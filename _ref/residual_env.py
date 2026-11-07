"""把 SmolVLA 包成固定基座，残差 PPO 在它输出的 6 维关节目标上叠加增量。

SmolVLA 冻结、不出梯度。每 n_action_steps 步它出一个 6 维绝对关节目标 a_base；
残差网络每一步输出 [-1,1]^4：(dx, dy, dz, 夹爪)。前三维修的是 TCP 位移
（最多 12 mm），用雅可比换成 5 个臂关节的增量，再和基座动作相加。

    a = clip(a_base + [dq(J⁻¹·12mm·r[:3]), 0.05·r[3]], action_low, action_high)

观测 33 维：6 关节角 + 7 TCP 位姿 + 7 目标方块位姿 + 3 相对位置
+ 6 当前基座动作 + 4 上一步残差。
奖励 = grasping + task_objective + completion_bonus
      + γΦ(s')−Φ(s)，Φ = −2.0·‖(p_obj−p_tcp)_xy‖（势能差分，每步都给）
      − 0.01‖r‖² − 0.05‖rₜ−rₜ₋₁‖²。
残差在 TCP 的 xy 进入方块 40 mm 以内时打开；夹住后的 30 步、或基座开始合爪后的 60 步
关闭（见 HOVER_XY、WINDOW_AFTER_GRASP、WINDOW_AFTER_CLOSE）。

用法：
    from residual_env import make_residual_env
    env = make_residual_env()
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(np.zeros(4, np.float32))

只供仿真用：目标方块和 TCP 的位姿直接读 MuJoCo 的 e.data，真机上没有这份特权信息。
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import eval_picklift6 as E
from collect_lerobot import POOL as POOL_TRAIN

# 残差现在直接给 TCP 位移（任务空间），再用雅可比换成关节增量。
# 关节空间那版做不到「误差多大就修多大」：基座失败时的 xy 偏差是 20~40 mm，
# 而 ±0.5°/关节只挪 2.5 mm，±2°/关节又会把已经对准的位置推飞。
TCP_DELTA_M = 0.003      # 每步最多挪 3 mm。12 mm/步时，探索噪声（0.22×12≈2.6 mm/步）
                         # 在 50 步的 chunk 里能累积 ~18 mm，足以把基座的接近轨迹带偏。
GRIP_DELTA_RAD = 0.05    # 夹爪关节最多改 0.05 rad
DQ_MAX = 0.12            # 单个关节每步最多改 0.12 rad，防奇异位形下解出大跳变
DAMP = 4e-4              # 阻尼最小二乘的 λ²（λ=0.02）
HORIZON = 300            # 一条训练回合最多 300 步；成功的通常 150~200 步结束
W_RES, W_DRES = 0.01, 0.05

# 残差窗口：TCP 的 xy 离方块中心 40 mm 内就打开。
# 原来用 15 mm，实测纯基座只有 6/20 个回合能进到这个半径，剩下 2/3 的失败回合
# 残差一步都用不上；40 mm 覆盖了大多数能靠近方块的回合。
# 基座第一次下合爪指令后再放 WINDOW_AFTER_CLOSE 步就关闭：合爪前的 xy 修正保留，
# 抬升/放置阶段不再继续加偏置（实测开窗回合中位数被改了 244 步，远超需要）。
HOVER_XY = 0.040
CLOSE_PCT = 30.0          # 夹爪指令低于这个百分比算「合爪」
WINDOW_AFTER_CLOSE = 60
WINDOW_AFTER_GRASP = 30   # 已经夹住之后再改关节只会毁掉后面的抬升/放置
# 势能 Φ = −W_XY·‖(p_obj−p_tcp)_xy‖，每步给 γΦ(s')−Φ(s)。抓空回合抬升恒 0，
# 只有这项能给出梯度。直接每步扣分会被「停在方块正上方」刷爆（0/60 那次），
# 差分版停在原地净收益为 0，只有真把 TCP 挪近才赚钱，最优策略不变。
W_XY = 2.0
GAMMA = 0.99              # 必须和 PPO 的 gamma 一致，否则势能差分不再保序

OBS_DIM = 6 + 7 + 7 + 3 + 6 + 4      # = 33（上一步残差 4 维：dx,dy,dz,爪）


def yaw_err(tcp_q, obj_q) -> float:
    """TCP 与方块绕世界 z 的朝向差，按方块 90° 对称折到 [−45°, 45°)。"""
    d = (yaw_of(tcp_q) - yaw_of(obj_q) + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
    return float(d)


def yaw_of(quat) -> float:
    """四元数 (w,x,y,z) 的偏航角。只看绕世界 z 的转角，方块绕 z 对称 90°。"""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class SeedCycler(gym.Wrapper):
    """每次 reset 换一个 seed，让训练看到不同的方块初始位置。"""

    def __init__(self, env: gym.Env, seed0: int = 0) -> None:
        super().__init__(env)
        self._next_seed = seed0

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:   # 调用方点名了 seed 就不动它
            self._next_seed += 1
            kwargs["seed"] = self._next_seed
        return self.env.reset(**kwargs)


class ResidualGraspEnv(gym.Env):
    """残差动作空间在外面，SmolVLA 和 MuJoCo 都在里面。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tag: str = "graspsegexpert_230000",
        n_action_steps: int | None = None,
        n_distractors: int | None = None,
        pool: list | None = None,
        tcp_delta_m: float = TCP_DELTA_M,
        grip_delta_rad: float = GRIP_DELTA_RAD,
        hover_xy: float = HOVER_XY,
        horizon: int = HORIZON,
        w_res: float = W_RES,
        w_dres: float = W_DRES,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), np.float32)

        self.tcp_delta = tcp_delta_m
        self.grip_delta = grip_delta_rad
        self.hover_xy = hover_xy
        self.horizon = horizon
        self.w_res, self.w_dres = w_res, w_dres

        # --- 基座策略 ---
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        ckpt = os.path.join(E.PROJECT, "checkpoints", tag, "pretrained_model")
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(ckpt)
        self.policy = SmolVLAPolicy.from_pretrained(ckpt)
        self.policy.eval()
        if n_action_steps is not None:
            self.policy.config.n_action_steps = int(n_action_steps)
        self.n_action_steps = int(self.policy.config.n_action_steps)
        pre, post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)}},
        )

        # --- 环境 ---
        self.env = E.make_env(
            n_distractors=n_distractors,
            pool=POOL_TRAIN if pool is None else pool,
        )
        self.adapter = E.SmolVLARolloutPolicy(self.policy, pre, post, self.env)
        # task_description 在 __init__ 时还是空串，_task_reset 里才填；reset 里再取一次。
        self.task = ""

        e = self.env.unwrapped
        self._act_low = np.asarray(e.action_space.low, np.float64)
        self._act_high = np.asarray(e.action_space.high, np.float64)
        self._base_rad = np.zeros(6, np.float64)
        self._prev_res = np.zeros(4, np.float64)
        self._slot = 0
        self._steps = 0
        # 残差窗口：TCP 进入悬停半径后置 True，本回合不再回退；
        # 基座第一次合爪的步号记在 _close_step，之后 WINDOW_AFTER_CLOSE 步关窗。
        self._armed = False
        self._close_step = -1
        self._grasp_step = -1
        self._grip_prev = None
        # 计时：env.step（含两路相机渲染）和基座推理各占多少
        self.t_env = 0.0
        self.t_base = 0.0
        self.n_total = 0
        # 再细分：真正跑模型的 _get_action_chunk、policy 层的准备、adapter 的预处理
        self.t_chunk = 0.0
        self.n_chunk = 0
        self.t_pol = 0.0
        _orig_pol = self.policy.select_action
        _orig_chunk = self.policy._get_action_chunk

        def _timed_chunk(*a, **kw):
            t = time.perf_counter()
            out = _orig_chunk(*a, **kw)
            self.t_chunk += time.perf_counter() - t
            self.n_chunk += 1
            return out

        def _timed_pol(*a, **kw):
            t = time.perf_counter()
            out = _orig_pol(*a, **kw)
            self.t_pol += time.perf_counter() - t
            return out

        self.policy._get_action_chunk = _timed_chunk
        self.policy.select_action = _timed_pol

    # --- 内部 ---
    def _target_pose(self) -> np.ndarray:
        e = self.env.unwrapped
        bid = mujoco.mj_name2id(
            e.model, mujoco.mjtObj.mjOBJ_BODY, f"pick_slot_{self._slot}")
        if bid < 0:
            raise RuntimeError(f"场上没有 pick_slot_{self._slot}")
        return np.concatenate(
            [e.data.xpos[bid], e.data.xquat[bid]]).astype(np.float64)

    def _res_obs(self, obs: dict) -> np.ndarray:
        e = self.env.unwrapped
        tcp = e._get_tcp_pose()
        obj = self._target_pose()
        return np.concatenate([
            np.asarray(obs["state"], np.float64),   # 6 关节角
            tcp,                                    # 3 位置 + 4 四元数
            obj,                                    # 3 + 4
            obj[:3] - tcp[:3],                      # 3 相对位置，省得网络自己减
            self._base_rad,                         # 6 当前基座动作
            self._prev_res,                         # 4 上一步残差（dx,dy,dz,爪）
        ]).astype(np.float32)

    def _base_action(self, obs: dict) -> np.ndarray:
        """问一次 SmolVLA。它内部按 n_action_steps 排队，非换挡的调用不解算模型。"""
        batch = {
            "observation.state": np.rad2deg(obs["state"]).astype(np.float32),
            "task": self.task,
            "observation.images.overhead": obs["overhead_camera"],
            "observation.images.wrist": obs["wrist_camera"],
        }
        deg = self.adapter.select_action(batch)
        lo, hi = self.adapter.g_lo_deg, self.adapter.g_hi_deg
        self.base_grip_pct = float((deg[5] - lo) / (hi - lo) * 100.0)
        # 第一次从张开跨到合爪的那一步，作为窗口的右端
        pct = self.base_grip_pct
        if (self._close_step < 0 and self._grip_prev is not None
                and self._grip_prev >= CLOSE_PCT > pct):
            self._close_step = self._steps
            # 合爪这一瞬 TCP 与方块的高度差 / 绕 z 朝向差。对齐却抓不住的回合
            # 要看是高度问题还是爪面法向问题（真机上后者是滑落的根因）。
            tcp = self.env.unwrapped._get_tcp_pose()
            obj = self._target_pose()
            self._close_dz = float(tcp[2]) - float(obj[2])
            d = (yaw_of(tcp[3:7]) - yaw_of(obj[3:7])) % (np.pi / 2.0)
            self._close_dyaw = float(min(d, np.pi / 2.0 - d))
        self._grip_prev = pct
        return np.clip(np.deg2rad(deg), self._act_low, self._act_high)

    def _dq_from_tcp(self, delta: np.ndarray) -> np.ndarray:
        """把 TCP 位移（世界系，米）换成 5 个臂关节的增量。

        阻尼最小二乘：dq = Jᵀ(JJᵀ + λ²I)⁻¹·delta。超出 DQ_MAX 时整体等比缩小，
        不逐关节夹，否则方向会被破坏（实测逐关节夹到 0.05 时，12 mm 只走出 2 mm）。
        腕转那一维试过，加进来 BC 起点从 4/20 掉到 2/20，成功率也没涨，已删。
        """
        e = self.env.unwrapped
        jacp = np.zeros((3, e.model.nv))
        jacr = np.zeros((3, e.model.nv))
        mujoco.mj_jacSite(e.model, e.data, jacp, jacr, e._tcp_site_id)
        J = jacp[:, e._arm_qvel_addrs]                   # 3 × 5
        dq = J.T @ np.linalg.solve(J @ J.T + DAMP * np.eye(3), delta)
        m = float(np.max(np.abs(dq))) if dq.size else 0.0
        if m > DQ_MAX:
            dq *= DQ_MAX / m
        return dq

    def _xy_error(self) -> float:
        """TCP 与目标方块在水平面上的距离（米）。"""
        e = self.env.unwrapped
        tcp = e._get_tcp_pose()[:3]
        obj = self._target_pose()[:3]
        return float(np.linalg.norm(tcp[:2] - obj[:2]))

    def _residual_allowed(self) -> bool:
        """悬停后打开；基座开始合爪后 WINDOW_AFTER_CLOSE 步关闭。

        没下过合爪指令就保持打开（这时抓取还没发生，截断只会让残差更晚才起作用）。
        """
        if not self._armed:
            return False
        if (self._grasp_step >= 0
                and self._steps - self._grasp_step > WINDOW_AFTER_GRASP):
            return False
        if self._close_step < 0:
            return True
        return self._steps - self._close_step <= WINDOW_AFTER_CLOSE

    # --- gym 接口 ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # 原始观测（含两路相机）留一份给采集脚本；对外返回的是 33 维残差观测。
        self.last_obs = obs
        self.task = self.env.unwrapped.task_description
        self.adapter.reset()
        self._slot = int(info.get("target_index", 0))
        self._base_rad = np.zeros(6, np.float64)
        self._prev_res = np.zeros(4, np.float64)
        self._steps = 0
        self._armed = False
        self._close_step = -1
        self._grasp_step = -1
        self._grip_prev = None
        self._close_dz = float("nan")
        self._close_dyaw = float("nan")
        self._phi = -W_XY * self._xy_error()
        # flow-matching 的噪声按 adapter 的 (回合号, 调用序号) 定种子。回合号默认是
        # 「本进程第几次 reset」，换个进程/换个顺序就会变，同一个 seed 的结果对不上。
        # 这里用评测 seed 直接定回合号，跨进程可复现，基线和残差也能逐条配对。
        if seed is not None:
            self.adapter._ep = int(seed) % 100_000
        # 第一次基座动作先算出来，观测里的「当前基座动作」才不是零
        self._base_rad = self._base_action(obs)
        self._armed = self._xy_error() <= self.hover_xy
        return self._res_obs(obs), info

    def step(self, action: np.ndarray):
        res = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        open_ = self._residual_allowed()
        if not open_:                        # 窗口外恒零，观测里记的也是零，保持自洽
            res = np.zeros(4, np.float64)
        dq = self._dq_from_tcp(self.tcp_delta * res[:3])
        delta = np.concatenate([dq, [self.grip_delta * res[3]]])
        base_used = self._base_rad.copy()
        act = np.clip(base_used + delta, self._act_low, self._act_high)

        t0 = time.perf_counter()
        obs, raw_reward, terminated, truncated, info = self.env.step(
            act.astype(np.float32))
        t1 = time.perf_counter()
        self.last_obs = obs
        self._steps += 1
        if self._grasp_step < 0 and info.get("is_grasped"):
            self._grasp_step = self._steps

        # 残差优化「抓住 + 抬升 + 成功」。reaching 仍不要：它是把 TCP 往方块挪
        # 的势能差，轨迹整体被毁也能刷到钱。grasping 要：抓空回合抬升恒为 0，
        # task_objective / completion_bonus 也恒为 0，只有它能给出这一下抓住没有。
        comp = info.get("reward_components") or {}
        reward = float(comp.get("grasping", 0.0)
                       + comp.get("task_objective", 0.0)
                       + comp.get("completion_bonus", 0.0))

        # 稠密项只在基座已经合爪之后的 60 步内给。之前是「开窗就给」，模型发现
        # 停在方块正上方（xy≈0）比真去抓更划算，0/60 就是这么来的。
        xy = self._xy_error()
        if xy <= self.hover_xy:
            self._armed = True
        # 势能差分：Φ(s) = −W_XY·xy。停在原地净收益为 0，挪近才拿得到。
        phi = -W_XY * xy
        reward += GAMMA * phi - self._phi
        self._phi = phi

        pen = (self.w_res * float(res @ res)
               + self.w_dres * float((res - self._prev_res) @ (res - self._prev_res)))
        self._prev_res = res
        self._base_rad = self._base_action(obs)
        t2 = time.perf_counter()
        self.t_env += t1 - t0
        self.t_base += t2 - t1
        self.n_total += 1

        truncated = bool(truncated) or self._steps >= self.horizon
        info["residual_norm"] = float(np.linalg.norm(res))
        info["reward_env"] = float(raw_reward)
        info["reward_used"] = float(reward)
        # 训练日志要用的逐回合明细：这一步残差指向哪、离方块多远、拿到多少
        info["res_vector"] = res.astype(np.float32)
        info["xy_error"] = float(xy)
        info["armed"] = int(self._armed)
        info["window_open"] = int(open_)
        info["close_step"] = int(self._close_step)
        info["close_dz"] = float(self._close_dz)
        info["close_dyaw"] = float(self._close_dyaw)
        info["grasp_step"] = int(self._grasp_step)
        info["reward_step"] = float(reward - pen)
        # 采 RECAP 数据要逐帧存「从这一帧算出的基座动作」和「实际下发的动作」：
        # 后面做优势条件化时，监督目标是实际执行的动作（含残差修正），不是基座原样。
        info["base_used"] = base_used.astype(np.float32)
        info["exec_used"] = act.astype(np.float32)
        return self._res_obs(obs), float(reward - pen), terminated, truncated, info

    def close(self):
        self.env.close()
        super().close()


def make_residual_env(**kwargs) -> gym.Env:
    return SeedCycler(ResidualGraspEnv(**kwargs))
