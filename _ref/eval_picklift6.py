"""在 MuJoCoPickLift-v1 里闭环跑训练好的 SmolVLA，报成功率。

    G:/anaconda/envs/sim2real/python.exe eval_picklift6.py            # 30 条，用 step 7000
    G:/anaconda/envs/sim2real/python.exe eval_picklift6.py 60 010000  # 60 条，用 step 10000

这是唯一能检验策略行不行的方式——训练 loss 低只说明它背下了训练集里那些轨迹。

闭环回路直接复用包自带的 so101_nexus.policy_adapters.RolloutRecorder：它负责组 batch
（状态转「度」、相机键改叫 observation.images.*）、把策略回的「度」转回弧度并裁剪到
action_space、从 info["success"] 取成败。自己只补三处：

  1. 控制模式用 pd_joint_pos。数据集里的 action 就是 6 个关节目标角，pd_joint_pos
     收的正好是绝对目标角，一一对应，不需要再过一遍 IK。
  2. 夹爪单位。数据集里夹爪记的是「占满行程的 0~100 百分比」，RolloutRecorder 一律按
     「度」送。两者只差第 6 个通道的一个仿射变换，来回换算一下，不动数据。
  3. 圆柱朝向。采集时圆柱每回合都被摆横，评估也得摆横，否则是拿没见过的姿态考它。
"""

from __future__ import annotations

import os
import sys
import time

# 本机连不上 huggingface.co。AutoProcessor.from_pretrained 会去列仓库里的
# 聊天模板，超时抛 ConnectTimeout——而它只捕获 httpx.NetworkError，接不住
# TimeoutException，于是直接崩。开离线开关让它回退到本地缓存（服务器训练时同款）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gymnasium as gym
import numpy as np
import torch

import so101_nexus.mujoco  # noqa: F401  注册 gym 环境
from so101_nexus import (
    CubeObject,
    CylinderObject,
    JointPositions,
    OverheadCamera,
    PickConfig,
    RobotConfig,
    SphereObject,
    WristCamera,
)
from so101_nexus.camera_utils import compute_angled_camera_params
from so101_nexus.policy_adapters import RolloutRecorder
from so101_nexus.processors.observation import Hwc2ChwImageObservationStep

from collect_lerobot import lay_cylinders_flat
from local_picklift import CAM_AZIM, CAM_ELEV, OW
from target_onehot import COLORS, onehot_from_task

# 本机是 Windows 路径。服务器上跑的时候用 PROJECT_ROOT 覆盖——_collect_dagger.py
# 靠它去 checkpoints/<step>/pretrained_model 找策略权重。
PROJECT = os.environ.get("PROJECT_ROOT", r"G:\sim2real\so101-smolvla-sim")
IMAGE_KEYS = ("observation.images.overhead", "observation.images.wrist")

# 被评的 checkpoint 是在 6 物体数据集上训的，场景必须还原成那 6 个物体：
# 3 方块 / 1 圆柱 / 2 球，见 datasets/so101_picklift_6obj/meta/tasks.parquet。
# 不能从 collect_lerobot 导入 POOL——那份已经为 3obj 数据集换成 4 个立方体了。
# 全用裸构造函数：三个类的 half_size 默认都走 _resolve_half_size，都解析成 0.0125。
POOL = [
    CubeObject(color="red"),
    CubeObject(color="yellow"),
    CubeObject(color="orange"),
    CylinderObject(color="green"),
    SphereObject(color="blue"),
    SphereObject(color="purple"),
]

# 训练集里的回合是 266~683 帧，留足余量；失败的那些会一直磨到上限才停。
MAX_STEPS = 900
# 采集时用的 seed 是 0..207，评估必须错开——否则是拿训练见过的场面考它。
EVAL_SEED = 10000


class LyingCylinderReset(gym.Wrapper):
    """每回合 reset 之后把圆柱摆横，再刷新一次 obs。"""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            lay_cylinders_flat(self.env)
        except RuntimeError:
            return obs, info  # 场上没圆柱，不用摆
        # 用「保持当前角度」走一步：摆横是直接写 qpos 再沉降的，obs 得重新取。
        # 但夹爪那一路必须 +0.2 rad，不能照搬「保持不动」。采集时给 ee_delta_pose
        # 的第 6 通道是增量语义，1.0 对应 +0.2 rad（实测 ctrl 从 −0.17453 挪到
        # 0.02544，差正好 0.2）。不补这一下，评测的夹爪会停在 reset 给的闭合位
        # −10.0°，而训练数据每条轨迹的起点都是 −8.79°，差 1.21°；腕部相机贴着
        # 爪子拍，这点角度被放大成 3% 的像素差。补上之后两边逐位相同（实测 0.0000）。
        action = np.asarray(obs["state"], dtype=np.float32).copy()
        action[5] += 0.2
        obs, _, _, _, info = self.env.step(action)
        return obs, info


def make_env(n_distractors: int | None = None, pool: list | None = None) -> gym.Env:
    """和采集时同场地、同相机，只是换成接收绝对关节角。

    pool 是场上的物体池，默认上面那份 6 个物体的 POOL。评 3obj 的 checkpoint
    必须传采集脚本里那份（collect_lerobot.POOL，4 个立方体），否则场上会摆出
    训练时根本没见过的东西——圆柱和球。
    n_distractors 默认「池里全上」；调小就只剩目标 + 几个陪衬。
    """
    objects = POOL if pool is None else pool
    cfg = PickConfig(
        obs_mode="visual",
        # 只要关节角 + 两路相机：视觉模式下 state 就是这 6 个关节，单位和数据集对得上。
        observations=[JointPositions(), WristCamera(), OverheadCamera()],
        objects=objects,
        n_distractors=len(objects) - 1 if n_distractors is None else n_distractors,
        robot=RobotConfig(ee_orientation_weight=OW),
    )
    env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                   control_mode="pd_joint_pos")

    # 全局相机改成斜前方（同 collect_lerobot.py / local_picklift.py）
    e = env.unwrapped
    cam, comp = e._overhead_obs_cam, e._overhead_cam_component
    p = compute_angled_camera_params(
        spawn_center=e.config.spawn_center,
        spawn_max_radius=e.config.spawn_max_radius,
        elevation=CAM_ELEV, azimuth=CAM_AZIM,
        fov_deg=comp.fov_deg, aspect=comp.width / comp.height,
    )
    cam.lookat[:] = p["lookat"]
    cam.distance, cam.elevation, cam.azimuth = p["distance"], p["elevation"], p["azimuth"]
    return LyingCylinderReset(env)


class SmolVLARolloutPolicy:
    """把 lerobot 的 SmolVLA 接成 RolloutRecorder 要的 select_action/reset。

    只需要补两件事：夹爪单位换算、图像 HWC uint8 -> CHW float32 [0,1]。
    模型的归一化器、分词器、动作反归一化都走 checkpoint 里存好的那套流水线。
    """

    def __init__(self, policy, preprocessor, postprocessor, env: gym.Env) -> None:
        self.policy = policy
        self.pre = preprocessor
        self.post = postprocessor
        self.to_chw = Hwc2ChwImageObservationStep(image_keys=IMAGE_KEYS)
        unwrapped = env.unwrapped
        ctrl = unwrapped.model.actuator(unwrapped._actuator_ids[5]).ctrlrange
        self.g_lo_deg, self.g_hi_deg = (float(v) for v in np.rad2deg(ctrl))
        # 噪声种子按 (回合号, 本回合第几次调用) 固定，见 select_action。
        self._ep, self._k = -1, 0
        # 存档声明了几维 state 就喂几维。补过目标 one-hot 的数据集训出来是 9 维，
        # 老存档仍是 6 维，这里跟着存档走，两边的存档同一份代码都能评。
        feat = policy.config.input_features["observation.state"]
        self.state_dim = int(np.prod(getattr(feat, "shape", None) or feat["shape"]))

    def reset(self) -> None:
        self.policy.reset()
        self._ep += 1
        self._k = 0

    def select_action(self, batch: dict) -> np.ndarray:
        state = np.asarray(batch["observation.state"], dtype=np.float32).copy()
        state[5] = (state[5] - self.g_lo_deg) / (self.g_hi_deg - self.g_lo_deg) * 100.0
        if self.state_dim == state.shape[0] + len(COLORS):
            state = np.concatenate([state, onehot_from_task(batch["task"])])

        # 必须是 tensor：流水线里加 batch 维和搬设备两步行都只认 tensor，
        # 递 numpy 会被原样放过，最后停在 CPU 上跟 cuda 的权重对不上。
        obs = {"observation.state": torch.from_numpy(state), "task": batch["task"]}
        obs.update({k: batch[k] for k in IMAGE_KEYS})
        obs = self.to_chw.observation(obs)

        obs = self.pre(obs)
        # flow matching 采样从一组随机高斯噪声起步（common/flow_matching.py:43 sample_noise），
        # lerobot 不设种子，每次调用都抽新的，于是同一个 seed 跑两次结果不同。
        # 按 (回合号, 本回合第几次) 固定住，评测才可复现。
        torch.manual_seed(self._ep * 100_000 + self._k)
        self._k += 1
        with torch.inference_mode():
            action = self.policy.select_action(obs)
        action = np.asarray(self.post(action).detach().cpu(), dtype=np.float32).reshape(-1)

        action[5] = action[5] / 100.0 * (self.g_hi_deg - self.g_lo_deg) + self.g_lo_deg
        return action


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    step = (sys.argv[2] if len(sys.argv) > 2 else "7000").zfill(6)
    ckpt = os.path.join(PROJECT, "checkpoints", step, "pretrained_model")
    if not os.path.isdir(ckpt):
        print(f"没有 {ckpt}")
        return 2

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"checkpoint : {ckpt}")
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    env = make_env()
    recorder = RolloutRecorder(
        env,
        SmolVLARolloutPolicy(policy, pre, post, env),
        max_steps_per_episode=MAX_STEPS,
    )
    print(f"{n} 条，seed {EVAL_SEED}..{EVAL_SEED + n - 1}，单条上限 {MAX_STEPS} 帧\n")

    tally: dict[str, list[int]] = {}
    ok = 0
    t0 = time.time()
    for i in range(n):
        r = recorder.record_episode(seed=EVAL_SEED + i)
        # reset 之后没再 reset 过，所以这里读到的还是这一条的任务
        task = env.unwrapped.task_description
        t = tally.setdefault(task, [0, 0])
        t[1] += 1
        if r.success:
            ok += 1
            t[0] += 1
        print(f"  [{i + 1:>3}/{n}] {ok:>3} 胜  {task:34s} "
              f"{'✓' if r.success else '✗'}  {r.n_steps:>3} 帧  "
              f"{(time.time() - t0) / 60:5.1f} 分")
        sys.stdout.flush()

    env.close()

    print(f"\n{'任务':36s} {'成功/尝试':>10}  成功率")
    for task, (good, tries) in sorted(tally.items()):
        print(f"  {task:34s} {good:>4}/{tries:<5} {good / tries * 100:5.1f}%")
    print(f"\n总成功率 {ok}/{n} = {ok / n * 100:.1f}%，"
          f"用了 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
