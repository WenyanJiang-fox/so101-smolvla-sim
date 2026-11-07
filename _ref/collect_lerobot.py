"""按 LeRobot v3 格式采集 PickLift 数据（多物体 + 正确 action + 只留成功的）。

    G:/anaconda/envs/sim2real/python.exe collect_lerobot.py                # 200 条
    G:/anaconda/envs/sim2real/python.exe collect_lerobot.py 20             # 20 条
    G:/anaconda/envs/sim2real/python.exe collect_lerobot.py 200 500        # 200 条，最多试 500 次

和 local_picklift.py 同一个策略（速度和相机常量直接从它那儿 import，只有一份），
区别是四件事：

  1. action 记的是「命令下去的关节目标角」，从 data.ctrl 读，不是当前关节角。
     策略发的是 7 维末端增量，环境内部 _action_to_ctrl 解算成 6 维关节目标写进
     data.ctrl —— 那才是"下一步该去的姿势"。
  2. 直接写成 LeRobot v3 数据集（meta/info.json + data/ + videos/），
     不是散装 parquet。格式对齐那份公开的 MuJoCoPickLift-v1。
  3. 物体池放红绿蓝三个立方体，环境随机指定目标，指令才真的在区分。
  4. 没抓起来的那条不写进数据集（clear_episode_buffer 丢掉），免得教模型学失败。

单位：前 5 个关节用「度」，夹爪用 0~100 百分比（跟包里 build_so101_motors 一致，
也和公开数据集一致）。当前关节角直接 degrees() 是错的，夹爪得换算。

配对：记录的是「发动作之前那一帧观察」+「这一步命令下去的目标角」——
即 obs[t] -> action[t] -> obs[t+1]，这是模仿学习要的配对。
"""
from __future__ import annotations

import os
import sys

import av
import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import so101_nexus.mujoco  # noqa: F401  注册 gym 环境
from so101_nexus import (
    CubeObject,
    EndEffectorPose,
    JointPositions,
    ObjectPose,
    OverheadCamera,
    PickConfig,
    RobotConfig,
    WristCamera,
)
from so101_nexus.camera_utils import compute_angled_camera_params
from so101_nexus.config import SO101_JOINT_NAMES

from local_picklift import (
    CAM_AZIM,
    CAM_ELEV,
    FPS,
    LYING_QUAT,
    OW,
    POS_SCALE,
    ROT_SCALE,
    SPEED_POS,
    SPEED_ROT,
    align_action,
)

# 输出位置。默认还是原来那份 rgb 数据集；做扰动实验时用环境变量指到新目录，
# 免得把已经采好的 300 条覆盖掉（LeRobotDataset.create 会直接写同一个 root）。
OUT_ROOT = os.environ.get(
    "OUT_ROOT", r"G:\sim2real\so101-smolvla-sim\_ref\datasets\so101_picklift_rgb")
REPO_ID = os.environ.get("REPO_ID", "local/so101_picklift_rgb")
VIDEO_BACKUP = os.environ.get(
    "VIDEO_BACKUP", r"G:\sim2real\so101-smolvla-sim\_ref\rgb_first_episode.mp4")

# 场上 3 个物体：目标 + 2 个陪衬，全是同尺寸立方体，只差颜色。
# 池只放红绿蓝三个，配 n_distractors=2 就是三个全上场、目标从这三个里抽，
# 指令文本只有「Pick up the red/green/blue cube.」三种。
# POOL_COLORS / N_DISTRACTORS 可以覆盖，用来做「只留一个方块、指令固定」的简化版：
#   POOL_COLORS=red N_DISTRACTORS=0
# 不设这两个变量时逐字等于原来的三色三物体，老脚本行为不变。
POOL = [CubeObject(color=c) for c in
        os.environ.get("POOL_COLORS", "red,green,blue").split(",")]
N_DISTRACTORS = int(os.environ.get("N_DISTRACTORS", "2"))

# run() 的兜底「停住」判据：TCP 单帧位移(mm) 和夹爪关节单帧转角(mrad) 都低于
# 阈值，连续 STILL_N 帧就认为这一段走不动了，别把 n 帧跑满。
# 用关节实际角度而不是指令角度：手指夹住方块后指令还在往下压，关节早就不动了，
# 拿指令当信号会一直等（实测合爪那段因此空转 39 帧）。
STILL_MM = 0.05
STILL_MRAD = 0.1
STILL_N = 5

# --- DART 式抖动（PERTURB=1 打开）---
# 仿 ataghof/molmoact2-so101-sim 的 collect/collect_demos.py:163：每一步都往
# 「执行下去的指令」上叠独立高斯噪声，录进数据集的仍是没叠噪声的干净指令。
# 于是每一帧都是「被抖歪的状态 -> 专家纠回的干净动作」，补上闭环缺的那类样本。
# 原来那种「某一步猛推 3 帧、推的那几帧还不录」是错的：幅度是专家满速的 3.3 倍，
# 而且只制造一个孤立的偏离点，不是持续的小偏离。
PERTURB = os.environ.get("PERTURB") == "1"
# σ 直接照抄参考实现 collect/collect_demos.py:49
#   NOISE_SIGMA = 0.006  # rad, DART wobble on the executed arm command
# 同一条 SO-101、同一个 so101-nexus 包、同样加在关节目标的前 5 维上，单位也是
# 弧度，跟我们 ctrl 一致，不需要换算。
# 不采用「按 DART 公式令 Σ = 策略误差协方差」那个取法：我们策略实测误差 0.64°
# （_offline_fit.py:108），折成 σ≈0.80°，是参考值的 2.3 倍，实测把专家打散了
# （seed 0 跑成 399 帧后失败）。
PERTURB_SIGMA = float(os.environ.get("PERTURB_SIGMA", "0.006"))

# retry 条：故意抓空 → 张开 → 再抓真的。占比照抄参考实现的 p=0.15
# （collect_demos.py:244 的 ["std","retry","midtask","release"], p=[.40,.15,.15,.30]；
# 我们是 PickLift，没有放置段，midtask/release 移不过来）。
RETRY = os.environ.get("RETRY") == "1"
RETRY_PROB = float(os.environ.get("RETRY_PROB", "0.15"))
# 抓空那一下的抬起段单独给个短上限。参考实现 collect_demos.py:261 里，
# 抓空后的抬起是 go(miss_z + 0.10, CLOSED, 25, maxstep=0.02)——只有 25 步；
# 而它正常抓取后的抬起是 go(obj_z + 0.13, CLOSED, 40, ...)。我们这两段都是 250。
# 250 帧在普通条上无所谓（兜底的「停住」判据会在 40 来帧就收段），但抓空那条
# 的 wrist_roll 会自转（−36°→−91°，约 0.2°/帧）带着 TCP 一直动，判据永不触发，
# 于是跑满 250 帧≈8.3 秒的静止画面。给个硬上限直接掐掉。
MISS_LIFT_CAP = int(os.environ.get("MISS_LIFT_CAP", "25"))

# ALIGN_NORMAL=1：抓取时把夹爪的开合方向对到方块的侧面法向上。
# 为什么需要：方块出生时 yaw 是随机的（spawn_utils.py:170 用 random_yaw_quat 乘
# 上去），而 align_action 只压 z 轴、绕 z 的转角从来不管，所以合爪时两个手指和
# 方块的相对角度是随机的，夹在棱上是常事——这正是滑落的来源。
# 不设这个变量时 x_des=None，逐位等于以前的专家。
ALIGN_NORMAL = os.environ.get("ALIGN_NORMAL") == "1"

# 「转到位」的残差阈值（弧度）。run() 拿它当 stop_rot 时，stop 还要求旋转残差小于
# 它才算这一段的终点——也就是「转好了再夹」。默认 5°。
ALIGN_TOL = float(np.deg2rad(float(os.environ.get("ALIGN_TOL_DEG", "5"))))

# 对齐段（第一段）的旋转速度。默认是正常速度的 1/3：转得慢，这一段的帧数就是三倍
# 左右，模型才有足够多的帧看到「对准」这件事。ALIGN_ROT_FRAC=1 就退回原速。
ALIGN_ROT = SPEED_ROT * float(os.environ.get("ALIGN_ROT_FRAC", "0.33"))

# 「只录末端段」的录制门：REC_FROM_HOVER>0 时（单位米），TCP 进到悬停点这个半径以内
# 才开始往数据集里写帧；在那之前「从远处走过来」的那段只走不录，动作照走、渲染照跑。
# 默认 0＝不设门，从第一帧就录，老脚本的行为逐字节不变。
# 悬停点 = 方块正上方 60mm，就是 pick() 里第一段的那个目标点 pos + [0, 0, 0.06]。
REC_FROM = float(os.environ.get("REC_FROM_HOVER", "0"))

# 分片用：每个采集进程给一段互不重叠的种子带，第 n 个进程用 SEED_START=a、试 a..a+max_try-1。
# 不设＝从 0 开始，老脚本行为不变（单进程本来就该从 0 起）。
SEED_START = int(os.environ.get("SEED_START", "0"))


def object_quat(e, obj_pos):
    """读离 obj_pos 最近那个自由关节的 wxyz 四元数，也就是目标方块的朝向。

    privileged_state 里只有物体的位置（p[13:16]），没有朝向，只能从 MuJoCo 读。
    「找最近的那个自由关节」照抄 local_picklift.lay_object_flat 的办法。
    """
    m, d = e.model, e.data
    jid, best = -1, np.inf
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        dist = float(np.linalg.norm(d.xpos[m.jnt_bodyid[j]] - obj_pos))
        if dist < best:
            jid, best = j, dist
    if jid < 0:
        return None
    a = int(m.jnt_qposadr[jid])
    return d.qpos[a + 3:a + 7].copy()


def normal_from_quat(q_wxyz, q_cur=None):
    """方块侧面法向（取水平分量）——夹爪的开合方向要对的正是它。

    正方体的 ±x、±y 四个方向两两等价（夹爪开合方向本身正反也等效），四组都能让
    手指夹在平行面上，所以有得挑。q_cur（夹爪当前姿态，wxyz）给了的话，在这四个
    里挑「从当前姿态转过去最近」的那个，省掉最多 135° 的白转；不给时还是老行为，
    取 +x。方块立起来时 x 轴接近竖直，退回去用 y 轴。
    """
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    cands: list[np.ndarray] = []
    for ax in (0, 1):
        v = R[:, ax].copy()
        v[2] = 0.0                  # 只取水平分量：接近方向仍然是竖直向下
        if np.linalg.norm(v) > 1e-6:
            v = v / np.linalg.norm(v)
            cands.append(v)
            cands.append(-v)
    if not cands:                   # 极端退化（x、y 都竖直）：和以前一样没得挑
        cands = [np.array([1.0, 0.0, 0.0])]
    if q_cur is None or len(cands) == 1:
        return cands[0]
    # 用「对齐还差多少弧度」挑，和 align_action 同一把尺子。
    errs = [float(np.linalg.norm(align_action(q_cur, [0, 0, -1], gain=1.0, x_des=c)))
            for c in cands]
    return cands[int(np.argmin(errs))]


def make_env():
    cfg = PickConfig(
        obs_mode="visual",
        observations=[JointPositions(), EndEffectorPose(), ObjectPose(),
                      WristCamera(), OverheadCamera()],
        objects=POOL,
        # 官方默认是 0（单物体场景），这里按官方示例开 2 个陪衬。
        n_distractors=N_DISTRACTORS,
        robot=RobotConfig(ee_orientation_weight=OW),
    )
    env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                   control_mode="pd_ee_delta_pose")

    # 全局相机改成斜前方（同 local_picklift.py）
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
    return env


def lay_cylinders_flat(env) -> int:
    """把场上所有圆柱槽位摆成躺倒（轴沿世界 Y），返回摆了几个。

    包不给朝向参数，只能在 env 建好之后改自由关节的 qpos，不动包代码。
    哪个槽位是什么形状，直接看 geom 类型——方块/球/圆柱三种一眼分得开。
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    laid = 0
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not name.startswith("pick_slot_"):
            continue
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        if not geoms or m.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            continue
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
        if jid < 0:
            continue
        adr = m.jnt_qposadr[jid]
        d.qpos[adr + 2] = float(m.geom_size[geoms[0]][0])   # 放到半径高度，正好躺桌面上
        d.qpos[adr + 3:adr + 7] = LYING_QUAT
        laid += 1
    # 池里没放圆柱时 laid==0，不报错，照常做下面的沉降。

    # 沉降一下让它落稳。机械臂同时锁在原位，别让 200 步物理把手臂放倒。
    d.ctrl[:6] = d.qpos[:6]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=100)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return laid


def run_episode(env, seed: int, keep_frames: bool):
    """跑一条，把帧喂进数据集。返回 (任务指令, 是否成功, 录制帧列表)。"""
    obs, info = env.reset(seed=seed)
    e = env.unwrapped
    task = e.task_description
    # 扰动用的随机源，跟着 seed 走，同一条可复现。
    prng = np.random.default_rng(seed * 7919 + 13)

    lay_cylinders_flat(env)
    # 沉降推进过物理，重取一次 obs/info 再开始录。
    obs, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))

    grip_lo, grip_hi = e.model.actuator(e._actuator_ids[5]).ctrlrange

    def to_dataset(rad6):
        """6 个关节(rad) -> 数据集单位：前 5 个度，夹爪 0~100 百分比。"""
        out = np.empty(6, dtype=np.float32)
        out[:5] = np.degrees(rad6[:5])
        out[5] = (rad6[5] - grip_lo) / (grip_hi - grip_lo) * 100.0
        return out

    def st():
        p = np.asarray(info["privileged_state"], dtype=np.float64)
        return p[0:6], p[6:9], p[9:13], p[13:16]

    # 录制门的开关。REC_FROM<=0（默认）时一开始就是开的，等于没有门。
    rec_armed = REC_FROM <= 0.0

    def rec(q_pre, obs_pre, ctrl_rad):
        if not rec_armed:
            return
        if keep_frames:
            frames.append(np.hstack([obs_pre["overhead_camera"], obs_pre["wrist_camera"]]))
        dataset.add_frame({
            "observation.images.overhead": obs_pre["overhead_camera"],
            "observation.images.wrist": obs_pre["wrist_camera"],
            "observation.state": to_dataset(q_pre),
            "action": to_dataset(ctrl_rad),
            "task": task,
        })

    def step_ctrl(ctrl_rad):
        """照给定的 6 个关节目标推进一步物理，返回是否终止。

        采集要的是「执行带噪的关节目标、录干净的关节目标」，而 env.step 内部才把
        action 解算成 ctrl（base_env.py:1009），外面插不进关节噪声。这里只把 step
        的必要三行用环境自己的私有方法拼出来（base_env.py:1009-1013），
        不改包代码、也不重算 reward（采集用不到 reward，只要 info["success"]）。
        """
        nonlocal obs, info
        e.data.ctrl[e._actuator_ids] = ctrl_rad
        e._advance_physics()
        obs, info = e._observe()
        return bool(e.config.terminate_on_success and info.get("success", False))

    def run(target, grip, n, stop=None, perturb=False, align=True, stop_rot=None,
            rot_scale=None, arm_at=None):
        """走一段。stop(dist, ctrl) 说到了就返回，另有回合结束和兜底的「停住」。

        align=False：这一步不压腕部对齐（a[3:6] 置 0，保持当前姿态）。合爪段和抬起段
        用它——夹住之后再转，转的是夹着的方块。
        stop_rot（弧度）给了的话，stop 还必须「旋转残差 < stop_rot」才算到了，
        用来保证转到位才进入下一段（转好了再夹）。
        rot_scale 给了的话，这一段按它当旋转速度（不给就是 SPEED_ROT）。调小 = 同样
        的角度要更多帧。只有对齐段用它。


        原来的退出判据是「走到目标 1mm 以内」。比例控制在目标附近是按指数逼近的
        （动作 = 距离/POS_SCALE，越近越慢），所以每一段末尾都有几十帧在以 <1mm/步
        爬行，最后彻底静止。实测 seed 5 那条：5 个停点各 16 帧，共 81 帧（27%）。
        换成「还差 R_SWITCH 就切下一段」，速度还没掉下来就走了，动作连成一条。
        兜底的「停住」只在目标根本够不到时（关节限位）才会触发。
        """
        nonlocal obs, info, n_noise, rec_armed
        prev_tcp, prev_q, still = None, None, 0
        rot_ok = stop_rot is None
        rot_gain = SPEED_ROT if rot_scale is None else rot_scale
        for i in range(n):
            q_pre, tcp, quat, obj_now = st()
            # 开录制门：TCP 进到 arm_at 这个点的 REC_FROM 米以内就开，开了不再关。
            # 判据用这一帧自己的 tcp，所以这一帧本身就被录进去。
            if not rec_armed and arm_at is not None:
                if float(np.linalg.norm(arm_at - tcp)) < REC_FROM:
                    rec_armed = True
            obs_pre = obs
            a = np.zeros(7, dtype=np.float32)
            a[:3] = np.clip((target - tcp) / APPROACH, -1.0, 1.0) * SPEED_POS
            # ALIGN_NORMAL=1 时把开合方向对到方块侧面法向。每步重读方块朝向：
            # 上一次抓空可能把方块碰转了。不设时 x_des=None，行为和以前一样。
            x_des = None
            if ALIGN_NORMAL:
                oq = object_quat(e, obj_now)
                if oq is not None:
                    x_des = normal_from_quat(oq, quat)
            if align:
                a[3:6] = align_action(quat, [0, 0, -1], x_des=x_des) * rot_gain
            else:
                a[3:6] = 0.0
            if stop_rot is not None and align:
                if x_des is None:
                    rot_ok = True        # 没开 ALIGN_NORMAL / 读不到方块朝向：没有残差可判
                elif float(np.linalg.norm(
                        align_action(quat, [0, 0, -1], gain=1.0, x_des=x_des))) < stop_rot:
                    # gain=1.0 拿回来的是没被缩放的旋转残差（弧度）。动作那一行照旧。
                    rot_ok = True
            a[6] = grip
            # 专家动作先解算成关节目标。等一下录的就是它，加噪声的是另一份副本。
            ctrl_clean = e._action_to_ctrl(a)
            ctrl_exec = ctrl_clean
            if perturb:
                # DART 抖动：噪声叠在「执行」的关节目标上，前 5 个臂关节抖、
                # 夹爪不抖（对齐 ataghof 的 q[:5] += ...），录进数据集的还是
                # 上面那份干净的。夹到关节限位时夹回去，和 env 自己一个口径。
                ctrl_exec = ctrl_clean.copy()
                ctrl_exec[:5] = np.clip(ctrl_exec[:5] + prng.normal(0.0, PERTURB_SIGMA, 5),
                                        e._target_low[:5], e._target_high[:5])
                n_noise += 1
            term = step_ctrl(ctrl_exec)
            rec(q_pre, obs_pre, ctrl_clean)
            if term:
                return
            if stop is not None and rot_ok and stop(float(np.linalg.norm(target - tcp)), q_pre):
                return
            tcp_moved = np.inf if prev_tcp is None else np.linalg.norm(tcp - prev_tcp) * 1000.0
            grip_moved = np.inf if prev_q is None else abs(q_pre[5] - prev_q[5]) * 1000.0
            prev_tcp, prev_q = tcp, q_pre
            still = still + 1 if (tcp_moved < STILL_MM and grip_moved < STILL_MRAD) else 0
            if still >= STILL_N:
                return

    frames: list[np.ndarray] = []
    n_noise = 0        # 这条里有多少帧是带噪执行下去的，跑完打出来确认抖动真的开了
    q0, tcp0, _, obj0 = st()
    obj = obj0.copy()

    # 切段半径。抓取点是终点必须走到，其余两站都提前切走。
    R_HOVER = 0.015    # 物体上方这一站，还差 15mm 就转下降
    R_GRASP = 0.004    # 抓取点，留 4mm
    # 逼近用的比例尺。local_picklift.POS_SCALE 是 20mm：离目标 20mm 以内速度就
    # 按距离线性衰减，越近越慢，最后几毫米要爬二十来帧（实测 0.2mm/帧）。
    # 收紧到 5mm 只改最后 20mm 内的速度，路径不变，到位就干脆停住。
    APPROACH = 0.005

    def pick(pos, lift_cap=250):
        """朝 pos 这个点跑一遍完整四段：靠近 → 下降 → 合爪 → 抬起。

        拆成函数是为了 retry——同一条里要跑两遍（先故意抓空，再抓真的）。
        原来第一段是「原地站 150 帧、只转腕+张爪」，那本身就是一段静止，直接朝
        上方走：腕部对齐由 align_action 每步都在做，夹爪同步张开，三件事合成一条。
        lift_cap 只给抓空那次用，见 MISS_LIFT_CAP 的注释。
        """
        # 第一段的终点是「到位 **并且** 转到位」——没转到位就在上方原地继续转，
        # 转好了才允许下降，也就是「转好了再夹」。
        run(pos + [0, 0, 0.06], 1.0, 400,
            stop=lambda d, q: d < R_HOVER and q[5] >= 1.70,
            perturb=PERTURB, stop_rot=ALIGN_TOL,
            rot_scale=ALIGN_ROT,
            arm_at=pos + np.array([0.0, 0.0, 0.06]))        # 移到上方（抖）+ 慢转到位
        # 下降段锁住腕子：对齐在悬停点已经做完，下降是纯竖直的一段，不该再转。
        run(pos + [0, 0, 0.004], 1.0, 200,
            stop=lambda d, q: d < R_GRASP,
            perturb=PERTURB, align=False)                   # 下降（抖）
        # 合爪这一段不抖，对齐参考实现 grasp_at 里的 hold(CLOSED, 30, noise=False)：
        # 抖动正好落在接触成形那几帧会把方块抖掉。这段也不写 stop——手指夹住方块后
        # 关节就不动了，交给 run() 里的兜底判据，一停就接抬起。原来写的
        # ctrl[5] <= grip_lo 永远不成立（方块挡着，指令停在 −7.4°），白等了 39 帧。
        run(pos + [0, 0, 0.004], -1.0, 60, align=False)     # 合爪（不抖），夹住后不再转
        run(np.array([pos[0], pos[1], 0.14]), -1.0, lift_cap,
            perturb=PERTURB, align=False)                   # 抬起（抖），夹好了就别转了

    # retry：先故意抓空、张开、再抓真的。参考 collect_demos.py:256-265，占比 15%。
    # 这是数据集里唯一带「抓失败后恢复」的样本，噪声制造不出来（噪声只让状态偏，
    # 不会让爪子真的空一次）。偏移量照抄它：22~34mm，随机正负，加在 y 上。
    if RETRY and prng.random() < RETRY_PROB:
        off = float(prng.uniform(0.022, 0.034) * prng.choice([-1.0, 1.0]))
        miss = obj.copy()
        miss[1] += off
        print(f"      [retry] {task}  先朝 y{off * 1000:+.0f}mm 的位置抓一次")
        pick(miss, lift_cap=MISS_LIFT_CAP)
        if float(info.get("lift_height", 0.0)) > float(e.config.lift_threshold):
            print("      [retry] 偏着也抓到了，这条算普通条")
        else:
            # 张开爪子，别真把空爪当夹住的带走
            run(np.array([miss[0], miss[1], 0.14]), 1.0, 120)
            # 重读物体位置：刚才那一下可能把方块蹭动了（参考实现同样重读 objp()）
            obj = st()[3]

    pick(obj)

    if PERTURB:
        print(f"      [抖动] {task}  带噪执行 {n_noise} 帧")
    success = bool(info["success"])
    if success:
        dataset.save_episode()
    else:
        dataset.clear_episode_buffer()   # 没抓起来的那条不写进去
    return task, success, frames


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n_ok = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_try = int(sys.argv[2]) if len(sys.argv) > 2 else n_ok * 2

    global dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env = make_env()
    o, _ = env.reset(seed=0)
    H, W = o["overhead_camera"].shape[:2]

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features={
            "observation.images.overhead": {
                "dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]},
            "observation.images.wrist": {
                "dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]},
            "observation.state": {
                "dtype": "float32", "shape": (6,), "names": list(SO101_JOINT_NAMES)},
            "action": {
                "dtype": "float32", "shape": (6,), "names": list(SO101_JOINT_NAMES)},
        },
        root=OUT_ROOT,
        robot_type="so101",
        use_videos=True,
        video_backend="pyav",
    )
    print(f"数据集 -> {OUT_ROOT}")
    print(f"目标 {n_ok} 条有效，最多试 {max_try} 次\n")

    tally: dict[str, list[int]] = {}
    ok = 0
    first_frames: list[np.ndarray] = []
    first_ok = False
    for attempt in range(max_try):
        if ok >= n_ok:
            break
        task, success, frames = run_episode(env, attempt + SEED_START,
                                            keep_frames=(not first_frames))
        t = tally.setdefault(task, [0, 0])
        t[1] += 1
        if success:
            ok += 1
            t[0] += 1
        # 第一条的帧无论成败都留：失败那条更需要看视频，原来只在 success 里赋值，
        # 一失败就出不来视频。
        if not first_frames and frames:
            first_frames = frames
            first_ok = success
        print(f"  [{attempt + 1:>3}] {ok:>3}/{n_ok}  seed={attempt + SEED_START:<3} {task:34s} "
              f"{'✓' if success else '✗'}")
        sys.stdout.flush()

    total = dataset.meta.total_frames
    dataset.finalize()
    env.close()

    if first_frames:
        with av.open(VIDEO_BACKUP, "w", format="mp4") as c:
            s = c.add_stream("libx264", rate=FPS)
            s.width, s.height = first_frames[0].shape[1], first_frames[0].shape[0]
            s.pix_fmt = "yuv420p"
            s.options = {"crf": "23", "preset": "veryfast"}
            for f in first_frames:
                for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    c.mux(pk)
            for pk in s.encode():
                c.mux(pk)
        print(f"\n第一条视频（{'成功' if first_ok else '失败'}那条）-> {VIDEO_BACKUP}"
              f"  ({len(first_frames)} 帧, {os.path.getsize(VIDEO_BACKUP)//1024} KB)")

    print(f"\n{'任务':36s} {'成功/尝试':>10}  成功率")
    for task, (good, tries) in sorted(tally.items()):
        print(f"  {task:34s} {good:>4}/{tries:<5} {good / tries * 100:5.1f}%")
    print(f"\n有效 {ok}/{n_ok} 条，共试 {sum(t[1] for t in tally.values())} 次，总帧数 {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
