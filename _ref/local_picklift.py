"""在我们本地跑一条 PickLift 数据，出视频 + parquet。

    G:/anaconda/envs/sim2real/python.exe local_picklift.py                  # 方块
    G:/anaconda/envs/sim2real/python.exe local_picklift.py cylinder         # 竖圆柱（默认 25mm 直径）
    G:/anaconda/envs/sim2real/python.exe local_picklift.py cylinder 0.04    # 竖圆柱，直径=高=80mm

用包自带的 pd_ee_delta_pose（末端差速），不自己解 IK——环境内部用 mj_jacSite 解算。
两个非默认参数（都是运行时配置，没改包代码）：
  - ee_orientation_weight 0.01 -> 0.05：默认姿态几乎不管，工具轴是斜的，手指会斜着
    撞在方块顶上压不下去。调到 0.05 工具轴才接近竖直，同时位置精度还保得住。
  - 其它全默认。
"""
from __future__ import annotations

import os
import sys

import av
import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import so101_nexus.mujoco  # noqa: F401  注册 gym 环境
from so101_nexus.camera_utils import compute_angled_camera_params
from so101_nexus import (
    CubeObject,
    CylinderObject,
    EndEffectorPose,
    JointPositions,
    ObjectPose,
    OverheadCamera,
    PickConfig,
    RobotConfig,
    SphereObject,
    WristCamera,
)

# 形状 -> 物体类。cylinder_lying 还是圆柱，只是建好之后把它摆横。
SHAPE_CLASS = {
    "cube": CubeObject,
    "cylinder": CylinderObject,
    "cylinder_lying": CylinderObject,
    "sphere": SphereObject,
}
DEFAULT_HALF = 0.0125

# 横躺姿态：绕世界 X 轴转 90°，圆柱轴从竖直转到沿世界 Y（横在机器人面前）。
LYING_QUAT = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)

ROOT = r"G:\sim2real\so101-smolvla-sim\_ref"
SEED = 0
POS_SCALE = 0.02
ROT_SCALE = 0.1
OW = 0.05
FPS = 30

# 速度旋钮：动作乘这个系数再发出去，1.0 就是环境的满速。
# 满速 = 每步最多 POS_SCALE = 20mm；0.15 就是 3mm/步，30fps 下约 9cm/s。
# 变慢后每段需要的步数变多，所以下面 run(...) 的步数预算一起放大了。
SPEED_POS = 0.15
SPEED_ROT = 0.35

# 全局相机视角。包默认给的是正俯视（elevation=-90, azimuth=0）。
# elevation 负 = 往下看；azimuth 0 = 站在机器人正前方（+X 侧）回看。
CAM_ELEV = -30.0
CAM_AZIM = 210.0
CAM_TAG = f"e{int(CAM_ELEV)}a{int(CAM_AZIM)}"


def align_action(q_wxyz, z_des, gain=ROT_SCALE, x_des=None):
    """把工具 z 轴转到 z_des 所需的归一化旋转增量。

    x_des=None（默认）：目标 x 轴由当前姿态的 x 投影到 ⊥z_des 的平面得到，
    也就是绕 z 的转角（yaw）完全不管，是原来的行为。
    x_des 给了：把它投影过去当目标 x 轴，用来把夹爪的开合方向对到某个方向上
    （方块侧面法向）。z_des 竖直时 x_des 是水平的，本来就正交。
    """
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    z = np.asarray(z_des, dtype=np.float64)
    z /= np.linalg.norm(z)
    src = R[:, 0] if x_des is None else np.asarray(x_des, dtype=np.float64)
    x = src - np.dot(src, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = R[:, 1] - np.dot(R[:, 1], z) * z
    x /= np.linalg.norm(x)
    R_des = np.column_stack([x, np.cross(z, x), z])
    return np.clip(Rotation.from_matrix(R_des @ R.T).as_rotvec() / gain, -1.0, 1.0)


def lay_object_flat(env, quat_wxyz: tuple[float, ...], z: float) -> None:
    """把物体直接摆成给定姿态：写它自由关节的 qpos，再让物理沉一下。

    包不给朝向参数，所以只能在 env 建好之后从外面改，不动包代码。
    """
    m, d = env.unwrapped.model, env.unwrapped.data

    # 找到物体的自由关节：身体位置离物体最近的那个。
    pos = np.asarray(env.unwrapped._compute_obs_components(), dtype=np.float64)
    obj_pos = pos[21:24] if pos.size >= 24 else np.zeros(3)
    jid, best = -1, np.inf
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        dist = float(np.linalg.norm(d.xpos[m.jnt_bodyid[j]] - obj_pos))
        if dist < best:
            jid, best = j, dist
    if jid < 0:
        raise RuntimeError("没找到物体的自由关节")

    adr = m.jnt_qposadr[jid]
    d.qpos[adr + 2] = z                       # 放到半径高度，正好躺在桌面上
    d.qpos[adr + 3:adr + 7] = quat_wxyz
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=200)           # 沉降
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    shape = sys.argv[1] if len(sys.argv) > 1 else "cube"
    if shape not in SHAPE_CLASS:
        print(f"形状只能是 {list(SHAPE_CLASS)}，收到 {shape!r}")
        return 2

    half = float(sys.argv[2]) if len(sys.argv) > 2 else None
    tag = shape if half is None else f"{shape}_{half * 2000:.0f}mm"
    tag += f"_slow{SPEED_POS:g}_{CAM_TAG}"

    obj_kwargs = {"color": "green"}
    if half is not None:
        obj_kwargs["half_size"] = half
    pool = [SHAPE_CLASS[shape](**obj_kwargs)]
    print(f"物体：{shape} {obj_kwargs}")

    cfg = PickConfig(
        obs_mode="visual",
        observations=[JointPositions(), EndEffectorPose(), ObjectPose(),
                      WristCamera(), OverheadCamera()],
        objects=pool,
        robot=RobotConfig(ee_orientation_weight=OW),
    )
    env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                   control_mode="pd_ee_delta_pose")

    # 把全局相机从「正俯视」改成「斜前方」。这个自由相机只在构造 env 时配置一次，
    # 之后每步渲染都不再动，所以这里改一次就够。角度用包自带的函数算，不自己写公式。
    _env = env.unwrapped
    _cam = _env._overhead_obs_cam
    _comp = _env._overhead_cam_component
    if _cam is not None and _comp is not None:
        _p = compute_angled_camera_params(
            spawn_center=_env.config.spawn_center,
            spawn_max_radius=_env.config.spawn_max_radius,
            elevation=CAM_ELEV,
            azimuth=CAM_AZIM,
            fov_deg=_comp.fov_deg,
            aspect=_comp.width / _comp.height,
        )
        _cam.lookat[:] = _p["lookat"]
        _cam.distance = _p["distance"]
        _cam.elevation = _p["elevation"]
        _cam.azimuth = _p["azimuth"]
        print(f"全局相机：elevation {_p['elevation']:.0f}° / azimuth {_p['azimuth']:.0f}° / "
              f"距离 {_p['distance']:.2f}m")

    obs, info = env.reset(seed=SEED)

    if shape == "cylinder_lying":
        _h = half if half is not None else DEFAULT_HALF
        lay_object_flat(env, LYING_QUAT, _h)
        # 沉降推进了物理，重取一次 obs/info 再开始录。
        obs, _, _, _, info = env.step(
            np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        print(f"已摆横：圆柱轴沿世界 Y，中心高度 {_h:.4f} m")

    task = env.unwrapped.task_description
    print(f"任务：{task}")

    frames: list[np.ndarray] = []
    rows: list[dict] = []

    def st():
        p = np.asarray(info["privileged_state"], dtype=np.float64)
        return p[0:6], p[6:9], p[9:13], p[13:16], p[16:20]

    q0, tcp0, _, obj0, oquat = st()
    print(f"物体位置 {np.round(obj0,4)}  朝向(wxyz) {np.round(oquat,3)}")

    def rec(phase):
        q, tcp, _, obj, _ = st()
        frames.append(np.hstack([obs["overhead_camera"], obs["wrist_camera"]]))
        rows.append({
            "phase": phase,
            "action": np.degrees(q).astype(np.float32),
            "observation.state": np.degrees(q).astype(np.float32),
            "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
            "obj_x": obj[0], "obj_y": obj[1], "obj_z": obj[2],
            "reward": float(info.get("lift_height", 0.0)),
            "success": float(bool(info["success"])),
            "grasped": float(bool(info["is_grasped"])),
        })

    def run(target, grip, n, phase, tol=None):
        nonlocal obs, info
        for i in range(n):
            q, tcp, quat, _, _ = st()
            a = np.zeros(7, dtype=np.float32)
            a[:3] = np.clip((target - tcp) / POS_SCALE, -1.0, 1.0) * SPEED_POS
            a[3:6] = align_action(quat, [0, 0, -1]) * SPEED_ROT
            a[6] = grip
            obs, _, term, trunc, info = env.step(a)
            rec(phase)
            if tol is not None and np.linalg.norm(target - tcp) < tol:
                return
            if term or trunc:
                return

    obj = obj0.copy()
    hold = obj + [0, 0, 0.06]

    rec("start")
    run(tcp0, 1.0, 150, "张爪+竖姿态")
    run(hold, 1.0, 250, "移到物体上方", tol=0.003)
    run(obj + [0, 0, 0.004], 1.0, 200, "下降", tol=0.003)
    run(obj + [0, 0, 0.004], -1.0, 60, "合爪")
    run(np.array([obj[0], obj[1], 0.14]), -1.0, 250, "抬起")

    df = pd.DataFrame(rows)
    OUT_PQ = os.path.join(ROOT, f"local_picklift_{tag}.parquet")
    OUT_MP4 = os.path.join(ROOT, f"local_picklift_{tag}.mp4")
    df.to_parquet(OUT_PQ)

    _, _, _, obj_end, _ = st()
    print(f"\n共 {len(frames)} 帧，{len(frames)/FPS:.1f} 秒")
    print(f"  抓起    : {bool(df['grasped'].max())}")
    print(f"  举高    : {df['tcp_z'].max() - obj[2]:.3f} m")
    print(f"  成功    : {bool(df['success'].max())}")

    with av.open(OUT_MP4, "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for f in frames:
            for pk in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    print(f"  视频    -> {OUT_MP4}  ({os.path.getsize(OUT_MP4)//1024} KB)")
    print(f"  数据    -> {OUT_PQ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
