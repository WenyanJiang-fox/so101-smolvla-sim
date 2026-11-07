"""scripted expert 自动采集：物理抓取 + success 过滤，存 LeRobot v3（与遥操作同口径）。

    G:/anaconda/envs/sim2real/python.exe collect_demos.py --episodes 50

和 teleop_collect.py 用同一套数据集口径（state/action 都是舵机 0~180°、双相机、
robot_type=so101），这样脚本数据和遥操作数据可以直接合起来训 SmolVLA。差别：
- 动作来自 scripted expert（IK 路点 + 物理抓取），不是人手。
- 只保留 e.success() 判定成功的 episode，失败的整条丢弃。

方块直接放在平桌面上（z=0，见 paths.py），默认 ±15mm / ±30° 抖动铺开分布。
再放宽之前先确认方块位置手臂够得着。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import paths
import sim_env

sys.path.insert(0, str(paths.MCP_SOURCES))
from mujoco_servo_controller import qpos_to_deg  # noqa: E402
from expert import Expert  # noqa: E402

REPO_ID = "so101_sim/scripted_pick_cube"

# 抖动范围（平桌面，方块直接放桌上，只要手臂够得着就行）
# 位置 ±30mm（方形，角落离取件点 42mm）；偏航 ±45°。
# 偏航放到 45° 靠的是 expert.ALIGN_WRIST 的腕转对齐：不对齐的话 45° 时方块在爪
# 开合轴上的跨距是 56.6mm > 爪全开 55.9mm，爪板根本伸不进去。
JITTER_XY = 0.030          # ±30mm
JITTER_YAW = np.pi / 6     # ±30°（±45° 要配腕转对齐，见 expert.ALIGN_WRIST）


def _action_deg(env):
    """当前 ctrl（关节弧度，= expert 的命令目标）→ 舵机 0~180°，和 state_deg 同口径。"""
    return np.array([qpos_to_deg(float(env.d.ctrl[a]), r)
                     for a, r in zip(env.acts, env.ranges)], dtype=np.float32)


def record_episode(e, env, frames, rng) -> bool:
    """跑一次 run_episode，把 30fps 帧写进 frames；成功返回 True，否则清空返回 False。"""
    step = [0]

    def on_frame():
        step[0] += 1
        if step[0] % sim_env.STEPS_PER_FRAME != 0:
            return
        obs = env.observation()
        frames.append({
            "observation.state": env.state_deg(),
            "action": _action_deg(env),
            f"observation.images.{paths.CAM_FRONT}":
                obs[f"observation.images.{paths.CAM_FRONT}"],
            f"observation.images.{paths.CAM_WRIST}":
                obs[f"observation.images.{paths.CAM_WRIST}"],
        })

    xy = (paths.PICK_X + rng.uniform(-JITTER_XY, JITTER_XY),
          paths.PICK_Y + rng.uniform(-JITTER_XY, JITTER_XY))
    yaw = float(rng.uniform(-JITTER_YAW, JITTER_YAW))
    trace = []
    # run_episode 返回 trace：不传 trace=[] 的话成功也返回 None，分不清 IK 失败（踩过）
    if e.run_episode(cube_xy=xy, cube_yaw=yaw, on_frame=on_frame, trace=trace) is None:
        frames.clear()
        return False
    if not e.success():
        frames.clear()
        return False
    return True


def main():
    # 进度是中文，重定向到日志时按默认编码写会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="scripted expert 自动采集")
    ap.add_argument("--episodes", type=int, default=50, help="要采的成功 episode 数")
    ap.add_argument("--root", default=str(paths.DATA_ROOT.parent / "so101_smolvla_sim_scripted"))
    ap.add_argument("--seed", type=int, default=None, help="抖动随机种子（不传则真随机）")
    a = ap.parse_args()

    env = sim_env.SO101SimEnv()
    e = Expert(controller=env.c)

    # 数据集（与 teleop_collect.py 完全同口径）
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(a.root)
    names = paths.JOINT_NAMES
    H = W = paths.IMG_H
    feats = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
    }
    for ck in (paths.CAM_FRONT, paths.CAM_WRIST):
        feats[f"observation.images.{ck}"] = {
            "dtype": "video", "shape": (H, W, 3),
            "names": ["height", "width", "channels"]}
    if (root / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(REPO_ID, root=str(root))
        print(f"续写已有数据集：{root}（已有 {ds.num_episodes} 条）")
    else:
        ds = LeRobotDataset.create(repo_id=REPO_ID, fps=paths.FPS, root=str(root),
                                   robot_type=paths.ROBOT_TYPE, use_videos=True,
                                   features=feats)
        print(f"新建数据集：{root}")

    env.observation()   # 预热离屏 renderer，别让第一次渲染落进数据里

    rng = np.random.default_rng(a.seed)
    saved = 0
    attempts = 0
    while saved < a.episodes and attempts < a.episodes * 4:
        attempts += 1
        frames = []
        if not record_episode(e, env, frames, rng):
            continue
        for f in frames:
            f["task"] = paths.TASK
            ds.add_frame(f)
        ds.save_episode()
        saved += 1
        print(f"已采 {saved}/{a.episodes}（尝试 {attempts} 次）", flush=True)

    ds.finalize()
    print(f"完成：{saved} 条成功，共尝试 {attempts} 次（成功率 {saved}/{attempts}），"
          f"数据在 {root}")


if __name__ == "__main__":
    main()
