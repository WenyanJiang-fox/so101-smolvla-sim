"""冒烟测试：场景载入 + scripted expert 取放 + sim_env 接口 + 双相机出图。

    python smoke_expert.py
"""
import sys

import numpy as np

import paths
import sim_env
from expert import Expert

# 抖动范围跟着采集脚本走，别各写一份（改了那边这边就跟着变，冒烟测试才有意义）
from collect_demos import JITTER_XY, JITTER_YAW  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    env = sim_env.SO101SimEnv()
    e = Expert(controller=env.c)   # 专家和 env 共用同一份仿真状态

    print(f"场景载入 OK: {paths.SCENE_XML.name}")
    print(f"  neq={e.m.neq} (weld 应恒为 0: {[int(e.d.eq_active[i]) for i in range(e.m.neq)]})")
    print(f"  关节={paths.JOINT_NAMES}  每帧 {sim_env.STEPS_PER_FRAME} 物理步")

    print("\n[IK 路点]")
    q = e.plan_pick()
    if q is None:
        print("  ❌ 有路点 IK 解不出，场景需要调整")
        return 1
    for k, v in q.items():
        if isinstance(v, list):
            print(f"  {k:12s} ({len(v)} 个竖直路点) 首={np.round(np.degrees(v[0]), 1).tolist()}")
        else:
            print(f"  {k:12s} " + " ".join(f"{np.degrees(x):7.1f}" for x in v))

    ok_n = 0
    n_trial = 5
    print(f"\n[执行取放 × {n_trial}（含 ±{JITTER_XY*1000:.0f}mm / "
          f"±{np.degrees(JITTER_YAW):.0f}° 抖动）]")
    print(f"  {'#':<4}{'抬起z':>9}{'释放触':>8}{'终点x':>9}{'终点y':>9}{'终点z':>9}{'结果':>8}")
    rng = np.random.default_rng(0)
    for i in range(n_trial):
        xy = (paths.PICK_X + rng.uniform(-JITTER_XY, JITTER_XY),
              paths.PICK_Y + rng.uniform(-JITTER_XY, JITTER_XY))
        yaw = float(rng.uniform(-JITTER_YAW, JITTER_YAW))
        trace = []
        if e.run_episode(cube_xy=xy, cube_yaw=yaw, trace=trace) is None:
            print(f"  {i+1:<4}  IK 失败")
            continue
        d = {s: (x, y, z, n, g) for s, x, y, z, n, g in trace}
        p = e.cube_pos()
        ok = e.success()
        ok_n += ok
        print(f"  {i+1:<4}{d['lifted'][2]*1000:9.1f}{d['released'][3]:8d}"
              f"{p[0]*1000:9.1f}{p[1]*1000:9.1f}{p[2]*1000:9.1f}"
              f"{'✅ 成功' if ok else '❌ 失败':>8}")
    print(f"  成功率 {ok_n}/{n_trial}")

    print("\n[sim_env 接口]")
    obs = env.reset(cube_xy=(paths.PICK_X, paths.PICK_Y))
    print(f"  observation 键: {sorted(obs)}")
    for k, v in obs.items():
        print(f"    {k:32s} {v.shape} {v.dtype}"
              + (f" 范围=[{v.min()},{v.max()}]" if v.ndim == 3 else ""))
    print(f"  state 初值(°) = {np.round(obs['observation.state'], 1).tolist()}")
    env.step(obs["observation.state"], n_frames=sim_env.STEPS_PER_FRAME)
    print(f"  空操作一步后 state(°) = {np.round(env.state_deg(), 1).tolist()}")
    print(f"  is_success = {env.is_success()}")
    return 0 if ok_n == n_trial else 1


if __name__ == "__main__":
    sys.exit(main())
