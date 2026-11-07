"""颜色轮换的纯专家采集。

    python -u collect_colors.py [n_seed] [seed_start]

每个 seed 是一个固定布局，采 N_COLOR 套配色 × 3 个目标 = 每条 seed N_COLOR*3 条，
全部是专家自己走完的整条轨迹（不像 DAgger 那样掺策略段）。

**轮换必须按布局整体做，不能只轮换一条轨迹**：同一个 seed 下，三套配色共用同一份
物体摆位（位置由 rng 决定，seed 一样抽出来的下标和位置就一样），只有「哪个颜色占
哪个槽位」变了。目标用 reset(options={"target_index": k}) 钉住，钉住不会移动物体。

于是对于固定的 (seed, k)，三套配色下**目标位置逐位相同**，只有指令里的颜色词不同。
这正是语言条件策略需要的那种反事实配对——「去第几个槽位」这条捷径不成立，因为
同一个槽位在不同配色下对应不同的颜色词。

数据集：collect_lerobot 那一套（60 帧一处不动，只有 dataset 换成这里的）。
OUT_ROOT / REPO_ID 走环境变量，并行时每个 worker 指一份自己的。
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 必须在 import collect_lerobot 之前设好：它在模块层（第 63-67 行）读这几个变量。
os.environ.setdefault(
    "OUT_ROOT", r"G:\sim2real\so101-smolvla-sim\_ref\datasets\so101_picklift_colors")
os.environ.setdefault("REPO_ID", "local/so101_picklift_colors")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

import collect_lerobot as C
from so101_nexus import CubeObject

BASE = ["red", "green", "blue"]
# 主题配色固定是 BASE；别的配色 = BASE 循环左移。左移保证三套配色下同一个槽位
# 拿到三个不同颜色，指令词也就三套都不一样。
N_COLOR = int(os.environ.get("N_COLOR", "2"))
COLORINGS = [BASE[i:] + BASE[:i] for i in range(N_COLOR)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 104

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    probe = C.make_env()
    o, _ = probe.reset(seed=0)
    H, W = o["overhead_camera"].shape[:2]
    probe.close()

    dataset = LeRobotDataset.create(
        repo_id=C.REPO_ID,
        fps=C.FPS,
        features={
            "observation.images.overhead": {
                "dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]},
            "observation.images.wrist": {
                "dtype": "video", "shape": (H, W, 3),
                "names": ["height", "width", "channels"]},
            "observation.state": {
                "dtype": "float32", "shape": (6,), "names": list(C.SO101_JOINT_NAMES)},
            "action": {
                "dtype": "float32", "shape": (6,), "names": list(C.SO101_JOINT_NAMES)},
        },
        root=C.OUT_ROOT,
        robot_type="so101",
        use_videos=True,
        video_backend="pyav",
    )
    C.dataset = dataset

    print(f"数据集 -> {C.OUT_ROOT}  ({C.REPO_ID})")
    print(f"{N_COLOR} 套配色 {COLORINGS}")
    print(f"seed {seed_start}..{seed_start + n_seed - 1}  "
          f"每 seed {N_COLOR * 3} 条 -> 最多 {n_seed * N_COLOR * 3} 条\n", flush=True)

    t0 = time.monotonic()
    saved = tried = 0
    tally: dict[str, list[int]] = {}
    # (seed, k) -> {配色名: (指令, 目标位置, 目标槽位)}
    pairing: dict[tuple[int, int], dict[str, tuple]] = {}
    pos_mismatch = 0

    for ci, cols in enumerate(COLORINGS):
        C.POOL = [CubeObject(color=c) for c in cols]
        env = C.make_env()
        slots = [s.obj.color for s in env.unwrapped._slots]
        if slots != cols:
            raise RuntimeError(f"槽位颜色 {slots} 和配色 {cols} 对不上")

        pin = {"k": None}
        got: dict = {}
        orig_reset = env.reset

        def patched_reset(seed=None, options=None, _orig=orig_reset, **kw):
            """接住 run_episode 里那句 env.reset(seed=seed)，注入 target_index。

            顺手把这一帧的指令、目标位置、目标槽位记下来——run_episode 跑完场面就
            变了，只能在这儿取。
            """
            opts = dict(options) if options else {}
            if pin["k"] is not None:
                opts["target_index"] = pin["k"]
            o, i = _orig(seed=seed, options=opts or None, **kw)
            e = env.unwrapped
            got["task"] = e.task_description
            got["pos"] = tuple(float(v) for v in e._get_target_pose()[:3])
            got["ti"] = int(i.get("target_index", -1))
            return o, i

        env.reset = patched_reset

        for seed in range(seed_start, seed_start + n_seed):
            for k in range(len(cols)):
                tried += 1
                pin["k"] = k
                task, success, _fr = C.run_episode(env, seed, keep_frames=False)
                key = "".join(c[0] for c in cols)
                pairing.setdefault((seed, k), {})[key] = (
                    got["task"], got["pos"], got["ti"])
                t = tally.setdefault(task, [0, 0])
                t[1] += 1
                if success:
                    saved += 1
                    t[0] += 1
                print(f"  seed {seed} {key} k={k}  目标槽位 {got['ti']}"
                      f"（{cols[got['ti']]}）  {task:28s} {'✓' if success else '✗'}"
                      f"   {time.monotonic() - t0:.0f}s", flush=True)

        env.close()
        print(f"配色 {cols} 采完  {time.monotonic() - t0:.0f}s\n", flush=True)

    # ---------- 轮换一致性 ----------
    print("--- 轮换一致性 ---")
    for (seed, k), per in sorted(pairing.items()):
        poss = {v[1] for v in per.values()}
        words = {v[0] for v in per.values()}
        if len(poss) != 1 or len(words) != len(per):
            pos_mismatch += 1
            print(f"  异常 seed {seed} k={k}  位置种类 {len(poss)}  指令 {sorted(words)}")
    n_pair = len(pairing)
    print(f"  {n_pair} 组 (seed,k)：目标位置跨配色一致且指令两两不同 "
          f"{n_pair - pos_mismatch}/{n_pair}，异常 {pos_mismatch}")

    # 同一个 seed 的三个槽位必须落在三个不同位置，否则「去第几个槽位」就退化成同一处
    for seed in range(seed_start, seed_start + n_seed):
        ps = [next(iter(pairing[(seed, k)].values()))[1]
              for k in range(3) if (seed, k) in pairing]
        print(f"  seed {seed}  三个目标位置互不相同：{len(set(ps)) == len(ps)}")

    total = dataset.meta.total_frames
    dataset.finalize()

    print(f"\n{'指令':30s} {'成功/尝试':>10}  成功率")
    for task, (good, tries) in sorted(tally.items()):
        print(f"  {task:28s} {good:>4}/{tries:<5} {good / tries * 100:5.1f}%")
    el = time.monotonic() - t0
    print(f"\n写入 {saved}/{tried} 条  总帧数 {total}  "
          f"总耗时 {el:.0f}s = {el / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
