"""同一场景换目标颜色，造反事实配对数据。

    G:/anaconda/envs/sim2real/python.exe -u _gen_counterfactual.py 208 [seed_start]

每个 seed 采 3 条：池里每种颜色各当一次目标，布局必须逐字节相同。

seed_start 是可选的起始 seed（默认 0），用来把采集切成互不重叠的片并行跑。
配对性质是「同一个 seed 内三色布局一致」，跨 seed 之间没有任何耦合，
所以分片不会破坏它——每片只需各写一份 OUT_ROOT/REPO_ID。

依据：包里 PickEnv._choose_slots（pick_env.py:210-229）支持
reset(seed=s, options={"target_index": k}) —— 当 k 已在该 seed 抽出的 active
集合里时，它只改任务名、不动布局。注释原话是这组对照就是「语言条件的策略
证明自己读了指令」所需要的东西。我们场上物体数（3）等于池大小，k 恒在抽出
的集合里，所以三条布局应当完全一致 —— 这一点脚本里每轮实测，不一致会打 '!'。

做法：包装 env.reset 注入 options，其余全复用 collect_lerobot 的 make_env /
run_episode / LeRobotDataset.create。现有文件一行都不改。
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 必须在 import collect_lerobot 之前设好：它在模块层（第 63-65 行）就读这两个
# 变量。用 setdefault，外面想换目录可以直接用环境变量顶掉。
os.environ.setdefault(
    "OUT_ROOT", r"G:\sim2real\so101-smolvla-sim\_ref\datasets\so101_picklift_cf")
os.environ.setdefault("REPO_ID", "local/so101_picklift_cf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

import collect_lerobot as C

# 当前要 pin 的池下标。None = 不 pin，用包里按 seed 抽出来的默认目标。
PIN = {"k": None}


def slot_colors(env) -> list[str]:
    return [s.obj.color for s in env.unwrapped._slots]


def slot_positions(env) -> dict[str, np.ndarray]:
    """按刚体名读出所有 pick_slot_* 的世界坐标，用来验三条布局是否一致。

    不读 qpos：被隐藏的槽位是 park 到世界外的，位置同样能反映布局；直接读
    xpos 不依赖哪个槽是目标。
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    out: dict[str, np.ndarray] = {}
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("pick_slot_"):
            out[name] = d.xpos[b].copy()
    return out


def same_layout(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if set(a) != set(b):
        return False
    return all(np.allclose(a[n], b[n], atol=1e-12) for n in a)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 208
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env = C.make_env()

    # 包装 reset：run_episode 里那句 env.reset(seed=seed) 会被这里接住并补上
    # options，所以不需要改 collect_lerobot.py。
    orig_reset = env.reset

    def pinned_reset(seed=None, options=None, **kw):
        opts = dict(options) if options else {}
        if PIN["k"] is not None:
            opts["target_index"] = PIN["k"]
        return orig_reset(seed=seed, options=opts or None, **kw)

    env.reset = pinned_reset

    o, _ = env.reset(seed=0)
    H, W = o["overhead_camera"].shape[:2]

    colors = slot_colors(env)
    n_col = len(colors)

    # features 必须和原数据集逐字一致，否则同一套训练配置对不上。
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
    # run_episode / rec() 里引用的是 collect_lerobot 的模块级 dataset。
    C.dataset = dataset

    print(f"数据集 -> {C.OUT_ROOT}  ({C.REPO_ID})")
    print(f"池 {colors}，每 seed 采 {n_col} 条，seed {seed_start}.."
          f"{seed_start + n_seed - 1} 共 {n_seed} 个 "
          f"-> 目标 {n_seed * n_col} 条\n")

    all_same = True
    complete = 0
    per_color: dict[str, list[int]] = {c: [0, 0] for c in colors}

    for seed in range(seed_start, seed_start + n_seed):
        # 先取一次不 pin 的布局当指纹，再逐色 pin 采。
        PIN["k"] = None
        env.reset(seed=seed)
        base = slot_positions(env)

        marks = []
        ok_all = True
        for k in range(n_col):
            PIN["k"] = k
            env.reset(seed=seed)
            same = same_layout(base, slot_positions(env))
            all_same &= same

            # run_episode 返回 (task, success, frames)，三个值（collect_lerobot.py:331）
            _, success, _ = C.run_episode(env, seed, keep_frames=False)
            per_color[colors[k]][1] += 1
            if success:
                per_color[colors[k]][0] += 1
            else:
                ok_all = False
            marks.append(f"{colors[k]}:{'OK' if success else 'NG'}{'' if same else '!'}")
        PIN["k"] = None

        complete += int(ok_all)
        print(f"seed {seed:<4} {'  '.join(marks)}", flush=True)

    print("\n--- 汇总 ---")
    for c in colors:
        got, tot = per_color[c]
        print(f"  目标={c:<6} 成功率 {got}/{tot}")
    print(f"  三种颜色全成功的 seed：{complete}/{n_seed}（配对完整的才算）")
    print(f"  同 seed 三条布局逐字节一致：{'是' if all_same else '否 —— 有布局被 pin 动了，见上面的 !'}")
    print(f"  总帧数 {dataset.meta.total_frames}")

    dataset.finalize()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
