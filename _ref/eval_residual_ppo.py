"""零噪声评测残差 PPO，同一批 seed 顺带跑一遍零残差基线。

    G:/anaconda/envs/sim2real/python.exe -u eval_residual_ppo.py \
        --model G:/.../run1/ckpt/res_60000_steps.zip --seeds 20

    # 只跑基线
    G:/anaconda/envs/sim2real/python.exe -u eval_residual_ppo.py --zero

评测 seed 从 10000 起，和训练用的 1..N 不重叠。每回合最多 300 步，
动作取策略均值（不采样），观测过训练时那份 VecNormalize 统计量。

和训练回合的区别只有一条：没有探索噪声，所以这里的成功率才和基线可比。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from residual_env import make_residual_env


FIELDS = ["cond", "seed", "success", "steps", "lift", "res_norm",
          "xy_min", "armed_steps", "window_steps", "close_step", "close_dz", "close_dyaw", "ret",
          "res0", "res1", "res2", "res3"]


def scripted_action(obs, step_m=0.003):
    """不学习的 seek 控制器：只按 xy 偏差往方块挪，用来判断「修 xy」这条路本身有没有用。

    观测里第 20:23 位是 p_obj − p_tcp。只取 xy，z 不动，每步最多 step_m。
    """
    rel = np.asarray(obs[20:23], np.float64)
    a = np.zeros(4, np.float32)
    a[:2] = np.clip(rel[:2] / step_m, -1.0, 1.0)
    return a


def run_episodes(env, vn, model, seeds, horizon, zero: bool,
                 out_path: str | None = None, cond: str = "",
                 scripted: bool = False):
    out = []
    # 逐条落盘：中途被杀掉也不至于一条记录都没有
    f = w = None
    if out_path:
        new = (not os.path.isfile(out_path)) or os.path.getsize(out_path) == 0
        f = open(out_path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
    try:
        for s in seeds:
            obs, info = env.reset(seed=s)
            norms, rsum = [], np.zeros(4)
            xy_min, armed, won, ret = 9.0, 0, 0, 0.0
            for t in range(horizon):
                if zero:
                    a = np.zeros(4, np.float32)
                elif scripted:
                    a = scripted_action(obs)
                else:
                    a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
                    a = np.asarray(a, np.float32).reshape(-1)
                obs, r, term, trunc, info = env.step(a)
                rn = float(info.get("residual_norm", 0.0))
                norms.append(rn)
                # 窗口开没开由环境说了算：确定性策略也可能在窗口内输出全零
                armed += int(bool(info.get("armed", rn > 0.0)))
                won += int(bool(info.get("window_open", rn > 0.0)))
                xy_min = min(xy_min, float(info.get("xy_error", 9.0)))
                ret += float(info.get("reward_step", 0.0))
                rsum += np.asarray(info.get("res_vector", np.zeros(4)), np.float64)
                if term or trunc:
                    break
            n = t + 1
            out.append({"cond": cond, "seed": s,
                        "success": int(bool(info.get("success"))),
                        "steps": n,
                        "lift": round(float(info.get("lift_height", 0.0)), 4),
                        "res_norm": round(float(np.mean(norms)), 4),
                        "xy_min": round(xy_min, 4),
                        "armed_steps": armed,
                        "window_steps": won,
                        "close_step": int(info.get("close_step", -1)),
                        "close_dz": round(float(info.get("close_dz", float("nan"))), 4),
                        "close_dyaw": round(float(info.get("close_dyaw", float("nan"))), 4),
                        "ret": round(ret, 4),
                        **{f"res{i}": round(float(rsum[i]) / n, 3)
                           for i in range(4)}})
            if w:
                w.writerow(out[-1])
                f.flush()
            print(f"  seed {s}: {'成功' if out[-1]['success'] else '失败'}  "
                  f"{n} 步  lift={out[-1]['lift']:.3f}  "
                  f"残差范数={out[-1]['res_norm']:.3f}  "
                  f"xy_min={out[-1]['xy_min']:.4f}  "
                  f"开窗/生效={armed}/{won} 合爪步={out[-1]['close_step']}",
                  flush=True)
    finally:
        if f:
            f.close()
    return out


def summarize(name: str, rows: list[dict]) -> float:
    k = sum(r["success"] for r in rows)
    n = len(rows)
    rate = k / n
    print(f"{name}: {k}/{n} = {rate:.2f}")
    return rate


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="graspsegexpert_230000")
    p.add_argument("--model", default=None)
    p.add_argument("--vec", default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--start", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--zero", action="store_true", help="只跑零残差基线")
    p.add_argument("--scripted", action="store_true",
                   help="不学习，用 seek 控制器直接修 xy")
    p.add_argument("--hover", type=float, default=None,
                   help="残差窗口半径（米）；给 2.0 等于从第一步就开")
    p.add_argument("--no-baseline", action="store_true",
                   help="只跑残差，不再顺带跑一遍零残差")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    seeds = list(range(args.start, args.start + args.seeds))
    env = make_residual_env(tag=args.tag, hover_xy=args.hover) if args.hover \
        else make_residual_env(tag=args.tag)

    model = vn = None
    if not args.zero and not args.scripted:
        if not args.model:
            raise SystemExit("要 --model、--zero 或 --scripted")
        vec_pkl = args.vec or os.path.join(
            os.path.dirname(args.model), "vec_" +
            os.path.basename(args.model).split("_")[1] + "_steps.pkl")
        vn = VecNormalize.load(vec_pkl, DummyVecEnv([lambda: env]))
        model = PPO.load(args.model, device="cpu")
        print(f"策略 {args.model}\n统计量 {vec_pkl}")

    out = args.out or (os.path.splitext(args.model)[0] + "_eval.csv"
                       if args.model else "eval_zero.csv")
    if os.path.isfile(out):              # 每次评测重新写，不和旧结果混在一起
        os.remove(out)

    if args.scripted:
        print("== seek 控制器（只修 xy，每步 ≤3 mm） ==")
        res = run_episodes(env, None, None, seeds, args.horizon, zero=False,
                           out_path=out, cond="scripted", scripted=True)
        summarize("seek", res)

    if vn is not None:
        print("== 残差策略（确定性） ==")
        res = run_episodes(env, vn, model, seeds, args.horizon, zero=False,
                           out_path=out, cond="residual")
        summarize("残差", res)

    if not args.no_baseline:
        print("== 零残差基线 ==")
        base = run_episodes(env, vn, model, seeds, args.horizon, zero=True,
                            out_path=out, cond="baseline")
        summarize("基线", base)

    print(f"明细 -> {out}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
