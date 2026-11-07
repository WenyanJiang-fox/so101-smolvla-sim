"""在 SmolVLA 上面训一个残差 PPO。

    # 冒烟：残差恒为 0，必须复现基座策略的行为（第 1 阶段验收）
    G:/anaconda/envs/sim2real/python.exe -u train_residual_ppo.py --smoke

    # 正式训练
    G:/anaconda/envs/sim2real/python.exe -u train_residual_ppo.py --timesteps 300000

产物在 checkpoints 同级的 residual_runs/<tag>/：模型、VecNormalize 统计量、
success.csv（每回合成没成）。服务器上用 PROJECT_ROOT 覆盖 E.PROJECT。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

import eval_picklift6 as E
from residual_env import make_residual_env


def build_env(args):
    env = make_residual_env(
        tag=args.tag,
        n_action_steps=args.nas,
        n_distractors=args.n_distractors,
        horizon=args.horizon,
        tcp_delta_m=args.tcp_delta,
        grip_delta_rad=args.grip_delta,
        hover_xy=args.hover,
    )
    return env


def smoke(args) -> int:
    """零残差跑一条，看末端抬升和成功情况；确认管道通了再开训。"""
    env = build_env(args)
    obs, info = env.reset(seed=args.seed)
    total = 0.0
    best_lift = 0.0
    for t in range(args.horizon):
        obs, r, term, trunc, info = env.step(np.zeros(4, np.float32))
        total += r
        best_lift = max(best_lift, float(info.get("lift_height", 0.0)))
        if t % 25 == 0:
            pct = env.get_wrapper_attr("base_grip_pct")
            print(f"  {t:>3} 帧  夹爪命令 {pct:6.1f}%  r={r:+.4f} "
                  f"lift={info.get('lift_height', 0):.3f} "
                  f"grasped={info.get('is_grasped')}")
        if term or trunc:
            break
    n = env.get_wrapper_attr("n_total")
    te = env.get_wrapper_attr("t_env")
    tb = env.get_wrapper_attr("t_base")
    print(f"零残差：{t + 1} 步  回报 {total:+.3f}  最大抬升 {best_lift:.3f}  "
          f"success={info.get('success')}")
    tc = env.get_wrapper_attr("t_chunk")
    tp = env.get_wrapper_attr("t_pol")
    nc = env.get_wrapper_attr("n_chunk")
    print(f"耗时：env.step {te / n * 1000:.1f} ms/步  "
          f"合计 {(te + tb) / n * 1000:.1f} ms/步")
    print(f"  基座 {tb / n * 1000:.1f} ms/步 = 模型 {tc / n * 1000:.1f}"
          f"（{nc} 次，{tc / max(nc, 1) * 1000:.0f} ms/次）"
          f" + policy准备 {(tp - tc) / n * 1000:.1f}"
          f" + adapter预处理 {(tb - tp) / n * 1000:.1f}")
    env.close()
    return 0


def train(args) -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out = args.out or os.path.join(E.PROJECT, "residual_runs", args.tag)
    ckpt_dir = os.path.join(out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(out, "success.csv")

    class SaveEveryN(BaseCallback):
        """模型和 VecNormalize 统计量成对落盘，续训才是逐位接上的。"""

        def __init__(self, every: int, vec):
            super().__init__(0)
            self.every = every
            self.vec = vec

        def _on_step(self) -> bool:
            if self.every and self.n_calls % self.every == 0:
                step = self.num_timesteps
                self.model.save(os.path.join(ckpt_dir, f"res_{step}_steps"))
                self.vec.save(os.path.join(ckpt_dir, f"vec_{step}_steps.pkl"))
                print(f"  [存档] {step} 步 -> {ckpt_dir}", flush=True)
            return True

    class SuccessLogger(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.eps = 0
            self.wins = 0
            self.res = []
            new = (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0
            if not new:                      # 续训接着上次的回合号往下数
                with open(csv_path, encoding="utf-8") as f:
                    self.eps = max(len(f.readlines()) - 1, 0)
            self._f = open(csv_path, "a", newline="", encoding="utf-8")
            self._w = csv.writer(self._f)
            # 逐回合累加器：DummyVecEnv 里只有一个环境，但按 i 分开写不会错
            self.acc = {}
            if new:
                self._w.writerow(["steps", "episode", "success", "len", "ret",
                                  "lift", "res_norm_mean", "xy_min",
                                  "res0", "res1", "res2", "res3"])

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            dones = self.locals.get("dones")
            # 不能用 `or []`：dones 是长度 1 的 ndarray，bool(np.array([False])) 是
            # False，`x or []` 会在未结束的步上悄悄换成空列表，累加器每步都被跳过。
            if infos is None or dones is None:
                return True
            for i, d in enumerate(dones):
                info = infos[i] if i < len(infos) else {}
                a = self.acc.setdefault(i, {"n": 0, "ret": 0.0, "xy": 9.0,
                                            "res": np.zeros(4)})
                a["n"] += 1
                a["ret"] += float(info.get("reward_step", 0.0))
                a["xy"] = min(a["xy"], float(info.get("xy_error", 9.0)))
                a["res"] = a["res"] + np.asarray(
                    info.get("res_vector", np.zeros(4)), np.float64)
                if not d:
                    continue
                n = max(a["n"], 1)
                self.eps += 1
                ok = int(bool(info.get("success", False)))
                self.wins += ok
                rn = float(info.get("residual_norm", 0.0))
                self.res.append(rn)
                self._w.writerow([self.num_timesteps, self.eps, ok, a["n"],
                                  round(a["ret"], 4),
                                  round(float(info.get("lift_height", 0.0)), 4),
                                  round(rn, 4), round(a["xy"], 4)]
                                 + [round(float(v) / n, 3) for v in a["res"]])
                self.acc[i] = {"n": 0, "ret": 0.0, "xy": 9.0, "res": np.zeros(4)}
                self._f.flush()
                if self.eps % 10 == 0:
                    print(f"  [ep {self.eps}] 成功率 {self.wins / self.eps:.2f}  "
                          f"最近10条残差范数 {np.mean(self.res[-10:]):.3f}", flush=True)
            return True

        def _on_training_end(self):
            self._f.close()

    vec = DummyVecEnv([lambda: build_env(args)])

    if args.resume:
        zip_path = args.resume
        step = os.path.basename(zip_path).split("_")[1]
        vec_pkl = os.path.join(os.path.dirname(zip_path), f"vec_{step}_steps.pkl")
        if not os.path.isfile(vec_pkl):
            raise FileNotFoundError(f"缺配套的统计量 {vec_pkl}，不能只续模型")
        vec = VecNormalize.load(vec_pkl, vec)
        model = PPO.load(zip_path, env=vec, device="cpu")
        print(f"续训自 {zip_path}（已 {model.num_timesteps} 步；"
              f"--timesteps 要给总步数）")
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if args.init_dir:
            vec = VecNormalize.load(
                os.path.join(args.init_dir, "init_vec.pkl"), vec)
            # 冻结统计量：不冻结的话 obs_rms 每步更新，策略看到的输入尺度一直在变，
            # 和 BC 时不一致（实测 5000 步后第 32 维偏移 158σ）。
            vec.training = False
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            # 初始标准差不能是默认的 1.0：探索噪声乘上动作上限就是每步的随机位移。
            # 0.082 对应每步 ~0.25 mm，50 步的 chunk 里只漂 ~1.8 mm，不会带偏接近轨迹。
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -2.5},
            seed=args.seed,
            verbose=0,
            device="cpu",     # 策略是 MLP，放 GPU 只会更慢；显存留给 SmolVLA
        )
        import torch
        if args.init_dir:
            # 从 seek 示范的监督学习权重起步：训练成功率直接站到 seek 那版（19/60）附近
            path = os.path.join(args.init_dir, "init_policy.pt")
            # torch 2.11 默认 weights_only=True，而政策对象存档里带 gymnasium.spaces.Box
            # 这类非张量全局，必须显式放开；文件是本机 warmstart 脚本自己写的。
            # BasePolicy.save 写的是 {"state_dict": ..., "data": ...}，不是裸 state_dict
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.policy.load_state_dict(ckpt["state_dict"])
            print(f"策略权重 <- {path}")
        else:
            # 动作头清零：开局残差恒为 0，行为和基座逐位一致，训练成功率从基座水平起步，
            # 只有真的能改进时才离开零点。
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()

    model.learn(total_timesteps=args.timesteps,
                callback=[SaveEveryN(args.save_freq, vec), SuccessLogger()],
                progress_bar=False,
                # 续训必须保住计数器，否则 --timesteps 会被当成「再跑这么多」
                reset_num_timesteps=not args.resume)

    model.save(os.path.join(out, "residual_ppo"))
    vec.save(os.path.join(out, "vecnormalize.pkl"))
    print(f"模型 -> {os.path.join(out, 'residual_ppo.zip')}")
    print(f"统计量 -> {os.path.join(out, 'vecnormalize.pkl')}")
    print(f"逐回合记录 -> {csv_path}")
    vec.close()
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="graspsegexpert_230000")
    p.add_argument("--nas", type=int, default=None)
    p.add_argument("--n-distractors", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--tcp-delta", type=float, default=0.003)
    p.add_argument("--hover", type=float, default=2.0,
                   help="残差窗口半径（米）；2.0 = 从第一步就开（评测里 19/60 vs 14/60）")
    p.add_argument("--grip-delta", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--save-freq", type=int, default=2500)
    p.add_argument("--init-dir", default=None,
                   help="warmstart_residual.py 的输出目录（init_policy.pt + init_vec.pkl）")
    p.add_argument("--resume", default=None,
                   help="ckpt/res_<N>_steps.zip；同名目录下必须有 vec_<N>_steps.pkl")
    args = p.parse_args()
    return smoke(args) if args.smoke else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
