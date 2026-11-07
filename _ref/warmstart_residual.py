"""用 seek 控制器当示范，先把残差策略的动作头训成它，再交给 PPO 微调。

    G:/anaconda/envs/sim2real/python.exe -u warmstart_residual.py --episodes 24

seek 控制器：观测里 p_obj − p_tcp 的 xy 除以 3 mm 直接当动作，z 和夹爪保持 0。
评测里它拿到 19/60（基座 12/60），而 PPO 从零学 5 轮都没学到这个行为，
所以先用监督学习把这一步固定下来，再让 PPO 在这个起点上改进。

产物写到 residual_runs/warmstart/：init_policy.pt（策略权重）、init_vec.pkl（观测统计量）。
训练时用 train_residual_ppo.py --init-dir 指过来。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

import eval_picklift6 as E
from eval_residual_ppo import scripted_action
from residual_env import make_residual_env


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="graspsegexpert_230000")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=os.path.join(
        E.PROJECT, "residual_runs", "warmstart"))
    args = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(args.out, exist_ok=True)

    def build():
        # 训练时残差从第一步就允许，和评测里跑出 19/60 的那版一致
        return make_residual_env(tag=args.tag, hover_xy=2.0)

    vec = DummyVecEnv([build])
    vn = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env = vec.envs[0]

    # --- 采示范 ---
    X, Y = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep + 1)
        for t in range(args.horizon):
            a = scripted_action(obs)
            X.append(np.asarray(obs, np.float32).copy())
            Y.append(a.copy())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        print(f"  示范 {ep + 1}/{args.episodes}：{len(X)} 帧", flush=True)

    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    vn.obs_rms.update(X)                      # 统计量和训练时用同一份
    # 示范里 dz 和夹爪两维恒为 0，方差接近 0，归一化会变成除以 ~0。
    # 给所有维度设方差下限，BC 和训练看到的是同一套尺度。
    vn.obs_rms.var = np.maximum(vn.obs_rms.var, 1e-2)
    # obs_rms 的统计量是 float64，一减就把整批示范变成 float64，喂给 float32 的网络会报
    # 「mat1 and mat2 must have the same dtype」。这里显式转回来。
    Xn = np.clip((X - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var),
                 -10.0, 10.0).astype(np.float32)

    model = PPO(
        "MlpPolicy", vn,
        learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.002,
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -2.5},
        seed=0, verbose=0, device="cpu",
    )
    pol = model.policy
    opt = torch.optim.Adam(
        list(pol.mlp_extractor.policy_net.parameters())
        + list(pol.action_net.parameters()), lr=args.lr)

    xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(Y)
    n = len(xt)
    for ep in range(args.epochs):
        idx = np.random.default_rng(ep).permutation(n)
        tot = 0.0
        for i in range(0, n, 256):
            b = idx[i:i + 256]
            pred = pol.action_net(pol.mlp_extractor.policy_net(xt[b]))
            loss = torch.nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep}: mse={tot / n:.4f}", flush=True)

    pol.save(os.path.join(args.out, "init_policy.pt"))
    vn.save(os.path.join(args.out, "init_vec.pkl"))
    print(f"初始化权重 -> {os.path.join(args.out, 'init_policy.pt')}")
    print(f"观测统计量 -> {os.path.join(args.out, 'init_vec.pkl')}")
    vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
