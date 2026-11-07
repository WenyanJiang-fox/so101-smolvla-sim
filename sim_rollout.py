"""闭环评估：让训好的 SmolVLA 策略在 MuJoCo 里自己抓方块。

    G:/anaconda/envs/sim2real/python.exe sim_rollout.py --episodes 5

和训练时完全同一口径：观测 = state(6,)+双相机，动作 = 6 个舵机目标角 0~180°。
策略一次出 50 步的动作块（chunk_size=50），内部排队，所以循环里每步只管取一个。

这是唯一能检验策略行不行的方式——训练 loss 低只说明它背下了训练集里那些轨迹。

**开跑前先看它打印的「用这个 checkpoint」那一行**：历史上这里写死过 010000，
训练存到 020000 了它还在跑老模型，结果拿"老模型的 0/8"当成"新模型不行"查了一整轮。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import paths
import sim_env

# 注意要指到 pretrained_model 这一层：lerobot 的 checkpoint 结构是
# <step>/pretrained_model/config.json，from_pretrained 找的是直接含 config.json 的目录，
# 指到 <step>/ 会报 "config.json not found"。
def latest_ckpt() -> Path:
    """checkpoints/ 下编号最大的那个。

    这里原来写死 "010000"，踩过一个查了一整轮的坑：训练存到 020000，评估却还在跑
    010000，于是"新模型 0/8"其实是老模型的结果。**永远别写死步数。**
    """
    root = paths.PROJECT / "checkpoints"
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not dirs:
        raise SystemExit(f"{root} 下没有编号 checkpoint")
    # 按 int 比大小，但**返回原目录名**：名字是零填充的 "010000"，
    # str(int("010000")) 会拼成不存在的 "10000"。
    return max(dirs, key=lambda p: int(p.name)) / "pretrained_model"


def to_batch(obs, task):
    """env.observation() → 策略输入。图像 uint8 HWC → float CHW /255，和数据集一致。"""
    batch = {"observation.state":
             torch.from_numpy(np.asarray(obs["observation.state"])).float().unsqueeze(0)}
    for ck in (paths.CAM_FRONT, paths.CAM_WRIST):
        img = np.ascontiguousarray(obs[f"observation.images.{ck}"])   # (H,W,3) uint8
        assert img.dtype == np.uint8, f"期望 uint8，拿到 {img.dtype}"
        t = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        batch[f"observation.images.{ck}"] = t
    batch["task"] = [task]
    return batch


def load_policy(ckpt: Path, device):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"加载策略 {ckpt} ...")
    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    policy.to(device)
    policy.eval()
    # 归一化器（均值/方差）就存在 checkpoint 里，必须一起加载，否则动作尺度全错
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"  参数量 {sum(p.numel() for p in policy.parameters())/1e6:.0f}M，"
          f"chunk={policy.config.chunk_size} 设备={device}")
    return policy, pre, post


def compose_frame(obs, label: str):
    """front | wrist 左右拼成一张图，左上角写 label，返回 cv2.VideoWriter 要的 BGR。

    MuJoCo 的 render() 给的是 **RGB**（数据集也是按 RGB 存的），cv2 要 BGR，不转的话
    红蓝互换——方块是红的，转反了一眼就能看出来。
    """
    import cv2
    f = obs[f"observation.images.{paths.CAM_FRONT}"]
    w = obs[f"observation.images.{paths.CAM_WRIST}"]
    c = cv2.cvtColor(np.hstack([f, w]), cv2.COLOR_RGB2BGR)
    cv2.putText(c, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return c


def main():
    ap = argparse.ArgumentParser(description="SmolVLA 闭环评估")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600, help="单条上限帧数（30fps，600=20s）")
    ap.add_argument("--jitter", type=float, default=0.015, help="方块位置抖动半径 m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="默认自动取 checkpoints/ 下编号最大的")
    ap.add_argument("--viewer", dest="viewer", action="store_true", default=True)
    ap.add_argument("--no-viewer", dest="viewer", action="store_false")
    ap.add_argument("--video", default=None,
                    help="把每条录成 front|wrist 拼接的 mp4 存到这个目录（不传则不录）")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(a.ckpt) if a.ckpt else latest_ckpt()
    print(f"用这个 checkpoint：{ckpt}")
    policy, pre, post = load_policy(ckpt, device)

    env = sim_env.SO101SimEnv()
    env.observation()      # 预热离屏 renderer

    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.m, env.d)
        print("已开 viewer（关掉窗口即停）")

    video_dir = None
    if a.video:
        import cv2
        video_dir = Path(a.video)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"录像输出到 {video_dir}（每条一个 mp4，front|wrist 拼接）")

    rng = np.random.default_rng(a.seed)
    results = []
    print(f"\n{'#':<3}{'方块xy(mm)':>16}{'偏航°':>7}{'结束z(mm)':>10}"
          f"{'离目标(mm)':>11}{'触点数':>7}{'用帧':>6}{'耗时s':>7}  结果")
    for ep in range(a.episodes):
        xy = (paths.PICK_X + rng.uniform(-a.jitter, a.jitter),
              paths.PICK_Y + rng.uniform(-a.jitter, a.jitter))
        yaw = float(rng.uniform(-np.pi / 6, np.pi / 6))
        obs = env.reset(cube_xy=xy, cube_yaw=yaw)
        policy.reset()          # 清掉上一条的动作块队列，否则会串味
        if viewer:
            viewer.sync()

        # 先写临时名，跑完知道成败再改成 ep01_OK.mp4 / ep01_FAIL.mp4——
        # 文件名不带结果的话，10 条视频混在一个目录里得逐个点开才知道看哪条。
        vw = None
        if video_dir is not None:
            h, w = obs[f"observation.images.{paths.CAM_FRONT}"].shape[:2]
            vw = cv2.VideoWriter(str(video_dir / f"_tmp{ep+1:02d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), paths.FPS, (w * 2, h))
            vw.write(compose_frame(obs, f"ep{ep+1} frame 0"))

        t0 = time.time()
        n = 0
        for n in range(1, a.steps + 1):
            batch = pre(to_batch(obs, paths.TASK))
            with torch.inference_mode():
                act = policy.select_action(batch)
            act = post(act)
            act_deg = act.squeeze(0).detach().cpu().numpy().astype(np.float64)
            # n_frames 的单位是 **30fps 帧**，step() 内部再乘 STEPS_PER_FRAME 换成物理步。
            # 这里原来传的是 n_frames=STEPS_PER_FRAME，等于 17×17=289 物理步 ≈0.58s/动作，
            # 比训练时慢 17 倍。后果：位置伺服有充足时间走完每个目标点，夹爪过早闭合、
            # 合在方块前面，然后整只爪子把方块推走——症状就是"10 条没一条把方块抬起来"。
            # 采集端同口径见 collect_demos.py 的 on_frame（每 17 物理步记一帧）。
            obs = env.step(act_deg)
            if vw is not None:
                vw.write(compose_frame(obs, f"ep{ep+1} frame {n}"))
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    print("viewer 被关掉了，提前结束")
                    a.episodes = ep + 1
                    break
            if env.is_success():
                break

        p = env.cube_pos()
        ok = env.is_success()
        if vw is not None:
            vw.release()
            tmp = video_dir / f"_tmp{ep+1:02d}.mp4"
            tmp.replace(video_dir / f"ep{ep+1:02d}_{'OK' if ok else 'FAIL'}.mp4")
        dist = float(np.hypot(p[0] - paths.PLACE_X, p[1] - paths.PLACE_Y)) * 1000
        results.append(ok)
        print(f"{ep+1:<3}{xy[0]*1000:7.1f},{xy[1]*1000:7.1f}{np.degrees(yaw):7.1f}"
              f"{p[2]*1000:10.1f}{dist:11.1f}{env.n_cube_contacts():7d}{n:6d}"
              f"{time.time()-t0:7.1f}  {'✅ 成功' if ok else '❌ 失败'}")

    n_ok = sum(results)
    print(f"\n成功率 {n_ok}/{len(results)}"
          + (f"（{n_ok/len(results)*100:.0f}%）" if results else ""))
    print("作为对照：scripted expert 在同一抖动下约 4/5。")
    if viewer:
        viewer.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
