"""录一条完整 episode 的视频，两个相机左右并排。

    G:/anaconda/envs/sim2real/python.exe record_episode.py --out episode.mp4
    G:/anaconda/envs/sim2real/python.exe record_episode.py --out b.mp4 --cube 0.28 -0.02 --yaw 20

就是把 expert 跑一次取放、把 30fps 的帧存下来。不判成功、不调参数、不写数据集。
--cube/--yaw 挪方块的位置和朝向（默认 = paths 里的固定值）。范围按采集时的抖动来。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import expert
import paths
import sim_env
from expert import Expert


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="录一条完整 episode")
    ap.add_argument("--out", default="episode.mp4")
    ap.add_argument("--cube", nargs=2, type=float, metavar=("X", "Y"),
                    help=f"方块位置（默认 {paths.PICK_X} {paths.PICK_Y}）")
    ap.add_argument("--yaw", type=float, default=0.0, help="方块偏航角（度）")
    a = ap.parse_args()

    env = sim_env.SO101SimEnv()
    e = Expert(controller=env.c)
    env.observation()

    cube_xy = tuple(a.cube) if a.cube else (paths.PICK_X, paths.PICK_Y)
    print(f"方块 {cube_xy[0]*1000:.0f},{cube_xy[1]*1000:.0f}mm  偏航 {a.yaw:.0f}°")

    frames, step = [], [0]

    def on_frame():
        step[0] += 1
        if step[0] % sim_env.STEPS_PER_FRAME == 0:
            obs = env.observation()
            frames.append(np.hstack([obs[f"observation.images.{paths.CAM_FRONT}"],
                                     obs[f"observation.images.{paths.CAM_WRIST}"]]))

    # 必须传一个 trace 列表：run_episode 是 `return trace`，trace 默认 None，
    # 所以不传的话它**成功也返回 None**，拿返回值判成败会全判成失败。
    kw = {} if a.cube is None else {"cube_xy": cube_xy, "cube_yaw": np.radians(a.yaw)}
    tr = []
    if e.run_episode(on_frame=on_frame, trace=tr, **kw) is None:
        print("IK 解不出路点，episode 没跑成——换组位置")
        return 1
    print(f"按姿态解出的夹持高度 z_bot = {e.pick_z_bot*1000:.1f}mm"
          f"（固定值 {expert.PICK_Z_BOT*1000:.0f}mm）")
    print(f"{'阶段':<12}{'方块 x,y,z (mm)':<26}{'触点':>4}{'爪角°':>8}")
    for st, x, y, z, c, g in tr:
        print(f"{st:<12}{x*1000:7.1f},{y*1000:7.1f},{z*1000:7.1f}      {c:>4}{g:>8.0f}")
    print(f"放置判据 place_xy={(paths.PLACE_X, paths.PLACE_Y)} → {'成功' if e.success() else '未达标'}")

    import av
    h, w, _ = frames[0].shape
    with av.open(a.out, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=paths.FPS)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "23", "preset": "veryfast"}
        for im in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(im), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    print(f"→ {a.out}（{len(frames)} 帧 ≈ {len(frames)/paths.FPS:.0f}s）"
          f"  左=front 右=wrist")


if __name__ == "__main__":
    sys.exit(main())
