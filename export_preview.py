"""把 LeRobot v3 数据集里的视频导成一条 episode 一个 mp4，方便肉眼过一遍。

    G:/anaconda/envs/sim2real/python.exe export_preview.py --episodes 12

数据集里 videos/ 是**分片**的（几十条 episode 拼在一个 mp4 里，靠 meta/episodes 的
from/to timestamp 切），直接点开分片文件看到的是接在一起的几十条，没法看。这里按
meta 切回来，两个相机左右并排成一个 1024×512 的 mp4，另外出一张首帧总览图。

    <root>/_preview/ep000.mp4 …
    <root>/_preview/_first_frames.png

用 PyAV（lerobot 的依赖，torchcodec 在这台机器上加载不了，会回落到它）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import paths

CAM_FRONT = paths.CAM_FRONT
CAM_WRIST = paths.CAM_WRIST


def _split_by_spans(container, spans):
    """只开**一次**解码器、一遍顺扫，把落在各 span 里的帧分桶。返回 {ep: [帧]}。

    不 seek：AV1 的 mp4 关键帧稀疏，seek 到任意时间点落不准。
    也不能每个 span 各起一个 decode() 生成器——上一个被中途 return 掉之后容器就
    解不出东西了（实测抛 EOFError，第 5 条就崩）。
    """
    want = sorted(spans)
    last = want[-1][2]
    out: dict[int, list] = {i: [] for i, _, _ in want}
    for idx, frame in enumerate(container.decode(video=0)):
        if idx >= last:
            break
        for i, lo, hi in want:
            if lo <= idx < hi:
                out[i].append(frame.to_ndarray(format="rgb24"))
                break
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="导出 episode 预览视频")
    # 和 collect_demos.py 同一个根：paths.DATA_ROOT 是遥操作那份（data/so101_smolvla_sim）
    ap.add_argument("--root", default=str(paths.DATA_ROOT.parent / "so101_smolvla_sim_scripted"))
    ap.add_argument("--episodes", type=int, default=12, help="导出前 N 条（0 = 全部）")
    ap.add_argument("--out", default=None, help="输出目录，默认 <root>/_preview")
    a = ap.parse_args()

    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(a.root)
    ds = LeRobotDataset("so101_sim/scripted_pick_cube", root=str(root))
    ep = ds.meta.episodes.to_pandas()
    n = len(ep) if a.episodes <= 0 else min(a.episodes, len(ep))
    outdir = Path(a.out) if a.out else root / "_preview"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(ep)} 条 episode，导出前 {n} 条 → {outdir}")

    cols = {c: (f"videos/observation.images.{c}/file_index",
                f"videos/observation.images.{c}/from_timestamp",
                f"videos/observation.images.{c}/to_timestamp")
            for c in (CAM_FRONT, CAM_WRIST)}
    # 每个分片开一次，顺序解一遍就把落在这个分片里的 episode 全部切出来
    jobs = {}
    for i in range(n):
        for c, (fi, ft, tt) in cols.items():
            f = int(ep[fi][i])
            jobs.setdefault((c, f), []).append(
                (i, round(float(ep[ft][i]) * ds.fps), round(float(ep[tt][i]) * ds.fps)))

    thumbs = {}
    for (c, f), spans in sorted(jobs.items()):
        path = root / "videos" / f"observation.images.{c}" / "chunk-000" / f"file-{f:03d}.mp4"
        spans.sort()
        with av.open(str(path)) as cont:
            got = _split_by_spans(cont, spans)
        for i, _, _ in spans:
            frames = got[i]
            if not frames:
                print(f"  ⚠️  ep{i} {c} 取到 0 帧")
                continue
            thumbs.setdefault(i, {})[c] = frames[0]
            _write(outdir / f"ep{i:03d}_{'front' if c == CAM_FRONT else 'wrist'}.mp4",
                   frames)
        print(f"  {c:28s} 分片 file-{f:03d} 出 {len([s for s in spans if got[s[0]]])} 条")

    _grid(thumbs, outdir / "_first_frames.png", n)
    print(f"总览图 → {outdir / '_first_frames.png'}")


def _write(dst: Path, frames: list[np.ndarray]):
    import av
    h, w, _ = frames[0].shape
    with av.open(str(dst), "w", format="mp4") as out:
        st = out.add_stream("libx264", rate=paths.FPS)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "23", "preset": "veryfast"}
        for im in frames:
            fr = av.VideoFrame.from_ndarray(im, format="rgb24")
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def _grid(thumbs: dict, dst: Path, n: int, cell: int = 128, cols: int = 10):
    """首帧缩略图铺一张网格：一眼看出方块位置/偏航铺开了没有。"""
    import av
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i in sorted(thumbs):
        im = thumbs[i].get(CAM_FRONT)
        if im is None:
            continue
        s = im.shape[0] // cell
        small = im[::s, ::s][:cell, :cell]
        r, c = divmod(i, cols)
        canvas[r * cell:r * cell + small.shape[0], c * cell:c * cell + small.shape[1]] = small
    with av.open(str(dst), "w") as out:
        st = out.add_stream("png")
        st.width, st.height, st.pix_fmt = canvas.shape[1], canvas.shape[0], "rgb24"
        fr = av.VideoFrame.from_ndarray(canvas, format="rgb24")
        for pkt in st.encode(fr):
            out.mux(pkt)


if __name__ == "__main__":
    main()
