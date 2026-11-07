"""把并行采的 N 份 shard 合并成一份。

    python merge_colors.py [份数] [标签]

标签默认 colors，读 <数据盘>/datasets/<标签>_p0..p<N-1>，
写出 <数据盘>/datasets/so101_picklift_<标签>。

合并会把各 shard 的视频重新编码一遍（concatenate_videos=True），所以比拷贝慢，别中途杀。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("DATASETS_ROOT", "/root/autodl-tmp/datasets"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tag = sys.argv[2] if len(sys.argv) > 2 else "colors"

    roots = [BASE / f"{tag}_p{i}" for i in range(n)]
    missing = [r for r in roots if not (r / "meta" / "info.json").is_file()]
    if missing:
        print("这些 shard 没有 meta/info.json：" + ", ".join(str(m) for m in missing))
        return 1

    aggr_root = BASE / f"so101_picklift_{tag}"
    if aggr_root.exists():
        print(f"{aggr_root} 已存在，先挪走再合，避免半成品混进去")
        return 1

    from lerobot.datasets.aggregate import aggregate_datasets

    print(f"{n} 份 shard -> {aggr_root}")
    aggregate_datasets(
        repo_ids=[f"local/{tag}_p{i}" for i in range(n)],
        aggr_repo_id=f"local/so101_picklift_{tag}",
        roots=roots,
        aggr_root=aggr_root,
    )
    print(f"\n合并完成 -> {aggr_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
