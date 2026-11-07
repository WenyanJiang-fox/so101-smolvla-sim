#!/bin/bash
# 8 路并行采颜色轮换数据。每个 worker 写自己一份数据集，采完再 aggregate 合并。
#
#   bash _launch_colors.sh [工人数] [每人几个 seed] [起始 seed] [标签]
#
# 默认 8 人 × 13 个 seed，seed 104..207，一共 104 个布局 × 2 套配色 × 3 个目标 = 624 条。
# 分离到 setsid 里跑，SSH 断了也不会被 SIGHUP 打死。日志在 /tmp/<标签>_<i>.log。
set -u

PY=/root/miniconda3/envs/sim2real/bin/python
PORT=/root/autodl-tmp/port
N_W=${1:-8}
N_S=${2:-13}
S0=${3:-104}
TAG=${4:-colors}

cd "$PORT" || exit 1
for i in $(seq 0 $((N_W - 1))); do
  s=$((S0 + i * N_S))
  root=/root/autodl-tmp/datasets/${TAG}_p${i}
  rm -rf "$root"
  env MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 \
      OUT_ROOT="$root" REPO_ID="local/${TAG}_p${i}" \
      setsid nohup "$PY" -u collect_colors.py "$N_S" "$s" \
      > "/tmp/${TAG}_${i}.log" 2>&1 &
  echo "worker $i  seed $s..$((s + N_S - 1))  -> $root  日志 /tmp/${TAG}_${i}.log"
done
echo "已全部拉起，日志里找「写入 N/M 条」就是跑完了"
