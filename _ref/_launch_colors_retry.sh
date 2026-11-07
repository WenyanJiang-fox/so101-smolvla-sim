#!/bin/bash
# 服务器上 8 路并行采 colors_retry：颜色轮换 + 故意抓空重抓。见 docs/训练数据谱系.md §1.10。
#
#   bash _launch_colors_retry.sh [工人数] [每人几个 seed] [起始 seed]
#
# 和 _launch_colors.sh 只差一处：多了 RETRY=1 RETRY_PROB=0.5。
# collect_colors.py 一个字没改——RETRY / RETRY_PROB 是 collect_lerobot.py:106-107 的
# 模块级常量，在这里用 env 传进去，子进程 import collect_lerobot 时就生效。
#
# 为什么改到服务器跑而不是本地：本机 8 核，6 路并发时写视频的 SVT-AV1 每路开 4 个
# 编码线程、24 线程挤 8 核互相饿死，实测 3 个 worker 0% CPU 卡死；降到 4 路才稳，
# 但速率只有 3.8 条/分钟，624 条要 2.6 小时。服务器 112 核，8 路轻松。
#
# 骨架和配色轮换逻辑与 colors 完全一致（104 layout × 2 配色 × 3 目标 = 624 条），
# 这样两批能一一对上做对照。
#
# 分片按 seed 切，不按 (配色, 目标) 切：同一个 layout 的 6 条必须落在同一个 worker 里，
# 否则 collect_colors.py 的轮换一致性检查（pairing）跨进程做不了。
# 默认 8 人 × 13 个 seed，seed 104..207。
#
# 与本地版一样带起飞前清场：删完目录复查，删不掉就中止，避免新 worker 撞上残缺目录
# （LeRobotDataset.create 的 exist_ok=False 会直接崩）。
set -u

PY=/root/miniconda3/envs/sim2real/bin/python
PORT=/root/autodl-tmp/port
DS=/root/autodl-tmp/datasets
N_W=${1:-8}
N_S=${2:-13}
S0=${3:-104}
TAG=colors_retry

cd "$PORT" || exit 1

# ---- 起飞前清场 ----
for i in $(seq 0 $((N_W - 1))); do
  root="$DS/${TAG}_p${i}"
  rm -rf "$root" 2>/dev/null
  if [ -e "$root" ]; then
    echo "p$i 的旧目录删不掉，有进程占着句柄：$root"
    echo "先把残留的采集进程杀干净。本轮中止，一个 worker 都没启动。"
    exit 1
  fi
done
echo "清场完成，$N_W 份 shard 目录都确认是空的"

# ---- 拉起 ----
for i in $(seq 0 $((N_W - 1))); do
  s=$((S0 + i * N_S))
  root="$DS/${TAG}_p${i}"
  env MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 \
      RETRY=1 RETRY_PROB=0.5 \
      OUT_ROOT="$root" REPO_ID="local/${TAG}_p${i}" \
      setsid nohup "$PY" -u collect_colors.py "$N_S" "$s" \
      > "/tmp/${TAG}_${i}.log" 2>&1 &
  echo "worker $i  seed $s..$((s + N_S - 1))  -> $root   日志 /tmp/${TAG}_${i}.log"
done

echo
echo "已全部拉起，日志里找「写入 N/M 条」就是跑完了。"
echo "采完合并："
echo "  cd $PORT && DATASETS_ROOT=$DS $PY merge_colors.py $N_W $TAG"
