#!/bin/bash
# 云端 SmolVLA 训练（AutoDL /root，conda env sim2real）。
#
#   bash /root/train_smolvla.sh smoke   # 50 步，验配置 + 看显存
#   bash /root/train_smolvla.sh         # 正式训练
#
# 数据和输出都放**数据盘** /root/autodl-tmp（/dev/sdb，50G），不放系统盘：
# 系统盘 overlay 只有 30G，miniconda 就吃掉 17G，5 个 checkpoint 再占 6.5G 就满了。
# 见 _upload_cloud.py 上传流程。
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sim2real

# SmolVLM2-500M 权重已由 _upload_model.py 手工放进 HF 缓存（云端直连 hf-mirror
# 下到 141MB 就断）。设成离线，让缺什么东西就立刻报错，而不是挂在网络上。
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
  STEPS=50; SAVE=1000000; LOG=10
else
  # 20000 步不是照抄上次的 10000：上次是 6 条 / 3168 帧，batch 8 → 396 步/epoch，
  # 10000 步等于 25 个 epoch（难怪过拟合也难怪照样 0/8）。这次 60 条 / 31680 帧，
  # 10000 步只有 2.5 个 epoch，对 VLA 明显不够。20000 步 ≈ 5 epoch ≈ 1.5h。
  # 想改回来：STEPS=10000 bash train_smolvla.sh
  STEPS="${STEPS:-20000}"; SAVE=2000; LOG=50
fi

lerobot-train \
  --dataset.repo_id=so101_sim/scripted_pick_cube \
  --dataset.root=/root/autodl-tmp/data/so101_smolvla_sim_scripted \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_peft=false \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --batch_size="${BATCH:-8}" \
  --steps=$STEPS \
  --save_freq=$SAVE \
  --log_freq=$LOG \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/so101_scripted_$MODE
