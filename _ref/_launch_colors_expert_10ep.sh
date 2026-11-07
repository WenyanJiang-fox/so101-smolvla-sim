#!/bin/bash
# 用颜色轮换的纯专家数据训 10 epoch，只训动作专家（VLM 冻结）。
# 2026-09-22 用户指定：「只训动作专家，10 epoch」，起点保持 cf_retry_expert/200000。
#
# === 这批数据是什么 ===
#
# so101_picklift_colors：104 个布局 × 2 套配色 × 3 个钉住的目标 = 624 条，实收 618 条
# （6 条专家自己没抓稳，clear_episode_buffer 丢掉）。78236 帧，三条指令 206/205/207。
# 全部是专家从初始位姿走完的整条轨迹，不含策略段。
#
# 关键是轮换怎么做的：同一个 seed 下两套配色共用同一份物体摆位（位置由 rng 决定，
# seed 一样抽出的下标和位置就一样），只有「哪个颜色占哪个槽位」变了。目标用
# reset(options={"target_index": k}) 钉住。于是固定的 (seed, k) 下，两套配色
# **目标位置逐位相同，只有指令里的颜色词不同**——这正是语言条件策略需要的反事实配对，
# 「去第几个槽位」这条捷径不成立：同一个槽位在不同配色下对应不同颜色词。
# 采集日志里 8 个 worker 全部报 39/39 组一致、异常 0。
#
# === 步数怎么来的 ===
#
# 看 epoch 不看步数。78236 帧 / batch 32 = 2445 步/epoch，10 epoch = 24450，
# 取整 24500。单任务、只训 action expert、视觉编码器冻结，10 epoch 在常规区间。
# steps 和两组 scheduler_decay_steps 必须同值，否则余弦退火对不上
# （lerobot/optim/schedulers.py:149 的自动缩放只在 num_training_steps < num_decay_steps 时触发）。
#
# === 不 resume 时哪组 LR 参数生效 ===
#
# lerobot/configs/train.py:246  active_cfg = self.trainable_config   （= 策略，不是顶层）
# lerobot/configs/train.py:275  if use_policy_training_preset and not resume:
# lerobot/configs/train.py:276      self.scheduler = active_cfg.get_scheduler_preset()
# 不 resume 时顶层 --scheduler.* 被策略 preset 覆盖，真正生效的是 --policy.* 那组；
# 顶层照写同值，只为万一 resume 语义变了不至于跑飞。
#
# LR 仍 0.00003 余弦退到 0，不动。batch 32 沿用上一轮（batch 8 时只占 3424 MiB，
# 有富余，所以乘 4）。
#
# save_freq=2450：10 个档，每档约 1 epoch，共约 13G。数据盘还剩 42G。
cd /root/autodl-tmp/scripts || exit 1

export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

OUT=/root/autodl-tmp/outputs/picklift_colors_expert
INIT=/root/autodl-tmp/outputs/picklift_cf_retry_expert/checkpoints/200000/pretrained_model

setsid nohup /root/miniconda3/envs/sim2real/bin/lerobot-train \
  --dataset.repo_id=local/so101_picklift_colors \
  --dataset.root=/root/autodl-tmp/datasets/so101_picklift_colors \
  --dataset.eval_split=0.0 \
  --policy.path=${INIT} \
  --rename_map='{"observation.images.overhead": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.n_action_steps=10 \
  --policy.compile_model=false \
  --policy.compile_mode=max-autotune \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.use_peft=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_amp=false \
  --policy.optimizer_lr=0.00003 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=24500 \
  --policy.scheduler_decay_lr=0.0 \
  --policy.push_to_hub=false \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=24500 \
  --scheduler.peak_lr=0.00003 \
  --scheduler.decay_lr=0.0 \
  --batch_size=32 \
  --steps=24500 \
  --eval_steps=0 \
  --save_freq=2450 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=${OUT} \
  < /dev/null > /root/autodl-tmp/traincolors_expert.log 2>&1 &

echo launched
