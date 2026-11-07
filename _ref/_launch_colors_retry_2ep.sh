#!/bin/bash
# 第 8 轮试跑：在 colors_retry 604 条上解冻微调 2 epoch，先看 blue 滑落后重抓有没有改善。
#
# 用户 2026-09-23 选「先跑短的看效果」：2 epoch 约 2.2 小时，有改善再续跑满 5 epoch。
#
# === 数据 ===
# so101_picklift_colors_retry，604 条 / 107266 帧（服务器 8 路采集，见 §1.10）。
# 其中 230 条带「故意抓空 -> 张开 -> 重抓」，这是这份数据唯一的新信息：
# 旧的 colors 618 条里 0 条抓空恢复，模型因此只学到「到点就合爪」。
#
# 这次不掺旧 colors，重抓占比 38%。掺进去会降到 19%（更保守），但既然是先看效果，
# 少一个变量——先确认这份数据到底有没有用。
#
# === 起点 ===
# picklift_colors_unfrozen/014670：视觉塔已经解冻训过 3 epoch（实测相对官方基座
# vision_model max|Δ| = 0.00844，而 cf_retry_expert/200000 是 0.00000000）。
# 它自己的 blue rollout 仍然失败 900 帧，所以这份数据还得上。
#
# === 两个开关必须一起给 ===
# smolvlm_with_expert.py:152 的 set_requires_grad 里，train_expert_only 那个分支会把
# 整个 self.vlm（含 vision_model）冻上，而且排在 freeze_vision_encoder 判断**之后**，
# 后写的覆盖先写的。只设 freeze_vision_encoder=false 没有任何用。
# 启动后看日志里的 num_learnable_params：
#   99,880,992  = 还是只训专家（参数没传进去）
#   392,904,096 = 解冻成功
#
# === 步数：看 epoch 不看步数 ===
# 107266 帧 / batch 16 = 6704 步/epoch，2 epoch = 13408 步。
# steps 和两组 scheduler_decay_steps 必须同值，否则余弦退火对不上
# （lerobot/optim/schedulers.py 的 build 开头会把 num_decay_steps 夹到 num_training_steps）。
#
# save_freq=6704 = 每 epoch 一档，2 个约 4.6G（全量微调存档实测约 2.3G），
# 数据盘还剩 21G，装得下。1 epoch 一档是为了万一语言被练坏了能往前退。
#
# === LR ===
# 3e-5 余弦到 0，warmup 500，和第 7 轮一模一样，不动。真跑飞了（loss 炸或者语言更差）先动它。
#
# eval_steps=0 + eval_split=0.0：全部数据进训练，且回避 eval 触发即卡死的老问题。
cd /root/autodl-tmp/scripts || exit 1

export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

OUT=/root/autodl-tmp/outputs/picklift_colors_retry_unfrozen
INIT=/root/autodl-tmp/outputs/picklift_colors_unfrozen/checkpoints/014670/pretrained_model

setsid nohup /root/miniconda3/envs/sim2real/bin/lerobot-train \
  --dataset.repo_id=local/so101_picklift_colors_retry \
  --dataset.root=/root/autodl-tmp/datasets/so101_picklift_colors_retry \
  --dataset.eval_split=0.0 \
  --policy.path=${INIT} \
  --rename_map='{"observation.images.overhead": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.n_action_steps=10 \
  --policy.compile_model=false \
  --policy.compile_mode=max-autotune \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.use_peft=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.use_amp=false \
  --policy.optimizer_lr=0.00003 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=13408 \
  --policy.scheduler_decay_lr=0.0 \
  --policy.push_to_hub=false \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=13408 \
  --scheduler.peak_lr=0.00003 \
  --scheduler.decay_lr=0.0 \
  --batch_size=16 \
  --steps=13408 \
  --eval_steps=0 \
  --save_freq=6704 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=${OUT} \
  < /dev/null > /root/autodl-tmp/traincolors_retry.log 2>&1 &

echo launched
