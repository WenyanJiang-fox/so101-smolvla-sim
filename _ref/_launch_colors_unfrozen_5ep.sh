#!/bin/bash
# 解冻 VLM，在颜色轮换数据上全量微调 5 epoch。2026-09-22 用户指定。
#
# === 为什么这轮要解冻 ===
#
# 现在 picklift_colors_expert/024500 的视觉塔逐位等于官方基座（实测 vision_model
# 197 张量 + connector + text_model 146 张量全部 max|Δ|=0.00000000，只有 lm_expert
# 差 0.345）。模型读颜色词靠的就是冻结的那块，24500 步一个字没动它，所以
# 「蓝抓成绿」治不了——把 nas 从 10 改到 16 结果一模一样，因为那只是推理参数。
# 离线量过：颜色变化对特征的扰动 0.0205，场景变化 0.1379，差 6.7 倍。信号太弱，
# 模型就滑回空间先验（就近抓）。解冻是唯一能把这个信号加强的办法。
#
# === 起点为什么是 24500 而不是 cf_retry_expert/200000 ===
#
# 24500 的视觉塔**仍然是官方基座那份**（这轮 expert-only + freeze_vision_encoder
# 练出来的，VLM 没动过），所以从 24500 起步和从 200000 起步，视觉塔是同一份，
# 不存在污染。差别只在动作专家：从 24500 起步，专家已经在这份 colors 数据上练过
# 10 epoch，解冻后只要让视觉塔追上，不用把动作从头再学一遍。
#
# === 保留 24500 ===
#
# 输出写到新目录 picklift_colors_unfrozen，picklift_colors_expert/ 下 10 个存档
# 一个字不动。本地也有一份 checkpoints/pickliftcolors_expert_024500。
#
# === 两个开关必须一起给 ===
#
# smolvlm_with_expert.py:152 的 set_requires_grad 里，train_expert_only 那个分支
# 会把整个 self.vlm（含 vision_model）全冻上，而且排在 freeze_vision_encoder
# 判断**之后**，后写的覆盖先写的。只把 freeze_vision_encoder 设 false 没有任何用。
# 启动后看日志 num_learnable_params：
#   99,880,992  = 还是只训专家（参数没传进去）
#   392,904,096 = 解冻成功
#
# === 步数：看 epoch 不看步数 ===
#
# 78236 帧 / batch 16 = 4890 步/epoch，5 epoch = 24450 步。
# steps 和两组 scheduler_decay_steps 必须同值，否则余弦退火对不上
# （lerobot/optim/schedulers.py:149 的自动缩放只在 num_training_steps < num_decay_steps 时触发）。
#
# batch 32 -> 16：解冻后优化器状态和梯度多出约 4.7 GB，24G 卡先保守。
# 启动后查 nvidia-smi，显存剩得多再考虑加。
#
# save_freq=4890：每 1 epoch 一个档，5 个约 12G（全量微调存档实测约 2.4G），
# 数据盘剩 30G 装得下。1 epoch 一档是为了万一语言被练坏了能往前退。
#
# LR 仍 3e-5 余弦退到 0，warmup 500，不动。batch 和轮数已经定了，LR 是第三个变量。
# 真跑飞了（loss 炸或者语言更差）先动它。
#
# eval_steps=0 + eval_split=0.0：全部数据进训练，且回避 eval 触发即卡死的老问题。
cd /root/autodl-tmp/scripts || exit 1

export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

OUT=/root/autodl-tmp/outputs/picklift_colors_unfrozen
INIT=/root/autodl-tmp/outputs/picklift_colors_expert/checkpoints/024500/pretrained_model

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
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.use_amp=false \
  --policy.optimizer_lr=0.00003 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=24450 \
  --policy.scheduler_decay_lr=0.0 \
  --policy.push_to_hub=false \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=24450 \
  --scheduler.peak_lr=0.00003 \
  --scheduler.decay_lr=0.0 \
  --batch_size=16 \
  --steps=24450 \
  --eval_steps=0 \
  --save_freq=4890 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=${OUT} \
  < /dev/null > /root/autodl-tmp/traincolors_unfrozen.log 2>&1 &

echo launched
