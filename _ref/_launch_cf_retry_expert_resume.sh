#!/bin/bash
# 从 picklift_cf_retry_expert 的 020000 真续训到 100000（2026-09-20 用户指定「继续训练练到100000」）。
#
# 用 --resume=true 而不是 --policy.path 起新轮：要的是「继续」——保留 AdamW 的
# 动量状态（training_state/ 394M 就在存档里），步数从 20000 接着记。
#
# --resume 必须带 --config_path 指向存档里的 train_config.json，否则
# lerobot/configs/train.py:191-194 直接抛 ValueError。
# 给了 --config_path 后它会走 train.py:200-204 那条分支：
#   policy_dir     = <config_path 的父目录>            = checkpoints/last/pretrained_model
#   checkpoint_path= policy_dir.parent                 = checkpoints/last  (-> 020000)
# 权重和优化器都从 020000 起。
#
# 为什么这次不担心 memory/lerobot-resume-scheduler-trap 里那三个坑：
#   ① LR 被存档里的 num_decay_steps 钉死 —— 我们是恒定 LR，decay_lr == peak_lr == 1e-4，
#      余弦退化成水平线，num_decay_steps 是多少都不影响。跑到 100000 步还是 1e-4。
#   ② 步数接着记（20000 -> 100000）—— 这正是「续训」想要的行为。
#   ③ output_dir 指向老目录、覆盖上一轮存档 —— 也是想要的，010000/020000 会留着，
#      新的从 040000 开始往后加。
#
# save_freq 从 10000 改成 20000：数据盘只剩 9.9G，一个存档 1.3G。
# 10000 会留 8 个档 = 10.4G 装不下；20000 留 4 个档（040000/060000/080000/100000）= 5.2G，
# 还剩约 4.7G。
#
# 80000 步 ÷ 4.53 步/秒 ≈ 4.9 小时。use_amp=false（这台 3080 Ti 实测过的配置）。
#
# eval_steps=0 + eval_split=0.0：全部数据进训练，且回避 eval 触发即卡死的老问题。
cd /root/autodl-tmp/scripts || exit 1

export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

OUT=/root/autodl-tmp/outputs/picklift_cf_retry_expert

setsid nohup /root/miniconda3/envs/sim2real/bin/lerobot-train \
  --resume=true \
  --config_path=${OUT}/checkpoints/last/pretrained_model/train_config.json \
  --dataset.repo_id=local/so101_picklift_cf_retry \
  --dataset.root=/root/autodl-tmp/datasets/so101_picklift_cf_retry \
  --dataset.eval_split=0.0 \
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
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=0.0001 \
  --policy.push_to_hub=false \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=20000 \
  --scheduler.peak_lr=0.0001 \
  --scheduler.decay_lr=0.0001 \
  --batch_size=8 \
  --steps=100000 \
  --eval_steps=0 \
  --save_freq=20000 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=${OUT} \
  < /dev/null > /root/autodl-tmp/traincf_retry_expert_resume.log 2>&1 &

echo launched
