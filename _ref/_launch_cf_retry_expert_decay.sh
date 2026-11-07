#!/bin/bash
# 从 picklift_cf_retry_expert 的 100000 续训到 200000，这一轮开始让学习率逐步下降
# （2026-09-21 用户指定「继续最新的checkpoint训练100000轮，学习率可以逐渐下降」）。
#
# 相比上一轮 _launch_cf_retry_expert_resume.sh 只改了学习率那一组，其余一字不动：
#   恒定 1e-4                      -> 余弦从 7.5e-5 降到 5e-5
#   --steps=100000                 -> 200000（总步数，跑满 100000 步增量）
#   --scheduler.decay_lr=0.0001    -> 0.00005
#   --scheduler.num_decay_steps=20000 -> 200000
#
# === 为什么不是「8.5e-5 起降」===
#
# 用户选的方案是 num_decay_steps=400000，指望余弦在 t=100000 处还高（0.854，即 8.5e-5）。
# 做不到。lerobot/optim/schedulers.py 的 CosineDecayWithWarmupSchedulerConfig.build 开头：
#
#     if num_training_steps < self.num_decay_steps:
#         scale_factor = num_training_steps / self.num_decay_steps
#         actual_decay_steps = num_training_steps        # <- 被夹到 steps
#
# 我们 --steps=200000，只要 num_decay_steps 写得比 200000 大，一律被夹成 200000，
# num_decay_steps 这个旋钮的实际上限就是 --steps。所以 D=400000 等价于 D=200000。
#
# 退而求其次，用 decay_lr 抬底：alpha = decay_lr / peak_lr = 0.5e-4/1e-4 = 0.5，
# lambda = (1-alpha)*cos + alpha，把曲线整体抬起来。
#   t=100000 -> lambda 0.75 -> lr 7.5e-5
#   t=200000 -> lambda 0.50 -> lr 5.0e-5   （终点正好落在用户要的 5e-5，且不退火到 0）
#
# === 为什么 resume 时改 LR 只认 --scheduler.* ===
#
# lerobot/configs/train.py:272-276：只有 `not resume` 才调 active_cfg.get_scheduler_preset()，
# resume 时 cfg.scheduler 直接来自存档的 train_config.json，再被顶层 CLI 覆盖。
# 所以有效开关是 --scheduler.*；--policy.scheduler_* 这组仍然写同样的值只为保险（本轮不生效）。
#
# 另一个点：resume 时 load_scheduler_state 会把存档里的 base_lrs=[1e-4] 和 last_epoch=100000
# 恢复回来（checkpoints/100000/training_state/scheduler_state.json 实测）。base_lrs 就是余弦的
# 峰值，所以峰值必然还是 1e-4，--scheduler.peak_lr 改不动它；last_epoch=100000 让 lambda 从
# 绝对步 100000 处开始取值，这正是上面算 7.5e-5 的依据。
#
# === 其余照旧 ===
#
# save_freq=20000：删掉旧的 picklift_cf_5k(12G) 后数据盘剩 17G，5 个新档 6.5G 装得下。
# eval_steps=0 + eval_split=0.0：全部数据进训练，且回避 eval 触发即卡死的老问题。
# 80000 步 ÷ 约 4.5 步/秒 ≈ 4.9 小时。
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
  --policy.scheduler_decay_steps=200000 \
  --policy.scheduler_decay_lr=0.00005 \
  --policy.push_to_hub=false \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=200000 \
  --scheduler.peak_lr=0.0001 \
  --scheduler.decay_lr=0.00005 \
  --batch_size=8 \
  --steps=200000 \
  --eval_steps=0 \
  --save_freq=20000 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=${OUT} \
  < /dev/null > /root/autodl-tmp/traincf_retry_expert_decay.log 2>&1 &

echo launched
