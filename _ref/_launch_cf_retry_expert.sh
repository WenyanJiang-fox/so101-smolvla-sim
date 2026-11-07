#!/bin/bash
# 只训动作专家，起点用官方基座 lerobot/smolvla_base（2026-09-20 用户指定）。
#
# 跟上一轮 _launch_cf_retry.sh 的差别只有两处：
#   --policy.path                             : 我们的 checkpoint -> lerobot/smolvla_base
#   train_expert_only / freeze_vision_encoder : false/false -> true/true
# 数据、LR、batch、步数、seed 全部不动，变量只有「起点 + 训练范围」。
#
# 为什么必须换起点：--policy.path 指向我们自己的 checkpoint 时，
# PreTrainedPolicy.from_pretrained 会先用官方的 HuggingFaceTB/SmolVLM2-500M
# 构造 VLM，紧接着 _load_as_safetensor 用那个 checkpoint 的整个 state_dict
# 把它覆盖掉（lerobot/policies/pretrained.py）。
# 所以只要起点是自己的 checkpoint，视觉塔就必然是我们练过的。
# 代价：动作专家也一起从官方起步，丢掉之前 120k 步。
#
# 为什么不用 AMP：这台机器是 3080 Ti（Ampere, sm_86），有 bf16 张量核。
# 2080 Ti 那台的 fp16 事故（GradScaler 拒收 fp16/bf16 梯度）在这台上不存在，
# 但也没必要——expert-only 本来就快，保持本机实测过的 use_amp=false。
# smolvlm_with_expert.py:94 在这台机器上是原版 bfloat16，不需要任何补丁。
#
# 为什么省：VLM 主干 requires_grad=false + eval()（smolvlm_with_expert.py:157），
# 只前向不反传、不建优化器状态。参考实现在这台机器上同配置实测
# 4.1 步/秒、显存 2.6GB（batch 8），20000 步约 1.4 小时。
#
# 数据：so101_picklift_cf_retry = cf 615 条 + retry45 45 条 = 660 条 / 89840 帧。
# 混 retry 的原因见 _launch_cf_retry.sh 文件头：cf 那 615 条 0 条抓空恢复样本。
#
# save_freq=10000：数据盘只剩 13G，一个存档约 1.8G（0.91G 权重 + 0.8G 优化器），
# 5000 会留 4 个档约 7.2G 太贴边；10000 留 2 个档（010000/020000）+ last。
#
# eval_steps=0 + eval_split=0.0：全部数据进训练，且回避 eval 触发即卡死的老问题。
cd /root/autodl-tmp/scripts || exit 1

export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

setsid nohup /root/miniconda3/envs/sim2real/bin/lerobot-train \
  --dataset.repo_id=local/so101_picklift_cf_retry \
  --dataset.root=/root/autodl-tmp/datasets/so101_picklift_cf_retry \
  --dataset.eval_split=0.0 \
  --policy.path=lerobot/smolvla_base \
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
  --steps=20000 \
  --eval_steps=0 \
  --save_freq=10000 \
  --log_freq=50 \
  --num_workers=16 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/picklift_cf_retry_expert \
  < /dev/null > /root/autodl-tmp/traincf_retry_expert.log 2>&1 &

echo launched
