# best（013408）从零复现配方

`G:\sim2real\so101-smolvla-sim\checkpoints\best\pretrained_model` 就是
`pickliftcolors_retry_unfrozen_013408` 的整份拷贝（md5 `5e8d0f02c6d94a8391ad69177ba8e7b2`）。
这份文档写清它是怎么一步步练出来的、每步要用哪个脚本、哪些东西现在还活着。

判据（为什么它是全项目最好）：训练场景里**唯一一次交付了非默认颜色**——同 seed 10000、
同布局、同权重，只换那句话：绿（默认）成功 135 帧、红（指令说红）成功 467 帧、蓝失败 900 帧。
其余各线（colorchain / align / cfalign / fenkai 1-3）在训练场景里全是失败。

---

## 0. 一句话链路

```
lerobot/smolvla_base（官方基座，HF 下载）
  └─[S1] cf_retry 660 条，只训专家，20000 步，LR 1e-4 恒定   -> picklift_cf_retry_expert/020000
       └─[S2] --resume 续到 100000，其余不动                  -> .../100000
            └─[S3] --resume 续到 200000，LR 改余弦 1e-4→5e-5  -> .../200000   ★ 公共起点
                 └─[S4] colors 618 条，**只训专家**，24500 步（10 ep），LR 3e-5 余弦→0
                      -> picklift_colors_expert/024500
                      └─[S5] colors 618 条，**全量解冻**，24450 步（5 ep），LR 3e-5 余弦→0
                           -> picklift_colors_unfrozen/014670（取 3 ep 那档）
                           └─[S6] colors_retry 604 条，**全量解冻**，13408 步（2 ep），LR 3e-5 余弦→0
                                -> pickliftcolors_retry_unfrozen/013408  = best
```

注意 **S1 是从官方基座重起的，不是从 `picklift345_full/100000` 续下来的**。
`picklift345_*` 那条线（rgb345 数据、全模型微调）不在 best 的链上。

---

## 1. 环境和场地（所有步共用）

- conda env `sim2real`，解释器 `/root/miniconda3/envs/sim2real/bin/python`，训练入口 `lerobot-train`（lerobot 0.6.1）
- `export HF_HOME=/root/autodl-tmp/hf` + `HF_HUB_OFFLINE=1`；AutoDL 直连 HF 不通，权重走 hf-mirror
- 场景：`MuJoCoPickLift-v1` + `PickConfig`，pool = red/green/blue 三个 2.5cm 立方体，
  两路相机 overhead + wrist，640×480，30 fps
- `observation.state` / `action` 都 6 维：前 5 关节**度**，夹爪 **0~100 百分比**
- 路径：数据集 `/root/autodl-tmp/datasets/`，输出 `/root/autodl-tmp/outputs/`，启动脚本 `/root/autodl-tmp/scripts/`

---

## 2. 数据怎么来的

| 数据集 | 条数/帧数 | 采集方式 |
|---|---|---|
| `so101_picklift_cf` | 615 / 78158 | `_gen_counterfactual.py 208`：208 seed × 3 色 = 624 次尝试。包一层 `env.reset` 注入 `options={"target_index": k}`，同 seed 三条布局逐字节相同，只有指令里的颜色词不同 |
| `so101_picklift_retry45` | 45 / 11682 | `collect_lerobot.py` 加 `RETRY=1 RETRY_PROB=0.15`。整套数据里**唯一**带「抓空→张开→重抓」的样本 |
| `so101_picklift_cf_retry` | 660 / 89840 | cf 615 + retry45 45 |
| `so101_picklift_colors` | 618 / 78236 | `collect_colors.py`，8 路并行，seed 104..207。104 layout × **2 套配色**（rgb / gbr，整体左移）× 3 个钉住目标 = 624 设计，实收 618 |
| `so101_picklift_colors_retry` | 604 / 107266 | 同上骨架，**只多 `RETRY=1 RETRY_PROB=0.5`**。触发重抓 252 次，真正带重抓分支的 230 条（38%） |

采集命令（服务器，`/root/autodl-tmp/port`，脚本 `_ref/_launch_colors.sh` / `_ref/_launch_colors_retry.sh`）：

```bash
# colors：8 个 worker，每个 13 seed，从 seed 104 起
bash _launch_colors.sh 8 13 104 colors
# colors_retry：同一份脚本，只多 RETRY=1 RETRY_PROB=0.5
bash _launch_colors_retry.sh 8 13 104 colors_retry
# 采完合并（_ref/merge_colors.py）：
cd /root/autodl-tmp/port && DATASETS_ROOT=/root/autodl-tmp/datasets python merge_colors.py 8 colors
```

配色轮换为什么关键：同一个 seed 下两套配色共用同一份物体摆位，只有「哪个颜色占哪个槽位」变了。
于是固定 `(seed, k)` 下两套配色的**目标位置逐位相同、只有指令里的颜色词不同**，
「去第几个槽位」这条捷径不成立。`RETRY_PROB` 不设 1.0 是因为行为克隆会把「每局先抓空一次」
学成固定动作；不设 0.15 是因为 retry45 只有 45 条，会被 615 条 cf 稀释掉。

---

## 3. 六步训练，逐步的完整命令

### S1 只训专家，20000 步（脚本 `_ref/_launch_cf_retry_expert.sh`）

```bash
setsid nohup /root/miniconda3/envs/sim2real/bin/lerobot-train \
  --dataset.repo_id=local/so101_picklift_cf_retry \
  --dataset.root=/root/autodl-tmp/datasets/so101_picklift_cf_retry \
  --dataset.eval_split=0.0 \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.overhead": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.train_expert_only=true  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=500 --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=0.0001 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=20000 --scheduler.peak_lr=0.0001 --scheduler.decay_lr=0.0001 \
  --batch_size=8 --steps=20000 --eval_steps=0 --save_freq=10000 \
  --num_workers=16 --seed=1000 --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/picklift_cf_retry_expert \
  < /dev/null > /root/autodl-tmp/traincfretry.log 2>&1 &
```

**`decay_lr == peak_lr` 就是「恒定 LR」的写法**（余弦退化成常数，见 memory `constant-lr-by-equal-decay-peak`）。

### S2 续到 100000（`_ref/_launch_cf_retry_expert_resume.sh`）

差别：把 `--policy.path` 换成 `--resume=true --config_path=<OUT>/checkpoints/last/pretrained_model/train_config.json`，
`--steps=100000`、`--save_freq=20000`。`--resume` 保留 AdamW 的动量，`--policy.path` 不会。

**resume 与不 resume 认的是两组不同的 scheduler 参数**：resume 走 `--scheduler.*`，
不 resume 走 `--policy.scheduler_*`。两组都写同一值最保险（memory `lerobot-resume-scheduler-trap`）。

### S3 续到 200000，LR 开始降（`_ref/_launch_cf_retry_expert_decay.sh`）

只改三处：`--steps=200000`、`--policy.scheduler_decay_lr=0.00005`、`--scheduler.decay_lr=0.00005`。
`num_decay_steps` 写得比 `--steps` 大也会被夹成 `--steps`。
产物 `picklift_cf_retry_expert/200000` **是后面所有实验的公共起点**。

### S4 colors 只训专家 10 epoch（`_ref/_launch_colors_expert_10ep.sh`）

```bash
INIT=/root/autodl-tmp/outputs/picklift_cf_retry_expert/checkpoints/200000/pretrained_model
OUT=/root/autodl-tmp/outputs/picklift_colors_expert
# 其余同上，关键差异：
#   --dataset.repo_id=local/so101_picklift_colors   --policy.path=$INIT
#   --policy.train_expert_only=true --policy.freeze_vision_encoder=true   （VLM 冻着）
#   --policy.optimizer_lr=0.00003  decay_lr=0.0      （3e-5 余弦到 0）
#   --batch_size=32  --steps=24500  --save_freq=2450
```

78236 帧 / 32 = 2445 步/epoch × 10 = 24500 步。实测 1.91 步/秒、3 小时 33 分、
末 loss 0.006、`num_learnable_params=99880992`（只训专家的值）。

### S5 colors 全量解冻 5 epoch（`_ref/_launch_colors_unfrozen_5ep.sh`）

```bash
INIT=/root/autodl-tmp/outputs/picklift_colors_expert/checkpoints/024500/pretrained_model
#   --policy.train_expert_only=false --policy.freeze_vision_encoder=false   ← 两个必须一起给
#   --policy.optimizer_lr=0.00003  decay_lr=0.0
#   --batch_size=16  --steps=24450  --save_freq=4890
```

**解冻必须两个开关一起给**：`smolvlm_with_expert.py` 的 `set_requires_grad` 里，
`train_expert_only` 那个分支排在 `freeze_vision_encoder` 判断之后、会把整个 `self.vlm` 冻上，
后写的覆盖先写的。只设 `freeze_vision_encoder=false` 没有任何用。
启动后核日志 `num_learnable_params`：`392904096` = 解冻成功，`99880992` = 还是只训专家。

### S6 colors_retry 全量解冻 2 epoch（`_ref/_launch_colors_retry_2ep.sh`）→ **best**

```bash
INIT=/root/autodl-tmp/outputs/picklift_colors_unfrozen/checkpoints/014670/pretrained_model
OUT=/root/autodl-tmp/outputs/picklift_colors_retry_unfrozen
#   --dataset.repo_id=local/so101_picklift_colors_retry   （只用 604 条，**不掺旧 colors 618**）
#   --policy.train_expert_only=false --policy.freeze_vision_encoder=false
#   --policy.optimizer_lr=0.00003  decay_lr=0.0  warmup 500
#   --batch_size=16  --steps=13408  --save_freq=6704   （107266/16 = 6704 步/epoch × 2）
```

13408 步那一档就是 **013408 = best**。`save_freq=6704` 是每 epoch 一档，
所以同一次运行还留下了 `006704`（1 epoch），要往前退用得上。

---

## 4. 现在还剩什么（2026-09-29 实测）

**还在（本地 `G:\sim2real\so101-smolvla-sim\checkpoints\`）**——链上每一档都在，
所以从任何一步接着往上练都可以：

- `pickliftcfretry_expert_020000` / `_040000` / `_100000` / `_200000`
- `pickliftcolors_expert_012250` / `_024500`
- `pickliftcolors_unfrozen_004890` / `_009780` / `_014670`
- `pickliftcolors_retry_unfrozen_006704` / **`_013408`（= best）** / `_cont3_020112`

**还在（本地数据集）**：`G:\sim2real\so101-smolvla-sim\_ref\datasets\` 下有
`so101_picklift_cf`、`so101_picklift_cf_retry`、`so101_picklift_retry45`、`so101_picklift_rgb`、
`so101_picklift_rgb345`、`so101_picklift_dagger`、`so101_picklift_dagger2` 等。

**已经没了**：训练 best 直接用的两份数据 `so101_picklift_colors`（618）
和 `so101_picklift_colors_retry`（604）**服务器和本地都没有**——
服务器 `/root/autodl-tmp/datasets/` 只剩 `so101_picklift_cfalign` /
`so101_picklift_cfalign_retry` / `so101_picklift_fenkai3` 和 fenkai3 分片；
本地 `_ref/datasets/` 下没有 `*colors*`。中间 checkpoint 也是服务器侧全清（`outputs/` 只剩
`picklift_cfalign_fresh` / `picklift_cfalign_retry_expert` / `picklift_fenkai{,2,3}_*`）。

**结论**：想真的「从零」复现，S1~S4 的数据 cf_retry 还在、S5/S6 的两份 colors 必须重采
（采集脚本都在，服务器 8 路并行约 20 分钟一份）。**只想再往上练或换数据实验，
直接从本地 `024500` / `014670` / `013408` 起就行，不用重采。**

---

## 5. 复现时必踩的四个坑

1. **不 resume 时认 `--policy.scheduler_*`，resume 时认 `--scheduler.*`** —— 两组都写同一值。
2. **`num_decay_steps` 上限就是 `--steps`**，写大了会被静默夹断。
3. **恒定 LR 的写法是 `decay_lr == peak_lr`**，不是把 decay_lr 设 0。
4. **`eval_steps=0` + `dataset.eval_split=0.0`** —— eval 一触发就卡死（老问题），别开。
