"""指令字符串 -> 目标颜色 one-hot（红/绿/蓝，固定这个顺序）。

只在这一处定义顺序，数据集转换（add_target_onehot.py）和推理侧
（eval_picklift6.py / lang_ablation.py）都从这里取，避免两边对不上。

为什么存在：同一 layout、同一 observation.state 的第 0 帧，模型对
「red / green / blue」三条指令的动作只差 0.2 度——第一步根本没按目标走。
这个 one-hot 是目标信息的干净版：直接从指令字符串解析，不经过 VLM 也不经过图像。

顺序必须和数据集里 action/state 的既有维度无关，只是追加在末尾的第 7/8/9 维。
"""

from __future__ import annotations

import numpy as np

COLORS = ("red", "green", "blue")
NAMES = tuple(f"target_{c}" for c in COLORS)


def onehot_from_task(task: str) -> np.ndarray:
    """从指令里认出颜色，返回 float32 的 (3,) one-hot。

    认不出就直接抛，不做兜底：数据集里出现第四种颜色必须立刻发现，
    静默给个全零会让实验结论不可信。
    """
    low = str(task).lower()
    hits = [i for i, c in enumerate(COLORS) if c in low]
    if len(hits) != 1:
        raise ValueError(f"指令 {task!r} 里认出的颜色不是恰好一个：{hits}")
    out = np.zeros(len(COLORS), dtype=np.float32)
    out[hits[0]] = 1.0
    return out
