"""C4 规划形式化层（R4-1 / R2-1 / R2-4 / AE-2）。

设计文档：`docs/work_records/paper/MAGNET/VT-2026-04191_修改工作记录/C4_规划形式化_设计_20260801.md`

🔴 **定位（§7 写作硬约束）**：候选生成 + 打分 + argmax 的范式由 **PDM `2306.07962`** 占据。
本层是「按 R4-1 意见实施的规划形式化」，**不是本文提出的方法**。

🔴 **import 纪律**：一律走包路径 `from src.planning_layer import ...`，
禁止裸 import 同名模块 —— 同一文件被两条 import 路径加载会变成两个模块对象，
`isinstance` 会静默失败（坑#18，08-01 已踩过一次）。
"""

from .candidates import Candidate, EgoSnapshot, generate, reference_speed
from .config import (
    PlanningLayerConfig,
    available_presets,
    preset,
    preset_from_env,
)
from .layer import constant_velocity_reference, refine, select
from .scoring import SceneContext, progress_scale, score, score_breakdown

__all__ = [
    "Candidate",
    "EgoSnapshot",
    "PlanningLayerConfig",
    "SceneContext",
    "available_presets",
    "constant_velocity_reference",
    "generate",
    "preset",
    "preset_from_env",
    "reference_speed",
    "refine",
    "progress_scale",
    "score",
    "score_breakdown",
    "select",
]
