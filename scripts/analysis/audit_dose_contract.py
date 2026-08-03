"""干预协议 §3 的**剂量口径**——MAGNET 与 PlanTF 两侧共用的唯一定义。

为什么单独成一个模块
--------------------
协议 §9 要求两侧实现「逐行对齐」。`run_interaction_audit.py`（MAGNET 侧）跑在 `nuplan` env，
`run_interaction_audit_plantf.py`（PlanTF 侧，G18）跑在 `plantf` env，而**后者没有
`torch_geometric`** ⇒ PlanTF 侧无法 import MAGNET 侧的脚本。

若两边各写一份剂量解析，`count_sat` / `count_1` / `frac_25` 的语义迟早会分叉，
而这三个剂量正是 primary estimand 的定义本身 —— 分叉就等于两侧在测不同的东西。
⇒ 本模块**只依赖 numpy 与标准库**，两个 env 都能 import。

⚠️ `run_interaction_audit.py` 目前仍持有自己的一份副本（G19 正在用它跑，运行期间不改动）。
`tests/test_audit_dose_contract.py` 逐项断言两份定义等价；G19 跑完后应把 MAGNET 侧
改为 import 本模块并删掉那份副本（记为待办 Q47）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["Dose", "DOSES", "round_half_up", "resolve_dose"]


@dataclass(frozen=True)
class Dose:
    """一个剂量（协议 §3.1）。

    :param mode: `'count'` = 两侧移除**个数**相同；`'fraction'` = 两侧移除**比例**相同。
    :param value: count 模式下 `absolute=False` 时按目标侧比例解析再取两侧较小值；
        `absolute=True` 时是绝对个数。fraction 模式下恒为比例。
    :param absolute: 显式区分「1 个」与「100%」—— 旧实现把 budget=1 解释成比例，
        协议的 `count_1` 剂量正好踩在这个歧义上。
    """

    name: str
    mode: str
    value: float
    absolute: bool = False


#: 冻结的剂量网格（协议 §3.1）。⚠️ fraction 口径禁止 ρ=1.0（两类实体全删 ⇒ 图为空）。
DOSES: Tuple[Dose, ...] = (
    Dose("count_sat", "count", 1.0),             # k = min(n_target, n_control)，功效最大
    Dose("count_1", "count", 1, absolute=True),  # R3-M2 字面版：移除单个周围车辆
    Dose("frac_25", "fraction", 0.25),           # fraction-matched primary
)


def round_half_up(x: float) -> int:
    """半数向上取整。

    Python 内建 `round` 是 banker's rounding（`round(0.5) == 0`），
    剂量解析用它会在 `n·ρ` 恰为 .5 时静默少删一个实体。
    """
    return int(np.floor(x + 0.5))


def resolve_dose(dose: Dose, n_target: int, n_control: int,
                 target_name: str = "target",
                 control_name: str = "control",
                 unit: str = "实体") -> Tuple[int, int, Optional[str]]:
    """把剂量解析成 `(k_target, k_control, skip_reason)`（协议 §3）。

    :param n_target: 目标侧实体池大小（MAGNET=非 ego 的 agent 节点；PlanTF=index 1..A-1 的有效 agent）。
    :param n_control: matched control 侧实体池大小（MAGNET=lane 节点；PlanTF=有效 map polygon）。
    :param unit: `skip_reason` 里的量词。🔴 **不是措辞偏好** —— `skip_reason` 会被写进
        audit shard 的 JSON 产物（`run_interaction_audit.py` 的
        `out[dose.name] = {..., "skipped": skip}`），已落盘的 G19 四 config 用的是
        **「节点」**。MAGNET 侧必须传 `unit="节点"` 才能逐位复现那批产物；
        PlanTF 侧沿用默认「实体」（其产物本来就是用默认值生成的）。
    :returns: 两侧各删几个；`skip_reason` 非 None 时本剂量在该 token 上跳过。
    """
    if n_target == 0 or n_control == 0:
        return 0, 0, f"缺少 {target_name} 或 {control_name} {unit}（n_t={n_target}, n_c={n_control}）"
    if dose.mode == "count":
        k = int(dose.value) if dose.absolute else round_half_up(n_target * dose.value)
        k = min(k, n_target, n_control)
        return (k, k, None) if k > 0 else (0, 0, f"count 剂量解析为 0 个{unit}")
    if dose.mode == "fraction":
        k_t, k_c = round_half_up(n_target * dose.value), round_half_up(n_control * dose.value)
        if k_t <= 0 or k_c <= 0:
            return 0, 0, f"fraction 剂量解析为 0 个{unit}（k_t={k_t}, k_c={k_c}）"
        return k_t, k_c, None
    raise ValueError(f"未知 matched 口径 {dose.mode!r}")
