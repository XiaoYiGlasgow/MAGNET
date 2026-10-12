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
#: 🔴 **这三个不得改动** —— 已落盘的 G19 / R1 shard 全部依赖它们逐位复现
#: （回归门见 `tests/test_audit_dose_contract.py`）。
DOSES: Tuple[Dose, ...] = (
    Dose("count_sat", "count", 1.0),             # k = min(n_target, n_control)，功效最大
    Dose("count_1", "count", 1, absolute=True),  # R3-M2 字面版：移除单个周围车辆
    Dose("frac_25", "fraction", 0.25),           # fraction-matched primary
)

#: 🆕 **剂量-反应扫描网格**（Q83，08-05 新增，**不进 primary family**）。
#:
#: 动机（08-05 实测）：`S = response(agent) − response(lane)` 的**符号完全由两侧删除
#: 个数之比决定**，与模型无关 —— 18 格（2 架构 × 3 seed × 3 剂量）无一例外：
#: 比值 1.00×（`count_1` / `count_sat`）⇒ S 恒**正**；比值 7.07×（`frac_25`）⇒ S 恒**负**。
#: ⇒ 「哪个模态更重要」这个问法本身是**剂量依赖的、良定义不了的**。
#:
#: 本网格把剂量比 `R = k_control / k_target` 变成**唯一变量**（固定 `k_target = 1`），
#: 用来估计**交换率 R\***：**删几个 lane 才等价于删 1 个 agent**。
#: R\* 是**剂量不变**的量，可跨模型/跨架构比较 —— 它替代了那个良定义不了的问法。
#:
#: 🔴 `ratio_1` 与 `count_1` 在语义上等价（1 vs 1），**故意保留**作为自校验点：
#: 两者的读数必须逐位相同，否则说明新模式的实现与冻结契约分叉了。
RATIO_DOSES: Tuple[Dose, ...] = tuple(
    Dose(f"ratio_{r}", "ratio", r) for r in (1, 2, 3, 5, 7, 10, 14, 20)
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
    if dose.mode == "ratio":
        # 剂量比 R 是唯一变量：固定 k_target = 1，k_control = R。
        # 🔴 **绝不 clamp**：若 n_control < R，实际比值就不是 R 了，
        # 混进曲线会把「S 在 R 处的取值」算错，而且**不会有任何报错**
        # —— 这正是剂量混杂本身的失效形态，不能在修它的实验里重演一次。
        r = int(dose.value)
        if n_target < 1:
            return 0, 0, f"ratio 剂量需要至少 1 个 target {unit}（n_t={n_target}）"
        if n_control < r:
            return 0, 0, (f"ratio 剂量比 R={r} 不可达：control 侧只有 {n_control} 个{unit}"
                          f"（clamp 会静默改变实际比值，故跳过）")
        return 1, r, None
    raise ValueError(f"未知 matched 口径 {dose.mode!r}")
