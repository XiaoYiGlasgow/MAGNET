"""路线 B 第 1 步 —— **ego 与周车的统一位置域 target**（Q108 方案 §10 第 1 项）。

🔎 2026-08-22 由 `scripts/training/joint_targets.py` 迁来并改名（Q110 决策 D-c）：
标签生成器在 `scripts/analysis/`、训练在 `scripts/training/`、评测在 `scripts/eval/`，
三个目录都已会 import `src.shared.utils.*`，而裸名 `joint_targets` 只在 `scripts/training/` 可见。

## 为什么需要这个模块

联合多智能体预测要在同一个损失里同时监督 ego 与周车，但两侧的存储口径不同：

===================  ==========================  =====================================
字段                 存储形态                    出处
===================  ==========================  =====================================
``GROUND_TRUTH``     ego 未来 **逐帧增量** 60×2  `dataset.py:358` → `data.y`
``AGENT_FUTURES``    周车未来 **绝对位置** N×60×3 `magnet_agent_aux_wiring.py` 直接切 `[:, :60, :2]`
===================  ==========================  =====================================

🔑 **两者其实已经在同一个坐标系里**（t0 时刻的 ego-centric 系），只差 ego 是增量存储：
ego 在 t0 位于原点，且 ``GROUND_TRUTH[0] == (0, 0)``（实测），
故 ``cumsum(GROUND_TRUTH)`` **就是** ego 的绝对 ego-centric 未来位置。
⇒ 统一动作 = **只对 ego 做 cumsum**，不需要任何坐标变换。

## 为什么统一到位置域而不是增量域

Q106 实测：现有四项损失全在增量域，恒定输出在增量域只差 **0.40 m**，而位置域 ADE 是 **11.06 m**
（压缩 28 倍），且 smoothness 项**主动奖励**恒定输出。而路线 B 的目标里 ~80% 的周车是静止的
（P0-c 实测），增量域会把「预测它永远不动」奖励到底。⇒ **位置域是唯一能罚到这个失败模式的口径。**

## 逐帧掩码（🔴 不可省）

``AGENT_FUTURES`` 在 track 结束后**逐帧补零**（P0-c 实测：**32.26%** 的 agent 未来被截断，
有效帧前缀连续）。**用最后一帧当终点会把「补零」读成「开到了原点」** —— 助手的 P0 v1 正是栽在这里
（把「到 ego 的距离」误报成「6s 位移」）。本模块的 ``agent_mask`` 逐帧标注有效性，
损失侧必须逐帧 mask。

## 节点顺序

周车侧直接复用 ``magnet_agent_aux_wiring.build_agent_aux_fields``：它的节点顺序与
``dataset.py`` 建 ``gnn_x`` 时**逐位一致**，槽位映射走 ``resolve_agent_slots()`` 的位置 join
（判据 最近 <0.05 m 且 次近 >0.5 m）。🔴 **该判据已在新档 fullret 上复核**：
800 场景 / 13 773 个 agent 节点，最近距离 max **0.000000**、次近中位 **5.874**、
**失败率 0.26%**（闸门 1%）⇒ 判据无需调整。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import sys
from pathlib import Path

import numpy as np
import torch

# `build_agent_aux_fields` 住在 `scripts/training/`（它是训练侧接线，且有自己的回归门），
# 本模块在 `src/shared/utils/` ⇒ 需要把那个目录挂上 sys.path。
# 🔴 与 `magnet_agent_aux_wiring.py:50-52` 同款写法，别改成相对导入 —— 那边是裸名导入 `magnet_agent_aux`。
_GNN_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_GNN_ROOT / "scripts" / "training"), str(_GNN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_agent_aux_wiring import build_agent_aux_fields  # noqa: E402

logger = logging.getLogger(__name__)

#: 未来帧数（10 Hz × 6 s），与 `AGENT_FUTURES` / `GROUND_TRUTH` 同源。
FUTURE_FRAMES: int = 60


class JointTargetError(RuntimeError):
    """target 构造前提不成立 —— 必须暴露，不得静默降级为错误监督。"""


def ego_increments_to_positions(y: np.ndarray) -> np.ndarray:
    """ego 逐帧增量 → 绝对 ego-centric 位置。

    :param y: 形状 (60, 2) 或 (1, 60, 2) 的逐帧增量（`GROUND_TRUTH` / `data.y`）。
    :returns: (60, 2) 的绝对 ego-centric 位置；第 0 帧即 ego 在 t0 的位置 (0, 0)。
    :raises JointTargetError: 形状不符，或首帧增量非零（说明口径与假设不同）。

    🔴 首帧断言不可删：整个「两侧同系」的结论依赖 `cumsum[0] == (0,0)`。
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (FUTURE_FRAMES, 2):
        raise JointTargetError(f"ego 增量形状 {arr.shape} != ({FUTURE_FRAMES}, 2)")
    if float(np.abs(arr[0]).sum()) > 1e-6:
        raise JointTargetError(
            f"ego 首帧增量非零（{arr[0]}）—— 与「cumsum[0] 即 t0 原点」的假设冲突，"
            f"两侧同坐标系的结论不再成立，须先查清口径")
    return np.cumsum(arr, axis=0).astype(np.float32)


def build_joint_targets(row, adjacency_config: str = "default", *,
                        strict: bool = True) -> Optional[Dict[str, torch.Tensor]]:
    """从一行 feature DataFrame 造出 ego + 周车的**统一位置域** target。

    :param row: `pd.read_pickle(pkl).iloc[0]`。
    :param adjacency_config: 必须与建图时一致，否则节点顺序对不上。
    :param strict: True 时任何前提不成立即抛错；False 时返回 None（该样本不参与联合损失）。
    :returns: dict，全部为 **绝对 ego-centric 位置域**，`None` 表示该样本不可用：

        =================  =====================  ==================================
        key                shape                  含义
        =================  =====================  ==================================
        ``ego_tgt``        (60, 2)                ego 未来位置
        ``ego_mask``       (60,) bool             ego 逐帧有效（GT 恒为全 True）
        ``agent_tgt``      (n_agent, 60, 2)       周车未来位置，行序 = 图中 agent 节点序
        ``agent_mask``     (n_agent, 60) bool     🔴 周车逐帧有效（补零帧为 False）
        ``is_agent``       (n_nodes,) bool        图节点是否为 agent，与 `gnn_x` 同序
        =================  =====================  ==================================
    """
    if "GROUND_TRUTH" not in row:
        if strict:
            raise JointTargetError("缺 GROUND_TRUTH")
        return None
    try:
        ego = ego_increments_to_positions(np.asarray(row["GROUND_TRUTH"]))
    except JointTargetError:
        if strict:
            raise
        return None

    out = build_agent_aux_fields(row, adjacency_config, strict=strict)
    if out is None:                                   # 无 agent 节点：仍可只监督 ego
        n_nodes = 0
        is_agent = torch.zeros(0, dtype=torch.bool)
        a_tgt = torch.zeros(0, FUTURE_FRAMES, 2)
        a_msk = torch.zeros(0, FUTURE_FRAMES, dtype=torch.bool)
    else:
        is_agent, a_tgt, a_msk = out
        n_nodes = int(is_agent.numel())
    if n_nodes and int(is_agent.sum()) != int(a_tgt.shape[0]):
        msg = (f"agent 节点数 {int(is_agent.sum())} != target 行数 {int(a_tgt.shape[0])} "
               f"—— 行序与节点序已错位，不得继续")
        if strict:
            raise JointTargetError(msg)
        logger.warning(msg)
        return None

    return {
        "ego_tgt": torch.from_numpy(ego),
        "ego_mask": torch.ones(FUTURE_FRAMES, dtype=torch.bool),
        "agent_tgt": a_tgt,
        "agent_mask": a_msk,
        "is_agent": is_agent,
    }


def masked_position_l1(pred: torch.Tensor, tgt: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """逐帧掩码的位置域 smooth-L1。

    :param pred: (..., T, 2) 预测位置。
    :param tgt: (..., T, 2) 目标位置。
    :param mask: (..., T) bool，False 的帧不参与。
    :returns: 标量；无有效帧时返回 0（**不是 nan**，否则整个 batch 的梯度会被污染）。
    """
    if mask.numel() == 0 or not bool(mask.any()):
        return pred.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(pred[mask], tgt[mask])


__all__ = ["FUTURE_FRAMES", "JointTargetError", "build_joint_targets",
           "ego_increments_to_positions", "masked_position_l1"]
