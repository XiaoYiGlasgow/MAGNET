"""闸⑥ 的 dataset 侧接线 —— **独立 wrapper，刻意不改 `src/shared/utils/dataset.py`**。

**为什么是 wrapper 而不是直接改 dataset.py**（阶段76 决定）

`dataset.py` 正被三个在跑的 MAGNET L2 训练消费。Python 不像 bash 按字节偏移读脚本，
改它**不会**让运行中的进程当场炸；但**只要有一个 run 崩溃重启，就会捡到新代码**，
于是同一条 scaling 曲线里混进两个代码版本。⇒ L2 全部跑完（≈08-04 15:00）之前只用 wrapper。

**本模块给 GraphData 补三个字段**

===================  =========================  ===================================
字段                  形状                        含义
===================  =========================  ===================================
``gnn_is_agent``      ``[N_nodes]`` bool         该 GNN 节点是不是 agent（ego/lane 为 False）
``agent_future_tgt``  ``[n_agent, 60, 2]`` float 对应 agent 的未来 (x, y)，**绝对 ego-centric 坐标**
``agent_future_mask`` ``[n_agent, 60]`` bool     该帧是否有效（**排除零填充**）
===================  =========================  ===================================

**三个必须处理的坑（全部实测，见阶段75/76 日志）**

1. 🔴 **节点 → 槽位不是恒等映射**：图里的 agent 是删掉 72.9% 静止车之后剩下的，
   而 ``AGENT_FUTURES`` 恒 20 槽。实测 **26/32 场景（81%）映射 ≠ 恒等**
   ⇒ 按下标接线会让 **100% 的监督信号错位**。用末帧位置精确 join 恢复
   （40 pkl / 116 个 agent 节点 **100% 唯一匹配**，最近 0.0000 m、次近 2.03 m）。

2. 🔴 **``AGENT_FUTURES`` 是绝对位置，不是增量** —— 与 ego 输出头（增量，训练时 ``cumsum``）
   **口径不同**。实测判别：``|未来[0] − 历史末帧|`` 中位 **0.137 m** vs ``|未来[0]|`` 中位 **46.10 m**。
   ⇒ aux 头直接回归绝对坐标；**论文写作时必须写明两个头口径不同**。

3. 🔴 **尾部零填充必须 mask**（track 中断）：实测 **9.6% 的帧是 (0,0) 填充**，
   **21% 的 agent 槽受影响**（p90 达 45%、最坏 98.3%）。
   不 mask 就是在教 aux 头预测 (0,0) —— 会把闸⑥ 推向"不升"档，
   再被读成「瓶颈不在监督」这一**错误结论**。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# `masked_aux_loss` 在 `magnet_agent_aux` 里定义（唯一实现），此处只转出去供 trainer 用。
# ⚠️ 早期本文件另写过一份同名函数 ⇒ 同一件事两个实现、其中一份没 mask。已合并。
from magnet_agent_aux import (  # noqa: E402
    AgentSlotJoinError,
    masked_aux_loss,
    resolve_agent_slots,
)
from src.shared.utils.dataset import (  # noqa: E402
    build_graph_from_subgraphs,
    get_adjacency_config,
    subgraph_pooling,
)

logger = logging.getLogger(__name__)

FUTURE_FRAMES = 60

#: 🔴 必须与 `dataset.py:400` 建图时用的值一致，否则重建出的节点集与 `gnn_x` 不同 ⇒ 目标错位。
#: 这里硬编码是无奈之举（`dataset.py` 没把它导出成常量）；`attach_agent_aux` 里的节点数断言
#: 是这一处不一致的**兜底探测器** —— 值改了而这里没跟上时，那条断言会先炸。
GRAPH_DISTANCE_THRESHOLD = 30.0


class AgentAuxWiringError(RuntimeError):
    """aux 接线前提不成立 —— 暴露而不是静默产出错目标。"""


def _agent_end_xy(row, sids) -> np.ndarray:
    """图中每个 agent 子图**末节点的终点** (x, y)，与建图所用坐标同源。

    ⚠️ `SUBGRAPH_ID_TO_NODE_INDICES[sid]` 是 **`(start, end)` 二元组**（见 `dataset.py:39`），
    不是节点下标列表 ⇒ `[-1]` 取到的是 `end`，即该子图最后一个 polyline 行。
    `POLYLINE_FEATURES` 的第 2:4 列是该向量的**终点** (x, y)，与 `AGENT_HISTORIES[:, -1, :2]`
    同一坐标系（实测精确吻合到 0.0000 m）。
    """
    idxs = row["SUBGRAPH_ID_TO_NODE_INDICES"]
    pf = np.asarray(row["POLYLINE_FEATURES"])
    return np.array([pf[idxs[s][-1]][2:4] for s in sids], dtype=np.float64)


def build_agent_aux_fields(row, adjacency_config: str = "default",
                           *, strict: bool = True, return_slots: bool = False):
    """从一行 feature DataFrame 算出闸⑥ 需要的三个字段。

    :param row: `pd.read_pickle(pkl).iloc[0]`。
    :param adjacency_config: **必须与建图时用的完全一致**，否则节点顺序对不上。
    :param strict: True 时任何一处对不上就抛错；False 时返回 None（该样本不参与 aux loss）。
    :param return_slots: True 时额外返回 `slots`（`[n_agent]`，-1 = 位置 join 失败）。
        🔴 **诊断必需**：只看 mask 分不出「join 失败」与「该 agent 未来本来就是零填充」——
        后者是已知数据性质（实测 21% 的槽受影响、最坏 98.3%），前者才是接线错误。
        把两者混在一个计数里，会让质量闸门在正常数据上误触发（08-04 实测踩过）。
    :returns: `(gnn_is_agent, agent_future_tgt, agent_future_mask[, slots])`，无 agent 节点时 None。
    """
    if "SUBGRAPH_ID_TO_NODE_INDICES" not in row or "AGENT_FUTURES" not in row:
        if strict:
            raise AgentAuxWiringError("缺 SUBGRAPH_ID_TO_NODE_INDICES 或 AGENT_FUTURES")
        return None

    types = row["SUBGRAPH_ID_TO_TYPE"]
    # 🔴 必须与 `dataset.py:394` 逐位同源：那里传的是 `POLYLINE_FEATURES`（astype float32），
    #    **不是** `SUBGRAPH_FEATURES`。传错会让 `subgraph_pooling` 拿到不可切片的对象，
    #    或更糟 —— 静默池化出另一套节点特征，导致节点顺序与 `gnn_x` 错位。
    features = np.asarray(row["POLYLINE_FEATURES"]).astype(np.float32)
    pooled = subgraph_pooling(features, row["SUBGRAPH_ID_TO_NODE_INDICES"])
    graph = build_graph_from_subgraphs(pooled, types,
                                       distance_threshold=GRAPH_DISTANCE_THRESHOLD,
                                       adjacency_builder=get_adjacency_config(adjacency_config))
    nodes = list(graph["nodes"])
    if not nodes:
        return None

    # 🔴 节点顺序必须与 dataset.py 建 gnn_x 时逐位一致（那里也是 `for i in gnn_graph['nodes']`）
    is_agent = torch.tensor([types[n] == "agent" for n in nodes], dtype=torch.bool)
    agent_sids = [n for n in nodes if types[n] == "agent"]
    if not agent_sids:
        empty = (is_agent, torch.zeros(0, FUTURE_FRAMES, 2),
                 torch.zeros(0, FUTURE_FRAMES, dtype=torch.bool))
        return empty + (np.zeros(0, dtype=np.int64),) if return_slots else empty

    # ── 坑 1：位置 join 恢复槽位（不得按下标猜）──────────────────────────
    hist_end = np.asarray(row["AGENT_HISTORIES"])[:, -1, :2].astype(np.float64)
    slots = resolve_agent_slots(_agent_end_xy(row, agent_sids), hist_end, strict=strict)

    af = np.asarray(row["AGENT_FUTURES"])            # [20, 60, 3] 绝对 ego-centric 位置（坑 2）
    # 🔴 帧数不足时 `af[slot, :60, :2]` 会**静默截短**，下游 pred/tgt 形状对不上或错位 ⇒ 先断言
    if af.ndim != 3 or af.shape[1] < FUTURE_FRAMES or af.shape[2] < 2:
        msg = (f"AGENT_FUTURES 形状 {af.shape} 不满足前提 "
               f"(>= [*, {FUTURE_FRAMES}, 2]) —— 不得静默截短")
        if strict:
            raise AgentAuxWiringError(msg)
        logger.warning(msg)
        return None
    tgt = np.zeros((len(agent_sids), FUTURE_FRAMES, 2), dtype=np.float32)
    msk = np.zeros((len(agent_sids), FUTURE_FRAMES), dtype=bool)
    for i, slot in enumerate(slots):
        if slot < 0:                                  # strict=False 时的未匹配节点
            continue
        xy = af[slot, :FUTURE_FRAMES, :2]
        tgt[i] = xy
        # ── 坑 3：零填充帧不参与 loss（实测 9.6% 的帧、21% 的槽受影响）──
        msk[i] = np.abs(xy).sum(axis=1) > 0

    out = (is_agent, torch.from_numpy(tgt), torch.from_numpy(msk))
    return out + (slots,) if return_slots else out


def attach_agent_aux(data, row, adjacency_config: str = "default", *, strict: bool = True):
    """把三个字段挂到 `build_graph_data_from_feature_df` 产出的 GraphData 上（原地）。

    ⚠️ **不修改 `dataset.py`** —— 调用方在建完图之后手动调一次即可。
    """
    out = build_agent_aux_fields(row, adjacency_config, strict=strict)
    if out is None:
        n = int(data.gnn_x.shape[0]) if hasattr(data, "gnn_x") else 0
        data.gnn_is_agent = torch.zeros(n, dtype=torch.bool)
        data.agent_future_tgt = torch.zeros(0, FUTURE_FRAMES, 2)
        data.agent_future_mask = torch.zeros(0, FUTURE_FRAMES, dtype=torch.bool)
        return data

    is_agent, tgt, msk = out
    n_graph = int(data.gnn_x.shape[0]) if hasattr(data, "gnn_x") else -1
    if n_graph >= 0 and is_agent.numel() != n_graph:
        msg = (f"节点数对不上：aux 侧重建得到 {is_agent.numel()}，"
               f"而 gnn_x 有 {n_graph} —— adjacency_config 是否与建图时一致？")
        if strict:
            raise AgentAuxWiringError(msg)
        logger.warning(msg)
        is_agent = torch.zeros(max(n_graph, 0), dtype=torch.bool)
        tgt = torch.zeros(0, FUTURE_FRAMES, 2)
        msk = torch.zeros(0, FUTURE_FRAMES, dtype=torch.bool)

    data.gnn_is_agent = is_agent
    data.agent_future_tgt = tgt
    data.agent_future_mask = msk
    return data


__all__ = ["AgentAuxWiringError", "attach_agent_aux", "build_agent_aux_fields", "masked_aux_loss"]
