"""R1 逐帧场景图的干预原语 —— 让 `run_interaction_audit.py` 能审计 `TemporalPredictor`。

为什么需要单独一份
------------------
`d2_semantic_intervention.py` 的 `zero_nodes` / `intervene_remove_nodes` 打在
`gnn_x` / `gnn_edge_index` 上，那是**旧架构**（建图前已沿时间轴 `np.mean` 压成 1 节点）的图。
R1 的 GNN 分支吃的是 `tg_x` / `tg_edge_index`（见 `models/temporal_predictor.py:92-94`），
直接沿用旧原语的后果不是报错而是**恒为 0 的 response** —— 会被读成「模型不依赖交互输入」，
而真相是「干预压根没作用在被测通路上」。这正是第八十四条要防的那类伪装成科学结论的缺陷。

口径：干预单位是**实体**，不是 tg 行
-----------------------------------
R1 里一个 agent 占 `T=19` 行（每帧一行），旧架构里占 1 行。
若按行计数，「删 1 个节点」在两侧会变成删掉不同数量的**实体** ⇒ 剂量合同（协议 §3）失配 19×。
⇒ 本模块的 `drop` 一律是**实体下标**（与 `gnn_x` 的行下标同一套编号，见
`verify_entity_alignment` 的运行期校验），内部再展开到 `t*N + k for all t`。
这样 `k_target` / `frac_target` 与旧架构逐字同义，ΔS 才是可配对的。

边的重建：重算而不是过滤
------------------------
删掉实体后，边按**幸存节点重新跑一遍 `build_frame_edges`** ——
得到的正是「该 agent 从未存在过」这张反事实图，与拓扑定义严格自洽。
对 `full` 与 `knn30` 两种拓扑，「重算」与「过滤掉关联边再重编号」逐位等价
（`tests/test_temporal_intervention.py::test_rebuild_equals_filter` 两种拓扑各验一遍）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent), str(_GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.temporal_graph import (  # noqa: E402
    NODE_FEATURE_DIM,
    NUM_FRAMES,
    TemporalGraphError,
    build_frame_edges,
    flatten_to_pyg_batch,
    rebuild_temporal_from_x,
)

__all__ = [
    "TemporalInterventionError",
    "attach_temporal_graph",
    "entity_order_from_x",
    "num_entities",
    "remove_temporal_entities",
    "verify_entity_alignment",
    "zero_temporal_entities",
]

if NODE_FEATURE_DIM != 9:   # 建图侧改了维度 ⇒ 本模块的剂量/对齐注释需同步复核
    raise ImportError(f"NODE_FEATURE_DIM 变成了 {NODE_FEATURE_DIM}，请复核本模块的口径说明")


class TemporalInterventionError(RuntimeError):
    """逐帧图干预的前置条件不成立 —— 必须暴露，不得静默产出 0 response。"""


def entity_order_from_x(x: torch.Tensor) -> List[int]:
    """还原 `rebuild_temporal_from_x` 的实体顺序（先随帧变化的 ego/agent，后 lane）。

    这段分类逻辑与 `rebuild_temporal_from_x` 必须一致 —— 它是本模块唯一「猜」得起的地方，
    故不复制判据、只复制**同一个**判据（行数 == NUM_FRAMES ⇒ dynamic，否则 lane）。

    :param x: `GraphData.x`，[N_rows, >=10]，第 9 列是 subgraph_ID。
    :returns: 实体的 subgraph_ID 列表，下标即 `tg_x` 每帧内的节点下标。
    """
    sid = x[:, 9].round().to(torch.int64)
    order, seen = [], set()
    for v in sid.tolist():
        if v not in seen:
            seen.add(v)
            order.append(v)

    dynamic, lane = [], []
    for v in order:
        n_rows = int((sid == v).sum())
        if n_rows == NUM_FRAMES:
            dynamic.append(v)
        elif n_rows >= 2:
            lane.append(v)
        else:
            raise TemporalInterventionError(f"子图 {v} 只有 {n_rows} 行，无法归类")
    return dynamic + lane


def verify_entity_alignment(x: torch.Tensor, gnn_node_order: Sequence[int]) -> None:
    """校验 `tg_x` 的实体顺序与 `gnn_x` 的行顺序**逐位相同**。

    为什么必须运行期校验而不是相信实测：`gnn_x` 的顺序是
    `subgraph_pooling()` 产出 dict 的键序，`tg_x` 的顺序是 `x` 里的出现顺序再按
    dynamic/lane 重排 —— 两条来源不同的路径。12 个真实样本上实测相同（08-04），
    但一旦某个样本上 lane 夹在 agent 中间，两者就会错位，
    而错位的后果是**干预打到了别的实体上**且不会报错。

    :param gnn_node_order: `gnn_x` 的行对应的 subgraph_ID 序列。
    :raises TemporalInterventionError: 顺序不一致（调用方应记 skip，不得猜对应关系）。
    """
    tg_order = entity_order_from_x(x)
    if list(gnn_node_order) != tg_order:
        raise TemporalInterventionError(
            f"实体顺序不一致：gnn_x 前 5 个 {list(gnn_node_order)[:5]} vs "
            f"tg 前 5 个 {tg_order[:5]}（n={len(gnn_node_order)}/{len(tg_order)}）")


def attach_temporal_graph(data, topology: str = "full"):
    """给 `GraphData` 挂上 `tg_x` / `tg_edge_index` / `tg_frame`（原对象被就地修改）。

    走的是**训练同一条路径** `rebuild_temporal_from_x`（零缓存重建），
    而不是 pkl 侧的 `build_temporal_graph` —— 两条路径已由
    `tests/test_temporal_graph.py::test_h` 验过 24 个(样本×拓扑)逐位相同，
    这里选训练那条，保证审计看到的就是训练时喂进模型的那张图。

    :returns: 传入的 `data`（便于链式调用）。
    :raises TemporalInterventionError: 建图失败（调用方记 skip）。
    """
    try:
        tg_x, tg_edge_index, tg_frame = rebuild_temporal_from_x(data.x, topology)
    except TemporalGraphError as exc:
        raise TemporalInterventionError(f"逐帧图重建失败: {exc}") from exc
    data.tg_x = tg_x
    data.tg_edge_index = tg_edge_index
    data.tg_frame = tg_frame
    return data


def _frames_view(tg_x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """把 [T*N, 9] 还原成 [T, N, 9]（`flatten_to_pyg_batch` 的逆操作）。"""
    if tg_x.shape[0] != NUM_FRAMES * n_nodes:
        raise TemporalInterventionError(
            f"tg_x 行数 {tg_x.shape[0]} ≠ T*N = {NUM_FRAMES}*{n_nodes}")
    return tg_x.view(NUM_FRAMES, n_nodes, tg_x.shape[1])


def _check_drop_bounds(drop: np.ndarray, n_nodes: int) -> np.ndarray:
    """校验实体下标在 `[0, n_nodes)` 内。

    🔴 **这是静默类缺陷的防线，不是形式检查**：越界下标在两个算子上都不报错，
    但后果相反且都错 ——
    `zero` 侧 `t*N + k` 会**折到下一帧的另一个实体**上（n=43、k=43 ⇒ 第 1 帧的 ego 被置零）；
    `remove` 侧 `np.setdiff1d` 直接忽略越界值 ⇒ **什么都没删**，response 却被当成真读数。
    """
    d = np.asarray(drop, dtype=np.int64)
    if d.size and (d.min() < 0 or d.max() >= n_nodes):
        raise TemporalInterventionError(
            f"实体下标越界：drop∈[{d.min()},{d.max()}]，实体数 {n_nodes}")
    return d


def _entity_rows(drop: np.ndarray, n_nodes: int) -> torch.Tensor:
    """实体下标 -> 它在 `tg_x` 里占的全部行下标（`t*N + k`，t 遍历 T 帧）。"""
    d = _check_drop_bounds(drop, n_nodes)
    if d.size == 0:
        return torch.zeros(0, dtype=torch.long)
    k = torch.as_tensor(d, dtype=torch.long)
    offsets = torch.arange(NUM_FRAMES, dtype=torch.long) * n_nodes
    return (offsets[:, None] + k[None, :]).reshape(-1)


def zero_temporal_entities(tg_x: torch.Tensor, drop: np.ndarray,
                           n_nodes: int) -> torch.Tensor:
    """把给定**实体**在全部 T 帧上的节点特征置零；图结构不变。

    与旧架构 `zero_nodes` 同义：节点数不变、边不变，只是这些实体没信息了。

    :param drop: 实体下标（空数组 ⇒ 恒等操作，返回逐位相同的副本）。
    :returns: 新的 `tg_x`；原张量不被修改。
    """
    out = tg_x.clone()
    rows = _entity_rows(np.asarray(drop), n_nodes)
    if rows.numel():
        out[rows] = 0.0
    return out


def remove_temporal_entities(tg_x: torch.Tensor, drop: np.ndarray, n_nodes: int,
                             topology: str = "full",
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """真正移除实体：每帧删掉这些节点，按幸存节点**重算**每帧的边。

    :param drop: 实体下标（空数组 ⇒ 恒等，返回与输入逐位相同的三元组）。
    :returns: `(tg_x, tg_edge_index, tg_frame)`，语义同 `flatten_to_pyg_batch`。
    :raises TemporalInterventionError: 删完没有剩余节点。
    """
    frames = _frames_view(tg_x, n_nodes)
    keep = np.setdiff1d(np.arange(n_nodes), _check_drop_bounds(drop, n_nodes))
    if keep.size == 0:
        raise TemporalInterventionError("移除后没有剩余节点")

    kept = frames[:, torch.as_tensor(keep, dtype=torch.long), :].contiguous()
    edges = build_frame_edges(kept.numpy(), topology=topology)
    payload = {
        "temporal_x": kept,
        "frame_edges": [torch.from_numpy(e) for e in edges],
        "num_frames": NUM_FRAMES,
        "num_nodes": int(keep.size),
    }
    return flatten_to_pyg_batch(payload)


def num_entities(tg_x: torch.Tensor) -> int:
    """由 `tg_x` 行数反推实体数（T 固定为 `NUM_FRAMES`）。"""
    rows = tg_x.shape[0]
    if rows % NUM_FRAMES != 0:
        raise TemporalInterventionError(f"tg_x 行数 {rows} 不是 T={NUM_FRAMES} 的整数倍")
    return rows // NUM_FRAMES
