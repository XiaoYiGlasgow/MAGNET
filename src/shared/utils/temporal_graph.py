"""R1 逐帧场景图构建 —— Q76 重构的数据侧。

背景（Q76，08-04 用户裁定）：现有 `dataset.py:45` 的 `subgraph_pooling` 用**零参数、零学习的
`np.mean`** 沿时间轴把每个实体的 19 帧运动压成 1 个节点（对照 VectorNet 同一步是 MLP + max-pool），
⇒ 送进 GNN 的是「20 帧的平均场景」而不是「t 时刻的场景」，交互的动态信息（相对速度、逼近、TTC）
在建图之前就没了。

本模块实现原始设计意图：**每个时刻一个场景图**（ego + agents + lanes），
图与图之间在时序上串联，交由上层的时序 Transformer 处理。

实测的数据前提（08-04 在 60 个真实样本上探查，非假设）::

    agent 子图行数  恒为 19（173/173）   每行 = 一个时间步，col6 timestamp 单调递增
    ego   子图行数  恒为 19（60/60）
    lane  子图行数  恒为 49（2400/2400） 🔴 行 = 空间折线段，timestamp 全 0，**不是时间步**

⇒ T = 19 固定，不需要 padding/mask。

R1 的设计决策（预注册，见 §9 重构路线）::

    D1  lane 每帧复用同一个池化节点，沿用 np.mean —— R1 的唯一变量是「agent 是否逐帧」，
        池化算子留给 R2，不在一格里混两个变量。
    D3  拓扑可配置：`full`（完全图，每帧边相同）/ `knn30`（30 m 距离阈值，**每帧边不同**）。
        后者才是逐帧建图的真正价值 —— agent 在移动，谁和谁相邻随帧变化。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: 每个实体的时间步数（实测恒为 19，见模块 docstring）
NUM_FRAMES: int = 19

#: 送进 GNN 的核心特征维度（与 `_extract_core_features` 一致，取 POLYLINE_FEATURES 前 9 列）
NODE_FEATURE_DIM: int = 9

#: 节点类型编码
NODE_TYPE_EGO: int = 0
NODE_TYPE_AGENT: int = 1
NODE_TYPE_LANE: int = 2

#: `knn30` 拓扑的距离阈值（m），沿用 `build_graph_from_subgraphs` 的既有默认值
KNN_DISTANCE_THRESHOLD: float = 30.0


class TemporalGraphError(ValueError):
    """逐帧建图失败；调用方须显式处理，不得静默跳过。"""


def _collect_entities(
    features: np.ndarray,
    subgraph_id_to_indices: Dict,
    subgraph_id_to_types: Dict,
) -> Tuple[list, list, list]:
    """按类型收集实体的原始行块。

    Args:
        features: POLYLINE_FEATURES，形状 [N_rows, 12]。
        subgraph_id_to_indices: 子图 ID -> (start, end) 行区间（闭区间）。
        subgraph_id_to_types: 子图 ID -> 'ego' / 'agent' / 'lane'。

    Returns:
        (dynamic_blocks, lane_blocks, node_types)
        dynamic_blocks: 每个元素是 [19, 9] 的 ego/agent 行块（随时间变化）。
        lane_blocks: 每个元素是 [49, 9] 的 lane 行块（静态）。
        node_types: 与 dynamic_blocks + lane_blocks 顺序对应的类型编码。
    """
    dynamic_blocks, lane_blocks, node_types = [], [], []

    # ego 与 agent 必须先于 lane，保证 index 0 = ego（与干预协议 §5 剔除 ego 的约定一致）
    for wanted, type_code in ((("ego",), NODE_TYPE_EGO), (("agent",), NODE_TYPE_AGENT)):
        for sid, (start, end) in subgraph_id_to_indices.items():
            if subgraph_id_to_types.get(sid) not in wanted:
                continue
            block = features[start:end + 1, :NODE_FEATURE_DIM]
            if block.shape[0] != NUM_FRAMES:
                raise TemporalGraphError(
                    f"子图 {sid}（{subgraph_id_to_types.get(sid)}）有 {block.shape[0]} 行，"
                    f"期望 {NUM_FRAMES} —— 实测数据恒为 19，出现例外必须查明而不是补零"
                )
            dynamic_blocks.append(block)
            node_types.append(type_code)

    for sid, (start, end) in subgraph_id_to_indices.items():
        if subgraph_id_to_types.get(sid) != "lane":
            continue
        block = features[start:end + 1, :NODE_FEATURE_DIM]
        if block.shape[0] < 1:
            continue
        # D1：lane 是静态几何，其「行」是空间折线段而非时间步 ⇒ 沿用 np.mean 压成 1 个节点
        lane_blocks.append(block.mean(axis=0))
        node_types.append(NODE_TYPE_LANE)

    return dynamic_blocks, lane_blocks, node_types


def _build_frame_features(
    dynamic_blocks: list,
    lane_blocks: list,
) -> np.ndarray:
    """拼出逐帧节点特征张量。

    Returns:
        形状 [T, N, 9]。前 len(dynamic_blocks) 个节点随帧变化，其余 lane 节点每帧复用。
    """
    n_dynamic, n_lane = len(dynamic_blocks), len(lane_blocks)
    n_nodes = n_dynamic + n_lane
    frames = np.zeros((NUM_FRAMES, n_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    if n_dynamic > 0:
        # [n_dynamic, 19, 9] -> [19, n_dynamic, 9]
        frames[:, :n_dynamic, :] = np.stack(dynamic_blocks, axis=0).transpose(1, 0, 2)
    if n_lane > 0:
        # lane 静态：同一份特征在 T 帧上广播
        frames[:, n_dynamic:, :] = np.stack(lane_blocks, axis=0)[None, :, :]

    return frames


def _node_positions(frame: np.ndarray) -> np.ndarray:
    """取某一帧每个节点的代表位置（用 x_start, y_start 两列）。"""
    return frame[:, :2]


def build_frame_edges(
    frames: np.ndarray,
    topology: str = "full",
) -> list:
    """为每一帧构建无向边（已补反向，GCN 需对称邻接）。

    Args:
        frames: [T, N, 9] 逐帧节点特征。
        topology: `full` = 完全图（每帧边相同）；`knn30` = 30 m 距离阈值（**每帧边不同**）。

    Returns:
        长度 T 的列表，每个元素是 [2, E_t] 的 int64 ndarray。

    Raises:
        TemporalGraphError: topology 取值非法。
    """
    num_frames, n_nodes = frames.shape[0], frames.shape[1]

    if topology == "full":
        if n_nodes < 2:
            empty = np.zeros((2, 0), dtype=np.int64)
            return [empty for _ in range(num_frames)]
        iu = np.triu_indices(n_nodes, k=1)
        directed = np.stack(iu, axis=0).astype(np.int64)
        undirected = np.concatenate([directed, directed[[1, 0], :]], axis=1)
        return [undirected for _ in range(num_frames)]

    if topology == "knn30":
        edges = []
        for t in range(num_frames):
            pos = _node_positions(frames[t])
            if n_nodes < 2:
                edges.append(np.zeros((2, 0), dtype=np.int64))
                continue
            dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            iu = np.triu_indices(n_nodes, k=1)
            keep = dist[iu] <= KNN_DISTANCE_THRESHOLD
            directed = np.stack([iu[0][keep], iu[1][keep]], axis=0).astype(np.int64)
            undirected = np.concatenate([directed, directed[[1, 0], :]], axis=1)
            edges.append(undirected)
        return edges

    raise TemporalGraphError(f"未知拓扑 {topology!r}，可选 'full' / 'knn30'")


def build_temporal_graph(
    data,
    topology: str = "full",
) -> Tuple[Optional[Dict], Optional[str]]:
    """从单行 feature DataFrame 构建逐帧场景图。

    与 `build_graph_data_from_feature_df` 并列，供 R1 的时序模型使用；
    **不修改**原函数，保证旧架构实验可复现、可对照。

    Args:
        data: 单行 pandas DataFrame（列同 `encoding_vectornet_features` 产出）。
        topology: 见 `build_frame_edges`。

    Returns:
        (payload, skip_reason)。payload 为 dict：
        `temporal_x` [T, N, 9] / `frame_edges` 长度 T 的边列表 /
        `node_types` [N] / `num_frames` / `num_nodes`。
        失败时 payload 为 None，skip_reason ∈ {'no_subgraph', 'no_entity'}。
    """
    if "SUBGRAPH_ID_TO_NODE_INDICES" not in data.columns:
        return None, "no_subgraph"

    features = data["POLYLINE_FEATURES"].values[0].astype(np.float32)
    subgraph_id_to_indices = data["SUBGRAPH_ID_TO_NODE_INDICES"].values[0]
    subgraph_id_to_types = data["SUBGRAPH_ID_TO_TYPE"].values[0]

    dynamic_blocks, lane_blocks, node_types = _collect_entities(
        features, subgraph_id_to_indices, subgraph_id_to_types
    )
    if len(dynamic_blocks) + len(lane_blocks) == 0:
        return None, "no_entity"

    frames = _build_frame_features(dynamic_blocks, lane_blocks)
    frame_edges = build_frame_edges(frames, topology=topology)

    payload = {
        "temporal_x": torch.from_numpy(frames),
        "frame_edges": [torch.from_numpy(e) for e in frame_edges],
        "node_types": torch.tensor(node_types, dtype=torch.long),
        "num_frames": NUM_FRAMES,
        "num_nodes": frames.shape[1],
    }
    return payload, None


def flatten_to_pyg_batch(payload: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 T 帧图摊平成一个 PyG 大图，供 GNN 一次前向算完 T 帧。

    这是 R1 的效率关键：GCN 只调用一次而不是 T 次，
    `global_mean_pool(x, frame_batch)` 直接得到 [T, hidden]。

    Args:
        payload: `build_temporal_graph` 的返回值。

    Returns:
        (x, edge_index, frame_batch)
        x: [T*N, 9]；edge_index: [2, sum_t E_t]（已按帧偏移）；frame_batch: [T*N] 帧索引。
    """
    temporal_x = payload["temporal_x"]
    num_frames, num_nodes = temporal_x.shape[0], temporal_x.shape[1]

    x = temporal_x.reshape(num_frames * num_nodes, -1)
    frame_batch = torch.arange(num_frames, dtype=torch.long).repeat_interleave(num_nodes)

    shifted = []
    for t, edges in enumerate(payload["frame_edges"]):
        if edges.numel() > 0:
            shifted.append(edges + t * num_nodes)
    edge_index = (
        torch.cat(shifted, dim=1) if shifted else torch.zeros(2, 0, dtype=torch.long)
    )
    return x, edge_index, frame_batch


# ────────────────────────────────────────────────────────────────────────
# 零缓存路径：从已有 GraphData.x 现场重建逐帧图
# ────────────────────────────────────────────────────────────────────────
#
# 为什么不把 tg_* 存进 InMemoryDataset 缓存：`tg_edge_index` 在 full 拓扑下是
# 42 864 条边/样本（int64 ⇒ 686 KB），L1 的 20 936 个样本合计 **≈14 GB**。
# 而 `GraphData.x` 本来就带着全部原始行（含 col9 = subgraph_ID），
# ⇒ 现场重排即可，**零缓存增长、零重新 process，可直接复用现有 L1 缓存起跑**。


def rebuild_temporal_from_x(
    x: torch.Tensor,
    topology: str = "full",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 `GraphData.x`（[N_rows, 12]）现场重建逐帧图。

    行顺序沿用 `build_graph_data_from_feature_df`：先 traj（ego 在最前）后 lane，
    每个子图的行在 x 里是连续的，col9 = subgraph_ID。

    Args:
        x: [N_rows, >=10] 的原始行特征。
        topology: `full` / `knn30`。

    Returns:
        (tg_x, tg_edge_index, tg_frame)，语义同 `flatten_to_pyg_batch`。

    Raises:
        TemporalGraphError: 出现行数既非 NUM_FRAMES 也不像 lane 的子图。
    """
    sid = x[:, 9].round().to(torch.int64)
    # 保持出现顺序（torch.unique 会排序，这里手动去重以保住 ego 在 index 0）
    order, seen = [], set()
    for v in sid.tolist():
        if v not in seen:
            seen.add(v)
            order.append(v)

    dynamic, lane = [], []
    for v in order:
        block = x[sid == v, :NODE_FEATURE_DIM]
        if block.shape[0] == NUM_FRAMES:
            dynamic.append(block)
        elif block.shape[0] >= 2:
            # 🔴 必须与 `_collect_entities` 用**同一种**求平均实现（numpy），
            # 否则 torch 与 numpy 的累加顺序差异会让两条路径产出相差 ~1e-5 的两张图
            # （训练走本函数、审计走 build_temporal_graph）。这不是精度问题，是口径问题。
            lane.append(torch.from_numpy(block.numpy().mean(axis=0)))
        else:
            raise TemporalGraphError(f"子图 {v} 只有 {block.shape[0]} 行，无法归类")

    n_dyn, n_lane = len(dynamic), len(lane)
    if n_dyn + n_lane == 0:
        raise TemporalGraphError("x 里没有可用子图")

    frames = torch.zeros(NUM_FRAMES, n_dyn + n_lane, NODE_FEATURE_DIM, dtype=torch.float32)
    if n_dyn:
        frames[:, :n_dyn, :] = torch.stack(dynamic, dim=0).transpose(0, 1)
    if n_lane:
        frames[:, n_dyn:, :] = torch.stack(lane, dim=0).unsqueeze(0)

    edges = build_frame_edges(frames.numpy(), topology=topology)
    payload = {
        "temporal_x": frames,
        "frame_edges": [torch.from_numpy(e) for e in edges],
        "num_frames": NUM_FRAMES,
        "num_nodes": frames.shape[1],
    }
    return flatten_to_pyg_batch(payload)


class AttachTemporalGraph:
    """PyG transform：在 `__getitem__` 时给样本挂上 `tg_x` / `tg_edge_index` / `tg_frame`。

    用法::

        ds = GraphDataset(root, transform=AttachTemporalGraph(topology="knn30"))
        loader = DataLoader(ds, batch_size=32, follow_batch=["tg_x"])
    """

    def __init__(self, topology: str = "full") -> None:
        self.topology = topology

    def __call__(self, data):
        try:
            tg_x, tg_edge_index, tg_frame = rebuild_temporal_from_x(data.x, self.topology)
        except TemporalGraphError as exc:
            # 🔴 不静默吞掉：记下来，让上层能看到有多少样本没挂上
            logger.warning("逐帧图重建失败，退回空张量：%s", exc)
            tg_x = torch.zeros(0, NODE_FEATURE_DIM, dtype=torch.float32)
            tg_edge_index = torch.zeros(2, 0, dtype=torch.long)
            tg_frame = torch.zeros(0, dtype=torch.long)
        data.tg_x = tg_x
        data.tg_edge_index = tg_edge_index
        data.tg_frame = tg_frame
        return data

    def __repr__(self) -> str:
        return f"{type(self).__name__}(topology={self.topology!r})"
