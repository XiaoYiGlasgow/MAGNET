"""R1 逐帧建图的单测（Q76 重构，数据侧）。

🔴 命门判据是 [E] 等价门：**逐帧特征沿时间轴取平均，必须逐位等于现有 `subgraph_pooling` 的输出**。
它证明新建图与旧管线读的是同一份数据、同一个子图划分 —— 沿用 Q49 `rebuild_agent_subgraph.py`
的做法（那次逐字节等价门 27000/27000 全过，才敢翻阈值）。

跑法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest tests/test_temporal_graph.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.shared.utils.dataset import subgraph_pooling  # noqa: E402
from src.shared.utils.temporal_graph import (  # noqa: E402
    NODE_TYPE_AGENT,
    NODE_TYPE_EGO,
    NODE_TYPE_LANE,
    NUM_FRAMES,
    build_frame_edges,
    build_temporal_graph,
    flatten_to_pyg_batch,
)

VAL_DIR = _REPO / "data/processed_features_magnet_e1_none_v3_split3/val"
N_SAMPLES = 30


def _samples(n: int = N_SAMPLES):
    for p in sorted(VAL_DIR.glob("features_*.pkl"))[:n]:
        yield p, pd.read_pickle(p)


@pytest.fixture(scope="module")
def built():
    """预先构建一批样本，避免每个用例重复读盘。"""
    out = []
    for p, df in _samples():
        payload, skip = build_temporal_graph(df, topology="full")
        out.append((p.stem, df, payload, skip))
    assert out, f"没有读到样本，检查 {VAL_DIR}"
    return out


def test_a_all_samples_build(built):
    """[A] 所有样本都能建图，且帧数恒为 19。"""
    for token, _, payload, skip in built:
        assert skip is None, f"{token} 建图失败：{skip}"
        assert payload["num_frames"] == NUM_FRAMES
        assert payload["temporal_x"].shape[0] == NUM_FRAMES


def test_b_node_layout(built):
    """[B] 节点顺序 = ego(1) + agents(K) + lanes(M)，且 index 0 恒为 ego。

    干预协议 §5 要求能把 ego 从干预池里剔除，靠的就是这个约定。
    """
    for token, _, payload, _ in built:
        types = payload["node_types"].numpy()
        assert types[0] == NODE_TYPE_EGO, f"{token} index 0 不是 ego"
        assert (types == NODE_TYPE_EGO).sum() == 1, f"{token} ego 节点不唯一"
        # 类型必须是分段的：ego 段、agent 段、lane 段
        agent_pos = np.where(types == NODE_TYPE_AGENT)[0]
        lane_pos = np.where(types == NODE_TYPE_LANE)[0]
        if agent_pos.size and lane_pos.size:
            assert agent_pos.max() < lane_pos.min(), f"{token} agent/lane 顺序交错"


def test_c_lane_static_agent_dynamic(built):
    """[C] lane 节点逐帧完全相同；agent 节点必须逐帧变化。

    这条直接检验重构的核心意图 —— 时间维真的被保留下来了。
    """
    n_checked_agent = 0
    for token, _, payload, _ in built:
        x = payload["temporal_x"]
        types = payload["node_types"].numpy()

        lane_idx = np.where(types == NODE_TYPE_LANE)[0]
        if lane_idx.size:
            lane_x = x[:, lane_idx, :]
            spread = (lane_x - lane_x[0:1]).abs().max().item()
            assert spread == 0.0, f"{token} lane 节点逐帧不一致（max|Δ|={spread}）"

        agent_idx = np.where(types == NODE_TYPE_AGENT)[0]
        for i in agent_idx:
            moved = (x[:, i, :] - x[0, i, :]).abs().max().item()
            assert moved > 0.0, f"{token} agent 节点 {i} 逐帧完全不变，时间维没进来"
            n_checked_agent += 1

    assert n_checked_agent > 0, "样本里一个 agent 都没有，该断言失去意义"


def test_d_topology(built):
    """[D] full 每帧边相同；knn30 至少在部分样本上逐帧不同（agent 在移动）。"""
    knn_varies = 0
    for token, df, payload, _ in built:
        full_edges = payload["frame_edges"]
        for t in range(1, len(full_edges)):
            assert torch.equal(full_edges[t], full_edges[0]), f"{token} full 拓扑逐帧不一致"

        knn_payload, _ = build_temporal_graph(df, topology="knn30")
        knn_edges = knn_payload["frame_edges"]
        sizes = {e.shape[1] for e in knn_edges}
        if len(sizes) > 1:
            knn_varies += 1
        for e in knn_edges:
            assert e.shape[1] <= full_edges[0].shape[1], f"{token} knn30 边多于完全图"

    assert knn_varies > 0, "knn30 在所有样本上都逐帧不变 —— 距离阈值可能没生效"


def test_e_equivalence_gate_vs_subgraph_pooling(built):
    """[E] 🔴 命门：逐帧特征沿时间轴平均 == 现有 `subgraph_pooling` 输出的前 9 维。

    证明新建图与旧管线读的是同一份数据、同一份子图划分。
    不过这一关，后面所有读数都不可信。

    🔴 **判据用相对误差而非绝对误差，理由是查证出来的不是拍的**（第八十四条：
    闸门触发时先验证闸门自身的度量）。首版用 `max|Δ| < 1e-5` 失败于 1.526e-05，
    用 float64 重算判定为 **float32 累加舍入**而非逻辑错误::

        numpy f32 vs float64 真值 : 6.023e-06   ← 现有管线自身的偏差
        torch f32 vs float64 真值 : 9.236e-06   ← 本实现自身的偏差
        两实现相对差 2.106e-07；float32 eps = 1.192e-07；19 次累加预期 ≈ 3.765e-05

    两个实现都落在 float64 真值附近、只是求和顺序不同 ⇒ 正确判据是**相对误差**，
    并同时与 float64 真值三方对照（将来出真错误时能与舍入区分开）。
    """
    max_rel_diff = 0.0
    max_rel_vs_f64 = 0.0
    n_entities = 0

    for token, df, payload, _ in built:
        features = df["POLYLINE_FEATURES"].values[0].astype(np.float32)
        features64 = df["POLYLINE_FEATURES"].values[0].astype(np.float64)
        sid_to_idx = df["SUBGRAPH_ID_TO_NODE_INDICES"].values[0]
        sid_to_type = df["SUBGRAPH_ID_TO_TYPE"].values[0]
        pooled_dict = subgraph_pooling(features, sid_to_idx)

        x = payload["temporal_x"]

        # 按与 _collect_entities 相同的顺序重建 sid 列表：ego -> agent -> lane
        ordered_sids = (
            [s for s in sid_to_idx if sid_to_type.get(s) == "ego"]
            + [s for s in sid_to_idx if sid_to_type.get(s) == "agent"]
            + [s for s in sid_to_idx if sid_to_type.get(s) == "lane"]
        )
        assert len(ordered_sids) == x.shape[1], f"{token} 节点数与子图数不符"

        for node_i, sid in enumerate(ordered_sids):
            if sid not in pooled_dict:
                continue
            start, end = sid_to_idx[sid]
            truth64 = features64[start:end + 1, :9].mean(axis=0)
            scale = max(float(np.abs(truth64).max()), 1.0)

            mine = x[:, node_i, :].mean(dim=0).numpy().astype(np.float64)
            theirs = np.asarray(pooled_dict[sid], dtype=np.float32)[:9].astype(np.float64)

            rel = float(np.abs(mine - theirs).max() / scale)
            rel_f64 = float(np.abs(mine - truth64).max() / scale)
            max_rel_diff = max(max_rel_diff, rel)
            max_rel_vs_f64 = max(max_rel_vs_f64, rel_f64)
            n_entities += 1

            assert rel < 1e-6, (
                f"{token} 子图 {sid}（{sid_to_type.get(sid)}）等价门失败："
                f"相对差 {rel:.3e} ≥ 1e-6，已超出 float32 舍入可解释范围"
            )
            assert rel_f64 < 1e-6, (
                f"{token} 子图 {sid} 与 float64 真值偏离 {rel_f64:.3e} —— 这是逻辑错误不是舍入"
            )

    assert n_entities > 0
    print(
        f"\n[E] 等价门通过：{n_entities} 个实体｜"
        f"与现管线相对差 max {max_rel_diff:.3e}｜与 float64 真值相对差 max {max_rel_vs_f64:.3e}"
        f"（float32 eps = {np.finfo(np.float32).eps:.3e}）"
    )


def test_f_pyg_flatten(built):
    """[F] 摊平成 PyG 大图后，形状与帧索引正确，且边不跨帧。"""
    for token, _, payload, _ in built:
        x, edge_index, frame_batch = flatten_to_pyg_batch(payload)
        t, n = payload["num_frames"], payload["num_nodes"]

        assert x.shape == (t * n, 9), f"{token} 摊平后形状错"
        assert frame_batch.shape == (t * n,)
        assert int(frame_batch.max()) == t - 1
        assert (frame_batch == 0).sum().item() == n

        if edge_index.numel() > 0:
            src_frame = frame_batch[edge_index[0]]
            dst_frame = frame_batch[edge_index[1]]
            assert torch.equal(src_frame, dst_frame), f"{token} 出现跨帧边"
            assert int(edge_index.max()) < t * n


def test_g_empty_graph_samples(built):
    """[G] 无 agent 的样本（实测约 20%）仍能建图，只是没有 agent 节点。"""
    n_empty = sum(
        1 for _, _, p, _ in built if (p["node_types"] == NODE_TYPE_AGENT).sum().item() == 0
    )
    for token, _, payload, skip in built:
        assert skip is None, f"{token} 意外被跳过"
        assert payload["num_nodes"] >= 1
    print(f"\n[G] {n_empty}/{len(built)} 个样本无 agent（与已知空图率同量级）")


def test_h_zero_cache_path_equivalence(built):
    """[H] 🔴 零缓存路径等价门：`rebuild_temporal_from_x` == `build_temporal_graph`。

    训练走 transform 从 `GraphData.x` 现场重建（省 14 GB 缓存），
    而审计/离线分析可能从 pkl 直接建 —— **两条路必须逐位一致**，否则读到的是两张图。
    """
    from src.shared.utils.dataset import build_graph_data_from_feature_df
    from src.shared.utils.temporal_graph import (
        flatten_to_pyg_batch,
        rebuild_temporal_from_x,
    )

    n_checked = 0
    for topology in ("full", "knn30"):
        for token, df, _, _ in built[:12]:
            gd, skip = build_graph_data_from_feature_df(df)
            if gd is None:
                continue
            payload, _ = build_temporal_graph(df, topology=topology)
            ref_x, ref_ei, ref_frame = flatten_to_pyg_batch(payload)
            got_x, got_ei, got_frame = rebuild_temporal_from_x(gd.x, topology)

            assert got_x.shape == ref_x.shape, f"{token}/{topology} tg_x 形状 {got_x.shape} vs {ref_x.shape}"
            dx = float((got_x - ref_x).abs().max())
            assert dx == 0.0, f"{token}/{topology} tg_x 不逐位相同 max|Δ|={dx:.3e}"
            assert torch.equal(got_frame, ref_frame), f"{token}/{topology} tg_frame 不一致"
            assert got_ei.shape == ref_ei.shape, (
                f"{token}/{topology} 边数 {got_ei.shape[1]} vs {ref_ei.shape[1]}"
            )
            assert torch.equal(got_ei, ref_ei), f"{token}/{topology} edge_index 不一致"
            n_checked += 1

    assert n_checked > 0
    print(f"\n[H] 零缓存路径等价门通过：{n_checked} 个(样本×拓扑)，tg_x/tg_frame/edge_index 全部逐位相同")
