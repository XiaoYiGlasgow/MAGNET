#!/usr/bin/env python3
"""旋钮 A（判据 20260810 §2 起跑前置门）：subgraph_pooling 聚合参数四道实现门。

[A] agg 缺省 / 'mean' 与旧实现（硬编码 np.mean）逐位等价 —— 默认路径零行为改变（候选96）；
[B] 'max' / 'sum' 输出与 'mean' 不同且形状同构；非法 agg 当场 ValueError；
[C] 全局 set/get 生效；GraphDataset 缓存名：mean 保旧名（现有缓存不失效），
    max/sum 带 agg 后缀（杜绝静默复用 mean 缓存 —— 本格头号风险门）;
[D] build_graph_data_from_feature_df 穿透：同一真实样本 mean vs max 的 gnn_x 不同、形状同。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.dataset import (  # noqa: E402
    GraphDataset,
    build_graph_data_from_feature_df,
    get_current_subgraph_agg,
    set_subgraph_agg,
    subgraph_pooling,
)

#: 判据 §2-4 指定用真实样本；该 token 属新档 val（08-08 已验证在盘）
REAL_PKL = _GNN / "data/processed_features_fullret_L1_logsplit/val/features_00030d6f192754e6.pkl"


def _toy():
    rng = np.random.default_rng(20260810)
    polyline = rng.normal(size=(60, 12)).astype(np.float32)
    idx = {0: (0, 18), 1: (19, 37), 2: (38, 56)}
    return polyline, idx


def test_a_mean_bitwise_equivalent() -> None:
    polyline, idx = _toy()
    for call in (lambda: subgraph_pooling(polyline, idx),          # 缺省
                 lambda: subgraph_pooling(polyline, idx, "mean")):  # 显式 mean
        pooled = call()
        for sid, (s, e) in idx.items():
            ref = np.mean(polyline[s:e + 1], axis=0)
            assert np.array_equal(pooled[sid], ref), f"subgraph {sid} 与旧实现不逐位相等"
    print("  ✅ [A] 缺省/mean 与旧实现逐位等价")


def test_b_max_sum_differ_and_invalid_raises() -> None:
    polyline, idx = _toy()
    ref = subgraph_pooling(polyline, idx, "mean")
    for agg, fn in (("max", np.max), ("sum", np.sum)):
        pooled = subgraph_pooling(polyline, idx, agg)
        assert set(pooled) == set(ref), f"{agg}: 键集合变了"
        for sid, (s, e) in idx.items():
            assert pooled[sid].shape == ref[sid].shape, f"{agg}: 形状变了"
            assert np.array_equal(pooled[sid], fn(polyline[s:e + 1], axis=0)), f"{agg}: 值不对"
            assert not np.array_equal(pooled[sid], ref[sid]), f"{agg}: 与 mean 相同（未生效）"
    try:
        subgraph_pooling(polyline, idx, "median")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 agg 未 raise")
    print("  ✅ [B] max/sum 生效且形状同构；非法 agg raise")


def test_c_global_and_cache_naming() -> None:
    assert get_current_subgraph_agg() == "mean", "默认全局应为 mean"
    ds = GraphDataset.__new__(GraphDataset)   # 不触发 __init__（不加载数据）
    ds.adjacency_config = "default"
    try:
        ds.subgraph_agg = "mean"
        assert ds.processed_file_names == ["dataset_default.pt"], "mean 必须保旧缓存名"
        for agg in ("max", "sum"):
            ds.subgraph_agg = agg
            assert ds.processed_file_names == [f"dataset_default_agg{agg}.pt"], f"{agg} 缓存名未区分"
        set_subgraph_agg("max")
        assert get_current_subgraph_agg() == "max"
        try:
            set_subgraph_agg("median")
        except ValueError:
            pass
        else:
            raise AssertionError("set_subgraph_agg 非法值未 raise")
    finally:
        set_subgraph_agg("mean")
    print("  ✅ [C] 全局 set/get + 缓存名区分（mean 保旧名）")


def test_d_build_graph_passthrough() -> None:
    import pandas as pd

    assert REAL_PKL.exists(), f"{REAL_PKL} 不在盘 —— 真实样本门无法跑"
    df = pd.read_pickle(REAL_PKL)
    try:
        set_subgraph_agg("mean")
        data_mean, r1 = build_graph_data_from_feature_df(df, "default")
        set_subgraph_agg("max")
        data_max, r2 = build_graph_data_from_feature_df(df, "default")
    finally:
        set_subgraph_agg("mean")
    assert r1 is None and r2 is None, f"真实样本建图被 skip: {r1}/{r2}"
    gm, gx = data_mean.gnn_x, data_max.gnn_x
    assert gm.shape == gx.shape, f"gnn_x 形状不同: {tuple(gm.shape)} vs {tuple(gx.shape)}"
    assert not np.array_equal(gm.numpy(), gx.numpy()), "mean 与 max 的 gnn_x 逐位相同 —— 穿透失败"
    assert np.array_equal(data_mean.x.numpy(), data_max.x.numpy()), \
        "context 分支 data.x 不应随 agg 变（聚合只作用于 GNN 建图侧）"
    print("  ✅ [D] 真实样本穿透：gnn_x 随 agg 变、data.x 不变、形状同构")


if __name__ == "__main__":
    test_a_mean_bitwise_equivalent()
    test_b_max_sum_differ_and_invalid_raises()
    test_c_global_and_cache_naming()
    test_d_build_graph_passthrough()
    print("✅ 4/4 通过")
