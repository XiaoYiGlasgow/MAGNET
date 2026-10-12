"""R1 逐帧图干预原语的单测 —— 判据写在见到审计读数之前。

命门是 **[A] 空操作校验**：不删任何实体时 `response` 必须**恰好** 0.0。
它不是形式检查 —— 若重建路径把图改坏了（顺序、边、帧索引任一处），
所有 response 都会被那份偏移污染，而污染后的读数长得跟真效应一模一样。

跑法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans && $PY tests/test_temporal_intervention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN), str(_GNN.parent), str(_GNN.parent / "models"),
           str(_GNN / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.d2_semantic_intervention import response  # noqa: E402
from scripts.analysis.temporal_intervention import (  # noqa: E402
    TemporalInterventionError,
    attach_temporal_graph,
    entity_order_from_x,
    num_entities,
    remove_temporal_entities,
    verify_entity_alignment,
    zero_temporal_entities,
)
from src.shared.utils.dataset import (  # noqa: E402
    build_graph_data_from_feature_df,
    get_adjacency_config,
    subgraph_pooling,
)
from src.shared.utils.temporal_graph import NUM_FRAMES, build_frame_edges  # noqa: E402
from temporal_predictor import TemporalPredictor  # noqa: E402

RAW_DIR = _GNN / "data" / "processed_features_scaling_L1_logsplit" / "val"
N_SAMPLES = 6


def _load_samples(n: int = N_SAMPLES):
    """取 n 个真实样本（排序后取前 n 个 ⇒ 可复现）。"""
    get_adjacency_config("default")
    out = []
    for pkl in sorted(RAW_DIR.glob("features_*.pkl"))[:n]:
        df = pd.read_pickle(pkl)
        data, reason = build_graph_data_from_feature_df(df, "default")
        if data is None:
            continue
        pooled = subgraph_pooling(df["POLYLINE_FEATURES"].values[0].astype(np.float32),
                                  df["SUBGRAPH_ID_TO_NODE_INDICES"].values[0])
        out.append((pkl.stem.replace("features_", ""), df, data, list(pooled.keys())))
    assert out, f"{RAW_DIR} 下没读到样本"
    return out


def _model() -> TemporalPredictor:
    """随机初始化的 R1 模型 —— 本组测试只验干预原语，不需要训练好的权重。

    🔴 必须 `eval()`：`temporal_transformer` 带 dropout=0.1，train 模式下
    空操作校验会因为随机 mask 而假失败（那会被误读成「重建路径坏了」）。
    """
    torch.manual_seed(20260804)
    m = TemporalPredictor()
    m.eval()
    return m


def test_a_noop_is_exactly_zero() -> None:
    """[A] 🔴 空操作：drop 为空时，两个算子的 response 都必须**恰好** 0.0。"""
    model, dev = _model(), torch.device("cpu")
    worst = 0.0
    for topo in ("full", "knn30"):
        for token, _df, data, _order in _load_samples():
            attach_temporal_graph(data, topo)
            n = num_entities(data.tg_x)
            empty = np.zeros(0, dtype=np.int64)

            zeroed = zero_temporal_entities(data.tg_x, empty, n)
            assert torch.equal(zeroed, data.tg_x), f"{token}/{topo}: zero 空操作改动了 tg_x"

            tg_x, tg_ei, tg_fr = remove_temporal_entities(data.tg_x, empty, n, topology=topo)
            assert torch.equal(tg_x, data.tg_x), f"{token}/{topo}: remove 空操作改动了 tg_x"
            assert torch.equal(tg_fr, data.tg_frame), f"{token}/{topo}: remove 空操作改动了 tg_frame"
            assert torch.equal(tg_ei, data.tg_edge_index), f"{token}/{topo}: 空操作改动了边"

            interv = data.clone()
            interv.tg_x, interv.tg_edge_index, interv.tg_frame = tg_x, tg_ei, tg_fr
            disp = abs(response(model, data, interv, dev)["mean_disp"])
            assert disp == 0.0, f"{token}/{topo}: 空操作 response={disp:.3e} ≠ 0"
            worst = max(worst, disp)
    print(f"  [A] 空操作 max|mean_disp| = {worst:.3e}（判据 == 0.0）")


def test_b_remove_drops_all_frames_of_the_entity() -> None:
    """[B] 删 1 个实体 = 删掉它在全部 T 帧的行；幸存实体的特征逐位不变。"""
    for topo in ("full", "knn30"):
        for token, _df, data, _order in _load_samples(3):
            attach_temporal_graph(data, topo)
            n = num_entities(data.tg_x)
            assert n >= 3, f"{token}: 实体数 {n} 太少，测不了"
            drop = np.array([1], dtype=np.int64)     # index 0 是 ego，删第一个非 ego 实体

            tg_x, _ei, tg_fr = remove_temporal_entities(data.tg_x, drop, n, topology=topo)
            assert tg_x.shape[0] == NUM_FRAMES * (n - 1), f"{token}: 行数 {tg_x.shape[0]}"
            assert torch.equal(tg_fr,
                               torch.arange(NUM_FRAMES).repeat_interleave(n - 1)), "帧索引错了"

            before = data.tg_x.view(NUM_FRAMES, n, -1)
            after = tg_x.view(NUM_FRAMES, n - 1, -1)
            keep = torch.as_tensor(np.setdiff1d(np.arange(n), drop))
            assert torch.equal(after, before[:, keep, :]), f"{token}: 幸存实体的特征被改动"
    print("  [B] remove 逐帧一致、幸存实体逐位不变 ✅")


def test_c_zero_only_touches_target_rows() -> None:
    """[C] zero 只把目标实体的 T 行置零，其余行逐位不变，图结构完全不动。"""
    for token, _df, data, _order in _load_samples(3):
        attach_temporal_graph(data, "full")
        n = num_entities(data.tg_x)
        drop = np.array([1, 2], dtype=np.int64)
        out = zero_temporal_entities(data.tg_x, drop, n)

        view_in, view_out = data.tg_x.view(NUM_FRAMES, n, -1), out.view(NUM_FRAMES, n, -1)
        assert float(view_out[:, drop, :].abs().max()) == 0.0, f"{token}: 目标行没被置零"
        keep = torch.as_tensor(np.setdiff1d(np.arange(n), drop))
        assert torch.equal(view_out[:, keep, :], view_in[:, keep, :]), f"{token}: 误伤了其它实体"
    print("  [C] zero 只动目标行 ✅")


def test_d_rebuild_equals_filter() -> None:
    """[D] 「按幸存节点重算边」与「过滤掉关联边再重编号」逐位等价（两种拓扑各验）。

    这条是 `remove_temporal_entities` 选择「重算」而不是「过滤」的依据 ——
    两者不等价的话，重算得到的就不是「该实体从未存在过」那张反事实图。
    """
    for topo in ("full", "knn30"):
        for token, _df, data, _order in _load_samples(3):
            attach_temporal_graph(data, topo)
            n = num_entities(data.tg_x)
            drop = np.array([1], dtype=np.int64)
            keep = np.setdiff1d(np.arange(n), drop)
            remap = -np.ones(n, dtype=np.int64)
            remap[keep] = np.arange(keep.size)

            _x, rebuilt, _f = remove_temporal_entities(data.tg_x, drop, n, topology=topo)

            # 过滤参考实现：逐帧取原边、去掉碰到被删节点的边、重编号、再按帧偏移
            frames = data.tg_x.view(NUM_FRAMES, n, -1)
            src_edges = build_frame_edges(frames.numpy(), topology=topo)
            shifted = []
            for t, e in enumerate(src_edges):
                if e.size == 0:
                    continue
                mask = np.isin(e[0], keep) & np.isin(e[1], keep)
                if not mask.any():
                    continue
                shifted.append(np.stack([remap[e[0][mask]], remap[e[1][mask]]]) + t * keep.size)
            filtered = (torch.from_numpy(np.concatenate(shifted, axis=1))
                        if shifted else torch.zeros(2, 0, dtype=torch.long))

            assert torch.equal(rebuilt, filtered), \
                f"{token}/{topo}: 重算 {tuple(rebuilt.shape)} vs 过滤 {tuple(filtered.shape)}"
    print("  [D] 重算 ≡ 过滤（full / knn30 均逐位相同）✅")


def test_e_entity_alignment_matches_gnn_order() -> None:
    """[E] 🔴 tg 的实体顺序与 `gnn_x` 的行顺序逐位相同 —— 干预下标才对得上。"""
    for token, _df, data, order in _load_samples():
        verify_entity_alignment(data.x, order)          # 不抛异常即通过
        assert entity_order_from_x(data.x) == list(order), token
        attach_temporal_graph(data, "full")
        assert num_entities(data.tg_x) == data.gnn_x.shape[0], \
            f"{token}: tg 实体数 {num_entities(data.tg_x)} ≠ gnn_x 行数 {data.gnn_x.shape[0]}"
    print(f"  [E] {N_SAMPLES} 个样本的实体顺序与 gnn_x 逐位一致 ✅")


def test_f_alignment_error_is_raised_not_guessed() -> None:
    """[F] 顺序对不上时必须抛错，**不得**猜对应关系（错位不报错才是最坏情况）。"""
    _token, _df, data, order = _load_samples(1)[0]
    bad = list(order)
    bad[0], bad[1] = bad[1], bad[0]
    try:
        verify_entity_alignment(data.x, bad)
    except TemporalInterventionError:
        print("  [F] 顺序错位被拦下 ✅")
        return
    raise AssertionError("顺序错位没有被拦下 —— 干预会静默打到别的实体上")


def test_g_intervention_actually_changes_output() -> None:
    """[G] 反向判据：真删掉 agent 时 response 必须 > 0。

    没有这一条，[A] 可以被一个「什么都不做」的实现骗过去（第五十一/八十九条：
    「关闭时逐位不变」不证明「打开时能跑」）。
    """
    model, dev = _model(), torch.device("cpu")
    n_changed = 0
    for token, _df, data, _order in _load_samples():
        attach_temporal_graph(data, "full")
        n = num_entities(data.tg_x)
        drop = np.array([1], dtype=np.int64)
        tg_x, tg_ei, tg_fr = remove_temporal_entities(data.tg_x, drop, n, topology="full")
        interv = data.clone()
        interv.tg_x, interv.tg_edge_index, interv.tg_frame = tg_x, tg_ei, tg_fr
        if abs(response(model, data, interv, dev)["mean_disp"]) > 0:
            n_changed += 1
    assert n_changed == N_SAMPLES, f"只有 {n_changed}/{N_SAMPLES} 个样本的输出被干预改变"
    print(f"  [G] 真删实体时 {n_changed}/{N_SAMPLES} 个样本 response > 0 ✅")


def test_h_out_of_range_drop_raises() -> None:
    """[H] 🔴 越界实体下标必须抛错 —— 两个算子的静默后果相反，且都错。

    `zero` 侧 `t*N + k` 会折到**下一帧的另一个实体**上（n=43、k=43 ⇒ 第 1 帧的 ego 被置零）；
    `remove` 侧 `np.setdiff1d` 直接忽略越界值 ⇒ **什么都没删**，response 却被当成真读数。
    两者都不报错，故必须有显式边界校验。
    """
    _token, _df, data, _order = _load_samples(1)[0]
    attach_temporal_graph(data, "full")
    n = num_entities(data.tg_x)

    for bad in (np.array([n], dtype=np.int64), np.array([-1], dtype=np.int64)):
        for fn, kw in ((zero_temporal_entities, {}),
                       (remove_temporal_entities, {"topology": "full"})):
            try:
                fn(data.tg_x, bad, n, **kw)
            except TemporalInterventionError:
                continue
            raise AssertionError(f"{fn.__name__} 收下了越界下标 {bad.tolist()}（n={n}）")

    # 边界内必须照常工作（防止校验写反把合法值也拦了）
    ok = np.array([0, n - 1], dtype=np.int64)
    zero_temporal_entities(data.tg_x, ok, n)
    remove_temporal_entities(data.tg_x, ok, n, topology="full")
    print(f"  [H] 越界 {{{n}, -1}} 两算子均抛错；边界内 {{0, {n - 1}}} 照常工作 ✅")


def main() -> None:
    tests = [
        ("A 空操作恒等（🔴 命门）", test_a_noop_is_exactly_zero),
        ("B remove 展开到全部 T 帧", test_b_remove_drops_all_frames_of_the_entity),
        ("C zero 只动目标行", test_c_zero_only_touches_target_rows),
        ("D 重算 ≡ 过滤", test_d_rebuild_equals_filter),
        ("E 实体顺序与 gnn_x 对齐", test_e_entity_alignment_matches_gnn_order),
        ("F 错位必须抛错", test_f_alignment_error_is_raised_not_guessed),
        ("G 干预确实改变输出（反向判据）", test_g_intervention_actually_changes_output),
        ("H 越界下标必须抛错（🔴 静默类防线）", test_h_out_of_range_drop_raises),
    ]
    failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001 单测入口要报全部失败
            failed += 1
            print(f"  ❌ {type(exc).__name__}: {exc}")
    print(f"\n{'✅' if failed == 0 else '❌'} {len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
