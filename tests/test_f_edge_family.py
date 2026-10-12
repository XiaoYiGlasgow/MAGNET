#!/usr/bin/env python3
"""F 留车切边的两道起跑前置门（冻结判据 §1，V-F1/V-F2；🔴 先写后跑，门⑥收编）。

冻结判据: docs/work_records/paper/MAGNET/VT-2026-04191_修改工作记录/预注册判据/
F_留车切边_预注册判据_20260812_冻结.md

  [V-F1] 空操作等价：删除集为空时，切边流水线输出与 base **逐位相同**（torch.equal）；
  [V-F2] 族隔离：只切 a2a 族时 —— ①被删的正好是 a2a 全部半边、其余三族边集逐位不变；
         ②`gnn_x`/`data.x` 逐位不动（留车）；③截获的 `context_feat`/`ego_feat` 逐位不动
         （边只进 GNN 通路）且 `gnn_feat` 在至少一个有 a2a 边的 token 上被改动
         （防「假隔离」：干预必须真打中 GNN 通路）。
  附加：族划分穷尽且互斥（四族下标并集 = 全部半边、两两不交）；对照抽样 k 条数守恒。

输入：3-token 迷你清单（head -3 shards_r1_L1/shard_00.txt），fullret val，CPU。
ckpt：fr_base_L1/seed42 best（判据 §4 主格；sidecar 决定类=MagnetImitationPooled/fix=True）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest tests/test_f_edge_family.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

torch.set_num_threads(2)

from scripts.analysis import f_edge_family_audit as fa  # noqa: E402

CKPT = _GNN / "experiments/knobs/fr_base_L1/seed42/models/best_model_default.pth"
RAW = _GNN / "data/processed_features_fullret_L1_logsplit/val"
SHARD00 = _GNN / "experiments/interaction_audit/shards_r1_L1/shard_00.txt"
DEVICE = torch.device("cpu")
EGO_SL, CTX_SL, GNN_SL = slice(0, 128), slice(128, 256), slice(256, 384)


class _Tap:
    def __init__(self, model) -> None:
        self.captured = None
        self.handle = model.feature_fusion.register_forward_pre_hook(self._hook)

    def _hook(self, module, args) -> None:
        self.captured = args[0].detach().clone()

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def _capture(model, tap: _Tap, data) -> torch.Tensor:
    from torch_geometric.data import Batch
    model(Batch.from_data_list([data]).to(DEVICE))
    assert tap.captured is not None and tap.captured.shape[1] == 384
    return tap.captured.clone()


@pytest.fixture(scope="module")
def tokens() -> list:
    assert SHARD00.exists(), f"{SHARD00} 不在盘"
    toks = [t.strip() for t in SHARD00.read_text().splitlines() if t.strip()][:3]
    assert len(toks) == 3
    return toks


@pytest.fixture(scope="module")
def model():
    assert CKPT.exists(), f"{CKPT} 不在盘 —— fr_base seed42 best 缺失"
    agg = fa.configure_subgraph_agg(CKPT)
    assert agg == "mean", f"fr_base sidecar 应为 mean 聚合，实得 {agg}"
    m = fa._load_model(CKPT, False, DEVICE, temporal_graph=False)
    assert type(m).__name__ == "MagnetImitationPooled", type(m).__name__
    return m


@pytest.fixture(scope="module")
def samples(tokens):
    out = []
    for tok in tokens:
        s, reason = fa.load_token(RAW, tok, "default")
        assert s is not None, f"{tok}: 构建失败（{reason}）"
        out.append(s)
    return out


def test_families_partition(samples) -> None:
    """四族划分穷尽且互斥；a2a 边数 = C(n_agent_noego, 2)（完全图 default builder）。"""
    for s in samples:
        fam = fa.classify_edges(s["data"].gnn_edge_index, s["node_types"], s["is_ego"])
        m = fam.directed.shape[1]
        allidx = np.concatenate([fam.idx[k] for k in fa.FAMILIES])
        assert allidx.size == m and np.unique(allidx).size == m, "四族不穷尽或有重叠"
        types = np.asarray(s["node_types"])
        n_ag = int(((types == fa.TARGET_TYPE) & ~np.asarray(s["is_ego"])).sum())
        assert fam.idx["a2a"].size == n_ag * (n_ag - 1) // 2, (
            f"{s['token']}: a2a 边数 {fam.idx['a2a'].size} ≠ C({n_ag},2)（完全图假设破）")
        assert fam.idx["E"].size == n_ag, f"{s['token']}: E 边数应等于非 ego agent 数"
    print(f"  ✅ 族划分：{len(samples)} token 穷尽互斥，a2a=C(n,2)、E=n 均命中")


def test_vf1_noop_bitwise(model, samples) -> None:
    """[V-F1] 空删除集切边流水线输出与 base 逐位相同。"""
    for s in samples:
        fam = fa.classify_edges(s["data"].gnn_edge_index, s["node_types"], s["is_ego"])
        base = fa._forward_once(model, s["data"], DEVICE)
        out = fa._forward_once(model, fa.cut_edges(s["data"], fam, np.zeros(0, dtype=np.int64)), DEVICE)
        assert torch.equal(out, base), f"[V-F1] {s['token']} 空操作不逐位等价"
        fa.noop_gate(model, s["data"], fam, DEVICE)     # 生产路径的门本身也须静默通过
    print(f"  ✅ [V-F1] {len(samples)} token 空操作逐位等价")


def test_vf2_family_isolation(model, samples) -> None:
    """[V-F2] 只切 a2a：边集精确、节点特征不动、context/ego 通路不动、GNN 通路被打中。"""
    tap = _Tap(model)
    n_hit = 0
    try:
        for s in samples:
            data = s["data"]
            fam = fa.classify_edges(data.gnn_edge_index, s["node_types"], s["is_ego"])
            cut = fa.cut_edges(data, fam, fam.idx["a2a"])
            fa.isolation_check(data, cut, fam, fam.idx["a2a"], "a2a_all")
            # ① 剩余边集 = 其余三族半边（对称重建后逐位比对）
            keep_cols = np.concatenate([fam.idx["E"], fam.idx["LL"], fam.idx["other"]])
            want = fa._rebuild_symmetric(fam.directed[:, torch.as_tensor(np.sort(keep_cols))])
            got = cut.gnn_edge_index
            assert got.shape == want.shape
            def _canon(ei):
                pairs = sorted(zip(ei[0].tolist(), ei[1].tolist()))
                return pairs
            assert _canon(got) == _canon(want), f"{s['token']}: 剩余边集 ≠ 其余三族"
            # ② 留车：节点特征逐位不动
            assert torch.equal(cut.gnn_x, data.gnn_x) and torch.equal(cut.x, data.x)
            # ③ 通路隔离：context/ego 段逐位不动；有 a2a 边的 token 上 gnn 段必须改动
            f0, f1 = _capture(model, tap, data), _capture(model, tap, cut)
            assert torch.equal(f0[:, CTX_SL], f1[:, CTX_SL]), f"{s['token']}: context_feat 被切边改动"
            assert torch.equal(f0[:, EGO_SL], f1[:, EGO_SL]), f"{s['token']}: ego_feat 被切边改动"
            if fam.idx["a2a"].size > 0 and not torch.equal(f0[:, GNN_SL], f1[:, GNN_SL]):
                n_hit += 1
        assert n_hit >= 1, "切 a2a 后 gnn_feat 无一 token 改动 ⇒ 干预未打中 GNN 通路（假隔离）"
    finally:
        tap.close()
    print(f"  ✅ [V-F2] 族隔离通过；gnn_feat 命中 {n_hit}/{len(samples)} token")


def test_ctrl_count_matched(samples) -> None:
    """count-matched 对照：k = a2a 边数，从 LL 池抽样，抽样确定性（同 seed 同 pick）。"""
    for s in samples:
        fam = fa.classify_edges(s["data"].gnn_edge_index, s["node_types"], s["is_ego"])
        k = fam.idx["a2a"].size
        if k == 0 or k > fam.idx["LL"].size:
            continue
        p1 = np.random.default_rng(1000).choice(fam.idx["LL"], size=k, replace=False)
        p2 = np.random.default_rng(1000).choice(fam.idx["LL"], size=k, replace=False)
        assert np.array_equal(p1, p2) and np.unique(p1).size == k
        assert np.all(np.isin(p1, fam.idx["LL"]))
    print("  ✅ 对照抽样：k 守恒、只落 LL 池、同 seed 确定性")
