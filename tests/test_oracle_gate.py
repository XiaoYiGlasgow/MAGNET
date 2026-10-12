#!/usr/bin/env python
"""E1 oracle 门接线的判据（2026-08-26 立）。

这条接线的危险性和 S3 标签同型：**对错了不会报错，只会训出一份没人能解释的产物**。
门是逐边的、边有几千条、标签是按 `(src, dst)` 存的而不是按边位置存的 —— 错位是静默的。

判据
----

[O1] **空操作门** —— `edge_gate = ones` 与 `edge_gate = None` 的前向在 CPU 上**逐位相同**。
     这一条单独就验证了整条透传链没有副作用（§8.2：GPU 前向不可复现，逐位判据必须 CPU）。
[O2] **作用域** —— 缓存里 `gate < 1` 的边**全部**落在 `ectx.scorable`（车↔车）上；
     lane 相关边与未打标边恒 `1.0`，与固定臂逐位一致。
[O3] **值正确** —— 独立重算逐目标 min-max，与缓存逐位相同；且 `gate ∈ [W_FLOOR, 1.0]`。
[O4] **批处理对齐** —— PyG 批处理后第 k 个样本的 `oracle_gate` 切片 == 单样本 gate
     （字段名不含 `index` ⇒ `cat_dim=0`，与 `gnn_edge_index` 沿 dim=1 拼接同序）。
[O5] **三档同位不同值** —— oracle / shuffled / random 的**被打标边集合完全相同**（只换值不换位置），
     但值不同；且 shuffled 是 oracle 的逐目标置换（多重集逐目标相同）。
[O6] **门确实进了模型** —— 真 oracle 门与全 1 门的前向**必须不同**（否则接线是死的）。

🔴 全程 CPU。

跑法::

    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    /home/team/anaconda3/envs/nuplan/bin/python tests/test_oracle_gate.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN), str(_GNN / "scripts" / "training"), str(_GNN / "scripts" / "analysis"),
           str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
LAB = _GNN / "data/edge_labels_s3_20260826/val"
CKPT = _GNN / "experiments/joint_s2_fixed/seed1/models/checkpoint_best.pt"
N_SAMPLE = 12                      # 逐位判据用小样本，够抓错位
_FAILED: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def _load(variant: str):
    import config  # noqa: F401
    from src.shared.utils.dataset import get_adjacency_config, set_subgraph_agg
    from magnet_oracle_gate_dataset import OracleGateGraphDataset
    get_adjacency_config("default")
    set_subgraph_agg("mean")
    return OracleGateGraphDataset(str(VAL), transform=None,
                                  adjacency_config="default", variant=variant)


def _model():
    from src.shared.utils.bn_calibration import load_model_for_inference
    return load_model_for_inference(CKPT, device=torch.device("cpu")).cpu().eval()


# ── [O1][O6] 空操作门 + 门确实起作用 ────────────────────────────────────────
def test_o1_o6_noop_and_effect() -> None:
    print("\n[O1][O6] 空操作门逐位恒等 + 真门必须改变输出（CPU）")
    from torch_geometric.data import Batch
    ds = _load("oracle")
    m = _model()
    max_noop, min_effect, n_eff = 0.0, float("inf"), 0
    for i in range(N_SAMPLE):
        d = ds[i]
        b = Batch.from_data_list([d], follow_batch=["gnn_x"])
        e = int(b.gnn_edge_index.shape[1])
        with torch.no_grad():
            p_none = m(b).traj
            p_ones = m(b, edge_gate=torch.ones(e)).traj
            p_gate = m(b, edge_gate=b.oracle_gate).traj
        max_noop = max(max_noop, float((p_none - p_ones).abs().max()))
        eff = float((p_none - p_gate).abs().max())
        if float(b.oracle_gate.min()) < 1.0:      # 该样本确实有非满权的边
            min_effect = min(min_effect, eff); n_eff += 1
    check(max_noop == 0.0, f"[O1] ones 门与无门逐位相同（{N_SAMPLE} 样本 max|Δ| = {max_noop:.1e}）")
    check(n_eff > 0, f"[O6] {N_SAMPLE} 个样本里有 {n_eff} 个带非满权边（否则测不出效应）")
    if n_eff:
        check(min_effect > 1e-4,
              f"[O6] 真 oracle 门**改变了**输出（最小 max|Δ| = {min_effect:.3e} m）—— 接线不是死的")


# ── [O2][O3] 作用域与值 ────────────────────────────────────────────────────
def test_o2_o3_scope_and_values() -> None:
    print("\n[O2][O3] 门只落在 scorable 车↔车边；值 == 独立重算的逐目标 min-max")
    from torch_geometric.data import Batch
    from magnet_joint_encoder import build_entity_index
    from magnet_joint_layers import build_edge_context, build_entity_geometry
    from build_oracle_gate_cache import W_FLOOR, _oracle_w, _match_edges
    from joint_usage_probe import resolve_tokens
    import json

    ds = _load("oracle")
    toks = resolve_tokens(str(VAL), len(ds))
    # 读回 S3 标签
    lab = {}
    for f in sorted(LAB.glob("labels_*.npz")):
        with np.load(f) as z:
            ptr = np.asarray(z["scene_ptr"])
            for k, tk in enumerate(str(t) for t in z["token"]):
                sl = slice(int(ptr[k]), int(ptr[k + 1]))
                lab[tk] = (np.asarray(z["src"])[sl], np.asarray(z["dst"])[sl],
                           np.asarray(z["U"])[sl])

    bad_scope = bad_val = bad_one = 0
    n_checked = 0
    rng_lo, rng_hi = 1.0, 0.0
    for i in range(N_SAMPLE):
        d = ds[i]
        g = d.oracle_gate.numpy()
        b = Batch.from_data_list([d], follow_batch=["gnn_x"])
        idx = build_entity_index(b)
        ectx = build_edge_context(b.gnn_edge_index, idx, build_entity_geometry(b.x, idx))
        scor = ectx.scorable.numpy()
        sub = g < 1.0
        bad_scope += int((sub & ~scor).sum())              # 非 scorable 边被改过 ⇒ 越界
        if toks[i] not in lab:
            bad_one += int((g != 1.0).sum())               # 无标签样本必须全 1
            continue
        src, dst, U = lab[toks[i]]
        w_ref = _oracle_w(src, dst, U)
        cols, rows = _match_edges(b.gnn_edge_index.numpy(), src, dst)
        bad_val += int((g[cols] != w_ref[rows]).sum())
        mask = np.ones(g.size, dtype=bool); mask[cols] = False
        bad_one += int((g[mask] != 1.0).sum())             # 未命中的边必须恰为 1.0
        if cols.size:
            rng_lo = min(rng_lo, float(g[cols].min())); rng_hi = max(rng_hi, float(g[cols].max()))
        n_checked += 1
    check(bad_scope == 0, f"[O2] 没有任何非 scorable 边被改动（越界 {bad_scope} 条）")
    check(bad_one == 0, f"[O2] 未打标的边恒为 1.0（异常 {bad_one} 条）")
    check(bad_val == 0, f"[O3] 命中边的值 == 独立重算（不符 {bad_val} 条，覆盖 {n_checked} 个样本）")
    check(rng_lo >= W_FLOOR - 1e-6 and rng_hi <= 1.0 + 1e-6,
          f"[O3] 值域落在 [{W_FLOOR}, 1.0]（实测 [{rng_lo:.4f}, {rng_hi:.4f}]）")


# ── [O4] 批处理对齐 ────────────────────────────────────────────────────────
def test_o4_batching() -> None:
    print("\n[O4] PyG 批处理后 oracle_gate 与 gnn_edge_index 仍逐位对齐")
    from torch_geometric.data import Batch
    ds = _load("oracle")
    items = [ds[i] for i in range(8)]
    b = Batch.from_data_list(items, follow_batch=["gnn_x"])
    off, bad = 0, 0
    for d in items:
        e = int(d.gnn_edge_index.shape[1])
        if not torch.equal(b.oracle_gate[off:off + e], d.oracle_gate):
            bad += 1
        off += e
    check(off == int(b.gnn_edge_index.shape[1]),
          f"[O4] 批内边总数一致（累计 {off} == 批 {int(b.gnn_edge_index.shape[1])}）")
    check(bad == 0, f"[O4] 8 个样本的 gate 切片逐位对齐（错位 {bad} 个）")


# ── [O5] 三档同位不同值 ────────────────────────────────────────────────────
def test_o5_variants() -> None:
    print("\n[O5] oracle / shuffled / random：改动的边**位置相同**、值不同；shuffled 是逐目标置换")
    from torch_geometric.data import Batch
    from build_oracle_gate_cache import _match_edges
    from joint_usage_probe import resolve_tokens
    # 🔴 「被改动的边」必须用**真实匹配位置**判定，不能用 `gate < 1.0` 近似 ——
    #    每个目标里 `w` 最大的那条边恰为 1.0，置换会把这个 1.0 挪到别的边上，
    #    于是 `<1.0` 的掩码在三档之间本来就不同（首版判据据此误报，2026-08-26）。
    dso, dss, dsr = _load("oracle"), _load("shuffled"), _load("random")
    toks = resolve_tokens(str(VAL), len(dso))
    lab = {}
    for fp in sorted(LAB.glob("labels_*.npz")):
        with np.load(fp) as z:
            ptr = np.asarray(z["scene_ptr"])
            for k, tk in enumerate(str(t) for t in z["token"]):
                sl = slice(int(ptr[k]), int(ptr[k + 1]))
                lab[tk] = (np.asarray(z["src"])[sl], np.asarray(z["dst"])[sl])
    n = diff_os = diff_or = ms_global = ms_target = 0
    e_tot = e_diff_s = e_diff_r = 0          # 边级计数（逐样本全有全无太脆）
    deg_of_identity = []
    for i in range(N_SAMPLE):
        if toks[i] not in lab:
            continue
        src, dst = lab[toks[i]]
        d = dso[i]
        b = Batch.from_data_list([d], follow_batch=["gnn_x"])
        cols, rows = _match_edges(b.gnn_edge_index.numpy(), src, dst)
        if cols.size == 0:
            continue
        n += 1
        go = d.oracle_gate.numpy()[cols]
        gs = dss[i].oracle_gate.numpy()[cols]
        gr = dsr[i].oracle_gate.numpy()[cols]
        dd0 = dst[rows]
        e_tot += go.size
        e_diff_s += int((go != gs).sum())
        e_diff_r += int((go != gr).sum())
        if not np.array_equal(go, gs):
            diff_os += 1
        else:
            # 🔴 恒等**不是缺陷**：目标内只有 1 条边时置换必然恒等，2 条时有 50% 概率恒等。
            #    记下该样本的目标度数分布，用它证明恒等有合法解释。
            deg_of_identity.append(int(np.bincount(np.unique(dd0, return_inverse=True)[1]).max()))
        if not np.array_equal(go, gr):
            diff_or += 1
        if Counter(np.round(go, 6)) == Counter(np.round(gs, 6)):
            ms_global += 1
        # 逐目标多重集（比全局多重集更强：置换只能在目标内部发生）
        dd = dst[rows]
        ok = True
        for t in np.unique(dd):
            m = dd == t
            if Counter(np.round(go[m], 6)) != Counter(np.round(gs[m], 6)):
                ok = False
                break
        ms_target += int(ok)
    check(n > 0, f"[O5] 有 {n} 个样本可比对")
    # 边级判据（逐样本恒等在低度数目标上是合法的，见上）
    check(e_tot > 0 and e_diff_s / e_tot > 0.5,
          f"[O5] shuffled 在**边级**上大幅不同于 oracle（{e_diff_s}/{e_tot} = "
          f"{100 * e_diff_s / max(e_tot, 1):.1f}% 的边被换了值）")
    check(e_tot > 0 and e_diff_r / e_tot > 0.5,
          f"[O5] random 在**边级**上大幅不同于 oracle（{e_diff_r}/{e_tot} = "
          f"{100 * e_diff_r / max(e_tot, 1):.1f}%）")
    check(all(d <= 2 for d in deg_of_identity),
          f"[O5] 逐样本恒等的 {n - diff_os} 个样本，其目标最大度数 {deg_of_identity} "
          f"—— 度数 ≤2 时置换恒等是数学必然，不是缺陷")
    check(ms_global == n, f"[O5] shuffled 的**全局**多重集与 oracle 相同（{ms_global}/{n}）")
    check(ms_target == n,
          f"[O5] shuffled 的**逐目标**多重集与 oracle 相同 ⇒ 置换只在目标内部（{ms_target}/{n}）")


if __name__ == "__main__":
    for f in (VAL, LAB, CKPT):
        if not f.exists():
            print(f"🔴 缺 {f}")
            sys.exit(1)
    torch.manual_seed(0)
    test_o1_o6_noop_and_effect()
    test_o2_o3_scope_and_values()
    test_o4_batching()
    test_o5_variants()
    if _FAILED:
        print(f"\n🔴 {len(_FAILED)} 项失败:")
        for f in _FAILED:
            print("   -", f)
        sys.exit(1)
    print("\n✅ test_oracle_gate 4/4 通过")
