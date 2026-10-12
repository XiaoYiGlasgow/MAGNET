"""S4 接线单测 —— `EdgeScorer.raw_logit` 与逐目标质量守恒归一 `conserve_target_mass`。

🔴 门的语义（六条，出数后不得改）：

  [N1] **质量守恒**  归一后每个目标的 scorable∧valid 权重和 == `d_i`（该目标的 scorable∧valid
       边数）。这是「可比的激活边预算」（Q109 §6.1）的落地：锚从固定臂量出来，不预设稀疏目标。
  [N2] **固定臂逐位不变**  `learn_topology=False` 走 `w=ones` 分支，压根不经过 scorer
       ⇒ E1 九格与 `joint_s2_fixed` 的读数**不受本次改动影响**。这条是回归保险。
  [N3] **无信息退化**  组内 `w` 全等时归一结果**逐位 1.0** —— 即恰好复现固定臂。
       这是守恒归一相对逐目标 min-max 的核心优势：87.6% 的 static/truncated 目标上
       (`U` p95 0.0077 ≈ teacher 分歧 0.00228) 噪声不会被展开成满量程。
  [N4] **作用域**  lane 参与的边恒 1.0（P6/D-b）；padding 边不分质量（改它不动 scorable 边）。
  [N5] **两条路径一致**  `forward == where(scorable, conserve(sigmoid(raw_logit)), 1)` 逐位。
  [N6] **梯度活着**  RankNet 损失对 `raw_logit` 有非零梯度，且能回传到 scorer 参数。
"""
from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN / "scripts" / "training"), str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_joint_encoder import EntityEncoder, build_entity_index          # noqa: E402
from magnet_joint_layers import (                                            # noqa: E402
    EdgeScorer, JointInteractionStack, _GeometryEncoder, build_edge_context,
    build_entity_geometry, conserve_target_mass)
from src.shared.utils.dataset import build_graph_data_from_feature_df        # noqa: E402

VAL_ROOT = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "val"
torch.manual_seed(20260826)

_PASS, _FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            _PASS.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            _FAIL.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    return deco


def _load(n: int = 4, seed: int = 42):
    fs = sorted(glob.glob(os.path.join(str(VAL_ROOT), "**", "features_*.pkl"), recursive=True))
    random.Random(seed).shuffle(fs)
    out = []
    for f in fs:
        df = pd.read_pickle(f)
        built = build_graph_data_from_feature_df(df, "default")
        d = built[0] if isinstance(built, tuple) else built
        if d is None:
            continue
        out.append(d)
        if len(out) == n:
            break
    return out


def _ctx(data, seed: int = 7):
    """编码器 → 几何 → 边上下文 → `h0` / `g_enc` / scorer。"""
    torch.manual_seed(seed)
    enc = EntityEncoder(dim=128).eval()
    scorer = EdgeScorer(dim=128).eval()
    geo = _GeometryEncoder(32).eval()
    idx = build_entity_index(data)
    geom = build_entity_geometry(data.x, idx)
    ectx = build_edge_context(data.gnn_edge_index, idx, geom)
    with torch.no_grad():
        h0 = enc(data.x, idx)
        g_enc = geo(ectx.g)
    return idx, ectx, h0, g_enc, scorer


SAMPLES = _load(4)
assert SAMPLES, f"{VAL_ROOT} 下没有可用样本"


@check("[N1] 质量守恒：stack 输出的逐目标 scorable 权重和 == d_i")
def _n1():
    worst = 0.0
    for d in SAMPLES:
        idx, ectx, h0, g_enc, _ = _ctx(d)
        torch.manual_seed(7)
        stack = JointInteractionStack(dim=128, learn_topology=True).eval()
        with torch.no_grad():
            w = stack(h0, ectx, idx).edge_weight
        m = (ectx.scorable & ectx.valid)
        dst = ectx.edge_index[1]
        n = idx.n_entity
        tot = torch.zeros(n).index_add(0, dst, w * m.to(w.dtype))
        cnt = torch.zeros(n).index_add(0, dst, m.to(w.dtype))
        hit = cnt > 0
        assert int(hit.sum()) > 0, "该样本没有任何 scorable 边 —— 换样本"
        worst = max(worst, float((tot[hit] - cnt[hit]).abs().max()))
    assert worst < 1e-4, f"权重和与 d_i 最大偏差 {worst:.3e}"
    print(f"        max|Σw − d_i| = {worst:.3e}")


@check("[N2] 固定臂逐位不变：learn_topology=False 仍是 w ≡ valid（不经过 scorer）")
def _n2():
    for d in SAMPLES:
        torch.manual_seed(7)
        stack = JointInteractionStack(dim=128, learn_topology=False).eval()
        enc = EntityEncoder(dim=128).eval()
        idx = build_entity_index(d)
        geom = build_entity_geometry(d.x, idx)
        ectx = build_edge_context(d.gnn_edge_index, idx, geom)
        with torch.no_grad():
            out = stack(enc(d.x, idx), ectx, idx)
        expect = ectx.valid.to(out.edge_weight.dtype)
        assert float((out.edge_weight - expect).abs().max()) == 0.0, "固定臂的 w 不再是 valid 的指示函数"


@check("[N3] 无信息退化：组内 w 全等 ⇒ 归一结果逐位 1.0（恰好复现固定臂）")
def _n3():
    for d in SAMPLES:
        idx, ectx, h0, g_enc, _ = _ctx(d)
        m = (ectx.scorable & ectx.valid)
        for c in (0.01, 0.5, 0.9999):
            w = torch.full((ectx.edge_index.shape[1],), float(c))
            out = conserve_target_mass(w, ectx, idx.n_entity)
            assert float((out[m] - 1.0).abs().max()) < 1e-5, \
                f"常数 w={c} 归一后不是 1.0（max|Δ| {float((out[m]-1.0).abs().max()):.3e}）"
            assert float((out[~m] - c).abs().max()) == 0.0, "非 scorable 边被动了"


@check("[N4] 作用域：lane 边恒 1.0；padding 边不分质量")
def _n4():
    for d in SAMPLES:
        idx, ectx, h0, g_enc, scorer = _ctx(d)
        with torch.no_grad():
            w = scorer(h0, ectx, g_enc)
        assert float((w[~ectx.scorable] - 1.0).abs().max()) == 0.0, "lane 参与的边不恒 1.0"

        # padding 边（valid=False）的原始权重改成任意值，scorable 边的结果必须逐位不变
        raw = torch.sigmoid(scorer.raw_logit(h0, ectx, g_enc)).detach()
        base = conserve_target_mass(raw, ectx, idx.n_entity)
        bad = raw.clone()
        pad = ~ectx.valid
        if int(pad.sum()) == 0:
            continue
        bad[pad] = 7.77
        got = conserve_target_mass(bad, ectx, idx.n_entity)
        m = (ectx.scorable & ectx.valid)
        assert float((got[m] - base[m]).abs().max()) == 0.0, "padding 边分走了质量"


@check("[N5] 两条路径一致：scorer.forward == where(scorable, sigmoid(raw_logit), 1)")
def _n5():
    # 🔴 `EdgeScorer.forward` 只出未归一的 σ；归一由 `JointInteractionStack` 在所有 mask
    #    之后做（见 N7）。这条门锁的是「raw_logit 与 forward 之间没有第二套变换」。
    for d in SAMPLES:
        idx, ectx, h0, g_enc, scorer = _ctx(d)
        with torch.no_grad():
            got = scorer(h0, ectx, g_enc)
            man = torch.sigmoid(scorer.raw_logit(h0, ectx, g_enc))
            man = torch.where(ectx.scorable, man, torch.ones_like(man))
        assert float((got - man).abs().max()) == 0.0, "forward 与手工重算不逐位相同"
        assert float(got[ectx.scorable].max()) < 1.0, "scorer 的原始输出应仍在 (0,1)"


@check("[N7] 归一在 mask 之后：被干预门切掉的边不占预算分母")
def _n7():
    """🔴 这条门是 2026-08-26 一个真 bug 的回归锁。

    第一版把归一放在 `EdgeScorer.forward` 里（即 `edge_gate` **之前**）⇒ 被切掉的边
    虽然 `w=0`，其质量仍留在分母中，造成两处失效：
      · V-2 实体隔离出现泄漏（`test_joint_adaptive` 实测 max|Δ| 5.334e-03）；
      · 「切边门 ≡ 重建 edge_index」不再成立（`test_joint_layers` 实测 max|Δ| 1.805e-02）。
    归一必须在 `valid` 与 `edge_gate` 之后，且 `active` 要排除被切的边。
    """
    for d in SAMPLES:
        idx, ectx, h0, g_enc, _ = _ctx(d)
        torch.manual_seed(7)
        stack = JointInteractionStack(dim=128, learn_topology=True).eval()
        scor = torch.nonzero(ectx.scorable & ectx.valid, as_tuple=True)[0]
        if scor.numel() < 8:
            continue
        gate = torch.ones(ectx.edge_index.shape[1])
        gate[scor[: scor.numel() // 3]] = 0.0            # 切掉三分之一的车↔车边
        with torch.no_grad():
            w = stack(h0, ectx, idx, edge_gate=gate).edge_weight
        assert float(w[gate == 0].abs().max()) == 0.0, "被切的边 w 不为 0"
        live = (ectx.scorable & ectx.valid) & (gate > 0)
        tot = torch.zeros(idx.n_entity).index_add(0, ectx.edge_index[1][live], w[live])
        cnt = torch.zeros(idx.n_entity).index_add(
            0, ectx.edge_index[1][live], torch.ones(int(live.sum())))
        hit = cnt > 0
        dmax = float((tot[hit] - cnt[hit]).abs().max())
        assert dmax < 1e-4, f"切边后逐目标权重和 ≠ 存活边数（max|Δ| {dmax:.3e}）—— 被切边仍占分母"


@check("[N6] 梯度活着：RankNet 损失能回传到 scorer 参数，且梯度非零")
def _n6():
    d = SAMPLES[0]
    idx, ectx, h0, g_enc, scorer = _ctx(d)
    scorer.train()
    s = scorer.raw_logit(h0, ectx, g_enc)
    m = torch.nonzero(ectx.scorable & ectx.valid, as_tuple=True)[0]
    assert m.numel() >= 4, "scorable 边太少，测不了"
    a, b = m[: m.numel() // 2], m[m.numel() // 2 : m.numel() // 2 * 2]
    loss = torch.nn.functional.softplus(-(s[a] - s[b])).mean()
    loss.backward()
    g = scorer.net[0].weight.grad
    assert g is not None and float(g.abs().max()) > 0, "scorer 第一层没有梯度"
    print(f"        loss={float(loss):.4f}  max|grad|={float(g.abs().max()):.3e}")


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for n, e in _FAIL:
        print(f"  FAIL {n}: {e}")
    sys.exit(1)
