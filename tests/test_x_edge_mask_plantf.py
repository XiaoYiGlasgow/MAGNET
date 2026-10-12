"""X —— PlanTF 边级干预原语的单测（判据 §1.4 的 V-X1 / V-X2 两道门）。

🔴 门的语义（见 `X_跨仪器边级与逐层响应_预注册判据_20260820`）：
  V-X1 空操作：干预集为空 ⇒ 与 vendor 原 `PlanningModel.forward` **逐位相同**；
  V-X2 族隔离：只屏蔽 a2a ⇒ **单层**时 ego 输出逐位不变（a2a 只改 agent 自身表示，
       需 ≥2 层堆叠才传导到读出端 —— 这条同时是「干预确实打在边上」的正面证据）。
两门任一不过 ⇒ 判据 §1.4 规定当场停，不产出任何读数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for p in (str(GNN), str(GNN / "scripts" / "analysis"), "/mnt/local/Xiao_Yi/external/planTF"):
    if p not in sys.path:
        sys.path.insert(0, p)

from x_edge_mask_plantf import (  # noqa: E402
    build_layout, make_attn_mask, forward_with_edge_mask, _n_total,
)
from src.models.planTF.layers.transformer_encoder_layer import TransformerEncoderLayer  # noqa: E402

DIM, HEADS, A, M = 64, 4, 6, 8


def _blocks(depth: int) -> torch.nn.ModuleList:
    torch.manual_seed(20260820)
    return torch.nn.ModuleList(
        TransformerEncoderLayer(dim=DIM, num_heads=HEADS, drop_path=0.0)
        for _ in range(depth)).eval()


def _x_kpm(n_pad_agent: int = 1, n_pad_map: int = 2):
    torch.manual_seed(7)
    x = torch.randn(1, A + M, DIM)
    a_pad = torch.zeros(A, dtype=torch.bool); a_pad[A - n_pad_agent:] = True
    m_pad = torch.zeros(M, dtype=torch.bool); m_pad[M - n_pad_map:] = True
    return x, a_pad, m_pad


# ── layout ──────────────────────────────────────────────────────────────────
def test_layout_pair_counts_match_combinatorics():
    """三族的无向对数必须等于组合数（有效 token 上）。"""
    _, a_pad, m_pad = _x_kpm(n_pad_agent=1, n_pad_map=2)
    lay = build_layout(a_pad, m_pad)
    n_other = int((~a_pad).sum()) - 1          # 去掉 ego
    n_map = int((~m_pad).sum())
    assert lay.pairs["a2a"].shape[0] == n_other * (n_other - 1) // 2
    assert lay.pairs["E"].shape[0] == n_other
    assert lay.pairs["LL"].shape[0] == n_map * (n_map - 1) // 2


def test_layout_excludes_padding_tokens():
    """padding 槽不得出现在任何族里（否则等于在屏蔽不存在的车）。"""
    _, a_pad, m_pad = _x_kpm()
    lay = build_layout(a_pad, m_pad)
    pad_global = set(np.flatnonzero(a_pad.numpy()).tolist()) | \
                 set((np.flatnonzero(m_pad.numpy()) + A).tolist())
    for fam, pr in lay.pairs.items():
        assert not (set(pr.flatten().tolist()) & pad_global), f"{fam} 含 padding token"


def test_a2a_never_touches_ego():
    """a2a 族一律不含 ego（index 0）—— 与 MAGNET 侧 E 族严格分离。"""
    _, a_pad, m_pad = _x_kpm()
    lay = build_layout(a_pad, m_pad)
    assert 0 not in set(lay.pairs["a2a"].flatten().tolist())
    assert all(p[0] == 0 for p in lay.pairs["E"])


# ── mask ────────────────────────────────────────────────────────────────────
def test_mask_is_symmetric_and_diagonal_free():
    _, a_pad, m_pad = _x_kpm()
    lay = build_layout(a_pad, m_pad)
    m = make_attn_mask(lay.n_total, lay.pairs["a2a"], torch.device("cpu"))
    assert torch.equal(m, m.T), "屏蔽必须对称（一个无向对 = 两个有向位）"
    assert not m.diagonal().any(), "对角线必须保留 self-attention"
    assert int(m.sum()) == 2 * lay.pairs["a2a"].shape[0]


def test_empty_pairs_return_none():
    """空干预集返回 None —— 比全 False 更强，连 mask 分支都不进。"""
    assert make_attn_mask(10, np.zeros((0, 2), dtype=np.int64), torch.device("cpu")) is None


# ── V-X1 空操作门 ───────────────────────────────────────────────────────────
def test_vx1_noop_bitwise_identical_to_vendor_forward():
    """mask=None 必须与 vendor 原调用（只传 key_padding_mask）逐位相同。"""
    x, a_pad, m_pad = _x_kpm()
    kpm = torch.cat([a_pad, m_pad]).unsqueeze(0)
    for depth in (1, 2, 4):
        blocks = _blocks(depth)
        with torch.no_grad():
            ref = x.clone()
            for blk in blocks:
                ref = blk(ref, key_padding_mask=kpm)          # vendor 原调用形态
            got = x.clone()
            for blk in _blocks(depth):
                got = blk(got, mask=None, key_padding_mask=kpm)
        assert torch.equal(ref, got), f"depth={depth} 空操作门失败"


# ── V-X2 族隔离门 ───────────────────────────────────────────────────────────
def test_vx2_a2a_does_not_move_ego_at_depth_one():
    """单层时切 a2a ⇒ ego 输出逐位不变（族隔离的操作化）。"""
    x, a_pad, m_pad = _x_kpm()
    kpm = torch.cat([a_pad, m_pad]).unsqueeze(0)
    lay = build_layout(a_pad, m_pad)
    m = make_attn_mask(lay.n_total, lay.pairs["a2a"], torch.device("cpu"))
    blocks = _blocks(1)
    with torch.no_grad():
        base = blocks[0](x.clone(), key_padding_mask=kpm)
        cut = _blocks(1)[0](x.clone(), mask=m, key_padding_mask=kpm)
    assert torch.equal(base[:, 0], cut[:, 0]), "单层 ego 必须逐位不变"
    assert not torch.equal(base[:, 1:A], cut[:, 1:A]), "agent token 必须变，否则干预没生效"


def test_vx2_effect_propagates_to_ego_with_depth():
    """≥2 层时必须传导到 ego，且 4 层强于 2 层 —— 干预打在边上的正面证据。"""
    x, a_pad, m_pad = _x_kpm()
    kpm = torch.cat([a_pad, m_pad]).unsqueeze(0)
    lay = build_layout(a_pad, m_pad)
    m = make_attn_mask(lay.n_total, lay.pairs["a2a"], torch.device("cpu"))

    def run(depth, mask):
        h = x.clone()
        with torch.no_grad():
            for blk in _blocks(depth):
                h = blk(h, mask=mask, key_padding_mask=kpm)
        return h

    d2 = (run(2, None)[:, 0] - run(2, m)[:, 0]).abs().max().item()
    d4 = (run(4, None)[:, 0] - run(4, m)[:, 0]).abs().max().item()
    assert d2 > 0, "depth=2 必须传导到 ego"
    assert d4 > d2, "层数越多传导越强"


def test_count_matched_control_can_match_a2a_size():
    """count-matched：LL 池必须够大，且抽出的对数与 a2a 精确相等。"""
    _, a_pad, m_pad = _x_kpm(n_pad_agent=1, n_pad_map=0)
    lay = build_layout(a_pad, m_pad)
    k = lay.pairs["a2a"].shape[0]
    assert lay.pairs["LL"].shape[0] >= k, "LL 池不足 ⇒ 判据 §2 要求当场记 skip"
    rng = np.random.default_rng(1000)
    pick = lay.pairs["LL"][rng.choice(lay.pairs["LL"].shape[0], size=k, replace=False)]
    assert pick.shape[0] == k
