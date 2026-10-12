"""`magnet_joint_encoder.py` 的单测 —— 覆盖 Q110 §7 的 V-1 / V-11 / V-12 / V-13。

🔴 门的语义：
  V-1  无 global pooling：输出是 `[n_entity, dim]` 而不是 `[1, dim]`，且改一个实体只动它自己那一行；
  V-11 padding 全程不参与：改 padding 实体/向量的特征，输出 **max|Δ| = 0**；
  V-12 批处理不串样本：`global_eid.max() == gnn_x.size(0)-1`，且逐样本分组与单样本跑一致；
  V-13 ego 恒为每样本的节点 0，且 `n_entity == gnn_x.size(0)`（token 不重复）。
另加：契约 C（类型按实体投票）、契约 A 违约时必须**响亮失败**。
"""
from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN / "scripts" / "training"), str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_joint_encoder import (  # noqa: E402
    TYPE_AGENT, EntityEncoder, EntityIndexError, build_entity_index)
from scripts.analysis.run_interaction_audit import _node_types_from_df  # noqa: E402
from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402

VAL_ROOT = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "val"
torch.manual_seed(20260822)

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
        out.append((d, df))
        if len(out) == n:
            break
    return out


def _batch(datas):
    """按训练侧口径批处理：**必须** `follow_batch=['gnn_x']`（契约 B）。"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(list(datas), follow_batch=["gnn_x"])


SAMPLES = _load(4)
assert SAMPLES, f"{VAL_ROOT} 下没有可用样本"


@check("V-13 单样本：n_entity == gnn_x 行数，ego 索引 = 0")
def _t1():
    for d, df in SAMPLES:
        idx = build_entity_index(d)
        assert idx.n_entity == d.gnn_x.shape[0], (idx.n_entity, d.gnn_x.shape[0])
        assert idx.ego_index.tolist() == [0], idx.ego_index.tolist()
        types, is_ego, _ = _node_types_from_df(df)
        assert int(np.flatnonzero(is_ego)[0]) == 0, "真值里 ego 不是节点 0，契约 D 被破坏"


@check("契约 C：实体类型按投票与真值 SUBGRAPH_ID_TO_TYPE 完全一致")
def _t2():
    for d, df in SAMPLES:
        idx = build_entity_index(d)
        types, _, _ = _node_types_from_df(df)
        truth = torch.tensor([t == "agent" for t in types])
        assert torch.equal(idx.entity_is_agent.cpu(), truth), \
            f"类型不一致：{int((idx.entity_is_agent.cpu() != truth).sum())} 个实体判错"


@check("P8：padding 判据与「gnn_x 除 col9 外整行全零」一致，且 agent 侧 padding 为 0")
def _t3():
    n_pad = n_tot = n_agent_pad = 0
    for d, _ in SAMPLES:
        idx = build_entity_index(d)
        cols = [c for c in range(12) if c != 9]
        expect = d.gnn_x[:, cols].abs().sum(dim=1) > 0
        assert torch.equal(idx.entity_valid, expect)
        n_pad += int((~idx.entity_valid).sum()); n_tot += idx.n_entity
        n_agent_pad += int((idx.entity_is_agent & ~idx.entity_valid).sum())
    assert n_pad > 0, "抽到的样本里一个 padding 都没有，判据可能失效"
    assert n_agent_pad == 0, f"{n_agent_pad} 个 agent 实体被判为 padding —— 实测应为 0"
    print(f"        （padding 占比 {100 * n_pad / n_tot:.1f}%）")


@check("V-12 批处理：global_eid 覆盖且不越界，逐样本与单样本分组一致")
def _t4():
    datas = [d for d, _ in SAMPLES]
    bt = _batch(datas)
    idx = build_entity_index(bt)
    assert idx.n_entity == sum(int(d.gnn_x.shape[0]) for d in datas)
    assert int(idx.global_eid.max()) == idx.n_entity - 1, \
        f"global_eid.max()={int(idx.global_eid.max())} != n_entity-1={idx.n_entity - 1}"
    # 🔴 直接用未偏移的 cluster 会撞：确认它确实不够用（契约 B 的反证）
    raw_max = int(bt.x[:, 9].round().long().max())
    assert raw_max < idx.n_entity - 1, "本批的 cluster 恰好没撞，换样本再测（契约 B 的反证需要撞）"
    # 逐样本的实体数与单样本一致
    offs = 0
    for d in datas:
        n = int(d.gnn_x.shape[0])
        assert int((idx.entity_batch[offs:offs + n]).unique().numel()) == 1
        offs += n
    assert idx.ego_index.tolist() == list(np.cumsum([0] + [int(d.gnn_x.shape[0]) for d in datas])[:-1])


@check("V-1 无 global pooling：输出 [n_entity, dim]，且 padding 行恒为 0")
def _t5():
    enc = EntityEncoder(dim=128).eval()
    bt = _batch([d for d, _ in SAMPLES])
    idx = build_entity_index(bt)
    with torch.no_grad():
        tok = enc(bt.x, idx)
    assert tok.shape == (idx.n_entity, 128), tok.shape
    assert float(tok[~idx.entity_valid].abs().max()) == 0.0, "padding 实体的 token 不为零"
    assert float(tok[idx.entity_valid].abs().max()) > 0.0, "有效实体的 token 全零，编码器没工作"


@check("V-11 padding 不参与：任意改动 padding 向量，输出 max|Δ| = 0")
def _t6():
    enc = EntityEncoder(dim=128).eval()
    bt = _batch([d for d, _ in SAMPLES])
    idx = build_entity_index(bt)
    with torch.no_grad():
        base = enc(bt.x, idx)
        x2 = bt.x.clone()
        pad_vec = ~idx.vec_valid
        assert int(pad_vec.sum()) > 0, "本批没有 padding 向量，测不出来"
        x2[pad_vec] = x2[pad_vec] + 137.0          # 任意大扰动
        x2[pad_vec, 9] = bt.x[pad_vec, 9]          # 🔴 子图 id 不能动，否则改的是分组不是特征
        idx2 = build_entity_index(
            type(bt)(**{**{k: v for k, v in bt.to_dict().items()}, "x": x2}))
        out = enc(x2, idx2)
    assert float((out - base).abs().max()) == 0.0, \
        f"padding 扰动泄漏到输出：max|Δ| = {float((out - base).abs().max()):.3e}"


@check("V-1 实体隔离（编码器层）：改一个实体的向量只动它自己那一行")
def _t7():
    enc = EntityEncoder(dim=128).eval()
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    target = int(torch.nonzero(idx.entity_valid & idx.entity_is_agent)[1])
    with torch.no_grad():
        base = enc(d.x, idx)
        x2 = d.x.clone()
        m = (idx.global_eid == target) & idx.vec_valid
        x2[m, :9] = x2[m, :9] + 3.0
        out = enc(x2, idx)
    delta = (out - base).abs().max(dim=1).values
    assert float(delta[target]) > 0, "目标实体没变，扰动没生效"
    others = torch.cat([delta[:target], delta[target + 1:]])
    assert float(others.max()) == 0.0, f"扰动泄漏到其它实体：max|Δ| = {float(others.max()):.3e}"


@check("契约 A 违约必须响亮失败（不得静默降级）")
def _t8():
    d = SAMPLES[0][0].clone()
    d.x = d.x.clone()
    d.x[:, 9] = d.x[:, 9] + float(d.gnn_x.shape[0])      # 制造越界的 subgraph_id
    try:
        build_entity_index(d)
    except EntityIndexError:
        return
    raise AssertionError("越界的 subgraph_id 没有被拦截")


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for n, e in _FAIL:
        print(f"  - {n}: {e}")
    sys.exit(1)
