"""`magnet_joint_layers.py` 的单测 —— 覆盖 Q110 §7 的 V-6，并把 V-1 / V-11 / V-12 推进到 MP 层。

🔴 门的语义：
  V-1  无 global pooling：MP 之后仍是 `[n_entity, dim]`；
  V-6  relation type 生效：换 `r_ij` 必须改变消息（否则 relation 是装饰，Q109 §11 风险表）；
  V-11 padding 全程不参与：扰动 padding 向量，输出 **max|Δ| = 0**（靠 `w=0 ⇒ 消息恰好 0`）；
  V-12 批处理不串样本：批跑逐样本切片 == 单样本跑。
另加四项本模块特有的契约门：
  * **几何**：agent 取最后一个有效向量的位姿；lane 朝向用**圆周均值**（🔴 跨 ±π 不得炸，D-3）；
  * **P6/D-b**：只有车辆↔车辆边被打分，其余 `w ≡ 1`；
  * **D-f**：`forward` 返回的 `edge_weight` 与 MP 内部用的是**同一份**（不得有第二个打分头）；
  * **干预口径**：`isolate_edge_gate` 与 `g1_dose_ladder_probe._isolate` 重建 `edge_index` **等价**。
"""
from __future__ import annotations

import dataclasses
import glob
import math
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
    REL_LANE_LANE, REL_VEH_VEH, EdgeScorer, JointInteractionStack, RelationMessagePassing,
    _GeometryEncoder, build_edge_context, build_entity_geometry, conserve_target_mass,
    isolate_edge_gate)
from scripts.analysis.g1_dose_ladder_probe import _isolate                   # noqa: E402
from scripts.analysis.run_interaction_audit import _node_types_from_df       # noqa: E402
from src.shared.utils.dataset import build_graph_data_from_feature_df        # noqa: E402

VAL_ROOT = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "val"
torch.manual_seed(20260825)

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
    from torch_geometric.data import Batch
    return Batch.from_data_list(list(datas), follow_batch=["gnn_x"])


def _pipeline(data, learn_topology: bool = True, seed: int = 7):
    """编码器 → 几何 → 边上下文 → 交互栈。返回 (idx, geom, ectx, h0, stack)。"""
    torch.manual_seed(seed)
    enc = EntityEncoder(dim=128).eval()
    stack = JointInteractionStack(dim=128, learn_topology=learn_topology).eval()
    idx = build_entity_index(data)
    geom = build_entity_geometry(data.x, idx)
    ectx = build_edge_context(data.gnn_edge_index, idx, geom)
    with torch.no_grad():
        h0 = enc(data.x, idx)
    return idx, geom, ectx, h0, stack


SAMPLES = _load(4)
assert SAMPLES, f"{VAL_ROOT} 下没有可用样本"


@check("几何契约：agent 位姿取最后一个有效向量；lane 朝向用圆周均值（跨 ±π 不炸）")
def _t1():
    for d, df in SAMPLES:
        idx = build_entity_index(d)
        geom = build_entity_geometry(d.x, idx)
        assert geom.pos.shape == (idx.n_entity, 2) and geom.vel.shape == (idx.n_entity, 2)
        # padding 实体的几何全部归零
        assert float(geom.pos[~idx.entity_valid].abs().max()) == 0.0
        assert float(geom.vel[~idx.entity_valid].abs().max()) == 0.0
        # lane 恒无速度、agent 无 route 位（col8 语义按 node type 分岔）
        lane = idx.entity_valid & ~idx.entity_is_agent
        assert float(geom.vel[lane].abs().max()) == 0.0, "lane 被赋了速度 —— col8 语义读错了"
        assert float(geom.on_route[idx.entity_is_agent].abs().max()) == 0.0
        # agent 的位置 = 它最后一个有效向量的终点
        for e in torch.nonzero(idx.entity_valid & idx.entity_is_agent)[:5, 0].tolist():
            rows = torch.nonzero((idx.global_eid == e) & idx.vec_valid)[:, 0]
            assert torch.allclose(geom.pos[e], d.x[rows.max(), 2:4]), f"实体 {e} 位置取错了向量"
    # 圆周均值的反证：两个朝向 +179° / −179°，算术均值给 0°（反向），圆周均值须给 ±180°
    d = SAMPLES[0][0].clone()
    x = d.x.clone()
    lane_e = int(torch.nonzero(~build_entity_index(d).entity_is_agent
                               & build_entity_index(d).entity_valid)[0])
    idx = build_entity_index(d)
    rows = torch.nonzero((idx.global_eid == lane_e) & idx.vec_valid)[:, 0]
    assert rows.numel() >= 2, "抽到的 lane 只有一个向量，构造不出 wrap"
    x[rows, 5] = math.radians(179.0)
    x[rows[0], 5] = math.radians(-179.0)
    d.x = x
    h = float(build_entity_geometry(d.x, build_entity_index(d)).heading[lane_e])
    assert abs(abs(h) - math.pi) < math.radians(3.0), \
        f"圆周均值失效：得 {math.degrees(h):.1f}°，算术均值会给约 0°（D-3 的 wrap 陷阱）"


@check("EdgeContext：四类 relation 与真值类型一致，scorable ≡ 车辆↔车辆，valid ≡ 两端非 padding")
def _t2():
    d, df = SAMPLES[0]
    idx = build_entity_index(d)
    ectx = build_edge_context(d.gnn_edge_index, idx, build_entity_geometry(d.x, idx))
    types, _, _ = _node_types_from_df(df)
    truth = torch.tensor([t == "agent" for t in types])
    src, dst = d.gnn_edge_index[0], d.gnn_edge_index[1]
    assert torch.equal(ectx.scorable, truth[src] & truth[dst]), "scorable 不等于车辆↔车辆边"
    assert torch.equal(ectx.valid, idx.entity_valid[src] & idx.entity_valid[dst])
    assert torch.equal(ectx.rel_type == REL_VEH_VEH, truth[src] & truth[dst])
    assert int(ectx.rel_type.unique().numel()) >= 3, "本样本 relation 类型少于 3 类，测不出分岔"
    assert float(ectx.g[~ectx.valid].abs().max()) == 0.0, "padding 边的几何没被清零"


@check("V-6 relation type 生效：换 r_ij 必须改变消息")
def _t3():
    d = SAMPLES[0][0]
    idx, geom, ectx, h0, _ = _pipeline(d)
    torch.manual_seed(11)
    layer, geo = RelationMessagePassing(dim=128).eval(), _GeometryEncoder(32).eval()
    w = torch.ones(ectx.edge_index.shape[1]) * ectx.valid.float()
    with torch.no_grad():
        g_enc = geo(ectx.g)
        base = layer(h0, ectx, g_enc, w, idx.entity_valid)
        flat = dataclasses.replace(ectx, rel_type=torch.full_like(ectx.rel_type, REL_LANE_LANE))
        alt = layer(h0, flat, g_enc, w, idx.entity_valid)
    assert float((alt - base).abs().max()) > 1e-6, \
        "把全部边改成同一个 relation type 后输出不变 —— relation 只是装饰"


@check("V-1 / V-12：MP 之后仍是 [n_entity, dim]，批跑逐样本切片 == 单样本跑")
def _t4():
    datas = [d for d, _ in SAMPLES]
    bt = _batch(datas)
    idx, geom, ectx, h0, stack = _pipeline(bt)
    with torch.no_grad():
        out = stack(h0, ectx, idx)
    assert out.tokens.shape == (idx.n_entity, 128), out.tokens.shape
    assert float(out.tokens[~idx.entity_valid].abs().max()) == 0.0, "padding token 没被归零"

    offs = 0
    for d in datas:
        i1, g1, e1, h1, s1 = _pipeline(d)
        s1.load_state_dict(stack.state_dict())
        with torch.no_grad():
            o1 = s1(h1, e1, i1)
        n = i1.n_entity
        delta = float((o1.tokens - out.tokens[offs:offs + n]).abs().max())
        assert delta < 1e-5, f"批跑与单样本跑不一致：max|Δ| = {delta:.3e}（串样本了？）"
        offs += n


@check("V-11 padding 不参与：任意改动 padding 向量，MP 输出 max|Δ| = 0")
def _t5():
    bt = _batch([d for d, _ in SAMPLES])
    idx, geom, ectx, h0, stack = _pipeline(bt)
    with torch.no_grad():
        base = stack(h0, ectx, idx)
    x2 = bt.x.clone()
    pad = ~idx.vec_valid
    assert int(pad.sum()) > 0, "本批没有 padding 向量，测不出来"
    x2[pad] = x2[pad] + 137.0
    x2[pad, 9] = bt.x[pad, 9]                     # 🔴 子图 id 不能动，否则改的是分组不是特征
    bt2 = type(bt)(**{**bt.to_dict(), "x": x2})
    idx2 = build_entity_index(bt2)
    torch.manual_seed(7)
    enc2 = EntityEncoder(dim=128).eval()
    with torch.no_grad():
        h2 = enc2(x2, idx2)
        ectx2 = build_edge_context(bt.gnn_edge_index, idx2, build_entity_geometry(x2, idx2))
        out = stack(h2, ectx2, idx2)
    dmax = float((out.tokens - base.tokens).abs().max())
    assert dmax == 0.0, f"padding 扰动泄漏到 MP 输出：max|Δ| = {dmax:.3e}"


@check("P6/D-b：只有车辆↔车辆边被打分，lane 参与的边 w ≡ 1，padding 边 w ≡ 0")
def _t6():
    d = SAMPLES[0][0]
    idx, geom, ectx, h0, stack = _pipeline(d)
    with torch.no_grad():
        w = stack(h0, ectx, idx).edge_weight
    keep = ectx.valid & ~ectx.scorable
    assert int(keep.sum()) > 0 and int((ectx.valid & ectx.scorable).sum()) > 0
    assert float((w[keep] - 1.0).abs().max()) == 0.0, "lane 参与的边被打了分（违反 D-b）"
    assert float(w[~ectx.valid].abs().max()) == 0.0, "padding 边的 w 不为 0（违反 P8）"
    veh = w[ectx.valid & ectx.scorable]
    # 🔴 2026-08-26 断言更新：`stack` 在所有 mask 之后加了**逐目标质量守恒归一**
    #    (`conserve_target_mass`)，`w` 的均值被钉在 1 ⇒ 单条边**可以 > 1**（这是设计：
    #    守恒意味着可增强也可衰减，不再是「只能衰减」）。原断言 `max < 1.0` 测的是
    #    `σ ∈ (0,1)` 这个实现细节，已被下面的守恒判据取代 —— 后者更强。
    assert float(veh.min()) > 0.0, "车辆边出现非正 w"
    assert float(veh.std()) > 1e-4, "车辆边的 w 是常数 —— scorer 没在工作"
    m = ectx.valid & ectx.scorable
    tot = torch.zeros(idx.n_entity).index_add(0, ectx.edge_index[1][m], w[m])
    cnt = torch.zeros(idx.n_entity).index_add(0, ectx.edge_index[1][m],
                                             torch.ones(int(m.sum())))
    hit = cnt > 0
    assert float((tot[hit] - cnt[hit]).abs().max()) < 1e-4, \
        f"逐目标权重和 ≠ d_i（max|Δ| {float((tot[hit]-cnt[hit]).abs().max()):.3e}）—— 守恒归一失效"


@check("D-f：forward 返回的 edge_weight 就是 MP 用的那一份（不存在第二个打分头）")
def _t7():
    d = SAMPLES[0][0]
    idx, geom, ectx, h0, stack = _pipeline(d)
    with torch.no_grad():
        out = stack(h0, ectx, idx)
        # 🔴 2026-08-26 更新：`stack` 在 mask 之后多了一步守恒归一，所以手工重算要带上它。
        #    门的**意图不变** —— 验的仍是「只有一个打分头，返回的就是 MP 用的那一份」。
        raw = stack.scorer(h0, ectx, stack.geo_enc(ectx.g)) * ectx.valid.float()
        manual = conserve_target_mass(raw, ectx, h0.shape[0],
                                      active=ectx.scorable & ectx.valid)
    assert torch.equal(out.edge_weight, manual), "返回的 w 与 scorer 逐位不同 —— 通路分叉了"
    # 固定拓扑臂：w ≡ 1（× valid），且 scorer 仍被实例化（参数量逐位相同，Q110 §4-1）
    torch.manual_seed(7)
    fixed = JointInteractionStack(dim=128, learn_topology=False).eval()
    with torch.no_grad():
        wf = fixed(h0, ectx, idx).edge_weight
    assert torch.equal(wf, ectx.valid.float()), "固定拓扑臂的 w 不是恒 1"
    n_learn = sum(p.numel() for p in stack.parameters())
    n_fixed = sum(p.numel() for p in fixed.parameters())
    assert n_learn == n_fixed, f"两臂参数量不同：{n_learn} vs {n_fixed}（Q110 §4-1 要求同容量）"
    print(f"        （交互栈参数量 {n_learn:,}；两臂逐位相同）")


@check("干预口径：isolate_edge_gate 与 g1_dose_ladder_probe._isolate 重建 edge_index 等价")
def _t8():
    d = SAMPLES[0][0]
    idx, geom, ectx, h0, stack = _pipeline(d)
    j = int(torch.nonzero(idx.entity_valid & idx.entity_is_agent)[1])
    ei2, n_cut = _isolate(d.gnn_edge_index, j, idx.n_entity)
    assert n_cut > 0
    ectx2 = build_edge_context(ei2, idx, geom)
    with torch.no_grad():
        by_gate = stack(h0, ectx, idx, edge_gate=isolate_edge_gate(ectx, torch.tensor([j])))
        by_rebuild = stack(h0, ectx2, idx)
    dmax = float((by_gate.tokens - by_rebuild.tokens).abs().max())
    assert dmax == 0.0, f"门与重建不等价：max|Δ| = {dmax:.3e}（CPU 上应恰好 0）"
    # 一阶隔离生效：孤立 j 之后，改 j 的 token 不再影响任何 i≠j
    h_alt = h0.clone()
    h_alt[j] = h_alt[j] + 5.0
    with torch.no_grad():
        alt = stack(h_alt, ectx2, idx)
    delta = (alt.tokens - by_rebuild.tokens).abs().max(dim=1).values
    others = torch.cat([delta[:j], delta[j + 1:]])
    assert float(others.max()) == 0.0, \
        f"孤立后仍有泄漏：max|Δ| = {float(others.max()):.3e}（应为 0；二阶泄漏走 decoder，V-2 另测）"


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for n, e in _FAIL:
        print(f"  - {n}: {e}")
    sys.exit(1)
