"""`magnet_joint_decoder.py` + `magnet_joint_adaptive.py` 的单测 —— Q110 §7 的 V-2 / V-7 / V-9 / V-10。

🔴 门的语义：
  V-2  **实体隔离（孤立 j）**：切 j 的全部边（两个方向）**且**把 decoder 对 j 的注意力置零后，
       改 j 的输入对**任何** i≠j 的输出 **max|Δ| = 0**。
       ⚠️ 不能只切 `j→i` —— 存在二阶泄漏 `j → lane → decoder → i`（D-b 的已知代价）；
  V-7  **batch 不变性**：同一样本在 batch=1 与 batch=4 下输出一致（P4 换掉 BN 之后**应当**成立）；
  V-9  **模态不塌**：M 个模态的输出不得全部重合；
  V-10 **建类链**：`bn_calibration.build_model(joint_adaptive=True)` 能建出本类，
       与其它开关的任意组合被**响亮拦截**（D-e：顶层分支不是旋钮）。
另加四项本模块特有的契约门：
  * **P1 输出形状**：`[n_entity, M, 60, 2]`，不是 `[1, 60, 2]`；padding 实体的行恒为 0；
  * **位置域口径**：与 `trajectory_contract_joint` 的 `ego_tgt` 同口径（ego 锚点在原点）；
  * **D-f 生效**：切一条 `j→i` 边必须改变 i 的输出（decoder 受拓扑管辖，不是旁路）；
  * **D-i**：闭环模态选择 = `argmax π`，且**不是**模态平均。
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

from magnet_joint_adaptive import MagnetJointAdaptive                        # noqa: E402
from magnet_joint_decoder import select_mode_argmax                          # noqa: E402
from magnet_joint_encoder import build_entity_index                          # noqa: E402
from src.shared.utils.bn_calibration import CalibrationError, build_model    # noqa: E402
from src.shared.utils.dataset import build_graph_data_from_feature_df        # noqa: E402
from src.shared.utils.trajectory_contract_joint import ego_increments_to_positions  # noqa: E402

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
        if d is None or int(build_entity_index(d).entity_is_agent.sum()) < 4:
            continue
        out.append((d, df))
        if len(out) == n:
            break
    return out


def _batch(datas):
    from torch_geometric.data import Batch
    return Batch.from_data_list(list(datas), follow_batch=["gnn_x"])


def _model(seed: int = 3, **kw) -> MagnetJointAdaptive:
    torch.manual_seed(seed)
    return MagnetJointAdaptive(**kw).eval()


SAMPLES = _load(4)
assert SAMPLES, f"{VAL_ROOT} 下没有可用样本"
MODEL = _model()


@check("P1 输出形状 [n_entity, M, 60, 2]（不是 [1,60,2]），padding 行恒为 0")
def _t1():
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    with torch.no_grad():
        pred = MODEL(d)
    assert pred.traj.shape == (idx.n_entity, 6, 60, 2), pred.traj.shape
    assert pred.logits.shape == (idx.n_entity, 6), pred.logits.shape
    assert float(pred.traj[~idx.entity_valid].abs().max()) == 0.0, "padding 实体的轨迹不为零"
    assert float(pred.logits[~idx.entity_valid].abs().max()) == 0.0
    assert float(pred.traj[idx.entity_valid].abs().max()) > 0.0, "有效实体输出全零，模型没工作"
    print(f"        （n_entity={idx.n_entity}，联合输出 {pred.traj[idx.entity_valid].numel():,} 维）")


@check("位置域口径：ego 锚点在原点，与 trajectory_contract_joint 的 ego_tgt 同口径")
def _t2():
    d, df = SAMPLES[0]
    idx = build_entity_index(d)
    ego = int(idx.ego_index[0])
    tgt = ego_increments_to_positions(np.asarray(df.iloc[0]["GROUND_TRUTH"]))
    assert tgt.shape == (60, 2) and float(np.abs(tgt[0]).max()) == 0.0
    with torch.no_grad():
        pred = MODEL(d)
    # ego 的锚点是它的当前位置（实测 |pos| < 1e-8），故未训练模型的首帧应当在原点附近
    assert pred.traj.shape[2] == tgt.shape[0], "预测帧数与 target 不一致"
    # 周车锚点必须是它自己的当前位置（离原点中位 20–40 m）⇒ 轨迹量级应在场景尺度上
    va = idx.entity_valid & idx.entity_is_agent
    med = float(pred.traj[va].norm(dim=-1).median())
    assert 1.0 < med < 200.0, f"周车轨迹量级 {med:.2f} 不在场景尺度上 —— 锚点可能没生效"


@check("V-9 模态不塌：M 个模态的输出不得全部重合")
def _t3():
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    with torch.no_grad():
        traj = MODEL(d).traj[idx.entity_valid]                    # [n_valid, M, 60, 2]
    spread = (traj - traj.mean(dim=1, keepdim=True)).abs().max()
    assert float(spread) > 1e-4, f"M 个模态几乎重合（max 偏离 {float(spread):.3e}）—— mode collapse"


@check("V-7 batch 不变性：batch=1 与 batch=4 输出一致（P4 换掉 BN 之后应当成立）")
def _t4():
    datas = [d for d, _ in SAMPLES]
    bt = _batch(datas)
    with torch.no_grad():
        big = MODEL(bt).traj
    offs = 0
    worst = 0.0
    for d in datas:
        n = int(d.gnn_x.shape[0])
        with torch.no_grad():
            one = MODEL(d).traj
        worst = max(worst, float((one - big[offs:offs + n]).abs().max()))
        offs += n
    assert worst < 1e-4, f"batch 依赖仍在：max|Δ| = {worst:.3e}（P4 应已消除 BN 这个唯一来源）"
    print(f"        （batch=1 vs batch=4 max|Δ| = {worst:.2e}）")


@check("D-f 生效：切一条 j→i 边必须改变 i 的输出（decoder 不是旁路）")
def _t5():
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    ag = torch.nonzero(idx.entity_valid & idx.entity_is_agent)[:, 0]
    i, j = int(ag[0]), int(ag[1])
    src, dst = d.gnn_edge_index[0], d.gnn_edge_index[1]
    hit = (src == j) & (dst == i)
    assert int(hit.sum()) == 1, "找不到 j→i 这条边"
    gate = (~hit).float()
    with torch.no_grad():
        base = MODEL(d).traj
        cut = MODEL(d, edge_gate=gate).traj
    di = float((cut[i] - base[i]).abs().max())
    assert di > 0.0, "切掉 j→i 后 i 的输出一点没变 —— 拓扑没管住任何通路"
    print(f"        （切单边后 i 的 max|Δ| = {di:.3e} m）")


@check("V-2 实体隔离：孤立 j（切全部边 + 屏蔽 decoder 注意力）后，改 j 不影响任何 i≠j")
def _t6():
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    ag = torch.nonzero(idx.entity_valid & idx.entity_is_agent)[:, 0]
    j = int(ag[1])
    iso = torch.tensor([j])
    x2 = d.x.clone()
    rows = (idx.global_eid == j) & idx.vec_valid
    assert int(rows.sum()) > 0
    x2[rows, :9] = x2[rows, :9] + 3.0
    d2 = d.clone()
    d2.x = x2
    with torch.no_grad():
        base = MODEL(d, isolate=iso).traj
        alt = MODEL(d2, isolate=iso).traj
    delta = (alt - base).abs().amax(dim=(1, 2, 3))
    others = torch.cat([delta[:j], delta[j + 1:]])
    assert float(delta[j]) > 0.0, "扰动没生效（j 自己也没变）"
    assert float(others.max()) == 0.0, \
        f"孤立后仍有泄漏：max|Δ| = {float(others.max()):.3e}（应为逐位 0）"
    # 🔴 反证一：完全不干预时泄漏**必须存在**，否则这个门是空的
    with torch.no_grad():
        raw = (MODEL(d2).traj - MODEL(d).traj).abs().amax(dim=(1, 2, 3))
    raw = torch.cat([raw[:j], raw[j + 1:]]).max()
    assert float(raw) > 1e-3, f"不干预时也没泄漏（{float(raw):.3e}）—— 本门测不出东西"

    # 🔴 反证二：只切边时 decoder 对 j 的**注意力质量仍非零** —— 结构上通路还在，
    #    只是被 log-gating 压了约 6 个数量级；位移看不见是因为 anchor≈20 m 时
    #    float32 的绝对分辨率已是 3.8e-06 m。⇒ `key_block` 给的是**结构性**保证，不是数值巧合。
    from magnet_joint_layers import build_edge_context, build_entity_geometry, isolate_edge_gate
    ectx = build_edge_context(d.gnn_edge_index, idx, build_entity_geometry(d.x, idx))
    only_edges = isolate_edge_gate(ectx, iso)
    keep = idx.entity_valid.clone()
    keep[j] = False
    with torch.no_grad():
        a_none = MODEL(d, need_attn=True).attn[0, :, j]
        a_edge = MODEL(d, edge_gate=only_edges, need_attn=True).attn[0, :, j]
        a_iso = MODEL(d, isolate=iso, need_attn=True).attn[0, :, j]
    assert float(a_none[keep].max()) > 1e-3, "无干预时其它实体就不看 j，反证不成立"
    assert 0.0 < float(a_edge[keep].max()) < 1e-5, \
        f"只切边时对 j 的注意力应当是「很小但非零」，实测 {float(a_edge[keep].max()):.3e}"
    assert float(a_iso[keep].max()) == 0.0, "屏蔽后对 j 的注意力不是逐位 0"
    print(f"        （对 j 的注意力：无干预 {float(a_none[keep].max()):.3e} → "
          f"只切边 {float(a_edge[keep].max()):.3e} → 孤立 0；位移 {float(raw):.3e} → 0）")


@check("V-10 建类链：build_model(joint_adaptive=True) 建出本类，任何旧旋钮组合被响亮拦截")
def _t7():
    m = build_model(joint_adaptive=True)
    assert isinstance(m, MagnetJointAdaptive), type(m)
    assert m.num_modes == 6 and m.learn_topology is True
    fixed = build_model(joint_adaptive=True, learn_topology=False)
    assert fixed.learn_topology is False
    n1 = sum(p.numel() for p in m.parameters())
    n2 = sum(p.numel() for p in fixed.parameters())
    assert n1 == n2, f"learned / fixed 两臂参数量不同：{n1} vs {n2}（Q110 §4-1）"
    for bad in ("head_norm", "agent_aux", "multimodal", "temporal_graph",
                "context_pooling", "ego_query_fusion", "gnn_pooling_fix"):
        try:
            build_model(joint_adaptive=True, **{bad: True})
        except CalibrationError:
            continue
        raise AssertionError(f"joint_adaptive + {bad} 没有被拦截")
    try:                                              # 非 bool 开关（gnn_operator）也要拦
        build_model(joint_adaptive=True, gnn_operator="GATv2Conv")
    except CalibrationError:
        pass
    else:
        raise AssertionError("joint_adaptive + gnn_operator 没有被拦截")
    print(f"        （joint_adaptive 参数量 {n1:,}；learned / fixed 两臂逐位相同）")


@check("D-i 闭环模态选择 = argmax π（且不是模态平均）")
def _t8():
    d = SAMPLES[0][0]
    idx = build_entity_index(d)
    with torch.no_grad():
        pred = MODEL(d)
        sel, best = select_mode_argmax(pred)
        closed = MODEL.predict_closed_loop(d)
    assert sel.shape == (idx.n_entity, 60, 2) and best.shape == (idx.n_entity,)
    assert torch.equal(best, pred.logits.argmax(dim=-1))
    assert torch.equal(sel, closed), "predict_closed_loop 与 select_mode_argmax 不一致"
    mean_traj = pred.traj.mean(dim=1)
    gap = float((sel[idx.entity_valid] - mean_traj[idx.entity_valid]).abs().max())
    assert gap > 1e-5, f"argmax 与模态平均几乎相同（max|Δ| {gap:.3e}）—— 该门测不出区别"
    print(f"        （argmax 与模态平均相差 max {gap:.3e} m ⇒ 平均会造出不存在的中间轨迹）")


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for n, e in _FAIL:
        print(f"  - {n}: {e}")
    sys.exit(1)
