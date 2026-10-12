"""`magnet_joint_loss.py` 的单测 —— joint WTA 损失、多模态指标、P5 监控。

🔴 门的语义（每条都带一个**反证**，避免测出一个恒真的东西）：
  * ego target ≡ `cumsum(data.y)`，与 `ego_increments_to_positions` **逐位一致**；
  * 周车 target 行序 ≡ `nonzero(gnn_is_agent)`；全零槽**不进** `is_target`；
  * 🔴 **FDE 取最后一个有效帧**：对截断样本，它与「第 59 帧」必须给出**不同**的数（反证）；
  * 逐帧 mask 生效：改补零帧的 target **不改变**损失（反证：改有效帧则必变）；
  * scene-level vs per-entity WTA：per-entity 的 minADE **≤** scene-level（数学必然）且实际不等；
  * 指标 ego / 周车**分开**报；`hit_rate` 与随机基线 1/K 可比；
  * P5：`MarginFunnelMonitor` 出数且 margin 的定义 ≤0 ⟺ 塌陷；`mode_usage` 求和为 1；
  * 损失可回传，且 **padding 实体没有梯度**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN / "scripts" / "training"), str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_agent_aux_dataset import AgentAuxGraphDataset                      # noqa: E402
from magnet_joint_adaptive import MagnetJointAdaptive                          # noqa: E402
from magnet_joint_encoder import EntityIndexError, build_entity_index          # noqa: E402
from magnet_joint_loss import (JointWTALoss, MarginFunnelMonitor, _best_mode,  # noqa: E402
                               build_joint_targets_batch, joint_metrics,
                               mode_diversity, mode_usage, per_mode_errors)
from magnet_joint_layers import build_entity_geometry                          # noqa: E402
from src.shared.utils.scene_strata import MOVING_DISP_M                        # noqa: E402
from src.shared.utils.dataset import get_adjacency_config                      # noqa: E402

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


get_adjacency_config("default")
DS = AgentAuxGraphDataset(str(VAL_ROOT), adjacency_config="default")


def _batch(n: int = 4, start: int = 0):
    from torch_geometric.data import Batch
    return Batch.from_data_list([DS[i] for i in range(start, start + n)],
                                follow_batch=["gnn_x"])


BT = _batch(4)
IDX = build_entity_index(BT)
TG = build_joint_targets_batch(BT, IDX)
torch.manual_seed(3)
MODEL = MagnetJointAdaptive().eval()
with torch.no_grad():
    PRED = MODEL(BT)


@check("ego target ≡ cumsum(data.y)，首帧 (0,0)；ego 恒全帧有效")
def _t1():
    y = BT.y.view(TG.n_graphs, 60, 2)
    for g, row in enumerate(IDX.ego_index.tolist()):
        expect = torch.cumsum(y[g], dim=0)
        assert torch.equal(TG.tgt[row], expect), f"样本 {g} 的 ego target 不等于 cumsum(y)"
        assert bool(TG.mask[row].all()), "ego 的逐帧 mask 不是全 True"
        assert float(TG.tgt[row][0].abs().max()) == 0.0, "ego 首帧不在原点"
    assert int(TG.is_ego.sum()) == TG.n_graphs


@check("周车 target 行序 ≡ nonzero(gnn_is_agent)；全零槽不进 is_target")
def _t2():
    rows = torch.nonzero(BT.gnn_is_agent.bool(), as_tuple=True)[0]
    assert int(rows.numel()) == int(BT.agent_future_tgt.shape[0])
    assert torch.equal(TG.tgt[rows], BT.agent_future_tgt)
    assert torch.equal(TG.mask[rows], BT.agent_future_mask)
    # ego 不在 gnn_is_agent 里（它是 ego/lane 为 False 的口径）
    assert not bool(BT.gnn_is_agent.bool()[IDX.ego_index].any()), "gnn_is_agent 把 ego 也算进去了"
    dead = rows[~BT.agent_future_mask.any(dim=1)]
    assert not bool(TG.is_target[dead].any()), "全帧补零的周车进了 is_target"
    n_ag = int((TG.is_target & ~TG.is_ego).sum())
    assert n_ag > 0
    print(f"        （批内监督实体 {int(TG.is_target.sum())} = ego {TG.n_graphs} + 周车 {n_ag}；"
          f"丢弃全零槽 {int(dead.numel())}）")


@check("🔴 FDE 取最后一个有效帧：对截断样本与「第 59 帧」必须不同（反证）")
def _t3():
    _ade, fde = per_mode_errors(PRED.traj, TG)
    err = torch.norm(PRED.traj - TG.tgt.unsqueeze(1), dim=-1)
    naive = err[:, :, -1]                                    # 错误做法：恒取第 59 帧
    trunc = TG.is_target & ~TG.mask[:, -1]                   # 被截断的周车
    assert int(trunc.sum()) > 0, "本批没有截断样本，反证做不了（换样本）"
    gap = float((fde[trunc] - naive[trunc]).abs().max())
    assert gap > 1e-3, f"两种 FDE 口径在截断样本上给出相同的数（max|Δ|={gap:.3e}）—— 门是空的"
    ok = TG.is_target & TG.mask[:, -1]
    assert torch.equal(fde[ok], naive[ok]), "未截断样本上两种口径应当逐位相同"
    print(f"        （截断实体 {int(trunc.sum())}/{int(TG.is_target.sum())}，"
          f"两种口径最大差 {gap:.2f} m —— 用第 59 帧会把补零读成「开到了原点」）")


@check("逐帧 mask 生效：改补零帧的 target 不改变损失（反证：改有效帧必变）")
def _t4():
    crit = JointWTALoss()
    base = float(crit(PRED, TG)[0])
    pad = ~TG.mask
    assert int(pad.sum()) > 0
    t2 = TG.tgt.clone()
    t2[pad] = t2[pad] + 50.0
    from dataclasses import replace
    assert abs(float(crit(PRED, replace(TG, tgt=t2))[0]) - base) < 1e-6, "补零帧影响了损失"
    t3 = TG.tgt.clone()
    t3[TG.mask] = t3[TG.mask] + 50.0
    assert abs(float(crit(PRED, replace(TG, tgt=t3))[0]) - base) > 1.0, "改有效帧竟不改变损失"


@check("scene-level vs per-entity WTA：per-entity minADE ≤ scene-level，且实际不等")
def _t5():
    ade, _ = per_mode_errors(PRED.traj, TG)
    sel = TG.is_target
    per_e = _best_mode(ade, TG, scene_level=False)
    scene = _best_mode(ade, TG, scene_level=True)
    ar = torch.arange(ade.shape[0])
    a_pe = float(ade[ar[sel], per_e[sel]].mean())
    a_sc = float(ade[ar[sel], scene[sel]].mean())
    assert a_pe <= a_sc + 1e-6, f"per-entity {a_pe:.4f} > scene-level {a_sc:.4f}（数学上不可能）"
    assert not torch.equal(per_e[sel], scene[sel]), "两种口径选出同一批模态 —— 门测不出区别"
    # scene-level 时同一场景内所有实体必须共用同一个模态
    for g in range(TG.n_graphs):
        s = sel & (TG.batch == g)
        if bool(s.any()):
            assert int(scene[s].unique().numel()) == 1, f"样本 {g} 的 scene-level 模态不唯一"
    print(f"        （minADE：per-entity {a_pe:.3f} m vs scene-level {a_sc:.3f} m，"
          f"差 {a_sc - a_pe:.3f} m = 联合一致性的代价）")


@check("指标 ego / 周车分开报，miss rate 带阈值，hit_rate 可与 1/K 比")
def _t6():
    m = joint_metrics(PRED, TG)
    for k in ("ego_minADE", "agent_minADE", "ego_minFDE", "agent_minFDE",
              "ego_missRate", "agent_missRate", "ego_hitRate", "agent_hitRate",
              "scene_minADE", "scene_minFDE", "miss_threshold_m"):
        assert k in m, f"指标缺 {k}"
    assert m["miss_threshold_m"] == 2.0
    assert 0.0 <= m["agent_missRate"] <= 1.0
    assert m["ego_n"] == float(TG.n_graphs)
    assert m["all_minADE"] >= m["all_minFDE"] * 0.0     # 只是形状检查
    # 未训练模型：hit_rate 应当在随机基线附近（1/6 ≈ 0.167），远离 1
    assert m["all_hitRate"] < 0.6, f"未训练模型的 hit_rate 竟有 {m['all_hitRate']:.3f}"
    print(f"        （未训练基线：ego minADE {m['ego_minADE']:.2f} / 周车 {m['agent_minADE']:.2f} m；"
          f"周车 missRate {m['agent_missRate']:.3f}；hitRate {m['all_hitRate']:.3f} vs 随机 0.167）")


@check("P5 监控：margin/funnel 出数（margin ≤0 ⟺ 塌陷），mode_usage 求和为 1")
def _t7():
    mon = MarginFunnelMonitor(MODEL.active_head())
    with torch.no_grad():
        MODEL(BT)
    st = mon.summary()
    mon.close()
    for k in ("margin_p1", "margin_median", "collapse_rate", "funnel_cos_median"):
        assert k in st and np.isfinite(st[k]), f"{k} 缺失或非有限"
    assert 0.0 <= st["collapse_rate"] <= 1.0
    assert -1.001 <= st["funnel_cos_median"] <= 1.001
    u = mode_usage(PRED, TG)
    tot = sum(v for k, v in u.items() if k.startswith("mode_usage_") and k[-1].isdigit())
    assert abs(tot - 1.0) < 1e-5, f"模态使用率求和 {tot}"
    assert u["mode_usage_max"] <= 1.0
    print(f"        （margin p1 {st['margin_p1']:.3f} / 中位 {st['margin_median']:.3f}，"
          f"塌陷率 {st['collapse_rate']:.3f}；funnel 余弦中位 {st['funnel_cos_median']:.3f}；"
          f"模态使用率 max {u['mode_usage_max']:.3f}）")


@check("损失可回传，且 padding 实体拿不到梯度")
def _t8():
    torch.manual_seed(3)
    model = MagnetJointAdaptive()
    model.train()
    x = BT.x.clone().requires_grad_(True)
    bt = BT.clone()
    bt.x = x
    idx = build_entity_index(bt)
    tg = build_joint_targets_batch(bt, idx)
    loss, comp = JointWTALoss()(model(bt), tg)
    loss.backward()
    assert comp["n_target"] > 0 and np.isfinite(comp["total"])
    g = x.grad
    assert g is not None and float(g[idx.vec_valid].abs().max()) > 0, "有效向量没有梯度"
    assert float(g[~idx.vec_valid].abs().max()) == 0.0, "padding 向量拿到了梯度（P8 被破坏）"
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0)
    print(f"        （loss={comp['total']:.4f} reg={comp['reg']:.4f} ce={comp['mode_ce']:.4f}；"
          f"{n_grad} 组参数收到非零梯度）")


@check("缺 aux 字段必须响亮失败（不得静默丢掉周车监督）")
def _t9():
    bt = BT.clone()
    bt.gnn_is_agent = None
    try:
        build_joint_targets_batch(bt, build_entity_index(bt))
    except EntityIndexError:
        return
    raise AssertionError("缺 `gnn_is_agent` 没有被拦截")


@check("🔴 is_moving 分层 ≡ scene_strata 冻结口径；反证：用第 59 帧算位移会分错层")
def _t10():
    pos0 = build_entity_geometry(BT.x, IDX).pos
    last = TG.mask.float().cumsum(dim=-1).argmax(dim=-1)
    end = TG.tgt.gather(1, last.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    disp = torch.norm(end - pos0, dim=-1)
    expect = (TG.is_target & ~TG.is_ego & (TG.mask.sum(dim=1) >= 60) & (disp > MOVING_DISP_M))
    assert torch.equal(TG.is_moving, expect), "is_moving 与 scene_strata 口径不一致"
    assert not bool(TG.is_moving[TG.is_ego].any()), "ego 被算进 moving 层（ego 必须单独成层）"
    # 三层互斥且并集 = is_target
    names = dict(TG.strata())
    ego_s, mov_s, sta_s, all_s = names["ego"], names["moving"], names["static"], names["all"]
    assert not bool((ego_s & mov_s).any() or (ego_s & sta_s).any() or (mov_s & sta_s).any())
    assert torch.equal(ego_s | mov_s | sta_s, all_s), "三层的并集不等于 is_target"
    # 🔴 反证：`is_full60` 这个前置条件是**吃重的**，不是装饰。
    #    去掉它并改用第 59 帧当终点（= P0 v1 的做法），截断的周车会被误判成「在动」——
    #    因为 `tgt[:, -1]` 是补零 ⇒ 位移退化成「该车到 ego 原点的距离」，几十米一抓一大把。
    #    （注意：**保留** full60 时两种终点口径必然相同，因为最后一个有效帧就是第 59 帧
    #      ⇒ 反证只能从 full60 这一侧做。）
    trunc = TG.is_target & ~TG.is_ego & ~TG.mask[:, -1]
    assert int(trunc.sum()) > 0, "本批没有截断样本，反证做不了"
    naive = torch.norm(TG.tgt[:, -1] - pos0, dim=-1)
    wrong = TG.is_target & ~TG.is_ego & (naive > MOVING_DISP_M)          # 无 full60 门
    n_bogus = int((wrong & trunc).sum())
    assert n_bogus > 0, "去掉 full60 门后没有截断车被误判 —— 门是空的，换样本"
    assert not bool((TG.is_moving & trunc).any()), "现口径把截断车算进了 moving"
    print(f"        （反证：去掉 full60 门 + 用第 59 帧 ⇒ {n_bogus}/{int(trunc.sum())} "
          f"辆截断车被误判成「在动」）")
    print(f"        （ego {int(ego_s.sum())} / moving {int(mov_s.sum())} / static {int(sta_s.sum())}；"
          f"静止占比 {float(sta_s.sum()) / max(float(all_s.sum()), 1.0):.3f}）")


@check("🔴 mode_usage / mode_diversity 必须分层；反证：模态压成同一条 ⇒ spread ≡ 0 而 modeMax 不变")
def _t11():
    mu, md = mode_usage(PRED, TG), mode_diversity(PRED, TG)
    for pre in ("ego_", "moving_", "static_", ""):
        assert f"{pre}mode_usage_max" in mu, f"缺 {pre}mode_usage_max"
        assert f"{pre}mode_spread" in md, f"缺 {pre}mode_spread"
    assert abs(sum(mu[f"mode_usage_{i}"] for i in range(6)) - 1.0) < 1e-5
    # 反证：把全部模态压成第 0 个模态 ⇒ 散度必须逐位归零，而 mode_usage 只看 logits ⇒ 不变
    from magnet_joint_decoder import JointPrediction
    flat = JointPrediction(traj=PRED.traj[:, :1].expand_as(PRED.traj).contiguous(),
                           logits=PRED.logits)
    md2 = mode_diversity(flat, TG)
    assert max(abs(v) for v in md2.values()) < 1e-6, "模态压平后散度没有归零 —— 测点错了"
    assert mode_usage(flat, TG)["mode_usage_max"] == mu["mode_usage_max"], \
        "mode_usage 受轨迹影响 —— 它只应读 logits"
    print(f"        （spread ego {md['ego_mode_spread']:.3f} m / moving "
          f"{md.get('moving_mode_spread', float('nan')):.3f} m / static "
          f"{md['static_mode_spread']:.3f} m；modeMax ego {mu['ego_mode_usage_max']:.3f} / "
          f"static {mu['static_mode_usage_max']:.3f}）")


@check("🔴 A 臂 ego_only 是**单变量 = 任务**：只缩 is_target，tgt/mask 逐位不变")
def _t12():
    tg_a = build_joint_targets_batch(BT, IDX, ego_only=True)
    assert torch.equal(tg_a.tgt, TG.tgt), "A 臂改动了 target 张量 —— 那就不是单变量了"
    assert torch.equal(tg_a.mask, TG.mask), "A 臂改动了 mask"
    assert torch.equal(tg_a.is_ego, TG.is_ego) and tg_a.n_graphs == TG.n_graphs
    assert torch.equal(tg_a.is_target, TG.is_target & TG.is_ego), "is_target 不是 B 臂与 ego 的交"
    assert int(tg_a.is_target.sum()) == TG.n_graphs, "A 臂的监督实体数应恰为场景数"
    assert not bool(tg_a.is_moving.any()), "A 臂里不该有 moving 层（周车不进监督）"
    # 反证：A 臂的损失只由 ego 决定 —— 把周车的 target 整体平移，损失必须逐位不变
    crit = JointWTALoss()
    base = float(crit(PRED, tg_a)[0])
    from dataclasses import replace
    t2 = TG.tgt.clone()
    t2[~TG.is_ego] = t2[~TG.is_ego] + 50.0
    moved = float(crit(PRED, replace(tg_a, tgt=t2))[0])
    assert abs(moved - base) < 1e-6, f"A 臂的损失受周车 target 影响（{base:.6f} -> {moved:.6f}）"
    b_base = float(crit(PRED, TG)[0])
    b_moved = float(crit(PRED, replace(TG, tgt=t2))[0])
    assert abs(b_moved - b_base) > 1.0, "反证失败：B 臂也不受周车影响，说明测点错了"
    print(f"        （A 臂监督 {int(tg_a.is_target.sum())} = 场景数；平移周车 target 后 "
          f"A 损失 {base:.4f}→{moved:.4f} 不变，B 损失 {b_base:.4f}→{b_moved:.4f} 剧变）")


@check("🔴 K=1（单模态对照臂）不得产生 NaN：散度按定义为 0，CE 恒 0")
def _t13():
    from magnet_joint_adaptive import MagnetJointAdaptive as _M
    torch.manual_seed(7)
    m1 = _M(num_modes=1, learn_topology=False).eval()
    with torch.no_grad():
        p1 = m1(BT)
    loss, comp = JointWTALoss()(p1, TG)
    allm = {**joint_metrics(p1, TG), **mode_usage(p1, TG), **mode_diversity(p1, TG)}
    nan = [k for k, v in allm.items() if isinstance(v, float) and v != v]
    assert not nan, f"K=1 下出现 NaN 键：{nan}"
    assert abs(comp["mode_ce"]) < 1e-9, f"单类 CE 应恒为 0，实得 {comp['mode_ce']}"
    assert allm["ego_mode_spread"] == 0.0 and allm["mode_spread"] == 0.0
    assert allm["ego_minADE"] == allm["ego_top1ADE"], "K=1 时 minADE 与 top1 必须相等"
    # 反证：K=2 时散度不再是 0（否则上面那条是恒真的）
    torch.manual_seed(7)
    with torch.no_grad():
        p2 = _M(num_modes=2, learn_topology=False).eval()(BT)
    assert mode_diversity(p2, TG)["ego_mode_spread"] > 0.0, "K=2 散度为 0 ⇒ 反证失败"
    print(f"        （K=1：CE={comp['mode_ce']:.1e} 散度=0 minADE≡top1={allm['ego_minADE']:.4f}；"
          f"K=2 散度={mode_diversity(p2, TG)['ego_mode_spread']:.4f}）")


print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for n, e in _FAIL:
        print(f"  - {n}: {e}")
    sys.exit(1)
