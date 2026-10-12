"""S2 闸门读数 —— 在**冻结 test** 上评 `MagnetJointAdaptive`（路线 B 新结构）。

## 🔴 闸要比的是 `top1 ego ADE`，不是 `ego_minADE`

日志里打的 `ego_minADE` 是 **minADE@6**，带 **oracle 成分（6 选 1）**，
**不得**与旧 MAGNET 的单输出 ADE 并列。**S2 前置健全性闸**（Q110 §6.1）比的是
**部署口径**的 top1（D-i = `argmax π`）对 `fr_base_seed1` 的 **2.925 / 7.378**。

## 🔴 但 top1 失败不等于结构失败 —— 必须同时给「最佳固定模态」

08-25 实测置信度头没学起来（mode CE ≈ 赢家分布熵）。若只报 top1，
「轨迹差」与「选模差」会被糊成一个数，而这两种情况下一步完全相反。
⇒ 本脚本同时报三档，**事前定义、一起报**：

| 档 | 含义 | 用途 |
|---|---|---|
| `top1` | `argmax π` 选出的模态 | **闸的判据**（部署口径） |
| `best_fixed` | 全集上**恒取同一个模态**（6 个里最好的那个） | 不用置信度头的下界 —— 它优于 top1 ⇒ 问题在选模不在轨迹 |
| `minADE@6` | 逐样本 6 选 1 | oracle 上界，**只作诊断** |

口径与 `eval_test_set.py` / J1 逐字一致：冻结 test、**batch=8**、位置域、
`follow_batch=['gnn_x']`（joint 必需）、架构由 sidecar 决定（不裸建）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_THIS = Path(__file__).resolve()
GNN_ROOT = _THIS.parents[2]
for _p in (str(GNN_ROOT), str(GNN_ROOT / "scripts" / "training"),
           str(GNN_ROOT.parent), str(GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402,F401
from src.shared.utils.bn_calibration import load_model_for_inference, read_architecture  # noqa: E402
from src.shared.utils.dataset import get_adjacency_config, set_subgraph_agg  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from magnet_agent_aux_dataset import AgentAuxGraphDataset  # noqa: E402
from magnet_joint_encoder import build_entity_index  # noqa: E402
from magnet_joint_decoder import FUTURE_FRAMES  # noqa: E402
from magnet_joint_loss import (MISS_THRESHOLD_M, MarginFunnelMonitor, _best_mode,  # noqa: E402
                               build_joint_targets_batch, mode_diversity, mode_usage,
                               per_mode_errors)

logger = logging.getLogger("evaljoint")

#: 闸的基准（`experiments/test_eval_20260818/fr_base_seed1/test_metrics.json`，同 test、同 batch=8）
GATE_REF = {"tag": "fr_base_seed1", "ade": 2.9251146017939376, "fde": 7.378103185605854}
#: 三 seed 均值与常速基线（`experiments/j1_constvel_20260825/test_frozen.json`）
FR_BASE_MEAN = {"ade": 2.8318, "fde": 7.0245}
CONST_VEL = {"ade": 2.8515, "fde": 7.8130}


def evaluate(ckpt: Path, test_root: Path, batch_size: int, device: str,
             ego_only: bool, limit_batches: int = 0) -> Dict[str, object]:
    sidecar = read_architecture(ckpt) or {}
    if not sidecar.get("joint_adaptive"):
        raise SystemExit(f"{ckpt} 的 sidecar 不是 joint_adaptive ⇒ 本脚本拒绝评（架构错配会污染读数）")
    set_subgraph_agg(sidecar.get("subgraph_agg") or "mean")
    meta = json.loads((ckpt.parent / "train_meta.json").read_text())
    # 🔴 A 臂（`--joint_ego_only`）用 **ego-only 监督**训练 ⇒ 周车头**从未被训练过**
    #    ⇒ agent/moving/static 三层的读数**没有意义**。指标函数是无状态的、不知道这件事，
    #    所以由本处（唯一知道 sidecar 的地方）显式标注，免得日后有人把 5.92 m 当成真读数引用。
    trained_ego_only = bool(meta.get("architecture", {}).get("joint_ego_only", False))
    dataset_config = meta.get("dataset_config") or "default"
    builder = get_adjacency_config(dataset_config)
    if dataset_config != "default" and type(builder).__name__ == "DefaultBuilder":
        raise SystemExit(f"邻接配置 {dataset_config!r} 未注册 ⇒ 拒绝评测")

    # E1：训练时挂了 oracle 门的 ckpt，**评测必须挂同一档门** —— 否则等于换了模型输入，
    # 读数既不是训练分布也不是部署分布。sidecar 里没有该字段 ⇒ 走原路径，逐位等价。
    oracle_gate = meta.get("oracle_gate")
    if oracle_gate:
        sys.path.insert(0, str(GNN_ROOT / "scripts" / "training"))
        from magnet_oracle_gate_dataset import OracleGateGraphDataset  # noqa: WPS433
        ds = OracleGateGraphDataset(str(test_root), transform=None,
                                    adjacency_config=dataset_config, variant=oracle_gate)
        logger.info("🔴 本 ckpt 训练时挂了 oracle 门 variant=%s ⇒ 评测同档挂载；"
                    "读数是**上界**不是可部署方法（标签用了 GT 锚定的未来信息）", oracle_gate)
    else:
        ds = AgentAuxGraphDataset(str(test_root), transform=None, adjacency_config=dataset_config)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, follow_batch=["gnn_x"])
    dev = torch.device(device)
    model = load_model_for_inference(ckpt, device=dev).to(dev).eval()
    n_modes = model.num_modes
    mon = MarginFunnelMonitor(model.active_head())

    strata = ("ego", "agent", "moving", "static", "all")
    sums: Dict[str, float] = {}
    cnts: Dict[str, float] = {}
    per_mode_sum = {s: torch.zeros(n_modes, dtype=torch.float64) for s in strata}
    ce_sum = 0.0
    winner_hist = torch.zeros(n_modes, dtype=torch.float64)
    n_ent_tot = 0
    # 🆕 逐时域曲线（ego，top1 口径）——旧模型实测 t=1 s 处是常速的 1.971 倍、
    #    要到 t=3.7 s 才反超（`j1b_horizon_curve.py`）。新结构有没有修好短时域，
    #    直接关系到「闭环靠短时域做控制」那条因果链。
    horizon_sum = torch.zeros(FUTURE_FRAMES, dtype=torch.float64)
    horizon_n = 0

    def add(key: str, val: float, w: float) -> None:
        sums[key] = sums.get(key, 0.0) + val * w
        cnts[key] = cnts.get(key, 0.0) + w

    t0 = time.time()
    with torch.no_grad():
        for bi, data in enumerate(loader):
            if limit_batches and bi >= limit_batches:
                break
            data = data.to(dev)
            idx = build_entity_index(data)
            tg = build_joint_targets_batch(data, idx, ego_only=ego_only)
            if not bool(tg.is_target.any()):
                continue
            pred = model(data, edge_gate=getattr(data, "oracle_gate", None))
            ade, fde = per_mode_errors(pred.traj, tg)
            top1 = pred.logits.argmax(dim=1)
            best_s = _best_mode(ade, tg, scene_level=True)
            sel = tg.is_target
            ce_sum += float(torch.nn.functional.cross_entropy(
                pred.logits[sel], best_s[sel], reduction="sum"))
            winner_hist += torch.bincount(best_s[sel].cpu(), minlength=n_modes).double()
            n_ent_tot += int(sel.sum())

            ego_sel = tg.is_target & tg.is_ego
            if bool(ego_sel.any()):
                rows_e = torch.nonzero(ego_sel, as_tuple=True)[0]
                sel_traj = pred.traj[rows_e, top1[rows_e]]                  # [n_ego, 60, 2]
                err_e = torch.norm(sel_traj - tg.tgt[rows_e], dim=-1)       # [n_ego, 60]
                horizon_sum += err_e.sum(dim=0).double().cpu()
                horizon_n += int(rows_e.numel())

            for sname, sub in tg.strata():
                n = int(sub.sum())
                if n == 0:
                    continue
                r = torch.arange(n, device=ade.device)
                a_s, f_s = ade[sub], fde[sub]
                add(f"{sname}.top1_ADE", float(a_s[r, top1[sub]].mean()), n)
                add(f"{sname}.top1_FDE", float(f_s[r, top1[sub]].mean()), n)
                add(f"{sname}.top1_missRate", float((f_s[r, top1[sub]] > MISS_THRESHOLD_M)
                                                    .float().mean()), n)
                add(f"{sname}.minADE", float(a_s.min(dim=1).values.mean()), n)
                add(f"{sname}.minFDE", float(f_s.min(dim=1).values.mean()), n)
                add(f"{sname}.minMissRate", float((f_s.min(dim=1).values > MISS_THRESHOLD_M)
                                                  .float().mean()), n)
                add(f"{sname}.scene_mstar_ADE", float(a_s[r, best_s[sub]].mean()), n)
                add(f"{sname}.hitRate", float((top1[sub] == a_s.argmin(dim=1)).float().mean()), n)
                per_mode_sum[sname] += a_s.sum(dim=0).double().cpu()
                add(f"{sname}.n_acc", float(n), n)
            # 逐层的 mode_usage / spread 必须按**该层**的实体数加权（与 run_epoch_joint 同规则）
            smap = dict(tg.strata())
            for k, v in {**mode_usage(pred, tg), **mode_diversity(pred, tg)}.items():
                pre = k.split("_", 1)[0]
                add(k, float(v), float(int(smap.get(pre, tg.is_target).sum())))
            if bi % 100 == 0:
                logger.info("… batch %d", bi)

    out: Dict[str, object] = {k: sums[k] / max(cnts[k], 1.0) for k in sums}
    for s in strata:
        n = cnts.get(f"{s}.n_acc", 0.0)
        if n:
            pm = (per_mode_sum[s] / n)
            out[f"{s}.per_mode_ADE"] = pm.tolist()
            out[f"{s}.best_fixed_ADE"] = float(pm.min())
            out[f"{s}.best_fixed_mode"] = int(pm.argmin())
            out[f"{s}.n"] = int(n)
    if horizon_n:
        hc = (horizon_sum / horizon_n)
        out["ego_horizon_curve"] = hc.tolist()
        # 🔴 索引 i ↔ t0 + i·0.1 s（首帧增量恒零、是锚点）⇒ 索引 9 是 **0.9 s** 不是 1.0 s。
        #    🔴 2026-08-31 更正：真实时刻是 **t0 + (i+1)·0.1 s**，故索引 59 = **6.0 s**。
        #    `process_nuplan_to_12d.py:97-116` 把第 0 个增量硬写成 [0,0] 再从 `list[1]` 起作差，
        #    而该列表第 k 项的时刻是 t0+(k+1)·0.1 s ⇒ cumsum[i] == pos((i+1)·0.1) − pos(0.1)
        #    （三 token 逐位验证 max|Δ| = 0.000000 m）。旧注「索引 59 = 5.9 s」是错的；
        #    我方一直报的「6 s」反而正确，**数字无需改动**。
        out["ego_horizon_at_s"] = {f"{(i + 1) * 0.1:.1f}": float(hc[i])
                                   for i in (9, 19, 29, 39, 49, 59)}
        out["horizon_index_note"] = ("索引 i ↔ t0 + (i+1)·0.1 s；索引 59 = 6.0 s（2026-08-31 查实更正，"
                                     "旧注「= 5.9 s」已作废）。空间锚点是 t0+0.1 s 的 ego rear_axle，"
                                     "非 t0 —— 头 0.1 s 位移已从每点整体减去")
    wf = winner_hist / max(winner_hist.sum(), 1.0)
    out.update({
        "checkpoint": str(ckpt), "test_root": str(test_root), "batch_size": batch_size,
        "dataset_config": dataset_config, "oracle_gate": oracle_gate,
        "n_entity": n_ent_tot, "num_modes": n_modes,
        "ego_only": ego_only, "trained_ego_only": trained_ego_only,
        "agent_metrics_meaningless": trained_ego_only,
        "agent_metrics_note": ("🔴 该 ckpt 用 ego-only 监督训练，周车头从未被训练 ⇒ "
                               "agent/moving/static 三层读数无意义，不得引用"
                               if trained_ego_only else ""),
        "eval_seconds": round(time.time() - t0, 1),
        "mode_ce": ce_sum / max(n_ent_tot, 1),
        "winner_dist": wf.tolist(),
        "winner_entropy": float(-(wf.clamp(min=1e-12) * wf.clamp(min=1e-12).log()).sum()),
        "ln_K": math.log(n_modes),
        "protocol": "saved-checkpoint 口径；冻结 test 仅用于最终 headline，不参与任何选择（R4-7）",
        **{f"margin.{k}": v for k, v in mon.summary().items()},
    })
    mon.close()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="S2 闸门：joint 模型在冻结 test 上的读数")
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--test_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=8,
                   help="🔴 冻结 test 口径 = 8，改它读数不可比")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ego_only", action="store_true", help="A 臂 ckpt（只监督 ego）")
    p.add_argument("--limit_batches", type=int, default=0, help="只跑前 N 批（冒烟用；0 = 全量）")
    a = p.parse_args()

    runs: List[Dict[str, object]] = []
    for c in a.checkpoint:
        logger.info("评 %s", c)
        runs.append(evaluate(Path(c).resolve(), Path(a.test_root).resolve(),
                             a.batch_size, a.device, a.ego_only, a.limit_batches))

    def m(key: str) -> float:
        vals = [float(r[key]) for r in runs if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    agg = {"n_ckpt": len(runs), "runs": runs, "gate_ref": GATE_REF,
           "fr_base_3seed_mean": FR_BASE_MEAN, "const_vel_baseline": CONST_VEL,
           "mean": {k: m(k) for k in ("ego.top1_ADE", "ego.top1_FDE", "ego.minADE", "ego.minFDE",
                                      "ego.best_fixed_ADE", "ego.hitRate",
                                      "agent.top1_ADE", "agent.minADE", "moving.top1_ADE",
                                      "static.top1_ADE", "mode_ce", "winner_entropy")}}
    gate_pass = agg["mean"]["ego.top1_ADE"] <= GATE_REF["ade"]
    agg["gate_top1_ego_ade_pass"] = bool(gate_pass)
    # 🔴 落盘挪到 main() 末尾 —— 逐时域聚合（`ego_horizon_curve_mean` / `crossover_frame_*`）
    #    是在下面的打印段才算出来的，先写会把它们漏掉（08-25 实测漏了）。

    print(f"\n=== S2 闸门读数 | {len(runs)} 个 ckpt | 冻结 test | batch={a.batch_size} ===")
    print(f"{'':34s} {'top1 ADE':>10s} {'top1 FDE':>10s} {'best_fixed':>11s} "
          f"{'minADE@6':>10s} {'hitRate':>9s}")
    for r in runs:
        cp = Path(str(r["checkpoint"]))
        # 🔴 带上文件名：`checkpoint_best`（按 minADE 选）与 `final_model_default`（末 epoch）
        #    是两个不同的口径，只打目录名两行会长得一模一样
        tag = f"{cp.parents[1].name}/{cp.stem[:12]}"
        print(f"{tag:34s} {r['ego.top1_ADE']:10.4f} {r['ego.top1_FDE']:10.4f} "
              f"{r['ego.best_fixed_ADE']:11.4f} {r['ego.minADE']:10.4f} {r['ego.hitRate']:9.4f}")
    mm = agg["mean"]
    print(f"{'均值':34s} {mm['ego.top1_ADE']:10.4f} {mm['ego.top1_FDE']:10.4f} "
          f"{mm['ego.best_fixed_ADE']:11.4f} {mm['ego.minADE']:10.4f} {mm['ego.hitRate']:9.4f}")
    print(f"\n{'闸基准 fr_base_seed1':24s} ADE {GATE_REF['ade']:.4f}  FDE {GATE_REF['fde']:.4f}")
    print(f"{'旧 MAGNET 三 seed 均值':24s} ADE {FR_BASE_MEAN['ade']:.4f}  FDE {FR_BASE_MEAN['fde']:.4f}")
    print(f"{'常速基线（最强 w=1）':24s} ADE {CONST_VEL['ade']:.4f}  FDE {CONST_VEL['fde']:.4f}")
    print(f"\n🔴 S2 前置健全性闸（top1 ego ADE ≤ {GATE_REF['ade']:.4f}）: "
          f"{'✅ 通过' if gate_pass else '❌ 未过 —— 按 Q110 §6.1 停下来查结构，不得进 S3'}")
    # 🔴 判据要算出来，不能写成固定句 —— 固定句在 CE < 熵 时会把结论说反。
    # 🔴 K=1 时更严重：CE / 赢家熵 / hitRate / best_fixed 四项**在数学上恒为平凡值**
    #    （CE=0、熵=0、hit=1、best_fixed≡top1），照原句会印出「置信度头不如最佳常数预测器」
    #    这种关于一个**根本不存在的头**的判词 ⇒ K=1 一律不打这两句，改打显式说明。
    n_modes_read = int(runs[0].get("num_modes", 0)) if runs else 0
    if n_modes_read <= 1:
        print(f"🔑 **单模态档（num_modes={n_modes_read}）：没有选模头** ⇒ mode CE / 赢家分布熵 / "
              f"hitRate / best_fixed 四项恒为平凡值（0 / 0 / 1 / ≡top1），**不作任何判读**。"
              f"\n   本档能回答的只有一件：去掉多模态之后，部署口径 top1 是多少。")
    else:
        ce, ent = mm["mode_ce"], mm["winner_entropy"]
        verdict = ("**不如**最佳常数预测器（无信息）" if ce >= ent
                   else f"略优于最佳常数预测器（低 {ent - ce:.4f} nats）")
        bf, t1 = mm["ego.best_fixed_ADE"], mm["ego.top1_ADE"]
        print(f"🔑 mode CE {ce:.4f} vs 赢家分布熵 {ent:.4f} ⇒ 置信度头{verdict}")
        print(f"🔑 best_fixed {bf:.4f} vs top1 {t1:.4f} ⇒ 恒取同一模态比用置信度头"
              f"{'**好**' if bf < t1 else '差'} {abs(bf - t1) / max(t1, 1e-9) * 100:.1f}%"
              f"{' ⇒ 选模头目前是净负担' if bf < t1 else ''}")
    if any(r.get("agent_metrics_meaningless") for r in runs):
        print(f"\n🔴🔴 周车读数**无意义、不得引用**：本批 ckpt 用 **ego-only 监督**训练"
              f"（sidecar `joint_ego_only=true`），周车头从未被训练过。")
        print(f"   （仅存证）top1 {mm['agent.top1_ADE']:.4f} / moving {mm['moving.top1_ADE']:.4f} "
              f"/ static {mm['static.top1_ADE']:.4f}")
    else:
        print(f"周车: top1 {mm['agent.top1_ADE']:.4f} / minADE {mm['agent.minADE']:.4f} "
              f"| moving top1 {mm['moving.top1_ADE']:.4f} | static top1 {mm['static.top1_ADE']:.4f}")
        print(f"   （常速基线 周车 0.8741 / moving 4.5696 / static 0.4809；"
              f"🔴 周车侧有约 0.15 m 契约地板）")
    # 🆕 逐时域对照：旧模型与常速的**完整 60 点曲线**读自 J1b 产物（不硬敲数字，规则 88）
    curves = [r["ego_horizon_curve"] for r in runs if "ego_horizon_curve" in r]
    ref_path = GNN_ROOT / "experiments/j1_constvel_20260825/horizon_curve_test.json"
    if curves and ref_path.exists():
        import statistics
        ref = json.loads(ref_path.read_text())
        old, cv = ref["model_curve_mean"], ref["cv_curve"]
        mc = [statistics.mean(c[i] for c in curves) for i in range(len(curves[0]))]
        agg["ego_horizon_curve_mean"] = mc
        print(f"\n=== 逐时域 ego 误差（top1 口径，{len(curves)} 个 ckpt 平均）===")
        print(f"{'t (s)':>6s} {'新结构':>9s} {'旧 MAGNET':>10s} {'常速':>9s} {'新/常速':>9s} {'新/旧':>8s}"
              f"   ← 🔴 t = 索引×0.1 s；末点 5.9 s 即项目惯例的「6 s / FDE」")
        for i_ in (9, 19, 29, 39, 49, 59):
            print(f"{i_ * 0.1:6.1f} {mc[i_]:9.4f} {old[i_]:10.4f} {cv[i_]:9.4f} "
                  f"{mc[i_] / cv[i_]:9.3f} {mc[i_] / old[i_]:8.3f}")
        xo_new = next((i_ for i_ in range(len(mc)) if mc[i_] < cv[i_]), None)
        xo_old = next((i_ for i_ in range(len(old)) if old[i_] < cv[i_]), None)
        agg["crossover_frame_new"] = xo_new
        agg["crossover_frame_old"] = xo_old
        f = lambda x: "从未反超" if x is None else f"t = {x * 0.1:.1f} s"  # noqa: E731
        print(f"  🔑 首次优于常速：**新结构 {f(xo_new)}** ｜ 旧 MAGNET {f(xo_old)}")
        print(f"     （短时域是闭环真正用到的一段 —— t=0.9 s 处旧模型是常速的 "
              f"{old[9] / cv[9]:.2f} 倍，新结构 {mc[9] / cv[9]:.2f} 倍）")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"落盘 {a.out}")


if __name__ == "__main__":
    main()
