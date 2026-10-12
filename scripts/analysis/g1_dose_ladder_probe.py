"""G1 前置②-b —— **剂量阶梯**：1.6 mm 到底是「通道没负载」还是「完全图上的 1/n 稀释」。

## 背景

`g1_degeneracy_probe.py`（阶段297）实测：切**一条** j→ego 边只让 6 s 预测轨迹动
**1.6–1.8 mm**（`rel_disp` 中位 8.5e-5 ~ 1.0e-4，base_scale 18–21 m），
比同数量单边 placebo 高约 3×、配对差 CI 排 0、两 split 复现。

信号方向对但量级极小。**两种完全不同的解释，处置方式相反**：

* **(甲) 剂量问题**：完全图上每个节点有 n−1≈82 个邻居，ego 的聚合失去 x_j 只占约 1/82，
  池化再摊到约 83 个节点 ⇒ 单边效应结构上就是 O(1/n²)。若如此，**逐边定义在这个 teacher 上
  是选错了剂量**，加大剂量应当看到近似成比例的增长。
* **(乙) 通道确实没负载**：加大剂量也不涨 ⇒ 与 F 门A（ΔU_F 仅 +0.0787 pt）、
  Q74 G2（使用度 0.72%）一致，路线 B 的 teacher 必须换。

判别方法就是把剂量拉上去看响应怎么变。

## 剂量阶梯（同一个目标 agent j，同一个基线前向）

======  ==========================================  ==================  ============================
剂量    干预                                        切掉的边数          count-matched 对照
======  ==========================================  ==================  ============================
`D1`    只切 (ego, j) 一条无向边                    1                   `P1` 随机一条 agent–agent 边
                                                                        （**不含 ego**，用来排除
                                                                        「ego 是特征离群点」这一
                                                                        竞争解释 —— 阶段297 遗留项）
`D2c`   切 j 与**全部车辆节点**（ego + 其他 agent）  n_agent−1           `P2c` 对随机 lane 节点做同样的
        之间的边，保留 j–lane 边                                        「只切它与车辆的边」
`D2`    孤立 j：切它的**全部**边（保留节点）        n−1                 `P2` 孤立一个随机 lane 节点
                                                                        （完全图上每个节点度相同
                                                                        ⇒ 天然逐条匹配）
`D3`    **删掉 j 节点本身**（R3-M2 字面要求的       n−1 边 + 1 节点      `P3` 删一个随机 lane 节点
        vehicle removal）
======  ==========================================  ==================  ============================

🔑 **为什么 `D2c` 必须单列**：`D2` 把 j–lane 边也切了，那不是交互通道；`D2c` 才是
Q88 枢纽洞察意义上的「留车切边」在单个 agent 上的版本。

🔴 **完全图让 count-matching 变得干净**：实测 E = C(n,2)，每个节点度恒为 n−1 ⇒
「孤立一个 lane 节点」与「孤立一个 agent 节点」切掉的边数**恒等**，不需要再解释 budget。

## 口径

响应度量、batch=1 部署口径、塌陷分离、log-cluster CI 全部沿用 `g1_degeneracy_probe`，
不新造口径。U（`delta_ade`）同样**只落盘不聚合** —— G1 判据尚未预注册冻结。

## 🔴 空操作校验

每个场景先跑一次「走完整条重建流水线但一个节点/一条边都不动」的空操作，位移必须恰好 0。
非 0 说明重建路径把图改坏了，而本轮**每一条** response 都会被同一份偏移污染
（`run_interaction_audit._temporal_noop_disp` 的同款防线，第十条的邻域）。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent), str(_GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.d2_semantic_intervention import (  # noqa: E402
    InterventionError,
    intervene_remove_nodes,
    split_directed_half,
)
from scripts.analysis.g1_degeneracy_probe import (  # noqa: E402
    MIN_AGENTS_FOR_RHO,
    ProbeError,
    _cluster_bootstrap_median,
    _forward_inc,
    _load_token_to_log,
    _response_from_base,
    _rewire,
)
from scripts.analysis.run_interaction_audit import (  # noqa: E402
    _clone_with,
    _load_feature_df,
    _load_model,
    _node_types_from_df,
    _probe_collapsed_head,
)
from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402

logger = logging.getLogger("g1_dose_ladder")

PROBE_VERSION = "G1_DOSE_LADDER_v1_20260822"

#: 剂量名 → 对应的 count-matched 对照名。产物与聚合都按这张表走，避免两处分叉。
DOSE_TO_CONTROL = {"D1": "P1", "D2c": "P2c", "D2": "P2", "D3": "P3"}

#: 空操作位移的容忍上限。重建路径应当恒等 ⇒ 恰好 0；GPU 非确定性底噪实测 max 5.5e-07
#: （阶段297），故留 1e-6 的余量，超了直接抛错而不是记个字段。
NOOP_TOL = 1e-6


def _cut_edges_between(edge_index: torch.Tensor, node: int,
                       others: Sequence[int]) -> Tuple[torch.Tensor, int]:
    """切掉 `node` 与 `others` 之间的全部无向边，保持成对存储约定。

    :returns: `(新 edge_index, 切掉的无向边数)`
    """
    fwd, e = split_directed_half(edge_index)
    if e == 0:
        raise InterventionError("图无边")
    other = torch.as_tensor(list(others), dtype=fwd.dtype)
    hit = (((fwd[0] == node) & torch.isin(fwd[1], other))
           | ((fwd[1] == node) & torch.isin(fwd[0], other)))
    n_cut = int(hit.sum())
    if n_cut == 0:
        raise InterventionError(f"节点 {node} 与给定集合之间没有边")
    new_fwd = fwd[:, ~hit]
    return torch.cat([new_fwd, new_fwd.flip(0)], dim=1), n_cut


def _isolate(edge_index: torch.Tensor, node: int, n_nodes: int) -> Tuple[torch.Tensor, int]:
    """孤立一个节点：切掉它的全部边（节点保留）。"""
    return _cut_edges_between(edge_index, node, [i for i in range(n_nodes) if i != node])


def _remove_node(data, node: int):
    """删掉一个节点并重建图 —— 与 `run_interaction_audit._apply_node_ops` 的 remove 分支同源。"""
    gnn_x, edge_index = intervene_remove_nodes(data.gnn_x, data.gnn_edge_index,
                                               np.array([node], dtype=np.int64))
    return _clone_with(data, gnn_x=gnn_x, gnn_edge_index=edge_index,
                       gnn_edge_attr=data.gnn_edge_attr.new_zeros(edge_index.shape[1]))


def _noop_check(model, data, inc_base: torch.Tensor, device: torch.device) -> float:
    """走完整条重建流水线但**什么都不删**，返回相对位移；调用方按 `NOOP_TOL` 判。"""
    gnn_x, edge_index = intervene_remove_nodes(data.gnn_x, data.gnn_edge_index,
                                               np.array([], dtype=np.int64))
    noop = _clone_with(data, gnn_x=gnn_x, gnn_edge_index=edge_index,
                       gnn_edge_attr=data.gnn_edge_attr.new_zeros(edge_index.shape[1]))
    r = _response_from_base(inc_base, _forward_inc(model, noop, device), None)
    bs = max(float(r["base_scale"]), 1e-12)
    return float(r["mean_disp"]) / bs


@torch.no_grad()
def probe_one_scene(model, data, node_types: Sequence[str], is_ego: np.ndarray,
                    device: torch.device, *, seed: int = 0,
                    max_agents: Optional[int] = None) -> Dict:
    """对单个场景跑完整剂量阶梯。

    :param max_agents: 每场景最多测几个 agent（None = 全测）。剂量阶梯每个 j 要 7 次前向，
        场景里 agent 多时成本线性上升，抽样只影响样本量不影响口径。
    """
    n_nodes = int(data.gnn_x.shape[0])
    if is_ego.size != n_nodes or len(node_types) != n_nodes:
        raise ProbeError(f"节点标注长度 != 图节点数 {n_nodes} —— 节点序已错位")
    ego_idx = np.flatnonzero(is_ego)
    if ego_idx.size != 1:
        return {"skip": f"ego 节点数 {ego_idx.size} != 1"}
    ego = int(ego_idx[0])
    types = [str(t).strip().lower() for t in node_types]
    agents = [i for i in range(n_nodes) if i != ego and types[i] == "agent"]
    lanes = [i for i in range(n_nodes) if types[i] == "lane"]
    if len(agents) < MIN_AGENTS_FOR_RHO:
        return {"skip": f"非 ego 的 agent 节点仅 {len(agents)} 个"}
    if not lanes:
        return {"skip": "无 lane 节点 ⇒ 无法构造 count-matched 对照"}

    _, e_directed = split_directed_half(data.gnn_edge_index)
    if e_directed != n_nodes * (n_nodes - 1) // 2:
        return {"skip": f"非完全图（E={e_directed}）"}

    collapsed = _probe_collapsed_head(model, data, device)
    inc_base = _forward_inc(model, data, device)
    y = getattr(data, "y", None)

    noop_rel = _noop_check(model, data, inc_base, device)
    if noop_rel > NOOP_TOL:
        raise ProbeError(f"空操作位移 {noop_rel:.3g} > {NOOP_TOL:.3g} —— 重建路径把图改坏了，"
                         f"本轮所有 response 都会被同一份偏移污染")

    rng = np.random.default_rng(seed)
    picked = list(agents)
    if max_agents is not None and len(picked) > max_agents:
        picked = sorted(rng.choice(picked, size=max_agents, replace=False).tolist())
    vehicles = [ego] + agents

    def _meas(interv) -> Dict[str, float]:
        r = _response_from_base(inc_base, _forward_inc(model, interv, device), y)
        bs = max(float(r["base_scale"]), 1e-12)
        return {"mean_disp": float(r["mean_disp"]),
                "rel_disp": float(r["mean_disp"]) / bs,
                "delta_ade_DO_NOT_READ_BEFORE_PREREG": float(r.get("delta_ade", float("nan")))}

    # agent–agent 边池（不含 ego）—— P1 的来源池，排除「ego 是特征离群点」这一竞争解释
    fwd_all, _ = split_directed_half(data.gnn_edge_index)
    a_set = torch.as_tensor(agents, dtype=fwd_all.dtype)
    aa_cols = np.flatnonzero((torch.isin(fwd_all[0], a_set)
                              & torch.isin(fwd_all[1], a_set)).cpu().numpy())

    recs: List[Dict] = []
    for j in picked:
        rec: Dict = {"j": j}
        others_veh = [v for v in vehicles if v != j]
        try:
            # ---- 处理臂 ------------------------------------------------------------
            e_d1, n1 = _cut_edges_between(data.gnn_edge_index, j, [ego])
            rec["D1"] = {**_meas(_rewire(data, e_d1)), "n_edges_cut": n1}

            e_d2c, n2c = _cut_edges_between(data.gnn_edge_index, j, others_veh)
            rec["D2c"] = {**_meas(_rewire(data, e_d2c)), "n_edges_cut": n2c}

            e_d2, n2 = _isolate(data.gnn_edge_index, j, n_nodes)
            rec["D2"] = {**_meas(_rewire(data, e_d2)), "n_edges_cut": n2}

            rec["D3"] = {**_meas(_remove_node(data, j)), "n_edges_cut": n2, "n_nodes_removed": 1}

            # ---- count-matched 对照 -------------------------------------------------
            if aa_cols.size > 0:
                col = int(rng.choice(aa_cols))
                u, v = int(fwd_all[0, col]), int(fwd_all[1, col])
                e_p1, m1 = _cut_edges_between(data.gnn_edge_index, u, [v])
                rec["P1"] = {**_meas(_rewire(data, e_p1)), "n_edges_cut": m1,
                             "pool": "agent-agent 边（不含 ego）"}

            ln = int(rng.choice(lanes))
            e_p2c, m2c = _cut_edges_between(data.gnn_edge_index, ln, vehicles)
            rec["P2c"] = {**_meas(_rewire(data, e_p2c)), "n_edges_cut": m2c, "control_node": ln}

            e_p2, m2 = _isolate(data.gnn_edge_index, ln, n_nodes)
            rec["P2"] = {**_meas(_rewire(data, e_p2)), "n_edges_cut": m2, "control_node": ln}

            rec["P3"] = {**_meas(_remove_node(data, ln)), "n_edges_cut": m2,
                         "n_nodes_removed": 1, "control_node": ln}
        except InterventionError as exc:
            rec["skip"] = str(exc)
        recs.append(rec)

    good = [r for r in recs if "skip" not in r]
    if len(good) < MIN_AGENTS_FOR_RHO:
        return {"skip": f"可用 agent 仅 {len(good)} 个"}

    out: Dict = {
        "n_nodes": n_nodes, "n_agents_total": len(agents), "n_agents_probed": len(good),
        "n_lane": len(lanes), "e_directed": int(e_directed),
        "collapsed": collapsed, "noop_rel_disp": noop_rel,
        "per_agent": recs,
    }
    # 逐场景中位数（配对差在场景内做，聚合时按 log-cluster 求 CI）
    for dose, ctrl in DOSE_TO_CONTROL.items():
        t = np.array([r[dose]["rel_disp"] for r in good if dose in r], dtype=float)
        c = np.array([r[ctrl]["rel_disp"] for r in good if ctrl in r], dtype=float)
        if t.size:
            out[f"{dose}_rel_median"] = float(np.median(t))
            out[f"{dose}_n_edges_cut"] = float(np.median(
                [r[dose]["n_edges_cut"] for r in good if dose in r]))
        if t.size and c.size:
            out[f"{ctrl}_rel_median"] = float(np.median(c))
            out[f"{dose}_minus_{ctrl}"] = float(np.median(t) - np.median(c))
    # 剂量比：D1 是分母，回答「加大剂量涨了几倍」
    d1 = out.get("D1_rel_median")
    if d1 and d1 > 0:
        for dose in ("D2c", "D2", "D3"):
            if f"{dose}_rel_median" in out:
                out[f"ratio_{dose}_over_D1"] = float(out[f"{dose}_rel_median"] / d1)
    out["resp_all_zero"] = bool(d1 == 0.0)
    return out


def _summarize(scenes: List[Dict], token_to_log: Dict[str, str],
               n_boot: int, boot_seed: int) -> Dict:
    def _agg(rows: List[Dict], key: str) -> Dict:
        vals, logs = [], []
        for r in rows:
            v = r.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
            logs.append(token_to_log.get(r["token"], f"__unknown_{r['token']}"))
        arr, cl = np.array(vals, dtype=float), np.array(logs)
        if arr.size == 0:
            return {"n": 0, "median": None, "ci95_log_cluster": [None, None], "n_logs": 0}
        lo, hi = _cluster_bootstrap_median(arr, cl, n_boot, boot_seed)
        return {"n": int(arr.size), "n_logs": int(np.unique(cl).size),
                "median": float(np.median(arr)), "mean": float(arr.mean()),
                "ci95_log_cluster": [lo, hi]}

    usable = [s for s in scenes if "skip" not in s]
    frozen = [s for s in usable if s.get("collapsed") is not True and not s.get("resp_all_zero")]
    keys: List[str] = []
    for dose, ctrl in DOSE_TO_CONTROL.items():
        keys += [f"{dose}_rel_median", f"{ctrl}_rel_median", f"{dose}_minus_{ctrl}",
                 f"{dose}_n_edges_cut"]
    keys += [f"ratio_{d}_over_D1" for d in ("D2c", "D2", "D3")]
    keys += ["n_agents_total", "n_nodes"]
    return {
        "n_scenes_loaded": len(scenes),
        "n_scenes_usable": len(usable),
        "n_scenes_collapsed": sum(1 for s in usable if s.get("collapsed") is True),
        "n_scenes_frozen": len(frozen),
        "noop_rel_disp_max": max([s.get("noop_rel_disp", 0.0) for s in usable], default=None),
        "frozen": {k: _agg(frozen, k) for k in keys},
        "sensitivity_keep_collapsed": {k: _agg(usable, k) for k in keys},
        "U_note": "delta_ade（GT 锚定效用 U）逐条落盘但未做任何统计 —— G1 判据尚未预注册冻结。",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 前置：剂量阶梯探针")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n_scenes", type=int, default=300)
    p.add_argument("--max_agents", type=int, default=8,
                   help="每场景最多测几个 agent（每个 j 要 7 次前向）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boot_seed", type=int, default=20260822)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--gnn_pooling_fix", action="store_true", default=True)
    p.add_argument("--no_gnn_pooling_fix", dest="gnn_pooling_fix", action="store_false")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    model = _load_model(args.ckpt, args.gnn_pooling_fix, device)
    model.eval()
    token_to_log = _load_token_to_log()

    files = sorted(args.data_root.glob("**/features_*.pkl"))
    if not files:
        raise ProbeError(f"{args.data_root} 下没有 features_*.pkl")
    random.Random(args.seed).shuffle(files)
    files = files[:args.n_scenes]
    logger.info("ckpt=%s device=%s 抽到 %d 个场景（每场景最多 %d 个 agent）",
                args.ckpt.name, device, len(files), args.max_agents)

    scenes: List[Dict] = []
    t0 = time.time()
    for k, f in enumerate(files, 1):
        token = f.stem.replace("features_", "")
        df = _load_feature_df(f)
        if df is None:
            scenes.append({"token": token, "skip": "pkl 读取失败"})
            continue
        built = build_graph_data_from_feature_df(df, args.adjacency_config)
        data = built[0] if isinstance(built, tuple) else built
        if data is None:
            scenes.append({"token": token, "skip": "建图失败"})
            continue
        types = _node_types_from_df(df)
        if types is None:
            scenes.append({"token": token, "skip": "节点类型还原失败"})
            continue
        node_types, is_ego, _ = types
        try:
            rec = probe_one_scene(model, data, node_types, is_ego, device,
                                  seed=args.boot_seed + k, max_agents=args.max_agents)
        except (ProbeError, InterventionError) as exc:
            rec = {"skip": str(exc)}
        scenes.append({"token": token, "log": token_to_log.get(token), **rec})
        if k % 25 == 0:
            logger.info("  %d/%d  用时 %.0f s", k, len(files), time.time() - t0)

    summary = _summarize(scenes, token_to_log, args.n_boot, args.boot_seed)
    payload = {
        "contract": PROBE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "ckpt": str(args.ckpt),
        "model_class": type(model).__name__,
        "data_root": str(args.data_root),
        "doses": {
            "D1": "只切 (ego, j) 一条边 | 对照 P1 = 随机一条 agent–agent 边（不含 ego）",
            "D2c": "切 j 与全部车辆节点之间的边，保留 j–lane | 对照 P2c = 对随机 lane 做同样操作",
            "D2": "孤立 j（切它全部边，保留节点） | 对照 P2 = 孤立随机 lane 节点",
            "D3": "删掉 j 节点本身 | 对照 P3 = 删随机 lane 节点",
        },
        "summary": summary,
        "per_scene": scenes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("写出 %s（用时 %.0f s）", args.out, time.time() - t0)
    fz = summary["frozen"]
    logger.info("冻结口径 n_scenes=%d / n_logs=%s  空操作最大位移 %.3g",
                summary["n_scenes_frozen"], fz["D1_rel_median"]["n_logs"],
                summary["noop_rel_disp_max"] or 0.0)
    for dose, ctrl in DOSE_TO_CONTROL.items():
        t, c, d = fz[f"{dose}_rel_median"], fz[f"{ctrl}_rel_median"], fz[f"{dose}_minus_{ctrl}"]
        ne = fz[f"{dose}_n_edges_cut"]["median"]
        if t["median"] is None:
            continue
        logger.info("  %-4s 切%-5s边  处理 %.4g | 对照(%s) %.4g | 配对差 %.4g  CI [%.4g, %.4g]",
                    dose, f"{ne:.0f}", t["median"], ctrl, c["median"],
                    d["median"], d["ci95_log_cluster"][0], d["ci95_log_cluster"][1])
    for dose in ("D2c", "D2", "D3"):
        a = fz[f"ratio_{dose}_over_D1"]
        if a["median"] is not None:
            logger.info("  倍率 %s/D1 = %.2f×  CI [%.2f, %.2f]",
                        dose, a["median"], a["ci95_log_cluster"][0], a["ci95_log_cluster"][1])


if __name__ == "__main__":
    main()
