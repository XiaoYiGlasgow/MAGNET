"""G1 前置 —— **退化探针**：切边响应能不能被一个无需学习的闭式量复现。

## 为什么必须先跑这个

主档 MAGNET 的图是**完全图**（`adjacency_config=default`，E = C(n,2)），GNN 分支是
`GCNConv → ReLU → GCNConv → global_mean_pool`（`models/transformer_predictor.py:18-58`）。
D-1 已实测：完全图上 `GCNConv` **精确退化为 `MLP(mean)`**（max|Δ| 2.608e-08）——
带自环的对称归一化让每个节点的聚合都等于全场景均值，故 layer-1 输出在所有节点上完全相同。

在这个前提下，切掉无向边 (ego, j) 之后：

* ``d̂_ego`` 与 ``d̂_j`` 从 n 降到 n−1，其余节点的度不变；
* ego 的聚合失去 x_j 这一项，即偏移量 ∝ ``x̄ − x_j``；
* j 的聚合失去 x_ego 这一项，即偏移量 ∝ ``x̄ − x_ego`` —— **对所有 j 是同一个常数**；
* 其余节点因邻居度改变而产生的偏移，j-依赖同样只经由 ``x_j``。

⇒ 固定场景后，**响应对 j 的依赖只经过 ``x_j`` 这一条通路**。

## 🔴 一处必须纠正的前提（助手 2026-08-22 核出）

「响应无法依赖 (i,j) 的关系」这个说法**不准确**：图是 **ego-centric 坐标系**，`x_j`
本身就编码了 j 相对 ego 的位置/朝向/速度 ⇒ 对 **j→ego** 这类边，退化并没有抹掉关系，
它约束的是**函数形态**（j 只能经由「与场景均值的偏差」进入 layer-1 的线性聚合）。

所以真正可证伪、且决定 G1 判读的问题是**经验问题**：

> 实测的切边响应对 j 的排序，能不能被一个**不需要学习的闭式标量**复现？

若能（本探针的 ρ ≈ 1），那么 G3 里「学出来的 scorer」学到的只是一个公式，
把它称作 counterfactual interaction utility 属于过度声称；G1 即使通过也只能报为
「通道携带 GT 相关信息」的下界，不能报为交互效用。

## 口径

* **响应 = `d2_semantic_intervention.response()` 的 `mean_disp`**（与 F 门A / 闸⑥ 同源，
  batch=1 部署口径，不换度量）；相对量 `rel_disp = mean_disp / base_scale`。
* **塌陷场景必须分开**：塌陷时任何干预的 response 恰好为 0 ⇒ 排序退化、ρ 无定义。
  本脚本同时报**冻结口径**（剔除塌陷场景）与**敏感性口径**（保留），两个都写进产物。
* **不确定性按 log-cluster 算**，不按场景数算（`token → log` 只认
  `experiments/scaling_ladder/s4_log_manifest_from_s1.json`）。

## 🔴 U（GT 锚定效用）只落盘、不聚合

`response()` 顺带返回 `delta_ade`，它正是 Q109 §3.3.2 的 U_{j→ego}。
**G1 的判据尚未预注册冻结**，此刻读 U 的分布会污染阈值的选取（方法学第七十九条邻域）。
⇒ 本脚本把 U **逐条写进产物**，但 summary 里**不做任何 U 的统计**，除非显式传
`--reveal_u`（那意味着判据已冻结，且必须在冻结件里注明先看过分布）。
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
    response,
    split_directed_half,
)
from scripts.analysis.run_interaction_audit import (  # noqa: E402
    _clone_with,
    _load_feature_df,
    _load_model,
    _node_types_from_df,
    _probe_collapsed_head,
)
from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402

logger = logging.getLogger("g1_degeneracy_probe")

#: 产物契约版本 —— 写进每个 JSON，便于事后对账。
PROBE_VERSION = "G1_DEGENERACY_PROBE_v1_20260822"

#: token → log 的唯一正确来源（§8.2；别的 manifest 覆盖不全）。
LOG_MANIFEST = _GNN_ROOT / "experiments" / "scaling_ladder" / "s4_log_manifest_from_s1.json"

#: 计算逐场景 Spearman 至少需要几个 agent —— 少于这个数排序统计没有意义。
MIN_AGENTS_FOR_RHO = 4

#: 缓存基线等价性的相对阈值。取 1e-5：比实测 GPU 底噪 max 5.5e-07 高 18 倍（不误杀），
#: 比单边 placebo 的 2.4e-05 低 2.4 倍（真出错时仍会被拦）。
CACHE_EQUIV_TOL = 1e-5

#: `POLYLINE_FEATURES` 第 2:4 列是向量终点 (x, y)（`magnet_agent_aux_wiring._agent_end_xy`
#: 的同源事实）；池化后 `gnn_x[:, 2:4]` 即该子图终点的均值，用作「到 ego 的距离」代理量。
XY_COLS = slice(2, 4)


class ProbeError(RuntimeError):
    """探针前提不成立 —— 必须暴露，不得静默降级成一个假读数。"""


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（并列取平均秩）。任一侧无变异时返回 nan，不返回 0。

    自己实现是为了不引入 scipy 依赖，且让「无变异 ⇒ nan」这一分支显式可见 ——
    塌陷场景的响应全为 0，若返回 0 会被误读成「不相关」。
    """
    if a.size != b.size or a.size < 3:
        return float("nan")
    ra, rb = _avg_rank(a), _avg_rank(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """并列取平均秩。"""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(v.size, dtype=float)
    # 处理并列：同值取平均秩
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if uniq.size < v.size:
        sums = np.zeros(uniq.size, dtype=float)
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return ranks


def _cluster_bootstrap_median(values: np.ndarray, clusters: np.ndarray,
                              n_boot: int, boot_seed: int) -> Tuple[float, float]:
    """按 log 重抽的 bootstrap 95% CI（中位数统计量）。

    与 `paired_usage_delta.cluster_bootstrap_ci` 同口径 —— 同一 log 内的场景不独立，
    按样本数算 CI 在本数据上会窄 4–12 倍（口径纪律）。
    """
    if values.size == 0:
        return float("nan"), float("nan")
    uniq = np.unique(clusters)
    if uniq.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(boot_seed)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        stats[b] = np.median(values[np.concatenate([idx[c] for c in pick])])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _load_token_to_log() -> Dict[str, str]:
    """读 token → log_name 映射。缺失即抛错 —— 没有它就只能按样本数算 CI，那是错的。"""
    if not LOG_MANIFEST.exists():
        raise ProbeError(f"缺 log manifest：{LOG_MANIFEST}")
    with open(LOG_MANIFEST, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return {r["token"]: r["log_name"] for r in rows}


def _cut_undirected_pair(edge_index: torch.Tensor, u: int, v: int) -> torch.Tensor:
    """删掉无向边 (u, v) 的**两个方向**，并保持「前半正向 + 后半反向」的存储约定。

    :raises InterventionError: 存储约定不成立，或该图里根本没有这条边。
    """
    fwd, e = split_directed_half(edge_index)
    if e == 0:
        raise InterventionError("图无边")
    hit = ((fwd[0] == u) & (fwd[1] == v)) | ((fwd[0] == v) & (fwd[1] == u))
    n_hit = int(hit.sum())
    if n_hit == 0:
        raise InterventionError(f"图中不存在无向边 ({u}, {v})")
    keep = ~hit
    new_fwd = fwd[:, keep]
    return torch.cat([new_fwd, new_fwd.flip(0)], dim=1)


def _cut_one_column(edge_index: torch.Tensor, col: int) -> torch.Tensor:
    """按**正向半边的列号**删掉一条无向边的两个方向（placebo 用，避开端点查找）。"""
    fwd, e = split_directed_half(edge_index)
    if not 0 <= col < e:
        raise InterventionError(f"列号 {col} 越界（E={e}）")
    keep = torch.ones(e, dtype=torch.bool)
    keep[col] = False
    new_fwd = fwd[:, keep]
    return torch.cat([new_fwd, new_fwd.flip(0)], dim=1)


def _rewire(data, edge_index: torch.Tensor):
    """换 `gnn_edge_index` 并同步 `gnn_edge_attr` 长度（与 audit 的 `_rewire_data` 同源）。"""
    return _clone_with(data, gnn_edge_index=edge_index,
                       gnn_edge_attr=data.gnn_edge_attr.new_zeros(edge_index.shape[1]))


def _forward_inc(model, data, device: torch.device) -> torch.Tensor:
    """batch=1 部署口径的一次前向 —— 与 `d2_semantic_intervention.response()` 内部逐字同源。"""
    from torch_geometric.data import Batch
    return model(Batch.from_data_list([data]).to(device))[:1].cpu().reshape(-1, 2)


def _response_from_base(inc_a: torch.Tensor, inc_b: torch.Tensor, y) -> Dict[str, float]:
    """把 `response()` 的度量部分抽出来，让基线前向可以**每场景只算一次**。

    🔴 与 `response()` 逐字同式（`d2_semantic_intervention.py:328-341`）。

    ⚠️ **GPU 上「同一输入两次前向」并不逐位相同** —— `GCNConv` 的 `scatter_add` 走原子加，
    顺序不定。2026-08-22 实测底噪（40 场景，cuda:0）：相对位移 median **9.3e-09**、
    p90 1.3e-07、**max 5.5e-07**；CPU 上恒为 **0**。本探针的信号量级是 1e-4，
    比该底噪高 **两个数量级**，故缓存基线安全；但等价性检查必须按底噪定阈，
    用严格相等会误杀场景（首版正是如此，train 上误杀 8 个）。
    """
    pos_a, pos_b = inc_a.cumsum(0), inc_b.cumsum(0)
    disp = (pos_a - pos_b).norm(dim=1)
    out = {
        "mean_disp": float(disp.mean()),
        "max_disp": float(disp.max()),
        "mean_abs_increment_diff": float((inc_a - inc_b).abs().mean()),
        "base_scale": float(pos_a.norm(dim=1).mean()),
    }
    if y is not None:
        gt = y.reshape(-1, 2).cumsum(0)
        out["delta_ade"] = float((pos_b - gt).norm(dim=1).mean() - (pos_a - gt).norm(dim=1).mean())
    return out


@torch.no_grad()
def probe_one_scene(model, data, node_types: Sequence[str], is_ego: np.ndarray,
                    device: torch.device, *, require_complete: bool = True,
                    verify_cache: bool = False, placebo_seed: int = 0,
                    cache_checks: Optional[List[float]] = None) -> Optional[Dict]:
    """对单个场景跑完整探针。

    :param node_types: 归一后的节点类型（ego 已被并进 `'agent'`），与 `gnn_x` 同序。
        🔴 **必须用它把 lane 节点排除**：新档每场景约 68 个 lane 节点、仅约 15 个 agent，
        「非 ego 节点」= agent + lane，直接当 j 会把 j→ego 的读数稀释成 lane→ego 为主。
    :returns: dict；`skip` 键存在表示该场景不可用。
    """
    n_nodes = int(data.gnn_x.shape[0])
    if is_ego.size != n_nodes or len(node_types) != n_nodes:
        raise ProbeError(f"节点标注长度({len(node_types)}/{is_ego.size}) != 图节点数 {n_nodes} "
                         f"—— 节点序已错位")
    ego_idx = np.flatnonzero(is_ego)
    if ego_idx.size != 1:
        return {"skip": f"ego 节点数 {ego_idx.size} != 1"}
    ego = int(ego_idx[0])
    agents = [i for i in range(n_nodes)
              if i != ego and str(node_types[i]).strip().lower() == "agent"]
    n_lane = sum(1 for t in node_types if str(t).strip().lower() == "lane")
    if len(agents) < MIN_AGENTS_FOR_RHO:
        return {"skip": f"非 ego 的 agent 节点仅 {len(agents)} 个（< {MIN_AGENTS_FOR_RHO}）",
                "n_nodes": n_nodes, "n_lane": n_lane}

    try:
        _, e_directed = split_directed_half(data.gnn_edge_index)
    except InterventionError as exc:
        return {"skip": f"边存储约定不成立：{exc}"}
    e_complete = n_nodes * (n_nodes - 1) // 2
    is_complete = (e_directed == e_complete)
    if require_complete and not is_complete:
        return {"skip": f"非完全图（E={e_directed} != C(n,2)={e_complete}）—— "
                        f"退化结论只对完全图成立"}

    collapsed = _probe_collapsed_head(model, data, device)

    # conv1 的真实输入：LayerNorm(9) 后的 9 维核心特征（逐节点，无 batch 依赖）
    x_norm = model._extract_core_features(data.gnn_x.to(device)).detach().cpu().numpy()
    x_raw = data.gnn_x[:, :9].detach().cpu().numpy()
    xy = data.gnn_x[:, XY_COLS].detach().cpu().numpy()

    mu_norm, mu_raw = x_norm.mean(axis=0), x_raw.mean(axis=0)
    ego_xy = xy[ego]

    # conv1 的线性映射 —— 解析推导说 layer-1 的扰动形如 `W₁ᵀ(x̄ − x_j)`，
    # 只用 ‖x̄ − x_j‖ 会忽略 W₁ 对 9 个维度的不等加权，是一个**偏弱**的解析代理量。
    w1 = model.gnn_encoder.conv1.lin.weight.detach().cpu().numpy()   # [64, 9]

    if cache_checks is None:
        cache_checks = []
    inc_base = _forward_inc(model, data, device)      # 🔑 每场景只算一次
    y = getattr(data, "y", None)

    recs: List[Dict] = []
    for k, j in enumerate(agents):
        try:
            cut = _cut_undirected_pair(data.gnn_edge_index, ego, j)
        except InterventionError as exc:
            recs.append({"j": j, "skip": str(exc)})
            continue
        data_cut = _rewire(data, cut)
        r = _response_from_base(inc_base, _forward_inc(model, data_cut, device), y)
        if verify_cache and k == 0:
            ref = response(model, data, data_cut, device)
            scale = max(float(ref["base_scale"]), 1e-12)
            worst = max(abs(r[kk] - ref[kk]) / scale for kk in ref)
            cache_checks.append(worst)
            if worst > CACHE_EQUIV_TOL:
                raise ProbeError(
                    f"缓存基线与 response() 的相对差 {worst:.3g} 超过阈 {CACHE_EQUIV_TOL:.3g}"
                    f"（实测 GPU 底噪 max 5.5e-07）—— 不是非确定性能解释的量级")
        base_scale = float(r["base_scale"])
        recs.append({
            "j": j,
            "mean_disp": float(r["mean_disp"]),
            "max_disp": float(r["max_disp"]),
            "base_scale": base_scale,
            # 相对量：与 §8 使用度 U = mean_disp / base_scale 同定义
            "rel_disp": float(r["mean_disp"] / base_scale) if base_scale > 0 else float("nan"),
            # 🔴 U（GT 锚定效用）：只落盘，summary 不统计（见模块 docstring）
            "delta_ade_DO_NOT_READ_BEFORE_PREREG": float(r.get("delta_ade", float("nan"))),
            # 闭式代理量（无需学习）
            "s_dev_norm": float(np.linalg.norm(x_norm[j] - mu_norm)),
            "s_dev_raw": float(np.linalg.norm(x_raw[j] - mu_raw)),
            "s_proj_w1": float(np.linalg.norm(w1 @ (mu_norm - x_norm[j]))),
            "d_ego": float(np.linalg.norm(xy[j] - ego_xy)),
        })

    # ---- 单边 placebo 底噪：同数量、来源池换成「不与 ego 相连的边」 -------------------
    # 🔴 这是**模型锚定的诊断量 S**，不是 G1 的 U —— 它只回答「单边切除的响应量级
    #    有没有超出任意单边扰动的底噪」，用来判读上面 `rel_disp` 那个 1e-4 是否有意义。
    fwd_all, e_all = split_directed_half(data.gnn_edge_index)
    not_ego = np.flatnonzero(((fwd_all[0] != ego) & (fwd_all[1] != ego)).cpu().numpy())
    placebo: List[Dict] = []
    if not_ego.size > 0:
        rng = np.random.default_rng(placebo_seed)
        for col in rng.choice(not_ego, size=min(len(agents), not_ego.size), replace=False):
            cut_p = _cut_one_column(data.gnn_edge_index, int(col))
            rp = _response_from_base(inc_base, _forward_inc(model, _rewire(data, cut_p), device), y)
            bs = float(rp["base_scale"])
            placebo.append({
                "col": int(col),
                "mean_disp": float(rp["mean_disp"]),
                "rel_disp": float(rp["mean_disp"] / bs) if bs > 0 else float("nan"),
                "delta_ade_DO_NOT_READ_BEFORE_PREREG": float(rp.get("delta_ade", float("nan"))),
            })

    good = [r for r in recs if "skip" not in r]
    if len(good) < MIN_AGENTS_FOR_RHO:
        return {"skip": f"可用 agent 仅 {len(good)} 个"}

    resp = np.array([r["mean_disp"] for r in good], dtype=float)
    out = {
        "n_nodes": n_nodes,
        "n_lane": n_lane,
        "n_agents": len(good),
        "e_directed": int(e_directed),
        "is_complete_graph": bool(is_complete),
        "collapsed": collapsed,
        "resp_all_zero": bool(np.all(resp == 0.0)),
        "rho_s_dev_norm": _spearman(resp, np.array([r["s_dev_norm"] for r in good])),
        "rho_s_dev_raw": _spearman(resp, np.array([r["s_dev_raw"] for r in good])),
        "rho_s_proj_w1": _spearman(resp, np.array([r["s_proj_w1"] for r in good])),
        "rho_d_ego": _spearman(resp, np.array([r["d_ego"] for r in good])),
        # 🔑 响应本身的量级 —— 若切一条 j→ego 边几乎不动轨迹，G1 会因为「没东西可监督」
        #    这种无聊原因失败，与「算子退化」是两回事，必须分开看。口径 = §8 的 U = mean_disp/base_scale。
        "rel_disp_median": float(np.median([r["rel_disp"] for r in good])),
        "rel_disp_max": float(np.max([r["rel_disp"] for r in good])),
        "base_scale": float(np.median([r["base_scale"] for r in good])),
        "edges": recs,
        "placebo": placebo,
    }
    if placebo:
        pl = np.array([p["rel_disp"] for p in placebo], dtype=float)
        tr = np.array([r["rel_disp"] for r in good], dtype=float)
        out["placebo_rel_disp_median"] = float(np.median(pl))
        # 处理臂中位 − placebo 中位（逐场景配对，聚合时按 log-cluster 求 CI）
        out["rel_disp_minus_placebo"] = float(np.median(tr) - np.median(pl))
        out["n_placebo"] = int(pl.size)
    return out


def _iter_scenes(data_root: Path, n_scenes: int, seed: int) -> List[Path]:
    """固定 seed 抽 n 个 pkl —— 与全仓抽样惯例一致（`random.Random(seed).shuffle`）。"""
    files = sorted(data_root.glob("**/features_*.pkl"))
    if not files:
        raise ProbeError(f"{data_root} 下没有 features_*.pkl")
    random.Random(seed).shuffle(files)
    return files[:n_scenes]


def _summarize(scenes: List[Dict], token_to_log: Dict[str, str],
               n_boot: int, boot_seed: int) -> Dict:
    """两口径聚合：冻结（剔塌陷）+ 敏感性（保留塌陷）。"""
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
            return {"n": 0, "median": None, "ci95": [None, None], "n_logs": 0}
        lo, hi = _cluster_bootstrap_median(arr, cl, n_boot, boot_seed)
        return {
            "n": int(arr.size),
            "n_logs": int(np.unique(cl).size),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "frac_ge_0.9": float((arr >= 0.9).mean()),
            "frac_ge_0.7": float((arr >= 0.7).mean()),
            "ci95_log_cluster": [lo, hi],
        }

    usable = [s for s in scenes if "skip" not in s]
    frozen = [s for s in usable if s.get("collapsed") is not True and not s.get("resp_all_zero")]
    keys = ("rho_s_dev_norm", "rho_s_dev_raw", "rho_s_proj_w1", "rho_d_ego",
            "rel_disp_median", "placebo_rel_disp_median", "rel_disp_minus_placebo",
            "rel_disp_max", "n_agents")
    return {
        "n_scenes_loaded": len(scenes),
        "n_scenes_usable": len(usable),
        "n_scenes_collapsed": sum(1 for s in usable if s.get("collapsed") is True),
        "n_scenes_resp_all_zero": sum(1 for s in usable if s.get("resp_all_zero")),
        "n_scenes_frozen": len(frozen),
        "frozen": {k: _agg(frozen, k) for k in keys},
        "sensitivity_keep_collapsed": {k: _agg(usable, k) for k in keys},
        "U_note": "delta_ade（GT 锚定效用 U）已逐条落盘但**未做任何统计** —— "
                  "G1 判据尚未预注册冻结，先看分布会污染阈值选取。",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 前置：切边响应的退化探针")
    p.add_argument("--ckpt", required=True, type=Path, help="calibrated checkpoint")
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n_scenes", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boot_seed", type=int, default=20260822)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--gnn_pooling_fix", action="store_true", default=True)
    p.add_argument("--no_gnn_pooling_fix", dest="gnn_pooling_fix", action="store_false")
    p.add_argument("--allow_incomplete_graph", action="store_true",
                   help="允许非完全图场景参与（默认拒绝：退化结论只对完全图成立）")
    p.add_argument("--reveal_u", action="store_true",
                   help="🔴 只有在 G1 判据已冻结后才允许 —— 会把 U 的统计写进 summary")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.reveal_u:
        logger.warning("🔴 --reveal_u 已开：这意味着 G1 判据必须已经冻结，且冻结件里要注明先看过 U 分布")

    device = torch.device(args.device)
    model = _load_model(args.ckpt, args.gnn_pooling_fix, device)
    model.eval()
    token_to_log = _load_token_to_log()

    files = _iter_scenes(args.data_root, args.n_scenes, args.seed)
    logger.info("ckpt=%s device=%s 抽到 %d 个场景", args.ckpt.name, device, len(files))

    scenes: List[Dict] = []
    verified = False          # 首个可用场景上实测「缓存基线 ≈ response()」，通过后不再重复
    cache_checks: List[float] = []
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
            reason = built[1] if isinstance(built, tuple) else "build 返回 None"
            scenes.append({"token": token, "skip": f"建图失败：{reason}"})
            continue
        types = _node_types_from_df(df)
        if types is None:
            scenes.append({"token": token, "skip": "节点类型还原失败"})
            continue
        node_types, is_ego, _ = types
        try:
            rec = probe_one_scene(model, data, node_types, is_ego, device,
                                  require_complete=not args.allow_incomplete_graph,
                                  verify_cache=(not verified),
                                  placebo_seed=args.boot_seed + k,
                                  cache_checks=cache_checks)
        except (ProbeError, InterventionError) as exc:
            rec = {"skip": str(exc)}
        if rec and "skip" not in rec:
            verified = True
        rec = {"token": token, "log": token_to_log.get(token), **(rec or {"skip": "None"})}
        scenes.append(rec)
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
        "adjacency_config": args.adjacency_config,
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),
        "sampling": {"n_requested": args.n_scenes, "seed": args.seed},
        "metric": "response.mean_disp（batch=1 部署口径，与 F 门A / 闸⑥ 同源）",
        "surrogates": {
            "s_dev_norm": "‖LayerNorm(x_j)[:9] − 场景均值‖₂ —— conv1 的真实输入空间",
            "s_dev_raw": "‖x_j[:9] − 场景均值‖₂ —— 未归一化空间",
            "s_proj_w1": "‖W₁(场景均值 − LayerNorm(x_j))‖₂ —— 计入 conv1 线性映射的解析代理量（更强）",
            "d_ego": "‖gnn_x[j, 2:4] − gnn_x[ego, 2:4]‖₂ —— 到 ego 的距离（池化终点近似）",
        },
        "cache_equivalence": {
            "tol_relative": CACHE_EQUIV_TOL,
            "observed_worst_relative": (max(cache_checks) if cache_checks else None),
            "n_checks": len(cache_checks),
            "gpu_nondeterminism_floor_measured_20260822": {
                "device": "cuda:0", "n_scenes": 40,
                "median": 9.32e-09, "p90": 1.26e-07, "max": 5.52e-07,
                "cpu_max": 0.0,
                "note": "同一输入两次前向的 rel_disp；信号 1e-4 高出两个数量级",
            },
        },
        "summary": summary,
        "per_scene": scenes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("写出 %s（用时 %.0f s）", args.out, time.time() - t0)
    fz = summary["frozen"]
    logger.info("冻结口径 n_scenes=%d / n_logs=%s", summary["n_scenes_frozen"],
                fz["rho_s_dev_norm"]["n_logs"])
    for key in ("rho_s_dev_norm", "rho_s_dev_raw", "rho_s_proj_w1", "rho_d_ego",
                "rel_disp_median", "placebo_rel_disp_median", "rel_disp_minus_placebo",
                "rel_disp_max", "n_agents"):
        a = fz[key]
        if a["median"] is None:
            logger.info("  %-16s (无可用样本)", key)
            continue
        logger.info("  %-24s n=%-4d median %.6g  CI95 [%.6g, %.6g]",
                    key, a["n"], a["median"], a["ci95_log_cluster"][0], a["ci95_log_cluster"][1])


if __name__ == "__main__":
    main()
