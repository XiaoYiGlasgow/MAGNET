#!/usr/bin/env python
"""E1 —— 把 S3 的反事实边效用标签烘成**逐边 `edge_gate`** 缓存（oracle / shuffled / random 三档）。

## 它回答什么

Q108 §7 判据 3 逐字：「使用理想或 teacher topology 时，联合预测或下游规划是否存在足够的性能上界……
**如果第三个条件不成立，说明 topology 即使预测正确，也难以改善最终任务**」。

## 为什么它是 S4/S5 的**上界**（而不是又一个对照臂）

`EdgeScorer` 的输出是 `w = sigmoid(·) ∈ (0,1)`，而固定臂是 `w ≡ 1`
⇒ **gating 机制只能衰减交互、永远不能增强**。scorer 能达到的最好情况就是「边权完全正确」，
也就是本文件生成的 oracle 门。**若 oracle 都赢不了完全图，任何 scorer 都赢不了。**

## 三档（后两档是 oracle 的 placebo，Q108 §7 对照集逐字列有这两个名字）

| 变体 | `w` 怎么来 | 它排除什么 |
|---|---|---|
| `oracle` | 逐目标 min-max 把真 `U` 映到 `[W_FLOOR, 1.0]` | —— |
| `shuffled` | 同一目标内部把 `w` **随机置换** | 「是这条边该得高权」还是「只要权重分布对就行」 |
| `random` | 从全 split 的 oracle `w` 经验分布**独立重抽** | 「任何非均匀权重都行」 |

## 🔴 三处口径，改之前先读

1. **标签用 `U` 不用 `I`**：`I = [U − placebo]_+` 有 **47.9%** 被 clamp 到 0、**4.0%** 的目标全零
   （ego 层 13.2%），排序会大量并列；`U` 只有 0.48% 的目标无可排序信息。且 `U` 与 placebo 抽样
   完全无关（placebo 只进 `I`，实测它在同一目标内 43,646 组 **100% 逐位恒定**）。
2. **归一化在「目标内部」做**：`U` 是重尾的（全局 median 5.6e-5、p95 0.024、max 64.2），
   全局归一会被少数目标的离群值压平；逐目标 min-max 对目标间的尺度差免疫
   （实测组内极差中位 1.85e-02，量级良性）。
3. **下界 `W_FLOOR` 不取 0**：`w=0` 在 MP 里是**逐位删边**、在 decoder 里给 `log(1e-6)≈−13.8` 的
   近乎屏蔽 ⇒ 那是**硬改拓扑**，不是软加权，会把「边权」这个变量和「边数」混起来。
   取 0.05 让最没用的边仍在图上但被压到 5%。**这是预注册的选择，出数后不得改。**

## 对齐怎么保证

不靠约定，靠断言（与 `magnet_agent_aux_dataset` 同一条纪律）：
每条缓存带 `n_nodes` / `n_edges` / `gnn_x_sum` 三个指纹，读取时逐样本核对。
下标↔token 走 `joint_usage_probe.resolve_tokens`（它自带「文件数 ≠ 样本数就抛错」的守卫）。

跑法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/training/build_oracle_gate_cache.py \\
        --root  /abs/.../processed_features_fullret_L1_logsplit/train \\
        --labels /abs/.../data/edge_labels_s3_20260826/train \\
        --variant oracle
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_GNN_ROOT / "scripts" / "analysis"),
           str(_DEVKIT), str(_DEVKIT / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("build_oracle_gate")

VARIANTS = ("oracle", "shuffled", "random")
#: 🔴 预注册：最没用的边压到 5%，**不压到 0**（0 = 硬删边，会把「边权」与「边数」两个变量混起来）。
W_FLOOR = 0.05
CACHE_NAME = "oracle_gate_{variant}_{cfg}.pt"


class OracleGateError(RuntimeError):
    """缓存构建期的任何不一致 —— 一律抛出，绝不静默降级。"""


def cache_path(root, variant: str, adjacency_config: str = "default") -> Path:
    return (Path(root) / "processed" / f"config_{adjacency_config}"
            / CACHE_NAME.format(variant=variant, cfg=adjacency_config))


def _load_labels(label_root: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
    """读 S3 合并产物，返回 `{token: {src, dst, U}}` 与 meta。"""
    meta = json.loads((label_root / "meta.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(label_root.glob("labels_*.npz")):
        with np.load(f) as z:
            ptr = np.asarray(z["scene_ptr"])
            toks = [str(t) for t in z["token"]]
            src, dst, U = np.asarray(z["src"]), np.asarray(z["dst"]), np.asarray(z["U"])
            for i, tk in enumerate(toks):
                sl = slice(int(ptr[i]), int(ptr[i + 1]))
                out[tk] = {"src": src[sl].copy(), "dst": dst[sl].copy(), "U": U[sl].copy()}
    if not out:
        raise OracleGateError(f"{label_root} 下没有可读的 labels_*.npz")
    return out, meta


def _oracle_w(src: np.ndarray, dst: np.ndarray, U: np.ndarray) -> np.ndarray:
    """逐目标 min-max 把 `U` 映到 `[W_FLOOR, 1.0]`。目标内全等 ⇒ 全给 1.0（无信息，不惩罚）。"""
    w = np.ones(U.size, dtype=np.float32)
    order = np.argsort(dst, kind="stable")
    d = dst[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, d.size]):
        sel = order[lo:hi]
        u = U[sel]
        umin, umax = float(u.min()), float(u.max())
        if umax - umin <= 0.0:
            w[sel] = 1.0
        else:
            w[sel] = (W_FLOOR + (1.0 - W_FLOOR) * (u - umin) / (umax - umin)).astype(np.float32)
    return w


def _permute_within_target(w: np.ndarray, dst: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """`shuffled` 档：同一目标内部把 `w` 随机置换（边际分布逐目标不变，分配全错）。"""
    out = w.copy()
    order = np.argsort(dst, kind="stable")
    d = dst[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, d.size]):
        sel = order[lo:hi]
        out[sel] = w[sel][rng.permutation(sel.size)]
    return out


#: 把 `(src, dst)` 打成一个 int64 键。`dst` 最大实测 64、实体上限远小于 KEY_BASE ⇒ 不会撞键。
KEY_BASE = 1_000_000


def _match_edges(ei: np.ndarray, src: np.ndarray, dst: np.ndarray):
    """向量化地把 `gnn_edge_index` 的列匹配到标签行。返回 `(cols, rows)` 两个等长下标数组。

    🔴 不用 Python 字典逐边查 —— train 上是 20936 场景 × 7289 边 ≈ **1.5 亿次**查表。
    这里走 `searchsorted`：O(E log L)，实测把 train 从数小时压到分钟级。
    """
    lab_key = src.astype(np.int64) * KEY_BASE + dst.astype(np.int64)
    order = np.argsort(lab_key, kind="stable")
    lab_sorted = lab_key[order]
    e_key = ei[0].astype(np.int64) * KEY_BASE + ei[1].astype(np.int64)
    pos = np.searchsorted(lab_sorted, e_key)
    pos_c = np.clip(pos, 0, lab_sorted.size - 1)
    hit = lab_sorted[pos_c] == e_key
    cols = np.flatnonzero(hit)
    rows = order[pos_c[cols]]
    return cols, rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="E1：把 S3 边效用标签烘成逐边 edge_gate 缓存")
    p.add_argument("--root", required=True, help="split leaf root（含 aux 缓存）")
    p.add_argument("--labels", required=True, help="S3 合并产物目录（含 labels_*.npz + meta.json）")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    import config  # noqa: F401
    from src.shared.utils.dataset import get_adjacency_config, set_subgraph_agg
    from magnet_agent_aux_dataset import AgentAuxGraphDataset
    from joint_usage_probe import resolve_tokens

    cp = cache_path(a.root, a.variant, a.adjacency_config)
    if cp.exists() and not a.force:
        logger.info("已存在，跳过：%s（要重建加 --force）", cp)
        return

    b = get_adjacency_config(a.adjacency_config)
    if a.adjacency_config != "default" and type(b).__name__ == "DefaultBuilder":
        raise OracleGateError(f"邻接配置 {a.adjacency_config!r} 未注册 ⇒ 会静默回退 default，拒绝继续")
    set_subgraph_agg("mean")
    ds = AgentAuxGraphDataset(a.root, adjacency_config=a.adjacency_config)
    toks = resolve_tokens(a.root, len(ds))
    labels, lmeta = _load_labels(Path(a.labels))
    logger.info("dataset %d 样本 | S3 标签 %d 场景 | variant=%s | teacher×%s | err=%s",
                len(ds), len(labels), a.variant, lmeta.get("n_teachers"), lmeta.get("err_metric"))

    rng = np.random.default_rng(a.seed)
    # ── 第一遍：算 oracle 门，同时收集全局 w 池（`random` 档要用）─────────────
    entries: List[Dict] = []
    pool: List[np.ndarray] = []
    n_hit = n_miss = 0
    n_edge_tot = n_lab_tot = 0
    for i in range(len(ds)):
        d = ds[i]
        ei = d.gnn_edge_index.numpy()
        e = int(ei.shape[1])
        gate = np.ones(e, dtype=np.float32)
        lab = labels.get(toks[i])
        n_lab = 0
        if lab is not None and lab["src"].size:
            w = _oracle_w(lab["src"], lab["dst"], lab["U"])
            cols, rows = _match_edges(ei, lab["src"], lab["dst"])
            gate[cols] = w[rows]          # 命中的边取 oracle w，其余（lane / 未打标）恒 1.0
            n_lab = int(cols.size)
            pool.append(w)
            n_hit += 1
        else:
            n_miss += 1
        n_edge_tot += e
        n_lab_tot += n_lab
        entries.append({"gate": gate, "n_nodes": int(d.gnn_x.shape[0]), "n_edges": e,
                        "gnn_x_sum": float(d.gnn_x.sum()), "n_labelled": n_lab,
                        "token": toks[i]})
        if (i + 1) % 2000 == 0:
            logger.info("  %d/%d  命中 %d  未打标 %d", i + 1, len(ds), n_hit, n_miss)

    # ── 第二遍：按变体改写 ───────────────────────────────────────────────────
    if a.variant != "oracle":
        big = np.concatenate(pool) if pool else np.array([1.0], dtype=np.float32)
        for i, ent in enumerate(entries):
            lab = labels.get(ent["token"])
            if lab is None or not lab["src"].size:
                continue
            ei = ds[i].gnn_edge_index.numpy()
            cc, kk = _match_edges(ei, lab["src"], lab["dst"])
            if cc.size == 0:
                continue
            if a.variant == "shuffled":
                w_full = _oracle_w(lab["src"], lab["dst"], lab["U"])
                w_new = _permute_within_target(w_full, lab["dst"], rng)
                ent["gate"][cc] = w_new[kk]
            else:                                  # random：从全局经验分布独立重抽
                ent["gate"][cc] = big[rng.integers(0, big.size, size=cc.size)]

    gates = np.concatenate([e["gate"] for e in entries])
    lab_mask = gates < 1.0                                   # 近似：被打标且非满权的边
    payload = {
        "entries": [{"gate": torch.from_numpy(e["gate"]), "n_nodes": e["n_nodes"],
                     "n_edges": e["n_edges"], "gnn_x_sum": e["gnn_x_sum"],
                     "n_labelled": e["n_labelled"]} for e in entries],
        "stats": {"n_samples": len(entries), "n_with_labels": n_hit, "n_without_labels": n_miss,
                  "n_edges_total": n_edge_tot, "n_edges_labelled": n_lab_tot,
                  "frac_edges_labelled": n_lab_tot / max(n_edge_tot, 1),
                  "w_mean_labelled": float(gates[lab_mask].mean()) if lab_mask.any() else 1.0,
                  "w_median_labelled": float(np.median(gates[lab_mask])) if lab_mask.any() else 1.0,
                  "w_mean_all": float(gates.mean())},
        "meta": {"variant": a.variant, "w_floor": W_FLOOR, "seed": a.seed,
                 "label_root": str(Path(a.labels).resolve()),
                 "label_meta": {k: lmeta.get(k) for k in
                                ("version", "teacher_sha256", "err_metric", "dose", "placebo",
                                 "n_teachers", "n_scenes_labelled", "n_pairs")},
                 "adjacency_config": a.adjacency_config,
                 "normalization": "per-target min-max of U -> [W_FLOOR, 1.0]",
                 "label_field": "U",
                 "caveats": [
                     "🔴 oracle 用了 GT 锚定标签（离线才算得出）⇒ 它是**上界**，不是可部署方法",
                     "🔴 未打标的边（lane 相关 / 非 scorable）恒 1.0，与固定臂一致",
                     "🔴 W_FLOOR=0.05 是预注册选择：不取 0 以免把「边权」变成「删边」"]},
    }
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cp)
    s = payload["stats"]
    logger.info("✅ 落盘 %s", cp)
    logger.info("   %d 样本｜有标签 %d / 无标签 %d｜打标边占比 %.2f%%｜"
                "被打标边的 w: mean %.4f median %.4f｜全边 w mean %.4f",
                s["n_samples"], s["n_with_labels"], s["n_without_labels"],
                100 * s["frac_edges_labelled"], s["w_mean_labelled"],
                s["w_median_labelled"], s["w_mean_all"])


if __name__ == "__main__":
    main()
