"""干预审计的聚合与统计 —— 把 `run_interaction_audit.py` 的 shard 产物变成可判读的结论。

统计合同全部来自冻结协议 `干预协议_INTERVENTION_PROTOCOL_v1_20260731.md` §7–§8，
本脚本**不引入任何协议之外的自由度**：

* 聚合顺序：**先在 token 内对 seed 取中位数**，再跨 token 统计（seed 只是「选中哪几个实体」
  的随机性，不是独立样本）；
* 配对量：``S_i = R_i(agent) − R_i(lane)``，同 token / 同 checkpoint / 同剂量 / 同算子；
* 分层：primary 只在 **non-collapsed** 子集上算（塌陷场景任何干预 response 恒为 0，
  混进来会把 estimand 稀释成假的零效应）；
* 点估计用**中位数**（分布右偏），区间用**按 log 聚类的 bootstrap**（val 60 个 log）；
* 检验：单边 Wilcoxon signed-rank（预注册方向 agent ≥ lane）+ sign test 稳健性，两侧 p 一并报；
* 多重性：两个 primary（`count_sat` 与 `frac_25` 的 `remove` 算子）做 **Holm(2)**；
  `count_1` 与 `zero` 算子为 exploratory，**不进 Holm、不作判定**。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/agg_interaction_audit.py \
        --audit_dir experiments/interaction_audit \
        --configs B_seed42_calibrated,B_seed42_raw,B_seed1_raw,B_seed2_raw \
        --log_manifest data/processed_features_magnet_e1_none_v3/log_manifest.json \
        --out experiments/interaction_audit/summary_20260731.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, wilcoxon

logger = logging.getLogger("agg_audit")

TARGET, CONTROL = "agent", "lane"
#: 预注册的两个 primary（协议 §8）：都是 `remove` 算子，分别对应两个 matched 口径。
PRIMARY_CELLS: Tuple[Tuple[str, str], ...] = (("count_sat", "remove"), ("frac_25", "remove"))
#: exploratory：报数但不进 Holm、不作判定。
EXPLORATORY_CELLS: Tuple[Tuple[str, str], ...] = (
    ("count_sat", "zero"), ("count_1", "remove"), ("count_1", "zero"), ("frac_25", "zero"),
)


def _load_records(config_dir: Path) -> List[Dict]:
    """读一个 config 目录下全部 shard 产物；缺 shard 只警告不静默丢。"""
    records: List[Dict] = []
    shards = sorted(config_dir.glob("shard_*.json"))
    for path in shards:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        records.extend(payload["records"])
    logger.info("%s: %d shard，%d 条记录", config_dir.name, len(shards), len(records))
    return records


def _cell_value(rec: Dict, dose: str, op: str, side: str) -> Optional[float]:
    """取一条记录在 (剂量, 算子, 侧) 上的 `mean_disp`；被 skip 的格返回 None。"""
    dd = rec.get("doses", {}).get(dose)
    if not isinstance(dd, dict) or "skipped" in dd:
        return None
    cell = dd.get(f"{op}_{side}")
    if not isinstance(cell, dict) or "mean_disp" not in cell:
        return None
    return float(cell["mean_disp"])


def _by_token(records: Sequence[Dict], dose: str, op: str) -> Dict[str, Dict]:
    """token 级聚合：对 seed 取中位数，同时带出分层所需的元信息。"""
    buf: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"t": [], "c": []})
    meta: Dict[str, Dict] = {}
    for rec in records:
        t = _cell_value(rec, dose, op, TARGET)
        c = _cell_value(rec, dose, op, CONTROL)
        if t is None or c is None:
            continue
        token = rec["token"]
        buf[token]["t"].append(t)
        buf[token]["c"].append(c)
        meta.setdefault(token, {
            "collapsed": bool(rec.get("collapsed_head")),
            "n_agent_noego": rec.get("n_agent_noego"),
            "n_lane": rec.get("n_lane"),
            "k_target": rec["doses"][dose].get("k_target"),
            "k_control": rec["doses"][dose].get("k_control"),
            "frac_target": rec["doses"][dose].get("frac_target"),
            "frac_control": rec["doses"][dose].get("frac_control"),
        })
    return {
        token: {
            "target": float(np.median(v["t"])),
            "control": float(np.median(v["c"])),
            "S": float(np.median(v["t"]) - np.median(v["c"])),
            "n_seeds": len(v["t"]),
            **meta[token],
        }
        for token, v in buf.items()
    }


def _cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int,
                          rng: np.random.Generator) -> Tuple[float, float]:
    """按 cluster（log）重抽的 bootstrap 95% CI —— 同一 log 内的场景不是独立样本。"""
    uniq = np.unique(clusters)
    if uniq.size < 2:
        return float("nan"), float("nan")
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sample = np.concatenate([idx_by_cluster[c] for c in pick])
        stats[b] = np.median(values[sample])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _test_cell(tokens: Dict[str, Dict], token_to_log: Dict[str, str],
               n_boot: int, rng: np.random.Generator) -> Dict:
    """一个 (剂量, 算子) 格的完整读数：分层计数 + 点估计 + CI + 两个检验。"""
    non = {k: v for k, v in tokens.items() if not v["collapsed"]}
    col = {k: v for k, v in tokens.items() if v["collapsed"]}
    out: Dict = {
        "n_tokens_total": len(tokens),
        "n_non_collapsed": len(non),
        "n_collapsed": len(col),
        "collapse_rate": round(len(col) / len(tokens), 4) if tokens else None,
    }
    if not non:
        out["skipped"] = "无 non-collapsed token"
        return out

    keys = sorted(non)
    s = np.array([non[k]["S"] for k in keys])
    tgt = np.array([non[k]["target"] for k in keys])
    ctl = np.array([non[k]["control"] for k in keys])
    logs = np.array([token_to_log.get(k, f"__unknown_{k}") for k in keys])

    lo, hi = _cluster_bootstrap_ci(s, logs, n_boot, rng)
    nz = s[s != 0]
    out.update({
        "median_target": float(np.median(tgt)),
        "median_control": float(np.median(ctl)),
        "median_S": float(np.median(s)),
        "ci95_S_logcluster": [lo, hi],
        "n_logs": int(np.unique(logs).size),
        "mean_frac_target": float(np.mean([non[k]["frac_target"] for k in keys])),
        "mean_frac_control": float(np.mean([non[k]["frac_control"] for k in keys])),
        "mean_k_target": float(np.mean([non[k]["k_target"] for k in keys])),
        "mean_k_control": float(np.mean([non[k]["k_control"] for k in keys])),
        "n_S_positive": int((s > 0).sum()),
        "n_S_negative": int((s < 0).sum()),
        "n_S_zero": int((s == 0).sum()),
        # 相对效应量：S 除以基线轨迹尺度的中位数量级，避免只报绝对米数
        "median_S_over_target": float(np.median(s) / np.median(tgt)) if np.median(tgt) else None,
    })
    if nz.size >= 10:
        out["wilcoxon_p_one_sided_greater"] = float(
            wilcoxon(nz, alternative="greater", zero_method="wilcox").pvalue)
        out["wilcoxon_p_two_sided"] = float(
            wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue)
        out["sign_test_p_one_sided_greater"] = float(
            binomtest(int((nz > 0).sum()), nz.size, 0.5, alternative="greater").pvalue)
    else:
        out["test_skipped"] = f"非零配对差只有 {nz.size} 个，不足以检验"
    return out


def _holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """Holm–Bonferroni（只覆盖 primary family）。"""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict] = {}
    prev = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[name] = {"p_raw": p, "p_holm": adj, "reject_at_0.05": adj < alpha}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="干预审计聚合与统计（协议 v1 §7–§8）")
    p.add_argument("--audit_dir", required=True, type=Path)
    p.add_argument("--configs", required=True,
                   help="逗号分隔的 config 子目录名；第一个视为 primary config")
    p.add_argument("--log_manifest", required=True, type=Path,
                   help="token→log 映射（cluster bootstrap 用）")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--boot_seed", type=int, default=20260731)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    a = parse_args()
    rng = np.random.default_rng(a.boot_seed)

    with open(a.log_manifest, encoding="utf-8") as f:
        token_to_log = {e["token"]: e["log_name"] for e in json.load(f)}

    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    summary: Dict[str, Dict] = {}
    for name in configs:
        cfg_dir = a.audit_dir / name
        if not cfg_dir.is_dir():
            logger.warning("跳过不存在的 config 目录：%s", cfg_dir)
            continue
        records = _load_records(cfg_dir)
        if not records:
            continue
        cells: Dict[str, Dict] = {}
        pvals: Dict[str, float] = {}
        for dose, op in list(PRIMARY_CELLS) + list(EXPLORATORY_CELLS):
            tokens = _by_token(records, dose, op)
            if not tokens:
                cells[f"{dose}.{op}"] = {"skipped": "该格全部 token 被 skip"}
                continue
            res = _test_cell(tokens, token_to_log, a.n_boot, rng)
            res["role"] = "primary" if (dose, op) in PRIMARY_CELLS else "exploratory"
            cells[f"{dose}.{op}"] = res
            if (dose, op) in PRIMARY_CELLS and "wilcoxon_p_one_sided_greater" in res:
                pvals[f"{dose}.{op}"] = res["wilcoxon_p_one_sided_greater"]

        n_tok = len({r["token"] for r in records})
        n_col = len({r["token"] for r in records if r.get("collapsed_head")})
        summary[name] = {
            "n_tokens": n_tok,
            "n_collapsed_tokens": n_col,
            "collapse_rate": round(n_col / n_tok, 4) if n_tok else None,
            "max_noop_disp": max(float(r.get("noop_disp") or 0.0) for r in records),
            "cells": cells,
            "holm_primary": _holm(pvals) if pvals else {},
        }

    payload = {
        "protocol": "INTERVENTION_PROTOCOL_v1_20260731",
        "audit_dir": str(a.audit_dir),
        "configs": configs,
        "primary_cells": [f"{d}.{o}" for d, o in PRIMARY_CELLS],
        "n_boot": a.n_boot, "boot_seed": a.boot_seed,
        "note": "S = response(agent) − response(lane)，同 token 配对；primary 只在 non-collapsed 上算；"
                "count-matched 与 fraction-matched 并列 primary，判读规则见协议 §3.2",
        "summary": summary,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for name, s in summary.items():
        print(f"\n=== {name} ===  n_token={s['n_tokens']}  塌陷率={s['collapse_rate']}  "
              f"noop_max={s['max_noop_disp']:.2e}")
        for cell, r in s["cells"].items():
            if "median_S" not in r:
                print(f"  {cell:20s} {r.get('skipped') or r.get('test_skipped')}")
                continue
            ci = r["ci95_S_logcluster"]
            p = r.get("wilcoxon_p_one_sided_greater")
            print(f"  {cell:20s} [{r['role'][:5]}] agent={r['median_target']:.4f} "
                  f"lane={r['median_control']:.4f} S={r['median_S']:+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
                  f"frac {r['mean_frac_target']:.2f}/{r['mean_frac_control']:.2f} "
                  f"p1={p if p is None else round(p, 6)} n={r['n_non_collapsed']}")
        if s["holm_primary"]:
            print("  Holm(primary):", json.dumps(s["holm_primary"], ensure_ascii=False))
    logger.info("写入 %s", a.out)


if __name__ == "__main__":
    main()
