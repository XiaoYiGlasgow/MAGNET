#!/usr/bin/env python
"""S3 标签的读数报告 —— 出数后**先看这三件**再动 S4。

1. **标签有没有信号**：`I > 0` 的比例、`U` 的分布（按 `dst_stratum` 分层）。
   🔴 §8.2 红线：不分层的读数一律不许用 —— 静止/截断周车占绝大多数，会主导任何聚合量。
2. **「模型用的 ≠ 真正有用的」成不成立**：`rho(U, S)`（GT 锚定 vs 模型锚定，Q109 §3.3.3）
   全局与逐层。🔴 **两者的一致性是诊断量，不得预设必然一致**。
3. **teacher 分歧有多大**：`U_sd_teacher` 的分布 —— 它量的是「换个 seed 当老师，标签会差多少」，
   直接决定单 teacher 的方案会有多危险。

另外报 S4 设计要用的一条：**within-target 排序对 placebo 噪声免疫**（placebo 是逐目标的公共偏移），
所以同时给「全局 rho」与「逐目标 rho」两个口径。

跑法::

    python scripts/analysis/s3_report.py --root data/edge_labels_s3_20260825/train --out <json>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger("s3_report")
STRATA = {0: "ego", 1: "moving_agent", 2: "static_or_truncated"}


def _load(root: Path) -> Dict[str, np.ndarray]:
    fs = sorted(root.glob("labels_*.npz"))
    if not fs:
        raise SystemExit(f"🔴 {root} 下没有 labels_*.npz（先跑 s3_merge_parts.py）")
    out: Dict[str, list] = {}
    scene_of, base = [], 0
    for f in fs:
        with np.load(f) as z:
            ptr = np.asarray(z["scene_ptr"])
            n_sc = int(ptr.size) - 1
            # 🔴 base 必须是**已累计的场景数**，不是已累计的对数。
            #    原实现写的是 `base = len(scene_of)`（逐对列表的长度）⇒ 单分片时侥幸为 0 而正确，
            #    多分片时把 n_scenes 撑到对数量级（train 5 分片实测报成 7,915,962 而真值 19,826）。
            #    分组本身不受影响（分片间取值区间不重叠），但计数字段会错，故加下面的交叉校验。
            scene_of.append(np.repeat(np.arange(base, base + n_sc, dtype=np.int64), np.diff(ptr)))
            base += n_sc
            for k in z.files:
                if k in ("scene_ptr", "token"):
                    continue
                out.setdefault(k, []).append(np.asarray(z[k]))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["scene_of"] = np.concatenate(scene_of)
    return d


def _dist(a: np.ndarray) -> Dict[str, float]:
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "frac_positive": float((a > 0).mean())}


def _rho(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size < 10 or float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return {"n": int(x.size), "rho": None, "p": None}
    r = spearmanr(x, y)
    return {"n": int(x.size), "rho": float(r.statistic), "p": float(r.pvalue)}


def _within_target_rho(d: Dict[str, np.ndarray], a: str, b: str, min_deg: int = 4) -> Dict:
    """逐 (场景, 目标) 分组算 rho 再取中位数 —— 这是 S4 若学 within-target 排序时的真实口径。"""
    key = d["scene_of"].astype(np.int64) * 100000 + d["dst"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    k, x, y = key[order], d[a][order], d[b][order]
    bounds = np.flatnonzero(np.diff(k)) + 1
    rs = []
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, k.size]):
        if hi - lo >= min_deg and np.std(x[lo:hi]) > 0 and np.std(y[lo:hi]) > 0:
            rs.append(float(spearmanr(x[lo:hi], y[lo:hi]).statistic))
    a_ = np.asarray(rs)
    return {"n_groups": int(a_.size),
            "median": float(np.median(a_)) if a_.size else None,
            "p25": float(np.percentile(a_, 25)) if a_.size else None,
            "p75": float(np.percentile(a_, 75)) if a_.size else None}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="S3 标签读数报告")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    root = Path(a.root)
    d = _load(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))

    rep: Dict = {"root": str(root), "n_pairs": int(d["U"].size),
                 "n_scenes": int(d["scene_of"].max() + 1),
                 "n_teachers": meta.get("n_teachers"), "err_metric": meta.get("err_metric"),
                 "dose": meta.get("dose"), "placebo": meta.get("placebo"),
                 "noop_max_observed": meta.get("noop_max_observed")}
    rep["overall"] = {"U": _dist(d["U"]), "I": _dist(d["I"]), "S": _dist(d["S"]),
                      "placebo": _dist(d["placebo"]), "U_sd_teacher": _dist(d["U_sd_teacher"])}
    rep["by_dst_stratum"] = {
        name: {"U": _dist(d["U"][d["dst_stratum"] == s]), "I": _dist(d["I"][d["dst_stratum"] == s]),
               "S": _dist(d["S"][d["dst_stratum"] == s]),
               "rho_U_S": _rho(d["U"][d["dst_stratum"] == s], d["S"][d["dst_stratum"] == s])}
        for s, name in STRATA.items()}
    # 🔴 交叉校验：读数侧自算的场景/对数必须与 `s3_merge_parts.py` 落盘的 meta 一致。
    #    两者由完全不同的代码路径得出，不一致说明其中一侧的索引口径坏了 —— 当场停，不出报告。
    for _k, _got in (("n_scenes_labelled", rep["n_scenes"]), ("n_pairs", rep["n_pairs"])):
        _want = meta.get(_k)
        if _want is not None and int(_want) != int(_got):
            raise SystemExit(f"🔴 读数与 meta 不一致：{_k} meta={_want} 读数={_got}"
                             f" —— 索引口径坏了，报告不予产出")

    rep["rho_U_S_overall"] = _rho(d["U"], d["S"])
    rep["rho_U_S_within_target"] = _within_target_rho(d, "U", "S")
    rep["rho_U_I_within_target"] = _within_target_rho(d, "U", "I")

    print(f"\n=== S3 标签读数 ({root.name}) ===")
    print(f"  {rep['n_scenes']} 场景 / {rep['n_pairs']} 对 | teacher×{rep['n_teachers']} "
          f"| err={rep['err_metric']} | 空操作门 max {rep['noop_max_observed']:.1e}")
    print(f"  I>0 全局 {100 * rep['overall']['I']['frac_positive']:.1f}%  "
          f"U 中位 {rep['overall']['U']['median']:+.5f}  "
          f"teacher 分歧 U_sd 中位 {rep['overall']['U_sd_teacher']['median']:.5f}")
    print("  逐层（🔴 §8.2：不分层不许引）：")
    for nm, v in rep["by_dst_stratum"].items():
        r = v["rho_U_S"]["rho"]
        print(f"    {nm:20s} n={v['U']['n']:>8}  U 中位 {v['U']['median']:+.5f}  "
              f"I>0 {100 * v['I']['frac_positive']:5.1f}%  rho(U,S)="
              f"{'None' if r is None else f'{r:+.4f}'}")
    w = rep["rho_U_S_within_target"]
    g = rep["rho_U_S_overall"]["rho"]
    fmt = lambda v: "None" if v is None else f"{v:+.4f}"   # noqa: E731
    print(f"  rho(U,S) 全局 {fmt(g)} | 逐目标中位 {fmt(w['median'])} (n_groups={w['n_groups']})")
    print("  🔴 rho(U,S) 是**诊断量**：Q109 §3.3.3 明写「两者的一致性不应预设必然一致」")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  落盘 {a.out}")


if __name__ == "__main__":
    main()
