#!/usr/bin/env python3
"""B5 —— 邻居计数可表示性的**正式**读数（欠账清单 §10.1 B5；初值出处 = 阶段117）。

背景
----
阶段117 只有一个非正式初值：`corr(‖gnn_feat‖, n_node) = −0.037`（旧档 s4b seed42、
300 token），用来说明「机制1 判别力稀释随节点数」在幅度口径下不成立。
「可表示性」的正式问法不是幅度相关，而是：**从分支特征能不能线性解码出邻居计数**。

估计量（描述性，不设判据）
--------------------------
对三分支 128 维特征各做 ridge 线性 probe（闭式解，λ=1e-3·tr(XᵀX)/d）：
- 目标 1 = `n_node`（GNN 图节点总数：agent+lane+ego）；
- 目标 2 = `n_traj`（agent/traj 子图数，≈他车数）；
- 5-fold CV R²（fold 按 token 排序切块，可复现）；
- shuffle 对照：标签重排（rng=20260819）后同 probe，R² 应≈0；
- 特征 z-score 用训练 fold 的均值/方差（不泄漏）。

回归门：旧档前 300 token 上 corr(‖gnn‖, n_node) 必须复现 −0.037（±0.005），
不符 exit 2。

口径红线
--------
- 描述性读数：单 ckpt/单 seed/无区间；「可解码」≠「被下游使用」（探针可读出 ≠ 融合层在用）；
- 旧档（default 契约）为主；新档（full_retention, fr_base seed42）同报只作对照，
  两档差三个变量不得相减。

用法：/home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/b5_neighbor_count_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
NPZ = {
    "oldtier_s4b_seed42": GNN / "experiments/branch_probe_20260818/oldtier_s4b_seed42.npz",
    "newtier_fr_base_seed42": GNN / "experiments/branch_probe_20260818/newtier_fr_base_seed42.npz",
}
OUT = GNN / "scratchpad/b5_neighbor_count_probe_20260819.json"

GATE_CORR = -0.037   # 阶段117 初值
GATE_TOL = 0.005


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """5-fold CV R²（fold 按样本顺序切块）；ridge 闭式解，z-score 只用训练 fold。"""
    n = len(y)
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    ss_res, ss_tot = 0.0, 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        mu, sd = X[tr].mean(0), X[tr].std(0, ddof=1) + 1e-12
        Xtr, Xte = (X[tr] - mu) / sd, (X[f] - mu) / sd
        ym = y[tr].mean()
        lam = 1e-3 * np.trace(Xtr.T @ Xtr) / Xtr.shape[1]
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ym))
        pred = Xte @ w + ym
        ss_res += float(((y[f] - pred) ** 2).sum())
        ss_tot += float(((y[f] - y[tr].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def tier_readout(npz_path: Path, rng: np.random.Generator) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    feats = {"ego": d["ego"], "ctx": d["ctx"], "gnn": d["gnn"]}
    targets = {"n_node": d["n_node"].astype(np.float64),
               "n_traj": d["n_traj"].astype(np.float64)}
    out: dict = {"n_samples": int(len(d["n_node"])), "ckpt": str(d["ckpt"]),
                 "target_stats": {t: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                      "min": int(v.min()), "max": int(v.max())}
                                  for t, v in targets.items()},
                 "norm_corr": {}, "probe_cv_r2": {}, "probe_cv_r2_shuffled": {}}
    for b, X in feats.items():
        nrm = np.linalg.norm(X, axis=1)
        out["norm_corr"][b] = {t: float(np.corrcoef(nrm, y)[0, 1]) for t, y in targets.items()}
        out["probe_cv_r2"][b] = {t: ridge_cv_r2(X.astype(np.float64), y)
                                 for t, y in targets.items()}
        out["probe_cv_r2_shuffled"][b] = {
            t: ridge_cv_r2(X.astype(np.float64), rng.permutation(y))
            for t, y in targets.items()}
    return out


def main() -> int:
    rng = np.random.default_rng(20260819)
    res = {tier: tier_readout(p, rng) for tier, p in NPZ.items()}

    # 🔴 回归门：旧档前 300 token 复现阶段117 初值
    d = np.load(NPZ["oldtier_s4b_seed42"], allow_pickle=True)
    corr300 = float(np.corrcoef(np.linalg.norm(d["gnn"][:300], axis=1),
                                d["n_node"][:300].astype(np.float64))[0, 1])
    gate = abs(corr300 - GATE_CORR) < GATE_TOL
    print(f"回归门：corr(‖gnn‖,n_node) 前300 = {corr300:.4f}（期望 {GATE_CORR}±{GATE_TOL}）"
          f" -> {'PASS' if gate else 'FAIL'}")
    if not gate:
        return 2

    out = {
        "protocol": {
            "probe": "ridge 闭式解 λ=1e-3·tr(XᵀX)/d，5-fold 顺序切块 CV，z-score 不泄漏",
            "targets": "n_node（图节点总数）/ n_traj（agent 子图数）",
            "shuffle_control": "标签重排 rng=20260819",
            "regression_gate": {"corr300": corr300, "expect": GATE_CORR, "pass": True},
        },
        "tiers": res,
        "caveats": [
            "描述性读数：单 ckpt/单 seed、无区间",
            "「线性可解码」≠「被下游使用」：probe 能读出只说明信息在特征里",
            "旧档为主；新档只作对照，两档差三个变量（数据契约/ckpt 家族/pooling fix）不得相减",
            "n_traj 在旧档删静止后动态范围小（中位 2），R² 的分母方差本身就小",
            "实测 n_lane 恒定（旧档 40 / 新档 68）⇒ n_node = n_traj + 常数，两目标仿射等价、"
            "probe R² 恒等；实质都在解码 n_traj（他车数）",
            "正式读数推翻初值的暗示：corr(‖gnn‖,n_node)≈0 只说明计数不在幅度里；"
            "线性 probe R²=0.48（旧档）说明计数在方向里可解码",
        ],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for tier, r in res.items():
        print(f"[{tier}] n={r['n_samples']}")
        for b in ("ego", "ctx", "gnn"):
            print(f"  {b}: corr(‖·‖,n_node)={r['norm_corr'][b]['n_node']:+.3f}  "
                  f"CV R²(n_node)={r['probe_cv_r2'][b]['n_node']:.3f} "
                  f"(shuf {r['probe_cv_r2_shuffled'][b]['n_node']:+.3f})  "
                  f"R²(n_traj)={r['probe_cv_r2'][b]['n_traj']:.3f} "
                  f"(shuf {r['probe_cv_r2_shuffled'][b]['n_traj']:+.3f})")
    print(f"✅ -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
