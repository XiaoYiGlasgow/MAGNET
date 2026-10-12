"""融合层三分支「权重占比 vs 贡献场景间方差占比」读数（A2 / A2′）。

背景：阶段113 用「权重占比不低（gnn 26.11%）但贡献方差占比只有 0.75%」解释了独立测得的
使用度 0.72% —— 权重再大，乘以一个**场景间近乎不变**的向量还是常数。阶段207 又查出
**新档**（full retention 64 槽）的权重分配与旧档反过来（gnn 范数 53.25% / 能量 69.41%），
于是留下欠账 **A2′：权重再分配有没有转化成信息**，必须在新档重算贡献方差占比才能答。

估计量（与阶段113 同口径，此处第一次写成脚本）：

* 切块：``feature_fusion.weight`` 是 ``[128, 384]``，输入是 ``cat[ego, context, gnn]``
  ⇒ 列块 0:128 / 128:256 / 256:384（见 ``transformer_predictor.py`` 的 forward）。
* 权重占比：**范数口径** ‖W_b‖_F / Σ‖W_b‖_F 与**能量口径** ‖W_b‖_F² / Σ‖W_b‖_F²
  —— 两个是同一次测量的两种范数（阶段207 已解析核对为恒等变换），按第八十三条**都报**。
* 贡献：c_b = feat_b · W_bᵀ ∈ ℝ^{N×128}，即该分支真正送进网络的量。
* 贡献场景间方差占比：tr(Cov(c_b)) / Σ_b tr(Cov(c_b))，Cov 在**场景维**上取（ddof=1）。

🔴 该分解**忽略了分支间的协方差**：三支贡献之和的总方差不等于三支总方差之和。
本脚本因此同报 ``cross_term_share`` = 1 − Σ_b tr(Cov(c_b)) / tr(Cov(Σ_b c_b))，
用来说明这个近似有多松；写作时若 cross term 大，占比只能作**描述性**读数。

输入 = ``branch_feature_probe.py`` 的 npz（含 ego/ctx/gnn 三段 128 维特征）+ 同一个 ckpt。
用法::

    python3 scripts/analysis/fusion_contribution_share.py \\
        --ckpt <abs.pth> --probe <abs.npz> --out <abs.json> [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("fusion_share")

BRANCHES = ("ego", "ctx", "gnn")
BLOCK = {"ego": slice(0, 128), "ctx": slice(128, 256), "gnn": slice(256, 384)}


def _fusion_weight(ckpt_path: Path) -> np.ndarray:
    """取 ``feature_fusion.weight``；ckpt 可能是 bare state_dict 或带 model_state_dict 的字典。"""
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    keys = [k for k in state if k.endswith("feature_fusion.weight")]
    if len(keys) != 1:
        raise KeyError(f"ckpt 里 feature_fusion.weight 命中 {len(keys)} 个键：{keys}")
    w = state[keys[0]].detach().cpu().numpy()
    if w.shape[1] != 384:
        raise ValueError(f"feature_fusion.weight 形状 {w.shape}，非 [*, 384]，切块口径不成立")
    return w


def _total_var(x: np.ndarray) -> float:
    """场景间总方差 tr(Cov)：对每一维在样本轴上取方差（ddof=1）再求和。"""
    return float(np.var(x, axis=0, ddof=1).sum())


def compute(w: np.ndarray, feats: Dict[str, np.ndarray]) -> Dict:
    w_norm = {b: float(np.linalg.norm(w[:, BLOCK[b]])) for b in BRANCHES}
    s_norm = sum(w_norm.values())
    s_energy = sum(v ** 2 for v in w_norm.values())

    contrib = {b: feats[b] @ w[:, BLOCK[b]].T for b in BRANCHES}     # [N,128]
    var_c = {b: _total_var(contrib[b]) for b in BRANCHES}
    s_var = sum(var_c.values())
    var_sum = _total_var(sum(contrib.values()))

    return {
        "n_samples": int(next(iter(feats.values())).shape[0]),
        "weight_share_frobenius_pct": {b: 100.0 * w_norm[b] / s_norm for b in BRANCHES},
        "weight_share_energy_pct": {b: 100.0 * w_norm[b] ** 2 / s_energy for b in BRANCHES},
        "contribution_var_share_pct": {b: 100.0 * var_c[b] / s_var for b in BRANCHES},
        "contribution_total_var": var_c,
        "feature_total_var": {b: _total_var(feats[b]) for b in BRANCHES},
        "feature_mean_norm": {b: float(np.linalg.norm(feats[b].mean(axis=0))) for b in BRANCHES},
        "feature_mean_dimwise_sd": {b: float(np.std(feats[b], axis=0, ddof=1).mean()) for b in BRANCHES},
        "cross_term_share": float(1.0 - s_var / var_sum) if var_sum > 0 else float("nan"),
        "note": ("cross_term_share = 1 − Σ_b tr(Cov(c_b)) / tr(Cov(Σ_b c_b))；"
                 "非零说明分支贡献相关，占比只作描述性读数"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--probe", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0,
                   help="只用前 N 个 token（排序后取）；0 = 全用。用于与阶段113 的 300 token 口径对齐")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    d = np.load(a.probe, allow_pickle=True)
    feats = {"ego": d["ego"], "ctx": d["ctx"], "gnn": d["gnn"]}
    if a.limit:
        feats = {k: v[: a.limit] for k, v in feats.items()}
    w = _fusion_weight(a.ckpt)

    res = compute(w, feats)
    res.update({"ckpt": str(a.ckpt), "probe": str(a.probe),
                "probe_ckpt": str(d["ckpt"]), "probe_raw_dir": str(d["raw_dir"]),
                "limit": a.limit})
    if str(d["ckpt"]) != str(a.ckpt):
        logger.warning("🔴 探针 npz 记录的 ckpt (%s) 与 --ckpt (%s) 不同 —— 权重与特征不同源，读数无效",
                       d["ckpt"], a.ckpt)
        res["MISMATCH"] = True

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("weight_share_frobenius_pct", "weight_share_energy_pct", "contribution_var_share_pct"):
        logger.info("%s: %s", key, {k: round(v, 2) for k, v in res[key].items()})
    logger.info("cross_term_share = %.4f  (n=%d) -> %s", res["cross_term_share"], res["n_samples"], a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
