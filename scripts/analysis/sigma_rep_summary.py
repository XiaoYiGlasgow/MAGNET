#!/usr/bin/env python
"""CP1 前置：从重复 run 估计 σ_rep，并给出预注册最小效应量 δ_min 的下界。

背景（D0#11）
------------
`train_magnet_imitation.py` 的 `set_seed` 只设了 `cudnn.deterministic=True`，**没有**设
`torch.use_deterministic_algorithms(True)`，PyG 的 scatter 走 atomicAdd ——
**同 commit + 同 seed 也不保证同数字**。

后果：cell 内的「seed 间方差」其实是 `seed 效应 + run-to-run 复现噪声` 的混合。
如果预注册的最小效应量比复现噪声还小，那条预注册承诺的是一个**测不出来**的差别。
本脚本把 σ_rep 单独测出来，作为 δ_min 的硬下界。

口径
----
* 指标一律读 `models/checkpoint_best.pt`（saved-checkpoint 口径），
  **禁用** `loss_record.json` 的 `best_val_ade/fde`（那是两条曲线各自的独立最小值，
  通常不在同一个 epoch）。
* σ_rep 用**同 cell、同 seed、同代码**的多次重复 run 的样本标准差（ddof=1）。
* σ_seed 用同 cell 跨 seed 的样本标准差 —— 注意它**已经包含** σ_rep。

δ_min 的推导
-----------
final 用 n_seeds 个 seed，两格之差按 seed 配对，其标准误：

    SE(effect) = σ_seed · sqrt(2 / n_seeds)

取「效应至少要有 2 个 SE」作为可检出下限：

    δ_min ≳ 2 · SE(effect) = 2·sqrt(2/n_seeds) · σ_seed

因 σ_seed ≥ σ_rep，σ_rep 给出 δ_min 的硬下界。

用法
----
    python sigma_rep_summary.py \
        --g1prime_dir experiments/ablation_2x2_f1/dev \
        --rep_dirs experiments/ablation_2x2_f1_rep1/dev \
                   experiments/ablation_2x2_f1_rep2/dev \
                   experiments/ablation_2x2_f1_rep3/dev \
        --cell B --seed 42 --n_seeds_final 5 \
        --out experiments/ablation_2x2_f1/sigma_rep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ("ade", "fde")


def read_checkpoint_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    """读 saved-checkpoint 口径的 val_ade / val_fde。

    :param run_dir: 形如 ``.../dev/B_seed42`` 的单 run 目录。
    :return: ``{"ade": .., "fde": .., "epoch": ..}``；缺文件或字段返回 None。
    """
    import torch

    ckpt_path = run_dir / "models" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        return None
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读 checkpoint 失败 %s: %s", ckpt_path, exc)
        return None
    if blob.get("val_ade") is None or blob.get("val_fde") is None:
        logger.warning("checkpoint 缺 val_ade/val_fde: %s", ckpt_path)
        return None
    return {
        "ade": float(blob["val_ade"]),
        "fde": float(blob["val_fde"]),
        # checkpoint 存 0-indexed epoch，日志/agg_2x2 打印 +1，这里统一转成 1-indexed
        "epoch": (int(blob["epoch"]) + 1) if blob.get("epoch") is not None else None,
    }


def is_complete(run_dir: Path) -> bool:
    """`loss_record.json` 存在才算跑完 —— 中断的 run 也会留下 checkpoint。"""
    return (run_dir / "loss_record.json").exists()


def chi2_upper_bound(s: float, dof: int, alpha: float = 0.05) -> Optional[float]:
    """σ 的单侧 (1-alpha) 上界：s · sqrt(dof / χ²_{alpha, dof})。"""
    if dof < 1 or not math.isfinite(s):
        return None
    try:
        from scipy.stats import chi2
    except ImportError:
        logger.warning("缺 scipy，跳过 χ² 上界")
        return None
    q = float(chi2.ppf(alpha, dof))
    if q <= 0:
        return None
    return float(s * math.sqrt(dof / q))


def collect_draws(g1prime_dir: Path, rep_dirs: Sequence[Path],
                  cell: str, seed: int) -> Tuple[List[Dict], List[str]]:
    """收集同 cell 同 seed 的所有重复 draw（G1' 本体算第一个 draw）。"""
    draws: List[Dict] = []
    skipped: List[str] = []
    for label, base in [("G1'", g1prime_dir)] + [
            (f"rep{i + 1}", d) for i, d in enumerate(rep_dirs)]:
        run_dir = base / f"{cell}_seed{seed}"
        if not run_dir.is_dir():
            skipped.append(f"{label}: 目录不存在 {run_dir}")
            continue
        if not is_complete(run_dir):
            skipped.append(f"{label}: 未跑完（无 loss_record.json）")
            continue
        m = read_checkpoint_metrics(run_dir)
        if m is None:
            skipped.append(f"{label}: checkpoint 不可读")
            continue
        draws.append({"label": label, "run_dir": str(run_dir), **m})
    return draws, skipped


def seed_spread(g1prime_dir: Path, cell: str, seeds: Sequence[int]) -> Dict[str, object]:
    """同 cell 跨 seed 的标准差（含 σ_rep）。"""
    vals: Dict[str, List[float]] = {m: [] for m in METRICS}
    used: List[int] = []
    for s in seeds:
        run_dir = g1prime_dir / f"{cell}_seed{s}"
        if not (run_dir.is_dir() and is_complete(run_dir)):
            continue
        m = read_checkpoint_metrics(run_dir)
        if m is None:
            continue
        used.append(s)
        for k in METRICS:
            vals[k].append(m[k])
    out: Dict[str, object] = {"cell": cell, "seeds_used": used, "n": len(used)}
    for k in METRICS:
        out[f"{k}_values"] = vals[k]
        out[f"sigma_seed_{k}"] = float(np.std(vals[k], ddof=1)) if len(vals[k]) >= 2 else None
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g1prime_dir", type=Path, required=True,
                   help="G1' 的 stage 目录，例如 experiments/ablation_2x2_f1/dev")
    p.add_argument("--rep_dirs", type=Path, nargs="*", default=[],
                   help="重复 run 的 stage 目录，例如 .../ablation_2x2_f1_rep1/dev")
    p.add_argument("--cell", default="B", help="在哪个 cell 上测重复噪声")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_list", default="42,1,2", help="估 σ_seed 用的 seed 列表")
    p.add_argument("--n_seeds_final", type=int, default=5, help="final 阶段的 seed 数")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]

    draws, skipped = collect_draws(args.g1prime_dir, args.rep_dirs, args.cell, args.seed)
    spread = seed_spread(args.g1prime_dir, args.cell, seeds)

    n = len(draws)
    dof = n - 1
    rep: Dict[str, object] = {"cell": args.cell, "seed": args.seed, "n_draws": n, "dof": dof,
                              "draws": draws, "skipped": skipped}
    for k in METRICS:
        vals = [d[k] for d in draws]
        rep[f"{k}_values"] = vals
        rep[f"sigma_rep_{k}"] = float(np.std(vals, ddof=1)) if n >= 2 else None
        rep[f"range_{k}"] = (float(max(vals) - min(vals))) if n >= 2 else None
        rep[f"sigma_rep_{k}_upper95"] = (
            chi2_upper_bound(rep[f"sigma_rep_{k}"], dof) if n >= 2 else None)

    # δ_min 下界：2·sqrt(2/n_seeds)·σ
    factor = 2.0 * math.sqrt(2.0 / args.n_seeds_final)
    delta: Dict[str, object] = {"n_seeds_final": args.n_seeds_final,
                                "factor_2SE": round(factor, 4)}
    for k in METRICS:
        s_rep = rep[f"sigma_rep_{k}"]
        s_rep_ub = rep[f"sigma_rep_{k}_upper95"]
        s_seed = spread.get(f"sigma_seed_{k}")
        cands = {"from_sigma_rep": (factor * s_rep) if s_rep else None,
                 "from_sigma_rep_upper95": (factor * s_rep_ub) if s_rep_ub else None,
                 "from_sigma_seed_dev": (factor * s_seed) if s_seed else None}
        present = [v for v in cands.values() if v]
        delta[k] = {**{kk: (round(v, 4) if v else None) for kk, v in cands.items()},
                    "delta_min_lower_bound": round(max(present), 4) if present else None}

    report = {"sigma_rep": rep, "sigma_seed_dev": spread, "delta_min": delta}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 74)
    print(f" σ_rep 估计 —— cell={args.cell} seed={args.seed}，口径 = saved checkpoint")
    print("=" * 74)
    if skipped:
        for s in skipped:
            print(f"   ⏭️ {s}")
    for d in draws:
        print(f"   {d['label']:<6} ADE={d['ade']:.4f}  FDE={d['fde']:.4f}  ep{d['epoch']}")
    print(f"   n_draws={n}  dof={dof}")
    if n < 2:
        print("   ⚠️ draw 不足 2 个，给不出 σ_rep。CP1 的 §1 判据链在此之前不成立。")
    else:
        for k in METRICS:
            ub = rep[f"sigma_rep_{k}_upper95"]
            print(f"   σ_rep({k.upper()}) = {rep[f'sigma_rep_{k}']:.4f}"
                  f"   极差 = {rep[f'range_{k}']:.4f}"
                  f"   95% 单侧上界 = {ub:.4f}" if ub else "")
        if n == 2:
            print("   ⚠️ n=2 只是**单点粗估**（E|X₁−X₂| = 1.128σ），只能用来设下界，不能当 σ 的可用估计。")
    print("-" * 74)
    print(f" σ_seed(dev, cell={args.cell}, seeds={spread['seeds_used']}, n={spread['n']})"
          "  —— 注意它已包含 σ_rep")
    for k in METRICS:
        v = spread.get(f"sigma_seed_{k}")
        print(f"   σ_seed({k.upper()}) = {v:.4f}" if v else f"   σ_seed({k.upper()}) = 不足 2 个 seed")
    print("-" * 74)
    print(f" δ_min 下界（final n_seeds={args.n_seeds_final}，2·SE 口径，系数 {factor:.4f}）")
    for k in METRICS:
        d = delta[k]
        print(f"   {k.upper()}: 由 σ_rep {d['from_sigma_rep']} / 由 σ_rep 上界"
              f" {d['from_sigma_rep_upper95']} / 由 σ_seed_dev {d['from_sigma_seed_dev']}"
              f"  ⇒ **δ_min ≥ {d['delta_min_lower_bound']}**")
    print("=" * 74)
    print(" 注：dev 数字只用于定门槛，不得进入论文任何表；报告 σ_rep 时必须写明 n、cell 与数据集。")
    if args.out:
        print(f" 报告: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
