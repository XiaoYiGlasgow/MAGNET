"""2x2 factorial aggregation for the Route x Perturbation ablation (VT-2026-04191).

Reads every run under a stage directory (``dev`` or ``final``) and produces the
seed-paired factorial analysis required by 阶段15 §5.1 (Z1).

Hard rules enforced by this script
----------------------------------
1. Metrics come **only** from ``models/checkpoint_best.pt`` (the model that was
   actually saved).  ``loss_record.json`` also stores ``best_val_ade`` /
   ``best_val_fde``, but those are the independent minima of two separate
   curves and generally correspond to *no* saved model.  Using them is
   forbidden by the project contract; this script refuses to read them.
2. Effects are computed **seed-paired**: every contrast is formed within a seed
   first, then aggregated across seeds.  Cross-seed unpaired means are never
   reported as effects.
3. Primary inference is a paired bootstrap / exact permutation CI.  Paired
   t-test and Wilcoxon are reported only as sensitivity checks, because n=3..5
   has almost no power to check normality.
4. Holm correction is applied across the reported metrics.
5. ``p > 0.05`` is never rendered as "equivalent".

Usage
-----
    python3 agg_2x2.py --stage_dir <.../experiments/ablation_2x2/dev> \
                       --out <.../dev_summary_stats.json>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 2x2 design: cell -> (route_mode, perturbation_enabled)
CELLS: Dict[str, Tuple[str, bool]] = {
    "B": ("none", False),
    "R": ("three_channel", False),
    "P": ("none", True),
    "RP": ("three_channel", True),
}
METRICS: Tuple[str, ...] = ("ade", "fde")
N_BOOTSTRAP = 20000
MAX_EXACT_PERM = 2 ** 14  # exact sign-flip permutation up to 14 seeds


@dataclass(frozen=True)
class RunRecord:
    """One训练 run, keyed by (cell, seed)."""

    tag: str
    cell: str
    seed: int
    route_mode: str
    perturbation: bool
    ckpt_epoch: Optional[int]
    ade: Optional[float]
    fde: Optional[float]
    epochs_run: Optional[int]
    early_stopped: Optional[bool]
    ckpt_sha256: Optional[str]

    @property
    def complete(self) -> bool:
        return self.ade is not None and self.fde is not None


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """Checkpoint SHA256 — required by the D0#5 provenance contract."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_metrics(ckpt: Path) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Read saved-checkpoint metrics. Never falls back to curve minima.

    ``epoch`` is stored 0-indexed by ``train_magnet_imitation.py`` (it comes from
    ``range(args.epochs)``) while the training log prints ``epoch + 1``.  We
    normalise to the 1-indexed value here so the report matches the log.
    """
    if not ckpt.exists():
        logger.warning("missing checkpoint: %s", ckpt)
        return None, None, None
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report and skip this run
        logger.error("cannot load %s: %s", ckpt, exc)
        return None, None, None
    epoch = blob.get("epoch")
    return (epoch + 1 if epoch is not None else None), blob.get("val_ade"), blob.get("val_fde")


def _classify_cell(route_mode: str, perturbation: bool) -> Optional[str]:
    for name, (rm, pert) in CELLS.items():
        if rm == route_mode and pert == perturbation:
            return name
    return None


def collect_runs(stage_dir: Path) -> List[RunRecord]:
    """Scan a stage directory and build one record per run."""
    runs: List[RunRecord] = []
    for record_path in sorted(stage_dir.glob("*/loss_record.json")):
        run_dir = record_path.parent
        with open(record_path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)

        route_mode = rec.get("route_mode", "none")
        pert_cfg = rec.get("perturbation") or {}
        perturbation = bool(pert_cfg.get("enabled", False))
        cell = _classify_cell(route_mode, perturbation)
        if cell is None:
            logger.warning("run %s does not match any 2x2 cell (route=%s perturb=%s)",
                           run_dir.name, route_mode, perturbation)
            continue

        ckpt_path = run_dir / "models" / "checkpoint_best.pt"
        epoch, ade, fde = _load_checkpoint_metrics(ckpt_path)
        epochs_run = rec.get("epochs_run")
        # early stopping 触发 <=> 实际跑的 epoch 数少于计划上限。`history` 与
        # `epochs_run` 是同一条曲线的两种记法，互相比较恒等，不能用来判早停；
        # 唯一可判的信号是 epochs_run 是否达到本 run 的 epoch 上限（由调用方传入）。
        runs.append(RunRecord(
            tag=run_dir.name, cell=cell, seed=int(rec.get("seed", -1)),
            route_mode=route_mode, perturbation=perturbation,
            ckpt_epoch=epoch, ade=ade, fde=fde,
            epochs_run=epochs_run, early_stopped=None,
            ckpt_sha256=sha256_file(ckpt_path),
        ))
    return runs


def mark_early_stopping(runs: Sequence[RunRecord], epoch_budget: Optional[int]) -> List[RunRecord]:
    """Fill ``early_stopped`` once the per-run epoch budget is known."""
    if epoch_budget is None:
        return list(runs)
    return [RunRecord(**{**r.__dict__,
                         "early_stopped": (r.epochs_run is not None and r.epochs_run < epoch_budget)})
            for r in runs]


def build_matrix(runs: Sequence[RunRecord], metric: str) -> Tuple[List[int], Dict[str, np.ndarray]]:
    """Return the seeds present in ALL four cells, and per-cell metric vectors."""
    by_cell: Dict[str, Dict[int, float]] = {c: {} for c in CELLS}
    for run in runs:
        if not run.complete:
            continue
        value = getattr(run, metric)
        if run.seed in by_cell[run.cell]:
            logger.warning("duplicate run for cell=%s seed=%s (%s)", run.cell, run.seed, run.tag)
        by_cell[run.cell][run.seed] = float(value)

    common = sorted(set.intersection(*(set(v.keys()) for v in by_cell.values()))) if all(by_cell.values()) else []
    matrix = {c: np.array([by_cell[c][s] for s in common], dtype=float) for c in CELLS} if common else {}
    return common, matrix


def _paired_bootstrap_ci(diffs: np.ndarray, alpha: float = 0.05,
                         n_boot: int = N_BOOTSTRAP, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap over the seed-level paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _sign_flip_p(diffs: np.ndarray) -> float:
    """Two-sided exact sign-flip permutation test on paired differences."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(diffs.mean())
    if 2 ** n <= MAX_EXACT_PERM:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = np.abs((signs * diffs).mean(axis=1))
        return float((stats >= observed - 1e-15).mean())
    rng = np.random.default_rng(0)
    signs = rng.choice([1.0, -1.0], size=(N_BOOTSTRAP, n))
    stats = np.abs((signs * diffs).mean(axis=1))
    return float((stats >= observed - 1e-15).mean())


def _sensitivity_tests(diffs: np.ndarray) -> Dict[str, Optional[float]]:
    """Paired t and Wilcoxon, reported as sensitivity analyses only."""
    out: Dict[str, Optional[float]] = {"paired_t_p": None, "wilcoxon_p": None}
    if len(diffs) < 2:
        return out
    try:
        from scipy import stats as sps
        out["paired_t_p"] = float(sps.ttest_1samp(diffs, 0.0).pvalue)
        if len(diffs) >= 5 and not np.allclose(diffs, 0.0):
            out["wilcoxon_p"] = float(sps.wilcoxon(diffs).pvalue)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scipy sensitivity tests unavailable: %s", exc)
    return out


def _effect(diffs: np.ndarray, seeds: Sequence[int]) -> Dict[str, object]:
    lo, hi = _paired_bootstrap_ci(diffs)
    same_sign = bool(len(diffs) > 0 and (np.all(diffs > 0) or np.all(diffs < 0)))
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    cohen_dz = float(diffs.mean() / sd) if len(diffs) > 1 and sd > 0 else float("nan")
    return {
        "n_seeds": len(diffs),
        "per_seed": {str(s): float(d) for s, d in zip(seeds, diffs)},
        "mean": float(diffs.mean()) if len(diffs) else float("nan"),
        "sd": sd,
        "ci95_bootstrap": [lo, hi],
        "ci_excludes_zero": bool(len(diffs) > 1 and (lo > 0 or hi < 0)),
        "all_seeds_same_sign": same_sign,
        "permutation_p": _sign_flip_p(diffs),
        "cohen_dz": cohen_dz,
        "sensitivity": _sensitivity_tests(diffs),
    }


def analyse_metric(matrix: Dict[str, np.ndarray], seeds: Sequence[int]) -> Dict[str, object]:
    """Seed-paired 2x2 main effects and interaction. Negative = improvement."""
    b, r, p, rp = matrix["B"], matrix["R"], matrix["P"], matrix["RP"]
    return {
        "cells": {
            c: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "per_seed": {str(s): float(x) for s, x in zip(seeds, v)}}
            for c, v in matrix.items()
        },
        "route_main_effect": _effect(((r + rp) - (b + p)) / 2.0, seeds),
        "perturbation_main_effect": _effect(((p + rp) - (b + r)) / 2.0, seeds),
        "interaction": _effect((rp - p) - (r - b), seeds),
    }


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment."""
    items = sorted(((k, v) for k, v in pvalues.items() if v == v), key=lambda kv: kv[1])
    m, adjusted, running = len(items), {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    for key, value in pvalues.items():
        adjusted.setdefault(key, value)
    return adjusted


def resolve_expectations(args: argparse.Namespace,
                         runs: Sequence[RunRecord]) -> Tuple[List[int], Optional[int], Dict]:
    """Work out which (cell, seed) grid was *intended*, so gaps are reported honestly.

    Order of authority: explicit CLI > ``<stage>_summary.json`` written by
    ``run_2x2_ablation.py`` > seeds actually observed on disk (last resort, and
    it under-reports missing cells while the stage is still running).
    """
    stage_dir = Path(args.stage_dir)
    summary_path = stage_dir.parent / f"{stage_dir.name}_summary.json"
    preset: Dict = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                preset = json.load(fh).get("resolved_preset", {}) or {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read %s: %s", summary_path, exc)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    elif preset.get("seeds"):
        seeds = [int(s) for s in preset["seeds"]]
    else:
        seeds = sorted({r.seed for r in runs})
        logger.warning("未拿到预期 seed 列表（%s 不存在或无 resolved_preset），"
                       "退回已观察到的 seed=%s —— 缺失格数会被低估", summary_path, seeds)

    budget = args.epoch_budget if args.epoch_budget is not None else preset.get("epochs")
    return seeds, (int(budget) if budget else None), preset


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def render(report: Dict[str, object]) -> str:
    lines: List[str] = []
    add = lines.append
    add("=" * 74)
    add(f" 2x2 Route x Perturbation — {report['stage_dir']}")
    add(f" runs found: {report['n_runs']}   complete: {report['n_complete']}/{report.get('n_expected', '?')}"
        f"   expected seeds: {report.get('expected_seeds')}")
    add(" 口径: saved checkpoint (checkpoint_best.pt) —— 训练曲线独立 minima 已禁用")
    add("=" * 74)

    if report.get("incomplete"):
        add("")
        add("⚠️ 未完成，下列 (cell, seed) 缺失，效应量仅供参考：")
        add("   " + ", ".join(report["incomplete"]))

    budget = report.get("epoch_budget")
    add("")
    add(f"── 训练预算与早停（epoch 上限 = {budget or '未知'}）" + "─" * 30)
    for run in report["runs"]:
        if not run.get("epochs_run"):
            continue
        if run.get("early_stopped") is None:
            note = "早停未知（未提供 epoch 上限）"
        else:
            note = "early stop ✅" if run["early_stopped"] else "⚠️ 跑满未早停"
        add(f"   {run['tag']:<12} epochs_run={run['epochs_run']:<4} ckpt@ep{run['ckpt_epoch']:<4} {note}")
    add("   ⚠️ 「跑满未早停」意味着 epoch 上限本身在限制模型 —— final 阶段必须放宽或说明。")

    for metric in METRICS:
        res = report["metrics"].get(metric)
        if not res:
            continue
        add("")
        add(f"── {metric.upper()} (m)  —— 负号 = 改善 " + "─" * 34)
        add(f"   {'cell':<6}{'mean':>9}{'sd':>9}   per-seed")
        for cell in ("B", "R", "P", "RP"):
            c = res["cells"][cell]
            per = "  ".join(f"{k}:{v:.3f}" for k, v in c["per_seed"].items())
            add(f"   {cell:<6}{c['mean']:>9.3f}{c['sd']:>9.3f}   {per}")
        for name, label in (("route_main_effect", "route 主效应"),
                            ("perturbation_main_effect", "perturb 主效应"),
                            ("interaction", "交互项")):
            e = res[name]
            lo, hi = e["ci95_bootstrap"]
            flag = "✅" if (e["ci_excludes_zero"] and e["all_seeds_same_sign"]) else "❌"
            add(f"   {flag} {label:<16} {e['mean']:+.4f}  95%CI [{_fmt(lo)}, {_fmt(hi)}]"
                f"  perm p={_fmt(e['permutation_p'], 3)}"
                f"  同号={'是' if e['all_seeds_same_sign'] else '否'}"
                f"  dz={_fmt(e['cohen_dz'], 2)}")

    add("")
    add("── Holm 校正后的 permutation p " + "─" * 42)
    for key, value in sorted(report["holm_adjusted_p"].items()):
        add(f"   {key:<42} {_fmt(value, 4)}")

    add("")
    add("── 判定门 D1 前判（阶段15 §6）" + "─" * 42)
    for line in report["gate"]:
        add("   " + line)
    add("")
    add("⚠️ p>0.05 不等于「等价」。要主张等价须预注册 margin 并做 TOST/非劣检验。")
    add("⚠️ dev 阶段结果只用于定方向与冻结参数，不得进入论文任何表。")
    add("=" * 74)
    return "\n".join(lines)


def gate_verdict(report: Dict[str, object]) -> List[str]:
    """Apply the three D1 pre-check criteria from 阶段15 §6."""
    out: List[str] = []
    for factor, label in (("route_main_effect", "route"), ("perturbation_main_effect", "perturbation")):
        verdicts = []
        for metric in METRICS:
            res = report["metrics"].get(metric)
            if not res:
                continue
            e = res[factor]
            ok = e["ci_excludes_zero"] and e["all_seeds_same_sign"]
            direction = "改善" if e["mean"] < 0 else "变差"
            verdicts.append(f"{metric.upper()}: {'稳定' if ok else '不稳定'}/{direction}")
        out.append(f"{label:<14} " + " | ".join(verdicts) if verdicts else f"{label}: 无数据")
    inter = [f"{m.upper()} {report['metrics'][m]['interaction']['mean']:+.4f}"
             for m in METRICS if m in report["metrics"]]
    out.append("interaction    " + " | ".join(inter) + "  （显著为负须在论文中报告，不得只报主效应）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="预期 seed 列表（逗号分隔）。不给则从 <stage>_summary.json 读，"
                             "再不行才退回「已观察到的 seed」——那会低估缺失格数。")
    parser.add_argument("--epoch_budget", type=int, default=None,
                        help="本阶段每个 run 的 epoch 上限，用于判定是否触发 early stopping。"
                             "不给则从 <stage>_summary.json 的 resolved_preset 读。")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs = collect_runs(args.stage_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.stage_dir}")

    expected_seeds, epoch_budget, preset = resolve_expectations(args, runs)
    runs = mark_early_stopping(runs, epoch_budget)

    report: Dict[str, object] = {
        "stage_dir": str(args.stage_dir),
        "metric_source": "models/checkpoint_best.pt (saved-checkpoint); curve minima disabled",
        "expected_seeds": expected_seeds,
        "epoch_budget": epoch_budget,
        "stage_preset": preset,
        "n_runs": len(runs),
        "runs": [r.__dict__ for r in runs],
        "metrics": {},
    }

    seeds_all: List[int] = []
    for metric in METRICS:
        seeds, matrix = build_matrix(runs, metric)
        if not matrix:
            logger.warning("metric %s: no seed present in all four cells yet", metric)
            continue
        seeds_all = seeds
        report["metrics"][metric] = analyse_metric(matrix, seeds)

    present = {(r.cell, r.seed) for r in runs if r.complete}
    report["incomplete"] = [f"{c}/seed{s}" for c in CELLS for s in expected_seeds
                            if (c, s) not in present]
    report["n_complete"] = len(present)
    report["n_expected"] = len(CELLS) * len(expected_seeds)
    report["seeds_used"] = seeds_all

    pvals: Dict[str, float] = {}
    for metric, res in report["metrics"].items():
        for name in ("route_main_effect", "perturbation_main_effect", "interaction"):
            pvals[f"{metric}.{name}"] = res[name]["permutation_p"]
    report["holm_adjusted_p"] = holm(pvals)
    report["gate"] = gate_verdict(report)

    text = render(report)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        args.out.with_suffix(".txt").write_text(text, encoding="utf-8")
        logger.info("written: %s (+ .txt)", args.out)


if __name__ == "__main__":
    main()
