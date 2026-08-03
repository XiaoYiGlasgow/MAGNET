"""2x2 factorial aggregation for the Route x Perturbation ablation (VT-2026-04191).

Reads every run under a stage directory (``dev`` or ``final``) and produces the
seed-paired factorial analysis required by 阶段15 §5.1 (Z1).

Hard rules enforced by this script
----------------------------------
1. Metrics come **only** from ``models/checkpoint_best.pt`` (the model that was
   actually saved), or — under the S4-b inference contract — from calibrated
   re-evaluation JSONs supplied via ``--metrics_json_dir``.  ``loss_record.json``
   stores ``best_val_ade``/``best_val_fde``, which are the independent minima of
   two separate curves and generally correspond to *no* saved model; reading
   them is forbidden by the project contract.
2. Effects are computed **seed-paired** (S5): every contrast is formed within a
   seed first, then aggregated across seeds.  Cross-seed unpaired means are
   never reported as effects.
3. **Q11**: Route and Perturbation on the primary metric are the only two
   confirmatory hypotheses.  They use a **one-sided** exact sign-flip test in
   the pre-registered direction (improvement = negative), and Holm correction is
   applied **only across those two**.  Everything else (FDE, interaction) is
   exploratory and reported with two-sided p, outside the Holm family.
4. **D0#14**: ``ci_excludes_zero ∧ all_seeds_same_sign`` is mathematically one
   criterion (same-sign ⟹ CI excludes zero) with a null false-positive rate of
   ≈2/2^n — identical to the permutation-p floor.  It is therefore kept as a
   *diagnostic* field only and must not be used as a decision gate.
5. **D0#15**: main effects are on the "per-factor level switch" scale (divided
   by 2), the interaction is on the difference-in-differences scale (not
   divided).  The two scales differ by 2.0×, so Route/Perturbation/interaction
   each need their own δ; they must never share one threshold.
6. **D0#16**: in ``--formal`` mode the expected seed list and epoch budget must
   be passed explicitly; falling back to an overwritable ``<stage>_summary.json``
   is refused, because a queue that rewrites it silently under-reports gaps.
7. ``p > 0.05`` is never rendered as "equivalent".

Usage
-----
    python3 agg_2x2.py --stage_dir <.../experiments/ablation_2x2_f1/dev> \
                       --seeds 42,1,2 --epoch_budget 45 \
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

#: Q11：confirmatory family = 主指标上的两个主效应，其余全部 exploratory
PRIMARY_METRIC = "ade"
PRIMARY_EFFECTS: Tuple[str, ...] = ("route_main_effect", "perturbation_main_effect")
#: 预注册方向：ADE/FDE 越低越好 ⇒ 改善 = 负
PREREGISTERED_DIRECTION = "negative"
#: D0#15：三个 estimand 的尺度不同，δ 不可共用
EFFECT_SCALE: Dict[str, str] = {
    "route_main_effect": "per-factor level switch (÷2)",
    "perturbation_main_effect": "per-factor level switch (÷2)",
    "interaction": "difference-in-differences (未除 2，约为主效应尺度的 2×)",
}


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


def _sign_flip_null(diffs: np.ndarray) -> np.ndarray:
    """Sign-flip null distribution of the mean paired difference (exact when feasible)."""
    n = len(diffs)
    if 2 ** n <= MAX_EXACT_PERM:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:
        signs = np.random.default_rng(0).choice([1.0, -1.0], size=(N_BOOTSTRAP, n))
    return (signs * diffs).mean(axis=1)


def _sign_flip_p(diffs: np.ndarray) -> float:
    """Two-sided exact sign-flip permutation test — exploratory items only (Q11)."""
    if len(diffs) == 0:
        return float("nan")
    null = _sign_flip_null(diffs)
    return float((np.abs(null) >= abs(diffs.mean()) - 1e-15).mean())


def _sign_flip_p_one_sided(diffs: np.ndarray, direction: str = PREREGISTERED_DIRECTION) -> float:
    """One-sided exact sign-flip test in the **pre-registered** direction (Q11).

    ``direction='negative'`` ⇒ H1: mean paired contrast < 0 (ADE/FDE improvement).
    With n=6 the raw p floor is ``1/64 = 0.015625``; after Holm over the two
    primary hypotheses the floor is ``0.03125``.
    """
    if len(diffs) == 0:
        return float("nan")
    null = _sign_flip_null(diffs)
    observed = diffs.mean()
    if direction == "negative":
        return float((null <= observed + 1e-15).mean())
    return float((null >= observed - 1e-15).mean())


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


def _effect(diffs: np.ndarray, seeds: Sequence[int], name: str,
            confirmatory: bool) -> Dict[str, object]:
    """One seed-paired contrast.

    :param confirmatory: True 只对主指标上的两个主效应成立（Q11）；决定 `p_primary_one_sided`
        是否参与 Holm family，也决定报表里怎么标注。
    """
    lo, hi = _paired_bootstrap_ci(diffs)
    same_sign = bool(len(diffs) > 0 and (np.all(diffs > 0) or np.all(diffs < 0)))
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    cohen_dz = float(diffs.mean() / sd) if len(diffs) > 1 and sd > 0 else float("nan")
    return {
        "n_seeds": len(diffs),
        "per_seed": {str(s): float(d) for s, d in zip(seeds, diffs)},
        "mean": float(diffs.mean()) if len(diffs) else float("nan"),
        "sd": sd,
        "scale": EFFECT_SCALE.get(name, "unspecified"),
        "role": "confirmatory" if confirmatory else "exploratory",
        "ci95_bootstrap": [lo, hi],
        "p_one_sided_prereg": _sign_flip_p_one_sided(diffs),
        "p_two_sided_exploratory": _sign_flip_p(diffs),
        "cohen_dz": cohen_dz,
        # D0#14：下面两项是同一条判据（同号 ⟹ CI 排零），假阳性率≈2/2^n = permutation p 下限。
        # 仅作诊断留存，**不得**作为判定门。
        "legacy_diagnostic_only": {
            "ci_excludes_zero": bool(len(diffs) > 1 and (lo > 0 or hi < 0)),
            "all_seeds_same_sign": same_sign,
            "note": "D0#14：同号与 CI 排零是同一条判据，不作判定门",
        },
        "sensitivity": _sensitivity_tests(diffs),
    }


def analyse_metric(matrix: Dict[str, np.ndarray], seeds: Sequence[int],
                   metric: str) -> Dict[str, object]:
    """Seed-paired 2x2 contrasts (S5). Negative = improvement.

    三个 estimand 直接按 seed 计算，不把两个 cell 当独立样本::

        route_s       = ((R_s + RP_s) - (B_s + P_s)) / 2
        perturb_s     = ((P_s + RP_s) - (B_s + R_s)) / 2
        interaction_s = (RP_s - P_s) - (R_s - B_s)
    """
    b, r, p, rp = matrix["B"], matrix["R"], matrix["P"], matrix["RP"]
    contrasts = {
        "route_main_effect": ((r + rp) - (b + p)) / 2.0,
        "perturbation_main_effect": ((p + rp) - (b + r)) / 2.0,
        "interaction": (rp - p) - (r - b),
    }
    out: Dict[str, object] = {
        "cells": {
            c: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "per_seed": {str(s): float(x) for s, x in zip(seeds, v)}}
            for c, v in matrix.items()
        },
    }
    for name, diffs in contrasts.items():
        confirmatory = (metric == PRIMARY_METRIC and name in PRIMARY_EFFECTS)
        out[name] = _effect(diffs, seeds, name, confirmatory)
    return out


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


def apply_external_metrics(runs: Sequence[RunRecord],
                           metrics_dir: Path) -> Tuple[List[RunRecord], Dict[str, object]]:
    """用 `eval_openloop.py` 产出的 JSON 覆盖 checkpoint 里的 diagnostic 指标（S4-b 口径）。

    每个 run 对应 ``<metrics_dir>/<tag>.json``。所有 run 的 ``inference_contract`` 必须一致 ——
    混口径正是提交版 Table II 作废的原因，这里直接报错而不是悄悄合并。
    """
    updated: List[RunRecord] = []
    contracts: Dict[str, str] = {}
    missing: List[str] = []
    for run in runs:
        path = metrics_dir / f"{run.tag}.json"
        if not path.exists():
            missing.append(run.tag)
            updated.append(RunRecord(**{**run.__dict__, "ade": None, "fde": None}))
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        contracts[run.tag] = str(blob.get("inference_contract", "UNKNOWN"))
        updated.append(RunRecord(**{**run.__dict__,
                                    "ade": float(blob["ade"]), "fde": float(blob["fde"])}))
    distinct = sorted(set(contracts.values()))
    if len(distinct) > 1:
        raise SystemExit(f"metrics JSON 混用了多种 inference contract {distinct}；禁止混口径汇总")
    if missing:
        logger.warning("%d 个 run 缺 metrics JSON，将按缺格处理：%s", len(missing), missing)
    return updated, {"source": str(metrics_dir), "inference_contract": distinct[0] if distinct else None,
                     "missing_runs": missing}


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

    # D0#16：queue 每个作业都覆写 <stage>_summary.json，formal 阶段不得依赖它
    if args.formal and (not args.seeds or args.epoch_budget is None):
        raise SystemExit("--formal 模式必须显式传 --seeds 与 --epoch_budget（D0#16："
                         f"{summary_path.name} 会被队列覆写，缺格会被静默低估）")

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
            role = "confirmatory" if e["role"] == "confirmatory" else "exploratory"
            p_show = (f"one-sided p={_fmt(e['p_one_sided_prereg'], 4)}" if role == "confirmatory"
                      else f"two-sided p={_fmt(e['p_two_sided_exploratory'], 3)}")
            add(f"   [{role[:4]}] {label:<16} {e['mean']:+.4f}  95%CI [{_fmt(lo)}, {_fmt(hi)}]"
                f"  {p_show}  dz={_fmt(e['cohen_dz'], 2)}  尺度={e['scale']}")

    add("")
    add("── Holm 校正（只覆盖 confirmatory family = 主指标的 Route/Perturbation）" + "─" * 6)
    for key, value in sorted(report["holm_adjusted_p_primary"].items()):
        add(f"   {key:<42} {_fmt(value, 4)}")
    add("   exploratory 项（FDE、interaction）不进 Holm family，raw two-sided p 见上。")

    add("")
    add("── 判定门 D1（Q11 one-sided + Holm，S5 分别定 δ）" + "─" * 24)
    for line in report["gate"]:
        add("   " + line)
    add("")
    add(f"⚠️ inference 口径：{report.get('metrics_source')}")
    add("⚠️ p>0.05 不等于「等价」。要主张等价须预注册 margin 并做 TOST/非劣检验。")
    add("⚠️ dev 阶段结果只用于定方向与冻结参数，不得进入论文任何表。")
    add("⚠️ 旧 batch=32 未校准数字只是 D0#13 诊断证据；CP1 数字只能来自 calibrated 口径。")
    add("=" * 74)
    return "\n".join(lines)


def gate_verdict(report: Dict[str, object], deltas: Dict[str, Optional[float]],
                 alpha: float) -> List[str]:
    """D1 判定（Q11 + S5）：Holm 后的 one-sided p ≤ α **且** 效应量达到各自的 δ_decision。

    δ 未签字时不给结论 —— 看到结果再定门槛是 §12.2 明令禁止的。
    """
    out: List[str] = []
    res = report["metrics"].get(PRIMARY_METRIC)
    holm_adj = report.get("holm_adjusted_p_primary", {})
    if not res:
        return [f"主指标 {PRIMARY_METRIC.upper()} 尚无完整 2×2 数据，无法判定"]

    for factor, label, key in (("route_main_effect", "Route", "route"),
                               ("perturbation_main_effect", "Perturbation", "perturb")):
        e = res[factor]
        p_adj = holm_adj.get(f"{PRIMARY_METRIC}.{factor}", float("nan"))
        delta = deltas.get(key)
        sig = p_adj == p_adj and p_adj <= alpha
        if delta is None:
            verdict = "δ_decision 未签字 ⇒ 不判定"
        elif e["mean"] <= -abs(delta) and sig:
            verdict = "PASS"
        else:
            reason = []
            if not sig:
                reason.append(f"Holm p={_fmt(p_adj, 4)}>α")
            if not (e["mean"] <= -abs(delta)):
                reason.append(f"效应 {e['mean']:+.4f} 未达 δ={-abs(delta):+.4f}")
            verdict = "FAIL（" + "，".join(reason) + "）"
        out.append(f"{label:<14} mean={e['mean']:+.4f}  one-sided p={_fmt(e['p_one_sided_prereg'], 4)}"
                   f"  Holm={_fmt(p_adj, 4)}  δ={'未签' if delta is None else f'{delta:.4f}'}"
                   f"  -> {verdict}")

    inter = report["metrics"][PRIMARY_METRIC]["interaction"]
    out.append(f"interaction    mean={inter['mean']:+.4f}（difference-in-differences 尺度，"
               f"约主效应的 2×）two-sided p={_fmt(inter['p_two_sided_exploratory'], 3)} "
               "—— exploratory，不共用主效应 δ，不进 confirmatory family")
    out.append(f"FDE / 其余指标全部 exploratory；confirmatory family 只有主指标 "
               f"{PRIMARY_METRIC.upper()} 上的 Route 与 Perturbation 两项（Q11）")
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
    parser.add_argument("--formal", action="store_true",
                        help="formal 模式（D0#16）：强制显式 --seeds/--epoch_budget，"
                             "且指标必须来自 --metrics_json_dir 的 calibrated 口径。")
    parser.add_argument("--metrics_json_dir", type=Path, default=None,
                        help="eval_openloop.py 产出的 <run>.json 目录；给了就用它覆盖 "
                             "checkpoint 里的 diagnostic 指标（S4-b calibrated 口径）。")
    parser.add_argument("--delta_route", type=float, default=None,
                        help="δ_decision(Route)，单位与主指标相同；未签字则不判定")
    parser.add_argument("--delta_perturb", type=float, default=None,
                        help="δ_decision(Perturbation)；与 Route 分别定（S5）")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs = collect_runs(args.stage_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.stage_dir}")

    expected_seeds, epoch_budget, preset = resolve_expectations(args, runs)
    runs = mark_early_stopping(runs, epoch_budget)

    metrics_source = "models/checkpoint_best.pt (saved-checkpoint, batch=32 未校准 diagnostic 口径)"
    external: Dict[str, object] = {}
    if args.metrics_json_dir:
        runs, external = apply_external_metrics(runs, args.metrics_json_dir)
        metrics_source = f"external: {external.get('inference_contract')} ({args.metrics_json_dir})"
    elif args.formal:
        raise SystemExit("--formal 模式禁止使用 checkpoint 里的未校准 batch=32 指标；"
                         "请先用 eval_openloop.py 生成 calibrated 口径并传 --metrics_json_dir")

    report: Dict[str, object] = {
        "stage_dir": str(args.stage_dir),
        "formal": bool(args.formal),
        "metrics_source": metrics_source,
        "external_metrics": external,
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
        report["metrics"][metric] = analyse_metric(matrix, seeds, metric)

    present = {(r.cell, r.seed) for r in runs if r.complete}
    report["incomplete"] = [f"{c}/seed{s}" for c in CELLS for s in expected_seeds
                            if (c, s) not in present]
    report["n_complete"] = len(present)
    report["n_expected"] = len(CELLS) * len(expected_seeds)
    report["seeds_used"] = seeds_all

    # Q11：Holm 只覆盖 confirmatory family（主指标上的两个主效应），用 one-sided p
    primary: Dict[str, float] = {}
    exploratory: Dict[str, float] = {}
    for metric, res in report["metrics"].items():
        for name in ("route_main_effect", "perturbation_main_effect", "interaction"):
            key = f"{metric}.{name}"
            if metric == PRIMARY_METRIC and name in PRIMARY_EFFECTS:
                primary[key] = res[name]["p_one_sided_prereg"]
            else:
                exploratory[key] = res[name]["p_two_sided_exploratory"]
    report["holm_adjusted_p_primary"] = holm(primary)
    report["raw_p_exploratory_two_sided"] = exploratory
    report["test_family"] = {
        "confirmatory": sorted(primary),
        "test": "one-sided exact sign-flip，预注册方向=改善（负）",
        "multiplicity": "Holm over 2 primary hypotheses",
        "alpha": args.alpha,
        "note": "n=6 时 one-sided raw p 下限 1/64=0.015625，Holm 后 0.03125",
    }
    deltas = {"route": args.delta_route, "perturb": args.delta_perturb}
    report["delta_decision"] = deltas
    report["gate"] = gate_verdict(report, deltas, args.alpha)

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
