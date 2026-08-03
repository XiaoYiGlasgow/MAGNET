#!/usr/bin/env python3
"""C4 规划层：任意两格闭环产物的**配对**不确定度量化。

背景（坑#45）：C4 三点验证一开始只有点估计，没有 CI。阶段57 补配对 bootstrap 后，
「修后那格」的 CI 含 0，直接推翻了已写进文档的 [E1] 第三档解读。那次是内联跑的，
没留脚本 ⇒ 本文件把它固化成可复用工具，任何新的 major contrast 都必须过这一关。

口径（与阶段57 逐条一致，改动须走干预协议 §11）：

* **配对单位 = 场景**，键 = ``(log_name, scenario)``；两格必须是同一批场景，
  不完全配对时默认直接报错（``--allow-partial`` 才降级为取交集）。
* 逐场景行只认 ``log_name`` 非空 —— 聚合 parquet 共 270 行 = 256 逐场景 +
  13 个 scenario_type 汇总 + 1 个 final_score。把汇总行当场景正是 07-31 修掉的 bug（坑#28）。
* **Δ = mean(B) − mean(A)**，A 为对照（通常是层 OFF），B 为实验组（层 ON）。
* 95% CI = **按场景重采样的配对 bootstrap**（默认 10 000 次，percentile 法）。
* 同时报**精确符号检验**（去平局后的二项检验）—— 地板效应下平局占 184–217/256，
  均值显著而符号不显著是常态，**两种检验必须同报**（阶段57 方法学观察）。
* 零分比例另报**二项噪声**（n=256 时 SE≈3.4 pt）；落在噪声内只能写
  「未观察到零分比例下降」，不得写「升了/降了」（坑#48）。

用法::

    # 单个对比
    python c4_paired_bootstrap.py --a g13a_20260730/calibrated --b c4_magnet_layer_idfix_20260802 \\
        --label-a "MAGNET 层 OFF" --label-b "MAGNET+层（修后）"

    # 回归门：复算阶段57 已发表的四个对比，与文档值对拍
    python c4_paired_bootstrap.py --regress
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

CLOSED_LOOP = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/closed_loop")

# 事前定死：bootstrap 次数与种子，改动即改口径。
N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 20260802

# 阶段57 已发表的四个对比，用作回归门。
# (label, A 相对路径, B 相对路径, Δ, CI_lo, CI_hi, 好, 差, 平)
PUBLISHED: Tuple[Tuple[str, str, str, float, float, float, int, int, int], ...] = (
    ("MAGNET 层OFF→ON（修前）", "g13a_20260730/calibrated", "c4_magnet_layer_20260801",
     -0.0730, -0.1283, -0.0203, 28, 44, 184),
    ("MAGNET identity 修复效应", "c4_magnet_layer_20260801", "c4_magnet_layer_idfix_20260802",
     +0.0484, +0.0162, +0.0824, 23, 16, 217),
    ("CV 层OFF→ON（修前）", "c4_cv_off_20260801", "c4_cv_on_20260801",
     -0.0079, -0.0694, +0.0528, 54, 50, 152),
    ("PlanTF 层OFF→ON", "plantf_R1_20260731", "c4_plantf_layer_20260802",
     +0.0024, -0.0690, +0.0763, 78, 93, 85),
)


@dataclass(frozen=True)
class PairedResult:
    """一个配对对比的完整读数（点估计 + 两种检验 + 零分比例）。"""

    label_a: str
    label_b: str
    n_pair: int
    n_log: int
    mean_a: float
    mean_b: float
    delta: float
    ci_lo: float
    ci_hi: float
    n_win: int
    n_loss: int
    n_tie: int
    sign_p: float
    zero_frac_a: float
    zero_frac_b: float
    zero_z: float

    @property
    def ci_excludes_zero(self) -> bool:
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)


def _newest_parquet(group_dir: Path) -> Path:
    """取该 group 下最新的 weighted-average 聚合 parquet。"""
    found = sorted(group_dir.rglob("*weighted_average*.parquet"))
    if not found:
        raise FileNotFoundError(f"{group_dir} 下没有聚合 parquet —— 该格可能还没跑完（第十条：rc=0 不算跑成）")
    return max(found, key=lambda p: p.stat().st_mtime)


def load_per_scene(group_dir: Path) -> pd.DataFrame:
    """读一格的逐场景分数，返回索引为 ``(log_name, scenario)`` 的 ``score`` 序列。"""
    parquet = _newest_parquet(group_dir)
    df = pd.read_parquet(parquet)
    if "log_name" not in df.columns:
        raise ValueError(f"{parquet} 没有 log_name 列，无法过滤汇总行（坑#28）")

    per = df[df["log_name"].notna()].copy()
    per["score"] = pd.to_numeric(per["score"], errors="coerce")
    if per["score"].isna().any():
        raise ValueError(f"{parquet} 有 {int(per['score'].isna().sum())} 行 score 解析失败")

    key_cols = ["log_name", "scenario"]
    if per.duplicated(key_cols).any():
        raise ValueError(f"{parquet} 的 (log_name, scenario) 不唯一，配对键不成立")
    return per.set_index(key_cols)[["score"]]


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Tuple[float, float]:
    """按场景重采样的配对 bootstrap，返回 Δ 的 95% percentile CI。"""
    rng = np.random.default_rng(seed)
    diff = scores_b - scores_a
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def compare(
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
    allow_partial: bool = False,
) -> PairedResult:
    """跑一个完整的配对对比。Δ = B − A。"""
    a = load_per_scene(dir_a)
    b = load_per_scene(dir_b)

    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    if len(only_a) or len(only_b):
        msg = f"两格场景不完全配对：只在 A 的 {len(only_a)} 个、只在 B 的 {len(only_b)} 个"
        if not allow_partial:
            raise ValueError(msg + "（要取交集请显式传 --allow-partial 并在读数里声明）")
        logger.warning("%s ⇒ 已按 --allow-partial 取交集", msg)

    common = a.index.intersection(b.index)
    sa = a.loc[common, "score"].to_numpy(dtype=float)
    sb = b.loc[common, "score"].to_numpy(dtype=float)

    delta = float(sb.mean() - sa.mean())
    ci_lo, ci_hi = paired_bootstrap(sa, sb, n_boot=n_boot, seed=seed)

    diff = sb - sa
    n_win = int((diff > 0).sum())
    n_loss = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    # 精确符号检验：去平局后的二项检验（p=0.5，双边）。
    sign_p = (
        float(stats.binomtest(n_win, n_win + n_loss, 0.5, alternative="two-sided").pvalue)
        if (n_win + n_loss) > 0
        else float("nan")
    )

    # 零分比例 + 二项噪声（两独立比例的 z，保守口径；n 相同）。
    zf_a = float((sa == 0).mean())
    zf_b = float((sb == 0).mean())
    n = float(sa.size)
    pooled = (zf_a + zf_b) / 2.0
    se = float(np.sqrt(2.0 * pooled * (1.0 - pooled) / n)) if 0.0 < pooled < 1.0 else float("nan")
    zero_z = (zf_b - zf_a) / se if se and not np.isnan(se) else float("nan")

    n_log = int(pd.MultiIndex.from_tuples(list(common)).get_level_values(0).nunique())
    return PairedResult(
        label_a=label_a,
        label_b=label_b,
        n_pair=int(sa.size),
        n_log=n_log,
        mean_a=float(sa.mean()),
        mean_b=float(sb.mean()),
        delta=delta,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_win=n_win,
        n_loss=n_loss,
        n_tie=n_tie,
        sign_p=sign_p,
        zero_frac_a=zf_a,
        zero_frac_b=zf_b,
        zero_z=zero_z,
    )


def format_result(res: PairedResult) -> str:
    """按 §8 口径（4 位小数）打印，并把两种检验的判读写全。"""
    verdict = "✅ CI 排除 0" if res.ci_excludes_zero else "🔴 CI 含 0（不可区分）"
    zero_verdict = "噪声内" if abs(res.zero_z) < 1.96 else "超出二项噪声"
    lines = [
        f"  A（对照）{res.label_a}: final {res.mean_a:.4f}｜零分 {100 * res.zero_frac_a:.1f}%",
        f"  B（实验）{res.label_b}: final {res.mean_b:.4f}｜零分 {100 * res.zero_frac_b:.1f}%",
        f"  n = {res.n_pair} 配对场景 / {res.n_log} log",
        f"  Δ(B−A) = {res.delta:+.4f}   95% CI [{res.ci_lo:+.4f}, {res.ci_hi:+.4f}]   {verdict}",
        f"  符号检验：好 {res.n_win} / 差 {res.n_loss} / 平 {res.n_tie}   p = {res.sign_p:.2g}"
        f"   {'（不显著）' if res.sign_p >= 0.05 else '（显著）'}",
        f"  零分比例 Δ = {100 * (res.zero_frac_b - res.zero_frac_a):+.1f} pt   z = {res.zero_z:.2f}"
        f"   ⇒ {zero_verdict}",
    ]
    if not res.ci_excludes_zero:
        lines.append("  ⚠️ 写作口径：只能写「不可区分」，不得写「降了/升了」（坑#45/#48）")
    return "\n".join(lines)


def run_regression(n_boot: int, seed: int) -> int:
    """回归门：复算阶段57 已发表的四个对比。

    好/差/平是确定性的 ⇒ **必须逐位一致**；CI 依赖 RNG 种子（原次种子未记录）
    ⇒ 只做 ±0.01 的容差检查。
    """
    ok = True
    for label, rel_a, rel_b, d0, lo0, hi0, w0, l0, t0 in PUBLISHED:
        try:
            res = compare(CLOSED_LOOP / rel_a, CLOSED_LOOP / rel_b, rel_a, rel_b,
                          n_boot=n_boot, seed=seed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ⛔ {label}: {exc}")
            ok = False
            continue

        d_ok = abs(res.delta - d0) < 5e-4
        c_ok = (res.n_win, res.n_loss, res.n_tie) == (w0, l0, t0)
        ci_ok = abs(res.ci_lo - lo0) < 0.01 and abs(res.ci_hi - hi0) < 0.01
        mark = "✅" if (d_ok and c_ok and ci_ok) else "❌"
        print(f"  {mark} {label}")
        print(f"       Δ {res.delta:+.4f} (文档 {d0:+.4f}) {'✓' if d_ok else '✗'}"
              f"｜好/差/平 {res.n_win}/{res.n_loss}/{res.n_tie}"
              f" (文档 {w0}/{l0}/{t0}) {'✓' if c_ok else '✗'}"
              f"｜CI [{res.ci_lo:+.4f},{res.ci_hi:+.4f}]"
              f" (文档 [{lo0:+.4f},{hi0:+.4f}]) {'✓' if ci_ok else '✗'}")
        ok = ok and d_ok and c_ok and ci_ok
    print(f"\n  回归门：{'✅ 全过' if ok else '❌ 有不一致，先查口径再用本脚本出新数字'}")
    return 0 if ok else 1


def _resolve(group: str) -> Path:
    """允许传相对 ``experiments/closed_loop`` 的名字，也允许绝对路径。"""
    path = Path(group)
    return path if path.is_absolute() else CLOSED_LOOP / group


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="对照格（层 OFF）目录名或绝对路径")
    parser.add_argument("--b", help="实验格（层 ON）目录名或绝对路径")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--allow-partial", action="store_true", help="场景不完全配对时取交集（须在读数里声明）")
    parser.add_argument("--regress", action="store_true", help="复算阶段57 已发表的四个对比")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.regress:
        print(f"\n=== 回归门（n_boot={args.n_boot}, seed={args.seed}）===")
        return run_regression(args.n_boot, args.seed)

    if not (args.a and args.b):
        parser.error("要么给 --a/--b，要么用 --regress")

    try:
        res = compare(_resolve(args.a), _resolve(args.b), args.label_a, args.label_b,
                      n_boot=args.n_boot, seed=args.seed, allow_partial=args.allow_partial)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print(f"\n=== 配对对比（n_boot={args.n_boot}, seed={args.seed}）===")
    print(format_result(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
