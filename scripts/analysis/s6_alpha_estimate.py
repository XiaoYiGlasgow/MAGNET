#!/usr/bin/env python3
"""S6 —— 实测 scaling 指数 α，并按事前判据决定下一级是否值得跑。

判据全部**事前写死**（Q60 / §2 S6，不得因数字不好看再改）：

  * 模型      : ``log(ADE) ~ log(N)``，**α = 斜率的绝对值**（ADE 随数据增加而下降 ⇒ 斜率为负）。
  * 主指标    : ``val/minADE1``（Q65 primary，= 闭环 planner 实际执行的那条）。
  * CI        : 🔴 **按 seed 层 bootstrap**，不是对观测点做普通回归。
                两级各 3 seed ⇒ 只有 2 个 N，普通回归残差自由度为 0、CI 无从谈起；
                噪声的真实来源是 seed，所以重采样单位必须是 seed（§9「聚类数据的 n 是簇数」）。
  * 功效门槛  : MAGNET 实测 σ_avg5=0.069 @ ADE≈2.135 ⇒ 相对噪声 3.23%，k=3 需 **α ≥ 0.045**。
                PlanTF 侧相对噪声由 L1 读数实测（2.66%，比值 0.82×）⇒ 门槛可沿用 0.045。
                若某级实测相对噪声显著高于 MAGNET，门槛按比例放大（脚本自动算）。
  * 决策      : α ≥ 门槛 ⇒ 跑下一级；α < 门槛 ⇒ **下一级跑了也测不出来，应停**。

⚠️ 两级估 α 的**已知局限**（必须随结论同报）：
  - 两点定一条直线 ⇒ 无法检验 log-log 线性假设本身；
  - CI 只反映 seed 噪声，不含「曲线其实是弯的」这一项模型误设风险；
  - ⇒ 本读数用于**投不投 L3 的决策**，不是论文里的最终 α。论文的 α 用三级拟合。

用法::

    python3 scripts/analysis/s6_alpha_estimate.py --tiers L1 L2
    python3 scripts/analysis/s6_alpha_estimate.py --tiers L1 L2 L3 --self_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
LADDER = GNN / "experiments" / "scaling_ladder"

# 事前写死的常量（见模块 docstring）
PRIMARY_TAG = "val/minADE1"
ALPHA_THRESHOLD_BASE = 0.045          # k=3 在 MAGNET 相对噪声 3.23% 下的 80% 功效门槛
MAGNET_REL_SD = 0.0323
N_BOOT = 10000
BOOT_SEED = 20260803

# 各级训练样本数（= splitter 的 train 部分，**不是** train+val）
# 由 s1 清单 × splitter 精确算出；L1 的 20936 已被 L1 冒烟实测逐位验证。
TIER_N_TRAIN = {"L1": 20936, "L2": 62903, "L3": 204408}
TIER_N_LOG = {"L1": 212, "L2": 635, "L3": 2062}


def load_tier(tier: str) -> Dict[str, float]:
    """读某级的读数 JSON，返回 {seed: minADE1}。"""
    p = LADDER / f"s3_readout_{tier}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在 —— 该级还没读数？先跑 s3_readout_last5.py --tier {tier}")
    d = json.loads(p.read_text())
    if d.get("SELF_TEST"):
        raise ValueError(f"{p} 是自测产物（last_n 偏离签字口径），不得用于 α 估计")
    return {k: float(v) for k, v in d["aggregate"][PRIMARY_TAG]["per_seed"].items()}


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """最小二乘斜率；两点时退化为差商（数学等价）。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _slope_from_seedpick(tiers: Sequence[str],
                         per_tier: Dict[str, Dict[str, float]],
                         picks: Dict[str, str]) -> float:
    """给定「每级选哪个 seed」，返回该组合的 log-log 斜率。"""
    xs = [math.log(TIER_N_TRAIN[t]) for t in tiers]
    ys = [math.log(per_tier[t][picks[t]]) for t in tiers]
    return _ols_slope(xs, ys)


def estimate_alpha(tiers: Sequence[str], n_boot: int = N_BOOT,
                   boot_seed: int = BOOT_SEED) -> Dict[str, object]:
    """点估计用各级 seed 均值；CI 用**按 seed 重采样**的 bootstrap。"""
    per_tier = {t: load_tier(t) for t in tiers}

    # 点估计：每级取 seed 均值
    xs = [math.log(TIER_N_TRAIN[t]) for t in tiers]
    ys = [math.log(sum(per_tier[t].values()) / len(per_tier[t])) for t in tiers]
    slope = _ols_slope(xs, ys)

    # bootstrap：每级独立地对自己的 seed 有放回重采样，重算斜率
    rng = random.Random(boot_seed)
    boots: List[float] = []
    seed_lists = {t: list(per_tier[t].keys()) for t in tiers}
    for _ in range(n_boot):
        ys_b = []
        for t in tiers:
            sl = seed_lists[t]
            resamp = [per_tier[t][rng.choice(sl)] for _ in sl]
            ys_b.append(math.log(sum(resamp) / len(resamp)))
        s = _ols_slope(xs, ys_b)
        if not math.isnan(s):
            boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]

    # 门槛：按各级实测相对噪声相对 MAGNET 的比值放大
    rel_sds = []
    for t in tiers:
        v = list(per_tier[t].values())
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
        rel_sds.append(sd / m)
    rel_sd_max = max(rel_sds)
    threshold = ALPHA_THRESHOLD_BASE * max(1.0, rel_sd_max / MAGNET_REL_SD)

    return {
        "tiers": list(tiers),
        "primary_tag": PRIMARY_TAG,
        "n_train_per_tier": {t: TIER_N_TRAIN[t] for t in tiers},
        "n_log_per_tier": {t: TIER_N_LOG[t] for t in tiers},
        "ade_mean_per_tier": {t: math.exp(y) for t, y in zip(tiers, ys)},
        "per_tier_per_seed": per_tier,
        "slope": slope,
        "alpha": abs(slope),
        "alpha_ci95": [abs(hi), abs(lo)] if slope < 0 else [abs(lo), abs(hi)],
        "slope_ci95": [lo, hi],
        "ci_excludes_zero": (lo < 0 and hi < 0) or (lo > 0 and hi > 0),
        "rel_sd_per_tier": {t: r for t, r in zip(tiers, rel_sds)},
        "alpha_threshold": threshold,
        "n_boot": len(boots),
        "boot_seed": boot_seed,
        "caveat": (
            "两级估 α：两点定一线，无法检验 log-log 线性假设；CI 只含 seed 噪声，"
            "不含模型误设风险。本读数用于投不投下一级的决策，论文的 α 用三级拟合。"
        ),
    }


def print_report(r: Dict[str, object]) -> str:
    lines: List[str] = []
    a = lines.append
    a(f"=== 实测 α（{'→'.join(r['tiers'])}，主指标 {r['primary_tag']}）===")
    for t in r["tiers"]:
        a(f"  {t}: N_train={r['n_train_per_tier'][t]:>7,}｜log={r['n_log_per_tier'][t]:>5}"
          f"｜ADE={r['ade_mean_per_tier'][t]:.4f}｜rel_sd={r['rel_sd_per_tier'][t]*100:.2f}%")
    a("")
    a(f"  斜率      = {r['slope']:+.4f}   95% CI [{r['slope_ci95'][0]:+.4f}, {r['slope_ci95'][1]:+.4f}]")
    a(f"  α = |斜率| = {r['alpha']:.4f}   95% CI [{r['alpha_ci95'][0]:.4f}, {r['alpha_ci95'][1]:.4f}]")
    a(f"  CI 排除 0 ? {'✅ 是' if r['ci_excludes_zero'] else '🔴 否（斜率与 0 不可区分）'}")
    a(f"  事前门槛   = {r['alpha_threshold']:.4f}"
      f"（基准 0.045 × 相对噪声放大系数）｜bootstrap n={r['n_boot']}，按 seed 重采样")
    a("")

    if r["slope"] > 0:
        a("  ⇒ 🔴 斜率为正 = 数据越多反而越差。这不是 scaling，先查管线（同 log 池？同口径？）。")
    elif r["alpha"] >= r["alpha_threshold"] and r["ci_excludes_zero"]:
        a(f"  ⇒ 🟢 **跑下一级**：α={r['alpha']:.4f} ≥ 门槛 {r['alpha_threshold']:.4f} 且 CI 排除 0，"
          "k=3 有 80% 功效测出斜率。")
    elif r["alpha"] >= r["alpha_threshold"]:
        a(f"  ⇒ 🟠 α={r['alpha']:.4f} 过门槛但 **CI 含 0** ⇒ 点估计够大、证据不足。"
          "两点估计的 CI 本就宽，建议跑下一级把点数提到 3 再判。")
    else:
        a(f"  ⇒ 🔴 **停，不跑下一级**：α={r['alpha']:.4f} < 门槛 {r['alpha_threshold']:.4f}，"
          "k=3 下测不出来（重演 k=1 没有检出力的老问题）。")
    a("")
    a(f"  ⚠️ {r['caveat']}")
    out = "\n".join(lines)
    print(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="实测 scaling 指数 α 并判下一级是否值得跑")
    p.add_argument("--tiers", nargs="+", required=True, choices=["L1", "L2", "L3"])
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--self_test", action="store_true", help="用合成数据自测拟合与判据")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if len(args.tiers) < 2:
        raise SystemExit("至少要两级才能估斜率")

    r = estimate_alpha(args.tiers)
    print_report(r)
    out = Path(args.out) if args.out else LADDER / f"s6_alpha_{'_'.join(args.tiers)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")


def _self_test() -> None:
    """合成数据自测：已知 α 能否被还原，且判据方向对不对。"""
    import tempfile
    ok = 0
    with tempfile.TemporaryDirectory() as td:
        global LADDER
        orig, LADDER = LADDER, Path(td)
        try:
            for true_alpha, expect in ((0.20, "跑"), (0.01, "停")):
                for tier in ("L1", "L2"):
                    n = TIER_N_TRAIN[tier]
                    base = 3.0 * (n / TIER_N_TRAIN["L1"]) ** (-true_alpha)
                    per = {str(s): base * (1 + 0.005 * i) for i, s in enumerate((42, 1, 2))}
                    (LADDER / f"s3_readout_{tier}.json").write_text(json.dumps(
                        {"aggregate": {PRIMARY_TAG: {"per_seed": per}}}))
                r = estimate_alpha(["L1", "L2"], n_boot=2000)
                err = abs(r["alpha"] - true_alpha)
                got = "跑" if r["alpha"] >= r["alpha_threshold"] else "停"
                good = err < 1e-6 and got == expect
                ok += good
                print(f"  {'✅' if good else '🔴'} 真值 α={true_alpha}: 还原 {r['alpha']:.6f}"
                      f"（误差 {err:.2e}）｜判据={got}（期望 {expect}）")
        finally:
            LADDER = orig
    print(f"\n自测 {ok}/2 {'通过' if ok == 2 else '🔴 失败'}")
    if ok != 2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
