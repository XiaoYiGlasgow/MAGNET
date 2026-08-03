#!/usr/bin/env python3
"""S2-pre 读数 —— L3 坏场景**全量普查**的结论化输出（2026-08-02 阶段66）。

## 为什么单独写一个读数脚本

`s2_merge_l3_scan.py` 负责「合并 + 写出剔除后清单」，是**产物**脚本。
本脚本只做**判读**，不写任何清单，回答提示词 A1 的四个问题：

1. 全量失败率是多少？（🔴 **这是普查不是抽样 ⇒ 直接引用，不需要 CI**）
2. 全崩的 log 有几个？分布是不是仍然双峰？
3. 剔除后 L1/L2/L3 各剩多少样本？阶梯比例还成立吗？
4. 🔴 **与 20 log 抽样的 11.8% 对比** —— 阶段65 审阅判定该值是「假精度」，
   按 log 聚类的正确 CI 是 **[−2.8%, 26.4%]**。普查值落在区间内则审阅结论被证实；
   落在区间外则连那个 CI 也偏窄，须记为方法学教训。

## 判读纪律（§9）

* 右偏/双峰分布**不能用一个中心统计量代表**（第十五条）⇒ 同时报 log 级的分位数与两端计数；
* 普查值**不配 CI**（配了反而是错的：总体已全测，不存在抽样误差）；
* Q61 已裁定 (a) 接受不均匀 ⇒ 报**每 log 存活帧数**的分布，供「上限 100 帧」措辞取证；
* 🔴 Q61 的再议触发条件（§10 事前写死）：**全量失败率 > 25%** ⇒ 重新讨论是否回退 `rig=True`。

用法::

    python s2_l3_scan_readout.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SUMMARY = GNN / "experiments/scaling_ladder/s2_l3_scan_summary.json"
FILTERED = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42_filtered.json"

# 阶段65 审阅给出的按 log 聚类 CI（20 log 抽样），用于检验普查值是否落在其中
SAMPLE_RATE = 0.118
SAMPLE_CI = (-0.028, 0.264)
Q61_REOPEN_THRESHOLD = 0.25


def quantiles(values: List[float], qs: List[float]) -> Dict[str, float]:
    """无 numpy 依赖的分位数（线性插值），values 会被排序。"""
    if not values:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    s = sorted(values)
    out: Dict[str, float] = {}
    for q in qs:
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        out[f"p{int(q * 100)}"] = s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return out


def main() -> int:
    if not SUMMARY.exists():
        logger.error("汇总不存在：%s ⇒ L3 扫描的接班还没跑完", SUMMARY)
        return 2
    rep: Dict[str, Any] = json.loads(SUMMARY.read_text())

    n_all = rep["n_scenarios"]
    n_bad = rep["n_bad"]
    rate = rep["failure_rate"]
    n_clean = rep["n_logs_clean"]
    n_partial = rep["n_logs_partial"]
    n_allbad = rep["n_logs_all_bad"]
    n_logs = n_clean + n_partial + n_allbad

    print("\n" + "=" * 78)
    print("S2-pre L3 坏场景全量普查 —— 读数（普查值，不配 CI）")
    print("=" * 78)

    print(f"\n【1】全量失败率（**普查**，非抽样）")
    print(f"  试算场景 {n_all:,}｜失败 {n_bad:,} ⇒ **失败率 {rate:.4%}**")
    print(f"  异常类型分布：{rep['kinds']}")
    print(f"  🔴 这是总体普查 ⇒ **直接引用，不需要也不应该配 CI**")

    print(f"\n【2】与 20 log 抽样的 11.8% 对比（检验阶段65 审阅结论 B）")
    lo, hi = SAMPLE_CI
    inside = lo <= rate <= hi
    print(f"  抽样点估计 {SAMPLE_RATE:.1%}｜按 log 聚类的 95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  普查真值 {rate:.2%} ⇒ {'✅ 落在 CI 内' if inside else '🔴 落在 CI 外'}")
    if inside:
        print("  ⇒ 审阅结论 B 成立：token 口径的 ±1.4% 是假精度，聚类 CI 才是对的。")
    else:
        print("  ⇒ 🔴 连聚类 CI 也没覆盖真值 ⇒ 记为方法学教训，20 log 的样本量对双峰分布不够。")
    diff = rate - SAMPLE_RATE
    print(f"  抽样偏差 {diff:+.2%}（普查 − 抽样）")

    print(f"\n【3】log 级分布（双峰是否仍成立；右偏/双峰不得用单一中心统计量代表）")
    print(f"  log 总数 {n_logs:,}：全干净 **{n_clean:,}**（{n_clean / n_logs:.1%}）"
          f"｜部分失败 **{n_partial:,}**（{n_partial / n_logs:.1%}）"
          f"｜**全崩 {n_allbad:,}**（{n_allbad / n_logs:.1%}）")
    partial = rep.get("logs_partial", {})
    if partial:
        fr = [f / t for f, t in partial.values() if t]
        q = quantiles(fr, [0.25, 0.5, 0.75, 0.9])
        print(f"  部分失败 log 的失败比例：p25 {q['p25']:.1%}｜中位 {q['p50']:.1%}"
              f"｜p75 {q['p75']:.1%}｜p90 {q['p90']:.1%}")
    # 双峰判据：两端（0% 与 100%）占了多少 log
    ends = (n_clean + n_allbad) / n_logs
    print(f"  🔑 两端（0% 与 100%）合计占 **{ends:.1%}** 的 log"
          f" ⇒ {'双峰成立' if ends > 0.8 else '双峰减弱，需重看'}")

    print(f"\n【4】剔除后阶梯（Q61 已裁定 (a) 接受不均匀，横轴须同报 log 数 + 样本数）")
    before = rep["tiers_before"]
    after = rep["tiers_after"]
    print(f"  {'级':<4}{'剔除前':>10}{'剔除后':>10}{'降幅':>9}")
    for tier in ("L1", "L2", "L3"):
        b, a = before[tier], after[tier]
        print(f"  {tier:<4}{b:>10,}{a:>10,}{(b - a) / b:>8.1%}")

    if FILTERED.exists():
        man = json.loads(FILTERED.read_text())
        print(f"\n  剔除后每级 log 数：", end="")
        for tier in ("L1", "L2", "L3"):
            print(f"{tier} {man['tiers'][tier]['n_log']:,}  ", end="")
        print()
        # 每 log 存活帧数分布 —— 「上限 100 帧」措辞的取证
        toks = man["tokens"]
        l3_logs = man["tiers"]["L3"]["logs"]
        counts = [len(toks.get(lg, [])) for lg in l3_logs]
        q = quantiles([float(c) for c in counts], [0.25, 0.5, 0.75])
        n_full = sum(1 for c in counts if c == man["frames_per_log"])
        print(f"  每 log 存活帧数：min {min(counts)}｜p25 {q['p25']:.0f}｜中位 {q['p50']:.0f}"
              f"｜p75 {q['p75']:.0f}｜max {max(counts)}")
        print(f"  仍满 {man['frames_per_log']} 帧的 log：{n_full:,}/{len(counts):,}"
              f"（{n_full / len(counts):.1%}）⇒ 措辞必须是「**上限** {man['frames_per_log']} 帧」")
        # 阶梯嵌套断言（剔除后仍须 L1 ⊂ L2 ⊂ L3）
        s1, s2, s3 = (set(man["tiers"][t]["logs"]) for t in ("L1", "L2", "L3"))
        print(f"  嵌套断言 L1⊂L2⊂L3：{'✅ 成立' if s1 <= s2 <= s3 else '🔴 被破坏'}")

    print(f"\n【5】Q61 再议触发判据（§10 事前写死）")
    trig = rate > Q61_REOPEN_THRESHOLD
    print(f"  阈值 {Q61_REOPEN_THRESHOLD:.0%}｜实测 {rate:.2%} ⇒ "
          f"{'🔴 **触发**，须重新讨论是否回退 rig=True' if trig else '✅ 未触发，Q61 的 (a) 继续有效'}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
