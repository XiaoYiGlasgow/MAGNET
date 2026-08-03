"""C4（R4-1 规划形式化）主表 + leave-one-out 消融的**事前判读**读表器。

🔴 本脚本写于 **2026-08-01 21:0x —— 八个格一个都还没跑完的时候**。
判据全部照抄 `C4_规划形式化_设计_20260801.md` §4.4 与主控表 §7 的 C4 六条，
目的就是让「读表」变成机械动作，不给见到分数之后再挑判据的余地。
**改判据必须走干预协议 §11 并重跑全部格。**

三条事前写死的判读（§4.4）
-------------------------
P1 `progress` 项是主效应
    `no_progress` 相对 `full`(= `magnet_layer`) 的 `ego_progress_along_expert_route`
    与 `ego_is_making_progress` **双双下降**，且降幅**大于其余三件消融的任一件**。
P2 规划层把分数抬离地板
    `magnet_layer` 的**零分场景比例 < 78.5%**（当前 calibrated 层 OFF 值，本脚本实测复算）。
P3 学习组件是否有贡献
    `magnet_layer − cv_on` 的 final 差 —— **无论多小都报绝对值**，并按 §7 措辞写。

🔴🔴 CV 那一列是诚实性支点：若「CV+层」≈「MAGNET+层」，结论**只能**写
「在该规划形式化下学习组件无可测量的额外贡献」，**不得**只报 MAGNET+层的绝对分数。

口径说明
--------
* 逐场景行只认 `log_name` 非空的行 —— 聚合 parquet 共 270 行 = 256 逐场景 + 13 个
  scenario_type 汇总 + 1 个 final_score；把汇总行当场景正是 07-31 修掉的那个 bug
  （零分率被记成 86.6%，真值 88.3%）。本脚本的零分算法已在
  `g13a_20260730/calibrated` 上复算出 **0.7852**，与主控表 §8 的 78.5% 逐位吻合。
* 层 OFF 的两个对照**复用已有产物、零 GPU**：
  MAGNET 层 OFF = `g13a_20260730/calibrated`（0.198）；PlanTF 层 OFF = `plantf_R1_20260731`（0.356）。

用法::

    python c4_readout.py                        # 默认读 *_20260801 那一批
    python c4_readout.py --date 20260801 --csv out.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# `extract_scores` 与本文件同目录，且**只取模块级常量与纯函数**（无类）
# ⇒ 坑#18「同一文件两条 import 路径 = 两个模块对象致 isinstance 静默失败」在此不适用。
# 宁可 import 也不复制列名契约，避免两处定义漂移。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_scores as _es  # noqa: E402

logger = logging.getLogger(__name__)

CLOSED_LOOP = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/closed_loop")

# 事前定死的零分基线：层 OFF 的 MAGNET calibrated（主控表 §8 = 78.5%）。
ZERO_FRAC_BASELINE = 0.785

# 主表 / 消融各格 -> (人话说明, 是不是消融格)
CELLS: Dict[str, Tuple[str, bool]] = {
    "magnet_layer": ("MAGNET + 层（主表核心格，= 消融的 full）", False),
    "cv_on": ("CV 常速外推 + 层（🔴 诚实性支点）", False),
    "cv_off": ("CV 常速外推，层 OFF（trivial 下界）", False),
    "no_progress": ("消融 −progress（事前预测：主效应）", True),
    "no_dynamics": ("消融 −dynamics（件①）", True),
    "no_collision": ("消融 −collision（件②）", True),
    "no_rules": ("消融 −rules（件③）", True),
}

# 层 OFF 的两个对照：复用已有产物，零 GPU。
REUSED: Dict[str, Tuple[Path, str]] = {
    "magnet_off": (CLOSED_LOOP / "g13a_20260730" / "calibrated", "MAGNET 层 OFF（calibrated，已有）"),
    "plantf_off": (CLOSED_LOOP / "plantf_R1_20260731", "PlanTF 层 OFF（R1 自训 20 k，已有）"),
}

PROGRESS_KEYS = ("progress", "making_prog")


def _newest_parquet(group_dir: Path) -> Optional[Path]:
    """取该 group 下最新的 weighted-average 聚合 parquet（tag 可能不止一个）。"""
    found = _es.find_aggregator_files(group_dir)
    if not found:
        return None
    return max(found.values(), key=lambda p: p.stat().st_mtime)


def read_cell(group_dir: Path) -> Optional[Dict[str, object]]:
    """读一个 group 的七项官方指标 + 零分场景比例 + 场景/log 计数。"""
    parquet = _newest_parquet(group_dir)
    if parquet is None:
        return None

    tag = parquet.relative_to(group_dir).parts[0]
    summary = _es.summarise_tag(tag, parquet)
    if summary is None:
        return None

    try:
        df = pd.read_parquet(parquet)
    except (OSError, ValueError) as exc:
        logger.error("读不了 %s：%s", parquet, exc)
        return None

    # 🔴 逐场景行只认 log_name 非空（见模块 docstring 的口径说明）。
    if "log_name" not in df.columns:
        logger.warning("%s 没有 log_name 列，零分比例不可信，置 None", group_dir.name)
        summary["zero_frac"] = None
        summary["n_scenario"] = None
        summary["n_log"] = None
        return summary

    per_scene = df[df["log_name"].notna()]
    scores = pd.to_numeric(per_scene[_es.SCORE_COLUMN], errors="coerce").dropna()
    summary["zero_frac"] = float(scores.eq(0).mean()) if len(scores) else None
    summary["n_scenario"] = int(len(scores))
    summary["n_log"] = int(per_scene["log_name"].nunique())
    summary["parquet"] = str(parquet.relative_to(CLOSED_LOOP))
    return summary


def collect(date_tag: str) -> Dict[str, Dict[str, object]]:
    """把八个格（含两个复用的层 OFF 对照）读成一张表；缺格返回时不报错，后面按缺格处理。"""
    out: Dict[str, Dict[str, object]] = {}
    for cell in CELLS:
        got = read_cell(CLOSED_LOOP / f"c4_{cell}_{date_tag}")
        if got is not None:
            out[cell] = got
    for key, (path, _desc) in REUSED.items():
        got = read_cell(path)
        if got is not None:
            out[key] = got
    return out


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "  --  "
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: Dict[str, Dict[str, object]]) -> None:
    labels = ["final_score"] + list(_es.SUB_METRICS.values())
    header = f"{'格':<16}" + "".join(f"{lab:>13}" for lab in labels) + f"{'零分比例':>11}{'场景':>7}{'log':>6}"
    print("\n=== C4 主表 + 消融（官方七项，nonreactive，256 场景配对）===")
    print(header)
    print("-" * len(header))
    order = list(CELLS) + list(REUSED)
    for key in order:
        row = rows.get(key)
        if row is None:
            print(f"{key:<16}" + "  （未跑完 / 无产物）")
            continue
        line = f"{key:<16}"
        for lab in labels:
            line += f"{_fmt(row.get(lab), 4):>13}"
        zf = row.get("zero_frac")
        line += f"{(f'{zf*100:.1f}%' if isinstance(zf, float) else '--'):>11}"
        line += f"{_fmt(row.get('n_scenario')):>7}{_fmt(row.get('n_log')):>6}"
        print(line)


def judge_p1(rows: Dict[str, Dict[str, object]]) -> List[str]:
    """P1：`no_progress` 是不是主效应。"""
    out: List[str] = ["\n【P1】progress 项是主效应？（§4.4 事前判据）"]
    full = rows.get("magnet_layer")
    if full is None:
        return out + ["  ⛔ 无法判定：`magnet_layer`(full) 那格没有产物。"]

    drops: Dict[str, Dict[str, float]] = {}
    for cell, (_desc, is_abl) in CELLS.items():
        if not is_abl or cell not in rows:
            continue
        drops[cell] = {
            k: float(full[k]) - float(rows[cell][k])  # 正数 = 关掉该件后**下降**
            for k in PROGRESS_KEYS
            if full.get(k) is not None and rows[cell].get(k) is not None
        }
    if "no_progress" not in drops:
        return out + ["  ⛔ 无法判定：`no_progress` 那格没有产物。"]

    npr = drops["no_progress"]
    both_down = all(v > 0 for v in npr.values()) and len(npr) == len(PROGRESS_KEYS)
    others = {c: d for c, d in drops.items() if c != "no_progress"}
    missing = [c for c in ("no_dynamics", "no_collision", "no_rules") if c not in drops]

    for cell, d in drops.items():
        out.append(
            "  {:<13} Δprogress {:+.4f}｜Δmaking_prog {:+.4f}   （正数 = 关掉后下降）".format(
                cell, d.get("progress", float("nan")), d.get("making_prog", float("nan"))
            )
        )
    if missing:
        out.append(f"  ⚠️ 缺 {', '.join(missing)} ⇒ 「降幅大于其余三件的任一件」这半条**判不了**，只能判前半条。")

    larger = all(
        all(npr[k] > o.get(k, float("-inf")) for k in npr) for o in others.values()
    ) if others else None

    if both_down and larger and not missing:
        out.append("  ✅ **P1 成立**：两项双双下降，且降幅大于其余三件的任一件。")
    elif both_down and (larger is not False) and missing:
        out.append("  🟡 P1 前半条成立（两项双双下降），后半条待缺格补齐。")
    elif not both_down:
        out.append("  ❌ **P1 不成立**：两项没有双双下降 ⇒ 事前预测的主效应未复现，如实写。")
    else:
        out.append("  ❌ **P1 不成立**：虽双双下降，但降幅**未**大于其余三件的每一件。")
    return out


def judge_p2(rows: Dict[str, Dict[str, object]]) -> List[str]:
    """P2：规划层有没有把分数抬离地板。"""
    out = ["\n【P2】规划层把分数抬离地板？（判据：MAGNET+层零分比例 < 78.5%）"]
    full = rows.get("magnet_layer")
    if full is None or full.get("zero_frac") is None:
        return out + ["  ⛔ 无法判定：`magnet_layer` 那格没有产物或缺 log_name 列。"]
    zf = float(full["zero_frac"])
    off = rows.get("magnet_off", {}).get("zero_frac")
    out.append(f"  MAGNET+层零分比例 = {zf*100:.1f}%（{full['n_scenario']} 场景 / {full['n_log']} log）")
    if isinstance(off, float):
        out.append(f"  对照 层 OFF calibrated = {off*100:.1f}%（判据基线写死 {ZERO_FRAC_BASELINE*100:.1f}%）")
    if zf < ZERO_FRAC_BASELINE:
        out.append(f"  ✅ **P2 成立**：{zf*100:.1f}% < {ZERO_FRAC_BASELINE*100:.1f}%。")
    else:
        out.append(
            f"  ❌ **P2 不成立**：{zf*100:.1f}% ≥ {ZERO_FRAC_BASELINE*100:.1f}% ⇒ 地板没被抬起。"
            "\n     🔴 这本身就是可写的结果：R4-1 要求的规划形式化已实施，且在该平台上不提分。"
            "\n     **不得为了拿正数去调权重/网格** —— 改动须走干预协议 §11 并重跑全部格。"
        )
    return out


def judge_p3(rows: Dict[str, Dict[str, object]]) -> List[str]:
    """P3：学习组件有没有可测量的额外贡献（🔴 诚实性支点）。"""
    out = ["\n【P3】学习组件的额外贡献？（🔴 §4.4 要求：无论多小都报绝对值）"]
    m, c = rows.get("magnet_layer"), rows.get("cv_on")
    if m is None or c is None:
        return out + ["  ⛔ 无法判定：`magnet_layer` 或 `cv_on` 缺产物 —— **不得只报 MAGNET+层的绝对分数**。"]

    dm = float(m["final_score"]) - float(c["final_score"])
    out.append(f"  MAGNET+层 {float(m['final_score']):.4f}   CV+层 {float(c['final_score']):.4f}")
    out.append(f"  **Δfinal = {dm:+.4f}**（绝对值 {abs(dm):.4f}）")
    cv_off = rows.get("cv_off")
    if cv_off is not None:
        out.append(f"  trivial 下界 CV 层 OFF = {float(cv_off['final_score']):.4f}")
    out.append(
        "  🔴 措辞（§7 C4 第 3 条，二选一，不得只报绝对分数）：\n"
        "     · 若两者不可区分 ⇒ 只能写「在该规划形式化下学习组件无可测量的额外贡献」；\n"
        "     · 若可区分 ⇒ 仍须同报 CV+层 的绝对分数与本 Δ。"
    )
    return out


def print_writing_constraints() -> None:
    print(
        "\n=== 🔴 写作硬约束（主控表 §7 的 C4 条目，读表即适用）===\n"
        "  1. 规划层是**已知技术** —— 候选生成+打分+argmax 由 PDM `2306.07962` 占据，\n"
        "     必须显式引用并定位为「按 R4-1 意见实施的规划形式化」，不得写成本文提出的方法；\n"
        "  2. 必须声明**目标函数与官方评分同源** ⇒ 提升部分是同义反复，必须同报 leave-one-out；\n"
        "  3. 必须报 **CV + 层** 的对照（见 P3）；\n"
        "  4. **不得用 C4 的结果反过来重新表述 C2/C3** —— C3 是负结果，已锁死；\n"
        "  5. dev / 单 seed 数字不进论文表；闭环产物必须落 experiments/；\n"
        "  6. 必须写明两处工程近似：碰撞用**质心距离 2.5 m** 代理官方包围盒相交（保守方向）；\n"
        "     可行驶区域用 patch 并集代理逐点查询（真地图 3600 点零差异，\n"
        "     可写 \"verified identical on a 3600-point grid\"）；\n"
        "  7. 阈值全部取自 `nuplan/planning/script/config/common/simulation_metric/`，\n"
        "     不得写成自己调出来的超参。\n"
        "  ⚠️ 主表第四格 **PlanTF + 层尚未接线** —— 要么补接线，要么在论文里写明该格缺失的理由。"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="20260801", help="C4 各 group 的日期后缀（守卫在 08-01 启动 ⇒ 默认 20260801）")
    parser.add_argument("--csv", type=Path, default=None, help="可选：把主表写成 CSV")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = collect(args.date)
    if not rows:
        logger.error("一个格都没读到 —— 确认 %s/c4_*_%s/ 是否已有产物", CLOSED_LOOP, args.date)
        return 1

    print_table(rows)
    ran = [c for c in CELLS if c in rows]
    print(f"\n已出产物的格：{len(ran)}/{len(CELLS)} —— {', '.join(ran) if ran else '无'}")

    for block in (judge_p1(rows), judge_p2(rows), judge_p3(rows)):
        print("\n".join(block))
    print_writing_constraints()

    if args.csv:
        pd.DataFrame(rows).T.to_csv(args.csv)
        print(f"\n已写 CSV：{args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
