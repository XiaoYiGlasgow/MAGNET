#!/usr/bin/env python3
"""S3/S6 —— PlanTF scaling ladder 的开环读数（Q60 + Q65 口径）。

读数口径（**已在见到末段数字之前签字，不得因数字不好看再改**）：

  - primary   : ``val/minADE1`` / ``val/minFDE1``
                = 概率最高那条模态（``sort_predictions(..., k=1)``），
                  正是闭环 planner 执行的那条（``planning_model.py:135`` 的
                  ``best_mode = probability.argmax(dim=-1)``）。
  - secondary : ``val/minADE6`` / ``val/minFDE6``
                🔴 best-of-6 **oracle** 选择（``min_ade.py:52`` 的 ``ade.min(-1)``），
                闭环里永远不会被执行 ⇒ **不得与 MAGNET 单模态 ADE 并列比较**。
  - 「末 5 epoch 均值」在 PlanTF 侧 = **末 10 次 val 读数**
    （``val_check_interval=0.5`` ⇒ 每 epoch 2 次 val）。
  - 🔴 **不用 ``checkpoint_best``**：它在抖动序列上取 min，实测有 0.2188 m 系统性向下偏倚。

输出三层数字：

  1. 每 seed 的末 N 次读数均值（= 该 seed 的点估计）；
  2. 三 seed 的 mean ± sd（= 该级的读数）；
  3. 🔴 **实测 σ_avg5 及其相对值** —— Q60 的功效前提 σ_avg5=0.069 是在 **MAGNET**
     （ADE≈2.1 m 尺度）上测的，PlanTF minADE1 在 ≈4.4–5.0 m 尺度，
     **α ≥ 0.045 这个门槛能否直接套用，必须用本项验证**。

用法::

    python3 scripts/analysis/s3_readout_last5.py --tier L1
    python3 scripts/analysis/s3_readout_last5.py --tier L1 --last_n 4 --self_test

``--self_test`` 用早段读数自测脚本正确性（此时末段尚未产生，不构成偷看）。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")

# Q65 签字口径：primary 在前，secondary 在后；顺序即报告顺序。
PRIMARY_TAGS = ("val/minADE1", "val/minFDE1")
SECONDARY_TAGS = ("val/minADE6", "val/minFDE6")
ALL_TAGS = PRIMARY_TAGS + SECONDARY_TAGS

# 「末 5 epoch」在 PlanTF 侧 = 末 10 次 val 读数（val_check_interval=0.5）。
DEFAULT_LAST_N = 10


def find_run_dir(tier: str, seed: int, root: Optional[Path] = None) -> Path:
    """定位某 seed 的 lightning 产物目录（含 tfevents）。"""
    base = (root or GNN / "experiments" / "scaling_ladder" / f"s3_plantf_{tier}") / f"seed{seed}"
    cands = sorted(base.glob("training/planTF/2*"))
    if not cands:
        raise FileNotFoundError(f"{base} 下没有 training/planTF/2* 产物目录")
    if len(cands) > 1:
        # 断点重跑会留多个；取最新的那个，但必须报出来（provenance）
        logger.warning("seed %d 有 %d 个产物目录，取最新：%s", seed, len(cands), cands[-1].name)
    return cands[-1]


def load_scalars(run_dir: Path, tags: Sequence[str]) -> Dict[str, List[float]]:
    """读 tfevents 里指定 tag 的完整 scalar 序列（按 step 升序）。"""
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    missing = [t for t in tags if t not in have]
    if missing:
        raise KeyError(
            f"{run_dir} 缺 tag {missing}；实际有 {sorted(have)}。"
            "⚠️ 不要静默换 tag —— 口径已由 Q65 签字锁定。"
        )
    return {t: [ev.value for ev in ea.Scalars(t)] for t in tags}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: Sequence[float]) -> float:
    """样本标准差（ddof=1）。n<2 时返回 nan，绝不返回 0 冒充「无噪声」。"""
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def readout_seed(run_dir: Path, last_n: int) -> Dict[str, object]:
    """单 seed 读数：末 last_n 次 val 读数的均值 + 该 seed 自身的段内 sd。"""
    series = load_scalars(run_dir, ALL_TAGS)
    n_read = len(series[PRIMARY_TAGS[0]])
    if n_read < last_n:
        raise ValueError(
            f"{run_dir} 只有 {n_read} 次 val 读数 < 末 {last_n} 次 ⇒ 训练未跑完，拒绝读数。"
        )
    out: Dict[str, object] = {"run_dir": str(run_dir), "n_val_readings": n_read}
    for t in ALL_TAGS:
        tail = series[t][-last_n:]
        out[t] = _mean(tail)
        out[f"{t}__tail_sd"] = _sd(tail)   # 段内抖动，用于说明为什么要取均值
    return out


def readout_tier(tier: str, seeds: Sequence[int], last_n: int,
                 root: Optional[Path] = None) -> Dict[str, object]:
    """整级读数：三 seed 的 mean ± sd，并实测 σ_avg5 的相对值。"""
    per_seed = {}
    for s in seeds:
        per_seed[s] = readout_seed(find_run_dir(tier, s, root), last_n)

    agg: Dict[str, object] = {}
    for t in ALL_TAGS:
        vals = [per_seed[s][t] for s in seeds]
        m, sd = _mean(vals), _sd(vals)
        agg[t] = {
            "mean": m,
            "sd": sd,
            # 🔴 Q60 的 α 门槛是在 MAGNET 的 ADE 尺度上定的；跨模型套用必须看相对噪声
            "rel_sd": sd / m if m else float("nan"),
            "per_seed": {str(s): per_seed[s][t] for s in seeds},
        }
    return {
        "tier": tier,
        "seeds": list(seeds),
        "last_n_readings": last_n,
        "readout_convention": (
            "Q60 末5epoch均值 + Q65：primary=val/minADE1|minFDE1（闭环执行的那条）；"
            "secondary=val/minADE6|minFDE6（best-of-6 oracle，不得与 MAGNET 单模态并列）；"
            "末5epoch=末10次val读数（val_check_interval=0.5）；不用 checkpoint_best"
        ),
        "per_seed_detail": {str(s): per_seed[s] for s in seeds},
        "aggregate": agg,
    }


def _fmt(v: float) -> str:
    return "  nan " if math.isnan(v) else f"{v:.4f}"


def print_report(res: Dict[str, object]) -> None:
    agg = res["aggregate"]  # type: ignore[index]
    print(f"\n=== {res['tier']} 开环读数（末 {res['last_n_readings']} 次 val 读数均值，"
          f"seeds {res['seeds']}）===")
    print("口径：", res["readout_convention"])

    for label, tags in (("🎯 primary（闭环实际执行的轨迹）", PRIMARY_TAGS),
                        ("🔴 secondary（best-of-6 ORACLE，禁止与 MAGNET 单模态并列）", SECONDARY_TAGS)):
        print(f"\n{label}")
        for t in tags:
            a = agg[t]
            ps = "｜".join(f"s{k}={v:.4f}" for k, v in a["per_seed"].items())
            print(f"  {t:<16} {a['mean']:.4f} ± {_fmt(a['sd'])}"
                  f"   (rel_sd {a['rel_sd']*100:.2f}%)   [{ps}]")

    # 🔴 Q60 功效前提的跨模型可移植性检验
    ade = agg[PRIMARY_TAGS[0]]
    print("\n--- 🔴 Q60 功效前提检验（σ_avg5 能否从 MAGNET 移植到 PlanTF）---")
    print(f"  MAGNET 实测 : σ_avg5 = 0.0690 m @ ADE≈2.135 m  ⇒ 相对噪声 3.23%")
    print(f"  PlanTF 实测 : σ      = {_fmt(ade['sd'])} m @ minADE1≈{ade['mean']:.3f} m"
          f"  ⇒ 相对噪声 {ade['rel_sd']*100:.2f}%")
    ratio = ade["rel_sd"] / 0.0323 if not math.isnan(ade["rel_sd"]) else float("nan")
    print(f"  相对噪声比  : {_fmt(ratio)}×")
    if not math.isnan(ratio):
        if ratio <= 1.2:
            print("  ⇒ 🟢 PlanTF 相对噪声不高于 MAGNET ⇒ α ≥ 0.045 的门槛可沿用，k=3 够")
        elif ratio <= 2.0:
            print(f"  ⇒ 🟠 相对噪声高 {ratio:.2f}× ⇒ 所需 α 门槛按同比例放大到 ≈{0.045*ratio:.3f}，"
                  "L1→L2 实测 α 后须用这个新门槛判 L3")
        else:
            print(f"  ⇒ 🔴 相对噪声高 {ratio:.2f}× ⇒ k=3 在 PlanTF 侧功效不足，"
                  "L3 判据必须重算（否则重演 k=1 测不出来的老问题）")
    print("  ⚠️ n=3 的 sd 本身极不稳定，本行只作量级判断，不作精确功效结论。")


def main() -> None:
    p = argparse.ArgumentParser(description="PlanTF scaling ladder 开环读数（Q60+Q65 口径）")
    p.add_argument("--tier", required=True, choices=["L1", "L2", "L3"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--last_n", type=int, default=DEFAULT_LAST_N,
                   help=f"末 N 次 val 读数（默认 {DEFAULT_LAST_N} = 末 5 epoch × 2 次/epoch）")
    p.add_argument("--root", type=str, default=None, help="覆盖产物根目录（自测用）")
    p.add_argument("--out", type=str, default=None, help="JSON 输出路径")
    p.add_argument("--self_test", action="store_true",
                   help="自测模式：允许 last_n 小于默认值，用早段读数验证脚本正确性")
    args = p.parse_args()

    if args.last_n != DEFAULT_LAST_N and not args.self_test:
        raise SystemExit(
            f"🔴 last_n={args.last_n} 偏离签字口径 {DEFAULT_LAST_N}。"
            "正式读数不得改 last_n；自测请显式加 --self_test。"
        )

    root = Path(args.root).resolve() if args.root else None
    res = readout_tier(args.tier, args.seeds, args.last_n, root)
    if args.self_test:
        res["SELF_TEST"] = True
        res["warning"] = "自测读数，last_n 偏离签字口径，不得作为正式结果引用"
    print_report(res)

    out = Path(args.out) if args.out else (
        GNN / "experiments" / "scaling_ladder" / f"s3_readout_{args.tier}"
        f"{'_selftest' if args.self_test else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
