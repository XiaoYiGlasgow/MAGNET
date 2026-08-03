#!/usr/bin/env python3
"""S1 —— 实测两条管线的**真实每 log 场景产出密度**（2026-08-02 阶段60）。

## 为什么必须实测

这个项目今天一天在同一个坑上摔了三次，**全部是把 cap 当成了密度**：

1. `PlanTF 200 log` —— 是 `scenario_filter` 里**请求**的 log 数；cap 20000 在第 60 个 log 撞满 ⇒ 实际 60。
2. `PlanTF 333 样本/log` —— 20000/60，同样是 cap 的产物；真实密度按 L1 观测 ≥794。
3. `MAGNET 50 样本/log` —— 600/600 个 log 全是**恰好 50**（min=p25=median=max=50）
   ⇒ 是 `run_e1_preprocess.py:122` 的 `--per_db_cap` 默认值，不是密度。

外推同样不可信（坑#43：用 train_boston 的密度估 val/test 为 12–16 G，实测 91 G + 90 G，**差 6 倍**）。
⇒ **本脚本不外推、不引用任何文档数字，只在真实 DB 上数。**

## 量什么

对同一批随机抽的 log，在**两套 ScenarioFilter** 下各数一遍：

| | scenario_types | expand_scenarios | remove_invalid_goals | cap |
|---|---|---|---|---|
| `magnet_current` | None | **False** | **False** | per-DB 50 |
| `plantf_native`  | None | **True**  | **True**  | 无 |

`plantf_native` 是 PlanTF 发表时的 recipe（`training_scenarios_1M.yaml`），
第二轮拟采用它作为**两条管线共享的枚举器**（理由见主控表 §0.0-c 修订版：
参照基线必须保持其发表 recipe，被研究对象迁就它）。

⚠️ 两者的差别只有三个 flag —— 08-02 查实 MAGNET 的 imitation 线预处理
（`run_e1_preprocess.py:92-97`）本来就是 `scenario_types=None` 的无类型随机采样，
**不是** v10 danger 线的 TTC/交互危险对挖掘。此前把两者混为一谈是错的。

## 产出

- 每 log 在两套 filter 下的场景数（min/median/max，不只报均值）
- 由中位密度推出四级阶梯各需多少 log，并对 2 707 的可用池判可达性
- **不给单一数字** —— 密度在 log 之间方差很大（PlanTF cache 实测每 log 32–3492），
  用中位数排期、用 p25 做保守下界

用法::

    python s1_measure_scenario_density.py --n-logs 8
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"

HELDOUT_DIR = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/closed_loop_sets/pittsburgh_heldout_clean")
TRAIN_SPLITS = ("train_boston", "train_pittsburgh")

# 事前定死的阶梯（主控表 §0.0-c）
TIERS: Dict[str, int] = {"L1": 25_000, "L2": 100_000, "L3": 300_000, "L4": 1_000_000}

# 两套 filter 的差异只有这三个 flag
FILTERS: Dict[str, Dict[str, object]] = {
    "magnet_current": {"expand_scenarios": False, "remove_invalid_goals": False, "limit_total_scenarios": 50},
    "plantf_native": {"expand_scenarios": True, "remove_invalid_goals": True, "limit_total_scenarios": None},
}


def sample_logs(n: int, seed: int) -> List[str]:
    """从训练池（已扣掉闭环 heldout）里随机抽 n 个 db 绝对路径。"""
    heldout = {p.stem for p in HELDOUT_DIR.glob("*.db")}
    pool: List[Path] = []
    for split in TRAIN_SPLITS:
        pool.extend(p for p in (Path(DATA_ROOT) / split).glob("*.db") if p.stem not in heldout)
    pool.sort()
    rng = random.Random(seed)
    rng.shuffle(pool)
    return [str(p) for p in pool[:n]]


def count_one(db: str, flags: Dict[str, object]) -> Tuple[int, float]:
    """在单个 db 上按给定 flag 数出场景数，返回 (场景数, 耗时秒)。"""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=[db], map_version=MAP_VERSION,
    )
    sf = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=flags["limit_total_scenarios"],
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        expand_scenarios=flags["expand_scenarios"],
        remove_invalid_goals=flags["remove_invalid_goals"],
        shuffle=True,
    )
    t0 = time.time()
    scenarios = builder.get_scenarios(sf, Sequential())
    return len(scenarios), time.time() - t0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-logs", type=int, default=8, help="抽几个 log 实测（密度方差大，别少于 5）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--available-logs", type=int, default=2707,
                        help="训练池可用 log 数（3207 − 300 闭环 − 200 开环验证）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    dbs = sample_logs(args.n_logs, args.seed)
    print(f"\n=== 实测 {len(dbs)} 个 log 的真实场景密度（seed={args.seed}）===")
    print("   ⚠️ 不引用任何文档数字；cap 与密度必须分清（今天已踩三次）\n")

    counts: Dict[str, List[int]] = {name: [] for name in FILTERS}
    for i, db in enumerate(dbs, 1):
        name_short = Path(db).stem
        row = []
        for fname, flags in FILTERS.items():
            try:
                n, secs = count_one(db, flags)
            except Exception as exc:  # 深层坏 DB：跳过不炸整轮（与 run_e1_preprocess 同策略）
                logger.warning("跳过 %s / %s: %s", name_short, fname, exc)
                n, secs = -1, 0.0
            if n >= 0:
                counts[fname].append(n)
            row.append(f"{fname}={n:>5}({secs:4.1f}s)")
        print(f"  [{i}/{len(dbs)}] {name_short[:40]:40s} {'  '.join(row)}")

    print("\n=== 密度汇总（每 log 场景数）===")
    summary: Dict[str, Dict[str, float]] = {}
    for fname, vals in counts.items():
        if not vals:
            print(f"  {fname}: 全部失败")
            continue
        v = sorted(vals)
        s = {
            "n_log": len(v), "min": v[0], "p25": v[len(v) // 4],
            "median": statistics.median(v), "max": v[-1], "mean": statistics.mean(v),
        }
        summary[fname] = s
        print(f"  {fname:16s}: min={s['min']:>5} p25={s['p25']:>5} "
              f"median={s['median']:>7.1f} max={s['max']:>5} (n={s['n_log']} log)")

    if "plantf_native" in summary:
        med = summary["plantf_native"]["median"]
        p25 = summary["plantf_native"]["p25"]
        print(f"\n=== 阶梯可达性（共享枚举器 = plantf_native，池 {args.available_logs} log）===")
        print(f"   用**中位数 {med:.0f}**排期，**p25 {p25}** 作保守下界\n")
        for name, target in TIERS.items():
            need_med = -(-target // int(med)) if med else 0
            need_p25 = -(-target // int(p25)) if p25 else 0
            # 🔴 p25 可能为 0（存在整条 log 被 remove_invalid_goals 清空的情况）
            # ⇒ need_p25 会被算成 0 而误判为「✅ 可达」。判据里的每个变量都要先验它是不是有效正整数。
            if not p25:
                ok = "⚠️ p25=0（有 log 被清空），保守下界不可用，只能按中位排期"
            elif need_p25 <= args.available_logs:
                ok = "✅"
            elif need_med <= args.available_logs:
                ok = "⚠️ 中位可达 / 保守不可达"
            else:
                ok = "❌ 不可达"
            print(f"  {name}: {target:>9,} 样本 ← 中位 {need_med:>5} log / 保守 {need_p25:>5} log   {ok}")

    print("\n⚠️ 这是密度，不是 cap。排期时两条管线必须消费同一份 (log, token) 清单，")
    print("   否则样本数对齐了、token 仍然不同（08-02 实测同 57 log 上交集仅 259/2850）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
