#!/usr/bin/env python3
"""S1 —— 两条管线**共享的 (log, token) 清单**（2026-08-02 阶段61）。

## 设计（阶段60 实测后定稿）

**固定每 log 帧数、只放大 log 数。** 这样级与级之间变的是「更多样的驾驶」，
而不是「同一批驾驶采得更密」—— 后者不是一个有意义的 data-scaling 实验。

| 级 | 样本数 | log 数 | 说明 |
|---|---|---|---|
| L1 | 25 000 | 250 | |
| L2 | 75 000 | 750 | |
| L3 | ~271 000 | 2707（**全部**） | 上限：用光了我们拥有的每一个 log |

~~L4 1M~~ **不做**：在恒定冗余度（100 帧/log）下需要约 10 000 个 log，
而两城只有 2707。硬凑 1M 意味着 369 帧/log，是 PlanTF 官方 1M（≈67 帧/log）的 **5.5× 冗余**，
两个「1M」不是同一个东西，画同一条横轴会误导。

## 为什么每 log 取 100 帧

接近 PlanTF 官方 1M 的 ≈67 帧/log，且是 MAGNET 现行 50 的 2×。
阶段60 实测每 log 可用帧数均值 2425（`remove_invalid_goals=True`），100 远在其下，
⇒ 绝大多数 log 都取得满，不会因密度不足而使各 log 权重不等。

## 确定性

🔴 **不用 `ScenarioFilter(shuffle=True)`** —— 那是 devkit 内部的 RNG，我们控制不了种子。
改为 `shuffle=False` 全量枚举 → 按 token 字典序排序 → 用**我们自己的 seeded RNG** 抽 100 个。
这样同一个 seed 任何时候重跑都得到逐位相同的清单。

## 嵌套性

L1 ⊂ L2 ⊂ L3：log 池洗牌一次后按前缀切；每 log 取的那 100 个 token 在所有包含该 log 的级里
**完全相同** ⇒ 天然嵌套，级间差异只有「新加进来的 log」。

## 两条管线怎么用它

两边都读同一个 json，按 `scenario_tokens=` 建 ScenarioFilter，**谁都不许自己再抽一次**。
⚠️ 08-02 实测：不共享清单时，同 57 个 log 上两边 token 交集只有 259/2850。

用法::

    python s1_build_shared_token_manifest.py --build          # 全量（2707 log，约 3 min）
    python s1_build_shared_token_manifest.py --build --max-logs 60   # 快速试
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"

HELDOUT_DIR = GNN / "data/closed_loop_sets/pittsburgh_heldout_clean"
TRAIN_SPLITS = ("train_boston", "train_pittsburgh")
OUT_DIR = GNN / "experiments/scaling_ladder"

FRAMES_PER_LOG = 100
SEED = 42

# 开环验证集：从池子里先切走，两模型共用；绝不进任何训练级。
N_VAL_LOGS = 200

# (级名, log 数)；L3 = 全部剩余
TIER_LOGS: Tuple[Tuple[str, Optional[int]], ...] = (("L1", 250), ("L2", 750), ("L3", None))

# 🔴 Q59 已由实测裁定 = `remove_invalid_goals: False`（阶段61，同一批 30 log 对拍）：
#      rig=False：空 log 6/30 (20%)、≥100 帧的 log 24/30 ⇒ 2707 池约 2166 可用 → 上限 **217k**
#      rig=True ：空 log 22/30 (73%)、≥100 帧的 log  8/30 ⇒ 2707 池约  722 可用 → 上限 **72k**
#   True 会把阶梯跨度压到 2.9×（看不出饱和），且分布是**双峰**的（log 要么 0 帧要么 ≥100 帧）
#   ⇒ 调低每 log 帧数救不回被清空的那些 log。
#   ⚠️ 这是对 PlanTF 发表 recipe 的**一处偏离**，必须在论文里显式声明；
#      但它**对两个模型同样施加**，配对比较不受影响。MAGNET 现行管线本来就是 False。
#      🔴 PlanTF 侧在 rig=False 下未测过 ⇒ cache 前必须先冒烟。
# expand_scenarios 阶段60 已证在 scenario_types=None 下完全无作用，留 False 更省。
FILTER_FLAGS = {
    "scenario_types": None,
    "num_scenarios_per_type": None,
    "limit_total_scenarios": None,
    "expand_scenarios": False,
    "remove_invalid_goals": False,
    "shuffle": False,          # 🔴 必须 False，确定性由我们自己的 RNG 提供
}


def build_pool() -> List[Path]:
    """训练+验证可用的 log（已扣掉闭环 heldout），确定性排序。"""
    heldout = {p.stem for p in HELDOUT_DIR.glob("*.db")}
    pool: List[Path] = []
    for split in TRAIN_SPLITS:
        pool.extend(p for p in (Path(DATA_ROOT) / split).glob("*.db") if p.stem not in heldout)
    return sorted(pool)


def tokens_of(db: Path) -> List[str]:
    """枚举单个 db 的全部场景 token，按字典序（确定性）。"""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=[str(db)], map_version=MAP_VERSION,
    )
    sf = ScenarioFilter(scenario_tokens=None, log_names=None, map_names=None,
                        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                        **FILTER_FLAGS)
    return sorted(s.token for s in builder.get_scenarios(sf, Sequential()))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--max-logs", type=int, default=None, help="只枚举前 N 个 log（快速试）")
    parser.add_argument("--frames-per-log", type=int, default=FRAMES_PER_LOG)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.build:
        parser.error("要 --build")

    pool = build_pool()
    rng = random.Random(args.seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    val_logs = shuffled[:N_VAL_LOGS]
    train_logs = shuffled[N_VAL_LOGS:]
    if args.max_logs:
        train_logs = train_logs[: args.max_logs]

    print(f"\n=== 枚举 (log, token)（seed={args.seed}，{args.frames_per_log} 帧/log）===")
    print(f"  池 {len(pool)} log = 开环验证 {len(val_logs)} + 训练可用 {len(train_logs)}")
    print(f"  filter: remove_invalid_goals={FILTER_FLAGS['remove_invalid_goals']}, shuffle=False（确定性）\n")

    picked: Dict[str, List[str]] = {}
    empty: List[str] = []
    broken: List[str] = []
    short: List[str] = []
    t0 = time.time()
    for i, db in enumerate(train_logs, 1):
        try:
            toks = tokens_of(db)
        except Exception as exc:
            broken.append(db.stem)
            logger.warning("坏 DB %s: %s", db.stem, exc)
            continue
        if not toks:
            empty.append(db.stem)
            continue
        if len(toks) < args.frames_per_log:
            short.append(db.stem)
        # 用自己的 RNG 抽，seed 绑 log 名 ⇒ 与枚举顺序无关，可逐位复现
        r = random.Random(f"{args.seed}:{db.stem}")
        picked[db.stem] = sorted(r.sample(toks, min(args.frames_per_log, len(toks))))
        if i % 200 == 0:
            print(f"  进度 {i}/{len(train_logs)}｜有效 {len(picked)}｜空 {len(empty)}｜坏 {len(broken)}"
                  f"｜{time.time() - t0:.0f}s")

    usable = [s for s in (d.stem for d in train_logs) if s in picked]
    print(f"\n  枚举完成 {time.time() - t0:.0f}s：有效 {len(usable)} log｜"
          f"空 {len(empty)}｜坏 {len(broken)}｜不足 {args.frames_per_log} 帧 {len(short)}")

    tiers: Dict[str, Dict[str, object]] = {}
    for name, n_log in TIER_LOGS:
        sel = usable if n_log is None else usable[:n_log]
        n_sample = sum(len(picked[s]) for s in sel)
        tiers[name] = {"n_log": len(sel), "n_sample": n_sample, "logs": sel}

    order = [n for n, _ in TIER_LOGS]
    for lo, hi in zip(order, order[1:]):
        if not set(tiers[lo]["logs"]).issubset(set(tiers[hi]["logs"])):
            raise AssertionError(f"{lo} 不是 {hi} 的子集")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"s1_shared_token_manifest_seed{args.seed}.json"
    out.write_text(json.dumps({
        "seed": args.seed,
        "frames_per_log": args.frames_per_log,
        "filter_flags": FILTER_FLAGS,
        "note": "两条管线必须读本文件、按 scenario_tokens 建 filter，谁都不许自己再抽一次",
        "n_pool": len(pool),
        "val_logs": [d.stem for d in val_logs],
        "empty_logs": empty, "broken_logs": broken, "short_logs": short,
        "tiers": {k: {"n_log": v["n_log"], "n_sample": v["n_sample"], "logs": v["logs"]}
                  for k, v in tiers.items()},
        "tokens": picked,
    }, ensure_ascii=False), encoding="utf-8")

    print("\n=== 阶梯 ===")
    for name, t in tiers.items():
        print(f"  {name}: {t['n_log']:>5} log → {t['n_sample']:>7,} 样本")
    print("  ✅ 嵌套性断言通过")
    print(f"\n  已写 {out}（{out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
