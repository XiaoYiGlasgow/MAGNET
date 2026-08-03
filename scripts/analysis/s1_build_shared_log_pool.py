#!/usr/bin/env python3
"""S1 —— 第二轮 scaling ladder 的**统一 log 池 + 分级清单**（2026-08-02 阶段58）。

为什么需要它（现有「预算对齐」不严格，实测比文档记的还差得多）：

============================  实测（本脚本 `--audit`）  ============================
                       log 数      样本数     样本/log     token 来源
  MAGNET  e1_none_v3     600       30 000        50       自家 danger/interaction 预处理
    └ train              480       24 000        50
  PlanTF  cache_L0        60       20 000       333       devkit challenge scenario_filter
==================================================================================

⇒ 文档（主控表 §0.0-c / §8）写的「PlanTF **200 log** / 100 样本每 log」**是错的**：
  实测 **60 log / 333 样本每 log**（目录法与 metadata csv 两法交叉验证一致）。
  200 是 `scenario_filter` 里**请求**的 log 数，cap 20000 在第 60 个 log 就撞满了 ——
  与 §9 纪律一（`Processing scenario i / N` 的 N 是分片计数不是全局数）**同一类误读，第三次**。

⇒ 真实错配比文档记的严重得多：log 数差 **8×**（480 vs 60，文档写 2.4×）、
  采样密度差 **6.7×**（50 vs 333，文档写 2×）。

🔴 **但最根本的错配不是数量，是 token 选取准则**：两者用的是**同一个 nuPlan token 空间**，
  可在共有的 57 个 log 上，MAGNET 2 850 token / PlanTF 19 784 token、**交集只有 259**。
  ⇒ 光对齐 log 数和样本数**不够**，必须让两条管线消费**同一份 (log, token) 清单**。

## 本脚本的设计

1. **共享 log 池** = `train_boston` + `train_pittsburgh` 全部 db，**扣掉闭环评测的 300 个
   `pittsburgh_heldout` log**（已实测两个训练池与它 **零重叠**，本脚本再断言一次）。
2. **确定性洗牌**（seed 固定），产出**嵌套**分级：L1 ⊂ L2 ⊂ L3 ⊂ L4 ——
   嵌套是为了让两条 scaling curve 的每一级都是「加数据」而不是「换数据」，
   否则级间差异里混入了 log 组成的变化。
3. 每级只定**log 清单**；token 由两条管线在**同一个官方 filter + 同一个 per-log cap +
   同一个确定性排序**下各自枚举 ⇒ 得到同一批 token，无需先把 2 907 个 db 全开一遍。
4. 产出 `s1_shared_log_pool_<seed>.json`，含每级 log 清单 + 全部计数 + 断言结果。

⚠️ **待裁定（写在这里，见 §4.2 Q56）**：token 池由**哪个 filter** 定义。
   默认取 **(a) 官方 14 类 challenge filter**（= 我方 `nuplan_challenge_scenarios.yaml`，
   已实测与官方 `test14-random.yaml` 14 类同名零差异）—— 理由是 benchmark 就是它，
   训练场景分布应与评测口径一致。代价：**MAGNET 必须按新 token 重做预处理**
   （S4 已预算 ≈103 GB / ≈96 h，本就在计划内）。

用法::

    python s1_build_shared_log_pool.py --audit          # 只做资产盘点与断言，不写文件
    python s1_build_shared_log_pool.py --build          # 生成分级清单
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SPLITS = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits")
HELDOUT_DIR = GNN / "data/closed_loop_sets/pittsburgh_heldout_clean"
MAGNET_MANIFEST = GNN / "data/processed_features_magnet_e1_none_v3/log_manifest.json"
MAGNET_SPLIT = GNN / "data/processed_features_magnet_e1_none_v3_split3/split3_manifest.json"
PLANTF_CACHE = GNN / "data/plantf_cache_L0"

TRAIN_SPLITS: Sequence[str] = ("train_boston", "train_pittsburgh")

# 事前定死的阶梯（主控表 §0.0-c）。MAGNET 只跑前三级，L4 仅 PlanTF。
TIERS: Dict[str, int] = {"L1": 25_000, "L2": 100_000, "L3": 300_000, "L4": 1_000_000}
MAGNET_MAX_TIER = "L3"

SEED_DEFAULT = 42
OUT_DEFAULT = GNN / "experiments/scaling_ladder"


@dataclass(frozen=True)
class Audit:
    """现有两条管线的实测资产（全部由本脚本现场数出来，不引用文档）。"""

    heldout_logs: frozenset
    magnet_logs: frozenset
    magnet_train_logs: frozenset
    magnet_tokens: int
    plantf_logs: frozenset
    plantf_tokens: int
    token_overlap: int
    shared_logs: int


def _db_stems(directory: Path) -> frozenset:
    return frozenset(p.stem for p in directory.glob("*.db"))


def audit() -> Audit:
    """现场盘点两条管线，并断言训练池与闭环评测集 log-disjoint。"""
    heldout = _db_stems(HELDOUT_DIR)

    magnet_rows = json.loads(MAGNET_MANIFEST.read_text())
    magnet_by_log: Dict[str, set] = collections.defaultdict(set)
    for row in magnet_rows:
        magnet_by_log[row["log_name"]].add(row["token"])
    split = json.loads(MAGNET_SPLIT.read_text())
    magnet_train = frozenset(split["logs"]["train"])

    plantf_by_log: Dict[str, set] = collections.defaultdict(set)
    for log_dir in PLANTF_CACHE.iterdir():
        if not log_dir.is_dir() or log_dir.name == "metadata":
            continue
        for type_dir in log_dir.iterdir():
            if type_dir.is_dir():
                plantf_by_log[log_dir.name].update(t.name for t in type_dir.iterdir() if t.is_dir())

    shared = set(magnet_by_log) & set(plantf_by_log)
    overlap = sum(len(magnet_by_log[log] & plantf_by_log[log]) for log in shared)

    result = Audit(
        heldout_logs=heldout,
        magnet_logs=frozenset(magnet_by_log),
        magnet_train_logs=magnet_train,
        magnet_tokens=sum(len(v) for v in magnet_by_log.values()),
        plantf_logs=frozenset(plantf_by_log),
        plantf_tokens=sum(len(v) for v in plantf_by_log.values()),
        token_overlap=overlap,
        shared_logs=len(shared),
    )

    # 🔴 硬断言：训练池与闭环评测集必须 log-disjoint，否则既有的 0.1978 / 0.3565 全部作废。
    for name, pool in (("MAGNET", result.magnet_logs), ("PlanTF", result.plantf_logs)):
        leak = pool & heldout
        if leak:
            raise AssertionError(f"{name} 训练池与闭环 heldout 有 {len(leak)} 个 log 重叠 ⇒ 分数被污染")
    return result


def print_audit(a: Audit) -> None:
    def density(tokens: int, logs: int) -> float:
        return tokens / logs if logs else float("nan")

    print("\n=== 现有资产实测（不引用文档）===")
    print(f"  闭环评测 heldout        : {len(a.heldout_logs):5d} log")
    print(f"  MAGNET 全池             : {len(a.magnet_logs):5d} log｜{a.magnet_tokens:6d} token"
          f"｜{density(a.magnet_tokens, len(a.magnet_logs)):6.1f} 样本/log")
    print(f"    └ train               : {len(a.magnet_train_logs):5d} log")
    print(f"  PlanTF L0 cache         : {len(a.plantf_logs):5d} log｜{a.plantf_tokens:6d} token"
          f"｜{density(a.plantf_tokens, len(a.plantf_logs)):6.1f} 样本/log")
    print(f"\n  🔴 log 数之比           : {len(a.magnet_train_logs) / max(len(a.plantf_logs), 1):.1f}×"
          f"（文档记 2.4×）")
    print(f"  🔴 采样密度之比         : "
          f"{density(a.plantf_tokens, len(a.plantf_logs)) / density(a.magnet_tokens, len(a.magnet_logs)):.1f}×"
          f"（文档记 2×）")
    print(f"  🔴 共有 {a.shared_logs} log 上的 token 交集: {a.token_overlap}"
          f"（⇒ 同一 token 空间，但选取准则不同）")
    print("\n  ✅ 断言通过：两个训练池与闭环 heldout 均 log-disjoint（0 重叠）")


def build_pool(seed: int, samples_per_log: int) -> Dict[str, object]:
    """建共享 log 池并切出**嵌套**分级清单。"""
    a = audit()

    available: List[str] = []
    per_split: Dict[str, int] = {}
    for split_name in TRAIN_SPLITS:
        stems = _db_stems(SPLITS / split_name)
        per_split[split_name] = len(stems)
        available.extend(sorted(stems - a.heldout_logs))
    available = sorted(set(available))

    rng = random.Random(seed)
    shuffled = available[:]
    rng.shuffle(shuffled)

    tiers: Dict[str, Dict[str, object]] = {}
    for name, target in TIERS.items():
        n_log = -(-target // samples_per_log)  # 向上取整
        if n_log > len(shuffled):
            logger.warning("%s 需要 %d log，池里只有 %d ⇒ 该级不可达", name, n_log, len(shuffled))
        tiers[name] = {
            "target_samples": target,
            "n_log": min(n_log, len(shuffled)),
            "reachable": n_log <= len(shuffled),
            "magnet_runs": name <= MAGNET_MAX_TIER,
            "logs": shuffled[: min(n_log, len(shuffled))],
        }

    # 嵌套性断言：L1 ⊂ L2 ⊂ L3 ⊂ L4
    order = list(TIERS)
    for lo, hi in zip(order, order[1:]):
        if not set(tiers[lo]["logs"]).issubset(set(tiers[hi]["logs"])):
            raise AssertionError(f"{lo} 不是 {hi} 的子集 ⇒ 级间差异混入 log 组成变化")

    return {
        "seed": seed,
        "samples_per_log": samples_per_log,
        "note": "两条管线必须消费同一份 log 清单 + 同一个官方 filter + 同一个 per-log cap，"
                "否则 token 不同（实测同 57 log 上 token 交集仅 259/2850）",
        "token_filter": "官方 14 类 challenge filter（待 Q56 签字；见模块 docstring）",
        "splits_used": per_split,
        "n_heldout_excluded": len(a.heldout_logs),
        "n_available": len(available),
        "audit": {
            "magnet_logs": len(a.magnet_logs),
            "magnet_train_logs": len(a.magnet_train_logs),
            "magnet_tokens": a.magnet_tokens,
            "plantf_logs": len(a.plantf_logs),
            "plantf_tokens": a.plantf_tokens,
            "token_overlap_on_shared_logs": a.token_overlap,
            "shared_logs": a.shared_logs,
        },
        "tiers": tiers,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", action="store_true", help="只盘点与断言，不写文件")
    parser.add_argument("--build", action="store_true", help="生成分级清单")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--samples-per-log", type=int, default=333,
                        help="每 log 取多少样本；默认 333 = PlanTF L0 实测密度")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.audit or not args.build:
            print_audit(audit())
            if not args.build:
                return 0

        pool = build_pool(args.seed, args.samples_per_log)
    except (AssertionError, FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"s1_shared_log_pool_seed{args.seed}.json"
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2))

    print(f"\n=== 分级清单（seed={args.seed}，{args.samples_per_log} 样本/log）===")
    print(f"  可用 log = {pool['n_available']}（已扣掉 {pool['n_heldout_excluded']} 个闭环 heldout）")
    for name, tier in pool["tiers"].items():
        mark = "✅" if tier["reachable"] else "❌不可达"
        who = "两模型" if tier["magnet_runs"] else "仅 PlanTF"
        print(f"  {name}: {tier['target_samples']:>9,} 样本 ← {tier['n_log']:>5} log  {mark}  ({who})")
    print(f"  ✅ 嵌套性断言通过：L1 ⊂ L2 ⊂ L3 ⊂ L4")
    print(f"\n  已写 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
