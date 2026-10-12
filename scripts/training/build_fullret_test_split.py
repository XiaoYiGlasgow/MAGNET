#!/usr/bin/env python3
"""新档（fullret L1）**独立 test split** 构建器 —— L-15 裁定 (A) 的落地（2026-08-18）。

## 为什么必须新建（三条实测，见 `progress_20260818.md` 阶段204）

1. 🔴 **splitter yaml 的 `test` 键是 `val` 的逐字副本** —— `gnn_scaling_L{1,2,3}.yaml` 与
   `gnn_selftrain_L0.yaml` 四个文件全部 `val == test`（程序化比对）。照它建 test，
   "test" 就是 early stopping / checkpoint 选择用的同一批 log，**正是 R4-7 原批评的再犯**；
2. 旧的 480/60/60 三分 split（`processed_features_magnet_e1_none_v3_split3`）属**旧数据契约**，
   新档 ckpt 用不了（好消息：它至今未被打开，`find experiments -name test_metrics.json` = 0）；
3. 新档已预处理 **227 344 token / 2292 log**，L1 只用了 236 log ⇒ **2056 log / 204 008 token 未用**，
   且与闭环评测的 117 log **交集为 0**（闭环用 pittsburgh_heldout 300，本就不在共享池里）
   ⇒ 从未用 log 里切 test，对**训练**与**闭环评测**同时 log-disjoint，且是纯 manifest/symlink 操作。

## 口径（建之前写死）

- **整 log 分配**（file 级 partition），同一 nuPlan log 绝不跨 split；
- 选取规则确定性可复现：`sorted(未用 log)` 后用 `random.Random(seed)` 抽 `--n_logs` 个，再排序落盘；
- token 来源 = `s4_log_manifest_from_s1.json`（L1 侧唯一正确 manifest，§8.2），
  与 train/val 同一份**官方 14 类 challenge filter + per-log cap**，口径一致；
- 物化方式与既有 train/val 完全一致：**absolute symlink** 指向 `processed_features_fullret/`，
  `GraphDataset.process()` 的 `os.walk` 可直接吃；
- 🔴 **三向硬断言**：test ∩ train = ∅、test ∩ val = ∅、test ∩ 闭环 log = ∅；缺失 pkl 当场停。

## 🔴 冻结纪律（test 只开一次）

建完写 `.TEST_SPLIT_FROZEN`（含规则、seed、log 清单 sha256、建立时间）。
**本脚本拒绝覆盖已存在的 test root** —— 想改规模/规则必须显式 `--force` 并在工作记录里说明理由，
否则「test opened once」失去指称对象（letter L420/L695 的承诺）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/training/build_fullret_test_split.py \\
        --n_logs 60 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_test_split")

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
MANIFEST = GNN / "experiments/scaling_ladder/s4_log_manifest_from_s1.json"
SPLITTER_YAML = Path("/mnt/local/Xiao_Yi/external/planTF/config/splitter/gnn_scaling_L1.yaml")
SOURCE_POOL = GNN / "data/processed_features_fullret"
SPLIT_ROOT = GNN / "data/processed_features_fullret_L1_logsplit"
CLOSED_LOOP_REF = GNN / "experiments/closed_loop/f_closed_loop_paired_calibrated_20260817/mask_none"


def closed_loop_logs(root: Path) -> Set[str]:
    """闭环评测实际用到的 log 名（从 aggregator parquet 读，不靠记忆）。"""
    import glob

    import pandas as pd
    hits = glob.glob(str(root / "**/aggregator_metric/*weighted_average*.parquet"), recursive=True)
    if not hits:
        logger.warning("未找到闭环 aggregator parquet（%s）⇒ 跳过闭环交集断言", root)
        return set()
    df = pd.read_parquet(hits[0])
    return set(df[df["log_name"].notna()]["log_name"].unique())


def main() -> None:
    p = argparse.ArgumentParser(description="新档 fullret L1 独立 test split 构建（L-15 (A)）")
    p.add_argument("--n_logs", type=int, default=60, help="test 取多少个 log（整 log 分配）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--force", action="store_true", help="🔴 覆盖已冻结的 test root（须在工作记录说明理由）")
    a = p.parse_args()

    test_root = a.split_root / "test"
    frozen = a.split_root / ".TEST_SPLIT_FROZEN"
    if frozen.exists() and not a.force:
        raise SystemExit(f"🔴 test split 已冻结（{frozen}）—— 拒绝重建。"
                         "test 只开一次；确需重建请显式 --force 并在工作记录写明理由。")

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    tok_by_log: Dict[str, List[str]] = defaultdict(list)
    for e in manifest:
        tok_by_log[e["log_name"]].append(e["token"])

    y = yaml.safe_load(SPLITTER_YAML.read_text(encoding="utf-8"))
    train_logs = set(y["log_splits"]["train"])
    val_logs = set(y["log_splits"]["val"])
    # 🔴 yaml 的 test 键是 val 的副本，故意不读它
    unused = sorted(set(tok_by_log) - train_logs - val_logs)
    logger.info("池 %d log / 已用 %d（train %d + val %d）/ 未用 %d",
                len(tok_by_log), len(train_logs | val_logs), len(train_logs), len(val_logs), len(unused))
    if len(unused) < a.n_logs:
        raise SystemExit(f"未用 log 只有 {len(unused)} 个，不足 {a.n_logs}")

    rng = random.Random(a.seed)
    test_logs = sorted(rng.sample(unused, a.n_logs))

    assert not (set(test_logs) & train_logs), "test ∩ train ≠ ∅"
    assert not (set(test_logs) & val_logs), "test ∩ val ≠ ∅"
    cl = closed_loop_logs(CLOSED_LOOP_REF)
    overlap_cl = set(test_logs) & cl
    assert not overlap_cl, f"test ∩ 闭环 log ≠ ∅：{sorted(overlap_cl)[:5]}"

    tokens = [t for lg in test_logs for t in sorted(tok_by_log[lg])]
    missing = [t for t in tokens if not (SOURCE_POOL / f"features_{t}.pkl").exists()]
    if missing:
        raise SystemExit(f"🔴 {len(missing)} 个 token 在 {SOURCE_POOL} 缺 pkl，前三：{missing[:3]}")

    test_root.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for t in tokens:
        link = test_root / f"features_{t}.pkl"
        if link.is_symlink() or link.exists():
            continue
        os.symlink(SOURCE_POOL / f"features_{t}.pkl", link)
        n_new += 1

    logs_sha = hashlib.sha256("\n".join(test_logs).encode()).hexdigest()[:16]
    meta = {
        "purpose": "独立 test split（headline 报数用，只开一次）",
        "ruling": "L-15 = (A) 新建 fullret test，只评 MAGNET（用户 2026-08-18 裁定）",
        "why_not_splitter_yaml": "gnn_scaling_L{1,2,3}.yaml 与 L0 的 test 键均为 val 的逐字副本 ⇒ 用它等于拿选模型的 val 报 headline（R4-7 再犯）",
        "source_pool": str(SOURCE_POOL),
        "manifest": str(MANIFEST),
        "splitter_yaml": str(SPLITTER_YAML),
        "selection_rule": f"sorted(pool logs − train − val) 后 random.Random({a.seed}).sample(·, {a.n_logs})，结果再排序",
        "seed": a.seed,
        "n_logs_pool_total": len(tok_by_log),
        "n_logs_train": len(train_logs),
        "n_logs_val": len(val_logs),
        "n_logs_unused_pool": len(unused),
        "n_logs_test": len(test_logs),
        "n_tokens_test": len(tokens),
        "test_logs_sha256_16": logs_sha,
        "test_logs": test_logs,
        "disjointness_asserted": ["test∩train=∅", "test∩val=∅",
                                  f"test∩closed_loop_logs=∅（n_cl={len(cl)}，源 {CLOSED_LOOP_REF.name}）"],
        "test_root": str(test_root),
        "note": "test 仅用于最终 headline 报数，不参与任何超参 / 拓扑 / checkpoint 选择（R4-7）；"
                "报数一律用 saved-checkpoint 口径，不得用训练曲线 minima。",
    }
    (a.split_root / "test_split_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen.write_text(json.dumps(
        {"frozen_by": str(Path(__file__).resolve()), "logs_sha256_16": logs_sha,
         "n_logs": len(test_logs), "n_tokens": len(tokens), "seed": a.seed},
        ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("test split 落盘：%d log / %d token（新建 symlink %d）；sha16=%s",
                len(test_logs), len(tokens), n_new, logs_sha)
    logger.info("manifest: %s", a.split_root / "test_split_manifest.json")


if __name__ == "__main__":
    main()
