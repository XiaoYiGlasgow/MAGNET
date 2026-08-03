#!/usr/bin/env python3
"""S4 —— MAGNET 侧按**共享清单**做 12D 全量预处理（第二轮 scaling ladder）。

## 与冒烟的关系

`s1_smoke_magnet_on_manifest.py` 已用 20 个 token 验过 [S1]–[S5] 全通过
（数量对得上 / schema 逐键相等 / 形状一致 / token 可回溯 / `remove_invalid_goals=False` 不崩）。
本脚本是**同一条代码路径**的全量分片版，不引入任何新的处理逻辑：

    build_scenarios(按 token)  →  batch_process_nuplan_data(scenarios, create_default_config(), out)

## 为什么逐 log 处理而不是一次性建 2455 个 db 的 builder

`NuPlanScenarioBuilder` 的 `db_files` 一次给 2455 个会把整张 log 表读进来，
内存与首次延迟都不可控；而每个 log 只有 ≤100 个 token（Q61 已裁定「**上限** 100 帧」），
逐 log 建 builder 的重复开销远小于一次性建。**逐 log 也让断点续跑成为可能。**

## 分片

按 log 取模（`i % n_shards == shard`），与 `s2_diagnose_rig_false_failures.py` 同一形状。
`features_<token>.pkl` 文件名全局唯一 ⇒ 多个分片写同一个输出目录是安全的。

## 断点续跑

默认跳过输出目录里已存在的 `features_<token>.pkl`。
⚠️ 这意味着**中途改了预处理配置就必须换输出目录**，否则新旧产物会混在一起。

## 终局判据（不看 rc）

落盘的 `features_*.pkl` 数 == 清单该级样本数（各分片各自写 `s4_shard_<i>.json` 供合并核对）。

用法::

    # 单分片试跑（前 2 个 log）
    python s4_preprocess_magnet_on_manifest.py --tier L3 --limit_logs 2

    # 全量第 k/14 片
    python s4_preprocess_magnet_on_manifest.py --tier L3 --shard 0 --n_shards 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
DEFAULT_MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42_filtered.json"
DEFAULT_OUT = GNN / "data/processed_features_scaling_L3"
SHARD_REPORT_DIR = GNN / "experiments/scaling_ladder/s4_preprocess"

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"


def build_scenarios_for_log(log_name: str, tokens: List[str]) -> list:
    """为单个 log 按 token 建场景。

    与 `s1_smoke_magnet_on_manifest.py::build_scenarios` 逐字段一致，
    唯一差别是 db 只取这一个 log 的。

    Args:
        log_name: log 名（不含 `.db`）。
        tokens: 该 log 上要处理的 scenario token。

    Returns:
        场景对象列表。

    Raises:
        FileNotFoundError: 该 log 在两城 split 下都找不到 db。
    """
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    dbs = sorted(str(p) for p in Path(DATA_ROOT).glob(f"*/{log_name}.db"))
    if not dbs:
        raise FileNotFoundError(f"{log_name}.db 在 {DATA_ROOT}/*/ 下找不到")

    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=dbs, map_version=MAP_VERSION,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None, scenario_tokens=list(tokens),
        log_names=None, map_names=None, num_scenarios_per_type=None,
        limit_total_scenarios=None, timestamp_threshold_s=None,
        ego_displacement_minimum_m=None, expand_scenarios=False,
        remove_invalid_goals=False,   # Q59 裁定，与共享清单口径一致
        shuffle=False,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="L3", choices=["L1", "L2", "L3"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="共享清单（默认用**剔除后**的那份）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="产物目录")
    parser.add_argument("--shard", type=int, default=0, help="分片序号（按 log 取模）")
    parser.add_argument("--n_shards", type=int, default=1, help="总分片数")
    parser.add_argument("--limit_logs", type=int, default=None, help="只处理前 N 个 log（试跑用）")
    parser.add_argument("--report_every", type=int, default=10, help="每处理几个 log 打一次进度")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(GNN))
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("清单不存在：%s", manifest_path)
        return 2

    manifest: Dict[str, Any] = json.loads(manifest_path.read_text())
    all_logs: List[str] = manifest["tiers"][args.tier]["logs"]
    tokens_by_log: Dict[str, List[str]] = manifest["tokens"]

    my_logs = [log for i, log in enumerate(all_logs) if i % args.n_shards == args.shard]
    if args.limit_logs is not None:
        my_logs = my_logs[: args.limit_logs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    SHARD_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_requested = sum(len(tokens_by_log.get(log, [])) for log in my_logs)
    logger.info("分片 %d/%d｜log %d 个｜token %d 个｜产物 %s",
                args.shard, args.n_shards, len(my_logs), n_requested, out_dir)

    from scripts.preprocessing.process_nuplan_to_12d import (  # noqa: E402
        batch_process_nuplan_data, create_default_config)

    config = create_default_config()
    t0 = time.time()
    n_done = 0          # 本次真正处理的 token 数
    n_skipped = 0       # 断点续跑跳过的
    failed_logs: List[Dict[str, str]] = []

    for idx, log_name in enumerate(my_logs, start=1):
        tokens = tokens_by_log.get(log_name, [])
        todo = [t for t in tokens if not (out_dir / f"features_{t}.pkl").exists()]
        n_skipped += len(tokens) - len(todo)
        if not todo:
            continue

        try:
            scenarios = build_scenarios_for_log(log_name, todo)
            if not scenarios:
                failed_logs.append({"log": log_name, "error": "建场景返回空列表"})
                continue
            batch_process_nuplan_data(scenarios, config, str(out_dir))
            n_done += len(scenarios)
        except Exception as exc:   # 一个 log 崩不许毁掉整个分片
            failed_logs.append({"log": log_name, "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("log %s 处理失败：%s: %s", log_name, type(exc).__name__, exc)

        if idx % args.report_every == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed * 3600 if elapsed > 0 else 0.0
            logger.info("  %d/%d log 已处理｜落盘 %d｜跳过 %d｜失败 log %d｜%.0f 样本/h",
                        idx, len(my_logs), n_done, n_skipped, len(failed_logs), rate)

    elapsed = time.time() - t0
    n_on_disk = sum(1 for log in my_logs for t in tokens_by_log.get(log, [])
                    if (out_dir / f"features_{t}.pkl").exists())
    report = {
        "tier": args.tier,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "n_logs": len(my_logs),
        "n_requested": n_requested,
        "n_processed_this_run": n_done,
        "n_skipped_existing": n_skipped,
        "n_on_disk": n_on_disk,
        "failed_logs": failed_logs,
        "elapsed_s": round(elapsed, 1),
        "finished_at": time.strftime("%F %T"),
    }
    report_path = SHARD_REPORT_DIR / f"s4_shard_{args.shard}.json"
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    logger.info("分片 %d 完成：落盘 %d / 请求 %d（跳过 %d，失败 log %d），用时 %.0f s ⇒ %s",
                args.shard, n_on_disk, n_requested, n_skipped, len(failed_logs), elapsed, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
