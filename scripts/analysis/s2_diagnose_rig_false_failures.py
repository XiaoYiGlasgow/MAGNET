"""S2 诊断 —— 量化 `remove_invalid_goals=False` 在 PlanTF 侧的坏场景比例。

背景（08-02 冒烟实测）：
    Q59 为了把 L3 上限从 ~72 k 抬到 245.5 k 裁定了 `remove_invalid_goals=False`。
    MAGNET 侧冒烟 [S5] 通过（它不做 route 修正）。**PlanTF 侧崩了**：
        route_utils.py:153 `BreadthFirstSearchRoadBlock.search(...)`
        → AttributeError: 'NoneType' object has no attribute 'id'
    根因：`route_roadblock_correction` 把 route 里每个 id 查地图，查不到就存 None
    （route_utils.py:157-163 的 `block or ...` 两次都失败），随后 BFS 起点为 None。
    而 `feature_preprocessor.py:105` 把任何异常 **re-raise 成 RuntimeError**
    ⇒ 一个坏场景就让整批 ray task 挂掉，不是跳过。

本脚本单进程逐个 token 试算 feature，catch 异常，回答三个问题：
    1. 坏场景占比多少？
    2. 失败是否集中在少数 log / 少数异常类型？
    3. 坏 token 能否事先枚举 —— 能的话，从共享清单里剔除即可，两条管线仍读同一份清单。

用法：
    python s2_diagnose_rig_false_failures.py --tokens_json <路径> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PLANTF = Path("/mnt/local/Xiao_Yi/external/planTF")
sys.path.insert(0, str(PLANTF))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (  # noqa: E402
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter  # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential  # noqa: E402

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SPLITS = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits")
MAPS = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps")


def build_scenarios(logs: List[str], tokens: List[str]) -> List[Any]:
    """按共享清单的 flags 建 scenario 列表（flags 与 S1 清单逐字段一致）。"""
    builder = NuPlanScenarioBuilder(
        data_root=[str(SPLITS / "train_boston"), str(SPLITS / "train_pittsburgh")],
        map_root=str(MAPS),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=tokens,
        log_names=logs,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def main() -> int:
    """跑诊断，把坏 token 清单写盘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens_json", type=Path,
        default=GNN / "experiments/scaling_ladder/s2_smoke_tokens.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="只试前 N 个场景")
    parser.add_argument(
        "--out", type=Path,
        default=GNN / "experiments/scaling_ladder/s2_rig_false_failures.json",
    )
    parser.add_argument("--tier", default=None, choices=["L1", "L2", "L3"],
                        help="直接从共享清单取某一级，与 --tokens_json 二选一")
    parser.add_argument("--shard", type=int, default=0, help="分片序号（按 log 取模）")
    parser.add_argument("--n_shards", type=int, default=1, help="总分片数")
    args = parser.parse_args()

    if args.tier is not None:
        manifest = json.loads(
            (GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42.json").read_text()
        )
        all_logs = manifest["tiers"][args.tier]["logs"]
        # 按 log 取模分片：同一个 log 的 token 不会被拆到两个分片，逐 log 失败率才算得准
        my_logs = [log for i, log in enumerate(all_logs) if i % args.n_shards == args.shard]
        requested = {
            "logs": my_logs,
            "tokens": [t for log in my_logs for t in manifest["tokens"][log]],
        }
        logger.info("分片 %d/%d：%d 个 log / %d 个 token",
                    args.shard, args.n_shards, len(my_logs), len(requested["tokens"]))
    else:
        requested = json.loads(args.tokens_json.read_text())
    scenarios = build_scenarios(requested["logs"], requested["tokens"])
    logger.info("请求 %d token → 建出 %d 个 scenario", len(requested["tokens"]), len(scenarios))
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    feature_builder = NuplanFeatureBuilder(radius=100, history_horizon=2)

    failures: List[Dict[str, str]] = []
    kinds: Counter = Counter()
    per_log_fail: Counter = Counter()
    per_log_total: Counter = Counter()

    for idx, scenario in enumerate(scenarios):
        per_log_total[scenario.log_name] += 1
        try:
            feature_builder.get_features_from_scenario(scenario)
        except Exception as error:  # noqa: BLE001 — 诊断脚本要的就是「任何异常」
            kind = f"{type(error).__name__}: {error}"
            kinds[kind] += 1
            per_log_fail[scenario.log_name] += 1
            failures.append(
                {
                    "token": scenario.token,
                    "log_name": scenario.log_name,
                    "scenario_type": scenario.scenario_type,
                    "error": kind,
                    "where": traceback.format_exc().strip().splitlines()[-3],
                }
            )
        if (idx + 1) % 25 == 0:
            logger.info("  %d/%d 已试，失败 %d", idx + 1, len(scenarios), len(failures))

    n = len(scenarios)
    logger.info("\n===== rig=False 坏场景诊断 =====")
    logger.info("  试算 %d 个场景，失败 %d 个（%.1f%%）", n, len(failures), 100.0 * len(failures) / max(n, 1))
    logger.info("  异常类型分布：")
    for kind, count in kinds.most_common(10):
        logger.info("    %4d× %s", count, kind[:140])
    logger.info("  逐 log 失败率：")
    for log in sorted(per_log_total):
        logger.info(
            "    %s  %d/%d = %.1f%%",
            log, per_log_fail[log], per_log_total[log],
            100.0 * per_log_fail[log] / per_log_total[log],
        )

    args.out.write_text(
        json.dumps(
            {
                "n_scenarios": n,
                "n_failures": len(failures),
                "failure_rate": len(failures) / max(n, 1),
                "kinds": dict(kinds),
                "per_log_total": dict(per_log_total),
                "per_log_fail": dict(per_log_fail),
                "failures": failures,
            },
            indent=1,
        )
    )
    logger.info("  已写 %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
