"""VT-2026-04191 D2 前置 · route 信号覆盖率统计。

背景：阶段⑫ E0 十样本验收中发现 1/10 样本 route 信号为空（3 个 roadblock_ids 经
`prune_route_by_connectivity` 在 50 m 半径内全被裁掉，devkit 原生行为，非 bug）。
十样本不足以定比例，全量重预处理前须在 E1 真实训练集口径上把这个数测准 —— 它直接决定
route 提分预期，进而影响 D8 赌注判断。

口径等价性：预处理写入 lane 节点 col8 的值就是
    build_lane_on_route_flags(lane_ids, route_lane_ids, config.LANE_NUM)
本脚本调用同一对函数、用同一 ego 参考点（`get_ego_past_trajectory(...)[-1]`，与
process_nuplan_to_12d.py 一致），故统计结果与真实预处理**零推断误差**，
无需跑完整特征编码（跳过 agent 提取与 map_process 插值，快一个量级）。

不使用 GPU（纯地图查询 + CPU）。

运行：
  cd /mnt/local/Xiao_Yi/nuplan-devkit
  PYTHONPATH=GNN_Trans:. /home/team/anaconda3/envs/nuplan/bin/python \
    GNN_Trans/scripts/preprocessing/stat_route_coverage.py --n_logs 600 --tokens_per_log 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_GNN_ROOT = Path(__file__).resolve().parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402  GNN_Trans/config.py
from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (  # noqa: E402
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter  # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential  # noqa: E402
from src.shared.utils.data_utils import (  # noqa: E402
    get_neighbor_vector_set_map,
    get_route_lane_ids,
    build_lane_on_route_flags,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stat_route_coverage")

_DATASET_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset"
_DATA = f"{_DATASET_ROOT}/nuplan-v1.1/splits/train_pittsburgh"
_MANIFEST = _GNN_ROOT / "data/processed_features_magnet_e1_full/log_manifest.json"


def sample_tokens_by_log(manifest_path: Path, n_logs: int, tokens_per_log: int,
                         seed: int) -> Dict[str, List[str]]:
    """从 E1 log_manifest 按 log 分层采样 token，保证覆盖尽可能多的 log。"""
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)

    by_log: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_log[r["log_name"]].append(r["token"])

    rng = random.Random(seed)
    logs = sorted(by_log.keys())
    rng.shuffle(logs)
    logs = logs[:n_logs]

    out: Dict[str, List[str]] = {}
    for lg in logs:
        toks = sorted(by_log[lg])
        rng.shuffle(toks)
        out[lg] = toks[:tokens_per_log]
    logger.info("采样 %d logs × %d tokens = %d 个 scenario（manifest 共 %d logs / %d tokens）",
                len(out), tokens_per_log, sum(len(v) for v in out.values()),
                len(by_log), len(rows))
    return out


def measure_scenario(scenario, cfg) -> dict:
    """复刻预处理口径，返回单个 scenario 的 route 覆盖情况。"""
    ego_states = list(scenario.get_ego_past_trajectory(
        iteration=0,
        num_samples=cfg["num_past_poses"],
        time_horizon=cfg["past_time_horizon"],
    ))
    if not ego_states:
        raise ValueError("ego 历史为空")

    current_ego_state = ego_states[-1]  # 与 process_nuplan_to_12d.py 同一参考点
    ego_position = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)

    route_roadblock_ids = scenario.get_route_roadblock_ids() or []
    _coords, _tl, lane_ids = get_neighbor_vector_set_map(
        scenario.map_api, cfg["map_features"], ego_position, cfg["map_radius"],
        route_roadblock_ids, [], return_lane_ids=True,
    )
    route_lane_ids = get_route_lane_ids(
        scenario.map_api, ego_position, cfg["map_radius"], route_roadblock_ids
    )
    flags = build_lane_on_route_flags(lane_ids, route_lane_ids, cfg["max_elements"]["LANE"])

    n_on_route = int(flags.sum())
    return {
        "n_roadblock_ids": len(route_roadblock_ids),
        "n_lanes_in_radius": len(lane_ids),
        "n_route_lane_ids": len(route_lane_ids),
        "n_on_route_lanes": n_on_route,          # 真正写进 col8 的 1 的 lane 条数
        "empty_route": n_on_route == 0,          # ★ 本脚本的核心指标
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 E1 训练集口径下的 route 信号覆盖率")
    parser.add_argument("--n_logs", type=int, default=600, help="采样多少个 log")
    parser.add_argument("--tokens_per_log", type=int, default=1, help="每个 log 采多少 scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_logs", type=int, default=50,
                        help="每批加载多少个 db（控制 builder 内存/耗时）")
    parser.add_argument("--out", type=str,
                        default=str(_GNN_ROOT / "docs/work_records/paper/MAGNET/"
                                    "VT-2026-04191_修改工作记录/route_coverage_stats.json"))
    args = parser.parse_args()

    cfg = {
        "num_past_poses": config.NUM_PAST_POSES,
        "past_time_horizon": config.PAST_TIME_HORIZON,
        "map_features": config.MAP_FEATURES,
        "map_radius": config.MAP_RADIUS,
        "max_elements": config.MAX_ELEMENTS,
    }
    logger.info("契约: %s", cfg)

    sampled = sample_tokens_by_log(_MANIFEST, args.n_logs, args.tokens_per_log, args.seed)
    log_names = sorted(sampled.keys())

    records: List[dict] = []
    n_fail = 0
    for bi in range(0, len(log_names), args.batch_logs):
        batch_logs = log_names[bi: bi + args.batch_logs]
        db_files = [f"{_DATA}/{lg}.db" for lg in batch_logs]
        db_files = [p for p in db_files if os.path.exists(p)]
        if not db_files:
            logger.warning("批 %d 无可用 db，跳过", bi // args.batch_logs)
            continue

        want_tokens = {t for lg in batch_logs for t in sampled[lg]}
        builder = NuPlanScenarioBuilder(
            data_root=_DATA, map_root=f"{_DATASET_ROOT}/maps",
            sensor_root=f"{_DATASET_ROOT}/nuplan-v1.1/sensor_blobs",
            db_files=db_files, map_version="nuplan-maps-v1.0",
        )
        sf = ScenarioFilter(
            scenario_types=None, scenario_tokens=sorted(want_tokens), log_names=None,
            map_names=None, num_scenarios_per_type=None, limit_total_scenarios=None,
            timestamp_threshold_s=None, ego_displacement_minimum_m=None,
            expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
        )
        scenarios = builder.get_scenarios(sf, Sequential())

        for sc in scenarios:
            try:
                rec = measure_scenario(sc, cfg)
                rec["token"] = sc.token
                rec["log_name"] = sc.log_name
                records.append(rec)
            except Exception as e:  # noqa: BLE001  统计脚本：单场景失败不应中断全批
                n_fail += 1
                logger.debug("token=%s 失败: %s: %s", sc.token, type(e).__name__, e)

        logger.info("批 %d/%d 完成：累计 %d 个样本（失败 %d）",
                    bi // args.batch_logs + 1,
                    (len(log_names) + args.batch_logs - 1) // args.batch_logs,
                    len(records), n_fail)

    if not records:
        logger.error("零样本，统计失败")
        sys.exit(1)

    n = len(records)
    empty = [r for r in records if r["empty_route"]]
    on_route = np.array([r["n_on_route_lanes"] for r in records], dtype=np.float64)
    n_rb = np.array([r["n_roadblock_ids"] for r in records], dtype=np.float64)
    p_empty = len(empty) / n
    se = float(np.sqrt(p_empty * (1 - p_empty) / n))  # 二项比例标准误

    summary = {
        "n_samples": n,
        "n_logs_sampled": len({r["log_name"] for r in records}),
        "n_failed": n_fail,
        "empty_route_count": len(empty),
        "empty_route_ratio": p_empty,
        "empty_route_ratio_95ci": [max(0.0, p_empty - 1.96 * se), min(1.0, p_empty + 1.96 * se)],
        "on_route_lanes_mean": float(on_route.mean()),
        "on_route_lanes_median": float(np.median(on_route)),
        "on_route_lanes_p10": float(np.percentile(on_route, 10)),
        "on_route_lanes_p90": float(np.percentile(on_route, 90)),
        "roadblock_ids_mean": float(n_rb.mean()),
        "roadblock_ids_zero_count": int((n_rb == 0).sum()),
        "contract": {k: str(v) for k, v in cfg.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("route 信号覆盖率统计（E1 训练集口径，与预处理 col8 零推断误差）")
    print("=" * 64)
    print(f"样本数            : {n}（覆盖 {summary['n_logs_sampled']} logs，失败 {n_fail}）")
    print(f"★ route 空信号比例: {p_empty:.1%}  95%CI [{summary['empty_route_ratio_95ci'][0]:.1%}, "
          f"{summary['empty_route_ratio_95ci'][1]:.1%}]  （{len(empty)}/{n}）")
    print(f"  其中 roadblock_ids 本身为 0 的: {summary['roadblock_ids_zero_count']}")
    print(f"on-route lane 条数 : mean={summary['on_route_lanes_mean']:.2f} "
          f"median={summary['on_route_lanes_median']:.0f} "
          f"p10={summary['on_route_lanes_p10']:.0f} p90={summary['on_route_lanes_p90']:.0f} "
          f"(共 {config.LANE_NUM} 条 lane)")
    print(f"roadblock_ids 均值 : {summary['roadblock_ids_mean']:.1f}")
    print(f"\n明细已写入: {out_path}")


if __name__ == "__main__":
    main()
