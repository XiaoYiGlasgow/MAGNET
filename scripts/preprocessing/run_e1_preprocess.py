"""E1 · 干净重预处理 driver（不改 E0 冻结的 process_nuplan_to_12d.py）。

现脚本 `__main__` 把 `limit_total_scenarios=10000` / `db_files=None` 写死，且 `db_files=None` 扫全部
1560 DB 会撞上个别坏 DB 在 builder 层报错炸整轮。本 driver：
  1) sqlite 轻校验剔除明显坏/空 DB（有 lidar_pc 表且非空）；
  2) **逐 DB 迭代**建 per-DB builder + get_scenarios（try/except 隔离——深层坏 DB 只跳它不炸整轮），
     每 DB 限量 `per_db_cap` 采样以跨更多 log，累积到 `limit` 停；
  3) 调 E0 生效的 `batch_process_nuplan_data`（内部写 features_<token>.pkl + log_manifest.json）。

log-disjoint 依赖 log 多样性：逐 DB 小额采样 → 30k 跨约 600-1000 log（1 DB≈1 log）。
注：sqlite 轻校验全过≠get_scenarios 一定成功（E0 撞的坏 DB 属深层损坏），故 try/except 隔离是关键。

运行（conda nuplan env）：
  # 先干跑校验，看好/坏 DB 数（不预处理）
  PYTHONPATH=GNN_Trans:nuplan-devkit /home/team/anaconda3/envs/nuplan/bin/python \
      GNN_Trans/scripts/preprocessing/run_e1_preprocess.py --validate_only
  # 正式跑（setsid 防杀）
  PYTHONPATH=GNN_Trans:nuplan-devkit /home/team/anaconda3/envs/nuplan/bin/python \
      GNN_Trans/scripts/preprocessing/run_e1_preprocess.py \
      --out_dir /abs/.../processed_features_magnet_e1_full --limit 30000
"""
from __future__ import annotations

import argparse
import glob
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import List

_THIS = Path(__file__).resolve()
GNN_TRANS_ROOT = _THIS.parents[2]          # scripts/preprocessing/x.py -> GNN_Trans
DEVKIT_ROOT = GNN_TRANS_ROOT.parent        # nuplan-devkit
for _p in (str(GNN_TRANS_ROOT), str(DEVKIT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_e1_preprocess")

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"


def validate_dbs(db_paths: List[str]) -> List[str]:
    """只读打开每个 DB，校验有 lidar_pc 表且非空；坏/空 DB 剔除。"""
    good, bad = [], []
    for db in db_paths:
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lidar_pc'")
            has_tbl = cur.fetchone() is not None
            has_row = False
            if has_tbl:
                cur.execute("SELECT 1 FROM lidar_pc LIMIT 1")
                has_row = cur.fetchone() is not None
            con.close()
            (good if has_row else bad).append(db)
        except Exception as e:  # 损坏/无法打开
            bad.append(db)
            logger.warning("坏 DB 剔除: %s (%s)", Path(db).name, e)
    logger.info("DB 校验: 好=%d, 坏/空=%d, 合计=%d", len(good), len(bad), len(db_paths))
    return good


def collect_scenarios_per_db(good_dbs: List[str], limit: int, per_db_cap: int, seed: int) -> list:
    """逐 DB 采样，try/except 隔离深层坏 DB；累积到 limit 停。返回 scenario 列表。"""
    import random
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    dbs = list(good_dbs)
    random.Random(seed).shuffle(dbs)
    worker = Sequential()
    collected: list = []
    skipped = 0
    t0 = time.time()
    for i, db in enumerate(dbs):
        if len(collected) >= limit:
            break
        try:
            builder = NuPlanScenarioBuilder(
                data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
                db_files=[db], map_version=MAP_VERSION,
            )
            sf = ScenarioFilter(
                scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                num_scenarios_per_type=None, limit_total_scenarios=per_db_cap, timestamp_threshold_s=None,
                ego_displacement_minimum_m=None, expand_scenarios=False,
                remove_invalid_goals=False, shuffle=True,
            )
            collected.extend(builder.get_scenarios(sf, worker))
        except Exception as e:  # 深层坏 DB，跳过不炸整轮
            skipped += 1
            logger.warning("跳过坏 DB %s: %s", Path(db).name, e)
        if i % 100 == 0:
            logger.info("进度 DB %d/%d，已收 %d scenario，跳坏 %d（%.0fs）",
                        i, len(dbs), len(collected), skipped, time.time() - t0)
    collected = collected[:limit]
    logger.info("采样完成：%d scenario，跳坏 DB %d 个，用时 %.0fs", len(collected), skipped, time.time() - t0)
    return collected


def build_and_process(good_dbs: List[str], out_dir: str, limit: int, per_db_cap: int, seed: int) -> None:
    from scripts.preprocessing.process_nuplan_to_12d import batch_process_nuplan_data, create_default_config
    scenarios = collect_scenarios_per_db(good_dbs, limit, per_db_cap, seed)
    if not scenarios:
        raise RuntimeError("未采到任何 scenario")
    batch_process_nuplan_data(scenarios, create_default_config(), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 干净重预处理 driver（校验跳坏 DB + 限量采样）")
    p.add_argument("--out_dir", type=str, default=None, help="输出目录（绝对路径）；--validate_only 时可省")
    p.add_argument("--limit", type=int, default=30000, help="总采样 scenario 数")
    p.add_argument("--per_db_cap", type=int, default=50, help="每 DB 最多采样数（越小跨 log 越多）")
    p.add_argument("--seed", type=int, default=42, help="DB 洗牌种子（可复现）")
    p.add_argument("--validate_only", action="store_true", help="只校验 DB 报好/坏数，不预处理")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_dbs = sorted(glob.glob(f"{DATA_ROOT}/*.db"))
    logger.info("发现 %d 个 DB，开始校验...", len(all_dbs))
    good = validate_dbs(all_dbs)
    if not good:
        raise RuntimeError("没有可用 DB")
    if args.validate_only:
        logger.info("validate_only：好 DB=%d，未预处理", len(good))
        return
    if not args.out_dir:
        raise ValueError("正式预处理需 --out_dir")
    build_and_process(good, args.out_dir, args.limit, args.per_db_cap, args.seed)


if __name__ == "__main__":
    main()
