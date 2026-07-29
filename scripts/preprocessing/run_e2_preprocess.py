"""E2 · route 三通道重预处理 driver（VT-2026-04191 阶段⑫ D2）。

与 `run_e1_preprocess.py` 的关键差别：**不重新采样 scenario，而是按 E1 的 `log_manifest.json`
精确重建同一批 30000 个 token。**

理由：E1 driver 的 `ScenarioFilter(shuffle=True)` 无固定 seed，同参重跑不保证命中同一批 scenario。
若新旧数据集样本不同，D8 的闭环对比就混入了「数据采样差异」这一混淆项，破坏主线
「诊断 → 干预 → 闭环验证」的因果归因严格性（阶段11 §4.2）。按 token 精确重建后：

  - 新旧数据集**逐样本配对**（同一批 30000 token、同一批 600 log）；
  - 可直接复用 E1 的 log-disjoint 划分，或对同一批 log 做统一三分（R4-7 的 test set）；
  - D8 的差异**唯一来源是 route 三通道**，这正是环②要检验的干预。

产物与 E1 同构：`features_<token>.pkl` + `log_manifest.json`（由 `batch_process_nuplan_data` 写）。

运行（conda nuplan env；CPU 任务，不占 GPU）：
  cd /mnt/local/Xiao_Yi/nuplan-devkit
  PYTHONPATH=GNN_Trans:. /home/team/anaconda3/envs/nuplan/bin/python \
      GNN_Trans/scripts/preprocessing/run_e2_preprocess.py \
      --out_dir /abs/.../processed_features_magnet_e2_route
  # 先 smoke 验证 driver 与测速：
  ... run_e2_preprocess.py --out_dir /tmp/e2_smoke --max_logs 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_THIS = Path(__file__).resolve()
GNN_TRANS_ROOT = _THIS.parents[2]
DEVKIT_ROOT = GNN_TRANS_ROOT.parent
for _p in (str(GNN_TRANS_ROOT), str(DEVKIT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_e2_preprocess")

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"
E1_MANIFEST = GNN_TRANS_ROOT / "data/processed_features_magnet_e1_full/log_manifest.json"


def load_e1_grouping(manifest_path: Path) -> Dict[str, List[str]]:
    """读 E1 manifest，返回 {log_name: [token, ...]}（保持 manifest 内出现顺序）。"""
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)
    by_log: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_log[r["log_name"]].append(r["token"])
    logger.info("E1 manifest: %d tokens / %d logs", len(rows), len(by_log))
    return dict(by_log)


def rebuild_scenarios(by_log: Dict[str, List[str]], max_logs: int | None) -> list:
    """逐 log（=逐 DB）用精确 token 列表重建 scenario；坏 DB try/except 隔离，只跳它不炸整轮。"""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    worker = Sequential()
    log_names = sorted(by_log.keys())
    if max_logs is not None:
        log_names = log_names[:max_logs]

    collected: list = []
    n_missing_db = 0
    n_bad = 0
    t0 = time.time()

    for i, log_name in enumerate(log_names, 1):
        db_path = f"{DATA_ROOT}/{log_name}.db"
        if not os.path.exists(db_path):
            n_missing_db += 1
            logger.warning("缺 DB，跳过 log %s", log_name)
            continue
        want = by_log[log_name]
        try:
            builder = NuPlanScenarioBuilder(
                data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
                db_files=[db_path], map_version=MAP_VERSION,
            )
            sf = ScenarioFilter(
                scenario_types=None, scenario_tokens=sorted(want), log_names=None, map_names=None,
                num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                ego_displacement_minimum_m=None, expand_scenarios=False,
                remove_invalid_goals=False, shuffle=False,   # ★ 精确复现：不洗牌、不限量
            )
            got = builder.get_scenarios(sf, worker)
            collected.extend(got)
            if len(got) != len(want):
                logger.warning("log %s: 期望 %d token，实得 %d", log_name, len(want), len(got))
        except Exception as e:  # noqa: BLE001  深层坏 DB，隔离
            n_bad += 1
            logger.warning("跳过坏 DB %s: %s: %s", log_name, type(e).__name__, e)

        if i % 50 == 0:
            logger.info("进度 log %d/%d，已重建 %d scenario（缺DB %d / 坏DB %d，%.0fs）",
                        i, len(log_names), len(collected), n_missing_db, n_bad, time.time() - t0)

    logger.info("重建完成：%d scenario，缺 DB %d，坏 DB %d，用时 %.0fs",
                len(collected), n_missing_db, n_bad, time.time() - t0)
    return collected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2 route 三通道重预处理（按 E1 token 精确重建）")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录（绝对路径）")
    p.add_argument("--manifest", type=str, default=str(E1_MANIFEST), help="E1 log_manifest.json")
    p.add_argument("--max_logs", type=int, default=None, help="只处理前 N 个 log（smoke 用）")
    p.add_argument("--route_mode", type=str, default="three_channel",
                   choices=["none", "three_channel"],
                   help="route 注入模式；none=三通道全零（复现旧基线），three_channel=启用")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from scripts.preprocessing.process_nuplan_to_12d import (
        batch_process_nuplan_data, create_default_config,
    )

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.glob("features_*.pkl")):
        logger.error("输出目录已有 features_*.pkl，拒绝覆盖：%s", out_dir)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = create_default_config()
    cfg["route_mode"] = args.route_mode
    logger.info("契约: map_features=%s max_elements=%s radius=%s agents=%s route_mode=%s",
                cfg["map_features"], cfg["max_elements"], cfg["map_radius"], cfg["num_agents"],
                cfg["route_mode"])

    by_log = load_e1_grouping(Path(args.manifest))
    scenarios = rebuild_scenarios(by_log, args.max_logs)
    if not scenarios:
        logger.error("零 scenario，中止")
        sys.exit(1)

    t0 = time.time()
    batch_process_nuplan_data(scenarios, cfg, str(out_dir))
    n = len(list(out_dir.glob("features_*.pkl")))
    logger.info("预处理完成：%d 个 features pkl，用时 %.0fs", n, time.time() - t0)

    # 产物元数据：代码版本 + 配置 + 数量（供复现与事故追溯）
    try:
        import subprocess
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(GNN_TRANS_ROOT),
                                 capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        git_rev = "unknown"
    meta = {
        "route_mode": args.route_mode,
        "n_features_pkl": n,
        "n_scenarios_rebuilt": len(scenarios),
        "source_manifest": str(args.manifest),
        "contract": {k: str(v) for k, v in cfg.items()},
        "git_rev": git_rev,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "driver": "run_e2_preprocess.py",
    }
    with open(out_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("preprocess_meta.json 已写出（route_mode=%s, git=%s）", args.route_mode, git_rev[:8])


if __name__ == "__main__":
    main()
