#!/usr/bin/env python3
"""S1 冒烟 —— 验证 MAGNET 的 12D 预处理**吃得下共享清单里的 token**（2026-08-02 阶段61）。

## 这是第二轮唯一未验证的技术风险

共享清单方案要求两条管线消费同一份 `(log, token)`。PlanTF 侧本来就是按 token 建 filter，
没有风险；**MAGNET 侧从来没有按外部 token 列表跑过** —— 它现行的
`run_e1_preprocess.py:collect_scenarios_per_db()` 是自己在每个 DB 上抽 50 个。

若 `batch_process_nuplan_data` 对场景来源有任何隐含假设（顺序、类型、goal 有效性…），
方案要改。**跑之前不能假设它能通。**

## 判据（事前写死）

- **[S1] 产出数量对得上**：请求 N 个 token → 落盘 N 个 `features_<token>.pkl`
- **[S2] schema 与现行产物逐键相同**：与 `processed_features_magnet_e1_none_v3` 里
  任取一个 pkl 对比 **key 集合完全相等**
- **[S3] 每个 key 的数组形状与现行产物一致**（逐 key 比 `shape`，允许第 0 维不同）
- **[S4] token 可回溯**：产出的 pkl 文件名对得上请求的 token
- 🔴 **[S5] `remove_invalid_goals=False` 引入的无效 goal 场景不会让它崩**
  —— 这是 Q59 翻成 False 之后新增的风险点，必须单独判

任一不过 ⇒ 共享清单方案对 MAGNET 侧不成立，需要改设计（不是改判据）。

用法::

    python s1_smoke_magnet_on_manifest.py --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42.json"
REFERENCE_DIR = GNN / "data/processed_features_magnet_e1_none_v3"
OUT_DIR = GNN / "experiments/scaling_ladder/smoke_magnet_manifest"

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"
MAP_VERSION = "nuplan-maps-v1.0"


def load_reference() -> Dict[str, Any]:
    """现行产物里任取一个 pkl 作 schema 基准。"""
    pkls = sorted(REFERENCE_DIR.glob("features_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"{REFERENCE_DIR} 下没有 features_*.pkl，无法做 schema 对比")
    with pkls[0].open("rb") as fh:
        return pickle.load(fh)


def pick_tokens(n: int) -> List[Dict[str, str]]:
    """从 L1 里跨 log 均匀取 n 个 (log, token)，别集中在一个 log 上。"""
    man = json.loads(MANIFEST.read_text())
    logs = man["tiers"]["L1"]["logs"]
    tokens = man["tokens"]
    picked: List[Dict[str, str]] = []
    i = 0
    while len(picked) < n and i < 1000:
        for log in logs:
            if len(picked) >= n:
                break
            toks = tokens.get(log, [])
            if i < len(toks):
                picked.append({"log_name": log, "token": toks[i]})
        i += 1
    return picked


def build_scenarios(pairs: List[Dict[str, str]]) -> list:
    """按 token 建场景（这正是共享清单方案要求两边都走的路径）。"""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    dbs = sorted({str(p) for log in {q["log_name"] for q in pairs}
                  for p in Path(DATA_ROOT).glob(f"*/{log}.db")})
    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=dbs, map_version=MAP_VERSION,
    )
    sf = ScenarioFilter(
        scenario_types=None, scenario_tokens=[q["token"] for q in pairs],
        log_names=None, map_names=None, num_scenarios_per_type=None,
        limit_total_scenarios=None, timestamp_threshold_s=None,
        ego_displacement_minimum_m=None, expand_scenarios=False,
        remove_invalid_goals=False,   # 与共享清单口径一致（Q59 = False）
        shuffle=False,
    )
    return builder.get_scenarios(sf, Sequential())


def shape_of(v: Any) -> Optional[tuple]:
    return tuple(np.asarray(v).shape) if isinstance(v, (np.ndarray, list)) else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--keep", action="store_true", help="保留产物（默认跑完删）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(GNN))

    results: Dict[str, str] = {}
    pairs = pick_tokens(args.n)
    print(f"\n=== MAGNET 预处理 × 共享清单 冒烟（请求 {len(pairs)} 个 token，"
          f"跨 {len({p['log_name'] for p in pairs})} 个 log）===\n")

    # [S5] 先判会不会崩
    try:
        scenarios = build_scenarios(pairs)
        print(f"  建场景成功：{len(scenarios)}/{len(pairs)}")
    except Exception as exc:
        print(f"  ❌ [S5] 建场景阶段就崩了：{type(exc).__name__}: {exc}")
        return 1

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.preprocessing.process_nuplan_to_12d import batch_process_nuplan_data, create_default_config
        batch_process_nuplan_data(scenarios, create_default_config(), str(OUT_DIR))
        results["S5"] = "✅ 未崩（remove_invalid_goals=False 的场景可处理）"
    except Exception as exc:
        print(f"  ❌ [S5] batch_process_nuplan_data 抛异常：{type(exc).__name__}: {exc}")
        return 1

    produced = sorted(OUT_DIR.glob("features_*.pkl"))
    requested = {p["token"] for p in pairs}
    produced_tokens = {p.stem.replace("features_", "") for p in produced}

    results["S1"] = (f"{'✅' if len(produced) == len(pairs) else '⚠️'} "
                     f"请求 {len(pairs)} → 落盘 {len(produced)}")
    missing = requested - produced_tokens
    extra = produced_tokens - requested
    results["S4"] = ("✅ token 逐个可回溯" if not missing and not extra
                     else f"⚠️ 缺 {len(missing)}｜多 {len(extra)}")

    if not produced:
        print("  ❌ 没有任何产物，后续判据无法评估")
        for k, v in results.items():
            print(f"  [{k}] {v}")
        return 1

    ref = load_reference()
    with produced[0].open("rb") as fh:
        new = pickle.load(fh)

    ref_keys, new_keys = set(ref), set(new)
    results["S2"] = ("✅ key 集合完全相等" if ref_keys == new_keys
                     else f"❌ 缺 {sorted(ref_keys - new_keys)}｜多 {sorted(new_keys - ref_keys)}")

    bad = []
    for k in sorted(ref_keys & new_keys):
        rs, ns = shape_of(ref[k]), shape_of(new[k])
        if rs is None or ns is None:
            continue
        if rs[1:] != ns[1:]:
            bad.append(f"{k}: 现行{rs} vs 新{ns}")
    results["S3"] = "✅ 各 key 形状一致（第 0 维不比）" if not bad else "❌ " + "；".join(bad[:4])

    print()
    for k in ("S1", "S2", "S3", "S4", "S5"):
        print(f"  [{k}] {results[k]}")

    ok = all(v.startswith("✅") for v in results.values())
    print(f"\n  {'✅ 冒烟通过 —— 共享清单方案对 MAGNET 侧成立' if ok else '🔴 有判据未过，方案需改（不是改判据）'}")
    if not args.keep:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
