"""外部基线对齐版：cache 落盘特征与进程内现算逐位一致性核对（CPU，几个 token，不训练）。

为什么需要它
------------
cache 冒烟（scratchpad/s6_cache_smoke_guard_20260913.sh）只证明了条数够、devkit 自报 0 失败；训练读的是 cache。
三份单测（test_plantf_aligned_builder / test_urban_driver_aligned / test_lanegcn_aligned）证明的是
「进程内现算的 builder 输出 == MAGNET」，而且只在冻结 test 的 4 个 token 上跑过。本脚本补上中间一环：

  C 段：cache 里的 .gz == 用 cache 当时 hydra 拼好的同一份 model 配置、在进程内现算的结果（逐位）
  单测段：在同一批 token（cache 冒烟的 token 属 train split）上重跑单测的 check_token（现算 == MAGNET）
  ⇒ cache == MAGNET

检查项
------
C1 每个 token 的 cache 目录里文件集合 == 模型全部 feature / target builder 的 unique name（.gz）
C2 逐个 builder：cache 反序列化后 serialize() 与现算 serialize() 相同 —— 结构、dtype、形状、整数与布尔逐位相同；
   浮点逐位相同，或差异 ≤ FLOAT_NOISE_TOL（逐条记入 float_noise，不静默吞掉）。
   容差的来由（09-13 实测）：Urban Driver 周车第 5 列 yaw rate 由 devkit ``compute_yaw_rate_from_state_tensors``
   → ``approximate_derivatives_tensor`` → ``_torch_savgol_filter``（卷积）算出，同一进程里对同一场景连续现算 8 次，
   静止车的 yaw rate 在 0 与 5.3e-15 之间跳（OMP 1 线程与默认线程都复现）⇒ 这是原模型 builder 自身的数值非确定性，
   不是 cache 与现算不一致
C2b（仅 PlanTF）单测里用默认构造的 NuplanFeatureBuilderAligned()，与 hydra 配置实例化的 builder 输出逐位相同
    ⇒ 单测段验证的 builder 就是 cache 用的 builder
U  单测 check_token 在同一 token 上通过；PlanTF 的 T1 需要 E8 在冻结 test 上的参照特征，train token 上不适用，记 N/A
另报：单线程现算每样本耗时（首个 token 含地图加载，不计入中位数），供定全量 cache 的 worker 后端

用法
----
  plantf 环境：/home/team/anaconda3/envs/plantf/bin/python tests/check_external_baselines_cache_parity.py \\
      --model plantf_aligned --cache_dir <abs> --hydra_config <abs>/code/hydra/config.yaml --out <abs.json>
  nuplan 环境：/home/team/anaconda3/envs/nuplan/bin/python tests/check_external_baselines_cache_parity.py \\
      --model urban_driver|lanegcn --cache_dir <abs> --hydra_config <abs> --out <abs.json>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for p in (str(GNN.parent), str(GNN / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_plantf_aligned_builder as base  # noqa: E402  设好 PlanTF / GNN_Trans 的 sys.path，复用读我方特征

OUR_ROOT = GNN / "data/processed_features_fullret_L1_logsplit"
T1_NA_MARK = " T1 缺 E8 参照特征"
#: 浮点舍入噪声容差（见 docstring C2）：实测 builder 自身非确定性 5.3e-15，远小于 float32 在 1 m 处的间隔 6e-8
FLOAT_NOISE_TOL = 1e-12
#: diff() 记下的浮点噪声 (路径, 元素数, 最大差)
FLOAT_NOISE: List[Tuple[str, int, float]] = []


# --------------------------------------------------------------------------- 工具


def pick_tokens(cache_dir: Path, n: int) -> List[Tuple[str, str, str]]:
    """按场景类型轮流取 token（每类内按 token 排序），保证覆盖尽量多的类型。返回 (log, 场景类型, token)。"""
    by_type: Dict[str, List[Tuple[str, str, str]]] = {}
    for d in sorted(cache_dir.glob("*/*/*")):
        if d.is_dir() and d.parts[-3] != "metadata":
            by_type.setdefault(d.parts[-2], []).append((d.parts[-3], d.parts[-2], d.parts[-1]))
    out: List[Tuple[str, str, str]] = []
    queues = [by_type[t] for t in sorted(by_type)]
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def our_split_root(token: str) -> Path:
    hits = [OUR_ROOT / s for s in ("train", "val", "test") if (OUR_ROOT / s / f"features_{token}.pkl").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{token} 在我方 train/val/test 中命中 {len(hits)} 处")
    return hits[0]


def build_scenario(e8, log_name: str, token: str):
    """与 MAGNET / E8 相同的取场景配方：不传 scenario_mapping、scenario_types=None。"""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    db = e8.db_path(log_name)
    builder = NuPlanScenarioBuilder(data_root=os.path.dirname(db), map_root=e8.MAP_ROOT, sensor_root=None,
                                    db_files=[db], map_version=e8.MAP_VERSION)
    sf = ScenarioFilter(scenario_types=None, scenario_tokens=[token], log_names=None, map_names=None,
                        num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                        ego_displacement_minimum_m=None, expand_scenarios=False,
                        remove_invalid_goals=False, shuffle=False)
    scenarios = builder.get_scenarios(sf, Sequential())
    if len(scenarios) != 1:
        raise RuntimeError(f"{token} 取到 {len(scenarios)} 个场景")
    return scenarios[0]


def diff(a: Any, b: Any, path: str = "") -> List[str]:
    """逐位比较两个 serialize() 结果，返回不同之处（空表示逐位相同）。"""
    import torch

    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{path} 类型 {type(a).__name__} vs {type(b).__name__}"]
        if set(a) != set(b):
            return [f"{path} 键不同：{sorted(map(str, set(a) ^ set(b)))}"]
        return [m for k in a for m in diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return [f"{path} 序列长度或类型不同"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path} dtype/形状 {a.dtype}{a.shape} vs {b.dtype}{b.shape}"]
        is_float = a.dtype.kind == "f"
        if np.array_equal(a, b, equal_nan=is_float):
            return []
        if is_float and np.array_equal(np.isnan(a), np.isnan(b)):
            gap = np.nan_to_num(np.abs(a.astype(np.float64) - b.astype(np.float64)), nan=0.0)
            if gap.max() <= FLOAT_NOISE_TOL:
                FLOAT_NOISE.append((path, int((gap > 0).sum()), float(gap.max())))
                return []
            return [f"{path} 值不同（最大差 {float(gap.max())}，超过浮点噪声容差 {FLOAT_NOISE_TOL}）"]
        gap = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.dtype.kind in "fiub" else -1
        return [f"{path} 值不同（最大差 {gap}）"]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if (type(a) is type(b) and a == b) else [f"{path} 标量 {a!r} vs {b!r}"]


# --------------------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("plantf_aligned", "urban_driver", "lanegcn"), required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--hydra_config", required=True, help="cache 运行时保存的 code/hydra/config.yaml")
    ap.add_argument("--n_tokens", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

    cfg = OmegaConf.load(args.hydra_config)
    if Path(cfg.cache.cache_path) != Path(args.cache_dir):
        raise SystemExit(f"hydra 配置的 cache_path {cfg.cache.cache_path} 与 --cache_dir 不一致")
    model = instantiate(cfg.model)
    builders = list(model.get_list_of_required_feature()) + list(model.get_list_of_computed_target())
    names = [b.get_feature_unique_name() for b in builders]
    store = FeatureCachePickle()

    e8 = base._load_e8()
    base._load_e8 = lambda: e8          # 单测里 UD / LaneGCN 每次重新加载 E8，钉成同一个已改根目录的实例
    if args.model == "urban_driver":
        import test_urban_driver_aligned as unit
    elif args.model == "lanegcn":
        import test_lanegcn_aligned as unit
    else:
        unit = base
        from src.feature_builders.nuplan_feature_builder_aligned import NuplanFeatureBuilderAligned

    picks = pick_tokens(Path(args.cache_dir), args.n_tokens)
    fails: List[str] = []
    rows: List[Dict[str, Any]] = []
    build_sec: List[float] = []
    for log_name, stype, tok in picks:
        root = our_split_root(tok)
        base.TEST_FEATURES = root       # 单测读我方特征的目录（按 token 所在 split）
        e8.TEST_ROOT = root             # E8 load_our_gt 读的目录
        sc = build_scenario(e8, log_name, tok)
        row: Dict[str, Any] = {"token": tok, "scenario_type": stype, "our_split": root.name}

        # C1 文件集合
        tok_dir = Path(args.cache_dir) / log_name / stype / tok
        files = sorted(p.name for p in tok_dir.iterdir())
        if files != sorted(f"{n}.gz" for n in names):
            fails.append(f"{tok} C1 cache 文件 {files} ≠ builder {names}")

        # C2 逐 builder 逐位比较
        t0 = time.time()
        fresh = {}
        for b, n in zip(builders, names):
            fresh[n] = b.get_features_from_scenario(sc) if hasattr(b, "get_features_from_scenario") else b.get_targets(sc)
        build_sec.append(time.time() - t0)
        n_cmp, noise_from = 0, len(FLOAT_NOISE)
        for b, n in zip(builders, names):
            cached = store.load_computed_feature_from_folder(tok_dir / n, b.get_feature_type())
            d = diff(cached.serialize(), fresh[n].serialize(), n)
            n_cmp += 1
            fails += [f"{tok} C2 {m}" for m in d[:5]]
        row["c2_builders_compared"] = n_cmp
        row["c2_float_noise"] = FLOAT_NOISE[noise_from:]

        # C2b PlanTF：单测用的默认构造 builder == hydra 配置 builder
        if args.model == "plantf_aligned":
            d = diff(NuplanFeatureBuilderAligned().get_features_from_scenario(sc).serialize(),
                     fresh["feature"].serialize(), "default_ctor")
            fails += [f"{tok} C2b {m}" for m in d[:5]]

        # U 单测恒等检查（现算 == MAGNET）
        unit_fails: List[str] = []
        stat = unit.check_token(e8, sc, unit_fails) if args.model == "plantf_aligned" else unit.check_token(sc, model, unit_fails)
        row["unit_T1_na"] = any(T1_NA_MARK in f for f in unit_fails)
        fails += [f"{tok} U {f}" for f in unit_fails if T1_NA_MARK not in f]
        row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in stat.items()
                    if not k.startswith("_") and k != "token"})
        rows.append(row)
        print(row, flush=True)

    steady = build_sec[1:] if len(build_sec) > 1 else build_sec
    summary = {
        "model": args.model, "cache_dir": args.cache_dir, "hydra_config": args.hydra_config,
        "n_tokens": len(rows), "builders": names,
        "fresh_build_sec_first": round(build_sec[0], 3) if build_sec else None,
        "fresh_build_sec_median_after_first": round(statistics.median(steady), 3) if steady else None,
        "rows": rows, "fails": fails, "pass": not fails and len(rows) == args.n_tokens,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"\n单线程现算：首个 {summary['fresh_build_sec_first']} s，其后中位 {summary['fresh_build_sec_median_after_first']} s / 样本")
    if not summary["pass"]:
        print(f"❌ {len(fails)} 项失败：")
        for f in fails:
            print("  -", f)
        return 1
    print(f"✅ C1 / C2{' / C2b' if args.model == 'plantf_aligned' else ''} / 单测恒等检查全部通过（{len(rows)} 个 token）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
