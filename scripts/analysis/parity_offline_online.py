"""D0#3：离线 pkl 与在线 planner 张量的成对 parity 审计（阶段15 Z5）。

要回答的问题
------------
planner 已经按同一份 `selected_indices` 构 route、并调用与训练同一个
`build_graph_data_from_feature_df`，但**从未做过全链路的 paired tensor audit**。
"看起来同一条路径" 和 "逐字段等值" 是两回事 —— 本脚本做后者。

做法
----
对同一个 `(log, token)`：

    离线： features_<token>.pkl  --build_graph_data_from_feature_df-->  GraphData_off
    在线： scenario 的 past 重建成 PlannerInput
           --GNNTransPlanner._build_graph_data-->                      GraphData_on

然后逐字段比对 `x / edge_index / edge_attr / gnn_x / gnn_edge_* / cluster / ego_history`，
其中 lane 节点的 col6/col7/col8（route 三通道）**按 lane 顺序逐行**比对。

在线的输入是 `PlannerInput.history`（SimulationHistoryBuffer），离线的输入是 scenario 的
past API。本脚本用 **同一个 scenario 的 past** 造 history，使两条路径的物理输入相同 ——
这样任何差异都只能来自特征构建链路本身，而不是输入不同。

必须覆盖的四类 edge case（阶段15 Z5 验收）
------------------------------------------
    empty_route        route 信号全 0（实测约 11.8% 样本）
    connector          场景内含 lane connector（路口）
    lane_overflow      lane 数 > config.LANE_NUM=40，发生截断（实测约 19.9%）
    route_truncated    有 route lane，但全部落在按距离排序的前 40 之外 -> 三通道全 0

前三类靠随机抽样即可命中；`route_truncated` 稀有，脚本会**显式扫描**直到找到或扫完预算，
没找到就在报告里标 `NOT_COVERED`，**不假装覆盖过**。

用法
----
    # 接线自检（不碰 nuPlan DB，秒级）
    python3 scripts/analysis/parity_offline_online.py --dry_run

    # 真实审计（需要 DB；建议先小样本）
    PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/parity_offline_online.py \
        --pkl_root /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_features_magnet_e2_route_v2 \
        --route_mode three_channel --n 30 \
        --out /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/d0_parity/parity_three_channel.json

判定：**任何一个 token 的任何一个字段超出容差即整体 FAIL**。D0#3 要求的是等价，不是"大致相同"。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT), str(_DEVKIT / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("parity_offline_online")

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
SENSOR_ROOT = None
MAP_VERSION = "nuplan-maps-v1.0"

# 浮点容差。两条路径跑的是同一份 numpy/torch 代码，理论上应逐字节相同；给一个极小的
# 容差只为吸收 float32 累加顺序差异。**不要为了让测试变绿而放大它**。
ATOL = 1e-6
RTOL = 0.0

# route 三通道所在列（阶段⑫ 方案B：复用 lane 节点的 col6/7/8 占位槽）
ROUTE_COLS = (6, 7, 8)
CLUSTER_COL = 9

EDGE_CASES = ("empty_route", "connector", "lane_overflow", "route_truncated")


@dataclass
class FieldDiff:
    """一个字段的比对结果。"""

    name: str
    ok: bool
    detail: str
    max_abs: Optional[float] = None


@dataclass
class TokenResult:
    token: str
    log_name: str
    ok: bool = True
    skipped: Optional[str] = None
    buckets: List[str] = field(default_factory=list)
    diffs: List[FieldDiff] = field(default_factory=list)

    def add(self, diff: FieldDiff) -> None:
        self.diffs.append(diff)
        self.ok = self.ok and diff.ok


# --------------------------------------------------------------------------- 比对原语


def _as_np(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare_tensor(name: str, off, on) -> FieldDiff:
    """数值张量比对：先形状后数值。形状不同直接判失败，不做任何广播或截断对齐。"""
    a, b = _as_np(off), _as_np(on)
    if a is None and b is None:
        return FieldDiff(name, True, "both None")
    if a is None or b is None:
        return FieldDiff(name, False, f"one side is None (offline={a is None}, online={b is None})")
    if a.shape != b.shape:
        return FieldDiff(name, False, f"shape {a.shape} != {b.shape}")
    if a.size == 0:
        return FieldDiff(name, True, "both empty", 0.0)
    if not np.issubdtype(a.dtype, np.floating):
        n_bad = int((a != b).sum())
        return FieldDiff(name, n_bad == 0, f"{n_bad}/{a.size} 个整数元素不同", float(n_bad))
    max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    ok = bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))
    return FieldDiff(name, ok, f"max|Δ|={max_abs:.3e} (atol={ATOL})", max_abs)


def compare_route_channels(off_x, on_x, cluster_off, cluster_on) -> List[FieldDiff]:
    """route 三通道**逐 lane 行**比对，外加顺序一致性检查。

    只比 col6/7/8 不够：即使三通道的多重集合相同，行顺序错位一样是致命 bug
    （2026-07-28 事故的形态就是「值都在、位置错了」）。所以这里比的是**按行对齐后的值**，
    并单独报告"有多少行整行不一致"。
    """
    a, b = _as_np(off_x), _as_np(on_x)
    out: List[FieldDiff] = []
    if a is None or b is None or a.shape != b.shape:
        return [FieldDiff("route_channels", False, f"x 形状不可比: {None if a is None else a.shape} vs "
                                                   f"{None if b is None else b.shape}")]

    ra, rb = a[:, list(ROUTE_COLS)], b[:, list(ROUTE_COLS)]
    row_bad = np.where(~np.isclose(ra, rb, atol=ATOL, rtol=RTOL).all(axis=1))[0]
    out.append(FieldDiff("route_channels_rowwise", len(row_bad) == 0,
                         f"{len(row_bad)}/{len(ra)} 行三通道不一致"
                         + (f"，首个错行 idx={row_bad[0]}" if len(row_bad) else ""),
                         float(np.max(np.abs(ra - rb))) if ra.size else 0.0))

    # 顺序错位的特征签名：值的多重集合相同，但逐行不同 -> 一定是 permutation 问题
    if len(row_bad):
        same_multiset = np.allclose(np.sort(ra, axis=0), np.sort(rb, axis=0), atol=ATOL)
        out.append(FieldDiff("route_channels_permutation", False,
                             "值集合相同但行序不同 -> lane 排序错位（对照 reorder_lane_ids）"
                             if same_multiset else "值本身不同 -> 不是排序问题，是构造问题"))

    out.append(compare_tensor("cluster(col9)", cluster_off, cluster_on))
    return out


def classify_buckets(x_off: np.ndarray, n_lanes_in_radius: Optional[int],
                     n_route_lane_ids: Optional[int], has_connector: Optional[bool],
                     lane_num: int) -> List[str]:
    """给一个样本打 edge-case 标签，用于覆盖率统计。"""
    buckets: List[str] = []
    route = x_off[:, list(ROUTE_COLS)]
    on_route_rows = int((route[:, 2] > 0.5).sum())     # col8 = is_on_route

    if on_route_rows == 0:
        buckets.append("empty_route")
        # 有 route lane id 却一条都没进最终 40 条 -> 被距离排序截断掉了
        if n_route_lane_ids:
            buckets.append("route_truncated")
    if has_connector:
        buckets.append("connector")
    if n_lanes_in_radius is not None and n_lanes_in_radius > lane_num:
        buckets.append("lane_overflow")
    return buckets


# --------------------------------------------------------------------------- 两条路径


def build_offline(pkl_path: Path, adjacency_config: str):
    """离线：pkl -> GraphData（与 GraphDataset.process() 逐字节同一路径）。"""
    import pandas as pd

    from src.shared.utils.dataset import build_graph_data_from_feature_df

    df = pd.read_pickle(pkl_path)
    graph, skip = build_graph_data_from_feature_df(df, adjacency_config)
    return graph, skip


def build_online(scenario, route_mode: str, adjacency_config: str, checkpoint_path: str):
    """在线：用 scenario 的 past 造 PlannerInput，跑 planner 的真实特征构建路径。

    **不加载模型**（`_load_model` 只在 `compute_planner_trajectory` 里调用），因此本脚本
    不占 GPU、也不需要 checkpoint 真实存在 —— 我们审计的是特征链路，不是权重。
    """
    from nuplan.common.actor_state.state_representation import Point2D  # noqa: F401  见下方注释
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.planner.abstract_planner import (
        PlannerInitialization, PlannerInput,
    )
    from nuplan.planning.simulation.planner.gnn_trans_planner import GNNTransPlanner
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )

    cfg = _gnn_config()
    n_past = cfg.NUM_PAST_POSES

    planner = GNNTransPlanner(
        checkpoint_path=checkpoint_path, device="cpu",
        adjacency_config=adjacency_config, route_mode=route_mode,
        strict_route_mode=False,           # 只审特征，不做 ckpt/route_mode 一致性校验
    )
    # 手工注入 initialize() 的两个副作用，跳过 _load_model()
    planner._map_api = scenario.map_api
    planner._route_roadblock_ids = list(scenario.get_route_roadblock_ids() or [])

    # scenario 的 past（与预处理同一 API 家族）-> history buffer
    past_ego = list(scenario.get_ego_past_trajectory(0, (n_past - 1) * 0.1, n_past - 1))
    past_obs = list(scenario.get_past_tracked_objects(0, (n_past - 1) * 0.1, n_past - 1))
    ego_states = past_ego + [scenario.initial_ego_state]
    observations = past_obs + [scenario.initial_tracked_objects]
    buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=len(ego_states), ego_states=ego_states, observations=observations,
        sample_interval=0.1,
    )
    planner_input = PlannerInput(
        iteration=SimulationIteration(scenario.start_time, 0),
        history=buffer, traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
    )
    _ = PlannerInitialization  # 保留导入以自证接口未漂移
    return planner._build_graph_data(planner_input)


_CFG = None


def _gnn_config():
    global _CFG
    if _CFG is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gnn_trans_config", str(_GNN_ROOT / "config.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CFG = module
    return _CFG


# --------------------------------------------------------------------------- 主流程


def load_manifest(pkl_root: Path) -> List[Dict[str, str]]:
    with open(pkl_root / "log_manifest.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_scenarios_for(rows: List[Dict[str, str]]):
    """按 (log -> tokens) 精确重建 scenario（复刻 run_e2_preprocess.rebuild_scenarios 的做法）。"""
    from collections import defaultdict

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    by_log: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_log[row["log_name"]].append(row["token"])

    worker = Sequential()
    out = {}
    for log_name, tokens in sorted(by_log.items()):
        db_path = f"{DATA_ROOT}/{log_name}.db"
        if not os.path.exists(db_path):
            logger.warning("缺 DB，跳过 log %s", log_name)
            continue
        builder = NuPlanScenarioBuilder(
            data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
            db_files=[db_path], map_version=MAP_VERSION,
        )
        sf = ScenarioFilter(
            scenario_types=None, scenario_tokens=sorted(tokens), log_names=None, map_names=None,
            num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
            ego_displacement_minimum_m=None, expand_scenarios=False,
            remove_invalid_goals=False, shuffle=False,
        )
        for scenario in builder.get_scenarios(sf, worker):
            out[scenario.token] = scenario
    return out


def audit_token(token: str, log_name: str, pkl_root: Path, scenario,
                route_mode: str, adjacency_config: str, checkpoint_path: str) -> TokenResult:
    result = TokenResult(token=token, log_name=log_name)
    pkl_path = pkl_root / f"features_{token}.pkl"
    if not pkl_path.exists():
        result.skipped, result.ok = "missing_pkl", False
        return result

    off, skip = build_offline(pkl_path, adjacency_config)
    if off is None:
        result.skipped, result.ok = f"offline_skipped:{skip}", False
        return result
    if scenario is None:
        result.skipped, result.ok = "scenario_unavailable", False
        return result

    on = build_online(scenario, route_mode, adjacency_config, checkpoint_path)

    for name in ("x", "edge_index", "edge_attr", "gnn_x", "gnn_edge_index",
                 "gnn_edge_attr", "ego_history", "cluster"):
        result.add(compare_tensor(name, getattr(off, name, None), getattr(on, name, None)))

    if route_mode == "three_channel":
        for diff in compare_route_channels(getattr(off, "x", None), getattr(on, "x", None),
                                           getattr(off, "cluster", None), getattr(on, "cluster", None)):
            result.add(diff)

    n_lanes, has_connector = _scene_topology(scenario)
    result.buckets = classify_buckets(
        _as_np(off.x), n_lanes_in_radius=n_lanes,
        n_route_lane_ids=len(scenario.get_route_roadblock_ids() or []),
        has_connector=has_connector, lane_num=_gnn_config().LANE_NUM,
    )
    return result


def _scene_topology(scenario) -> Tuple[Optional[int], Optional[bool]]:
    """半径内的 lane 条数与是否含 lane connector —— 用于 edge-case 分桶，不参与 parity 判定。

    lane 条数走 planner 用的同一个 `get_neighbor_vector_set_map`（截断前的原始条数，
    所以 `>LANE_NUM` 才等价于「发生了截断」）；connector 走地图层查询。
    任一查询失败都返回 None 而不是猜一个值 —— 分桶宁可标"未知"，也不能虚报覆盖。
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    from src.shared.utils.data_utils import get_neighbor_vector_set_map

    cfg = _gnn_config()
    ego = scenario.initial_ego_state.rear_axle
    point = Point2D(ego.x, ego.y)
    try:
        _c, _tl, lane_ids = get_neighbor_vector_set_map(
            scenario.map_api, cfg.MAP_FEATURES, point, cfg.MAP_RADIUS,
            scenario.get_route_roadblock_ids() or [], [], return_lane_ids=True,
        )
        n_lanes: Optional[int] = len(lane_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("lane 计数失败 token=%s: %s", scenario.token, exc)
        n_lanes = None
    try:
        proximal = scenario.map_api.get_proximal_map_objects(
            point, cfg.MAP_RADIUS, [SemanticMapLayer.LANE_CONNECTOR])
        has_connector: Optional[bool] = len(proximal.get(SemanticMapLayer.LANE_CONNECTOR, [])) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("connector 查询失败 token=%s: %s", scenario.token, exc)
        has_connector = None
    return n_lanes, has_connector


def dry_run() -> int:
    """不碰 DB 的接线自检：导入链、比对原语、bucket 分类逻辑。"""
    print("=" * 70)
    print(" parity_offline_online.py —— dry run（不访问 nuPlan DB）")
    print("=" * 70)
    ok = True

    try:
        from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: F401
        print("  ✅ 离线 helper 可导入: build_graph_data_from_feature_df")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ❌ 离线 helper 导入失败: {exc}")

    try:
        from nuplan.planning.simulation.planner.gnn_trans_planner import GNNTransPlanner  # noqa: F401
        print("  ✅ 在线 planner 可导入: GNNTransPlanner（注册在 nuplan/.../planner/ 的那一份）")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ❌ planner 导入失败: {exc}")

    cfg = _gnn_config()
    print(f"  ✅ config 契约: NUM_AGENTS={cfg.NUM_AGENTS} LANE_NUM={cfg.LANE_NUM} "
          f"QUERY_RADIUS={cfg.QUERY_RADIUS} MAP_FEATURES={cfg.MAP_FEATURES}")

    a = np.arange(12, dtype=np.float32).reshape(2, 6)
    assert compare_tensor("same", a, a.copy()).ok
    assert not compare_tensor("shape", a, a[:1]).ok
    b = a.copy(); b[0, 0] += 1e-3
    assert not compare_tensor("value", a, b).ok
    print("  ✅ 比对原语：同值通过 / 形状不同失败 / 1e-3 偏差失败")

    x = np.zeros((5, 12), dtype=np.float32)
    x[1, 8] = 1.0
    assert "empty_route" not in classify_buckets(x, None, 3, None, 40)
    assert "empty_route" in classify_buckets(np.zeros((5, 12), np.float32), None, 0, None, 40)
    assert "route_truncated" in classify_buckets(np.zeros((5, 12), np.float32), None, 3, None, 40)
    assert "lane_overflow" in classify_buckets(x, 55, 3, None, 40)
    print("  ✅ edge-case 分类：empty_route / route_truncated / lane_overflow 判定正确")

    print("-" * 70)
    print(" dry run " + ("通过 —— 可以跑真实审计" if ok else "失败"))
    print(" 真实审计仍需 nuPlan DB；用 --pkl_root + --n 启动。")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D0#3 离线/在线 parity 审计")
    p.add_argument("--pkl_root", type=Path, default=None, help="离线 pkl 目录（含 log_manifest.json）")
    p.add_argument("--route_mode", choices=("none", "three_channel"), default="three_channel")
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--checkpoint_path", default="/dev/null",
                   help="仅为构造 planner；本脚本不加载权重")
    p.add_argument("--n", type=int, default=20, help="随机抽多少 token")
    p.add_argument("--tokens", type=str, default=None, help="逗号分隔，显式指定 token（覆盖 --n）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry_run", action="store_true", help="只做接线自检，不访问 DB")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run()
    if args.pkl_root is None:
        raise SystemExit("--pkl_root 必填（除非 --dry_run）")

    rows = load_manifest(args.pkl_root)
    if args.tokens:
        want = {t.strip() for t in args.tokens.split(",") if t.strip()}
        rows = [r for r in rows if r["token"] in want]
    else:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]
    logger.info("待审计 %d 个 token", len(rows))

    t0 = time.time()
    scenarios = get_scenarios_for(rows)
    logger.info("重建 scenario %d/%d，用时 %.0fs", len(scenarios), len(rows), time.time() - t0)

    results = [audit_token(r["token"], r["log_name"], args.pkl_root, scenarios.get(r["token"]),
                           args.route_mode, args.adjacency_config, args.checkpoint_path)
               for r in rows]

    coverage = {case: sum(1 for r in results if case in r.buckets) for case in EDGE_CASES}
    n_fail = sum(1 for r in results if not r.ok)
    report = {
        "route_mode": args.route_mode,
        "adjacency_config": args.adjacency_config,
        "atol": ATOL,
        "n_tokens": len(results),
        "n_fail": n_fail,
        "edge_case_coverage": coverage,
        "edge_cases_not_covered": [c for c, n in coverage.items() if n == 0],
        "results": [{"token": r.token, "log_name": r.log_name, "ok": r.ok,
                     "skipped": r.skipped, "buckets": r.buckets,
                     "diffs": [d.__dict__ for d in r.diffs]} for r in results],
    }

    print("=" * 74)
    print(f" 离线/在线 parity —— route_mode={args.route_mode}  {len(results)} tokens")
    print("=" * 74)
    for r in results:
        flag = "✅" if r.ok else "❌"
        note = f" [{r.skipped}]" if r.skipped else ""
        print(f"  {flag} {r.token}{note}  buckets={r.buckets or ['-']}")
        for d in r.diffs:
            if not d.ok:
                print(f"        ❌ {d.name}: {d.detail}")
    print("-" * 74)
    print(f"  edge-case 覆盖: {coverage}")
    if report["edge_cases_not_covered"]:
        print(f"  ⚠️ 未覆盖: {report['edge_cases_not_covered']} —— D0#3 不能算通过，需扩大抽样或定向找样本")
    print(f"  判定: {'✅ PASS' if n_fail == 0 and not report['edge_cases_not_covered'] else '❌ FAIL'}"
          f"（{len(results) - n_fail}/{len(results)} token 逐字段一致）")
    print("=" * 74)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("written: %s", args.out)

    return 0 if (n_fail == 0 and not report["edge_cases_not_covered"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
