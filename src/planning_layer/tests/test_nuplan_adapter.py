"""C4 规划层接线单测 —— 判据 [I]–[L]（设计文档 §8 第 2 步的三件事 + 回归命门）。

判据同样**先于实现写死**（见 `C4_规划形式化_设计_20260801.md` §3.3 第 4 条的做法）：

| 判据 | 内容 |
|---|---|
| [I] | `DetectionsTracks` → `agents_xy`：形状 / 常速正确性 / 静态物与 NaN / 半径 / 端到端避障 |
| [J] | `map_api` → `speed_limit` + `drivable_fn`：官方口径、patch 外不假罚、**车身尺寸真的传到 `refine`** |
| [K] | `route_roadblock_ids` → `route_xy`：顺序与连通、平行 lane 选取、缓存、塌陷不瞬移 |
| [L] | **命门**：不设 `MAGNET_PLANNING_LAYER` 时，planner 轨迹与接线前算法逐位相同 |

跑法：
    /home/team/anaconda3/envs/nuplan/bin/python -m pytest \\
        src/planning_layer/tests/test_nuplan_adapter.py -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.planning_layer import SceneContext, preset, refine  # noqa: E402
from src.planning_layer import nuplan_adapter as A  # noqa: E402

DT = 0.1
T = 60


# ─────────────────────────────────────────────────────────────────────
# 假 devkit 对象（不依赖 nuPlan DB，单测必须能离线跑）
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _P:
    x: float
    y: float


@dataclass
class _FakeAgent:
    center: _P
    velocity: Optional[_P]


class _FakeTracks:
    """行为对齐 `TrackedObjects.get_tracked_objects_of_types`：只回可动类型。"""

    def __init__(self, agents: List[_FakeAgent], statics: Optional[List[_FakeAgent]] = None):
        self._agents = agents
        self._statics = statics or []

    def get_tracked_objects_of_types(self, types):  # noqa: ANN001
        return list(self._agents)

    def __iter__(self):
        return iter(self._agents + self._statics)


class _FakeObservation:
    def __init__(self, tracks: _FakeTracks):
        self.tracked_objects = tracks


@dataclass
class _FakeVehicleParams:
    width: float = 2.297
    front_length: float = 4.049
    rear_length: float = 1.127

    @property
    def half_width(self) -> float:
        return self.width / 2.0


class _FakeFootprint:
    vehicle_parameters = _FakeVehicleParams()


@dataclass
class _FakePose:
    x: float
    y: float
    heading: float


class _FakeDyn:
    def __init__(self, vx: float):
        self.rear_axle_velocity_2d = _P(vx, 0.0)


class _FakeEgoState:
    def __init__(self, x: float, y: float, heading: float, speed: float):
        self.rear_axle = _FakePose(x, y, heading)
        self.dynamic_car_state = _FakeDyn(speed)
        self.car_footprint = _FakeFootprint()


class _FakeLane:
    def __init__(self, lane_id: str, pts: np.ndarray, speed_limit: Optional[float] = None):
        self.id = lane_id
        self.speed_limit_mps = speed_limit
        self._pts = pts
        self.outgoing_edges: List["_FakeLane"] = []
        self.incoming_edges: List["_FakeLane"] = []

    @property
    def baseline_path(self):
        return self

    @property
    def discrete_path(self):
        return [_FakePose(float(p[0]), float(p[1]), 0.0) for p in self._pts]


class _FakeRoadblock:
    def __init__(self, rb_id: str, lanes: List[_FakeLane]):
        self.id = rb_id
        self.interior_edges = lanes


class _FakeMapApi:
    """只实现适配层用到的四个方法；调用次数记账用于验证缓存。"""

    def __init__(self, roadblocks: Optional[dict] = None, lanes_at_point: Optional[List[_FakeLane]] = None):
        self._roadblocks = roadblocks or {}
        self._lanes_at_point = lanes_at_point or []
        self.n_proximal_calls = 0

    def get_proximal_map_objects(self, point, radius, layers):  # noqa: ANN001
        self.n_proximal_calls += 1
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        out = {layer: [] for layer in layers}
        out[SemanticMapLayer.ROADBLOCK] = [
            _FakeRoadblock(k, v) for k, v in self._roadblocks.items()
        ]
        return out

    def get_map_object(self, object_id, layer):  # noqa: ANN001
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        if layer != SemanticMapLayer.ROADBLOCK:
            return None
        lanes = self._roadblocks.get(object_id)
        return _FakeRoadblock(object_id, lanes) if lanes else None

    def get_all_map_objects(self, point, layer):  # noqa: ANN001
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        if layer == SemanticMapLayer.LANE:
            return list(self._lanes_at_point)
        return []


def straight_line(x0: float, x1: float, y: float, n: int = 20) -> np.ndarray:
    return np.stack([np.linspace(x0, x1, n), np.full(n, y)], axis=1)


# ─────────────────────────────────────────────────────────────────────
# [I] agent 占据前推
# ─────────────────────────────────────────────────────────────────────
def test_I_rollout_shape_and_constant_velocity():
    obs = _FakeObservation(_FakeTracks([_FakeAgent(_P(10.0, 0.0), _P(2.0, -1.0))]))
    out = A.agents_rollout(obs, T, DT, (0.0, 0.0))
    assert out.shape == (T, 1, 2), "形状必须是 [T, N, 2] 且 T == horizon"
    assert np.allclose(out[0, 0], [10.0, 0.0])
    assert np.allclose(out[10, 0], [10.0 + 2.0 * 1.0, -1.0 * 1.0]), "常速外推必须按 t=i·dt"


def test_I_nan_velocity_becomes_static_not_nan():
    """devkit 的 agent 速度可能是 NaN（官方 `_extract_agent_tensor` 也专门判过）。"""
    obs = _FakeObservation(_FakeTracks([_FakeAgent(_P(5.0, 1.0), _P(float("nan"), 0.0))]))
    out = A.agents_rollout(obs, T, DT, (0.0, 0.0))
    assert np.all(np.isfinite(out)), "NaN 速度必须退化成静止，不得把 NaN 传进打分"
    assert np.allclose(out[-1, 0], [5.0, 1.0])


def test_I_far_agents_and_empty_scene():
    far = _FakeObservation(_FakeTracks([_FakeAgent(_P(500.0, 0.0), _P(0.0, 0.0))]))
    assert A.agents_rollout(far, T, DT, (0.0, 0.0)) is None, "半径外应被丢弃"
    empty = _FakeObservation(_FakeTracks([]))
    assert A.agents_rollout(empty, T, DT, (0.0, 0.0)) is None, "无 agent 应返回 None（不是空数组）"


def test_I_end_to_end_avoids_a_real_blocking_agent():
    """端到端：把 `DetectionsTracks` 走完整适配路径接进 `refine`，必须绕开挡道车。

    与判据 [C] 同源，但 [C] 直接喂 `agents_xy`，这里测的是**适配层本身**没把
    坐标 / 时间步 / 单位搞错。
    """
    ego = A.ego_snapshot(_FakeEgoState(0.0, 0.0, 0.0, 8.0))
    ref = np.stack([np.arange(T) * DT * 8.0, np.zeros(T)], axis=1)
    obs = _FakeObservation(_FakeTracks([_FakeAgent(_P(20.0, 0.0), _P(0.0, 0.0))]))
    ctx = SceneContext(agents_xy=A.agents_rollout(obs, T, DT, (ego.x, ego.y)))
    out = refine(ref, ego, ctx=ctx, cfg=preset("full"))
    d = np.min(np.hypot(out[:, 0] - 20.0, out[:, 1] - 0.0))
    assert d >= 2.5, f"选中的候选仍与挡道车相撞（最小间距 {d:.2f} m）"


# ─────────────────────────────────────────────────────────────────────
# [J] 限速 + 可行驶区域
# ─────────────────────────────────────────────────────────────────────
def test_J_speed_limit_from_lane():
    api = _FakeMapApi(lanes_at_point=[_FakeLane("l1", straight_line(0, 10, 0), speed_limit=8.94)])
    assert A.speed_limit_mps(api, _FakeEgoState(0, 0, 0, 5)) == pytest.approx(8.94)


def test_J_speed_limit_missing_returns_none_not_zero():
    """拿不到限速必须返回 None（`rules_term` 跳过该项），返回 0 会把所有候选判超速。"""
    api = _FakeMapApi(lanes_at_point=[_FakeLane("l1", straight_line(0, 10, 0), speed_limit=None)])
    assert A.speed_limit_mps(api, _FakeEgoState(0, 0, 0, 5)) is None
    assert A.speed_limit_mps(_FakeMapApi(), _FakeEgoState(0, 0, 0, 5)) is None


def test_J_drivable_closure_matches_polygon_and_is_permissive_outside_patch():
    """闭包在 patch 内按并集判定；patch 外按「未知」判为可行驶（宁可漏罚不假罚）。"""
    gpd = pytest.importorskip("geopandas")
    from shapely import box

    corridor = box(-500.0, -3.0, 500.0, 3.0)

    class _MapWithLayer:
        def _get_vector_map_layer(self, layer):  # noqa: ANN001
            return gpd.GeoDataFrame(geometry=[corridor])

    closure = A.DrivableAreaClosure(_MapWithLayer(), patch_radius_m=50.0)
    closure.update((0.0, 0.0))
    pts = np.array([[0.0, 0.0], [0.0, 10.0], [1000.0, 1000.0]])
    inside = closure(pts)
    assert inside[0] and not inside[1], "patch 内必须按多边形判定"
    assert inside[2], "patch 之外是未知区域，不得计为越界"


def test_J_drivable_closure_rebuilds_only_after_threshold():
    gpd = pytest.importorskip("geopandas")
    from shapely import box

    class _CountingMap:
        def __init__(self):
            self.n = 0

        def _get_vector_map_layer(self, layer):  # noqa: ANN001
            self.n += 1
            return gpd.GeoDataFrame(geometry=[box(-500, -3, 500, 3)])

    api = _CountingMap()
    closure = A.DrivableAreaClosure(api, patch_radius_m=50.0, rebuild_after_m=25.0)
    closure.update((0.0, 0.0))
    closure.update((10.0, 0.0))     # 未超阈值 ⇒ 不重建
    assert api.n == 1, "小位移不得重建 patch（闭环性能命门）"
    closure.update((40.0, 0.0))     # 超阈值 ⇒ 重建
    assert api.n == 2


def test_J_ego_dimensions_reach_refine():
    """🔴 回归判据：车身尺寸必须真的传到 `refine()` 内部。

    08-01 接线时抓到的第 10 个实现错：`layer._with_origin` 手写字段列表，
    漏掉 `ego_half_width/front/rear` 三个字段，而 `refine()` 在 `origin is None` 时
    **必然**走那条路径 ⇒ 可行驶区域退化成只查质心，与官方判四角不一致。
    这里用「只有车角会出界的走廊」把差异逼出来。
    """
    ego = A.ego_snapshot(_FakeEgoState(0.0, 0.0, 0.0, 6.0))
    ref = np.stack([np.arange(T) * DT * 6.0, np.zeros(T)], axis=1)
    seen = {"n_pts": 0}

    def drivable(pts: np.ndarray) -> np.ndarray:
        seen["n_pts"] = max(seen["n_pts"], len(pts))
        return np.ones(len(pts), dtype=bool)

    dims = A.ego_dimensions(_FakeEgoState(0.0, 0.0, 0.0, 6.0))
    refine(ref, ego, ctx=SceneContext(drivable_fn=drivable, **dims), cfg=preset("full"))
    assert seen["n_pts"] == 4 * T, (
        f"给了车身尺寸就必须按四角查（期望 {4 * T} 点，实际 {seen['n_pts']}）"
    )


# ─────────────────────────────────────────────────────────────────────
# [K] route 中心线
# ─────────────────────────────────────────────────────────────────────
def _two_roadblock_map():
    l1 = _FakeLane("A1", straight_line(0, 50, 0))
    l2a = _FakeLane("B1", straight_line(50, 100, 0))
    l2b = _FakeLane("B2", straight_line(50, 100, 3.7))   # 平行车道
    l1.outgoing_edges = [l2a]
    return _FakeMapApi(roadblocks={"A": [l1], "B": [l2b, l2a]})


def test_K_route_follows_order_and_connectivity():
    api = _two_roadblock_map()
    xy = A.RouteCenterline(api).get(["A", "B"], (0.0, 0.0))
    assert xy is not None and len(xy) > 2
    assert xy[0, 0] == pytest.approx(0.0), "必须从 route 起点开始（裁剪交给 trim_to_nearest）"
    assert xy[-1, 0] == pytest.approx(100.0), "必须按 route 顺序拼到末端"
    # B 里先列的是 B2（y=3.7），但连通的是 B1（y=0）⇒ 必须选 B1
    assert np.max(np.abs(xy[:, 1])) < 1e-9, "平行车道里必须选与上一条 lane 连通的那条"


def test_K_empty_route_returns_none():
    assert A.RouteCenterline(_FakeMapApi()).get([], (0.0, 0.0)) is None


def test_K_route_cache_rebuilds_only_after_threshold():
    api = _two_roadblock_map()
    rc = A.RouteCenterline(api, rebuild_after_m=20.0)
    rc.get(["A", "B"], (0.0, 0.0))
    rc.get(["A", "B"], (5.0, 0.0))
    assert api.n_proximal_calls == 1, "小位移不得重建 route（闭环性能命门）"
    rc.get(["A", "B"], (40.0, 0.0))
    assert api.n_proximal_calls == 2


def test_K_collapsed_upstream_with_real_route_does_not_teleport():
    """塌陷 + route 起点远在 ego 后方 —— 走完整适配路径不得把 ego 拽回去（坑#20）。"""
    api = _two_roadblock_map()
    ego = A.ego_snapshot(_FakeEgoState(50.0, 0.0, 0.0, 5.0))
    route_xy = A.RouteCenterline(api).get(["A", "B"], (ego.x, ego.y))
    collapsed = np.tile(np.array([[ego.x, ego.y]]), (T, 1))   # 上游恒 bias
    out = refine(collapsed, ego, cfg=preset("full"), route_xy=route_xy)
    assert out[0, 0] == pytest.approx(50.0, abs=0.05), "首点必须仍在 ego 处，不得瞬移回 route 起点"
    assert out[-1, 0] > 55.0, "塌陷兜底后仍须前进（针对不作为失效）"


# ─────────────────────────────────────────────────────────────────────
# [L] 组装
# ─────────────────────────────────────────────────────────────────────
def test_L_build_scene_context_is_self_consistent():
    ego_state = _FakeEgoState(3.0, -2.0, 0.3, 7.0)
    obs = _FakeObservation(_FakeTracks([_FakeAgent(_P(12.0, -2.0), _P(1.0, 0.0))]))
    ctx = A.build_scene_context(ego_state, obs, _FakeMapApi(), T, DT, speed_limit=13.4)
    assert ctx.agents_xy.shape == (T, 1, 2)
    assert ctx.speed_limit == pytest.approx(13.4)
    assert ctx.ego_half_width == pytest.approx(2.297 / 2)
    assert ctx.ego_front_m == pytest.approx(4.049)
    assert ctx.origin is None and ctx.progress_direction is None, (
        "origin / 方向必须留给 refine() 统一兜底，避免两处各写一套"
    )
