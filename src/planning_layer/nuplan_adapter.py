"""C4 规划层 —— nuPlan devkit 适配层（把仿真里的 devkit 对象翻成规划层要的纯数据）。

分层理由：`config/geometry/candidates/scoring/layer` 五个核心模块**只依赖 numpy**
（坑#7：`plantf` env 没有 `torch_geometric`）；一切 devkit 依赖集中在本文件，
核心模块因此可以被单测在无 devkit 的环境里直接跑。

本文件负责设计文档 §8 第 2 步的三件事：
  ① `DetectionsTracks` → 常速前推的 agent 占据 `[T, N, 2]`；
  ② `map_api` → 限速 `speed_limit` 与可行驶区域闭包 `drivable_fn`；
  ③ `route_roadblock_ids` → route 中心线 `route_xy`（**上游塌陷时的兜底几何**）。

🔴 性能纪律：闭环一格 ≈80 min 是按「planner 每步 ~0.1 s」估的。地图查询是这里唯一
可能把它拖垮的东西，故：
  * 可行驶区域**按 patch 取一次并 union + 预备**，逐点判定走 `shapely.contains_xy`
    （向量化 C 实现，8400 点约 1 ms）；
  * patch 与 route 中心线都带**位移触发的重建**（ego 移动超过阈值才重建），
    不是每步重建。
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np

from .candidates import EgoSnapshot
from .scoring import SceneContext

logger = logging.getLogger(__name__)

# ── 参数（与官方判据无关的纯工程量，可调；与判据相关的阈值一律在 config.py）──
AGENT_RADIUS_M = 60.0          # 只前推这个半径内的 agent（更远的在 6 s 内够不着）
DRIVABLE_PATCH_RADIUS_M = 120.0  # 可行驶区域 patch 半径
DRIVABLE_REBUILD_AFTER_M = 25.0  # ego 移动超过它才重建 patch
ROUTE_RADIUS_M = 150.0         # route 中心线的提取半径
ROUTE_REBUILD_AFTER_M = 20.0   # ego 移动超过它才重建 route 中心线


def ego_snapshot(ego_state: Any) -> EgoSnapshot:
    """`EgoState` → 规划层的最小 ego 快照。

    ⚠️ 用**后轴**位姿：planner 的输出轨迹、训练特征、`two_stage_controller` 三者都以后轴为原点。
    速度取 `rear_axle_velocity_2d.x`（车体纵向），与候选生成里的 `v0` 同一口径。
    """
    return EgoSnapshot(
        x=float(ego_state.rear_axle.x),
        y=float(ego_state.rear_axle.y),
        heading=float(ego_state.rear_axle.heading),
        speed=float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
    )


def ego_dimensions(ego_state: Any) -> dict:
    """车身尺寸（相对**后轴**），用于按四角查可行驶区域。

    `VehicleParameters` 的定义：`front_length` = 后轴→前保险杠、`rear_length` = 后轴→后保险杠，
    正好与轨迹点的参考系一致，无需再做 `rear_axle_to_center` 平移。
    """
    p = ego_state.car_footprint.vehicle_parameters
    return {
        "ego_half_width": float(p.half_width),
        "ego_front_m": float(p.front_length),
        "ego_rear_m": float(p.rear_length),
    }


# ─────────────────────────────────────────────────────────────────────
# ① agent 占据：常速前推
# ─────────────────────────────────────────────────────────────────────
def agents_rollout(
    observation: Any,
    horizon: int,
    dt: float,
    ego_xy: Sequence[float],
    radius_m: float = AGENT_RADIUS_M,
) -> Optional[np.ndarray]:
    """`DetectionsTracks` → ``[T, N, 2]`` 逐时刻 agent 位置（常速外推）。

    :param observation: `PlannerInput.history.observations[-1]`，即 `DetectionsTracks`
    :param horizon: **必须等于规划 horizon** —— `scoring._validate_agents` 会校验，
        形状不符时抛的是明确错误而不是 numpy 的广播错。
    :returns: ``[T, N, 2]``；场景里没有可动 agent 时返回 None（`SceneContext` 允许 None）

    实现口径：
      * 只取**可动**类型（VEHICLE / PEDESTRIAN / BICYCLE）—— 静态障碍没有 `velocity` 属性；
      * `agent.velocity` 是**全局系**（devkit 自己的 `agent_global_velocities` 命名，
        `agents_preprocessing.py:530` 与本仓 `data_utils.global_velocity_to_local` 双向印证）；
      * 速度可能是 NaN（devkit 在 `_extract_agent_tensor` 里也专门判过）⇒ 按 0 处理，
        即退化成「静止占据」，方向保守；
      * 半径过滤只为控规模，不影响判据 —— 60 m 外的车在 6 s 内即使对向也够不到。
    """
    tracked = getattr(observation, "tracked_objects", None)
    if tracked is None:
        return None
    try:
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

        # ⚠️ 不能用 devkit 的 `AGENT_TYPES` —— 它**包含 `EGO`**
        #    （`tracked_objects_types.py:57`）。观测里正常不会出现 ego，但一旦出现，
        #    ego 会被当成一辆挡在自己身上的车，`collision_term` 直接给满罚。
        objects: List[Any] = tracked.get_tracked_objects_of_types([
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ])
    except (ImportError, AttributeError):  # 非 devkit 对象（单测里的假 tracks）
        objects = list(tracked)

    origin = np.asarray(ego_xy, dtype=float)
    pos, vel = [], []
    for obj in objects:
        c = obj.center
        p = np.array([float(c.x), float(c.y)])
        if float(np.hypot(*(p - origin))) > radius_m:
            continue
        v = getattr(obj, "velocity", None)
        vx, vy = (float(v.x), float(v.y)) if v is not None else (0.0, 0.0)
        if not (np.isfinite(vx) and np.isfinite(vy)):
            vx = vy = 0.0
        pos.append(p)
        vel.append([vx, vy])

    if not pos:
        return None
    p0 = np.asarray(pos, dtype=float)                       # [N, 2]
    v0 = np.asarray(vel, dtype=float)                       # [N, 2]
    t = (np.arange(horizon, dtype=float) * dt)[:, None, None]  # [T, 1, 1]
    return p0[None, :, :] + t * v0[None, :, :]              # [T, N, 2]


# ─────────────────────────────────────────────────────────────────────
# ② 可行驶区域闭包 + 限速
# ─────────────────────────────────────────────────────────────────────
class DrivableAreaClosure:
    """把官方 `DRIVABLE_AREA` 向量层裁到 ego 附近，做成一个可向量化查询的闭包。

    🔴 **为什么不直接每点调 `map_api.is_in_layer`**：官方实现是
    `vector_layer.contains(Point)`，对**全图**逐多边形做一次 geopandas 逐元素判定。
    本层一步要判 35 候选 × 60 步 × 4 角 = 8400 点 ⇒ 逐点调用会把闭环拖垮。
    这里改成「取一次 patch → `union_all` → `shapely.contains_xy` 向量化」，语义相同。

    ⚠️ **与官方的两处差异（必须写进论文的方法学说明）**：
      1. 官方判「点在**某一个**多边形内」，这里判「点在**并集**内」——
         相邻多边形的公共边界上，官方判 False、这里判 True。差异只发生在零测度集上，
         方向是**更宽松**，故不会制造假的越界惩罚。
      2. patch 之外的点按「未知」处理，记为**在区域内**（不罚）—— 宁可漏罚也不假罚。
         patch 半径 120 m 覆盖 6 s × 30 m/s 的最坏行程 + 余量。

    :param map_api: `AbstractMap`（实际是 `NuPlanMap`）
    """

    def __init__(
        self,
        map_api: Any,
        patch_radius_m: float = DRIVABLE_PATCH_RADIUS_M,
        rebuild_after_m: float = DRIVABLE_REBUILD_AFTER_M,
    ) -> None:
        self._map_api = map_api
        self._radius = float(patch_radius_m)
        self._rebuild_after = float(rebuild_after_m)
        self._geom = None
        self._center: Optional[np.ndarray] = None
        self._bounds: Optional[tuple] = None

    def update(self, ego_xy: Sequence[float]) -> None:
        """按需重建 patch —— ego 相对上次构建点移动超过阈值才重建。"""
        p = np.asarray(ego_xy, dtype=float)
        if self._center is not None and float(np.hypot(*(p - self._center))) <= self._rebuild_after:
            return
        self._geom = self._build_patch(p)
        self._center = p
        self._bounds = (
            p[0] - self._radius, p[1] - self._radius,
            p[0] + self._radius, p[1] + self._radius,
        )

    def _build_patch(self, center: np.ndarray):
        """取 patch 内的 drivable 多边形并 union。失败时返回 None（=全部按可行驶处理）。"""
        from shapely import box

        try:
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer

            layer = self._map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
        except Exception as e:  # noqa: BLE001 —— 地图后端不支持就降级，不让闭环崩
            logger.warning("[PlanningLayer] 取 DRIVABLE_AREA 向量层失败(%s)，本层不做可行驶区域检查", e)
            return None
        if layer is None or len(layer) == 0:
            return None
        patch = box(center[0] - self._radius, center[1] - self._radius,
                    center[0] + self._radius, center[1] + self._radius)
        idx = layer.sindex.query(patch, predicate="intersects")
        if len(idx) == 0:
            logger.warning(
                "[PlanningLayer] ego 周围 %.0f m 内没有 drivable 多边形，本步不做可行驶区域检查",
                self._radius,
            )
            return None
        geom = layer.geometry.iloc[idx].union_all()
        # `prepare` 给并集建一次空间索引；之后 `contains_xy` 从 O(多边形数) 降到 O(log)。
        # 实测（真地图 3600 点）：17.3 ms → 0.4 ms（对官方逐点 328 ms 是 910×），
        # 而本层一步要判 35×60×4 = 8400 点 ⇒ 约 1 ms/步，对 80 min 的一格闭环可忽略。
        # 证据脚本：`scripts/eval_closed_loop/validate_drivable_closure.py`
        from shapely import prepare

        prepare(geom)
        return geom

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """``[M, 2]`` → ``[M]`` bool（True = 在可行驶区域内）。"""
        from shapely import contains_xy

        pts = np.asarray(pts, dtype=float)
        if self._geom is None:
            return np.ones(len(pts), dtype=bool)
        inside = np.asarray(contains_xy(self._geom, pts[:, 0], pts[:, 1]), dtype=bool)
        if self._bounds is not None:  # patch 之外按「未知」处理，不罚
            x0, y0, x1, y1 = self._bounds
            outside_patch = (pts[:, 0] < x0) | (pts[:, 0] > x1) | (pts[:, 1] < y0) | (pts[:, 1] > y1)
            inside = inside | outside_patch
        return inside


def speed_limit_mps(map_api: Any, ego_state: Any) -> Optional[float]:
    """ego 当前所在 lane / lane connector 的限速（m/s），取不到返回 None。

    口径照抄官方 `speed_limit_compliance._get_speed_limit_violation`：
      * 在 Lane 上 ⇒ 该 lane 的 `speed_limit_mps`；
      * 在 LaneConnector 上 ⇒ 取其**进出边**的限速最大值（官方就是取 max，宽松方向）；
      * 任何一环拿不到（新地图 GPKG 可能没有限速数据）⇒ None，`rules_term` 会跳过该项。
    """
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    except ImportError:
        return None

    p = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    lanes = map_api.get_all_map_objects(p, SemanticMapLayer.LANE)
    if lanes:
        limits = [ln.speed_limit_mps for ln in lanes]
    else:
        connectors = map_api.get_all_map_objects(p, SemanticMapLayer.LANE_CONNECTOR)
        limits = []
        for conn in connectors:
            limits.extend([e.speed_limit_mps for e in conn.outgoing_edges + conn.incoming_edges])
    limits = [float(v) for v in limits if v is not None and np.isfinite(v)]
    return max(limits) if limits else None


# ─────────────────────────────────────────────────────────────────────
# ③ route 中心线
# ─────────────────────────────────────────────────────────────────────
class RouteCenterline:
    """从 `route_roadblock_ids` 拼一条沿行进方向的中心线折线，带位移触发的重建缓存。

    🔴 这是**上游塌陷时的兜底几何**（基准闭环塌陷率 0.594，是本层的主场景）。
    ⚠️ 拿到的折线**起点通常在 ego 后方** —— `candidates.generate` 会先
    `geometry.trim_to_nearest` 裁到 ego 投影点，否则会把车瞬移回 route 起点（坑#20）。
    """

    def __init__(
        self,
        map_api: Any,
        radius_m: float = ROUTE_RADIUS_M,
        rebuild_after_m: float = ROUTE_REBUILD_AFTER_M,
    ) -> None:
        self._map_api = map_api
        self._radius = float(radius_m)
        self._rebuild_after = float(rebuild_after_m)
        self._xy: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None

    def get(self, route_roadblock_ids: Sequence[str], ego_xy: Sequence[float]) -> Optional[np.ndarray]:
        p = np.asarray(ego_xy, dtype=float)
        if self._center is not None and float(np.hypot(*(p - self._center))) <= self._rebuild_after:
            return self._xy
        self._xy = self._build(route_roadblock_ids, p)
        self._center = p
        return self._xy

    def _build(self, route_roadblock_ids: Sequence[str], ego: np.ndarray) -> Optional[np.ndarray]:
        if not route_roadblock_ids:
            return None
        try:
            from nuplan.common.actor_state.state_representation import Point2D
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
                prune_route_by_connectivity,
            )
        except ImportError:
            return None

        point = Point2D(float(ego[0]), float(ego[1]))
        layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        layers = self._map_api.get_proximal_map_objects(point, self._radius, layer_names)
        nearby_ids = {obj.id for name in layer_names for obj in layers[name]}
        # 保序裁剪（devkit 约定 route_roadblock_ids 沿行进方向有序）
        pruned = prune_route_by_connectivity(list(route_roadblock_ids), nearby_ids)
        if not pruned:
            return None

        polyline: List[np.ndarray] = []
        prev_lane = None
        for rb_id in pruned:
            rb = self._map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK)
            if rb is None:
                rb = self._map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if rb is None or not rb.interior_edges:
                continue
            lane = _pick_lane(rb.interior_edges, prev_lane, polyline, ego)
            if lane is None:
                continue
            pts = np.array([[s.x, s.y] for s in lane.baseline_path.discrete_path], dtype=float)
            if len(pts) >= 2:
                polyline.append(pts)
            prev_lane = lane

        if not polyline:
            return None
        return np.concatenate(polyline, axis=0)


def _pick_lane(edges: List[Any], prev_lane: Any, polyline: List[np.ndarray], ego: np.ndarray):
    """在一个 roadblock 的平行 lane 里选一条。

    ⚠️ **不能随便挑一条** —— 平行 lane 之间横向差一个车道宽，随机挑会让兜底路径在
    roadblock 边界上横跳，`trim_to_nearest` 之后仍会留下一个折角。
    选法：优先接着上一条 lane 的 `outgoing_edges` 走（保证纵向连续），
    没有连接关系时按「baseline 起点离**当前路径末端**（首个 roadblock 则离 ego）最近」选。
    """
    if prev_lane is not None:
        outgoing_ids = {e.id for e in prev_lane.outgoing_edges}
        connected = [e for e in edges if e.id in outgoing_ids]
        if connected:
            return connected[0]
    anchor = polyline[-1][-1] if polyline else ego
    best, best_d = None, float("inf")
    for e in edges:
        path = e.baseline_path.discrete_path
        if not path:
            continue
        d = float(np.hypot(path[0].x - anchor[0], path[0].y - anchor[1]))
        if d < best_d:
            best, best_d = e, d
    return best


# ─────────────────────────────────────────────────────────────────────
# 组装
# ─────────────────────────────────────────────────────────────────────
def build_scene_context(
    ego_state: Any,
    observation: Any,
    map_api: Any,
    horizon: int,
    dt: float,
    drivable: Optional[DrivableAreaClosure] = None,
    speed_limit: Optional[float] = None,
) -> SceneContext:
    """把一步仿真的 devkit 输入组装成 `SceneContext`。

    `origin` / `progress_direction` 故意留空 —— 由 `layer.refine()` 统一兜底
    （origin=ego 位置、方向=route 方向或 ego 朝向），避免两处各写一套兜底。
    """
    ego_xy = (float(ego_state.rear_axle.x), float(ego_state.rear_axle.y))
    if drivable is not None:
        drivable.update(ego_xy)
    return SceneContext(
        agents_xy=agents_rollout(observation, horizon, dt, ego_xy),
        speed_limit=speed_limit,
        drivable_fn=drivable,
        **ego_dimensions(ego_state),
    )
