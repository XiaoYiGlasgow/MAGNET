"""GNN+Transformer MAGNET planner（路线B，VT-2026-04191 阶段2/3 重写）。

在线特征构建**完全复用训练同一链路**：
  encoding_vectornet_features(E0 契约参数) → build_graph_data_from_feature_df（与 GraphDataset.process 同一 helper）
契约对齐 E0：历史 20帧/2s、未来 60帧/6s、agents=20、radius=50、地图 LANE-only（max 40×50）、
单一 default edge builder、EGO_HISTORY[20,2] 局部逐帧增量。

模型输出 60×2 局部增量 → cumsum → 旋转回全局 → 纯参考轨迹；运动学可行性由 simulation 层
two_stage_controller（tracker=lqr/ilqr + kinematic_bicycle_model）强制，planner **不**在内部搓运动学。

删除全部回退（匀速基线+0.02×output / 随机初始化 / Linear(1,120) / mock 特征 / 第二套 edge builder /
静止后备 / 异常吞噬 / debug 写文件）——加载或推理失败即 raise，闭环评测计为失败、不静默替换。
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from typing import List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    Point2D, StateSE2, StateVector2D, TimePoint,
)
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

# GNN_Trans 源码（config / src.shared.utils）+ parent models（TransformerPredictor E0 生效副本）上 sys.path
_DEVKIT_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit"
_GNN_TRANS_ROOT = f"{_DEVKIT_ROOT}/GNN_Trans"
_MODELS_DIR = f"{_DEVKIT_ROOT}/models"
for _p in (_GNN_TRANS_ROOT, _MODELS_DIR, _DEVKIT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HEADING_EPS = 1e-3  # 位移过小则沿用上一朝向（避免 atan2 抖动）
_GNN_CONFIG = None    # GNN_Trans/config.py（按文件路径显式加载，避开 run_simulation 里 'config' 名冲突）


def _load_gnn_config():
    """按文件路径显式加载 GNN_Trans/config.py（E0 契约常量单一源）。

    run_simulation 进程里 sys.modules 已缓存别处的 'config' 模块，裸 `import config` 会取到错模块，
    故用 importlib 直接从已知文件路径加载，保证拿到 GNN_Trans 的 config（NUM_AGENTS/MAP_*/RADIUS 等）。
    """
    global _GNN_CONFIG
    if _GNN_CONFIG is None:
        spec = importlib.util.spec_from_file_location("gnn_trans_config", f"{_GNN_TRANS_ROOT}/config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GNN_CONFIG = module
    return _GNN_CONFIG


class GNNTransPlanner(AbstractPlanner):
    """基于 MAGNET TransformerPredictor 的 imitation planner（路线B，官方闭环评测）。"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        horizon_seconds: float = 6.0,
        sampling_time: float = 0.1,
        adjacency_config: str = "default",
        route_mode: str = "none",
        strict_route_mode: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if route_mode not in ("none", "three_channel"):
            raise ValueError(f"[GNNTransPlanner] 非法 route_mode={route_mode!r}，只允许 none/three_channel")
        self._route_mode = route_mode
        self._strict_route_mode = strict_route_mode
        self._checkpoint_path = checkpoint_path
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._horizon_seconds = horizon_seconds
        self._sampling_time = sampling_time
        self._num_output_points = int(round(horizon_seconds / sampling_time))  # 6s/0.1s=60
        self._adjacency_config = adjacency_config

        self.future_trajectory_sampling = TrajectorySampling(
            num_poses=self._num_output_points,
            time_horizon=horizon_seconds,
            interval_length=sampling_time,
        )
        self._model = None
        self._map_api = None
        self._route_roadblock_ids: List[str] = []  # 由 initialize 注入（D1-2 route 契约）
        self._diag_logged = False  # 首个 planner 调用打印一次 history 帧数（契约核对）
        logger.info(
            "[GNNTransPlanner] device=%s horizon=%.1fs dt=%.2fs points=%d adjacency=%s route_mode=%s",
            self._device, horizon_seconds, sampling_time, self._num_output_points,
            adjacency_config, route_mode,
        )

    # ---- checkpoint 与 route_mode 一致性（旧 ckpt 从未见过 route 通道，喂进去会失配）----
    def _check_route_mode_consistency(self) -> None:
        """比对 checkpoint 训练时的 route_mode 与当前 planner 配置。

        训练脚本会在 checkpoint 同目录写 sidecar `train_meta.json`（含 route_mode 等）。
        旧的 E1 checkpoint 没有该 sidecar，按 `route_mode="none"` 处理 —— 这正是实测结论：
        把 route 三通道喂给从未见过它的 E1 checkpoint，会造成预测漂移
        （10 个 E0 样本：ADE 2.45 m / FDE 7.10 m / 最大终点偏移 20.74 m），
        即阶段5 的旧基线在无该保护时**无法复现**。
        """
        meta_path = os.path.join(os.path.dirname(self._checkpoint_path), "train_meta.json")
        ckpt_mode = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    ckpt_mode = json.load(f).get("route_mode")
            except (OSError, ValueError) as e:
                logger.warning("[GNNTransPlanner] 读取 %s 失败: %s", meta_path, e)
        if ckpt_mode is None:
            ckpt_mode = "none"
            logger.warning(
                "[GNNTransPlanner] checkpoint 无 train_meta.json sidecar，按 route_mode='none' 处理: %s",
                self._checkpoint_path,
            )
        if ckpt_mode != self._route_mode:
            msg = (f"[GNNTransPlanner] route_mode 失配：checkpoint 训练用 '{ckpt_mode}'，"
                   f"planner 配置 '{self._route_mode}'。喂入模型未见过的输入会使结果不可解释。"
                   f"（如确需如此，设 strict_route_mode=False 降级为 warning）")
            if self._strict_route_mode:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("[GNNTransPlanner] route_mode 一致性校验通过: %s", ckpt_mode)

    # ---- 模型加载（无回退：失败即 raise）----
    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformer_predictor import TransformerPredictor  # parent models/（E0 生效副本）

        model = TransformerPredictor().to(self._device)
        if not os.path.exists(self._checkpoint_path):
            raise FileNotFoundError(f"[GNNTransPlanner] checkpoint 不存在: {self._checkpoint_path}")
        state_dict = torch.load(self._checkpoint_path, map_location=self._device)  # E1 bare state_dict
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        logger.info("[GNNTransPlanner] 已加载 checkpoint: %s", self._checkpoint_path)
        self._check_route_mode_consistency()

    @classmethod
    def get_requirements(cls) -> List[Type[Observation]]:
        return [DetectionsTracks]

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        # VT-2026-04191 D1-2：route 契约扩展。原实现丢弃 route_roadblock_ids（阶段⑨ 归因中
        # 「route/reference-line 缺失」列为闭环 drivable_area_compliance=0.296 的首要结构性缺口）。
        self._route_roadblock_ids = list(initialization.route_roadblock_ids or [])
        logger.info("[GNNTransPlanner] route_roadblock_ids=%d 条", len(self._route_roadblock_ids))
        self._load_model()

    def name(self) -> str:
        return "GNNTransPlanner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    # ---- 主入口：模型推理 → 参考轨迹（无异常吞噬）----
    def compute_planner_trajectory(self, current_input: PlannerInput) -> InterpolatedTrajectory:
        self._load_model()
        from torch_geometric.data import Batch

        graph_data = self._build_graph_data(current_input)
        batch = Batch.from_data_list([graph_data]).to(self._device)  # data.batch=zeros(N)，batch_size=1
        with torch.no_grad():
            output = self._model(batch)  # [1, 120] = 60×2 局部增量
        return self._output_to_trajectory(output, current_input.history.ego_states[-1])

    # ---- 在线特征构建：完全对齐 E0 训练契约 + 复用训练同一图构建 helper ----
    def _build_graph_data(self, current_input: PlannerInput):
        config = _load_gnn_config()
        from src.shared.utils.data_utils import (
            sampled_past_ego_states_to_tensor,
            sampled_past_timestamps_to_tensor,
            sampled_tracked_objects_to_tensor_list,
            agent_past_process,
            get_neighbor_vector_set_map,
            map_process,
            encoding_vectornet_features,
            get_route_lane_progress,
            build_route_channels,
            reorder_lane_ids,
        )
        from src.shared.utils.dataset import build_graph_data_from_feature_df

        history = current_input.history
        current_ego_state = history.ego_states[-1]
        if not self._diag_logged:  # 首帧核对 history 长度是否满足 20 帧契约
            logger.info("[GNNTransPlanner] history 首帧: ego_states=%d observations=%d",
                        len(history.ego_states), len(history.observations))
            self._diag_logged = True

        # 1. ego 历史 20帧/2s（E0 契约；模型 ego 分支锁 20 帧）
        ego_states = list(history.ego_states[-config.NUM_PAST_POSES:])
        past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
        past_timestamps = sampled_past_timestamps_to_tensor(ego_states)

        # 2. 历史 tracked objects：传 DetectionsTracks 本身（与预处理 get_tracked_objects_at_iteration 同类型；
        #    _extract_agent_tensor 会取其 .tracked_objects=TrackedObjects。切勿传 obs.tracked_objects，否则再钻一层进普通 list）
        past_tracked_objects = list(history.observations[-config.NUM_PAST_POSES:])
        while len(past_tracked_objects) < config.NUM_PAST_POSES:
            past_tracked_objects.insert(0, DetectionsTracks(TrackedObjects([])))  # 空帧占位
        tracked_list, tracked_types = sampled_tracked_objects_to_tensor_list(
            past_tracked_objects[-config.NUM_PAST_POSES:]
        )

        # 3. agents=20（E0 契约，原 planner 用 15）
        ego_past, agents_past, _ = agent_past_process(
            past_ego_states, past_timestamps, tracked_list, tracked_types, config.NUM_AGENTS
        )

        # 4. 地图 LANE-only / radius=50 / max LANE 40×50（E0 契约；tl 仍传 [] 与训练同）
        #    route 契约：传入真实 route_roadblock_ids 并取回与 coords['LANE'] 同序的 lane_ids，
        #    据此生成 route 三通道（方案B：写 lane 节点 col6/col7/col8，不扩节点维度）。
        ego_position = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
        coords, tl_data, lane_ids = get_neighbor_vector_set_map(
            self._map_api, config.MAP_FEATURES, ego_position, config.MAP_RADIUS,
            self._route_roadblock_ids, [], return_lane_ids=True,
        )
        route_lane_progress = get_route_lane_progress(
            self._map_api, ego_position, config.MAP_RADIUS, self._route_roadblock_ids
        )
        # 取回 selected_indices —— 生效的 convert_feature_layer_to_fixed_size 会按到 ego 的距离
        # 重排并截断 lane，route 三通道必须用同一 permutation 对齐（2026-07-28 事故修复）。
        vector_map, selected_indices = map_process(
            current_ego_state.rear_axle, coords, tl_data, config.MAP_FEATURES,
            config.MAX_ELEMENTS, config.MAX_POINTS, config.INTERPOLATION,
            return_selected_indices=True,
        )
        if self._route_mode == "three_channel":
            ordered_lane_ids = reorder_lane_ids(lane_ids, selected_indices.get("LANE", []))
            lane_route_channels = build_route_channels(
                ordered_lane_ids, vector_map["lanes"],
                route_lane_progress, config.LANE_NUM, config.MAP_RADIUS,
            )
        else:  # route_mode="none"：强制全零，复现无 route 的旧基线
            lane_route_channels = None

        # 5. EGO_HISTORY [20,2] 局部逐帧增量（决策D；复刻预处理 current_heading 参考系）
        ego_history_local = self._build_ego_history_local(
            ego_states, current_ego_state.rear_axle.heading
        )
        ego_future_placeholder = np.zeros((config.NUM_FUTURE_POSES, 2), dtype=np.float32)  # 推理不需 GT
        t_ref = past_timestamps[-1].item()

        feature_df = encoding_vectornet_features(
            agents_past, None, ego_past, ego_future_placeholder, vector_map,
            past_timestamps, t_ref, ego_history=ego_history_local,
            lane_route_channels=lane_route_channels,
        )

        # 6. 与训练同一图构建（gnn_x 均值池化 + DefaultBuilder 无向补反向 + cluster col9）
        graph_data, skip_reason = build_graph_data_from_feature_df(feature_df, self._adjacency_config)
        if skip_reason is not None:
            raise RuntimeError(
                f"[GNNTransPlanner] 在线图构建被跳过（{skip_reason}）：无有效 polyline，"
                f"ego@({ego_position.x:.1f},{ego_position.y:.1f})——失败即失败，不静默替换。"
            )
        return graph_data

    @staticmethod
    def _build_ego_history_local(ego_states: list, current_heading: float) -> torch.Tensor:
        """[20,2] 车辆局部逐帧增量，参考朝向=current_heading（预处理 L119-127 同约定）。

        全局→局部：local_dx = g_dx·cos_h + g_dy·sin_h；local_dy = -g_dx·sin_h + g_dy·cos_h。首帧 [0,0]。
        仿真早期 history 不足 20 帧时前补 [0,0]（模型 ego 分支须 [20,2]）。
        """
        cos_h, sin_h = np.cos(current_heading), np.sin(current_heading)
        inc = [[0.0, 0.0]]
        for i in range(1, len(ego_states)):
            g_dx = ego_states[i].rear_axle.x - ego_states[i - 1].rear_axle.x
            g_dy = ego_states[i].rear_axle.y - ego_states[i - 1].rear_axle.y
            inc.append([g_dx * cos_h + g_dy * sin_h, -g_dx * sin_h + g_dy * cos_h])
        arr = np.asarray(inc, dtype=np.float32)
        n_pad = 20 - arr.shape[0]
        if n_pad > 0:
            arr = np.concatenate([np.zeros((n_pad, 2), dtype=np.float32), arr], axis=0)
        return torch.from_numpy(arr[-20:])

    # ---- 模型增量 → 全局参考轨迹（不搓运动学/不夹速度；可行性交 two_stage_controller）----
    def _output_to_trajectory(
        self, model_output: torch.Tensor, current_state: EgoState
    ) -> InterpolatedTrajectory:
        inc = model_output.detach().cpu().numpy().reshape(-1, 2)  # [60,2] 局部逐帧增量（首帧≈[0,0]）
        local_pos = np.cumsum(inc, axis=0)                        # [60,2] 局部累计位移（X前/Y左）

        h = current_state.rear_axle.heading
        cos_h, sin_h = np.cos(h), np.sin(h)
        ox, oy = current_state.rear_axle.x, current_state.rear_axle.y
        # 局部→全局（预处理「全局→局部」的逆变换）
        gx = local_pos[:, 0] * cos_h - local_pos[:, 1] * sin_h + ox
        gy = local_pos[:, 0] * sin_h + local_pos[:, 1] * cos_h + oy

        # 训练原点=future[0]=t+0.1（increments[0]=[0,0]），故 cumsum[i] 是 t+0.1·(i+1) 相对 future[0] 的位置。
        # 在线锚定当前 ego：把 cumsum[i] 放在 t0+i·dt（cumsum[0]≈当前 ego），则首个未来点=cumsum[1]=真实约 0.9m 位移，
        # 消除「首点=[0,0]→控制器每步零速→ego 闭环停滞」滞后（cumsum[0] 放 t0+0.1 会令 progress~5%）。
        trajectory: List[EgoState] = [current_state]  # t0（≈ cumsum[0]）
        t0 = current_state.time_point.time_us
        dt_us = int(self._sampling_time * 1e6)
        veh_params = current_state.car_footprint.vehicle_parameters
        prev_x, prev_y, prev_h = ox, oy, h
        for i in range(1, self._num_output_points):  # cumsum[1..59] → t0+0.1 .. t0+5.9
            x, y = float(gx[i]), float(gy[i])
            dx, dy = x - prev_x, y - prev_y
            dist = float(np.hypot(dx, dy))
            heading = float(np.arctan2(dy, dx)) if dist > _HEADING_EPS else prev_h
            speed = dist / self._sampling_time  # 车体纵向速度 ≈ |Δ|/dt
            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(speed, 0.0),  # 车体系：纵向 speed，横向 0
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(t0 + i * dt_us),
                vehicle_parameters=veh_params,
                is_in_auto_mode=True,
                angular_vel=0.0,
                angular_accel=0.0,
            )
            trajectory.append(state)
            prev_x, prev_y, prev_h = x, y, heading
        return InterpolatedTrajectory(trajectory)
