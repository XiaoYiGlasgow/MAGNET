import os
import sys
import torch
import numpy as np
from typing import Type

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import config
from models.transformer_predictor import TransformerPredictor

# 添加nuplan项目路径以便导入data_utils
nuplan_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(nuplan_root)

try:
    from GNN_Trans.utils.data_utils import (
        sampled_past_ego_states_to_tensor,
        sampled_past_timestamps_to_tensor,
        sampled_tracked_objects_to_tensor_list,
        agent_past_process,
        get_neighbor_vector_set_map,
        map_process,
        encoding_vectornet_features
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import data_utils functions: {e}")
    # 提供备用导入或跳过复杂数据处理

# nuplan imports
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint, StateSE2


class GraphData:
    """模拟torch_geometric的Data类结构"""
    def __init__(self, x=None, edge_index=None, gnn_x=None, gnn_edge_index=None, y=None):
        self.x = x
        self.edge_index = edge_index
        self.gnn_x = gnn_x
        self.gnn_edge_index = gnn_edge_index
        self.y = y


class GNNTransformerPlanner(AbstractPlanner):
    """
    GNN + Transformer based trajectory planner.
    """
    
    def __init__(self, model_path: str = None, **kwargs):
        """
        初始化规划器
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置默认配置参数（在initialize中可能会被覆盖）
        self.num_past_poses = config.NUM_PAST_POSES  # 40 (4秒，10Hz)
        self.num_future_poses = config.NUM_FUTURE_POSES  # 30 (3秒，10Hz)
        self.sampling_frequency = 10.0  # Hz

        # 地图特征配置
        self.map_features = ["LANE", "CROSSWALK", "ROUTE_LANES"]
        self.max_elements = {
            "LANE": 40,
            "CROSSWALK": 20,
            "ROUTE_LANES": 20
        }
        self.max_points = {
            "LANE": 50,
            "CROSSWALK": 30,
            "ROUTE_LANES": 50
        }
        self.interpolation_method = "linear"
        self.map_radius = 50.0  # 地图提取半径(米)

        # 加载训练好的模型
        self.model = TransformerPredictor().to(self.device)
        if model_path is None:
            model_path = os.path.join(config.WEIGHT_PATH, 'best_model.pth')
            
        if os.path.exists(model_path):
            print(f"✅ Loading model from: {model_path}")
            try:
                if self.device.type == 'cuda':
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                print(f"✅ Model loaded successfully on {self.device}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
        else:
            print(f"⚠️ Model file not found: {model_path}, using untrained model")

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        初始化规划器
        """
        self.map_api = initialization.map_api
        self.goal = initialization.mission_goal

        # 配置参数
        self.num_past_poses = config.NUM_PAST_POSES  # 40 (4秒，10Hz)
        self.num_future_poses = config.NUM_FUTURE_POSES  # 30 (3秒，10Hz)
        self.sampling_frequency = 10.0  # Hz

        # 地图特征配置
        self.map_features = ["LANE", "CROSSWALK", "ROUTE_LANES"]
        self.max_elements = {
            "LANE": 40,
            "CROSSWALK": 20,
            "ROUTE_LANES": 20
        }
        self.max_points = {
            "LANE": 50,
            "CROSSWALK": 30,
            "ROUTE_LANES": 50
        }
        self.interpolation_method = "linear"
        self.map_radius = 50.0  # 地图提取半径(米)

        print(f"✅ GNNTransformerPlanner initialized on {self.device}")

    def name(self) -> str:
        """返回规划器名称."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """返回需要的观测类型."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        计算规划轨迹的主要方法
        """
        try:
            # 1. 构造模型输入
            model_input = self._construct_model_input(current_input)
            
            # 2. 模型推理
            with torch.no_grad():
                model_output = self.model(model_input)
            
            # 3. 转换输出为轨迹
            trajectory = self._convert_model_output_to_trajectory(
                model_output, current_input.history.ego_states[-1]
            )

            return trajectory
            
        except Exception as e:
            print(f"❌ Error in compute_planner_trajectory: {e}")
            import traceback
            traceback.print_exc()

            # 返回备用轨迹
            return self._create_fallback_trajectory(current_input.history.ego_states[-1])

    def _construct_model_input(self, current_input: PlannerInput) -> GraphData:
        """
        从nuPlan输入构造模型需要的GraphData格式
        """
        try:
            # 简化版本：创建基本的图数据结构用于测试
            # 这里先实现一个简单版本，确保代码可以运行

            # 创建虚拟的图数据（实际应用中需要从真实数据提取）
            num_nodes = 100  # 假设有100个节点

            # 创建节点特征 [num_nodes, 9] for Transformer
            x = torch.randn(num_nodes, 9).to(self.device)

            # 创建GNN节点特征 [num_nodes, 12]
            gnn_x = torch.randn(num_nodes, 12).to(self.device)

            # 创建简单的边连接
            edge_list = []
            for i in range(min(num_nodes-1, 50)):  # 限制边的数量
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])

            if edge_list:
                edge_index = torch.LongTensor(edge_list).t().to(self.device)
                gnn_edge_index = edge_index.clone()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
                gnn_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)

            # 创建占位符y
            y = torch.zeros(1, self.num_future_poses * 2).to(self.device)  # [1, 60]

            print(f"✅ Created model input: x.shape={x.shape}, edge_index.shape={edge_index.shape}")

            return GraphData(
                x=x,
                edge_index=edge_index,
                gnn_x=gnn_x,
                gnn_edge_index=gnn_edge_index,
                y=y
            )

        except Exception as e:
            print(f"❌ Error in _construct_model_input: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _convert_model_output_to_trajectory(self, model_output: torch.Tensor,
                                          current_ego_state: EgoState) -> AbstractTrajectory:
        """
        将模型输出转换为nuPlan轨迹格式
        """
        try:
            # model_output shape: [batch_size, OUT_CHANNELS] = [1, 60]
            # 60 = NUM_FUTURE_POSES * 2 = 30 * 2 (x, y coordinates for 30 time steps)

            if model_output.dim() == 2:
                model_output = model_output[0]  # 移除batch维度: [60]

            # 重塑为 [30, 2] 格式
            predicted_positions = model_output.reshape(self.num_future_poses, 2)  # [30, 2]
            predicted_positions = predicted_positions.cpu().numpy()

            # 当前ego状态
            current_position = np.array([
                current_ego_state.rear_axle.x,
                current_ego_state.rear_axle.y
            ])
            current_heading = current_ego_state.rear_axle.heading

            # 创建轨迹点
            ego_states = []
            dt = 1.0 / self.sampling_frequency  # 0.1 seconds

            # 将预测的相对位移转换为绝对位置
            # 假设模型输出的是相对于当前位置的增量
            absolute_positions = np.zeros_like(predicted_positions)
            absolute_positions[0] = current_position + predicted_positions[0]

            for i in range(1, self.num_future_poses):
                absolute_positions[i] = absolute_positions[i-1] + predicted_positions[i]

            for i in range(self.num_future_poses):
                # 时间点
                time_point = TimePoint(
                    int((current_ego_state.time_point.time_us + (i + 1) * dt * 1e6))
                )
                
                # 计算航向角（基于相邻点的方向）
                if i == 0:
                    # 第一个点：基于当前状态和预测点的方向
                    dx = absolute_positions[i, 0] - current_position[0]
                    dy = absolute_positions[i, 1] - current_position[1]
                else:
                    # 后续点：基于相邻预测点的方向
                    dx = absolute_positions[i, 0] - absolute_positions[i-1, 0]
                    dy = absolute_positions[i, 1] - absolute_positions[i-1, 1]

                if np.sqrt(dx**2 + dy**2) > 1e-6:  # 避免除零
                    heading = np.arctan2(dy, dx)
                else:
                    heading = current_heading if i == 0 else ego_states[-1].rear_axle.heading

                # 创建EgoState
                ego_state = EgoState(
                    car_footprint=current_ego_state.car_footprint,
                    dynamic_car_state=current_ego_state.dynamic_car_state,
                    tire_steering_angle=current_ego_state.tire_steering_angle,
                    is_in_auto_mode=current_ego_state.is_in_auto_mode,
                    time_point=time_point,
                    rear_axle=StateSE2(
                        x=float(absolute_positions[i, 0]),
                        y=float(absolute_positions[i, 1]),
                        heading=float(heading)
                    )
                )

                ego_states.append(ego_state)

            # 创建插值轨迹
            trajectory = InterpolatedTrajectory(ego_states)

            print(f"✅ Generated trajectory with {len(ego_states)} states")
            return trajectory
            
        except Exception as e:
            print(f"❌ Error in _convert_model_output_to_trajectory: {e}")
            import traceback
            traceback.print_exc()

            # 返回备用轨迹
            return self._create_fallback_trajectory(current_ego_state)

    def _create_fallback_trajectory(self, current_ego_state: EgoState) -> AbstractTrajectory:
        """
        创建备用轨迹（直线前进）
        """
        try:
            ego_states = []
            dt = 1.0 / self.sampling_frequency  # 0.1 seconds

            # 获取当前状态
            current_velocity = np.sqrt(
                current_ego_state.dynamic_car_state.rear_axle_velocity_2d.x**2 +
                current_ego_state.dynamic_car_state.rear_axle_velocity_2d.y**2
            )

            # 如果速度太小，设置最小前进速度
            if current_velocity < 1.0:
                current_velocity = 5.0  # 5 m/s

            # 创建直线前进轨迹
            for i in range(self.num_future_poses):
                time_point = TimePoint(
                    int((current_ego_state.time_point.time_us + (i + 1) * dt * 1e6))
                )

                # 沿当前航向角前进
                distance = current_velocity * (i + 1) * dt
                new_x = current_ego_state.rear_axle.x + distance * np.cos(current_ego_state.rear_axle.heading)
                new_y = current_ego_state.rear_axle.y + distance * np.sin(current_ego_state.rear_axle.heading)

                ego_state = EgoState(
                    car_footprint=current_ego_state.car_footprint,
                    dynamic_car_state=current_ego_state.dynamic_car_state,
                    tire_steering_angle=current_ego_state.tire_steering_angle,
                    is_in_auto_mode=current_ego_state.is_in_auto_mode,
                    time_point=time_point,
                    rear_axle=StateSE2(
                        x=float(new_x),
                        y=float(new_y),
                        heading=current_ego_state.rear_axle.heading
                    )
                )
                ego_states.append(ego_state)

            trajectory = InterpolatedTrajectory(ego_states)
            print(f"⚠️ Using fallback trajectory with {len(ego_states)} states")
            return trajectory

        except Exception as e:
            print(f"❌ Error creating fallback trajectory: {e}")
            raise e
