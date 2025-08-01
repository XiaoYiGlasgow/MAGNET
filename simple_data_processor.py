#!/usr/bin/env python3

"""
简化的数据处理脚本 - 将nuPlan数据转换为12维特征
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

def process_data_to_12d():
    """
    处理数据为12维特征的主函数
    """
    print("🚀 开始处理数据为12维特征向量...")

    try:
        # 导入必要的模块
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

        # 数据路径配置
        data_paths = [
            "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini",
            "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train",
            "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/val"
        ]

        # 寻找可用的数据路径
        available_data_root = None
        for path in data_paths:
            if os.path.exists(path):
                available_data_root = path
                print(f"✅ 找到数据路径: {path}")
                break

        if not available_data_root:
            print("❌ 未找到nuPlan数据，请检查数据路径")
            return False

        # 创建场景构建器
        scenario_builder = NuPlanScenarioBuilder(
            data_root=available_data_root,
            map_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps",
            sensor_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs",
            db_files=None,
            map_version="nuplan-maps-v1.0"
        )

        # 创建场景过滤器 - 限制处理数量
        scenario_filter = ScenarioFilter(
            scenario_types=None,  # 所有场景类型
            scenario_tokens=None,  # 所有场景token
            log_names=None,  # 所有日志
            map_names=None,  # 所有地图
            num_scenarios_per_type=None,  # 每种类型的场景数量
            limit_total_scenarios=5,  # 先处理5个场景进行测试
            timestamp_threshold_s=None,  # 时间戳阈值
            ego_displacement_minimum_m=None,  # ego最小位移
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False
        )

        # 获取场景
        worker = Sequential()
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

        print(f"📊 找到 {len(scenarios)} 个场景待处理")

        if not scenarios:
            print("❌ 未找到场景数据")
            return False

        # 处理配置
        config = {
            'num_past_poses': 40,
            'num_future_poses': 30,
            'past_time_horizon': 4.0,
            'future_time_horizon': 3.0,
            'num_agents': 20,
            'map_features': ["LANE", "CROSSWALK", "ROUTE_LANES"],
            'map_radius': 50.0,
            'max_elements': {"LANE": 40, "CROSSWALK": 20, "ROUTE_LANES": 20},
            'max_points': {"LANE": 50, "CROSSWALK": 30, "ROUTE_LANES": 50},
            'interpolation_method': 'linear'
        }

        # 输出目录
        output_dir = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理每个场景
        processed_count = 0
        for i, scenario in enumerate(scenarios):
            try:
                print(f"\n处理场景 {i+1}/{len(scenarios)}: {scenario.scenario_name}")

                # 导入处理函数
                from GNN_Trans.utils.data_utils import (
                    sampled_past_ego_states_to_tensor,
                    sampled_past_timestamps_to_tensor,
                    sampled_tracked_objects_to_tensor_list,
                    agent_past_process,
                    get_neighbor_vector_set_map,
                    map_process,
                    encoding_vectornet_features
                )

                # 提取ego轨迹
                ego_states = scenario.get_ego_past_trajectory(
                    iteration=0,
                    num_samples=config['num_past_poses'],
                    time_horizon=config['past_time_horizon']
                )

                ego_future_states = scenario.get_ego_future_trajectory(
                    iteration=0,
                    num_samples=config['num_future_poses'],
                    time_horizon=config['future_time_horizon']
                )

                past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
                past_timestamps = sampled_past_timestamps_to_tensor(ego_states)
                ego_future = np.array([[state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                                     for state in ego_future_states], dtype=np.float32)

                # 提取其他车辆
                past_tracked_objects = []
                for j in range(config['num_past_poses']):
                    tracked_objects = scenario.get_tracked_objects_at_iteration(j)
                    past_tracked_objects.append(tracked_objects)

                # 修复生成器问题 - 将生成器转换为列表
                if hasattr(past_tracked_objects[0], '__iter__') and not hasattr(past_tracked_objects[0], '__len__'):
                    past_tracked_objects = [list(obj) if hasattr(obj, '__iter__') else obj for obj in past_tracked_objects]

                tracked_objects_tensor_list, tracked_objects_types = sampled_tracked_objects_to_tensor_list(
                    past_tracked_objects
                )

                ego_past, agents_past, agent_indices = agent_past_process(
                    past_ego_states,
                    past_timestamps,
                    tracked_objects_tensor_list,
                    tracked_objects_types,
                    config['num_agents']
                )

                # 提取地图
                from nuplan.common.actor_state.state_representation import Point2D
                current_ego_state = ego_states[-1]
                ego_position = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)

                coords, traffic_light_data = get_neighbor_vector_set_map(
                    scenario.map_api,
                    config['map_features'],
                    ego_position,
                    config['map_radius'],
                    [],
                    []
                )

                vector_map = map_process(
                    current_ego_state.rear_axle,
                    coords,
                    traffic_light_data,
                    config['map_features'],
                    config['max_elements'],
                    config['max_points'],
                    config['interpolation_method']
                )

                # 生成12维特征
                T_ref = past_timestamps[-1].item()
                feature_df = encoding_vectornet_features(
                    agents_past,
                    None,
                    ego_past,
                    ego_future,
                    vector_map,
                    past_timestamps,
                    T_ref
                )

                # 保存结果
                output_file = output_dir / f"features_{scenario.scenario_name}.pkl"
                feature_df.to_pickle(output_file)

                print(f"✅ 已保存: {output_file}")
                processed_count += 1

            except Exception as e:
                print(f"❌ 场景处理失败: {e}")
                continue

        print(f"\n🎉 处理完成！成功处理 {processed_count} 个场景")
        print(f"📁 输出目录: {output_dir}")

        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保nuPlan环境正确安装")
        return False
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

if __name__ == "__main__":
    success = process_data_to_12d()
    exit(0 if success else 1)
