#!/usr/bin/env python3

"""
nuPlan数据处理为12维特征向量
完整的数据处理流程
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import config

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

from GNN_Trans.utils.data_utils import (
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    agent_past_process,
    agent_future_process,
    get_neighbor_vector_set_map,
    map_process,
    encoding_vectornet_features
)

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import VectorFeatureLayer
from nuplan.common.actor_state.state_representation import Point2D


def process_nuplan_scenario_to_12d_features(scenario, config):
    """
    将单个nuPlan场景处理为12维特征向量

    Args:
        scenario: nuPlan场景对象
        config: 配置参数

    Returns:
        feature_df: 包含12维特征向量的DataFrame
    """

    print(f"处理场景: {scenario.scenario_name}")

    # 1. 提取ego历史轨迹
    print("1. 提取ego历史轨迹...")
    ego_states = scenario.get_ego_past_trajectory(
        iteration=0,
        num_samples=config['num_past_poses'],
        time_horizon=config['past_time_horizon']
    )

    # 将生成器转换为列表
    ego_states = list(ego_states)

    past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
    past_timestamps = sampled_past_timestamps_to_tensor(ego_states)

    print(f"   Ego历史轨迹: {past_ego_states.shape}")

    # 2. 提取ego未来轨迹（用于ground truth）
    print("2. 提取ego未来轨迹...")
    ego_future_states = scenario.get_ego_future_trajectory(
        iteration=0,
        num_samples=config['num_future_poses'],
        time_horizon=config['future_time_horizon']
    )

    # 获取当前ego位置作为参考点 - 使用未来轨迹的第一个点作为起始点
    ego_future_states_list = list(ego_future_states)

    if not ego_future_states_list:
        raise ValueError("未来轨迹为空")

    # 使用未来轨迹第一个点作为起始参考点
    first_future_state = ego_future_states_list[0]
    current_x = first_future_state.rear_axle.x
    current_y = first_future_state.rear_axle.y
    current_heading = first_future_state.rear_axle.heading

    # 保存ego的绝对位置，在坐标转换之前
    ego_last_absolute_position = np.array([current_x, current_y], dtype=np.float32)
    print(f"   保存ego绝对位置: [{ego_last_absolute_position[0]:.4f}, {ego_last_absolute_position[1]:.4f}]")

    # 修复坐标系：将全局坐标增量转换为车辆局部坐标增量
    # 与推理阶段保持一致：X轴指向车辆前方，Y轴指向车辆左侧
    cos_h = np.cos(current_heading)
    sin_h = np.sin(current_heading)

    ego_future_increments = []

    # 第0个增量为0 (保持在起始位置)
    ego_future_increments.append([0.0, 0.0])

    # 从第1个点开始计算增量并转换为车辆局部坐标系
    for i in range(1, len(ego_future_states_list)):
        prev_state = ego_future_states_list[i-1]
        curr_state = ego_future_states_list[i]

        # 计算全局坐标系下的增量
        global_dx = curr_state.rear_axle.x - prev_state.rear_axle.x
        global_dy = curr_state.rear_axle.y - prev_state.rear_axle.y

        # 转换为车辆局部坐标系（全局 → 局部的逆变换）
        # 局部坐标系：X轴指向车辆前方，Y轴指向车辆左侧
        local_dx = global_dx * cos_h + global_dy * sin_h
        local_dy = -global_dx * sin_h + global_dy * cos_h

        ego_future_increments.append([local_dx, local_dy])

    ego_future = np.array(ego_future_increments, dtype=np.float32)

    print(f"   ✅ Ego未来轨迹（车辆局部坐标系）: {ego_future.shape}")
    print(f"   📊 局部坐标范围: x[{ego_future[:, 0].min():.2f}, {ego_future[:, 0].max():.2f}], y[{ego_future[:, 1].min():.2f}, {ego_future[:, 1].max():.2f}]")
    print(f"   🧭 车辆朝向: {current_heading:.4f} rad ({np.degrees(current_heading):.1f}°)")

    # 3. 提取其他车辆轨迹
    print("3. 提取其他车辆轨迹...")
    past_tracked_objects = []
    for i in range(config['num_past_poses']):
        tracked_objects = scenario.get_tracked_objects_at_iteration(i)
        past_tracked_objects.append(tracked_objects)

    tracked_objects_tensor_list, tracked_objects_types = sampled_tracked_objects_to_tensor_list(
        past_tracked_objects
    )

    # 处理agent数据
    ego_past, agents_past, agent_indices = agent_past_process(
        past_ego_states,
        past_timestamps,
        tracked_objects_tensor_list,
        tracked_objects_types,
        config['num_agents']
    )

    print(f"   ✅ 其他车辆轨迹: {len(agents_past)}个车辆")

    # 4. 提取其他车辆未来轨迹
    print("4. 提取其他车辆未来轨迹...")
    present_tracked_objects = scenario.initial_tracked_objects
    future_tracked_objects = [
        tracked_objects
        for tracked_objects in scenario.get_future_tracked_objects(
            iteration=0,
            time_horizon=config['future_time_horizon'],
            num_samples=config['num_future_poses']
        )
    ]
    sampled_future_observations = [present_tracked_objects] + future_tracked_objects
    future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(
        sampled_future_observations
    )
    agents_future = agent_future_process(
        scenario.initial_ego_state,
        future_tracked_objects_tensor_list,
        config['num_agents'],
        agent_indices
    )

    # 5. 提取地图信息
    print("4. 提取地图信息...")
    current_ego_state = ego_states[-1]
    ego_position = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)

    # 获取地图特征
    coords, traffic_light_data = get_neighbor_vector_set_map(
        scenario.map_api,
        config['map_features'],
        ego_position,
        config['map_radius'],
        [],  # route_roadblock_ids，可以根据需要添加
        []   # traffic_light_status_data，可以根据需要添加
    )

    # 处理地图数据
    vector_map = map_process(
        current_ego_state.rear_axle,
        coords,
        traffic_light_data,
        config['map_features'],
        config['max_elements'],
        config['max_points'],
        config['interpolation_method']
    )

    print(f"   ✅ 地图信息: {len(vector_map['lanes'])}条车道")

    # 6. 生成12维特征向量
    print("5. 生成12维特征向量...")

    # 获取参考时间
    T_ref = past_timestamps[-1].item()

    # 调用编码函数生成12维特征
    feature_df = encoding_vectornet_features(
        agents_past,
        agents_future,
        ego_past,
        ego_future,
        vector_map,
        past_timestamps,
        T_ref,
        ego_history=past_ego_states  # 传递ego历史轨迹数据
    )

    # ✅ 关键修复：手动设置正确的ego绝对位置
    feature_df.at[0, 'EGO_LAST_POS'] = ego_last_absolute_position.tolist()
    print(f"   手动设置EGO_LAST_POS: [{ego_last_absolute_position[0]:.4f}, {ego_last_absolute_position[1]:.4f}]")

    print(f"   12维特征向量生成完成")
    print(f"   特征矩阵形状: {feature_df.iloc[0]['POLYLINE_FEATURES'].shape}")

    return feature_df


def process_nuplan_scenario_to_12d_features_silent(scenario, config):
    """
    静默处理单个nuPlan场景为12维特征向量（完全无输出）
    """
    import sys
    import os
    from io import StringIO

    # 重定向所有输出到空
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        # 1. 提取ego历史轨迹
        ego_states = scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=config['num_past_poses'],
            time_horizon=config['past_time_horizon']
        )
        ego_states = list(ego_states)
        past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
        past_timestamps = sampled_past_timestamps_to_tensor(ego_states)

        # 2. 提取ego未来轨迹
        ego_future_states = scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=config['num_future_poses'],
            time_horizon=config['future_time_horizon']
        )

        # 获取当前ego位置作为参考点 - 使用未来轨迹的第一个点作为起始点
        ego_future_states_list = list(ego_future_states)

        if not ego_future_states_list:
            raise ValueError("未来轨迹为空")

        # 使用未来轨迹第一个点作为起始参考点
        first_future_state = ego_future_states_list[0]
        current_x = first_future_state.rear_axle.x
        current_y = first_future_state.rear_axle.y
        current_heading = first_future_state.rear_axle.heading

        # 保存ego的绝对位置，在坐标转换之前
        ego_last_absolute_position = np.array([current_x, current_y], dtype=np.float32)

        # 修复坐标系：将全局坐标增量转换为车辆局部坐标增量
        # 与推理阶段保持一致：X轴指向车辆前方，Y轴指向车辆左侧
        cos_h = np.cos(current_heading)
        sin_h = np.sin(current_heading)

        ego_future_increments = []

        # 第0个增量为0 (保持在起始位置)
        ego_future_increments.append([0.0, 0.0])

        # 从第1个点开始计算增量并转换为车辆局部坐标系
        for i in range(1, len(ego_future_states_list)):
            prev_state = ego_future_states_list[i-1]
            curr_state = ego_future_states_list[i]

            # 计算全局坐标系下的增量
            global_dx = curr_state.rear_axle.x - prev_state.rear_axle.x
            global_dy = curr_state.rear_axle.y - prev_state.rear_axle.y

            # 转换为车辆局部坐标系（全局 → 局部的逆变换）
            # 局部坐标系：X轴指向车辆前方，Y轴指向车辆左侧
            local_dx = global_dx * cos_h + global_dy * sin_h
            local_dy = -global_dx * sin_h + global_dy * cos_h

            ego_future_increments.append([local_dx, local_dy])

        ego_future = np.array(ego_future_increments, dtype=np.float32)

        # 3. 提取其他车辆轨迹
        past_tracked_objects = []
        for i in range(config['num_past_poses']):
            tracked_objects = scenario.get_tracked_objects_at_iteration(i)
            past_tracked_objects.append(tracked_objects)

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

        # 4. 提取其他车辆未来轨迹
        present_tracked_objects = scenario.initial_tracked_objects
        future_tracked_objects = [
            tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=config['future_time_horizon'],
                num_samples=config['num_future_poses']
            )
        ]
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(
            sampled_future_observations
        )
        agents_future = agent_future_process(
            scenario.initial_ego_state,
            future_tracked_objects_tensor_list,
            config['num_agents'],
            agent_indices
        )

        # 5. 提取地图信息
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

        # 6. 生成12维特征向量
        T_ref = past_timestamps[-1].item()
        feature_df = encoding_vectornet_features(
            agents_past,
            agents_future,
            ego_past,
            ego_future,
            vector_map,
            past_timestamps,
            T_ref,
            ego_history=past_ego_states  # 传递ego历史轨迹数据
        )

        # ✅ 关键修复：手动设置正确的ego绝对位置
        feature_df.at[0, 'EGO_LAST_POS'] = ego_last_absolute_position.tolist()

        return feature_df

    finally:
        # 恢复输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def batch_process_nuplan_data(scenario_list, config, output_dir):
    """
    批量处理多个nuPlan场景

    Args:
        scenario_list: nuPlan场景列表
        config: 配置参数
        output_dir: 输出目录
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_scenarios = len(scenario_list)
    print(f"🚀 开始批量处理 {total_scenarios} 个场景")
    print(f"📁 输出目录: {output_dir}")

    processed_count = 0
    failed_count = 0

    # 添加进度条和时间估算
    import time
    start_time = time.time()

    for i, scenario in enumerate(scenario_list):
        try:
            # 计算进度百分比
            progress = (i + 1) / total_scenarios * 100

            # 估算剩余时间
            if i > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_scenario = elapsed_time / i
                remaining_scenarios = total_scenarios - i
                estimated_remaining_time = avg_time_per_scenario * remaining_scenarios

                print(f"\n📊 处理进度: {i+1}/{total_scenarios} ({progress:.1f}%)")
                print(f"⏰ 已用时间: {elapsed_time/60:.1f}分钟")
                print(f"⏳ 预计剩余: {estimated_remaining_time/60:.1f}分钟")
                print(f"🎯 处理场景: {scenario.scenario_name}")
            else:
                print(f"\n📊 处理进度: {i+1}/{total_scenarios} ({progress:.1f}%)")
                print(f"🎯 处理场景: {scenario.scenario_name}")

            # 处理单个场景
            feature_df = process_nuplan_scenario_to_12d_features(scenario, config)

            # 保存结果
            scenario_name = scenario.scenario_name
            output_file = output_dir / f"features_{scenario_name}.pkl"
            feature_df.to_pickle(output_file)

            print(f"✅ 场景 {scenario_name} 处理完成")
            processed_count += 1

        except Exception as e:
            print(f"❌ 场景 {scenario.scenario_name} 处理失败: {e}")
            failed_count += 1
            continue

    # 最终统计
    total_time = time.time() - start_time
    print(f"\n📊 批量处理完成:")
    print(f"   ✅ 成功处理: {processed_count} 个场景")
    print(f"   ❌ 处理失败: {failed_count} 个场景")
    print(f"   ⏰ 总用时: {total_time/60:.1f}分钟")
    print(f"   📁 结果保存在: {output_dir}")


def create_default_config():
    """创建默认配置参数"""
    return {
        # 时间相关参数
        'num_past_poses': 20,      # 2秒历史，10Hz
        'num_future_poses': 60,    # 6秒未来，10Hz
        'past_time_horizon': 2.0,  # 秒
        'future_time_horizon': 6.0, # 秒

        # 车辆相关参数
        'num_agents': 20,          # 最多处理20个其他车辆

        # 地图相关参数
        'map_features': ["LANE", "CROSSWALK", "ROUTE_LANES"],
        'map_radius': 50.0,        # 地图提取半径(米)
        'max_elements': {
            "LANE": 40,
            "CROSSWALK": 20,
            "ROUTE_LANES": 20
        },
        'max_points': {
            "LANE": 50,
            "CROSSWALK": 30,
            "ROUTE_LANES": 50
        },
        'interpolation_method': 'linear'
    }


def demo_process_single_scenario():
    """演示处理单个场景"""

    print("🧪 演示：处理单个nuPlan场景为12维特征向量")

    # 导入nuPlan相关模块
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    try:
        # 创建场景构建器 - 使用mini数据集
        data_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini"
        scenario_builder = NuPlanScenarioBuilder(
            data_root=data_root,
            map_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps",
            sensor_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs",
            db_files=None,
            map_version="nuplan-maps-v1.0"
        )

        # 创建场景过滤器
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=1,  # 只处���1个场景用于演示
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False
        )
        
        # 使用Sequential执行器
        worker = Sequential()
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

        if scenarios:
            scenario = scenarios[0]  # 使用第一个场景
            print(f"使用场景: {scenario.scenario_name}")
            config = create_default_config()

            # 处理场景
            feature_df = process_nuplan_scenario_to_12d_features(scenario, config)

            # 分析结果
            polyline_features = feature_df.iloc[0]['POLYLINE_FEATURES']

            print(f"\n📊 处理结果分析:")
            print(f"   特征矩阵形状: {polyline_features.shape}")
            print(f"   12维特征包含:")
            print(f"     [0:2] x_start, y_start")
            print(f"     [2:4] x_end, y_end")
            print(f"     [4]   heading_start")
            print(f"     [5]   heading_end")
            print(f"     [6]   normalized_timestamp")
            print(f"     [7]   v_start")
            print(f"     [8]   v_end")
            print(f"     [9]   subgraph_ID")
            print(f"     [10]  start_time")
            print(f"     [11]  end_time")

            # 保存示例结果
            output_dir = Path(config.PROCESSED_DATA_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "demo_12d_features.pkl"
            feature_df.to_pickle(output_file)

            print(f"\n✅ 演示完成！结果保存到: {output_file}")
            return True

        else:
            print("❌ 未找到测试场景")
            return False

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='nuPlan数据处理为12维特征向量')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--batch', type=str, help='批量处理，指定输出目录')

    args = parser.parse_args()

    if args.demo:
        print("🎯 演示模式")
        success = demo_process_single_scenario()
        exit(0 if success else 1)
    elif args.batch:
        print("🎯 批量处理模式")
        output_dir = args.batch

        # 导入nuPlan相关模块
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

        try:
            # 创建场景构建器 - 使用完整数据集
            data_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
            scenario_builder = NuPlanScenarioBuilder(
                data_root=data_root,
                map_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps",
                sensor_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs",
                db_files=None,
                map_version="nuplan-maps-v1.0"
            )

            # 创建场景过滤器 - 先处理少量场景
            scenario_filter = ScenarioFilter(
                scenario_types=None,
                scenario_tokens=None,
                log_names=None,
                map_names=None,
                num_scenarios_per_type=None,
                limit_total_scenarios=10000,  # 只处理1万个场景
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=True  # 随机打乱
            )

            # 使用Sequential执行器
            worker = Sequential()
            scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

            config = create_default_config()

            # 批量处理所有场景
            batch_process_nuplan_data(scenarios, config, output_dir)

        except Exception as e:
            print(f"❌ 批量处理失败: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    else:
        print("""
🔧 nuPlan数据处理为12维特征向量

使用方法:
1. 运行演示:
   python process_nuplan_to_12d.py --demo

2. 批量处理:
   python process_nuplan_to_12d.py --batch /path/to/output

12维特征向量格式:
[x_start, y_start, x_end, y_end, heading_start, heading_end, 
 normalized_timestamp, v_start, v_end, subgraph_ID, start_time, end_time]
""")

        # 默认运行演示
        print("默认运行演示...")
        demo_process_single_scenario()
