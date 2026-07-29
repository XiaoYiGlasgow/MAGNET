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
import config as dm_config  # 模块 config；避免与函数内局部 dict `config` 同名遮蔽

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
    encoding_vectornet_features,
    get_route_lane_progress,
    build_route_channels,
    reorder_lane_ids
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
    # D0#3 修复 M2（2026-07-29）：与 `_silent` 版本及 planner 的 history buffer 同构 ——
    # 19 帧真实过去 + 当前帧，锚点落在 t0。
    print("1. 提取ego历史轨迹...")
    n_past = config['num_past_poses']
    past_dt = float(config['past_time_horizon']) / n_past
    ego_states = list(scenario.get_ego_past_trajectory(
        iteration=0,
        num_samples=n_past - 1,
        time_horizon=(n_past - 1) * past_dt
    )) + [scenario.initial_ego_state]

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

    ego_future_states_list = list(ego_future_states)

    if not ego_future_states_list:
        raise ValueError("未来轨迹为空")

    # D0#3 修复 M3（2026-07-29）：局部系参考位姿改用当前 ego（t0），与 planner 同系。
    ref_state = scenario.initial_ego_state
    current_x = ref_state.rear_axle.x
    current_y = ref_state.rear_axle.y
    current_heading = ref_state.rear_axle.heading

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

    # 决策D：EGO_HISTORY 改存 [20,2] 车辆局部坐标系逐帧增量
    # 与 ego_future 同约定、同参考朝向 current_heading（未来第0点=t=0 当前 ego）
    ego_history_increments = [[0.0, 0.0]]
    for i in range(1, len(ego_states)):
        g_dx = ego_states[i].rear_axle.x - ego_states[i-1].rear_axle.x
        g_dy = ego_states[i].rear_axle.y - ego_states[i-1].rear_axle.y
        ego_history_increments.append([
            g_dx * cos_h + g_dy * sin_h,     # local_dx（前向）
            -g_dx * sin_h + g_dy * cos_h,    # local_dy（左向）
        ])
    ego_history_local = torch.from_numpy(np.array(ego_history_increments, dtype=np.float32))  # [20,2]

    print(f"   ✅ Ego未来轨迹（车辆局部坐标系）: {ego_future.shape}")
    print(f"   📊 局部坐标范围: x[{ego_future[:, 0].min():.2f}, {ego_future[:, 0].max():.2f}], y[{ego_future[:, 1].min():.2f}, {ego_future[:, 1].max():.2f}]")
    print(f"   🧭 车辆朝向: {current_heading:.4f} rad ({np.degrees(current_heading):.1f}°)")

    # 3. 提取其他车辆轨迹
    # D0#3 修复 M1（2026-07-29，information leakage）：原 `get_tracked_objects_at_iteration(0..19)`
    # 取的是 t0…t0+0.95 s 的**未来**（database_interval=0.05）。改为真实过去 19 帧 + 当前帧。
    print("3. 提取其他车辆轨迹...")
    past_tracked_objects = list(scenario.get_past_tracked_objects(
        iteration=0,
        time_horizon=(n_past - 1) * past_dt,
        num_samples=n_past - 1
    )) + [scenario.initial_tracked_objects]

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

    # 获取地图特征（D1-2 route 契约：传真实 route_roadblock_ids + 取同序 lane_ids）
    route_roadblock_ids = scenario.get_route_roadblock_ids() or []
    coords, traffic_light_data, lane_ids = get_neighbor_vector_set_map(
        scenario.map_api,
        config['map_features'],
        ego_position,
        config['map_radius'],
        route_roadblock_ids,
        [],  # traffic_light_status_data，可以根据需要添加
        return_lane_ids=True
    )
    route_lane_progress = get_route_lane_progress(
        scenario.map_api, ego_position, config['map_radius'], route_roadblock_ids
    )

    # 处理地图数据（取回 selected_indices：lane 排序的唯一真相源）
    vector_map, selected_indices = map_process(
        current_ego_state.rear_axle,
        coords,
        traffic_light_data,
        config['map_features'],
        config['max_elements'],
        config['max_points'],
        config['interpolation_method'],
        return_selected_indices=True
    )
    # route 三通道必须在 map_process 之后、按其最终 lane 顺序构造
    if config.get('route_mode', 'none') == 'three_channel':
        ordered_lane_ids = reorder_lane_ids(lane_ids, selected_indices.get('LANE', []))
        lane_route_channels = build_route_channels(
            ordered_lane_ids, vector_map['lanes'],
            route_lane_progress, config['max_elements']['LANE'], config['map_radius']
        )
    else:
        lane_route_channels = None   # route_mode='none' -> col6/7/8 保持全 0

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
        ego_history=ego_history_local,  # [20,2] 局部增量（决策D）
        lane_route_channels=lane_route_channels  # route 三通道 -> lane col6/col7/col8
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
        # D0#3 修复 M2（2026-07-29）：与 planner 的 SimulationHistoryBuffer 同构 ——
        # 19 帧真实过去 + 当前帧，锚点 ego_states[-1] = t0（原实现为 20 帧纯过去，锚点落在
        # t0-0.1，导致地图查询点与在线相差约 1 m）。t_ref 亦随之对齐到 t0。
        n_past = config['num_past_poses']
        past_dt = float(config['past_time_horizon']) / n_past
        ego_states = list(scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=n_past - 1,
            time_horizon=(n_past - 1) * past_dt
        )) + [scenario.initial_ego_state]
        past_ego_states = sampled_past_ego_states_to_tensor(ego_states)
        past_timestamps = sampled_past_timestamps_to_tensor(ego_states)

        # 2. 提取ego未来轨迹
        ego_future_states = scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=config['num_future_poses'],
            time_horizon=config['future_time_horizon']
        )

        ego_future_states_list = list(ego_future_states)

        if not ego_future_states_list:
            raise ValueError("未来轨迹为空")

        # D0#3 修复 M3（2026-07-29）：局部系参考位姿改用**当前 ego**（t0），与 planner 的
        # `_output_to_trajectory(output, history.ego_states[-1])` 完全同系。
        # 原实现用 future[0]（t0+0.1）作参考朝向，与在线相差 1 帧。
        # 注意：increments 的时间口径不变（仍 increments[0]=[0,0]、cumsum[i] 覆盖 i·dt 的位移），
        # 与 planner 把 cumsum[i] 放在 t0+i·dt 的做法在**时长上自洽**；残留的 0.1 s 相位差
        # 见阶段⑰ M4，属两侧协同改动，本次不动。
        ref_state = scenario.initial_ego_state
        current_x = ref_state.rear_axle.x
        current_y = ref_state.rear_axle.y
        current_heading = ref_state.rear_axle.heading

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

        # 决策D：EGO_HISTORY 改存 [20,2] 车辆局部坐标系逐帧增量
        # 与 ego_future 同约定、同参考朝向 current_heading（未来第0点=t=0 当前 ego）
        ego_history_increments = [[0.0, 0.0]]
        for i in range(1, len(ego_states)):
            g_dx = ego_states[i].rear_axle.x - ego_states[i-1].rear_axle.x
            g_dy = ego_states[i].rear_axle.y - ego_states[i-1].rear_axle.y
            ego_history_increments.append([
                g_dx * cos_h + g_dy * sin_h,     # local_dx（前向）
                -g_dx * sin_h + g_dy * cos_h,    # local_dy（左向）
            ])
        ego_history_local = torch.from_numpy(np.array(ego_history_increments, dtype=np.float32))  # [20,2]

        # 3. 提取其他车辆轨迹
        # D0#3 修复 M1（2026-07-29，information leakage）：原实现用
        # `get_tracked_objects_at_iteration(i), i=0..19`，而 nuPlan 的 database_interval=0.05，
        # 实测 iteration 19 = t0+0.95 s —— 那是 agent 的**未来**、且采样率 20 Hz。
        # planner 在线只能给 t-1.9…t 的真实过去（10 Hz），两者窗口方向相反、时长差 2×，
        # 实测 11/12 样本连选中的 20 个 agent 都不是同一批。改为与 planner 同构的
        # 19 帧真实过去 + 当前帧。
        past_tracked_objects = list(scenario.get_past_tracked_objects(
            iteration=0,
            time_horizon=(n_past - 1) * past_dt,
            num_samples=n_past - 1
        )) + [scenario.initial_tracked_objects]

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

        # D1-2 route 契约：取真实 route_roadblock_ids + 与 coords['LANE'] 同序的 lane_ids
        route_roadblock_ids = scenario.get_route_roadblock_ids() or []
        coords, traffic_light_data, lane_ids = get_neighbor_vector_set_map(
            scenario.map_api,
            config['map_features'],
            ego_position,
            config['map_radius'],
            route_roadblock_ids,
            [],
            return_lane_ids=True
        )
        route_lane_progress = get_route_lane_progress(
            scenario.map_api, ego_position, config['map_radius'], route_roadblock_ids
        )

        vector_map, selected_indices = map_process(
            current_ego_state.rear_axle,
            coords,
            traffic_light_data,
            config['map_features'],
            config['max_elements'],
            config['max_points'],
            config['interpolation_method'],
            return_selected_indices=True
        )
        if config.get('route_mode', 'none') == 'three_channel':
            ordered_lane_ids = reorder_lane_ids(lane_ids, selected_indices.get('LANE', []))
            lane_route_channels = build_route_channels(
                ordered_lane_ids, vector_map['lanes'],
                route_lane_progress, config['max_elements']['LANE'], config['map_radius']
            )
        else:
            lane_route_channels = None   # route_mode='none' -> col6/7/8 保持全 0

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
            ego_history=ego_history_local,  # [20,2] 局部增量（决策D）
            lane_route_channels=lane_route_channels  # route 三通道 -> lane col6/col7/col8
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

    # E1(log-disjoint): 旁路 manifest，记录成功写盘样本的 token→log_name（不改 pkl schema）
    log_manifest = []

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

            # E1(log-disjoint): 写盘成功后才记录，保证 manifest 行与 pkl 文件一一对应
            log_manifest.append({"token": scenario_name, "log_name": scenario.log_name})

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

    # E1(log-disjoint): 批末写 token→log_name 旁路 manifest，供 split_log_disjoint.py 按 log 分组
    import json
    manifest_path = output_dir / "log_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(log_manifest, f, ensure_ascii=False, indent=2)
    print(f"   🗂  log manifest 写出: {manifest_path}（{len(log_manifest)} 条）")


def create_default_config():
    """创建默认配置参数（唯一源=config.py，决策C；地图 LANE-only，决策A）"""
    return {
        # 时间相关参数（读 config.py）
        'num_past_poses': dm_config.NUM_PAST_POSES,          # 20
        'num_future_poses': dm_config.NUM_FUTURE_POSES,      # 60
        'past_time_horizon': float(dm_config.PAST_TIME_HORIZON),     # 2.0
        'future_time_horizon': float(dm_config.FUTURE_TIME_HORIZON), # 6.0

        # 车辆相关参数
        'num_agents': dm_config.NUM_AGENTS,                  # 20

        # 地图相关参数（LANE-only，决策A）
        'map_features': dm_config.MAP_FEATURES,              # ["LANE"]
        'map_radius': float(dm_config.MAP_RADIUS),           # 50.0
        'max_elements': dm_config.MAX_ELEMENTS,              # {"LANE": 40}
        'max_points': dm_config.MAX_POINTS,                  # {"LANE": 50}
        'interpolation_method': dm_config.INTERPOLATION,     # "linear"

        # route 注入模式（阶段⑬）：'none'=三通道强制全零（复现无 route 旧基线）；
        # 'three_channel'=启用 proximity/progress/is_on_route。默认 none 保持旧行为。
        'route_mode': getattr(dm_config, 'ROUTE_MODE', 'none')
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
            output_dir = Path(dm_config.PROCESSED_DATA_DIR)  # 修局部 dict `config` 遮蔽致的 AttributeError
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
