#!/usr/bin/env python3
"""
简单测试脚本 - 验证基本导入和数据加载
"""

import sys
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')

print("🔍 开始测试...")

try:
    print("1. 测试基本导入...")
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    print("   ✅ 基本导入成功")

    print("2. 测试数据路径...")
    import os
    data_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini"
    print(f"   数据路径存在: {os.path.exists(data_root)}")

    print("3. 测试场景构建器创建...")
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps",
        sensor_root="/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs",
        db_files=None,
        map_version="nuplan-maps-v1.0"
    )
    print("   ✅ 场景构建器创建成功")

    print("4. 测试场景加载...")
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=1,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False
    )

    worker = Sequential()
    print("   正在加载场景...")
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    print(f"   ✅ 找到 {len(scenarios)} 个场景")

    if scenarios:
        scenario = scenarios[0]
        print(f"   第一个场景: {scenario.scenario_name}")
        print("   ✅ 基本测试完成！")
    else:
        print("   ❌ 未找到场景")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
