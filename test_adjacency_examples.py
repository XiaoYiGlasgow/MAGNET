#!/usr/bin/env python3
"""
邻接矩阵自定义示例
演示如何控制不同交通元素之间的连接关系
"""

import sys
import os
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

from utils.gnn_module import AdjacencyMatrixBuilder
import numpy as np

def example_1_no_ego_road_connection():
    """示例1: 禁止ego与道路相关元素连接"""
    print("示例1: 禁止ego与道路相关元素连接")

    builder = AdjacencyMatrixBuilder()

    # 禁止ego与车道线连接
    builder.set_connection_rule('ego_to_lane', False)
    # 禁止ego与其他道路元素连接
    builder.set_connection_rule('ego_to_road_elements', False)
    # 但允许ego与其他车辆连接
    builder.set_connection_rule('ego_to_vehicle', True)

    print("配置:")
    print("- ego -> 车辆: ✓")
    print("- ego -> 车道线: ✗")
    print("- ego -> 道路元素: ✗")

    return builder

def example_2_exclude_specific_vehicles():
    """示例2: 排除特定车辆不与ego连接"""
    print("\n示例2: 排除特定车辆")

    builder = AdjacencyMatrixBuilder()

    # 假设subgraph_id=5和7是干扰车辆，不希望它们影响ego
    builder.exclude_subgraph(5)
    builder.exclude_subgraph(7)

    print("配置:")
    print("- 排除subgraph_id=5的车辆")
    print("- 排除subgraph_id=7的车辆")
    print("- 这些车辆不会与任何其他元素连接")

    return builder

def example_3_only_nearby_vehicles():
    """示例3: 只与近距离车辆连接"""
    print("\n示例3: 只与近距离车辆连接")

    builder = AdjacencyMatrixBuilder()

    # 设置较小的距离阈值，只连接近距离车辆
    builder.set_connection_rule('distance_threshold', 10.0)  # 10米内

    # 禁止车辆之间的连接，只保留ego-vehicle连接
    builder.set_connection_rule('vehicle_to_vehicle', False)
    builder.set_connection_rule('vehicle_to_lane', False)

    print("配置:")
    print("- 距离阈值: 10米")
    print("- 车辆间连接: ✗")
    print("- 车辆-车道连接: ✗")
    print("- 只保留ego与近距离车辆的连接")

    return builder

def example_4_selective_connection():
    """示例4: 选择性连接特定车辆"""
    print("\n示例4: 选择性连接")

    builder = AdjacencyMatrixBuilder()

    # 只允许特定的subgraph之间连接
    # 假设0是ego，1,2,3是重要车辆，其他车辆被忽略
    builder.set_only_connect_subgraphs([0, 1, 2, 3])

    print("配置:")
    print("- 只允许subgraph_id=[0,1,2,3]之间连接")
    print("- 其他车辆完全被忽略")

    return builder

def example_5_ego_centric():
    """示例5: 以ego为中心的连接模式"""
    print("\n示例5: ego为中心的连接模式")

    builder = AdjacencyMatrixBuilder()

    # ego可以连接到所有类型
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', True)
    builder.set_connection_rule('ego_to_road_elements', True)

    # 但其他元素之间不连接
    builder.set_connection_rule('vehicle_to_vehicle', False)
    builder.set_connection_rule('vehicle_to_lane', False)
    builder.set_connection_rule('lane_to_lane', False)

    print("配置:")
    print("- ego连接所有元素: ✓")
    print("- 其他元素间连接: ✗")
    print("- 形成以ego为中心的星型连接")

    return builder

def test_adjacency_matrix():
    """测试邻接矩阵构建"""
    print("\n" + "="*50)
    print("测试邻接矩阵构建")
    print("="*50)

    # 创建测试数据
    subgraph_vectors = {
        0: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # ego
        1: np.array([5.0, 2.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 车辆1
        2: np.array([3.0, -1.0, 0.0, 0.9, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), # 车辆2
        3: np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 车道线
        4: np.array([50.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 远距离车辆
    }

    subgraph_types = {
        0: 0,  # EGO
        1: 1,  # VEHICLE
        2: 1,  # VEHICLE
        3: 2,  # LANE
        4: 1,  # VEHICLE
    }

    # 测试不同配置
    configs = [
        ("默认配置", AdjacencyMatrixBuilder()),
        ("禁止ego-道路连接", example_1_no_ego_road_connection()),
        ("排除车辆2", example_2_exclude_specific_vehicles()),
        ("只连接近距离", example_3_only_nearby_vehicles()),
        ("ego为中心", example_5_ego_centric()),
    ]

    for config_name, builder in configs:
        print(f"\n【{config_name}】")
        result = builder.build_adjacency_matrix(subgraph_vectors, subgraph_types)
        print(f"节点: {result['nodes']}")
        print(f"边: {result['edges']}")
        print(f"边类型: {result['edge_types']}")

if __name__ == "__main__":
    # 运行所有示例
    example_1_no_ego_road_connection()
    example_2_exclude_specific_vehicles()
    example_3_only_nearby_vehicles()
    example_4_selective_connection()
    example_5_ego_centric()

    # 测试邻接矩阵构建
    test_adjacency_matrix()
