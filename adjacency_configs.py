#!/usr/bin/env python3
"""
邻接矩阵配置文件
在这里修改你想要的连接规则
"""

from utils.gnn_module import AdjacencyMatrixBuilder

def get_adjacency_config(config_name="default"):
    """
    获取不同的邻接矩阵配置

    Args:
        config_name: 配置名称，支持:
            - "default": 默认全连接
            - "ego_only": 只有ego连接其他元素
            - "no_ego_road": ego不连接道路元素
            - "nearby_only": 只连接近距离元素
            - "selective": 选择性连接特定元素
    """

    if config_name == "default":
        return get_default_config()
    elif config_name == "ego_only":
        return get_ego_only_config()
    elif config_name == "no_ego_road":
        return get_no_ego_road_config()
    elif config_name == "nearby_only":
        return get_nearby_only_config()
    elif config_name == "selective":
        return get_selective_config()
    else:
        print(f"Unknown config: {config_name}, using default")
        return get_default_config()

def get_default_config():
    """默认配置：全连接模式"""
    builder = AdjacencyMatrixBuilder()

    # 所有元素都可以连接
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', True)
    builder.set_connection_rule('ego_to_road_elements', True)
    builder.set_connection_rule('vehicle_to_vehicle', True)
    builder.set_connection_rule('vehicle_to_lane', True)
    builder.set_connection_rule('lane_to_lane', True)
    builder.set_connection_rule('distance_threshold', 30.0)

    return builder

def get_ego_only_config():
    """ego为中心配置：只有ego连接其他元素，其他元素间不连接"""
    builder = AdjacencyMatrixBuilder()

    # ego连接所有类型
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', True)
    builder.set_connection_rule('ego_to_road_elements', True)

    # 其他元素间不连接
    builder.set_connection_rule('vehicle_to_vehicle', False)
    builder.set_connection_rule('vehicle_to_lane', False)
    builder.set_connection_rule('lane_to_lane', False)

    builder.set_connection_rule('distance_threshold', 25.0)

    return builder

def get_no_ego_road_config():
    """禁止ego连接道路元素"""
    builder = AdjacencyMatrixBuilder()

    # ego只连接车辆，不连接道路
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', False)          # 🚫 禁止连接车道线
    builder.set_connection_rule('ego_to_road_elements', False) # 🚫 禁止连接道路元素

    # 其他连接保持默认
    builder.set_connection_rule('vehicle_to_vehicle', True)
    builder.set_connection_rule('vehicle_to_lane', True)
    builder.set_connection_rule('lane_to_lane', True)

    builder.set_connection_rule('distance_threshold', 20.0)

    return builder

def get_nearby_only_config():
    """只连接近距离元素"""
    builder = AdjacencyMatrixBuilder()

    # 使用较小的距离阈值
    builder.set_connection_rule('distance_threshold', 15.0)  # 🎯 只连接15米内的元素

    # 减少车辆间的连接
    builder.set_connection_rule('vehicle_to_vehicle', False)  # 🚫 车辆间不连接
    builder.set_connection_rule('lane_to_lane', False)        # 🚫 车道间不连接

    # 保持ego和vehicle-lane连接
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', True)
    builder.set_connection_rule('vehicle_to_lane', True)

    return builder

def get_selective_config():
    """选择性连接配置（示例）"""
    builder = AdjacencyMatrixBuilder()

    # 正常的连接规则
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', True)
    builder.set_connection_rule('vehicle_to_vehicle', True)
    builder.set_connection_rule('distance_threshold', 25.0)

    # 排除特定的干扰车辆（这里是示例，实际使用时需要根据具体情况修改）
    # builder.exclude_subgraph(5)  # 排除subgraph_id=5
    # builder.exclude_subgraph(8)  # 排除subgraph_id=8

    # 或者只连接特定的重要元素
    # builder.set_only_connect_subgraphs([0, 1, 2, 3])  # 只连接这几个subgraph

    return builder

def get_custom_config_for_experiments():
    """
    实验专用配置
    根据你的实验需求自定义这个函数
    """
    builder = AdjacencyMatrixBuilder()

    # 🔬 在这里添加你的实验配置
    # 例如：

    # 实验1：测试ego是否需要道路信息
    builder.set_connection_rule('ego_to_lane', False)
    builder.set_connection_rule('ego_to_road_elements', False)

    # 实验2：测试不同距离阈值的影响
    builder.set_connection_rule('distance_threshold', 10.0)

    # 实验3：测试车辆间交互的重要性
    builder.set_connection_rule('vehicle_to_vehicle', False)

    return builder

# 使用示例
if __name__ == "__main__":
    # 测试不同配置
    configs = {
        "default": get_default_config(),
        "ego_only": get_ego_only_config(),
        "no_ego_road": get_no_ego_road_config(),
        "nearby_only": get_nearby_only_config(),
        "selective": get_selective_config(),
    }

    print("可用的邻接矩阵配置:")
    for name, builder in configs.items():
        print(f"- {name}: {builder.connection_rules}")
