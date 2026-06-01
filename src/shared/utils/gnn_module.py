# gnn_module.py - 完整版，支持自定义邻接矩阵构建
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

class AdjacencyMatrixBuilder:
    """
    可自定义的邻接矩阵构建器
    支持灵活配置不同交通元素之间的连接关系
    """

    def __init__(self):
        # 定义交通元素类型
        self.ELEMENT_TYPES = {
            'EGO': 0,           # 自车
            'VEHICLE': 1,       # 其他车辆
            'LANE': 2,          # 车道线
            'CROSSWALK': 3,     # 人行横道
            'TRAFFIC_LIGHT': 4, # 交通信号灯
            'STOP_SIGN': 5,     # 停止标志
            'ROAD_EDGE': 6      # 道路边界
        }

        # 默认连接规则
        self.connection_rules = {
            'ego_to_vehicle': True,      # ego是否连接到其他车辆
            'ego_to_lane': True,         # ego是否连接到车道线
            'ego_to_road_elements': True, # ego是否连接到道路元素
            'vehicle_to_vehicle': True,   # 车辆之间是否连接
            'vehicle_to_lane': True,     # 车辆是否连接到车道线
            'lane_to_lane': True,        # 车道线之间是否连接
            'distance_threshold': 30.0,   # 距离阈值(米)
            'angle_threshold': 45.0,      # 角度阈值(度)
        }

        # 排除列表：特定的subgraph ID不参与连接
        self.excluded_subgraphs = set()

        # 仅连接列表：只有这些subgraph ID之间可以连接
        self.only_connect_subgraphs = None  # None表示不限制

    def set_connection_rule(self, rule_name: str, value):
        """设置连接规则"""
        if rule_name in self.connection_rules:
            self.connection_rules[rule_name] = value
        else:
            print(f"Warning: Unknown rule '{rule_name}'")

    def exclude_subgraph(self, subgraph_id: int):
        """排除特定subgraph不参与任何连接"""
        self.excluded_subgraphs.add(subgraph_id)

    def set_only_connect_subgraphs(self, subgraph_ids: List[int]):
        """设置只有指定的subgraph之间可以连接"""
        self.only_connect_subgraphs = set(subgraph_ids)

    def clear_restrictions(self):
        """清除所有限制"""
        self.excluded_subgraphs.clear()
        self.only_connect_subgraphs = None

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两个位置之间的欧几里得距离"""
        return np.linalg.norm(pos1[:2] - pos2[:2])

    def _calculate_angle_diff(self, heading1: float, heading2: float) -> float:
        """计算两个朝向之间的角度差(度)"""
        diff = abs(heading1 - heading2)
        return min(diff, 360 - diff)

    def _should_connect(self,
                       id1: int, type1: int, pos1: np.ndarray,
                       id2: int, type2: int, pos2: np.ndarray) -> bool:
        """判断两个subgraph是否应该连接"""

        # 检查排除列表
        if id1 in self.excluded_subgraphs or id2 in self.excluded_subgraphs:
            return False

        # 检查仅连接列表
        if self.only_connect_subgraphs is not None:
            if id1 not in self.only_connect_subgraphs or id2 not in self.only_connect_subgraphs:
                return False

        # 计算距离
        distance = self._calculate_distance(pos1, pos2)
        if distance > self.connection_rules['distance_threshold']:
            return False

        # 根据类型组合判断是否连接
        ego_type = self.ELEMENT_TYPES['EGO']
        vehicle_type = self.ELEMENT_TYPES['VEHICLE']
        lane_type = self.ELEMENT_TYPES['LANE']

        # ego到车辆的连接
        if (type1 == ego_type and type2 == vehicle_type) or (type1 == vehicle_type and type2 == ego_type):
            return self.connection_rules['ego_to_vehicle']

        # ego到车道线的连接
        if (type1 == ego_type and type2 == lane_type) or (type1 == lane_type and type2 == ego_type):
            return self.connection_rules['ego_to_lane']

        # ego到其他道路元素的连接
        if type1 == ego_type or type2 == ego_type:
            return self.connection_rules['ego_to_road_elements']

        # 车辆之间的连接
        if type1 == vehicle_type and type2 == vehicle_type:
            return self.connection_rules['vehicle_to_vehicle']

        # 车辆到车道线的连接
        if (type1 == vehicle_type and type2 == lane_type) or (type1 == lane_type and type2 == vehicle_type):
            return self.connection_rules['vehicle_to_lane']

        # 车道线之间的连接
        if type1 == lane_type and type2 == lane_type:
            return self.connection_rules['lane_to_lane']

        # 默认连接其他类型
        return True

    def build_adjacency_matrix(self,
                             subgraph_vectors: Dict[int, np.ndarray],
                             subgraph_types: Dict[int, int]) -> Dict:
        """
        构建自定义的邻接矩阵

        Args:
            subgraph_vectors: {subgraph_id: pooled_vector}
            subgraph_types: {subgraph_id: element_type}

        Returns:
            dict: {'nodes': [...], 'edges': [...], 'edge_types': [...]}
        """
        nodes = list(subgraph_vectors.keys())
        edges = []
        edge_types = []

        # 构建连接
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:  # 避免重复连接和自连接
                    continue

                # 获取位置信息(假设前3维是x,y,z坐标)
                pos1 = subgraph_vectors[node1][:3]
                pos2 = subgraph_vectors[node2][:3]

                # 获取类型信息
                type1 = subgraph_types.get(node1, 1)  # 默认为车辆类型
                type2 = subgraph_types.get(node2, 1)

                # 判断是否应该连接
                if self._should_connect(node1, type1, pos1, node2, type2, pos2):
                    edges.append((node1, node2))
                    # 根据连接类型设置边类型
                    edge_type = self._get_edge_type(type1, type2)
                    edge_types.append(edge_type)

        return {
            'nodes': nodes,
            'edges': edges,
            'edge_types': edge_types
        }

    def _get_edge_type(self, type1: int, type2: int) -> int:
        """根据连接的两个节点类型确定边类型"""
        ego_type = self.ELEMENT_TYPES['EGO']
        vehicle_type = self.ELEMENT_TYPES['VEHICLE']
        lane_type = self.ELEMENT_TYPES['LANE']

        # 定义边类型
        if (type1 == ego_type and type2 == vehicle_type) or (type1 == vehicle_type and type2 == ego_type):
            return 0  # ego-vehicle连接
        elif (type1 == ego_type and type2 == lane_type) or (type1 == lane_type and type2 == ego_type):
            return 1  # ego-lane连接
        elif type1 == vehicle_type and type2 == vehicle_type:
            return 2  # vehicle-vehicle连接
        elif (type1 == vehicle_type and type2 == lane_type) or (type1 == lane_type and type2 == vehicle_type):
            return 3  # vehicle-lane连接
        elif type1 == lane_type and type2 == lane_type:
            return 4  # lane-lane连接
        else:
            return 5  # 其他类型连接


# 保持向后兼容的简化函数
def subgraph_pooling(polyline, subgraph_id_to_indices):
    """简化的子图池化"""
    pooled_vectors = {}
    for subgraph_id, (start, end) in subgraph_id_to_indices.items():
        if end >= start and start < len(polyline):
            segment = polyline[start:end+1]
            if len(segment) > 0:
                # 简单的平均池化
                pooled = np.mean(segment, axis=0)
                # 确保是12维
                if len(pooled) < 12:
                    pooled = np.pad(pooled, (0, 12 - len(pooled)))
                elif len(pooled) > 12:
                    pooled = pooled[:12]
                pooled_vectors[subgraph_id] = pooled
    return pooled_vectors


def build_graph_from_subgraphs(subgraph_vectors, subgraph_types, distance_threshold=30.0,
                               adjacency_builder=None, **kwargs):
    """
    构建图结构，支持自定义邻接矩阵

    Args:
        subgraph_vectors: 子图向量字典
        subgraph_types: 子图类型字典
        distance_threshold: 距离阈值
        adjacency_builder: 自定义的邻接矩阵构建器
        **kwargs: 其他参数
    """
    if adjacency_builder is None:
        # 使用默认的全连接构建器
        adjacency_builder = AdjacencyMatrixBuilder()
        adjacency_builder.set_connection_rule('distance_threshold', distance_threshold)

    return adjacency_builder.build_adjacency_matrix(subgraph_vectors, subgraph_types)


# 示例使用函数
def create_custom_adjacency_config():
    """创建自定义邻接矩阵配置的示例"""
    builder = AdjacencyMatrixBuilder()

    # 示例1：禁止ego与道路元素连接
    builder.set_connection_rule('ego_to_lane', False)
    builder.set_connection_rule('ego_to_road_elements', False)

    # 示例2：排除特定车辆(假设subgraph_id=5是某辆干扰车)
    builder.exclude_subgraph(5)

    # 示例3：设置更小的连接距离
    builder.set_connection_rule('distance_threshold', 15.0)

    return builder
