import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add nuplan imports for real-time data conversion
try:
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
except ImportError:
    print("Warning: nuPlan imports not available, some functions may not work")

# 导入项目配置
try:
    import config
except ImportError:
    class Config:
        EMBED_DIM = 256
        NUM_HEADS = 8
        FF_DIM = 512
        NUM_LAYERS = 6
        HIDDEN_DIM = 128
        OUT_CHANNELS = 60
    config = Config()

# 全局变量记录当前使用的邻接矩阵配置
CURRENT_ADJACENCY_CONFIG = "default"

# 内联定义所有需要的函数，避免导入错误
def subgraph_pooling(polyline, subgraph_id_to_indices):
    """简化的子图池化"""
    pooled_vectors = {}
    for subgraph_id, (start, end) in subgraph_id_to_indices.items():
        if end >= start and start < len(polyline):
            segment = polyline[start:end+1]
            if len(segment) > 0:
                pooled = np.mean(segment, axis=0)
                if len(pooled) < 12:
                    pooled = np.pad(pooled, (0, 12 - len(pooled)))
                elif len(pooled) > 12:
                    pooled = pooled[:12]
                pooled_vectors[subgraph_id] = pooled
    return pooled_vectors

def build_graph_from_subgraphs(subgraph_vectors, subgraph_types, distance_threshold=30.0, adjacency_builder=None, **kwargs):
    """构建图结构"""
    if adjacency_builder is not None:
        return adjacency_builder.build_adjacency_matrix(subgraph_vectors, subgraph_types)

    # 默认全连接图
    nodes = list(subgraph_vectors.keys())
    edges = []
    edge_types = []

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            edges.append((nodes[i], nodes[j]))
            edge_types.append(0)

    return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

def get_adjacency_config(config_name="default"):
    """获取邻接矩阵配置"""
    global CURRENT_ADJACENCY_CONFIG
    CURRENT_ADJACENCY_CONFIG = config_name

    class DefaultBuilder:
        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types):
            nodes = list(subgraph_vectors.keys())
            edges = []
            edge_types = []

            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    edges.append((nodes[i], nodes[j]))
                    edge_types.append(0)

            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    class EgoOnlyBuilder:
        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types):
            nodes = list(subgraph_vectors.keys())
            edges = []
            edge_types = []

            # ego节点通常是ID为0的节点
            ego_node = 0 if 0 in nodes else None

            if ego_node is not None:
                for node in nodes:
                    if node != ego_node:
                        edges.append((ego_node, node))
                        edge_types.append(0)

            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    class NoEgoRoadBuilder:
        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types):
            nodes = list(subgraph_vectors.keys())
            edges = []
            edge_types = []

            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    type_i = subgraph_types.get(node_i, 'unknown')
                    type_j = subgraph_types.get(node_j, 'unknown')

                    # Exclude ego-road connections, only connect agent-agent and agent-lane
                    if ((type_i == 'agent' and type_j == 'agent') or  # agent-agent
                        (type_i == 'agent' and type_j == 'lane') or   # agent-lane
                        (type_i == 'lane' and type_j == 'agent')):   # lane-agent
                        edges.append((node_i, node_j))
                        edge_types.append(0)

            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    class NearbyOnlyBuilder:
        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types, distance_threshold=30.0):
            # nearby_only 阈值=30m（E0 固定；E4 再扫 5/10/15/20/30m 做拓扑消融）
            nodes = list(subgraph_vectors.keys())
            edges = []
            edge_types = []

            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]

                    # Calculate distance between nodes
                    pos_i = np.array(subgraph_vectors[node_i][:2])
                    pos_j = np.array(subgraph_vectors[node_j][:2])
                    distance = np.linalg.norm(pos_i - pos_j)

                    if distance <= distance_threshold:
                        edges.append((node_i, node_j))
                        edge_types.append(0)

            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    class SelectiveBuilder:
        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types):
            nodes = list(subgraph_vectors.keys())
            edges = []
            edge_types = []

            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    type_i = subgraph_types.get(node_i, 'unknown')
                    type_j = subgraph_types.get(node_j, 'unknown')

                    # 选择性连接：vehicle-vehicle, vehicle-lane
                    # 支持字符串类型和数字类型
                    if isinstance(type_i, str):
                        # 字符串类型: 'ego', 'agent', 'lane'
                        vehicle_types = ['ego', 'agent']
                        lane_types = ['lane']

                        if ((type_i in vehicle_types and type_j in vehicle_types) or  # vehicle-vehicle
                            (type_i in vehicle_types and type_j in lane_types) or    # vehicle-lane
                            (type_i in lane_types and type_j in vehicle_types)):     # lane-vehicle
                            edges.append((node_i, node_j))
                            edge_types.append(0)
                    else:
                        # 数字类型: 0=vehicle, 1=lane, 2=crosswalk
                        if ((type_i == 0 and type_j == 0) or  # vehicle-vehicle
                            (type_i == 0 and type_j == 1) or  # vehicle-lane
                            (type_i == 1 and type_j == 0)):   # lane-vehicle
                            edges.append((node_i, node_j))
                            edge_types.append(0)

            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    class TypedRelationalBuilder:
        """关系型语义边构建器（#2 R-GCN 用）：拓扑与 DefaultBuilder 完全一致（全连接、无距离
        过滤），**仅**给每条边打真语义关系类型 —— 由节点 type('ego'/'agent'/'lane')决定，供子类
        RGCNConv 消费。GCN 基线忽略 edge_types、R-GCN 消费之，故 GCN-vs-RGCN 在同一拓扑上隔离出
        "关系型消息传递"净增益（RQ2）。父类 GCN 与 danger 线不读 edge_attr → 零风险。
        关系编码：ego-agent=0, ego-lane=1, agent-agent=2, agent-lane=3, lane-lane=4, other=5（num_relations=6）。"""
        _TYPE_ID = {'ego': 0, 'agent': 1, 'lane': 2}
        _PAIR_TO_REL = {(0, 1): 0, (0, 2): 1, (1, 1): 2, (1, 2): 3, (2, 2): 4, (0, 0): 5}

        def _rel(self, t1, t2):
            a, b = self._TYPE_ID.get(t1, 1), self._TYPE_ID.get(t2, 1)
            lo, hi = (a, b) if a <= b else (b, a)
            return self._PAIR_TO_REL.get((lo, hi), 5)

        def build_adjacency_matrix(self, subgraph_vectors, subgraph_types):
            nodes = list(subgraph_vectors.keys())
            edges, edge_types = [], []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    edges.append((node_i, node_j))
                    edge_types.append(self._rel(
                        subgraph_types.get(node_i, 'agent'),
                        subgraph_types.get(node_j, 'agent')))
            return {'nodes': nodes, 'edges': edges, 'edge_types': edge_types}

    # 根据配置返回相应的Builder
    if config_name == "default":
        return DefaultBuilder()
    elif config_name == "ego_only":
        return EgoOnlyBuilder()
    elif config_name == "no_ego_road":
        return NoEgoRoadBuilder()
    elif config_name == "nearby_only":
        return NearbyOnlyBuilder()
    elif config_name == "selective":
        return SelectiveBuilder()
    elif config_name == "typed_relational":
        return TypedRelationalBuilder()
    else:
        return DefaultBuilder()

def get_current_adjacency_config():
    """获取当前使用的邻接矩阵配置名称"""
    return CURRENT_ADJACENCY_CONFIG

def get_sequential_edge_index(num_nodes, start=0):
    """生成序列边索引，使用相对索引而不是绝对索引"""
    if num_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    # 使用相对索引：0, 1, 2, ... num_nodes-1
    edge_indices = np.array(
        [[i, i + 1] for i in range(num_nodes - 1)],
        dtype=np.int64
    ).T
    return edge_indices



# 自定义GraphData类，继承自PyTorch Geometric的Data类
class GraphData(Data):
    """自定义图数据类，添加ego_history属性支持"""
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None,
                 cluster=None, valid_len=None, time_step_len=None, traj_id_to_indices=None,
                 ego_last_pos=None, gnn_x=None, gnn_edge_index=None, gnn_edge_attr=None,
                 ego_history=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.cluster = cluster
        self.valid_len = valid_len
        self.time_step_len = time_step_len
        self.traj_id_to_indices = traj_id_to_indices
        self.ego_last_pos = ego_last_pos
        self.gnn_x = gnn_x
        self.gnn_edge_index = gnn_edge_index
        self.gnn_edge_attr = gnn_edge_attr
        self.ego_history = ego_history  # 确保ego_history属性被正确定义

    def __inc__(self, key: str, value, *args, **kwargs):
        """定义增量策略，用于批处理"""
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'gnn_edge_index':
            return self.gnn_x.size(0) if self.gnn_x is not None else 0
        elif key == 'cluster':
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """定义拼接维度策略"""
        if key in ['ego_last_pos', 'ego_history']:
            return 0  # 沿着batch维度拼接
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def to(self, device, **kwargs):
        """重写to方法，确保自定义属性也被正确转移"""
        # 先调用父类的to方法
        result = super().to(device, **kwargs)

        # 确保自定义属性也被转移到正确设备
        custom_attrs = ['cluster', 'valid_len', 'time_step_len', 'traj_id_to_indices',
                       'ego_last_pos', 'gnn_x', 'gnn_edge_index', 'gnn_edge_attr', 'ego_history']

        for attr_name in custom_attrs:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(result, attr_name, attr_value.to(device, **kwargs))
                else:
                    setattr(result, attr_name, attr_value)

        return result

    def cuda(self, device=None):
        """重写cuda方法"""
        return self.to(torch.device('cuda' if device is None else f'cuda:{device}'))

    def cpu(self):
        """重写cpu方法"""
        return self.to(torch.device('cpu'))

    def keys(self):
        """Return all available keys including custom ones."""
        keys = list(super().keys())
        custom_keys = []
        for key in ['cluster', 'ego_history', 'ego_last_pos', 'gnn_x', 'gnn_edge_index', 'gnn_edge_attr', 'traj_id_to_indices', 'valid_len', 'time_step_len']:
            if hasattr(self, key) and getattr(self, key) is not None:
                custom_keys.append(key)
        return keys + custom_keys


def build_graph_data_from_feature_df(data, adjacency_config="default"):
    """从单条 feature DataFrame 构建一个 GraphData（离线 process() 与在线 planner 共用）。

    `data` 为单行 DataFrame，列 = encoding_vectornet_features 的产出
    （POLYLINE_FEATURES / GROUND_TRUTH / TRAJ_ID_TO_INDICES / LANE_ID_TO_INDICES /
    SUBGRAPH_ID_TO_NODE_INDICES / SUBGRAPH_ID_TO_TYPE / EGO_LAST_POS / EGO_HISTORY）。
    离线读 pkl 得到它、在线 encoding_vectornet_features 直接返回它，二者走同一转换，
    保证在线 planner 的图（gnn_x 均值池化 + DefaultBuilder 无向补反向 + cluster col9）
    与训练逐字段一致。

    :param data: 单行 pandas DataFrame（.values[0] 取值）
    :param adjacency_config: 邻接矩阵配置名（对齐训练 best_model_default.pth 的 "default"）
    :return: (GraphData 或 None, skip_reason 或 None)；skip_reason ∈ {'empty_traj','no_polylines'}
    """
    features = data['POLYLINE_FEATURES'].values[0].astype(np.float32)
    cluster = features[:, 9].astype(np.int64)  # col9=subgraph_ID（原 -1=col11 end_time 取错列）
    y = data['GROUND_TRUTH'].values[0].astype(np.float32)

    # 确保Ground Truth是2维数组 (60, 2)
    if y.ndim == 3 and y.shape[0] == 1:
        y = y.squeeze(0)
    assert y.ndim == 2 and y.shape[1] == 2, f"Ground Truth维度错误: 期望(N, 2), 实际{y.shape}"

    traj_id_dict = data['TRAJ_ID_TO_INDICES'].values[0]
    lane_id_dict = data['LANE_ID_TO_INDICES'].values[0]

    if len(traj_id_dict) == 0 and len(lane_id_dict) == 0:
        return None, 'empty_traj'

    x_list, edge_list, cluster_list = [], [], []
    cluster_id = 0

    # Process trajectory polylines
    for polyline_id, (start, end) in traj_id_dict.items():
        if end < start:
            continue
        traj = features[start:end + 1]
        if traj.shape[0] < 2:
            continue

        x_list.append(traj)
        edge_index = get_sequential_edge_index(len(traj), start)
        if edge_index.ndim == 2 and edge_index.shape[1] > 0:
            edge_list.append(edge_index)
            cluster_list.append(cluster_id)
            cluster_id += 1

    # Process lane polylines
    if len(lane_id_dict) > 0:
        lane_offset = max(end for _, end in traj_id_dict.values()) + 1 if len(traj_id_dict) > 0 else 0
        for lane_id, (start, end) in lane_id_dict.items():
            if end < start:
                continue
            lane = features[lane_offset + start: lane_offset + end + 1]
            if lane.shape[0] < 2:
                continue

            x_list.append(lane)
            edge_index = get_sequential_edge_index(len(lane), lane_offset + start)
            if edge_index.ndim == 2 and edge_index.shape[1] > 0:
                edge_list.append(edge_index)
                cluster_list.append(cluster_id)
                cluster_id += 1

    if len(x_list) == 0:
        return None, 'no_polylines'

    x_combined = np.vstack(x_list)
    edge_combined = np.hstack(edge_list) if len(edge_list) > 0 else np.zeros((2, 0), dtype=np.int64)
    traj_tensor = torch.zeros(1, 3, dtype=torch.long)

    # 处理GNN数据
    pooled_tensor = torch.zeros(1, 12)
    edge_index_gnn = torch.zeros(2, 0, dtype=torch.long)
    edge_attr_gnn = torch.zeros(0, dtype=torch.long)

    if 'SUBGRAPH_ID_TO_NODE_INDICES' in data.columns:
        try:
            subgraph_id_to_indices = data['SUBGRAPH_ID_TO_NODE_INDICES'].values[0]
            subgraph_id_to_types = data['SUBGRAPH_ID_TO_TYPE'].values[0]
            pooled_dict = subgraph_pooling(features, subgraph_id_to_indices)

            custom_builder = get_adjacency_config(adjacency_config)
            gnn_graph = build_graph_from_subgraphs(
                pooled_dict,
                subgraph_id_to_types,
                distance_threshold=30.0,
                adjacency_builder=custom_builder
            )

            if len(gnn_graph['nodes']) > 0:
                pooled_tensor = torch.tensor([pooled_dict[i] for i in gnn_graph['nodes']], dtype=torch.float)
                if len(gnn_graph['edges']) > 0:
                    directed = np.array(gnn_graph['edges']).T                              # [2, E] builder 存单向 i<j
                    # 决策B：无向边补反向（GCN 需对称邻接）
                    mapped_edges = np.concatenate([directed, directed[[1, 0], :]], axis=1)  # [2, 2E]
                    edge_index_gnn = torch.tensor(mapped_edges, dtype=torch.long)
                    # edge_attr 与双向边一一对应（原 [:len(mapped_edges)]=[:2] 把 E 条错截成 2 条）
                    edge_types_sym = gnn_graph['edge_types'] + gnn_graph['edge_types']      # 长 2E，前正向后反向
                    edge_attr_gnn = torch.tensor(edge_types_sym, dtype=torch.long)
                    assert edge_index_gnn.shape[1] == edge_attr_gnn.shape[0], \
                        f"edge_index({edge_index_gnn.shape[1]}) 与 edge_attr({edge_attr_gnn.shape[0]}) 长度不一致"
        except Exception as e:
            print(f"Warning: GNN data processing failed: {e}")

    # 处理ego_last_pos
    ego_last_pos = np.zeros(2, dtype=np.float32)
    if 'EGO_LAST_POS' in data.columns:
        ego_last_pos_raw = data['EGO_LAST_POS'].values[0]
        if ego_last_pos_raw is not None:
            if isinstance(ego_last_pos_raw, (list, tuple)):
                ego_last_pos = np.array(ego_last_pos_raw, dtype=np.float32)[:2]
            elif isinstance(ego_last_pos_raw, np.ndarray):
                ego_last_pos = ego_last_pos_raw.astype(np.float32)[:2]

    # 提取EGO_HISTORY数据
    ego_history_data = None
    if 'EGO_HISTORY' in data.columns:
        ego_history_raw = data['EGO_HISTORY'].values[0]
        if ego_history_raw is not None and isinstance(ego_history_raw, np.ndarray):
            ego_history_data = torch.tensor(ego_history_raw, dtype=torch.float32)

    graph_data = GraphData(
        x=torch.tensor(x_combined, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        cluster=torch.tensor(cluster, dtype=torch.long),
        edge_index=torch.tensor(edge_combined, dtype=torch.long),
        valid_len=torch.tensor([cluster.max() if len(cluster) > 0 else 0], dtype=torch.long),
        time_step_len=torch.tensor([x_list[0].shape[0] if len(x_list) > 0 else 1], dtype=torch.long),
        traj_id_to_indices=traj_tensor,
        ego_last_pos=torch.tensor(ego_last_pos, dtype=torch.float32).reshape(2),
        gnn_x=pooled_tensor,
        gnn_edge_index=edge_index_gnn,
        gnn_edge_attr=edge_attr_gnn,
        ego_history=ego_history_data
    )
    return graph_data, None


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # 获取当前邻接矩阵配置
        self.adjacency_config = get_current_adjacency_config()
        print(f"[INFO] Loading GraphDataset from: {root} with config: {self.adjacency_config}")
        super().__init__(root, transform, pre_transform)

        pt_path = self.processed_paths[0]
        if os.path.exists(pt_path):
            print(f"[✅] Loading from existing {pt_path}")
            self.data, self.slices = torch.load(pt_path)
        else:
            print(f"[⚠️] {pt_path} not found, triggering process() with config: {self.adjacency_config}")
            self.process()
            self.data, self.slices = torch.load(pt_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # 根据邻接矩阵配置命名处理后的文件
        return [f'dataset_{self.adjacency_config}.pt']

    @property
    def processed_dir(self):
        # 根据邻接矩阵配置创建不同的processed子目录
        config_dir = os.path.join(self.root, 'processed', f'config_{self.adjacency_config}')
        os.makedirs(config_dir, exist_ok=True)
        return config_dir


    def download(self):
        pass

    def process(self):
        abs_root = os.path.abspath(self.root)
        print(f"[INFO] Loading GraphDataset from: {abs_root}")

        # 递归查找所有pkl文件
        data_path_list = []
        for root, dirs, files in os.walk(abs_root):
            for file in files:
                if file.endswith("pkl"):
                    data_path_list.append(os.path.join(root, file))

        data_path_list = sorted(data_path_list)
        print(f"[INFO] Found {len(data_path_list)} .pkl files.")

        graph_data_list = []
        stats = {
            'total_files': len(data_path_list),
            'processed': 0,
            'skipped_empty_traj': 0,
            'skipped_no_polylines': 0,
            'skipped_errors': 0
        }

        for data_path in tqdm(data_path_list):
            try:
                data = pd.read_pickle(data_path)
                # 抽出的共享 helper：离线 process() 与在线 planner 同一图构建，保证在线==离线逐字段一致
                graph_data, skip_reason = build_graph_data_from_feature_df(data, self.adjacency_config)
                if skip_reason == 'empty_traj':
                    stats['skipped_empty_traj'] += 1
                    continue
                if skip_reason == 'no_polylines':
                    stats['skipped_no_polylines'] += 1
                    continue

                graph_data_list.append(graph_data)
                stats['processed'] += 1

            except Exception as e:
                stats['skipped_errors'] += 1
                print(f"[ERROR] Failed processing: {os.path.basename(data_path)} -> {e}")
                continue

        print(f"\nProcessing Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Successfully processed: {stats['processed']} ({stats['processed'] / stats['total_files'] * 100:.1f}%)")
        print(f"   Skipped - empty trajectories: {stats['skipped_empty_traj']}")
        print(f"   Skipped - no polylines: {stats['skipped_no_polylines']}")
        print(f"   Skipped - errors: {stats['skipped_errors']}")

        if len(graph_data_list) == 0:
            raise RuntimeError(f"No valid data found in {self.root}. Please check your .pkl files.")

        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"\nAll valid data saved to {self.processed_paths[0]}")

def create_graph_data_from_history(ego_states, detections, map_api):
    """
    Convert nuPlan real-time data to GNN_Trans graph format.

    :param ego_states: List of ego vehicle states from history buffer
    :param detections: DetectionsTracks observation with tracked objects
    :param map_api: Map API for lane information
    :return: GraphData object for GNN_Trans model
    """
    try:
        # Extract ego trajectory history (use actual available history, no padding)
        ego_history = []
        if len(ego_states) > 1:
            for i in range(min(len(ego_states) - 1, len(ego_states) - 1)):
                idx = len(ego_states) - 1 - i
                if idx > 0:
                    prev_state = ego_states[idx - 1]
                    curr_state = ego_states[idx]

                    # Calculate increment (dx, dy) in global coordinates
                    dx = curr_state.rear_axle.x - prev_state.rear_axle.x
                    dy = curr_state.rear_axle.y - prev_state.rear_axle.y
                    ego_history.append([dx, dy])

        # Use actual history length, no artificial padding
        ego_history = np.array(ego_history, dtype=np.float32) if ego_history else np.zeros((0, 2), dtype=np.float32)

        # Get current ego state
        current_ego = ego_states[-1]
        ego_pos = np.array([current_ego.rear_axle.x, current_ego.rear_axle.y], dtype=np.float32)
        ego_heading = current_ego.rear_axle.heading
        ego_velocity = current_ego.dynamic_car_state.speed

        # Build node features from detected objects
        node_features = []
        object_positions = []

        # Add ego vehicle as first node
        ego_features = np.array([
            0.0, 0.0,  # Relative position (ego at origin)
            ego_velocity * np.cos(ego_heading),  # vx
            ego_velocity * np.sin(ego_heading),  # vy
            ego_heading,  # heading
            0.0,  # relative heading
            1.0,  # is_ego
            0.0, 0.0,  # padding to 9 features
        ], dtype=np.float32)
        node_features.append(ego_features)
        object_positions.append(ego_pos)

        # Add detected objects
        if detections and hasattr(detections, 'tracked_objects'):
            for obj in detections.tracked_objects.tracked_objects:
                # Calculate relative position
                obj_pos = np.array([obj.center.x, obj.center.y])
                rel_pos = obj_pos - ego_pos

                # Rotate to ego coordinate frame
                cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
                rel_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
                rel_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h

                # Object features
                obj_features = np.array([
                    rel_x, rel_y,  # Relative position
                    obj.velocity.x, obj.velocity.y,  # Velocity
                    obj.center.heading,  # Absolute heading
                    obj.center.heading - ego_heading,  # Relative heading
                    0.0,  # is_ego (false)
                    float(obj.tracked_object_type.value),  # Object type
                    0.0,  # padding
                ], dtype=np.float32)

                node_features.append(obj_features)
                object_positions.append(obj_pos)

        # Convert to tensors
        if len(node_features) == 0:
            x = torch.tensor(ego_features.reshape(1, -1), dtype=torch.float32)
        else:
            x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Create simple sequential edges
        num_nodes = x.shape[0]
        edge_index = get_sequential_edge_index(num_nodes)

        # Create dummy target
        y = torch.zeros(60, 2, dtype=torch.float32)

        # Create cluster information
        cluster = torch.zeros(num_nodes, dtype=torch.long)

        # Create minimal GNN data
        gnn_x = torch.zeros(1, 12, dtype=torch.float32)
        gnn_edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Create graph data
        graph_data = GraphData(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=y,
            cluster=cluster,
            ego_history=torch.tensor(ego_history, dtype=torch.float32),
            ego_last_pos=torch.tensor(ego_pos, dtype=torch.float32),
            gnn_x=gnn_x,
            gnn_edge_index=gnn_edge_index,
        )

        return graph_data

    except Exception as e:
        print(f"Error in create_graph_data_from_history: {e}")
        # Return minimal fallback data
        return GraphData(
            x=torch.zeros(1, 9, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.zeros(60, 2, dtype=torch.float32),
            cluster=torch.zeros(1, dtype=torch.long),
            ego_history=torch.zeros(20, 2, dtype=torch.float32),
            ego_last_pos=torch.zeros(2, dtype=torch.float32),
            gnn_x=torch.zeros(1, 12, dtype=torch.float32),
            gnn_edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
