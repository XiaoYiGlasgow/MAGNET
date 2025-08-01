import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from utils.gnn_module import subgraph_pooling, build_graph_from_subgraphs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def get_sequential_edge_index(num_nodes, start=0):
    edge_indices = np.array(
        [[i, i + 1] for i in range(start, start + num_nodes - 1)],
        dtype=np.int64
    ).T
    return edge_indices


from torch_geometric.data import Data

class GraphData(Data):
    def __init__(self, x=None, edge_index=None, y=None, cluster=None,
                 valid_len=None, time_step_len=None, traj_id_to_indices=None,
                 ego_current_abs=None,
                 gnn_x=None, gnn_edge_index=None, gnn_edge_attr=None,
                 **kwargs):
        super().__init__(x=x, edge_index=edge_index, y=y, cluster=cluster,
                         valid_len=valid_len, time_step_len=time_step_len,
                         traj_id_to_indices=traj_id_to_indices,
                         ego_current_abs=ego_current_abs,
                         gnn_x=gnn_x, gnn_edge_index=gnn_edge_index, gnn_edge_attr=gnn_edge_attr,
                         **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'gnn_edge_index':
            return self.gnn_x.size(0)
        if key == 'cluster':
            return int(self.cluster.max().item()) + 1
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'gnn_edge_index']:
            return 1
        if key in ['x', 'y', 'cluster', 'gnn_x', 'gnn_edge_attr']:
            return 0
        if key == 'traj_id_to_indices':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        print("[INFO] Loading GraphDataset from:", root)
        super().__init__(root, transform, pre_transform)

        pt_path = self.processed_paths[0]
        if os.path.exists(pt_path):
            print(f"[✅] Loading from existing {pt_path}")
            self.data, self.slices = torch.load(pt_path)
        else:
            print(f"[⚠️] {pt_path} not found, triggering process()")
            self.process()
            self.data, self.slices = torch.load(pt_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']


    def download(self):
        pass

    def process(self):
        abs_root = os.path.abspath(self.root)
        print(f"[INFO] Looking in: {abs_root}")

        data_path_list = sorted(
            os.path.join(abs_root, path)
            for path in os.listdir(abs_root)
            if path.endswith("pkl")
        )

        print(f"[INFO] Found {len(data_path_list)} .pkl files.")

        graph_data_list = []

        # Add detailed statistics
        stats = {
            'total_files': len(data_path_list),
            'processed': 0,
            'skipped_empty_traj': 0,
            'skipped_empty_lane': 0,
            'skipped_no_polylines': 0,
            'skipped_no_gnn_data': 0,
            'skipped_errors': 0
        }

        for data_path in tqdm(data_path_list):
            try:
                data = pd.read_pickle(data_path)
                features = data['POLYLINE_FEATURES'].values[0].astype(np.float32)
                cluster = features[:, -1].astype(np.int64)
                y = data['GROUND_TRUTH'].values[0].astype(np.float32)
                traj_id_dict = data['TRAJ_ID_TO_INDICES'].values[0]
                lane_id_dict = data['LANE_ID_TO_INDICES'].values[0]

                # ✅ 放宽检查条件：只要有轨迹或车道数据之一即可
                if len(traj_id_dict) == 0 and len(lane_id_dict) == 0:
                    stats['skipped_empty_traj'] += 1
                    continue

                x_list, edge_list, cluster_list = [], [], []
                cluster_id = 0

                # Process trajectory polylines
                for polyline_id, (start, end) in traj_id_dict.items():
                    if end < start:
                        continue

                    traj = features[start:end + 1]
                    if traj.shape[0] < 2:  # ✅ 只需要至少2个点
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
                        if lane.shape[0] < 2:  # ✅ 只需要至少2个点
                            continue

                        x_list.append(lane)
                        edge_index = get_sequential_edge_index(len(lane), lane_offset + start)
                        if edge_index.ndim == 2 and edge_index.shape[1] > 0:
                            edge_list.append(edge_index)
                            cluster_list.append(cluster_id)
                            cluster_id += 1

                # ✅ 放宽检查：只要有一些有效数据即可
                if len(x_list) == 0:
                    stats['skipped_no_polylines'] += 1
                    continue

                # ✅ 处理 GNN 数据，即使缺失也继续处理
                pooled_tensor = torch.zeros(1, 12)  # 默认值
                edge_index_gnn = torch.zeros(2, 0, dtype=torch.long)  # 空边
                edge_attr_gnn = torch.zeros(0, dtype=torch.long)  # 空属性

                # 尝试处理 GNN 数据，如果失败就使用默认值
                try:
                    if ('SUBGRAPH_ID_TO_NODE_INDICES' in data.columns and
                            'SUBGRAPH_ID_TO_TYPE' in data.columns):

                        subgraph_id_to_indices = data['SUBGRAPH_ID_TO_NODE_INDICES'].values[0]
                        subgraph_id_to_types = data['SUBGRAPH_ID_TO_TYPE'].values[0]

                        from utils.gnn_module import subgraph_pooling, build_graph_from_subgraphs

                        pooled_dict = subgraph_pooling(features, subgraph_id_to_indices)
                        gnn_graph = build_graph_from_subgraphs(
                            pooled_dict,
                            subgraph_id_to_types,
                            distance_threshold=30.0
                        )

                        if len(gnn_graph['nodes']) > 0:
                            pooled_tensor = torch.tensor([pooled_dict[i] for i in gnn_graph['nodes']],
                                                         dtype=torch.float)
                            if len(gnn_graph['edges']) > 0:
                                edge_index_gnn = torch.tensor(gnn_graph['edges'], dtype=torch.long).T.contiguous()
                                edge_attr_gnn = torch.tensor(gnn_graph['edge_types'], dtype=torch.long)
                except Exception as e:
                    # 如果 GNN 处理失败，继续使用默认值
                    pass

                # ✅ 构建图数据
                x_combined = np.vstack(x_list)
                edge_combined = np.hstack(edge_list) if len(edge_list) > 0 else np.zeros((2, 0), dtype=np.int64)

                traj_tensor = torch.tensor(
                    [[int(k), int(v[0]), int(v[1])] for k, v in traj_id_dict.items()],
                    dtype=torch.long
                ) if len(traj_id_dict) > 0 else torch.zeros(0, 3, dtype=torch.long)

                # ✅ 确保 ego_current_abs 存在
                ego_current_abs = data.get('EGO_CURRENT_ABS', [np.zeros(2)])
                if not isinstance(ego_current_abs, list):
                    ego_current_abs = [ego_current_abs]

                graph_data = GraphData(
                    x=torch.tensor(x_combined, dtype=torch.float32),
                    y=torch.tensor(y, dtype=torch.float32),
                    cluster=torch.tensor(cluster, dtype=torch.long),
                    edge_index=torch.tensor(edge_combined, dtype=torch.long),
                    valid_len=torch.tensor([cluster.max() if len(cluster) > 0 else 0], dtype=torch.long),
                    time_step_len=torch.tensor([x_list[0].shape[0] if len(x_list) > 0 else 1], dtype=torch.long),
                    traj_id_to_indices=traj_tensor,
                    ego_current_abs=torch.tensor(ego_current_abs, dtype=torch.float32),
                    gnn_x=pooled_tensor,
                    gnn_edge_index=edge_index_gnn,
                    gnn_edge_attr=edge_attr_gnn
                )

                graph_data_list.append(graph_data)
                stats['processed'] += 1

            except Exception as e:
                stats['skipped_errors'] += 1
                print(f"[❌ ERROR] Failed processing: {os.path.basename(data_path)} → {e}")
                continue

        # ✅ 输出详细统计
        print(f"\n📊 Processing Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(
            f"   Successfully processed: {stats['processed']} ({stats['processed'] / stats['total_files'] * 100:.1f}%)")
        print(f"   Skipped - empty trajectories: {stats['skipped_empty_traj']}")
        print(f"   Skipped - no polylines: {stats['skipped_no_polylines']}")
        print(f"   Skipped - errors: {stats['skipped_errors']}")

        if len(graph_data_list) == 0:
            raise RuntimeError(f"[❌] No valid data found in {self.root}. Please check your .pkl files.")

        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"\n[✅] All valid data saved to {self.processed_paths[0]}")