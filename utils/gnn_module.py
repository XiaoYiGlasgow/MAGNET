# 简化版 gnn_module.py
import numpy as np

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

def build_graph_from_subgraphs(subgraph_vectors, subgraph_types, distance_threshold=30.0):
    """简化的图构建"""
    nodes = list(subgraph_vectors.keys())
    edges = []
    edge_types = []
    
    # 简单的全连接图
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            edges.append((nodes[i], nodes[j]))
            edge_types.append(0)  # 默认边类型
    
    return {
        'nodes': nodes,
        'edges': edges, 
        'edge_types': edge_types
    }
