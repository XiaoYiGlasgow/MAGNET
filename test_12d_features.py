#!/usr/bin/env python3

"""
简单测试：nuPlan数据处理为12维特征向量
使用现有的数据处理函数进行测试
"""

import os
import sys
import torch
import numpy as np

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

def test_12d_feature_processing():
    """测试12维特征处理功能"""

    print("🧪 测试12维特征向量处理功能")
    print("="*50)

    # 1. 测试导入功能
    print("1️⃣ 测试导入功能...")
    try:
        from GNN_Trans.utils.data_utils import encoding_vectornet_features
        print("✅ encoding_vectornet_features 导入成功")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 2. 创建模拟数据测试特征处理
    print("\n2️⃣ 创建模拟数据测试特征处理...")

    # 模拟ego历史轨迹 (40个时间点，每个点7维：x,y,heading,vx,vy,ax,ay)
    num_past_poses = 40
    ego_past = np.random.randn(num_past_poses, 7).astype(np.float32)
    ego_past[:, :2] = np.cumsum(ego_past[:, :2] * 0.1, axis=0)  # 累积位置
    ego_past[:, 2] = np.linspace(0, 0.5, num_past_poses)  # 航向角
    ego_past[:, 3:5] = np.random.uniform(5, 15, (num_past_poses, 2))  # 速度

    print(f"   ✅ 模拟ego轨迹: {ego_past.shape}")

    # 模拟ego未来轨迹 (30个时间点)
    num_future_poses = 30
    ego_future = np.random.randn(num_future_poses, 3).astype(np.float32)
    ego_future[:, :2] = np.cumsum(ego_future[:, :2] * 0.1, axis=0)  # 累积位置
    ego_future[:, 2] = np.linspace(0.5, 1.0, num_future_poses)  # 航向角

    print(f"   ✅ 模拟ego未来轨迹: {ego_future.shape}")

    # 模拟其他车辆轨迹
    num_agents = 5
    agents_past = []
    for i in range(num_agents):
        agent_traj = np.random.randn(num_past_poses, 7).astype(np.float32)
        agent_traj[:, :2] = np.cumsum(agent_traj[:, :2] * 0.1, axis=0)
        agent_traj[:, 2] = np.linspace(0, 0.3, num_past_poses)
        agent_traj[:, 3:5] = np.random.uniform(3, 12, (num_past_poses, 2))
        agents_past.append(agent_traj)

    print(f"   ✅ 模拟其他车辆轨迹: {len(agents_past)}个车辆")

    # 模拟地图数据
    vector_map = {
        'lanes': [
            np.random.randn(20, 3).astype(np.float32),  # 车道1
            np.random.randn(15, 3).astype(np.float32),  # 车道2
        ],
        'crosswalks': [
            np.random.randn(10, 3).astype(np.float32),  # 人行横道1
        ],
        'route_lanes': [
            np.random.randn(25, 3).astype(np.float32),  # 路线车道1
        ]
    }

    print(f"   ✅ 模拟地图数据: {len(vector_map['lanes'])}条车道")

    # 模拟时间戳
    time_stamps_past = torch.linspace(0, 4.0, num_past_poses)  # 4秒历史
    T_ref = 4.0  # 参考时间

    print(f"   ✅ 模拟时间戳: {time_stamps_past.shape}")

    # 3. 调用特征处理函数
    print("\n3️⃣ 调用12维特征处理函数...")

    try:
        feature_df = encoding_vectornet_features(
            agents_past,
            None,  # agents_future在推理时不需要
            ego_past,
            ego_future,
            vector_map,
            time_stamps_past,
            T_ref
        )

        print("✅ 12维特征处理成功!")

        # 4. 分析生成的特征
        print("\n4️⃣ 分析生成的12维特征...")

        polyline_features = feature_df.iloc[0]['POLYLINE_FEATURES']
        ground_truth = feature_df.iloc[0]['GROUND_TRUTH']

        print(f"   特征矩阵形状: {polyline_features.shape}")
        print(f"   Ground truth形状: {ground_truth.shape}")

        # 验证12维特征的结构
        if polyline_features.shape[1] == 12:
            print("✅ 确认为12维特征向量")
            print("   12维特征结构:")
            print("     [0:2] x_start, y_start")
            print("     [2:4] x_end, y_end")
            print("     [4]   heading_start")
            print("     [5]   heading_end")
            print("     [6]   normalized_timestamp")
            print("     [7]   v_start")
            print("     [8]   v_end")
            print("     [9]   subgraph_ID")
            print("     [10]  start_time")
            print("     [11]  end_time")
        else:
            print(f"❌ 特征维度不正确: {polyline_features.shape[1]} (期望12维)")
            return False

        # 5. 显示特征样本
        print(f"\n5️⃣ 特征样本 (前3行):")
        for i in range(min(3, polyline_features.shape[0])):
            print(f"   Row {i}: {polyline_features[i]}")

        # 6. 保存测试结果
        print(f"\n6️⃣ 保存测试结果...")

        output_dir = "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/processed_data"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/test_12d_features.pkl"
        feature_df.to_pickle(output_file)

        print(f"✅ 测试结果保存到: {output_file}")

        return True

    except Exception as e:
        print(f"❌ 特征处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """测试与GNN Transformer模型的兼容性"""

    print("\n🔗 测试与GNN Transformer模型的兼容性")
    print("="*50)

    try:
        # 导入规划器
        from GNN_Trans.planner.gnn_transformer_planner import GNNTransformerPlanner, GraphData

        # 创建规划器实例
        planner = GNNTransformerPlanner()
        print(f"✅ 规划器初始化成功，设备: {planner.device}")

        # 创建模拟的12维特征数据
        num_nodes = 100
        polyline_features = np.random.randn(num_nodes, 12).astype(np.float32)

        # 为Transformer使用前9维特征
        x = torch.FloatTensor(polyline_features[:, :9]).to(planner.device)

        # 为GNN使用完整的12维特征
        gnn_x = torch.FloatTensor(polyline_features).to(planner.device)

        # 创建简单的边连接
        edge_list = []
        for i in range(min(num_nodes-1, 50)):
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])

        edge_index = torch.LongTensor(edge_list).t().to(planner.device)
        gnn_edge_index = edge_index.clone()

        # 创建GraphData对象
        graph_data = GraphData(
            x=x,
            edge_index=edge_index,
            gnn_x=gnn_x,
            gnn_edge_index=gnn_edge_index,
            y=torch.zeros(1, 60).to(planner.device)
        )

        print(f"✅ GraphData创建成功:")
        print(f"   x (Transformer): {graph_data.x.shape}")
        print(f"   gnn_x (GNN): {graph_data.gnn_x.shape}")
        print(f"   edge_index: {graph_data.edge_index.shape}")

        # 测试模型推理
        with torch.no_grad():
            output = planner.model(graph_data)

        print(f"✅ 模型推理成功:")
        print(f"   输出形状: {output.shape}")
        print(f"   期望形状: [1, 60] (30个时间步 × 2坐标)")

        if output.shape == (1, 60):
            print("✅ 输出格式正确!")
            return True
        else:
            print(f"❌ 输出格式不正确")
            return False

    except Exception as e:
        print(f"❌ 模型兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""

    print("🚀 nuPlan数据处理为12维特征向量 - 完整测试")
    print("="*60)

    # 测试1: 12维特征处理功能
    test1_success = test_12d_feature_processing()

    # 测试2: 模型兼容性
    test2_success = test_model_compatibility()

    # 总结测试结果
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print(f"{'='*60}")
    print(f"✅ 12维特征处理: {'通过' if test1_success else '失败'}")
    print(f"✅ 模型兼容性: {'通过' if test2_success else '失败'}")

    if test1_success and test2_success:
        print("\n🎉 所有测试通过！12维特征向量处理系统正常工作")
        print("\n📋 下一步你可以:")
        print("1. 处理真实的nuPlan数据")
        print("2. 训练你的GNN Transformer模型")
        print("3. 运行完整的轨迹规划测试")
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息")

    return test1_success and test2_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
