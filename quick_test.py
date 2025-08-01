#!/usr/bin/env python3
"""
简化版基础功能测试 - 确保GNN_Trans能正常运行
"""
import os
import sys
import torch

# 设置路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

def main():
    print("🔍 开始基础功能验证...")
    
    # 1. 测试配置加载
    try:
        import config
        print("✅ 配置加载成功")
        print(f"   训练路径: {config.TRAIN_PATH}")
        print(f"   输出维度: {config.OUT_CHANNELS}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 测试模型创建
    try:
        from models.transformer_predictor import TransformerPredictor
        model = TransformerPredictor()
        print("✅ 模型创建成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 3. 测试前向传播
    try:
        # 创建模拟数据
        class MockData:
            def __init__(self):
                self.x = torch.randn(50, 12)
                self.y = torch.randn(1, 60)
                self.gnn_x = torch.randn(50, 12)
                self.gnn_edge_index = torch.randint(0, 50, (2, 100))
        
        data = MockData()
        with torch.no_grad():
            output = model(data)
        print("✅ 前向传播成功")
        print(f"   输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    # 4. 检查数据路径
    if os.path.exists(config.TRAIN_PATH):
        pkl_files = [f for f in os.listdir(config.TRAIN_PATH) if f.endswith('.pkl')]
        print(f"✅ 数据路径存在，包含 {len(pkl_files)} 个pkl文件")
    else:
        print(f"⚠️ 数据路径不存在: {config.TRAIN_PATH}")
    
    print("\n🎉 基础功能验证完成！你的GNN_Trans基础版本可以运行。")
    return True

if __name__ == "__main__":
    main()
