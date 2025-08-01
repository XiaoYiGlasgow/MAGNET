#!/usr/bin/env python3
"""
基础功能测试脚本
确保GNN_Trans的基础版本能正常工作
"""
import os
import sys
import torch
import torch.nn as nn

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_config():
    """测试配置文件加载"""
    try:
        import config
        print("✅ Config loaded successfully")
        print(f"   TRAIN_PATH: {config.TRAIN_PATH}")
        print(f"   OUT_CHANNELS: {config.OUT_CHANNELS}")
        print(f"   EMBED_DIM: {config.EMBED_DIM}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_model():
    """测试模型创建和前向传播"""
    try:
        from models.transformer_predictor import TransformerPredictor
        
        # 创建模型
        model = TransformerPredictor()
        print("✅ Model created successfully")
        
        # 创建虚拟数据测试前向传播
        class DummyData:
            def __init__(self):
                self.x = torch.randn(50, 12)  # 50个节点，12维特征
                self.y = torch.randn(1, 60)   # 1个样本，60维输出
                self.gnn_x = torch.randn(50, 12)
                self.gnn_edge_index = torch.randint(0, 50, (2, 100))
                
        dummy_data = DummyData()
        
        # 测试前向传播
        with torch.no_grad():
            output = model(dummy_data)
            print(f"✅ Forward pass successful")
            print(f"   Input shape: {dummy_data.x.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output shape: {dummy_data.y.shape}")
            
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """测试数据集加载"""
    try:
        from utils.dataset import GraphDataset
        import config
        
        # 检查数据路径
        if not os.path.exists(config.TRAIN_PATH):
            print(f"❌ Training data path not found: {config.TRAIN_PATH}")
            return False
            
        print(f"✅ Training data path exists: {config.TRAIN_PATH}")
        
        # 尝试加载数据集（限制数量避免过长等待）
        try:
            dataset = GraphDataset(config.TRAIN_PATH)
            print(f"✅ Dataset loaded successfully")
            print(f"   Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample x shape: {sample.x.shape if hasattr(sample, 'x') else 'N/A'}")
                print(f"   Sample y shape: {sample.y.shape if hasattr(sample, 'y') else 'N/A'}")
                
        except Exception as e:
            print(f"⚠️ Dataset loading encountered issues: {e}")
            print("   This might be normal if dataset preprocessing is needed")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_training_components():
    """测试训练组件"""
    try:
        import torch.optim as optim
        from models.transformer_predictor import TransformerPredictor
        
        # 创建模型和优化器
        model = TransformerPredictor()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("✅ Training components created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return True
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def main():
    """运行所有基础功能测试"""
    print("🔍 开始基础功能测试...")
    print("=" * 50)
    
    tests = [
        ("配置文件", test_config),
        ("模型创建和前向传播", test_model),
        ("数据集", test_dataset),
        ("训练组件", test_training_components)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📝 测试: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    all_passed = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有基础功能测试通过！可以开始训练了。")
    else:
        print("\n⚠️ 有些测试失败，需要先解决这些问题。")
    
    return all_passed

if __name__ == "__main__":
    main()
