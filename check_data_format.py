#!/usr/bin/env python3

"""
检查已有的12维特征数据格式
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_existing_data():
    """检查已有的12维特征数据"""
    
    # 检查现有数据文件
    data_dir = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/processed_data")
    
    print("🔍 检查已有的12维特征数据...")
    
    # 检查测试数据
    test_file = data_dir / "test_12d_features.pkl"
    if test_file.exists():
        print(f"\n✅ 找到测试数据: {test_file}")
        
        try:
            # 加载数据
            data = pd.read_pickle(test_file)
            print(f"   数据类型: {type(data)}")
            print(f"   数据大小: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            
            if isinstance(data, pd.DataFrame):
                print(f"   DataFrame形状: {data.shape}")
                print(f"   列名: {list(data.columns)}")
                
                # 检查POLYLINE_FEATURES
                if 'POLYLINE_FEATURES' in data.columns:
                    features = data.iloc[0]['POLYLINE_FEATURES']
                    print(f"   特征矩阵形状: {features.shape}")
                    print(f"   特征矩阵类型: {type(features)}")
                    print(f"   前5行特征样例:")
                    print(features[:5])
                    
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
    
    # 检查训练数据目录
    train_dir = data_dir / "train"
    if train_dir.exists():
        print(f"\n✅ 找到训练数据目录: {train_dir}")
        train_files = list(train_dir.glob("*.pkl"))
        print(f"   包含 {len(train_files)} 个pkl文件")
        
        if train_files:
            # 检查第一个文件
            sample_file = train_files[0]
            print(f"   示例文件: {sample_file.name}")
            
            try:
                sample_data = pd.read_pickle(sample_file)
                print(f"   示例数据类型: {type(sample_data)}")
                if hasattr(sample_data, 'shape'):
                    print(f"   示例数据形状: {sample_data.shape}")
            except Exception as e:
                print(f"   ❌ 读取示例失败: {e}")
    
    # 检查测试数据目录  
    test_dir = data_dir / "test"
    if test_dir.exists():
        print(f"\n✅ 找到测试数据目录: {test_dir}")
        test_files = list(test_dir.glob("*.pkl"))
        print(f"   包含 {len(test_files)} 个pkl文件")
    
    print(f"\n📋 数据处理建议:")
    print(f"   1. 如果需要更多数据，可以使用现有的处理流程")
    print(f"   2. 12维特征格式已确定，可直接用于训练")
    print(f"   3. 数据保存为pandas DataFrame格式")

if __name__ == "__main__":
    check_existing_data()
