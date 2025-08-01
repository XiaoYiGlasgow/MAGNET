# GNN_Trans - Graph Neural Network with Transformer for Trajectory Prediction

这个项目实现了基于图神经网络(GNN)和Transformer的轨迹预测模型，适用于自动驾驶场景。

## 项目结构

- `train.py` - 模型训练脚本
- `evaluate.py` - 模型评估脚本
- `inference.py` - 模型推理脚本
- `config/` - 配置文件目录
- `models/` - 模型定义
- `planner/` - 规划器相关代码
- `utils/` - 工具函数
- `simple_demo.py` - 简单演示脚本

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 训练模型
```bash
python train.py
```

3. 评估模型
```bash
python evaluate.py
```

## 配置

主要配置文件位于 `config/` 目录下，可以根据需要调整模型参数和训练设置。

## 演示

运行演示脚本查看模型效果：
```bash
python simple_demo.py
```
