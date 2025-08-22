# GNN_Trans 远程批量训练指南

## 概览
本指南介绍如何使用改进的训练系统进行远程批量训练，支持详细进度监控和自动保存。

## 主要功能

### 1. 增强的 train.py 功能
- ✅ 详细的日志记录（文件+控制台双输出）
- ✅ 实时进度保存（progress.json）
- ✅ 完整的训练指标记录（ADE/FDE/Loss/LR）
- ✅ 自动文件夹管理（包含训练配置信息）
- ✅ 早停机制和最佳模型保存

### 2. 强大的进度监控系统
- ✅ 实时进度条显示
- ✅ 详细训练指标监控
- ✅ 多任务状态概览
- ✅ 自动刷新功能

## 快速开始

### 1. 单次训练
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

# 运行训练（会提示选择邻接矩阵配置）
python train.py
```

### 2. 远程训练（推荐）
```bash
# 在后台运行训练
nohup python train.py > training_output.log 2>&1 &

# 查看后台任务
jobs

# 获取进程ID
ps aux | grep "python train.py"
```

### 3. 数据集配置
在训练前，修改 `train.py` 中的设置：
```python
# 使用完整数据集
USE_SMALL_DATASET = False

# 使用小数据集测试
USE_SMALL_DATASET = True
SMALL_DATASET_SIZE = 1000
```

### 4. 邻接矩阵配置选项
1. **default** - 默认全连接模式（信息最丰富）
2. **ego_only** - ego为中心的星型连接（计算高效）
3. **no_ego_road** - 禁止ego连接道路元素（测试道路信息重要性）
4. **nearby_only** - 只连接15米内的元素（平衡性能和效率）

## 批量训练

### 1. 使用现有的批量训练脚本
项目中已经有了一个全面的批量训练脚本：`auto_train_all_configs.sh`

```bash
# 赋予脚本可执行权限
chmod +x auto_train_all_configs.sh

# 推荐在远程训练时后台运行
nohup ./auto_train_all_configs.sh &

# 检查是否成功启动
jobs
ps aux | grep auto_train_all_configs
```

### 2. 监控批量训练进度
```bash
# 检查当前正在运行的训练
nvidia-smi  # 检查GPU使用情况

# 查看主批量日志
tail -f batch_training_logs/batch_training_YYYYMMDD_HHMMSS.log

# 查看特定配置的训练日志
tail -f batch_training_logs/config_1_default_YYYYMMDD_HHMMSS.log
```

### 3. 高级进度监控
使用增强的进度监控脚本：

```bash
# 列出所有活动/最近的训练文件夹
python check_training_progress.py

# 实时监控特定训练进度
python check_training_progress.py --folder training_20250822_153453_default_ds70000_bs8_lr1e-04

# 自定义刷新间隔（每5秒）
python check_training_progress.py --folder [folder_name] --interval 5
```

## 远程训练最佳实践

### 1. 使用 Screen 会话（推荐）
```bash
# 创建新会话
screen -S gnn_batch_training

# 运行批量训练
./auto_train_all_configs.sh

# 分离会话（训练继续运行）
# 按 Ctrl+A, 然后按 D

# 重新连接会话
screen -r gnn_batch_training

# 查看所有会话
screen -ls

# 如有需要，强制结束会话
screen -X -S gnn_batch_training quit
```

### 2. 使用 nohup（替代方案）
```bash
# 在后台运行
nohup ./auto_train_all_configs.sh &

# 查看进程
ps aux | grep auto_train_all_configs

# 如有需要，结束进程
kill [PID]
```

### 3. 监控 GPU 使用
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 查看哪个进程在使用GPU
nvidia-smi pmon

# 如有需要，结束GPU进程
sudo kill [PID]
```

## 监控功能详解

### 增强的进度显示
`check_training_progress.py` 脚本提供：

```
🔍 MONITORING TRAINING: training_20250822_153453_default_ds70000_bs8_lr1e-04
📅 Current Time: 2025-08-22 18:30:15
================================================================================
📊 Training Status: Running

📈 Progress Overview:
   [████████████████████████████████████████████████████████████] 75.0%
   Epoch: 75/100

📊 Current Metrics:
   Train Loss: 0.045231
   Val Loss: 0.048567
   ADE: 1.2543m
   FDE: 2.8765m
   LR: 5.00e-05

⏱️  Time Information:
   Elapsed: 2.3h
   Remaining: 0.8h
   Total Estimated: 3.1h
   Last Update: 2025-08-22 18:29:45

📝 Recent Log Entries:
   2025-08-22 18:29:45 - INFO - ✅ Epoch  75/100 Results:
   2025-08-22 18:29:45 - INFO -    📈 Train Loss: 0.045231
   2025-08-22 18:29:45 - INFO -    📉 Val Loss:   0.048567
   2025-08-22 18:29:45 - INFO -    🎯 ADE:        1.2543 meters
   2025-08-22 18:29:45 - INFO -    🎯 FDE:        2.8765 meters

🔄 Refreshing in 10 seconds... (Ctrl+C to stop)
```

## 文件结构

### 训练结果文件夹结构
```
training_20250822_153453_default_ds70000_bs8_lr1e-04/
├── models/
│   ├── best_model_default.pth
│   └── final_model_default.pth
├── results/
│   └── loss_record_default_ep100_bs8_lr0.0001.json
├── logs/
│   ├── training_log.txt
│   ├── training_config_default.json
│   └── progress_history.jsonl
├── plots/
│   └── training_analysis_default_ep100_lr0.0001.png
├── progress.json
└── final_results.json
```

### 批量训练日志结构
```
batch_training_logs/
├── batch_training_20250822_160000.log        # 主批量日志
├── config_1_default_20250822_160000.log      # 配置1详细日志
├── config_2_ego_only_20250822_162000.log     # 配置2详细日志
├── config_3_no_ego_road_20250822_164000.log  # 配置3详细日志
└── config_4_nearby_only_20250822_170000.log  # 配置4详细日志
```

## 故障排除

### 1. 检查训练状态
```bash
# 检查训练是否在运行
ps aux | grep "python train.py"
ps aux | grep "auto_train_all_configs"

# 检查GPU使用情况
nvidia-smi

# 检查磁盘空间
df -h
```

### 2. 常见问题

#### 训练卡住或无进展
```bash
# 检查系统资源
top
htop

# 检查GPU内存
nvidia-smi

# 如有必要，重启训练
pkill -f "python train.py"
```

#### 日志文件未更新
```bash
# 检查文件权限
ls -la training_*/logs/

# 检查磁盘空间
df -h

# 手动检查进度文件
cat training_*/progress.json
```

### 3. 紧急停止
```bash
# 停止所有训练进程
pkill -f "python train.py"
pkill -f "auto_train_all_configs"

# 检查进程是否已结束
ps aux | grep train
```

## 提示和技巧

1. **始终使用 screen 或 nohup** 进行长时间训练
2. **监控磁盘空间** - 训练会生成大量日志文件
3. **检查GPU内存** 在开始批量训练前
4. **定期保存** - 系统会在每个epoch结束时自动保存进度
5. **使用 tail -f** 实时监控日志
6. **每个配置预留 2-4 小时** 完整数据集训练时间

## 示例工作流程

```bash
# 1. 启动screen会话
screen -S training

# 2. 进入项目目录
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

# 3. 确保使用完整数据集
# 编辑 train.py: USE_SMALL_DATASET = False

# 4. 开始批量训练
./auto_train_all_configs.sh

# 5. 分离screen（Ctrl+A, D）

# 6. 从另一个终端监控
python check_training_progress.py

# 7. 检查批量进度
tail -f batch_training_logs/batch_training_*.log

# 8. 重新连接查看最终结果
screen -r training
```
