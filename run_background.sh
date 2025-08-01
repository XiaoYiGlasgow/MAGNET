#!/bin/bash
# 后台运行GNN_Trans训练脚本 - 使用nohup

cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

echo "开始后台训练GNN_Trans模型..."
echo "当前目录: $(pwd)"
echo "日志将保存到 training.log"
echo "进程ID将保存到 training.pid"

# 使用nohup后台运行训练脚本
nohup python train.py > training.log 2>&1 &

# 保存进程ID
echo $! > training.pid

echo "GNN_Trans训练已在后台启动"
echo "进程ID: $(cat training.pid)"
echo "查看实时日志: tail -f training.log"
echo "查看进程状态: ps -p $(cat training.pid)"
echo "停止训练: kill $(cat training.pid)"
