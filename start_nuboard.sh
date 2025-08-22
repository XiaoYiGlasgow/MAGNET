#!/bin/bash
# Quick start nuBoard with existing results

echo "==================================="
echo "启动 GNN_Trans nuBoard 可视化"
echo "==================================="

# 切换到 nuplan-devkit 目录
cd /mnt/local/Xiao_Yi/nuplan-devkit

# 启动 nuBoard
echo "启动 nuBoard 服务器..."
echo "访问地址: http://localhost:5006"
echo "按 Ctrl+C 停止服务"
echo "==================================="

python nuplan/planning/script/run_nuboard.py \
    simulation_path=/mnt/local/Xiao_Yi/nuplan_results

echo "nuBoard 服务已停止"
