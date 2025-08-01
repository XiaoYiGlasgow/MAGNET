#!/bin/bash
# 使用screen会话后台运行

SESSION_NAME="gnn_training"

echo "创建screen会话: $SESSION_NAME"
echo "训练完成后可以用以下命令查看："
echo "  screen -r $SESSION_NAME  # 重新连接会话"
echo "  screen -ls              # 查看所有会话"
echo "  按 Ctrl+A+D 可以分离会话"

# 创建新的screen会话并运行训练
screen -dmS $SESSION_NAME bash -c "cd $(pwd) && python train.py; exec bash"

echo "Screen会话已创建并启动训练"
echo "使用 'screen -r $SESSION_NAME' 重新连接"
