#!/bin/bash
# 使用tmux会话后台运行GNN_Trans训练（推荐）

SESSION_NAME="gnn_training"

cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

echo "创建tmux会话: $SESSION_NAME"
echo "当前目录: $(pwd)"
echo "训练完成后可以用以下命令查看："
echo "  tmux attach -t $SESSION_NAME  # 重新连接会话"
echo "  tmux ls                      # 查看所有会话"
echo "  按 Ctrl+B+D 可以分离会话"

# 检查tmux会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "会话 $SESSION_NAME 已存在，正在连接..."
    tmux attach -t $SESSION_NAME
else
    # 创建新的tmux会话并运行训练
    tmux new-session -d -s $SESSION_NAME -c "$(pwd)"
    tmux send-keys -t $SESSION_NAME "python train.py" Enter
    echo "Tmux会话已创建并启动GNN_Trans训练"
    echo "使用 'tmux attach -t $SESSION_NAME' 重新连接"
fi
