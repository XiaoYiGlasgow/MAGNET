#!/bin/bash
# 监控GNN_Trans训练状态

cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

echo "=== GNN_Trans训练监控 ==="
echo "时间: $(date)"
echo

# 检查进程状态
if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 训练进程正在运行 (PID: $PID)"
        echo "内存使用: $(ps -p $PID -o pid,ppid,cmd,%mem,%cpu --no-headers)"
    else
        echo "❌ 训练进程已停止"
    fi
else
    echo "⚠️  未找到训练进程ID文件"
fi

echo

# 检查日志文件
if [ -f "training.log" ]; then
    echo "📊 最新训练日志 (最后10行):"
    echo "----------------------------------------"
    tail -10 training.log
    echo "----------------------------------------"
    echo "日志文件大小: $(du -h training.log | cut -f1)"
else
    echo "⚠️  未找到训练日志文件"
fi

echo
echo "💡 有用的命令:"
echo "  实时查看日志: tail -f training.log"
echo "  查看完整日志: less training.log"
echo "  停止训练: kill \$(cat training.pid)"
echo "  重新连接tmux: tmux attach -t gnn_training"
