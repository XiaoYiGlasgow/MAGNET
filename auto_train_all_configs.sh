#!/bin/bash

# 自动训练所有4种邻接矩阵配置的脚本
# 使用方法: nohup ./auto_train_all_configs.sh &

# 设置工作目录
WORK_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
LOG_DIR="$WORK_DIR/batch_training_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建日志目录
mkdir -p "$LOG_DIR"

# 主日志文件
MAIN_LOG="$LOG_DIR/batch_training_${TIMESTAMP}.log"

# 配置信息
declare -a CONFIGS=("1" "2" "3" "4")
declare -a CONFIG_NAMES=("default" "ego_only" "no_ego_road" "nearby_only")

echo "========================================" | tee -a "$MAIN_LOG"
echo "批量训练开始时间: $(date)" | tee -a "$MAIN_LOG"
echo "工作目录: $WORK_DIR" | tee -a "$MAIN_LOG"
echo "日志目录: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "将依次训练4种配置..." | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# 进入工作目录
cd "$WORK_DIR" || exit 1

# 循环训练每种配置
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    config_name="${CONFIG_NAMES[$i]}"

    echo "" | tee -a "$MAIN_LOG"
    echo "🚀 开始训练配置 $config ($config_name)" | tee -a "$MAIN_LOG"
    echo "开始时间: $(date)" | tee -a "$MAIN_LOG"

    # 为每个配置创建单独的日志文件
    CONFIG_LOG="$LOG_DIR/config_${config}_${config_name}_${TIMESTAMP}.log"

    # 运行训练，自动输入配置选择
    echo "正在运行: python train.py (自动选择配置 $config)" | tee -a "$MAIN_LOG"
    echo "$config" | python train.py > "$CONFIG_LOG" 2>&1

    # 检查训练结果
    if [ $? -eq 0 ]; then
        echo "✅ 配置 $config ($config_name) 训练成功完成!" | tee -a "$MAIN_LOG"
    else
        echo "❌ 配置 $config ($config_name) 训练失败!" | tee -a "$MAIN_LOG"
    fi

    echo "结束时间: $(date)" | tee -a "$MAIN_LOG"
    echo "详细日志: $CONFIG_LOG" | tee -a "$MAIN_LOG"

    # 短暂休息，避免系统负载过高
    if [ $i -lt $((${#CONFIGS[@]} - 1)) ]; then
        echo "休息10秒后开始下一个配置..." | tee -a "$MAIN_LOG"
        sleep 10
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "🎉 所有配置训练完成!" | tee -a "$MAIN_LOG"
echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# 生成训练结果摘要
echo "" | tee -a "$MAIN_LOG"
echo "📊 训练结果文件夹:" | tee -a "$MAIN_LOG"
ls -la training_* 2>/dev/null | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "查看进度命令:" | tee -a "$MAIN_LOG"
echo "tail -f $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "ls -la $LOG_DIR" | tee -a "$MAIN_LOG"
