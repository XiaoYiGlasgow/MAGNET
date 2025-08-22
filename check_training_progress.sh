#!/bin/bash

# Training Progress Monitor Script - Real-time Version
# Usage: ./check_training_progress.sh [config_number]

WORK_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
LOG_DIR="$WORK_DIR/batch_training_logs"

# If config number is specified, only show detailed info for that config
CONFIG_FILTER=$1

# Draw progress bar function
draw_progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percentage=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))

    printf "["
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d/%d (%d%%)" $current $total $percentage
}

# Extract batch progress from log files
get_batch_progress() {
    local log_file=$1
    # Look for tqdm progress indicators like "50%|█████     | 500/1000 [02:30<02:30, 3.33it/s]"
    local batch_info=$(tail -20 "$log_file" | grep -E "it/s\]|batch" | tail -1)
    if [[ $batch_info =~ ([0-9]+)/([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    else
        echo "0 0"
    fi
}

clear
echo "========================================"
echo "🔍 GNN_Trans Training Progress Monitor $(date)"
echo "========================================"

# Check if training processes are running
echo "📊 Process Status:"
PYTHON_PROCESSES=$(ps aux | grep "python train.py" | grep -v grep)
BATCH_PROCESSES=$(ps aux | grep "auto_train_all_configs" | grep -v grep)

if [ -n "$PYTHON_PROCESSES" ] || [ -n "$BATCH_PROCESSES" ]; then
    echo "✅ Training processes found"
    [ -n "$BATCH_PROCESSES" ] && echo "   Batch script running"
    [ -n "$PYTHON_PROCESSES" ] && echo "   Python training active"
else
    echo "❌ No training processes found"
fi

echo ""
echo "🖥️ GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")"

echo ""
echo "========================================"

# Show real-time training progress
for config_num in 1 2 3 4; do
    config_names=("" "default" "ego_only" "no_ego_road" "nearby_only")
    config_name=${config_names[$config_num]}

    # If config filter is specified, only show that config
    if [ -n "$CONFIG_FILTER" ] && [ "$CONFIG_FILTER" != "$config_num" ]; then
        continue
    fi

    CONFIG_LOG=$(ls -t "$LOG_DIR"/config_${config_num}_${config_name}_*.log 2>/dev/null | head -1)

    echo "🔧 Config $config_num ($config_name):"

    if [ -n "$CONFIG_LOG" ] && [ -f "$CONFIG_LOG" ]; then
        # Extract training information
        TOTAL_EPOCHS=$(grep -o "Total epochs: [0-9]\+" "$CONFIG_LOG" | tail -1 | grep -o "[0-9]\+" || echo "100")
        CURRENT_EPOCH=$(grep -o "Epoch [0-9]\+/" "$CONFIG_LOG" | tail -1 | grep -o "[0-9]\+" || echo "0")

        # Get batch progress
        read CURRENT_BATCH TOTAL_BATCH <<< $(get_batch_progress "$CONFIG_LOG")

        # Get latest metrics
        LATEST_LOSS=$(grep "Train Loss:" "$CONFIG_LOG" | tail -1 | grep -o "[0-9.]\+" | head -1)
        LATEST_VAL=$(grep "Val Loss:" "$CONFIG_LOG" | tail -1 | grep -o "[0-9.]\+" | head -1)
        LATEST_ADE=$(grep "ADE:" "$CONFIG_LOG" | tail -1 | grep -o "[0-9.]\+" | head -1)
        LATEST_FDE=$(grep "FDE:" "$CONFIG_LOG" | tail -1 | grep -o "[0-9.]\+" | head -1)

        if [ "$CURRENT_EPOCH" -gt 0 ]; then
            # Show epoch progress
            echo -n "   📈 Epoch Progress: "
            draw_progress_bar $CURRENT_EPOCH $TOTAL_EPOCHS
            echo ""

            # Show batch progress if available
            if [ "$TOTAL_BATCH" -gt 0 ]; then
                echo -n "   🔄 Batch Progress: "
                draw_progress_bar $CURRENT_BATCH $TOTAL_BATCH
                echo ""
            fi

            # Show metrics in compact format
            echo -n "   📊 Metrics: "
            [ -n "$LATEST_LOSS" ] && echo -n "Loss:$LATEST_LOSS "
            [ -n "$LATEST_VAL" ] && echo -n "Val:$LATEST_VAL "
            [ -n "$LATEST_ADE" ] && echo -n "ADE:${LATEST_ADE}m "
            [ -n "$LATEST_FDE" ] && echo -n "FDE:${LATEST_FDE}m"
            echo ""

            # Show latest activity
            LATEST_OUTPUT=$(tail -3 "$CONFIG_LOG" | grep -E "(Epoch|it/s)" | tail -1)
            if [ -n "$LATEST_OUTPUT" ]; then
                echo "   📋 Latest: $(echo "$LATEST_OUTPUT" | cut -c1-80)..."
            fi
        else
            echo "   ⏳ Initializing..."
        fi

        # Check status
        if grep -q "Training complete" "$CONFIG_LOG"; then
            echo "   ✅ COMPLETE"
        elif grep -q "Early stopping" "$CONFIG_LOG"; then
            echo "   🛑 EARLY STOP"
        elif [ "$CURRENT_EPOCH" -gt 0 ]; then
            echo "   🚀 TRAINING"
        fi
    else
        echo "   ❌ NOT STARTED"
    fi

    echo ""
done

echo "========================================"
echo "💡 Commands: watch -n 5 './check_training_progress.sh' | tail -f batch_training.out"
