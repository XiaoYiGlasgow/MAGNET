#!/bin/bash
# 快速打开可视化报告

REPORT_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/GNN_Trans_params/visualization_report_20250806_170528"
HTML_FILE="$REPORT_DIR/evaluation_report.html"

if [ -f "$HTML_FILE" ]; then
    echo "打开可视化报告..."
    xdg-open "$HTML_FILE"
else
    echo "报告文件不存在: $HTML_FILE"
fi
