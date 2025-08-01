#!/bin/bash

# 每日代码更新脚本
# 使用方法: ./daily_update.sh "你的提交信息"

# 进入项目根目录
cd /mnt/local/Xiao_Yi

# 获取当前日期
DATE=$(date +"%Y-%m-%d")

# 如果用户提供了提交信息，使用用户提供的，否则使用默认信息
if [ $# -eq 0 ]; then
    COMMIT_MSG="Daily update - $DATE"
else
    COMMIT_MSG="$1 - $DATE"
fi

echo "正在更新代码到GitHub..."
echo "提交信息: $COMMIT_MSG"

# 添加所有更改的文件
git add .

# 检查是否有文件需要提交
if git diff --staged --quiet; then
    echo "没有文件需要提交，代码已是最新状态。"
    exit 0
fi

# 提交更改
git commit -m "$COMMIT_MSG"

# 推送到GitHub
if git push; then
    echo "✅ 代码已成功更新到GitHub!"
    echo "仓库地址: https://github.com/XiaoYiGlasgow/GNN_Trans_Private.git"
else
    echo "❌ 推送失败，请检查网络连接或权限设置"
    exit 1
fi
