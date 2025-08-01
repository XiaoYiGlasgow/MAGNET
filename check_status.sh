#!/bin/bash

# Git 状态查看脚本
echo "=== 检查文件颜色原因 ==="
cd /mnt/local/Xiao_Yi

echo "1. 未跟踪的文件（橙色）："
git ls-files --others --exclude-standard

echo ""
echo "2. 已修改但未暂存的文件（橙色）："
git diff --name-only

echo ""
echo "3. 已暂存但未提交的文件（绿色）："
git diff --staged --name-only

echo ""
echo "4. 最后一次提交的文件："
git show --name-only --pretty=format:

echo ""
echo "5. 远程仓库状态："
git remote show origin 2>/dev/null || echo "远程仓库连接可能有问题"
