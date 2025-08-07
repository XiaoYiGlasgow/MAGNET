#!/bin/bash

# 每日GitHub更新脚本
# 自动提交和推送GNN_Trans项目的更新

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_PATH="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"

echo -e "${GREEN}=== 每日GitHub更新脚本 ===${NC}"
echo "项目路径: $PROJECT_PATH"
echo "开始时间: $(date)"
echo ""

# 检查项目路径是否存在
if [ ! -d "$PROJECT_PATH" ]; then
    echo -e "${RED}错误: 项目路径不存在: $PROJECT_PATH${NC}"
    exit 1
fi

# 进入项目目录
cd "$PROJECT_PATH" || exit 1

# 检查是否是git仓库
if [ ! -d ".git" ]; then
    echo -e "${RED}错误: 当前目录不是Git仓库${NC}"
    exit 1
fi

# 获取当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}当前分支: $CURRENT_BRANCH${NC}"

# 检查Git状态
echo -e "${YELLOW}检查Git状态...${NC}"
git status --porcelain > /tmp/git_status.txt

if [ ! -s /tmp/git_status.txt ]; then
    echo -e "${GREEN}没有新的更改需要提交${NC}"
    echo "最后一次提交: $(git log -1 --format='%h - %s (%cr)')"
    exit 0
fi

echo -e "${YELLOW}发现以下更改:${NC}"
git status --short

# 添加所有更改
echo -e "${YELLOW}添加所有更改到暂存区...${NC}"
git add .

# 生成提交信息
DATE=$(date '+%Y-%m-%d')
TIME=$(date '+%H:%M:%S')
COMMIT_MSG="Daily update $DATE $TIME

- Auto-generated daily commit
- Updated training and evaluation results
- Code improvements and bug fixes"

echo -e "${YELLOW}准备提交信息:${NC}"
echo "$COMMIT_MSG"
echo ""

# 提交更改
echo -e "${YELLOW}提交更改...${NC}"
if git commit -m "$COMMIT_MSG"; then
    echo -e "${GREEN}✅ 提交成功${NC}"
else
    echo -e "${RED}❌ 提交失败${NC}"
    exit 1
fi

# 推送到远程仓库
echo -e "${YELLOW}推送到GitHub...${NC}"
if git push origin "$CURRENT_BRANCH"; then
    echo -e "${GREEN}✅ 推送成功${NC}"
    echo -e "${GREEN}🎉 每日更新完成!${NC}"
    echo "查看更新: https://github.com/XiaoYiGlasgow/GNN_Trans"
else
    echo -e "${RED}❌ 推送失败${NC}"
    echo "请检查网络连接和GitHub凭据"
    exit 1
fi

echo ""
echo "完成时间: $(date)"
echo -e "${GREEN}=== 脚本执行完毕 ===${NC}"
