# 教程 04：Claude 常用提示词模板

这份文档给 `GNN_Trans` 项目提供一套可直接复用的 Claude 提示词模板。

目标只有两个：

1. 让 Claude 更稳定地调用你已经安装的 skills
2. 让每次任务都带上你关心的约束：先检查、分步修改、最小验证、中文记录

---

## 1. 先记住这几个 skill

你当前在 `/mnt/local/Xiao_Yi/.claude/skills` 下已经安装的高频 skill 包括：

- `gnn-trans-danger-model`：项目上下文专用，适合 danger generation、trajectory prediction、NuPlan 仿真、LSTM baseline、可视化排查
- `bug-detective`：定位 bug、异常、报错根因
- `daily-coding`：小步改代码、补实现、重构局部逻辑
- `verification-loop`：改完后做最小验证
- `results-analysis`：分析实验结果、比较模型、生成结论
- `results-report`：写实验总结、复盘、阶段报告
- `progress-log-update`：追加白天过程记录，适合开发修改和研究进展
- `progress-log-organize`：整理当天日志，刷新 `## 日终整理（供快速读取）`
- `progress-log-read`：恢复前情，查看之前做到哪
- `readme-updater`：同步更新说明文档
- `webapp-testing`：排查 NuBoard 或页面交互问题
- `codebase-memory-mcp`：Code Graph skill，适合符号查找、调用链追踪、blast radius 分析，当前已做项目级接入

---

## 2. 通用总模板

当你不想每次从头组织提示词时，先用这个总模板：

```text
这是 /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans 项目。

先使用 gnn-trans-danger-model 理解项目上下文。
如果是报错或异常，先用 bug-detective 定位根因，不要直接改。
如果需要改代码，用 daily-coding 小步修改。
改完必须用 verification-loop 做最小验证。
最后用 progress-log-update 把修改、验证结果和后续建议写入工作记录。
如果今天工作结束，再用 progress-log-organize 刷新日终摘要。

要求：
1. 全程使用中文
2. 先检查再下结论
3. 不要混淆“轨迹预测”和“危险场景生成”两条研究线
4. 优先给出实际文件、命令、日志和输出路径
```

---

## 3. 调 bug 模板

适合报错、结果异常、输出不符合预期、训练不收敛、仿真跑飞等场景。

```text
请用 gnn-trans-danger-model + bug-detective 检查这个问题。

问题现象：
<把报错、异常输出、日志片段贴在这里>

要求：
1. 先定位根因，再决定是否改代码
2. 明确指出问题属于轨迹预测还是危险场景生成
3. 如果涉及多个可能原因，按概率从高到低排序
4. 如果要改代码，用 daily-coding 小步修改
5. 改完用 verification-loop 做最小验证
6. 先用 progress-log-update 追加中文工作记录
7. 如果今天准备收工，再用 progress-log-organize 整理日终摘要
```

简化版：

```text
用 gnn-trans-danger-model 和 bug-detective 看这个报错，先定位根因，不要直接大改。修复后做最小验证，用 progress-log-update 记录过程；如果今天结束，再用 progress-log-organize 整理摘要。
```

---

## 4. 小步改代码模板

适合你已经知道要改什么，只希望 Claude 稳定执行。

```text
请用 gnn-trans-danger-model + daily-coding 完成这次修改。

修改目标：
<写清楚要改的函数、脚本或行为>

执行要求：
1. 先读相关文件，确认现有实现
2. 只做和目标直接相关的小步修改
3. 不要顺手重构无关代码
4. 改完用 verification-loop 做最小验证
5. 把变更摘要和验证结果写入 progress-log-update
6. 如果这是今天最后一项工作，再用 progress-log-organize 刷新日终整理
```

如果你想强调“先给计划再动手”，可以直接这样说：

```text
先给一个短的可执行计划，再按步骤修改代码。改完必须验证，并更新工作记录。
```

---

## 5. 跑实验与结果分析模板

适合比较 GNN_Trans、LSTM、IDM，或者分析 danger generation 的训练/评估输出。

```text
请用 gnn-trans-danger-model + results-analysis 分析这批实验结果。

输入材料：
- 日志路径：<log 路径>
- 结果目录：<result 路径>
- 对比对象：<例如 GNN_Trans / LSTM / IDM>

要求：
1. 先说明这些结果属于哪条研究线
2. 提取关键指标并做表格对比
3. 区分“观察到的事实”和“你的推断”
4. 如果结论不稳，明确指出证据不足
5. 最后给出下一步最值得做的 1 到 3 件事
6. 如需沉淀成文档，再用 results-report 输出中文总结
```

简化版：

```text
用 gnn-trans-danger-model 和 results-analysis 看这批结果，给我一个中文结论，区分事实和推断，并指出下一步最值得做什么。
```

---

## 6. 写实验总结模板

适合把已完成的实验整理成报告、阶段复盘或汇报稿。

```text
请用 gnn-trans-danger-model + results-report 写一份中文实验总结。

材料来源：
- 分析结论：<analysis 文档或聊天结论>
- 日志路径：<log 路径>
- 结果目录：<result 路径>

要求：
1. 先写本次实验做了什么
2. 再写关键结果和最重要的结论
3. 然后写存在的问题和风险
4. 最后写下一步行动建议
5. 所有结论都尽量绑定到实际日志、指标或文件
```

---

## 7. 更新 README 或教程模板

适合代码结构变化后同步说明文档。

```text
请用 gnn-trans-danger-model + readme-updater 更新相关文档。

目标文件：
<例如 docs/README.md 或 docs/tutorials/03_danger_scene_generation.md>

要求：
1. 先检查当前代码和文档是否一致
2. 只更新已经被代码或目录结构确认的内容
3. 不要写未经验证的承诺性描述
4. 保持中文、直接、可执行
5. 改完说明你核对了哪些文件
```

---

## 8. 恢复上下文模板

适合你隔了一天回来，先想知道“上次做到哪了”。

```text
请用 progress-log-read + gnn-trans-danger-model 恢复这个项目的上下文。

我现在最关心的是：
<例如 danger generation 训练进展 / LSTM baseline / NuBoard 可视化>

要求：
1. 先总结最近一次工作做到哪
2. 指出当前相关文件、结果目录和待办事项
3. 如果有正在运行或未完成的实验，明确说明
4. 输出使用中文
```

---

## 9. 工作记录怎么配合用

这两个 skill 现在分工是固定的，不建议混着用：

- `progress-log-update`：白天随做随记，负责**追加过程记录**
- `progress-log-organize`：一天结束时使用，负责**整理并覆盖 `## 日终整理（供快速读取）`**

你可以把它理解成：

- `update` = 写过程
- `organize` = 写摘要

最稳的用法是：

```text
白天每完成一项明确工作，就用 progress-log-update 追加记录。
当天准备结束时，再用 progress-log-organize 提炼今日核心成果、当前状态、关键结果、明日优先任务和风险提醒。
```

如果你要显式指定日志风格，可以直接这样说：

```text
更新工作记录时，请用 progress-log-update：
1. 如果是改代码，用“修改文件/修改内容/原因/预期效果”格式
2. 如果是实验分析或方法判断，用“当前结论/关键结果或指标/相关文件或脚本/下一步”格式
3. 不要直接改日终整理区

今天结束后，再用 progress-log-organize 刷新日终整理区。
```

---

## 10. NuBoard / 页面问题模板

适合“页面能开但行为不对”“一直转圈”“按钮没反应”这类问题。

```text
请用 gnn-trans-danger-model + webapp-testing 排查这个 NuBoard / 页面问题。

现象：
<描述页面现象>

要求：
1. 先确认运行入口、输出 bundle、页面实际行为
2. 检查控制台报错、请求状态和关键交互
3. 区分“后端数据没生成”和“前端渲染异常”
4. 如果需要改代码，小步修改后做最小验证
5. 最后更新工作记录
```

---

## 11. 推荐使用习惯

为了让 Claude 在这个项目里更稳定，建议你每次提示词都尽量包含下面 4 类信息：

1. **任务类型**：调 bug、改代码、分析结果、写文档
2. **研究线**：轨迹预测，还是危险场景生成
3. **证据位置**：报错文本、日志路径、结果目录、目标文件
4. **执行约束**：先检查、分步修改、最小验证、中文记录

最常用的一句短模板是：

```text
这是 GNN_Trans 项目。先用 gnn-trans-danger-model 理解上下文；如果是异常先用 bug-detective 定位根因；需要改代码时用 daily-coding 小步修改；改完用 verification-loop 验证；最后用 progress-log-update 记录。如果今天工作结束，再用 progress-log-organize 整理日终摘要。
```

这句话已经足够覆盖大多数日常开发任务。

---

## 12. codebase-memory-mcp / Code Graph 怎么用

### 当前状态

你现在已经安装了这个 skill：

- 目录：`/mnt/local/Xiao_Yi/.claude/skills/codebase-memory-mcp`
- 来源：`alinaqi/claude-bootstrap` 的 `skills/code-graph`

现在这套项目级接入已经补齐了：

- skill 已安装
- `GNN_Trans/.mcp.json` 已配置
- 本机 `codebase-memory-mcp` 二进制已放到 `/.claude/bin`

也就是说，当前已经不是“只有说明书”的状态了，而是已经具备了让 Claude 调用 code graph MCP 工具的基础条件。

### 已完成的接入

- 项目配置：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/.mcp.json`
- 本地二进制：`/mnt/local/Xiao_Yi/.claude/bin/codebase-memory-mcp`
- 本地包装脚本：`~/.claude/install-graph-tools.sh`

### 本次新增文件范围

为了保证不影响你原来的项目逻辑，这次只新增或追加了这些文件：

- 新增：`/mnt/local/Xiao_Yi/.claude/bin/codebase-memory-mcp`
- 新增：`/mnt/local/Xiao_Yi/.claude/install-graph-tools.sh`
- 新增：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/.mcp.json`
- 追加一行：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/.gitignore` 中的 `.code-graph/`

不会修改你的训练脚本、模型、数据、实验结果和教程结构。

### 它适合做什么

这个 skill 适合下面这些任务：

- 查一个函数、类、方法定义在哪里
- 看谁调用了某个函数
- 查一个改动的影响范围
- 快速理解模块依赖关系
- 在大仓库里少读文件、先查结构

对 `GNN_Trans` 来说，尤其适合：

- 从入口脚本追到 `src/` 里的真实实现
- 区分 trajectory prediction 和 danger generation 两条线的调用边界
- 查 `train_diffusion_gnn_trans.py`、`generate_danger_scenes.py`、`run_nuboard_simulation.py` 这类入口会连到哪些模块
- 在改共享工具函数前先看 blast radius

### 先用它做什么，不要用它做什么

优先用 `codebase-memory-mcp` 的场景：

- 先找入口函数、类、方法，而不是先通读大文件
- 先查调用链和依赖，再决定要不要改共享代码
- 先查影响范围，再动 `utils/`、`planner/`、`visualization/` 这类公共位置
- 先看项目结构，再决定精读哪个目录

不必强行用它的场景：

- 你只是想看一段日志、报错文本、配置常量
- 你已经明确知道就要改某一个文件里的几行代码
- 问题主要是数据内容、实验结果、图表，而不是代码结构

### 高价值问法

先给一条总原则：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查代码图，不要一上来就通读文件。先定位相关符号、调用链和影响范围，再决定读哪些文件。
```

下面这些是更贴近你日常工作的高频问法。

#### 1. 查训练主流程

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查 `train_diffusion_gnn_trans.py` 对应的关键函数入口、调用链和依赖模块，再告诉我训练主流程是怎么串起来的。
```

#### 2. 查危险场景生成入口

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查 `generate_danger_scenes.py` 会调用哪些 danger generation 模块，并标出真正负责 SDEdit 推理的函数。
```

#### 3. 查 NuBoard / planner 调用链

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查 `run_nuboard_simulation.py` 到 planner / visualization 的调用链，再决定要读哪些文件。
```

#### 4. 查 danger score 影响范围

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 分析如果我要修改 `danger score` 相关逻辑，哪些脚本、模块和评估流程会受影响。
```

#### 5. 查某个类/函数到底被谁用

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查 `<函数名或类名>` 的定义位置、直接调用者、间接调用链，再告诉我哪些入口脚本最终会走到这里。
```

#### 6. 改共享代码前先查 blast radius

```text
这是 GNN_Trans 项目。改这个函数之前，先用 codebase-memory-mcp 查它的 blast radius，告诉我会影响哪些模块、脚本和评估流程，再决定是否值得改。
```

#### 7. 做重构前先盘结构

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 看这个目录或模块的结构，帮我区分入口文件、核心实现、工具模块和可能受影响的公共代码。
```

#### 8. 查两条研究线的边界

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 帮我区分 trajectory prediction 和 danger generation 两条研究线分别经过哪些入口、核心模块和共享模块。
```

#### 9. 查某个 bug 涉及哪条链路

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查这次报错对应函数所在的调用链，告诉我它属于哪条研究线、从哪个入口进来、最可能在哪一层出问题。
```

#### 10. 查改动后该重点回归什么

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 看这次改动影响到了哪些调用链，然后告诉我最应该回归验证的脚本、评估和可视化入口。
```

### 和其他 skill 怎么配合

推荐搭配方式：

- `codebase-memory-mcp + gnn-trans-danger-model`
  - 先用图查询找结构，再用项目 skill 判断哪条研究线、哪些路径重要
- `codebase-memory-mcp + bug-detective`
  - 先查调用链和影响范围，再定位 bug 根因
- `codebase-memory-mcp + daily-coding`
  - 先查 blast radius，再做小步修改

基础组合模板：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查代码图理解结构，再用 gnn-trans-danger-model 结合项目上下文分析；如果确认要改，再用 daily-coding 小步修改，并用 verification-loop 验证。
```

### 推荐组合模板

调试类：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查代码路径和调用链，再用 bug-detective 定位根因；如果确认要改，再小步修改并做最小验证。
```

重构类：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查这个改动的 blast radius，再用 daily-coding 小步修改，不要顺手改无关代码，改完做最小验证。
```

理解代码类：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查结构，再告诉我这段功能属于哪条研究线、入口在哪、核心实现在哪、共享模块有哪些。
```

### 首次验证怎么问

重启 Claude 后，在 `GNN_Trans` 项目里可以先试这句：

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查 `train_diffusion_gnn_trans.py` 对应的关键入口和调用链，再告诉我训练主流程经过了哪些模块。
```

如果 Claude 能真的调用 code graph 工具，它应该优先走图查询，而不是一上来通读很多文件。

如果你想保守一点，也可以这样说：

```text
这是 GNN_Trans 项目。如果 codebase-memory-mcp 工具可用，就先查代码图；如果当前不可用，再退回到只读最关键入口文件。
```

---

## 12. 1 行提示词速查表

下面这些都是可以直接复制的短句版本，适合你日常快速发给 Claude。

### 调 bug

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + bug-detective 查这个问题，先定位根因，再决定是否改代码，修完后做最小验证并更新中文工作记录。
```

### 小步改代码

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + daily-coding 小步完成这次修改，不要改无关代码，改完用 verification-loop 验证，并更新工作记录。
```

### 先计划再改

```text
这是 GNN_Trans 项目，先给一个短的可执行计划，再用 daily-coding 按步骤修改，改完必须验证，并更新中文记录。
```

### 看实验结果

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + results-analysis 看这批结果，区分事实和推断，给出中文结论和下一步最值得做的事。
```

### 写实验总结

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + results-report 把这次实验写成中文总结，包含做了什么、结果、问题和下一步。
```

### 更新 README / 教程

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + readme-updater 同步更新文档，只写已经被代码和目录结构确认的内容。
```

### 恢复上下文

```text
这是 GNN_Trans 项目，用 progress-log-read + gnn-trans-danger-model 告诉我上次做到哪了、现在相关文件和结果目录在哪里、下一步该做什么。
```

### 排查 NuBoard / 页面

```text
这是 GNN_Trans 项目，用 gnn-trans-danger-model + webapp-testing 排查这个 NuBoard 或页面问题，先确认运行入口、bundle、控制台报错和实际页面行为。
```

### 训练/评估前先检查

```text
这是 GNN_Trans 项目，在运行训练或评估前先帮我检查环境、工作目录、模型路径、数据路径和输出目录，确认无误后再执行。
```

### 先查代码图再读文件

```text
这是 GNN_Trans 项目。先用 codebase-memory-mcp 查代码图，定位入口、调用链和影响范围，再决定读哪些文件。
```

### 统一万能版

```text
这是 GNN_Trans 项目。先用 gnn-trans-danger-model 理解上下文；异常先用 bug-detective；改代码用 daily-coding；改完用 verification-loop；最后用 progress-log-update 写中文过程记录。如果今天工作结束，再用 progress-log-organize 整理日终摘要。
```

---

## 13. 我常用的 Claude 提示词

这一节不是通用模板，而是更贴近你在 `GNN_Trans` 里最常见的工作方式。可以直接复制，再把路径、报错或目标替换掉。

### 先看上下文，再动手

```text
这是 GNN_Trans 项目。先告诉我这件事属于 trajectory prediction 还是 danger generation，再列出最相关的入口文件、核心实现和结果目录，然后再决定下一步。
```

### 先检查再改

```text
这是 GNN_Trans 项目。先检查现状，不要直接改。先告诉我问题最可能在哪、证据是什么、要改哪些文件，再开始动手。
```

### 先计划再执行

```text
这是 GNN_Trans 项目。先给一个短的可执行计划，每一步都写清楚要看什么文件、改什么内容、怎么验证；然后按步骤执行。
```

### 改完必须验证

```text
这是 GNN_Trans 项目。改完不要只给我 diff，必须告诉我你做了什么最小验证、结果是什么、还有什么没验证到。
```

### 只做小步改动

```text
这是 GNN_Trans 项目。只做和当前目标直接相关的小步修改，不要顺手重构无关代码，不要扩大改动范围。
```

### 绑定真实证据

```text
这是 GNN_Trans 项目。优先引用真实文件、日志、命令输出和结果路径，不要只给抽象判断。
```

### 区分事实和推断

```text
这是 GNN_Trans 项目。回答时把“已经确认的事实”和“你的推断”分开写，特别是实验结果分析和 bug 根因判断。
```

### 结果写回工作记录

```text
这是 GNN_Trans 项目。完成后把修改内容、验证结果和下一步建议写回中文工作记录。
```

### 适合你日常的万能组合

```text
这是 GNN_Trans 项目。先用 gnn-trans-danger-model 理解上下文；如果结构复杂就先用 codebase-memory-mcp 查调用链；如果是异常先用 bug-detective；需要改代码时用 daily-coding 小步修改；改完用 verification-loop 验证；最后更新中文工作记录。
```
