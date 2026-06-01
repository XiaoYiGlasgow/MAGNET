# GNN_Trans 文档索引

> 当前工作区按两条项目线理解：
> - **项目 A**：轨迹预测 / baseline / planning
> - **项目 B**：危险场景生成 / diffusion / danger visualization

---

## 首次阅读路线

1. **总入口** — [tutorials/README.md](tutorials/README.md)
2. **项目 A** — [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) → [tutorials/02_trajectory_prediction.md](tutorials/02_trajectory_prediction.md)
3. **项目 B** — [DANGER_GENERATION.md](DANGER_GENERATION.md) → [tutorials/03_danger_scene_generation.md](tutorials/03_danger_scene_generation.md)
4. **结果与分析**
   - 项目 A 结论：[reports/summary.md](reports/summary.md)
   - 项目 A 分析细节：[experiments/analysis/gnn_trans_ablation/analysis/analysis.md](../experiments/analysis/gnn_trans_ablation/analysis/analysis.md)
   - 项目 B 最优结果：见 [DANGER_GENERATION.md](DANGER_GENERATION.md) §当前最优结果

---

## 文档分区

| 目录/文件 | 适用项目 | 用途 |
|-----------|---------|------|
| [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) | A | 项目 A 主入口导航 |
| [DANGER_GENERATION.md](DANGER_GENERATION.md) | B | 项目 B 主入口导航 |
| [tutorials/](tutorials/) | A+B | 首次使用按步骤阅读 |
| [reference/](reference/) | A+B | 技术参考与排错 |
| [reports/](reports/) | A+B | 结果总结、汇报材料 |
| [work_records/](work_records/) | A+B | NuPlan 原版文档、进度日志、论文材料 |
| [archive/](archive/) | A+B | 已归档的旧记录 |

---

## reference/ — 技术参考

| 文件 | 适用项目 | 内容 |
|------|---------|------|
| [danger_score_documentation.md](reference/danger_score_documentation.md) | B | 危险评分模块原理与接口 |
| [diffusion_generator_documentation.md](reference/diffusion_generator_documentation.md) | B | Diffusion 生成器整体设计 |
| [visualization_workflow_guide.md](reference/visualization_workflow_guide.md) | B | 可视化排错与代码位置速查 |

## reports/ — 结果与汇报

| 文件 | 适用项目 | 内容 |
|------|---------|------|
| [summary.md](reports/summary.md) | A | 项目 A 结果总汇总（论文写作草稿） |
| [results_analysis_report.md](reports/results_analysis_report.md) | A | 项目 A 阶段性大报告（完整统计过程） |
| [danger_scene_generation_talk.md](reports/danger_scene_generation_talk.md) | B | 项目 B 汇报讲稿 |

## work_records/ — 进度与原版文档

| 路径 | 内容 |
|------|------|
| [work_records/README.md](work_records/README.md) | NuPlan 原版 README |
| [work_records/baselines.md](work_records/baselines.md) | NuPlan baseline 说明 |
| [work_records/competition.md](work_records/competition.md) | NuPlan 竞赛规则 |
| [work_records/dataset_setup.md](work_records/dataset_setup.md) | NuPlan 数据集下载与配置 |
| [work_records/faq.md](work_records/faq.md) | NuPlan 常见问题 |
| [work_records/paper/](work_records/paper/) | 论文 LaTeX 源文件 |
| [work_records/archive/](work_records/archive/) | 每日进度日志（2026-04-22 起） |

---

## 其他

- [folder_cleanup_plan.md](folder_cleanup_plan.md) — 文件夹整理计划与目录结构说明
- [archive/](archive/) — 已归档的旧记录（IDM/LSTM 编辑记录、旧任务计划）

---

## 归档

历史开发日志和已完成的计划文档见 [archive/](archive/)。
