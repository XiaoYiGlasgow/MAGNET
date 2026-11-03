# GNN_Trans Documentation Index

> The current workspace is organised as two project lines:
> - **Project A**: trajectory prediction / baseline / planning
> - **Project B**: danger scenario generation / diffusion / danger visualization

---

## First-time reading path

1. **Main entry** — [tutorials/README.md](tutorials/README.md)
2. **Project A** — [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) → [tutorials/02_trajectory_prediction.md](tutorials/02_trajectory_prediction.md)
3. **Project B** — [DANGER_GENERATION.md](DANGER_GENERATION.md) → [tutorials/03_danger_scene_generation.md](tutorials/03_danger_scene_generation.md)
4. **Results and analysis**
   - Project A conclusions: [reports/summary.md](reports/summary.md)
   - Project A analysis details: [experiments/analysis/gnn_trans_ablation/analysis/analysis.md](../experiments/analysis/gnn_trans_ablation/analysis/analysis.md)
   - Project B best results: see [DANGER_GENERATION.md](DANGER_GENERATION.md) §Current best results

---

## Documentation sections

| Directory/File | Project | Purpose |
|-----------|---------|------|
| [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) | A | Main entry navigation for Project A |
| [DANGER_GENERATION.md](DANGER_GENERATION.md) | B | Main entry navigation for Project B |
| [tutorials/](tutorials/) | A+B | Step-by-step reading for first-time use |
| [reference/](reference/) | A+B | Technical reference and troubleshooting |
| [reports/](reports/) | A+B | Result summaries and presentation materials |
| [work_records/](work_records/) | A+B | Original NuPlan documentation, progress logs, paper materials |
| [archive/](archive/) | A+B | Archived old records |

---

## reference/ — Technical reference

| File | Project | Content |
|------|---------|------|
| [danger_score_documentation.md](reference/danger_score_documentation.md) | B | Principles and interface of the danger scoring module |
| [diffusion_generator_documentation.md](reference/diffusion_generator_documentation.md) | B | Overall design of the Diffusion generator |
| [visualization_workflow_guide.md](reference/visualization_workflow_guide.md) | B | Visualisation troubleshooting and quick code-location lookup |

## reports/ — Results and presentations

| File | Project | Content |
|------|---------|------|
| [summary.md](reports/summary.md) | A | Overall results summary for Project A (paper writing draft) |
| [results_analysis_report.md](reports/results_analysis_report.md) | A | Project A interim full report (complete statistical procedure) |
| [danger_scene_generation_talk.md](reports/danger_scene_generation_talk.md) | B | Project B presentation script |

## work_records/ — Progress and original documentation

| Path | Content |
|------|------|
| [work_records/README.md](work_records/README.md) | Original NuPlan README |
| [work_records/baselines.md](work_records/baselines.md) | NuPlan baseline description |
| [work_records/competition.md](work_records/competition.md) | NuPlan competition rules |
| [work_records/dataset_setup.md](work_records/dataset_setup.md) | NuPlan dataset download and configuration |
| [work_records/faq.md](work_records/faq.md) | NuPlan FAQ |
| [work_records/paper/](work_records/paper/) | Paper LaTeX source files |
| [work_records/archive/](work_records/archive/) | Daily progress logs (from 2026-04-22) |

---

## Other

- [folder_cleanup_plan.md](folder_cleanup_plan.md) — Folder reorganisation plan and directory structure description
- [archive/](archive/) — Archived old records (IDM/LSTM editing records, old task plans)

---

## Archive

Historical development logs and completed plan documents are in [archive/](archive/).
