"""G13 地板效应判定用：逐场景分数分布 + 事件数统计（汇报 §8.4 口径）。

先报**事件数**再报分数：独立 log 数 / 场景数 / collision 与 offroad 阳性事件数 /
每个 planner 的有效样本量。分数分布同时报 `==0.000` 的场景数 —— 双峰即地板效应信号。

用法：
    python3 scripts/eval_closed_loop/g13_event_counts.py <group_dir> [--csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _latest_run(exp_dir: Path) -> Path | None:
    cands = sorted(exp_dir.glob("*/*/aggregator_metric"))
    return cands[-1].parent if cands else None


def _read_metric(run_dir: Path, name: str) -> pd.DataFrame | None:
    p = run_dir / "metrics" / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def analyse(group: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for exp_dir in sorted(group.iterdir()):
        if not exp_dir.is_dir():
            continue
        run_dir = _latest_run(exp_dir)
        if run_dir is None:
            continue

        agg_files = list((run_dir / "aggregator_metric").glob("*weighted_average*.parquet"))
        if not agg_files:
            continue
        agg = pd.read_parquet(agg_files[0])
        # 逐场景行：scenario 非空且非 "final_score" 汇总行
        per_scene = agg[agg["scenario"].notna() & (agg["scenario"] != "final_score")]

        score_col = "score" if "score" in per_scene.columns else per_scene.columns[-1]
        scores = pd.to_numeric(per_scene[score_col], errors="coerce").dropna()

        row: Dict = {
            "tag": exp_dir.name,
            "n_scenario": int(len(scores)),
            "n_log": int(per_scene["log_name"].nunique()) if "log_name" in per_scene else -1,
            "n_zero": int((scores <= 1e-9).sum()),
            "pct_zero": round(float((scores <= 1e-9).mean() * 100), 1) if len(scores) else float("nan"),
            "mean": round(float(scores.mean()), 4) if len(scores) else float("nan"),
            "median": round(float(scores.median()), 4) if len(scores) else float("nan"),
            "p75": round(float(scores.quantile(0.75)), 4) if len(scores) else float("nan"),
            "max": round(float(scores.max()), 4) if len(scores) else float("nan"),
        }

        # 阳性事件数：collision / offroad（指标 statistics 里 0 = 违规）
        for metric, key in (("no_ego_at_fault_collisions", "n_collision_scene"),
                            ("drivable_area_compliance", "n_offroad_scene")):
            df = _read_metric(run_dir, metric)
            if df is None or "metric_score" not in df.columns:
                row[key] = -1
                continue
            per = df.groupby("scenario_name")["metric_score"].min()
            row[key] = int((per < 1.0).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("tag")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    out = analyse(args.group)
    if out.empty:
        print("未找到任何 aggregator_metric，检查 group 目录", file=sys.stderr)
        return 1
    print("=" * 110)
    print(" G13 事件数与逐场景分数分布（先报事件数，再报分数 —— 汇报 §8.4）")
    print("=" * 110)
    print(out.to_string())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv)
        print(f"\nwritten: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
