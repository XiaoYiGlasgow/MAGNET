"""Official nuPlan closed-loop score extractor (rebuild of the lost stage-5 tool).

Reads ``aggregator_metric/*weighted_average*.parquet`` under every experiment tag in a
group directory and prints the final score plus the six headline sub-metrics, together
with a per-``scenario_type`` breakdown.

The stage-5 harness lived in ``scratchpad/`` and no longer exists, so the Boston-250
numbers (0.122 / 0.095) currently have no reproducible artefact trail. This module is
the auditable replacement; keep it under version control.

Usage::

    python extract_scores.py <group_dir> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Official aggregator column -> short label used in the work records.
SUB_METRICS: Dict[str, str] = {
    "drivable_area_compliance": "drivable",
    "no_ego_at_fault_collisions": "collision",
    "ego_progress_along_expert_route": "progress",
    "ego_is_making_progress": "making_prog",
    "ego_is_comfortable": "comfort",
    "time_to_collision_within_bound": "ttc",
}
SCORE_COLUMN = "score"
FINAL_ROW_MARKER = "final_score"


def find_aggregator_files(group_dir: Path) -> Dict[str, Path]:
    """Map experiment tag -> newest weighted-average parquet under that tag."""
    found: Dict[str, Path] = {}
    for parquet in sorted(group_dir.rglob("*weighted_average*.parquet")):
        try:
            tag = parquet.relative_to(group_dir).parts[0]
        except ValueError:  # pragma: no cover - rglob keeps paths inside group_dir
            continue
        prev = found.get(tag)
        if prev is None or parquet.stat().st_mtime > prev.stat().st_mtime:
            found[tag] = parquet
    return found


def _final_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the aggregate row, which nuPlan marks with scenario_type=final_score."""
    if "scenario_type" not in df.columns:
        return None
    rows = df[df["scenario_type"] == FINAL_ROW_MARKER]
    return rows.iloc[0] if len(rows) else None


def summarise_tag(tag: str, parquet: Path) -> Optional[Dict[str, object]]:
    try:
        df = pd.read_parquet(parquet)
    except (OSError, ValueError) as exc:
        logger.error("Cannot read %s: %s", parquet, exc)
        return None

    row = _final_row(df)
    if row is None:
        logger.warning("%s: no '%s' row in %s", tag, FINAL_ROW_MARKER, parquet.name)
        return None

    out: Dict[str, object] = {"tag": tag, "final_score": row.get(SCORE_COLUMN)}
    for column, label in SUB_METRICS.items():
        out[label] = row.get(column)
    n_rows = df[df["scenario_type"] != FINAL_ROW_MARKER]
    out["n_scenario_types"] = int(n_rows["scenario_type"].nunique()) if len(n_rows) else 0
    return out


def breakdown_by_type(tag: str, parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    if "scenario_type" not in df.columns:
        return pd.DataFrame()
    per_type = df[df["scenario_type"] != FINAL_ROW_MARKER].copy()
    keep = ["scenario_type", SCORE_COLUMN] + [c for c in SUB_METRICS if c in per_type.columns]
    per_type = per_type[[c for c in keep if c in per_type.columns]]
    per_type = per_type.rename(columns={SCORE_COLUMN: tag})
    return per_type.set_index("scenario_type")[[tag]]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_dir", type=Path, help="Directory containing experiment tags")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.group_dir.is_dir():
        logger.error("Not a directory: %s", args.group_dir)
        return 1

    files = find_aggregator_files(args.group_dir)
    if not files:
        logger.error("No *weighted_average*.parquet found under %s", args.group_dir)
        return 1

    summaries = [s for s in (summarise_tag(t, p) for t, p in sorted(files.items())) if s]
    if not summaries:
        logger.error("No aggregate rows recovered")
        return 1

    summary_df = pd.DataFrame(summaries).set_index("tag")
    pd.set_option("display.width", 200)
    print("\n=== final score + sub-metrics ===")
    print(summary_df.round(3).to_string())

    frames = [breakdown_by_type(tag, path) for tag, path in sorted(files.items())]
    frames = [f for f in frames if not f.empty]
    if frames:
        print("\n=== per scenario_type (final score) ===")
        print(pd.concat(frames, axis=1).round(3).to_string())

    if args.csv:
        summary_df.to_csv(args.csv)
        logger.info("Wrote %s", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
