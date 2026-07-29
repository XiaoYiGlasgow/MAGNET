"""Rebuild the Pittsburgh held-out closed-loop evaluation pool (stage-8 protocol).

Stage 8 defined the in-distribution closed-loop set as the Pittsburgh logs that E1 never
touched: ``train_pittsburgh`` (1560 DBs) minus the 600 logs used by the E1/E2 split, giving
a 960-DB held-out pool that is then screened for corrupt databases. The original scripts
(``step0_heldout_pgh.py``, ``pittsburgh_validate.py``) lived in a session scratchpad that no
longer exists, so this module rebuilds them under version control.

Two stages:

``--build``    write the held-out log list and assert zero overlap with the E1 logs.
``--validate`` deep-screen each held-out DB by instantiating the scenario builder and
               requesting the 14 nuPlan challenge scenario types, then symlink the clean
               databases into an evaluation directory.

Usage::

    python build_pittsburgh_heldout.py --build
    python build_pittsburgh_heldout.py --validate --limit 300 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SPLITS_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits")
PITTSBURGH_ROOT = SPLITS_ROOT / "train_pittsburgh"
MAPS_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps")
SPLIT_MANIFEST = GNN_ROOT / "data/processed_features_magnet_e1_full_split3/split3_manifest.json"
HARNESS_DIR = GNN_ROOT / "scripts/eval_closed_loop"

HELDOUT_LIST = HARNESS_DIR / "logs_pittsburgh_heldout.txt"
CLEAN_LIST = HARNESS_DIR / "logs_pittsburgh_heldout_clean.txt"
CLEAN_DIR = GNN_ROOT / "data/closed_loop_sets/pittsburgh_heldout_clean"

CHALLENGE_TYPES = [
    "starting_left_turn",
    "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "stopping_with_lead",
    "high_lateral_acceleration",
    "high_magnitude_speed",
    "low_magnitude_speed",
    "traversing_pickup_dropoff",
    "waiting_for_pedestrian_to_cross",
    "behind_long_vehicle",
    "stationary_in_traffic",
    "near_multiple_vehicles",
    "changing_lane",
    "following_lane_with_lead",
]


def load_e1_logs() -> List[str]:
    """All 600 log names consumed by the E1/E2 three-way split."""
    with open(SPLIT_MANIFEST) as handle:
        manifest = json.load(handle)
    logs = manifest["logs"]
    used = sorted(set(logs["train"]) | set(logs["val"]) | set(logs["test"]))
    if len(used) != 600:
        logger.warning("Expected 600 E1 logs, manifest gives %d", len(used))
    return used


def build_heldout() -> List[str]:
    all_logs = sorted(p.stem for p in PITTSBURGH_ROOT.glob("*.db"))
    used = set(load_e1_logs())
    heldout = [name for name in all_logs if name not in used]

    overlap = set(heldout) & used
    if overlap:  # hard assertion: this is the leakage red line from stage 8 §6
        raise AssertionError(f"held-out overlaps E1 logs: {sorted(overlap)[:5]}")

    HELDOUT_LIST.write_text("\n".join(heldout) + "\n")
    logger.info("total=%d  e1_used=%d  heldout=%d -> %s",
                len(all_logs), len(used), len(heldout), HELDOUT_LIST)
    return heldout


def _screen_one(log_name: str) -> Tuple[str, bool, str]:
    """Instantiate the scenario builder on a single DB and request challenge scenarios."""
    os.environ.setdefault("NUPLAN_DATA_ROOT", str(SPLITS_ROOT.parent))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(MAPS_ROOT))
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential

        builder = NuPlanScenarioBuilder(
            data_root=str(PITTSBURGH_ROOT),
            map_root=str(MAPS_ROOT),
            sensor_root=None,
            db_files=[str(PITTSBURGH_ROOT / f"{log_name}.db")],
            map_version="nuplan-maps-v1.0",
        )
        scenario_filter = ScenarioFilter(
            scenario_types=CHALLENGE_TYPES,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=1,
            limit_total_scenarios=None,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
        )
        scenarios = builder.get_scenarios(scenario_filter, Sequential())
        if not scenarios:
            # Not a corrupt database: this log simply holds no challenge-type scenario.
            return log_name, False, "no_challenge_scenarios"
        return log_name, True, f"ok n_scenarios={len(scenarios)}"
    except Exception as exc:  # noqa: BLE001 - screening must never abort the sweep
        return log_name, False, f"db_error {type(exc).__name__}: {exc}"[:180]


def validate(limit: Optional[int], workers: int) -> None:
    if not HELDOUT_LIST.exists():
        raise FileNotFoundError(f"run --build first: {HELDOUT_LIST} missing")
    candidates = [line.strip() for line in HELDOUT_LIST.read_text().splitlines() if line.strip()]
    logger.info("screening %d held-out DBs with %d workers (target clean=%s)",
                len(candidates), workers, limit or "all")

    clean: List[str] = []
    no_scenarios = 0
    db_errors: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_screen_one, name): name for name in candidates}
        for done, future in enumerate(as_completed(futures), 1):
            name, ok, detail = future.result()
            if ok:
                clean.append(name)
            elif detail.startswith("db_error"):
                db_errors.append(f"{name}: {detail}")
            else:
                no_scenarios += 1
            if done % 100 == 0:
                logger.info("  %d/%d screened, clean=%d no_scenarios=%d db_error=%d",
                            done, len(candidates), len(clean), no_scenarios, len(db_errors))
            if limit and len(clean) >= limit:
                logger.info("reached target clean=%d, stopping early", limit)
                break

    clean = sorted(clean)[: limit or len(clean)]
    CLEAN_LIST.write_text("\n".join(clean) + "\n")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name in clean:
        link = CLEAN_DIR / f"{name}.db"
        if not link.exists():
            link.symlink_to(PITTSBURGH_ROOT / f"{name}.db")
    # no_scenarios is normal filtering; db_error is the corruption signal stage 8 tracked.
    logger.info("clean=%d  no_challenge_scenarios=%d  db_error=%d", len(clean), no_scenarios, len(db_errors))
    for line in db_errors[:10]:
        logger.warning("db_error %s", line)
    logger.info("lists -> %s ; symlinks -> %s", CLEAN_LIST, CLEAN_DIR)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Write the held-out log list")
    parser.add_argument("--validate", action="store_true", help="Deep-screen held-out DBs")
    parser.add_argument("--limit", type=int, default=300, help="Target number of clean DBs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.build or args.validate):
        parser.error("choose --build and/or --validate")
    if args.build:
        build_heldout()
    if args.validate:
        validate(args.limit, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
