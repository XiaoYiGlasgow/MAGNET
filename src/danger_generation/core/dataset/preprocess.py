#!/usr/bin/env python3
"""
Danger pair preprocessing from raw NuPlan Pittsburgh data.

Produces correctly formatted pkl files in dataset/danger_pairs/:
  - ego_history:      (N_past, 5) [x, y, heading, vx, vy] ego-centric absolute
  - ego_future_gt:    (N_fut, 2)  [Δx, Δy] vehicle-local increments (correct velo)
  - key_agent_history:(N_past, 5) ego-centric absolute
  - key_agent_future: (N_fut, 5)  ego-centric absolute [x, y, heading, vx, vy]
  - gt_danger:        float, danger score from correct trajectories

Usage:
    python preprocess.py --n_scenarios 100
    python preprocess.py --all
"""

import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, '/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.insert(0, '/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

from src.shared.utils.ttc_filter import (
    TTCConfig,
    PairSelectionConfig,
    compute_agent_ttc_sequence,
    check_dimension_risk,
    select_key_agent_pair,
)
from .trajectory_utils import (
    global_to_ego_centric,
    build_agent_future_5col,
)

logger = logging.getLogger(__name__)

# ─── constants ───────────────────────────────────────────────────────────────

DATA_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_ROOT  = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
MAP_VER   = "nuplan-maps-v1.0"

N_PAST   = 40   # 4 s @ 10 Hz
N_FUTURE = 60   # 6 s @ 10 Hz
DT       = 0.1  # s

OUTPUT_DIR = Path(__file__).parent / "danger_pairs"


# ─── ego extraction ───────────────────────────────────────────────────────────

def _ego_states_to_5col(states: list, ref_x: float, ref_y: float, ref_h: float) -> np.ndarray:
    """Convert a list of NuPlan EgoState objects → (T, 5) ego-centric array.

    NuPlan's rear_axle_velocity_2d is in the vehicle's OWN local frame at that
    historical time (x = forward, y = left).  To bring it into the reference
    ego-centric frame we must compose:
        R(hist_heading) [local→global]  then  R(-ref_heading) [global→ego-centric]
    = R(hist_heading - ref_heading) applied to the local velocity vector.
    """
    out = np.zeros((len(states), 5), dtype=np.float32)
    for t, s in enumerate(states):
        rx, ry, rh = global_to_ego_centric(
            s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading,
            ref_x, ref_y, ref_h,
        )
        spd = s.dynamic_car_state.rear_axle_velocity_2d
        lvx, lvy = spd.x, spd.y            # vehicle-local (forward, left)
        rel_h = s.rear_axle.heading - ref_h  # heading relative to reference frame
        rvx = lvx * np.cos(rel_h) - lvy * np.sin(rel_h)
        rvy = lvx * np.sin(rel_h) + lvy * np.cos(rel_h)
        out[t] = [rx, ry, rh, rvx, rvy]
    return out


def extract_ego_history(scenario, ref_x: float, ref_y: float, ref_h: float) -> np.ndarray:
    """Return ego history (N_PAST, 5) in ego-centric absolute coordinates."""
    states = list(scenario.get_ego_past_trajectory(
        iteration=0, num_samples=N_PAST, time_horizon=N_PAST * DT
    ))
    if len(states) < N_PAST:
        return None
    return _ego_states_to_5col(states[-N_PAST:], ref_x, ref_y, ref_h)


def extract_ego_future_increments(
    scenario, ref_x: float, ref_y: float, ref_h: float
) -> np.ndarray:
    """Return ego future as (N_FUTURE, 2) vehicle-local increments (Δx, Δy).

    Frame 0 = (0, 0) (ego stays at origin in the first future frame by definition).
    Frame t = displacement from frame t-1 in vehicle-local coordinates.
    Velocity from these increments: vx = Δx / DT, vy = Δy / DT.
    """
    states = list(scenario.get_ego_future_trajectory(
        iteration=0, num_samples=N_FUTURE, time_horizon=N_FUTURE * DT
    ))
    if len(states) < N_FUTURE:
        return None

    # Use first future state as reference (identical to current ego state)
    cos_h = np.cos(ref_h)
    sin_h = np.sin(ref_h)

    increments = np.zeros((N_FUTURE, 2), dtype=np.float32)
    # frame 0: displacement from ego origin → first future position
    # In vehicle-local frame this equals the global displacement rotated by heading.
    # But since the reference frame already IS the vehicle frame, frame 0 = (0,0).
    # Subsequent frames are deltas between consecutive future positions.
    for i in range(1, N_FUTURE):
        prev = states[i - 1]
        curr = states[i]
        gdx = curr.rear_axle.x - prev.rear_axle.x
        gdy = curr.rear_axle.y - prev.rear_axle.y
        # rotate global delta → vehicle-local
        increments[i, 0] = gdx * cos_h + gdy * sin_h
        increments[i, 1] = -gdx * sin_h + gdy * cos_h

    return increments


# ─── agent extraction ────────────────────────────────────────────────────────

def _build_agent_lookup(past_obj_lists: list, present_objs, ref_x, ref_y, ref_h):
    """Build {track_token: [(t, rx, ry, rh, speed)]} from NuPlan tracked objects."""
    lookup: dict[str, list] = {}
    all_frames = list(past_obj_lists) + [present_objs]
    for t_idx, frame in enumerate(all_frames):
        objs = frame.tracked_objects if hasattr(frame, 'tracked_objects') else frame
        for obj in objs:
            tok = obj.track_token
            rx, ry, rh = global_to_ego_centric(
                obj.center.x, obj.center.y, obj.center.heading, ref_x, ref_y, ref_h
            )
            spd = obj.velocity
            gvx = spd.x if hasattr(spd, 'x') else 0.0
            gvy = spd.y if hasattr(spd, 'y') else 0.0
            cos_h = np.cos(-ref_h)
            sin_h = np.sin(-ref_h)
            rvx = gvx * cos_h - gvy * sin_h
            rvy = gvx * sin_h + gvy * cos_h
            if tok not in lookup:
                lookup[tok] = []
            lookup[tok].append((t_idx, rx, ry, rh, rvx, rvy))
    return lookup


def extract_agent_histories(scenario, ref_x, ref_y, ref_h) -> dict[str, np.ndarray]:
    """Return {track_token: (N_PAST, 5)} for agents with complete history."""
    past_lists = list(scenario.get_past_tracked_objects(
        iteration=0, time_horizon=N_PAST * DT, num_samples=N_PAST
    ))
    present = scenario.initial_tracked_objects
    lookup = _build_agent_lookup(past_lists, present, ref_x, ref_y, ref_h)

    result = {}
    for tok, frames in lookup.items():
        if len(frames) < N_PAST:
            continue
        # take last N_PAST frames
        frames = sorted(frames, key=lambda x: x[0])[-N_PAST:]
        arr = np.array([[r, ry, rh, rvx, rvy] for (_, r, ry, rh, rvx, rvy) in frames],
                       dtype=np.float32)
        result[tok] = arr
    return result


def extract_agent_futures(scenario, agent_histories: dict, ref_x, ref_y, ref_h) -> dict[str, np.ndarray]:
    """Return {track_token: (N_FUTURE, 5)} for agents that appear in future frames."""
    future_lists = list(scenario.get_future_tracked_objects(
        iteration=0, time_horizon=N_FUTURE * DT, num_samples=N_FUTURE
    ))

    # collect future positions per track_token
    future_pos: dict[str, list] = {}
    for frame in future_lists:
        objs = frame.tracked_objects if hasattr(frame, 'tracked_objects') else frame
        for obj in objs:
            tok = obj.track_token
            rx, ry, rh = global_to_ego_centric(
                obj.center.x, obj.center.y, obj.center.heading, ref_x, ref_y, ref_h
            )
            if tok not in future_pos:
                future_pos[tok] = []
            future_pos[tok].append((rx, ry, rh))

    result = {}
    for tok, positions in future_pos.items():
        if tok not in agent_histories:
            continue
        last_hist = agent_histories[tok][-1]  # (5,)
        pos_arr = np.array(positions, dtype=np.float32)  # (T, 3)

        # pad or truncate to N_FUTURE
        T = len(pos_arr)
        if T < N_FUTURE:
            # constant-velocity extrapolation for missing tail frames
            vx, vy = last_hist[3], last_hist[4]
            last_p = pos_arr[-1] if T > 0 else last_hist[:3]
            pad_rows = []
            for k in range(1, N_FUTURE - T + 1):
                pad_rows.append([
                    last_p[0] + vx * k * DT,
                    last_p[1] + vy * k * DT,
                    last_p[2],
                ])
            pos_arr = np.vstack([pos_arr, pad_rows]).astype(np.float32)
        else:
            pos_arr = pos_arr[:N_FUTURE]

        result[tok] = build_agent_future_5col(pos_arr, last_hist[:2], dt=DT)

    return result


# ─── TTC + pair selection bridge ─────────────────────────────────────────────

def _tokens_to_int_ids(agent_histories: dict) -> tuple[dict, dict]:
    """Assign integer IDs to track tokens for compatibility with ttc_filter."""
    token_to_id = {tok: i + 1 for i, tok in enumerate(sorted(agent_histories))}
    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


def run_ttc_analysis(ego_history: np.ndarray, agent_histories: dict) -> dict:
    """Run TTC analysis on ego vs all agents. Returns scenario_result dict."""
    ttc_cfg = TTCConfig()
    token_to_id, id_to_token = _tokens_to_int_ids(agent_histories)

    agent_results = []
    for tok, hist in agent_histories.items():
        aid = token_to_id[tok]
        lon_ttc, lat_ttc = compute_agent_ttc_sequence(hist, ego_history, ttc_cfg)
        lon_risk = check_dimension_risk(lon_ttc, ttc_cfg.LONGITUDINAL_TTC_THRESHOLD)
        lat_risk = check_dimension_risk(lat_ttc, ttc_cfg.LATERAL_TTC_THRESHOLD)
        agent_results.append({
            'agent_id': aid,
            'track_token': tok,
            'agent_data': hist,
            'lon_ttc': lon_ttc,
            'lat_ttc': lat_ttc,
            'lon_risk': lon_risk,
            'lat_risk': lat_risk,
            'keep': lon_risk or lat_risk,
        })

    return {
        'ego_data': ego_history,
        'agent_results': agent_results,
        'num_frames': len(ego_history),
        'file_name': '',
    }


# ─── boundary agents ─────────────────────────────────────────────────────────

BOUNDARY_RADIUS = 50.0  # metres: include agents within this distance of ego

def build_boundary_agents(
    scenario_result: dict,
    key_agent_tok: str,
    agent_hists: dict,
    agent_futs: dict,
) -> list:
    """Collect all agents except the key agent as boundary agents.

    Includes all agents that either (a) have TTC risk, or (b) are within
    BOUNDARY_RADIUS of ego at the last history frame.  Their histories are
    always stored; futures are included when available.

    Returns:
        List of dicts compatible with generate_danger_scenes.py IDM code:
        [{agent_id, agent_data (N_past,5), agent_future (N_fut,5) or None,
          lon_ttc, lat_ttc, lon_risk, lat_risk}]
    """
    boundary = []
    for r in scenario_result['agent_results']:
        tok = r['track_token']
        if tok == key_agent_tok:
            continue

        hist = agent_hists[tok]  # (N_past, 5)

        # distance filter: use last history position
        dist = float(np.linalg.norm(hist[-1, :2]))
        if not r['keep'] and dist > BOUNDARY_RADIUS:
            continue

        boundary.append({
            'agent_id':     r['agent_id'],
            'track_token':  tok,
            'agent_data':   hist,                     # (N_past, 5) history
            'agent_future': agent_futs.get(tok),      # (N_fut, 5) or None
            'lon_ttc':      r['lon_ttc'],
            'lat_ttc':      r['lat_ttc'],
            'lon_risk':     r['lon_risk'],
            'lat_risk':     r['lat_risk'],
        })

    return boundary


# ─── save ────────────────────────────────────────────────────────────────────

def save_danger_pair(
    scenario_name: str,
    ego_history: np.ndarray,
    ego_future_gt: np.ndarray,
    agent_history: np.ndarray,
    agent_future: np.ndarray,
    boundary_agents: list,
    selection_result: dict,
    gt_danger: float,
    output_dir: Path,
) -> Path:
    """Save a correctly formatted danger pair pkl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_id = selection_result['agent_id']
    out_path = output_dir / f"danger_pair_{scenario_name}_agent{agent_id}.pkl"

    data = {
        'scenario_name':       scenario_name,
        'ego_history':         ego_history,       # (N_past, 5) ego-centric abs
        'ego_future_gt':       ego_future_gt,      # (N_fut, 2) vehicle-local incr
        'key_agent_id':        agent_id,
        'key_agent_history':   agent_history,      # (N_past, 5) ego-centric abs
        'key_agent_future':    agent_future,       # (N_fut, 5) ego-centric abs
        'gt_danger':           gt_danger,
        'key_agent_lon_ttc':   selection_result['lon_ttc'],
        'key_agent_lat_ttc':   selection_result['lat_ttc'],
        'selection_reason':    selection_result.get('selection_reason', 'ttc'),
        # boundary agents: histories for GNN context + IDM reactive simulation
        'boundary_agents':     boundary_agents,   # list of dicts
        'num_boundary_agents': len(boundary_agents),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    return out_path


# ─── per-scenario pipeline ────────────────────────────────────────────────────

def _compute_gt_danger(ego_history, ego_future_incr, agent_history, agent_future):
    """Compute ground-truth danger score from correct trajectories."""
    try:
        from src.shared.utils.danger_score_torch import (
            compute_single_trajectory_danger_torch,
            DangerScoreConfigTorch,
        )
        import torch

        # ego: increments → absolute positions → build 5-col
        from .trajectory_utils import increments_to_positions, compute_velocity_from_increments
        abs_pos = increments_to_positions(ego_future_incr)        # (T, 2)
        vels    = compute_velocity_from_increments(ego_future_incr, DT)  # (T, 2)
        dx      = ego_future_incr[:, 0]
        dy      = ego_future_incr[:, 1]
        headings = np.arctan2(dy, dx)
        ego_future_5col = np.stack([
            ego_history[-1, 0] + abs_pos[:, 0],
            ego_history[-1, 1] + abs_pos[:, 1],
            headings, vels[:, 0], vels[:, 1],
        ], axis=1).astype(np.float32)

        ego_f = torch.from_numpy(ego_future_5col).unsqueeze(0)
        agt_f = torch.from_numpy(agent_future[:, :5]).unsqueeze(0)

        danger_cfg = DangerScoreConfigTorch()
        result = compute_single_trajectory_danger_torch(ego_f, agt_f, danger_cfg, traj_type="future")
        return float(result['total_score'])
    except Exception as e:
        logger.debug(f"gt_danger computation failed: {e}")
        return 0.0


def process_scenario(scenario, output_dir: Path) -> Optional[Path]:
    """Full pipeline for one NuPlan scenario. Returns saved path or None."""
    name = scenario.scenario_name

    # reference pose = current ego state
    ref = scenario.initial_ego_state.rear_axle
    ref_x, ref_y, ref_h = ref.x, ref.y, ref.heading

    ego_hist = extract_ego_history(scenario, ref_x, ref_y, ref_h)
    if ego_hist is None:
        return None

    ego_fut_incr = extract_ego_future_increments(scenario, ref_x, ref_y, ref_h)
    if ego_fut_incr is None:
        return None

    agent_hists = extract_agent_histories(scenario, ref_x, ref_y, ref_h)
    if not agent_hists:
        return None

    agent_futs = extract_agent_futures(scenario, agent_hists, ref_x, ref_y, ref_h)

    scenario_result = run_ttc_analysis(ego_hist, agent_hists)

    ttc_cfg  = TTCConfig()
    sel_cfg  = PairSelectionConfig()
    selection = select_key_agent_pair(scenario_result, ttc_cfg, sel_cfg)

    if not selection.get('selected', False):
        return None

    tok = next(
        r['track_token'] for r in scenario_result['agent_results']
        if r['agent_id'] == selection['agent_id']
    )

    agt_hist = agent_hists[tok]
    agt_fut  = agent_futs.get(tok)
    if agt_fut is None:
        # constant-velocity fallback
        last = agt_hist[-1]
        agt_fut = np.zeros((N_FUTURE, 5), dtype=np.float32)
        for t in range(N_FUTURE):
            agt_fut[t, 0] = last[0] + last[3] * (t + 1) * DT
            agt_fut[t, 1] = last[1] + last[4] * (t + 1) * DT
            agt_fut[t, 2] = last[2]
            agt_fut[t, 3] = last[3]
            agt_fut[t, 4] = last[4]

    boundary_agents = build_boundary_agents(scenario_result, tok, agent_hists, agent_futs)

    gt_danger = _compute_gt_danger(ego_hist, ego_fut_incr, agt_hist, agt_fut)

    return save_danger_pair(
        scenario_name=name,
        ego_history=ego_hist,
        ego_future_gt=ego_fut_incr,
        agent_history=agt_hist,
        agent_future=agt_fut,
        boundary_agents=boundary_agents,
        selection_result=selection,
        gt_danger=gt_danger,
        output_dir=output_dir,
    )


# ─── main ────────────────────────────────────────────────────────────────────

def build_scenarios(n_scenarios: Optional[int] = None, scenario_types: Optional[list] = None):
    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT,
        map_root=MAP_ROOT,
        sensor_root=None,
        db_files=None,
        map_version=MAP_VER,
    )
    sf = ScenarioFilter(
        scenario_types=scenario_types,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=n_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=10.0,  # skip near-static scenes
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
    )
    worker = Sequential()
    return builder.get_scenarios(sf, worker)


def run(n_scenarios: Optional[int], output_dir: Path, skip_existing: bool = False,
        scenario_types: Optional[list] = None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build set of already-processed scenario names for fast lookup
    existing: set = set()
    if skip_existing:
        for p in output_dir.glob("danger_pair_*.pkl"):
            # filename: danger_pair_{scenario_name}_agent{id}.pkl
            parts = p.stem.split('_agent')[0]  # "danger_pair_{name}"
            sc_name = parts[len("danger_pair_"):]
            existing.add(sc_name)
        logger.info(f"Skipping {len(existing)} already-processed scenarios")

    scenarios = build_scenarios(n_scenarios, scenario_types=scenario_types)
    total = len(scenarios)
    logger.info(f"Processing {total} scenarios → {output_dir}")

    saved, failed, skipped = 0, 0, 0
    for i, sc in enumerate(scenarios, 1):
        if skip_existing and sc.scenario_name in existing:
            skipped += 1
            logger.debug(f"[{i}/{total}] skip (exists): {sc.scenario_name}")
            continue
        try:
            path = process_scenario(sc, output_dir)
            if path:
                saved += 1
                logger.info(f"[{i}/{total}] saved: {path.name}")
            else:
                logger.debug(f"[{i}/{total}] no dangerous pair: {sc.scenario_name}")
        except Exception as e:
            failed += 1
            logger.warning(f"[{i}/{total}] failed {sc.scenario_name}: {e}")

    logger.info(
        f"Done. Saved {saved} new danger pairs, {skipped} skipped, "
        f"{failed} failures out of {total} scenarios."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--n_scenarios', type=int, help='Number of scenarios to process')
    group.add_argument('--all', action='store_true', help='Process all Pittsburgh scenarios')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip scenarios whose danger_pair pkl already exists')
    parser.add_argument('--scenario_types', type=str, default=None,
                        help='Comma-separated NuPlan scenario types to filter, e.g. '
                             'traversing_intersection,traversing_traffic_light_intersection')
    args = parser.parse_args()

    n = None if args.all else args.n_scenarios
    types = [t.strip() for t in args.scenario_types.split(',')] if args.scenario_types else None
    run(n, Path(args.output), skip_existing=args.skip_existing, scenario_types=types)
