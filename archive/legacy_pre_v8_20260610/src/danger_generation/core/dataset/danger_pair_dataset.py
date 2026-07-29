"""
DangerPairDataset — clean Dataset class for danger trajectory generation.

Expects pkl files produced by preprocess.py with schema:
  ego_history:      (N_past, 5)  [x, y, heading, vx, vy] ego-centric absolute
  ego_future_gt:    (N_fut, 2)   [Δx, Δy] vehicle-local increments
  key_agent_history:(N_past, 5)  ego-centric absolute
  key_agent_future: (N_fut, 5)   ego-centric absolute [x, y, heading, vx, vy]
  gt_danger:        float
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData

from .trajectory_utils import (
    increments_to_positions,
    compute_velocity_from_increments,
    positions_to_increments,
)

import pickle

# Ensure utils/ (danger_score, danger_score_torch, …) is importable
_UTILS_DIR = Path(__file__).resolve().parents[3] / "shared" / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from danger_score import DangerScoreConfig, compute_pair_svo, compute_danger_score  # noqa: E402

_DANGER_CFG = DangerScoreConfig()

N_PAST        = 40
N_FUTURE      = 60
DT            = 0.1
MAX_LANE_SEGS   = 900   # padding/容量上限，与 features pkl 固定结构对齐（60 lanes × 15 segs）
SUBGRAPH_ID_COL = 9     # POLYLINE_FEATURES col-9 是 subgraph_id（非 type），须经 SUBGRAPH_ID_TO_TYPE 查表


# ─── graph feature builder ───────────────────────────────────────────────────

def _traj_to_polyline_features(traj_5col: np.ndarray) -> np.ndarray:
    """Convert (T, 5) [x, y, h, vx, vy] trajectory to (T, 9) polyline features.

    Feature layout (matches the 9-dim expected by GNN-Trans encoder):
      [x, y, heading, vx, vy, Δx, Δy, Δheading, valid_mask]
    """
    T = len(traj_5col)
    feat = np.zeros((T, 9), dtype=np.float32)
    feat[:, :5] = traj_5col

    # motion increments (first frame diff from self → zero)
    prev = np.vstack([traj_5col[:1], traj_5col[:-1]])
    feat[:, 5] = traj_5col[:, 0] - prev[:, 0]  # Δx
    feat[:, 6] = traj_5col[:, 1] - prev[:, 1]  # Δy
    feat[:, 7] = traj_5col[:, 2] - prev[:, 2]  # Δheading
    feat[:, 8] = 1.0                             # valid mask

    return feat


def _build_graph_data(
    ego_history: np.ndarray,
    agent_history: np.ndarray,
    boundary_agents: list,
) -> GraphData:
    """Build GraphData from ego + key_agent + boundary agent history polylines.

    Node 0: ego, Node 1: key_agent, Nodes 2+: boundary agents (up to N_BOUNDARY).
    Each node is represented by the last frame of its history (9-dim features).
    Edges connect all pairs within BOUNDARY_RADIUS from ego.
    """
    N_BOUNDARY = 10  # cap to keep graph size manageable

    node_feats = [
        _traj_to_polyline_features(ego_history)[-1],     # (9,)
        _traj_to_polyline_features(agent_history)[-1],   # (9,)
    ]
    for ba in boundary_agents[:N_BOUNDARY]:
        hist = ba.get('agent_data')
        if hist is not None and len(hist) > 0:
            node_feats.append(_traj_to_polyline_features(hist)[-1])

    x = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32)  # (N, 9)
    N = x.size(0)

    # Full bidirectional graph (small enough to be all-to-all)
    edges = [[i, j] for i in range(N) for j in range(N) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)

    ego_history_tensor = torch.tensor(ego_history[:, :2], dtype=torch.float32)
    return GraphData(x=x, gnn_x=x.clone(), gnn_edge_index=edge_index,
                     ego_history=ego_history_tensor)


# ─── dataset ─────────────────────────────────────────────────────────────────

class DangerPairDataset(Dataset):
    """Dataset for diffusion-based danger trajectory generation.

    Loads from pkl files produced by preprocess.py.
    All trajectory data is correctly formatted:
      - ego_future_gt as (Δx, Δy) increments with real velocity
      - agent_future as absolute ego-centric positions with real velocity
    """

    def __init__(
        self,
        danger_pair_dir: str,
        max_samples: Optional[int] = None,
        min_gt_danger: float = 0.0,
        features_data_dir: Optional[str] = None,
    ):
        self.danger_pair_dir = Path(danger_pair_dir)
        self.min_gt_danger = min_gt_danger
        self.features_data_dir = Path(features_data_dir) if features_data_dir else None

        pattern = 'danger_pair_*.pkl'
        files = sorted(self.danger_pair_dir.glob(pattern))
        if max_samples:
            files = files[:max_samples]

        self.samples = []
        n_skip = 0
        for f in files:
            try:
                s = self._load(f)
                if s is not None:
                    self.samples.append(s)
                else:
                    n_skip += 1
            except Exception as e:
                n_skip += 1

        print(f"[DangerPairDataset] loaded {len(self.samples)} samples "
              f"({n_skip} skipped) from {self.danger_pair_dir}")

    def _load(self, pkl_path: Path) -> Optional[dict]:
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        ego_hist   = raw['ego_history'].astype(np.float32)          # (N_past, 5)
        ego_fut_gt = raw['ego_future_gt'].astype(np.float32)        # (N_fut, 2)
        agt_hist        = raw['key_agent_history'].astype(np.float32)    # (N_past, 5)
        agt_fut         = raw['key_agent_future'].astype(np.float32)     # (N_fut, 5)
        boundary_agents = raw.get('boundary_agents', [])
        gt_danger       = float(raw.get('gt_danger', 0.0))

        if gt_danger < self.min_gt_danger:
            return None

        # ensure shapes
        ego_hist   = ego_hist[-N_PAST:]
        agt_hist   = agt_hist[-N_PAST:]
        ego_fut_gt = ego_fut_gt[:N_FUTURE]
        agt_fut    = agt_fut[:N_FUTURE]

        if len(ego_hist) < N_PAST or len(agt_hist) < N_PAST:
            return None
        if len(ego_fut_gt) < N_FUTURE or len(agt_fut) < N_FUTURE:
            return None

        # ── ego future in 5-col for danger scoring ──────────────────────────
        abs_pos  = increments_to_positions(ego_fut_gt)          # (N_fut, 2) cumulative
        vels     = compute_velocity_from_increments(ego_fut_gt, DT)
        dx, dy   = ego_fut_gt[:, 0], ego_fut_gt[:, 1]
        headings = np.arctan2(dy, dx).astype(np.float32)
        ego_fut_5col = np.stack([
            ego_hist[-1, 0] + abs_pos[:, 0],
            ego_hist[-1, 1] + abs_pos[:, 1],
            headings,
            vels[:, 0],
            vels[:, 1],
        ], axis=1).astype(np.float32)

        # ── agent increments (for joint diffusion space) ────────────────────
        agt_incr = positions_to_increments(agt_fut[:, :2], agt_hist[-1, :2])  # (N_fut, 2)

        # ── joint increment vector for diffusion model ──────────────────────
        joint_increments = np.concatenate([
            ego_fut_gt.flatten(),   # 120-dim ego increments
            agt_incr.flatten(),     # 120-dim agent increments
        ]).astype(np.float32)       # → 240-dim

        # ── graph data ───────────────────────────────────────────────────────
        graph_data = _build_graph_data(ego_hist, agt_hist, boundary_agents)

        # ── SVO + inherent danger (core innovation) ───────────────────────
        svo_info = compute_pair_svo(
            ego_hist, ego_fut_5col, agt_hist, agt_fut, _DANGER_CFG
        )
        danger_result = compute_danger_score(
            ego_hist, ego_fut_5col, agt_hist, agt_fut, _DANGER_CFG
        )
        svo_angle        = float(svo_info['ego_svo_angle_deg'])
        inherent_danger  = float(danger_result['inherent_danger']['total_score'])
        outcome_danger   = float(danger_result['outcome_danger']['total_score'])

        # ── lane polylines (ego-centric, for lane deviation penalty) ──────────
        scenario_name = raw.get('scenario_name', pkl_path.stem)
        lane_polylines = self._load_lane_polylines(scenario_name)

        return {
            'graph_data':        graph_data,
            'ego_history':       ego_hist,
            'agent_history':     agt_hist,
            'ego_future_gt':     ego_fut_5col,     # (N_fut, 5) for danger scoring
            'agent_future_gt':   agt_fut,          # (N_fut, 5) for danger scoring
            'gt_increments':     ego_fut_gt.flatten(),    # (120,) ego Δ for model
            'joint_increments':  joint_increments,        # (240,)
            'ego_last_pos':      ego_hist[-1, :2],
            'ego_last_heading':  float(ego_hist[-1, 2]),
            'ego_last_vx':       float(ego_hist[-1, 3]),
            'ego_last_vy':       float(ego_hist[-1, 4]),
            'svo_angle':         svo_angle,
            'inherent_danger':   inherent_danger,
            'gt_outcome_danger': outcome_danger,
            'danger_target':     outcome_danger + 2.0,
            'gt_danger':         gt_danger,
            'boundary_agents':   boundary_agents,
            'scenario_name':     scenario_name,
            'lane_polylines':    lane_polylines,   # (MAX_LANE_SEGS, 4) padded
        }

    def _load_lane_polylines(self, scenario_name: str) -> np.ndarray:
        """Load lane centerline segments from features pkl for this scenario.

        Returns:
            (MAX_LANE_SEGS, 4) float32 array: [x_start, y_start, x_end, y_end]
            in ego-centric local frame. Padded with zeros if fewer segments.
        """
        out = np.zeros((MAX_LANE_SEGS, 4), dtype=np.float32)
        if self.features_data_dir is None:
            return out

        feat_path = self.features_data_dir / f"features_{scenario_name}.pkl"
        if not feat_path.exists():
            return out

        try:
            df = pd.read_pickle(str(feat_path))
            row = df.iloc[0]
            poly = row['POLYLINE_FEATURES']   # (N, 12)

            # col-9 是 subgraph_id（非 type），须经 SUBGRAPH_ID_TO_TYPE 查表筛 type=='lane' 的行。
            # col 0:4 = [x_start, y_start, x_end, y_end]（相邻行首尾相连）。
            sid_to_type = row.get('SUBGRAPH_ID_TO_TYPE', {})
            sids = poly[:, SUBGRAPH_ID_COL].astype(int)
            lane_mask = np.array(
                [sid_to_type.get(s, '') == 'lane' for s in sids], dtype=bool
            )
            lane_segs = poly[lane_mask, :4].astype(np.float32)  # (K, 4)

            n = min(len(lane_segs), MAX_LANE_SEGS)
            if n > 0:
                out[:n] = lane_segs[:n]
        except Exception:
            pass  # return zeros on any read failure

        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
