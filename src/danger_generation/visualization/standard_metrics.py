"""
Standard evaluation metrics aligned with adversarial trajectory generation literature.

References: AdvDiffuser (ICCV 2023), SafeBench (NeurIPS 2022), KING (ECCV 2022)

Metrics:
  - Collision Rate       : fraction of scenarios where ego collides with agent
  - Mean Min Distance    : average minimum inter-vehicle distance
  - Mean TTC             : average time-to-collision (inf if no near-miss)
  - Kinematic Feasibility: fraction with accel <= MAX_ACCEL and jerk <= MAX_JERK
  - Jerk JSD             : Jensen-Shannon divergence of jerk distributions (gen vs GT)
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde


# ── Vehicle geometry ─────────────────────────────────────────────────────────
EGO_LENGTH   = 4.5   # m
EGO_WIDTH    = 2.0   # m
AGENT_LENGTH = 4.5   # m
AGENT_WIDTH  = 2.0   # m

COLLISION_DIST = (EGO_WIDTH + AGENT_WIDTH) / 2   # simplified circle radius: 2.0 m

# ── Physical feasibility thresholds (KING / AdvDiffuser convention) ───────────
MAX_ACCEL = 8.0   # m/s²
MAX_JERK  = 15.0  # m/s³

DT = 0.1  # NuPlan timestep


def _inter_distances(ego_traj: np.ndarray, agent_traj: np.ndarray) -> np.ndarray:
    """Euclidean distance between ego and agent centers at each timestep. (T,)"""
    ego_pos   = ego_traj[:, :2]
    agent_pos = agent_traj[:, :2]
    T = min(len(ego_pos), len(agent_pos))
    diff = ego_pos[:T] - agent_pos[:T]
    return np.sqrt((diff ** 2).sum(axis=1))


def compute_collision(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    threshold: float = COLLISION_DIST,
) -> bool:
    """True if ego and agent come within `threshold` metres at any timestep."""
    dists = _inter_distances(ego_traj, agent_traj)
    return bool(np.any(dists < threshold))


def compute_min_distance(ego_traj: np.ndarray, agent_traj: np.ndarray) -> float:
    """Minimum inter-vehicle distance over the trajectory horizon."""
    return float(np.min(_inter_distances(ego_traj, agent_traj)))


def compute_ttc(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    threshold: float = COLLISION_DIST,
    dt: float = DT,
) -> float:
    """
    Time-to-Collision: first time (seconds) the distance drops below `threshold`.
    Returns np.inf if no near-miss occurs.
    """
    dists = _inter_distances(ego_traj, agent_traj)
    near = np.where(dists < threshold)[0]
    if len(near) == 0:
        return float('inf')
    return float(near[0] * dt)


def _longitudinal_accels(traj: np.ndarray, dt: float = DT) -> np.ndarray:
    """
    Longitudinal (scalar) accelerations from a (T, 5) trajectory array.
    Uses speed magnitude differences, avoiding centripetal contamination.
    """
    speeds = np.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    return np.abs(np.diff(speeds)) / dt


def _longitudinal_jerks(traj: np.ndarray, dt: float = DT) -> np.ndarray:
    """Longitudinal jerk from a (T, 5) trajectory array."""
    return np.abs(np.diff(_longitudinal_accels(traj, dt))) / dt


def compute_kinematic_feasibility(eval_samples) -> float:
    """
    Fraction of generated trajectories satisfying MAX_ACCEL and MAX_JERK bounds,
    computed from longitudinal (scalar) acceleration to avoid centripetal artifact.
    """
    if not eval_samples:
        return float('nan')
    feasible = sum(
        1 for s in eval_samples
        if _longitudinal_accels(s.gen_trajectory).max() <= MAX_ACCEL
        and _longitudinal_jerks(s.gen_trajectory).max() <= MAX_JERK
    )
    return feasible / len(eval_samples)


def compute_jerk_jsd(eval_samples, _unused=None, bins: int = 30) -> float:
    """
    Jensen-Shannon divergence between longitudinal jerk distributions of
    generated vs GT trajectories. Range [0, 1]; lower is better.
    """
    jerks_gen = np.array([_longitudinal_jerks(s.gen_trajectory).max()
                          for s in eval_samples])
    jerks_gt  = np.array([_longitudinal_jerks(s.gt_trajectory).max()
                          for s in eval_samples])

    if len(jerks_gen) < 2 or len(jerks_gt) < 2:
        return float('nan')

    lo = min(jerks_gen.min(), jerks_gt.min())
    hi = max(jerks_gen.max(), jerks_gt.max()) + 1e-8
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(jerks_gen, bins=edges, density=True)
    q, _ = np.histogram(jerks_gt,  bins=edges, density=True)
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()

    return float(jensenshannon(p, q))


def compute_standard_metrics(
    eval_samples,           # List[EvalSample] from matplotlib_visualizer
    metrics_gen: Optional[List[Dict]] = None,   # unused, kept for compatibility
    metrics_gt:  Optional[List[Dict]] = None,   # unused, kept for compatibility
) -> Dict:
    """
    Compute the full suite of standard metrics over a set of evaluated samples.

    Args:
        eval_samples : list of EvalSample objects (must have gen_trajectory,
                       gt_trajectory, agent_future fields)

    Returns:
        Dictionary with all standard metrics.
    """
    collision_flags: List[bool] = []
    min_distances:   List[float] = []
    ttcs:            List[float] = []

    for sample in eval_samples:
        if not hasattr(sample, 'agent_future') or sample.agent_future is None:
            continue
        ego_traj   = sample.gen_trajectory    # (60, 5)
        agent_traj = sample.agent_future      # (60, 5)

        collision_flags.append(compute_collision(ego_traj, agent_traj))
        min_distances.append(compute_min_distance(ego_traj, agent_traj))
        ttcs.append(compute_ttc(ego_traj, agent_traj))

    n = len(collision_flags)
    finite_ttcs = [t for t in ttcs if np.isfinite(t)]

    return {
        'collision_rate':        float(sum(collision_flags) / n) if n else float('nan'),
        'mean_min_distance':     float(np.mean(min_distances)) if min_distances else float('nan'),
        'mean_ttc':              float(np.mean(finite_ttcs)) if finite_ttcs else float('inf'),
        'near_miss_rate':        float(len(finite_ttcs) / n) if n else float('nan'),
        'kinematic_feasibility': compute_kinematic_feasibility(eval_samples),
        'jerk_jsd':              compute_jerk_jsd(eval_samples, eval_samples),  # gen vs GT via .gen_trajectory / .gt_trajectory
        'n_evaluated':           n,
    }
