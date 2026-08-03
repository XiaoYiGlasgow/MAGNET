"""
[v6.0] Differentiable Bezier control-point utility for DPS-style trajectory guidance (六B).

Provides two operations:
  1. fit_bezier_v6_0    — least-squares fit of a (T, D) trajectory to N control points
  2. upsample_bezier_v6_0 — evaluate the Bezier curve at T uniformly-spaced parameter values

Using these, the DPS guidance loop can:
  a. Project x̂_0 ego trajectory into N-control-point Bezier space
  b. Evaluate SVO / realism loss on the *smooth* upsampled version
  c. Backprop gradients through the Bezier layer to obtain a smooth guidance signal

All operations retain the autograd graph through the Bezier basis matrices, so
gradients flow back to the original trajectory tensor used in (a).
"""

import math
import torch


# ── Basis matrices (precomputed, registered on first call) ──────────────────

def _bernstein_basis_v6_0(n: int, t: torch.Tensor) -> torch.Tensor:
    """[v6.0] Bernstein basis matrix for degree-(n-1) Bezier curve.

    Args:
        n: number of control points (degree = n−1).
        t: (T,) parameter values in [0, 1].

    Returns:
        (T, n) basis matrix B where B[i, j] = C(n−1,j) * t[i]^j * (1−t[i])^(n−1−j).
    """
    T = t.shape[0]
    degree = n - 1
    B = torch.zeros(T, n, dtype=t.dtype, device=t.device)
    for j in range(n):
        coeff = math.comb(degree, j)
        B[:, j] = coeff * (t ** j) * ((1 - t) ** (degree - j))
    return B  # (T, n)


def fit_bezier_v6_0(
    traj: torch.Tensor,
    n_ctrl: int = 10,
) -> torch.Tensor:
    """[v6.0] Least-squares fit of a trajectory to Bezier control points.

    Solves: argmin_{P} ||B @ P - traj||^2  via the normal equations B^T B P = B^T traj.
    The solve is exact (no regularisation) and retains the autograd graph through `traj`.

    Args:
        traj: (T, D) trajectory to fit, must retain grad if you want gradients through it.
        n_ctrl: number of control points (degree = n_ctrl−1, default 10).

    Returns:
        ctrl: (n_ctrl, D) control point matrix (differentiable w.r.t. traj).
    """
    T, D = traj.shape
    device = traj.device
    dtype  = traj.dtype

    # Parameter values t_i = i / (T-1)
    t_vals = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)  # (T,)
    B  = _bernstein_basis_v6_0(n_ctrl, t_vals)                         # (T, n_ctrl)

    # Normal equations: (B^T B + λI) P = B^T traj
    # Tikhonov regularisation (λ=1e-4) prevents ill-conditioning for high-degree polynomials
    BtB = B.T @ B                                             # (n_ctrl, n_ctrl)
    BtB = BtB + 1e-4 * torch.eye(n_ctrl, dtype=dtype, device=device)
    Btt = B.T @ traj                                          # (n_ctrl, D)

    # Solve via Cholesky for stability (positive-definite after regularisation)
    ctrl = torch.linalg.solve(BtB, Btt)                       # (n_ctrl, D)
    return ctrl


def upsample_bezier_v6_0(
    ctrl: torch.Tensor,
    T_out: int = 60,
) -> torch.Tensor:
    """[v6.0] Evaluate a Bezier curve defined by control points at T_out uniform points.

    Args:
        ctrl: (n_ctrl, D) control points (can retain grad graph from fit_bezier_v6_0).
        T_out: number of output time steps (default 60 = NuPlan future horizon).

    Returns:
        traj_smooth: (T_out, D) smoothly interpolated trajectory.
    """
    n_ctrl = ctrl.shape[0]
    device = ctrl.device
    dtype  = ctrl.dtype

    t_vals = torch.linspace(0.0, 1.0, T_out, device=device, dtype=dtype)
    B = _bernstein_basis_v6_0(n_ctrl, t_vals)   # (T_out, n_ctrl)
    return B @ ctrl                              # (T_out, D)


def smooth_trajectory_via_bezier_v6_0(
    traj: torch.Tensor,
    n_ctrl: int = 10,
) -> torch.Tensor:
    """[v6.0] Round-trip: fit Bezier and immediately upsample back to original length.

    Equivalent to projecting the trajectory onto the n_ctrl-dimensional Bezier subspace,
    producing a smoothed version that still has the same length T.

    Args:
        traj: (T, D) input trajectory.
        n_ctrl: number of control points.

    Returns:
        (T, D) smoothed trajectory.
    """
    T = traj.shape[0]
    ctrl = fit_bezier_v6_0(traj, n_ctrl)
    return upsample_bezier_v6_0(ctrl, T_out=T)


def bezier_guidance_v6_0(
    ego_dxdy_flat: torch.Tensor,
    n_ctrl: int = 10,
) -> torch.Tensor:
    """[v6.0] Project ego guidance vector through Bezier space for smooth output (六B).

    Given a (B, 120) flat ego guidance gradient, this function:
      1. Reshapes to (B, 60, 2)
      2. For each sample: project through fit→upsample cycle
      3. Returns the smoothed guidance as (B, 120)

    The Bezier projection acts as a differentiable spatial smoother: the resulting
    guidance vector only moves the trajectory in directions that correspond to
    smooth n_ctrl-DoF curves, suppressing high-frequency artefacts.

    Args:
        ego_dxdy_flat: (B, 120) raw ego guidance gradient (no grad needed here).
        n_ctrl: number of Bezier control points.

    Returns:
        (B, 120) smooth ego guidance.
    """
    B = ego_dxdy_flat.shape[0]
    ego_dxdy = ego_dxdy_flat.reshape(B, 60, 2)   # (B, 60, 2)

    smoothed = []
    for i in range(B):
        s = smooth_trajectory_via_bezier_v6_0(ego_dxdy[i], n_ctrl=n_ctrl)
        smoothed.append(s)

    return torch.stack(smoothed).reshape(B, 120)   # (B, 120)


# ── Tests ────────────────────────────────────────────────────────────────────

def _test_bezier_v6_0():
    """[v6.0] Quick sanity checks for Bezier utility functions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[bezier_torch_v6_0] test on {device}")

    # 1. Shape and smoothed output
    T, D = 60, 2
    traj = torch.randn(T, D, device=device)
    smoothed = smooth_trajectory_via_bezier_v6_0(traj, n_ctrl=20)
    assert smoothed.shape == (T, D), f"Shape mismatch: {smoothed.shape}"

    # 2. Smoothing reduces high-frequency energy
    # Inject high-freq noise and verify smoothed version is smoother
    noisy = traj + 2.0 * torch.randn_like(traj)
    smooth_traj = smooth_trajectory_via_bezier_v6_0(noisy, n_ctrl=10)
    diff_noisy  = noisy.diff(dim=0).norm().item()
    diff_smooth = smooth_traj.diff(dim=0).norm().item()
    assert diff_smooth < diff_noisy, "Smoothing did not reduce variation"

    # 3. Gradient flows through fit + upsample
    traj_g = torch.randn(T, D, device=device, requires_grad=True)
    smoothed_g = smooth_trajectory_via_bezier_v6_0(traj_g, n_ctrl=10)
    loss = smoothed_g.sum()
    loss.backward()
    assert traj_g.grad is not None and traj_g.grad.abs().sum() > 0, \
        "No gradient flowed through Bezier"

    # 4. Batch guidance
    B = 4
    flat = torch.randn(B, 120, device=device)
    out  = bezier_guidance_v6_0(flat, n_ctrl=10)
    assert out.shape == (B, 120), f"Batch guidance shape: {out.shape}"

    print("  All tests passed.")


if __name__ == "__main__":
    _test_bezier_v6_0()
