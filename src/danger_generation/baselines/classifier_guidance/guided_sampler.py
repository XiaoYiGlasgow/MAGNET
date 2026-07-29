"""
Guided DDPM sampler for classifier guidance baseline.

Implements test-time classifier guidance (inspired by Safe-Sim, ECCV 2024):
at each denoising step, the predicted clean trajectory x_0_pred is decoded,
the danger score gradient is computed, and it is used to shift x_0_pred
toward more dangerous configurations before computing x_{t-1}.

Model output format: 240-dim = ego(120) + agent(120), each as (dx, dy) increments
  → first 120 dims: ego position increments (60 × 2)
  → guidance is applied only to the ego part

References:
  - Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021
  - Safe-Sim (Feng et al., ECCV 2024): classifier-free + constraint guidance for AV
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from utils.danger_score_torch import compute_danger_score_torch


# ── Differentiable (dx, dy) increment decoder ─────────────────────────────────

def decode_increments_torch(
    ego_increments_120: torch.Tensor,   # (B, 120) or (120,)
    start_pos: np.ndarray,              # (2,)
    start_vx: float,
    start_vy: float,
    dt: float = 0.1,
) -> torch.Tensor:
    """
    Differentiable decoder: (dx, dy) position increments → (B, 60, 5) trajectory.

    Mirrors the cumulative-sum part of recover_trajectory_from_increment()
    without the non-differentiable Savitzky-Golay smoothing.

    Column order: [x, y, heading, vx, vy]
    """
    squeezed = ego_increments_120.dim() == 1
    if squeezed:
        ego_increments_120 = ego_increments_120.unsqueeze(0)

    B = ego_increments_120.shape[0]
    device = ego_increments_120.device
    dtype = ego_increments_120.dtype

    inc = ego_increments_120.view(B, 60, 2)  # (B, 60, 2)

    # Initial state tensors
    pos0 = torch.tensor(start_pos, device=device, dtype=dtype).view(1, 1, 2)        # (1,1,2)
    v0   = torch.tensor([start_vx, start_vy], device=device, dtype=dtype).view(1, 1, 2)  # (1,1,2)

    # Positions: cumulative sum of increments from start_pos
    pos_cumsum = torch.cumsum(inc, dim=1)               # (B, 60, 2)
    positions = pos0 + pos_cumsum                        # (B, 60, 2)

    # Velocities: central finite differences; v[0] kept as start velocity
    pos_prev = torch.cat([pos0.expand(B, 1, 2), positions[:, :-1, :]], dim=1)  # (B, 60, 2)
    velocities = (positions - pos_prev) / dt             # (B, 60, 2)
    velocities = torch.cat([v0.expand(B, 1, 2), velocities[:, 1:, :]], dim=1)  # keep v0

    # Headings from velocity direction
    headings = torch.atan2(velocities[:, :, 1], velocities[:, :, 0])  # (B, 60)

    traj = torch.stack([
        positions[:, :, 0],   # x
        positions[:, :, 1],   # y
        headings,             # heading
        velocities[:, :, 0],  # vx
        velocities[:, :, 1],  # vy
    ], dim=2)                                           # (B, 60, 5)

    return traj.squeeze(0) if squeezed else traj


# ── Single guided DDPM step ────────────────────────────────────────────────────

def guided_sample_step(
    model,
    x_t: torch.Tensor,            # (B, 240) current noisy joint trajectory
    condition: torch.Tensor,      # (B, 256) condition vector
    t: int,
    sample: Dict,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    One reverse-diffusion step with danger-score classifier guidance.

    Guidance is applied by shifting x_0_pred (ego part) in the direction of
    increasing danger before computing the posterior mean x_{t-1}.
    """
    B = x_t.shape[0]
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

    # ── 1. Predict noise and reconstruct x_0_pred ───────────────────────────
    with torch.no_grad():
        noise_pred = model.denoiser(x_t, condition, t_tensor)      # (B, 240)

    alpha_bar_t = model.alpha_bars[t]
    x_0_pred = (x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

    # ── 2. Compute danger gradient on CPU (avoids device-mismatch in
    #        compute_danger_score_torch which creates internal CPU tensors) ──
    ego_0_cpu = x_0_pred[:, :120].detach().cpu().requires_grad_(True)   # (B,120) CPU

    ego_traj_cpu = decode_increments_torch(
        ego_0_cpu,
        start_pos=sample["start_pos"],
        start_vx=float(sample["start_vx"]),
        start_vy=float(sample["start_vy"]),
        dt=0.1,
    )  # (B, 60, 5) CPU

    ego_h_cpu   = torch.tensor(sample["ego_history"],        dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    agent_h_cpu = torch.tensor(sample["key_agent_history"],  dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    agent_f_cpu = torch.tensor(sample["key_agent_future"],   dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

    danger_result = compute_danger_score_torch(ego_h_cpu, ego_traj_cpu, agent_h_cpu, agent_f_cpu)
    danger_loss   = danger_result["outcome_danger"].mean()
    grad_cpu = torch.autograd.grad(danger_loss, ego_0_cpu)[0]   # (B, 120) CPU

    # Move gradient back to original device
    grad = grad_cpu.to(device)                                   # (B, 120) CUDA / CPU

    # ── 3. Apply guidance: shift ego x_0_pred toward higher danger ──────────
    with torch.no_grad():
        x_0_guided = x_0_pred.clone()
        x_0_guided[:, :120] = x_0_guided[:, :120] + guidance_scale * grad

        # ── 4. Standard DDPM posterior mean and sample ───────────────────────
        mean = (
            model.posterior_mean_coef1[t] * x_0_guided
            + model.posterior_mean_coef2[t] * x_t
        )

        if t > 0:
            sigma  = model.posterior_variance[t].sqrt()
            x_prev = mean + sigma * torch.randn_like(x_t)
        else:
            x_prev = mean

    return x_prev


# ── Full guided generation loop ────────────────────────────────────────────────

def guided_generate(
    model,
    condition: torch.Tensor,
    sample: Dict,
    guidance_scale: float = 50.0,
    guide_every: int = 10,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run the full guided DDPM reverse process.

    Guidance is applied every `guide_every` denoising steps to keep runtime
    manageable (100 steps default → 10 backward passes per sample).

    Returns:
        ego_increments : (120,) numpy array of (dx, dy) increments for ego
    """
    if device is None:
        device = next(model.parameters()).device

    num_steps  = model.config.num_diffusion_steps
    traj_dim   = model.config.trajectory_dim   # 240

    x = torch.randn(1, traj_dim, device=device)

    model.eval()
    for t in reversed(range(num_steps)):
        apply_guidance = (t % guide_every == 0)

        if apply_guidance:
            x = guided_sample_step(
                model, x, condition, t, sample, guidance_scale, device
            )
        else:
            with torch.no_grad():
                x = model.sample_step(x, condition, t)

    return x.squeeze(0)[:120].detach().cpu().numpy()  # ego increments only
