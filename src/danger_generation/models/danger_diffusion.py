"""
危险场景Diffusion生成器

基于DDPM的去噪扩散模型，用于生成比GT更危险的轨迹。

核心组件：
1. Sinusoidal时间步嵌入
2. Transformer-based去噪网络
3. 条件引导生成
4. Danger超越采样策略

训练目标：
- 重建GT轨迹（学习基本轨迹能力）
- Danger超越（生成轨迹的danger > GT的danger）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from shared.utils.svo_torch import (SvoTorchConfig, svo_dps_loss,
                                    compute_ego_svo_torch,
                                    compute_ego_svo_torch_timevarying,
                                    compute_svo_soft_cumulative_torch,
                                    default_svo_soft_cfg)
from shared.utils.bezier_torch_v6_0 import bezier_guidance_v6_0
from shared.utils.danger_score_torch import (lane_deviation_score_torch,
                                             compute_single_trajectory_danger_torch,
                                             DangerScoreConfigTorch)


# ============================================================
# 配置参数
# ============================================================

@dataclass
class DiffusionConfig:
    """Diffusion模型配置"""

    # 基础参数
    condition_dim: int = 256        # 条件向量维度
    trajectory_dim: int = 240       # 轨迹维度 (60步 × 4: ego dx/dy + agent dx/dy)
    num_steps: int = 60             # 轨迹时间步数

    # Diffusion参数
    num_diffusion_steps: int = 100  # 去噪步数
    beta_start: float = 1e-4        # beta起始值
    beta_end: float = 0.02          # beta结束值
    beta_schedule: str = 'linear'   # beta调度类型: 'linear', 'cosine'

    # 去噪网络参数
    denoiser_hidden_dim: int = 256
    denoiser_num_heads: int = 8
    denoiser_num_layers: int = 4
    denoiser_dropout: float = 0.1

    # 条件注入方式
    condition_injection: str = 'cross_attention'  # 'addition', 'concat', 'cross_attention'

    # Map conditioning
    use_map_conditioning: bool = False
    lane_encoder_out_dim: int = 256

    # Road boundary guidance（road-level 宽松边界：所有车道并集，软约束"尽量在路面"，
    # 允许少量超出。road_half_width = 到最近车道中心线的容差，越大边界越宽松）
    road_half_width: float = 3.0


# ============================================================
# Lane Encoder
# ============================================================

class LaneEncoder(nn.Module):
    """PointNet-style encoder for lane centerline segments.

    Input: [B, N, 4] padded ego-centric segments (x0,y0,x1,y1); N=900 in current data.
    Zero rows are padding — excluded via valid mask before pooling.
    Output: [B, out_dim] lane condition vector.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.seg_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, lane_polylines: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_polylines: [B, N, 4]
        Returns:
            [B, out_dim]
        """
        valid = lane_polylines.abs().sum(-1) > 0          # [B, N]
        seg_feat = self.seg_mlp(lane_polylines)            # [B, N, out_dim]
        valid_f = valid.float().unsqueeze(-1)              # [B, N, 1]
        pooled = (seg_feat * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)  # [B, out_dim]
        return self.out_proj(pooled)


# ============================================================
# 时间步嵌入
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal位置嵌入 - 用于时间步编码

    将时间步t编码为向量，类似于Transformer的位置编码。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch] 时间步索引 (0 to num_diffusion_steps-1)

        Returns:
            [batch, dim] 时间步嵌入
        """
        device = x.device
        half_dim = self.dim // 2

        # 计算嵌入维度
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # x [batch] -> [batch, half_dim]
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)

        # Sinusoidal编码
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class TimeStepMLP(nn.Module):
    """
    时间步MLP编码器

    将sinusoidal嵌入进一步编码为高维向量。
    """

    def __init__(self, time_dim: int, output_dim: int):
        super().__init__()

        self.time_emb = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch] 时间步索引

        Returns:
            [batch, output_dim] 时间步嵌入
        """
        t_emb = self.time_emb(t)
        return self.mlp(t_emb)


# ============================================================
# 去噪网络
# ============================================================

class TrajectoryDenoiser(nn.Module):
    """
    轨迹去噪网络

    使用Transformer架构，接收带噪声的轨迹和条件，预测噪声。
    支持多种条件注入方式。
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()

        self.config = config

        # 时间步编码
        self.time_encoder = TimeStepMLP(config.condition_dim, config.condition_dim)

        # 轨迹嵌入
        # 输入：轨迹增量 [batch, 240] -> [batch, 60, hidden_dim]
        self.traj_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, config.denoiser_hidden_dim)
        )

        # 条件注入层（根据注入方式）
        if config.condition_injection == 'cross_attention':
            # Cross-attention注入
            self.condition_proj = nn.Linear(config.condition_dim, config.denoiser_hidden_dim)

            # Cross-attention层
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.denoiser_hidden_dim,
                num_heads=config.denoiser_num_heads // 2,
                batch_first=True
            )

        elif config.condition_injection == 'concat':
            # 拼接注入
            self.condition_proj = nn.Linear(config.condition_dim, config.denoiser_hidden_dim)
            # 输入维度变大
            self.input_proj = nn.Linear(config.denoiser_hidden_dim * 2, config.denoiser_hidden_dim)

        else:  # 'addition'
            # 加法注入（最简单）
            self.condition_proj = nn.Linear(config.condition_dim, config.denoiser_hidden_dim)

        # Transformer去噪层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.denoiser_hidden_dim,
            nhead=config.denoiser_num_heads,
            dim_feedforward=config.denoiser_hidden_dim * 4,
            dropout=config.denoiser_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.denoiser_num_layers)

        # 输出层
        # 输出：预测噪声 [batch, 240]
        self.output_proj = nn.Sequential(
            nn.Linear(config.denoiser_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        # 最终输出reshape
        self.final_proj = nn.Identity()  # 用于调整维度

        print(f"[TrajectoryDenoiser] 初始化完成")
        print(f"  - Hidden dim: {config.denoiser_hidden_dim}")
        print(f"  - Num heads: {config.denoiser_num_heads}")
        print(f"  - Num layers: {config.denoiser_num_layers}")
        print(f"  - Condition injection: {config.condition_injection}")

    def forward(
        self,
        noisy_traj: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：预测噪声

        Args:
            noisy_traj: [batch, 240] 带噪声的轨迹增量 (ego_120 || agent_120 concatenated)
            condition: [batch, 256] 条件向量
            t: [batch] 时间步索引

        Returns:
            [batch, 240] 预测的噪声 (ego_120 || agent_120 concatenated)
        """
        batch_size = noisy_traj.size(0)
        device = noisy_traj.device

        # 1. 时间步编码
        time_embed = self.time_encoder(t)  # [batch, 256]

        # 2. 轨迹嵌入
        # 输入格式: [batch, 240] = [ego_120 || agent_120] (concat)
        # 重排为: [batch, 60, 4] = (ego_dx_t, ego_dy_t, agent_dx_t, agent_dy_t) per timestep
        ego_inc   = noisy_traj[:, :120].view(batch_size, self.config.num_steps, 2)
        agent_inc = noisy_traj[:, 120:].view(batch_size, self.config.num_steps, 2)
        traj_seq  = torch.cat([ego_inc, agent_inc], dim=-1)  # [batch, 60, 4] interleaved
        traj_embed = self.traj_embed(traj_seq)  # [batch, 60, hidden_dim]

        # 3. 组合条件
        # 时间嵌入 + 条件向量
        combined_condition = condition + time_embed  # [batch, 256]
        condition_embed = self.condition_proj(combined_condition)  # [batch, hidden_dim]

        # 4. 条件注入
        if self.config.condition_injection == 'cross_attention':
            # Cross-attention: condition作为key/value
            condition_kv = condition_embed.unsqueeze(1).expand(-1, self.config.num_steps, -1)  # [batch, 60, hidden_dim]

            # Cross-attention
            traj_with_condition, _ = self.cross_attn(
                query=traj_embed,
                key=condition_kv,
                value=condition_kv
            )
            traj_input = traj_embed + traj_with_condition  # 残差连接

        elif self.config.condition_injection == 'concat':
            # 拼接注入
            condition_seq = condition_embed.unsqueeze(1).expand(-1, self.config.num_steps, -1)
            traj_input = torch.cat([traj_embed, condition_seq], dim=-1)  # [batch, 60, hidden_dim*2]
            traj_input = self.input_proj(traj_input)  # [batch, 60, hidden_dim]

        else:  # 'addition'
            # 加法注入：直接加到每个时间步
            condition_seq = condition_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
            traj_input = traj_embed + condition_seq  # [batch, 60, hidden_dim]

        # 5. Transformer去噪
        denoised = self.transformer(traj_input)  # [batch, 60, hidden_dim]

        # 6. 输出预测噪声
        # [batch, 60, hidden_dim] -> [batch, 60, 4] interleaved -> [batch, 240] concat
        noise_pred = self.output_proj(denoised)  # [batch, 60, 4]
        # 逆重排: interleaved (ego_dx_t, ego_dy_t, agent_dx_t, agent_dy_t) -> concat (ego_120 || agent_120)
        ego_out    = noise_pred[:, :, :2].reshape(batch_size, 120)
        agent_out  = noise_pred[:, :, 2:].reshape(batch_size, 120)
        noise_pred = torch.cat([ego_out, agent_out], dim=-1)  # [batch, 240]

        return noise_pred


# ============================================================
# Diffusion主模型
# ============================================================

class DangerSceneDiffusion(nn.Module):
    """
    危险场景Diffusion生成器

    基于DDPM的去噪扩散模型，用于生成比GT更危险的轨迹。

    训练流程：
    1. 随机采样时间步 t
    2. 添加噪声：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    3. 预测噪声：noise_pred = denoiser(x_t, condition, t)
    4. 计算loss：MSE(noise_pred, noise)

    生成流程：
    1. 初始化纯噪声 x_T
    2. 逐步去噪：从T到1
    3. 最终得到 x_0（生成的轨迹增量）
    """

    def __init__(self, config: DiffusionConfig = None):
        super().__init__()

        if config is None:
            config = DiffusionConfig()

        self.config = config

        # 去噪网络
        self.denoiser = TrajectoryDenoiser(config)

        # 注册Diffusion参数
        self._register_diffusion_params()

        print(f"[DangerSceneDiffusion] 初始化完成")
        print(f"  - Num diffusion steps: {config.num_diffusion_steps}")
        print(f"  - Beta schedule: {config.beta_schedule}")
        print(f"  - Trajectory dim: {config.trajectory_dim}")

    def _register_diffusion_params(self):
        """注册DDPM的beta、alpha等参数"""
        num_steps = self.config.num_diffusion_steps

        # Beta schedule
        if self.config.beta_schedule == 'linear':
            betas = torch.linspace(self.config.beta_start, self.config.beta_end, num_steps)
        elif self.config.beta_schedule == 'cosine':
            # Cosine schedule (更平滑)
            betas = self._cosine_beta_schedule(num_steps)
        else:
            betas = torch.linspace(self.config.beta_start, self.config.beta_end, num_steps)

        # 计算alpha
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])

        # 注册buffer（不参与梯度计算）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)

        # 计算后验分布参数
        posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alpha_bars_prev) / (1 - alpha_bars))
        self.register_buffer('posterior_mean_coef2', (1 - alpha_bars_prev) * torch.sqrt(alphas) / (1 - alpha_bars))

    def _cosine_beta_schedule(self, num_steps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule

        参考：Improved DDPM (Nichol & Dhariwal, 2021)
        """
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加噪声（前向扩散）

        Args:
            x_0: [batch, 120] 原始轨迹增量（GT）
            t: [batch] 时间步索引

        Returns:
            (x_t, noise): 带噪声的轨迹和实际添加的噪声
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # 获取alpha_bar_t
        alpha_bar_t = self.alpha_bars[t]  # [batch]

        # 生成噪声
        noise = torch.randn_like(x_0)  # [batch, 120]

        # 添加噪声：x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        x_t = alpha_bar_t.sqrt().unsqueeze(-1) * x_0 + \
              (1 - alpha_bar_t).sqrt().unsqueeze(-1) * noise

        return x_t, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        单步去噪

        Args:
            x_t: [batch, 120] 当前带噪声的轨迹
            condition: [batch, 256] 条件向量
            t: [batch] 当前时间步

        Returns:
            [batch, 120] 去噪后的轨迹（预测x_0）
        """
        # 预测噪声
        noise_pred = self.denoiser(x_t, condition, t)

        # 获取参数
        alpha_bar_t = self.alpha_bars[t]
        alpha_t = self.alphas[t]

        # 从x_t和预测噪声恢复x_0
        # x_0 = (x_t - sqrt(1 - alpha_bar) * noise_pred) / sqrt(alpha_bar)
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt().unsqueeze(-1) * noise_pred) / \
                   alpha_bar_t.sqrt().unsqueeze(-1)

        return x_0_pred

    def sample_step(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        采样步骤（生成时的单步去噪）

        使用DDPM采样公式：
        x_{t-1} = mean + sigma * noise

        Args:
            x_t: [batch, 120]
            condition: [batch, 256]
            t: 当前时间步（整数）

        Returns:
            [batch, 120] x_{t-1}
        """
        batch_size = x_t.size(0)
        device = x_t.device

        # 时间步tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 预测噪声
        noise_pred = self.denoiser(x_t, condition, t_tensor)

        # 计算后验均值
        # mean = coef1 * x_0_pred + coef2 * x_t
        alpha_bar_t = self.alpha_bars[t]
        alpha_t = self.alphas[t]

        # x_0预测
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

        # 后验均值
        mean = self.posterior_mean_coef1[t] * x_0_pred + \
               self.posterior_mean_coef2[t] * x_t

        # 添加噪声（除了最后一步）
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.posterior_variance[t].sqrt()
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    def generate(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        从噪声生成轨迹（完整去噪过程）

        Args:
            condition: [batch, 256] 条件向量
            num_samples: 每个条件生成的样本数（用于多样性）
            return_intermediate: 是否返回中间步骤（用于分析）

        Returns:
            [batch, 120] 生成的轨迹增量
            或 [batch, T, 120] 如果return_intermediate=True
        """
        batch_size = condition.size(0) if num_samples == 1 else condition.size(0) * num_samples
        device = condition.device

        # 如果num_samples > 1，扩展condition
        if num_samples > 1:
            condition = condition.unsqueeze(1).expand(-1, num_samples, -1)
            condition = condition.view(batch_size, -1)

        # 初始化纯噪声
        x = torch.randn(batch_size, self.config.trajectory_dim, device=device)

        # 记录中间步骤
        if return_intermediate:
            intermediate = torch.zeros(batch_size, self.config.num_diffusion_steps, self.config.trajectory_dim, device=device)

        # 逐步去噪
        for t in reversed(range(self.config.num_diffusion_steps)):
            x = self.sample_step(x, condition, t)

            if return_intermediate:
                intermediate[:, t] = x

        if return_intermediate:
            return x, intermediate

        # Guard against NaN/Inf; no kinematic clamp — model outputs (Δx,Δy) space
        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        return x

    def sdedit_generate(
        self,
        x_gt_kinematic: torch.Tensor,
        condition: torch.Tensor,
        k: int = 40,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """[DEPRECATED] SDEdit for old kinematic (Δv,Δθ) parameterisation.

        Use DangerSceneGenerator._sdedit_with_guidance() instead — it works in
        (Δx,Δy) space and adds per-step danger gradient guidance.

        Args:
            x_gt_kinematic: [batch, 240] GT in kinematic (Δv,Δθ) space
            condition: [batch, 256] scene condition vector
            k: noise level (1..num_diffusion_steps-1); higher → more diverse, less GT-anchored
            num_samples: number of stochastic draws per batch item

        Returns:
            [batch*num_samples, 240] generated kinematic trajectories
        """
        batch_size = x_gt_kinematic.size(0)
        device = x_gt_kinematic.device

        if num_samples > 1:
            x_gt_kinematic = (
                x_gt_kinematic.unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(batch_size * num_samples, -1)
            )
            condition = (
                condition.unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(batch_size * num_samples, -1)
            )

        k = min(k, self.config.num_diffusion_steps - 1)
        effective_batch = x_gt_kinematic.size(0)

        # Forward: add partial noise at level k
        eps = torch.randn_like(x_gt_kinematic)
        alpha_bar_k = self.alpha_bars[k]
        x = alpha_bar_k.sqrt() * x_gt_kinematic + (1.0 - alpha_bar_k).sqrt() * eps

        # Reverse: denoise from k → 0
        for t in reversed(range(k)):
            x = self.sample_step(x, condition, t)

        # Cumulative kinematic clamp (same as generate())
        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=0.0)
        x_view = x.view(effective_batch, 60, 4)
        x_view[:, :, 0].clamp_(0.0, 2.0)
        x_view[:, :, 1].clamp_(-math.pi, math.pi)
        x_view[:, :, 2].clamp_(0.0, 2.0)
        x_view[:, :, 3].clamp_(-math.pi, math.pi)
        return x_view.view(effective_batch, -1)

    def sdedit_generate_dx_dy(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        k: int = 50,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """SDEdit for (Δx,Δy) parameterization.

        Adds partial noise to a baseline trajectory at level k, then denoises
        back to 0 guided by condition. Lower k → stays closer to baseline;
        higher k → more freedom to deviate toward dangerous behaviour.

        Args:
            x_0: [batch, 240] baseline trajectory in (Δx,Δy) space (GT or predicted)
            condition: [batch, 256] scene condition vector
            k: noise level in [1, num_diffusion_steps-1]
            num_samples: stochastic draws per batch item

        Returns:
            [batch*num_samples, 240] generated trajectories
        """
        batch_size = x_0.size(0)
        device = x_0.device

        if num_samples > 1:
            x_0 = x_0.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
            condition = condition.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)

        k = min(k, self.config.num_diffusion_steps - 1)

        eps = torch.randn_like(x_0)
        alpha_bar_k = self.alpha_bars[k]
        x = alpha_bar_k.sqrt() * x_0 + (1.0 - alpha_bar_k).sqrt() * eps

        for t in reversed(range(k)):
            x = self.sample_step(x, condition, t)

        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        return x

    # ── B2（v10 CFG）：classifier-free guidance 采样 ──────────────────────────
    # Phase 0 体检证 v9 的 danger 条件被忽略（conditioning collapse）。配合 B1 的属性 dropout
    # 重训后，danger 在 cond/uncond 两路产生真差异，这里在每个去噪步对预测噪声做 CFG 放大：
    #   eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)
    # cond_condition = 真 danger_target 的 condition；uncond_condition = force_null_danger 的 condition
    # （scene/svo/agent 分量不变，仅 danger 置 null）→ 放大的是"危险属性"这一个自由度。
    def sample_step_cfg(
        self,
        x_t: torch.Tensor,
        cond_condition: torch.Tensor,
        uncond_condition: torch.Tensor,
        t: int,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """CFG 版单步采样（替换 sample_step 的 noise 预测为 CFG 组合，后验公式不变）。

        Args:
            x_t:               [batch, trajectory_dim]
            cond_condition:    [batch, 256] danger=真目标
            uncond_condition:  [batch, 256] danger=null（其余分量与 cond 相同）
            t:                 当前时间步（整数）
            guidance_scale:    w。w=0 → 纯 uncond；w=1 → 纯 cond；w>1 → 放大危险属性。
        """
        batch_size = x_t.size(0)
        device = x_t.device
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 两路噪声预测 + CFG 组合
        noise_cond = self.denoiser(x_t, cond_condition, t_tensor)
        noise_uncond = self.denoiser(x_t, uncond_condition, t_tensor)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        alpha_bar_t = self.alpha_bars[t]
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

        mean = self.posterior_mean_coef1[t] * x_0_pred + \
               self.posterior_mean_coef2[t] * x_t
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.posterior_variance[t].sqrt()
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean
        return x_t_minus_1

    @torch.no_grad()
    def generate_cfg(
        self,
        cond_condition: torch.Tensor,
        uncond_condition: torch.Tensor,
        guidance_scale: float = 1.0,
        x_0: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """CFG 生成，统一覆盖两种起点（应对 Phase 0 的"降 x0 依赖"判定）：

        - x_0 is None      → 从纯噪声完整去噪（最弱 x0 依赖，danger 条件须扛全部信息）；
        - x_0 给 + k 给     → SDEdit：x_0 加噪到 level k 再去噪（k 越大越脱离 x0 基线）。

        cond/uncond_condition 由 encoder 分别用真 danger_target 与 force_null_danger=True 构建。
        返回 [batch, trajectory_dim]。
        """
        device = cond_condition.device
        batch_size = cond_condition.size(0)

        if x_0 is None:
            x = torch.randn(batch_size, self.config.trajectory_dim, device=device)
            t_start = self.config.num_diffusion_steps
        else:
            k = min(k if k is not None else self.config.num_diffusion_steps - 1,
                    self.config.num_diffusion_steps - 1)
            eps = torch.randn_like(x_0)
            alpha_bar_k = self.alpha_bars[k]
            x = alpha_bar_k.sqrt() * x_0 + (1.0 - alpha_bar_k).sqrt() * eps
            t_start = k

        for t in reversed(range(t_start)):
            x = self.sample_step_cfg(x, cond_condition, uncond_condition, t,
                                     guidance_scale=guidance_scale)

        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        return x

    def sdedit_with_svo_guidance(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: Optional[torch.Tensor] = None,
        k: int = 50,
        svo_target_deg: float = -45.0,
        svo_guidance_scale: float = 1.0,
        svo_config: Optional[SvoTorchConfig] = None,
    ) -> torch.Tensor:
        """SDEdit with DPS-style SVO gradient guidance.

        Adds partial noise at level k, then denoises back to 0 while applying
        a per-step SVO gradient correction that steers the generated trajectory
        toward the target SVO angle (default -45°, maximally competitive/aggressive).

        The guidance uses the approximate DPS formulation: at each denoising step t,
        we compute x̂_0 from x_t (treating the noise prediction as a constant),
        evaluate the SVO loss on x̂_0, and correct x_t by the gradient scaled by
        sqrt(1-ᾱ_t) * svo_guidance_scale before computing x_{t-1}.

        Args:
            x_0: (B, 240) baseline trajectory in (dx, dy) space.
            condition: (B, 256) scene condition vector.
            ego_history: (B, T_hist, 5) or (T_hist, 5) ego historical trajectory.
            agent_history: (B, T_hist, 5) or (T_hist, 5) agent history, optional.
            k: SDEdit noise level in [1, num_diffusion_steps-1].
            svo_target_deg: target SVO angle in degrees. -45 = maximally competitive.
            svo_guidance_scale: DPS step size multiplier (larger = stronger guidance).
            svo_config: SvoTorchConfig, uses defaults if None.

        Returns:
            (B, 240) generated trajectories.
        """
        batch_size = x_0.size(0)
        device = x_0.device

        k = min(k, self.config.num_diffusion_steps - 1)
        if svo_config is None:
            svo_config = SvoTorchConfig()

        # ── SDEdit: add partial noise at level k ────────────────────────────
        eps = torch.randn_like(x_0)
        alpha_bar_k = self.alpha_bars[k]
        x = alpha_bar_k.sqrt() * x_0 + (1.0 - alpha_bar_k).sqrt() * eps

        # ── Denoising loop with SVO guidance ────────────────────────────────
        for t in reversed(range(k)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[t]
            alpha_t = self.alphas[t]

            # Standard noise prediction (no grad through denoiser params)
            with torch.no_grad():
                noise_pred = self.denoiser(x, condition, t_tensor)

            # ── DPS guidance step ────────────────────────────────────────
            if svo_guidance_scale > 0.0:
                # Leaf tensor for gradient computation
                x_leaf = x.detach().requires_grad_(True)
                # Approximate x̂_0 (noise_pred treated as constant)
                x0_approx = (x_leaf - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
                x0_approx = torch.nan_to_num(x0_approx, nan=0.0, posinf=4.0, neginf=-4.0)

                svo_loss = svo_dps_loss(
                    x0_approx, ego_history, agent_history,
                    target_svo_deg=svo_target_deg, config=svo_config
                )
                svo_grad = torch.autograd.grad(svo_loss, x_leaf)[0]  # (B, 240)

                # Normalize gradient to unit direction so svo_guidance_scale has
                # a consistent meaning (meters of displacement per step) regardless
                # of loss magnitude or noise level.
                g = svo_grad.detach()
                g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                guidance = svo_guidance_scale * (g / g_norm)
                x_corrected = x.detach() - guidance
            else:
                x_corrected = x

            # ── DDPM posterior step from corrected x_t ───────────────────
            x_0_pred = (x_corrected - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

            mean = self.posterior_mean_coef1[t] * x_0_pred + \
                   self.posterior_mean_coef2[t] * x_corrected

            if t > 0:
                noise = torch.randn_like(x_corrected)
                sigma = self.posterior_variance[t].sqrt()
                x = mean + sigma * noise
            else:
                x = mean

        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        return x

    # ============================================================
    # [v6.0] Private helpers
    # ============================================================

    def _kinematic_realism_loss_v6_0(
        self,
        ego_dxdy_flat: torch.Tensor,
        v_max: float = 0.9,
        a_max: float = 0.06,
    ) -> torch.Tensor:
        """[v6.0] Penalise physically implausible ego speed and acceleration.

        NuPlan physical limits (dt=0.1s): v_max≈0.9 m/step (9 m/s),
        a_max≈0.06 m/step velocity change per step.

        Args:
            ego_dxdy_flat: (B, 120) ego future increments, retains grad.
            v_max: maximum plausible step displacement (m/step).
            a_max: maximum plausible speed change per step.

        Returns:
            Scalar penalty (differentiable).
        """
        B = ego_dxdy_flat.shape[0]
        ego_dxdy = ego_dxdy_flat.reshape(B, 60, 2)       # (B, 60, 2)
        step_norms = ego_dxdy.norm(dim=-1)                # (B, 60) m/step
        speed_penalty = F.relu(step_norms - v_max).pow(2).mean()
        speed_diff = step_norms.diff(dim=-1)              # (B, 59)
        accel_penalty = F.relu(speed_diff.abs() - a_max).pow(2).mean()
        return speed_penalty + accel_penalty

    def _kinematic_project(
        self,
        dxdy_flat: torch.Tensor,
        v_max: float = 0.9,
        a_max: float = 0.06,
    ) -> torch.Tensor:
        """[方式二] 把增量轨迹硬投影到运动学可行域：逐帧 speed≤v_max、相邻帧 |Δspeed|≤a_max。

        只改每步位移的"大小"、保留"方向"（碰撞梯度负责定方向去哪汇聚，投影负责定速度上限），
        替代旧的安全夹子（g/g_norm+clip）来守真实性——让大梯度也驱动得动位移、但轨迹不出物理域。
        生成后投影，无需梯度。返回同形 (B,120)。

        Args:
            dxdy_flat: (B,120) ego 或 agent 未来增量。
            v_max: 每步最大位移（m/step，9 m/s）。
            a_max: 相邻帧最大速度变化（m/step）。
        """
        B = dxdy_flat.shape[0]
        dxdy = dxdy_flat.reshape(B, 60, 2)
        norms = dxdy.norm(dim=-1)                              # (B,60) 每步速度大小
        direction = dxdy / norms.clamp(min=1e-8).unsqueeze(-1)  # 单位方向（保留）
        # 顺序投影：t=0 先夹 v_max；t≥1 把速度夹进 [prev-a_max, prev+a_max] 且 ≤v_max
        new_norms = norms.clone()
        new_norms[:, 0] = norms[:, 0].clamp(max=v_max)
        for t in range(1, 60):
            lo = (new_norms[:, t - 1] - a_max).clamp(min=0.0)
            hi = (new_norms[:, t - 1] + a_max).clamp(max=v_max)
            new_norms[:, t] = torch.min(torch.max(norms[:, t], lo), hi)
        projected = direction * new_norms.unsqueeze(-1)
        return projected.reshape(B, 120)

    def _lowpass_guidance_v6_0(
        self,
        guidance: torch.Tensor,
        cutoff_ratio: float = 0.3,
    ) -> torch.Tensor:
        """[v6.0] FFT temporal low-pass filter on guidance vector (六A).

        Removes high-frequency artefacts from the gradient signal to improve
        temporal smoothness of guidance-induced trajectory changes.

        Args:
            guidance: (B, 240) raw guidance vector (same layout as trajectory).
            cutoff_ratio: fraction of temporal frequency bins to keep (0–1).

        Returns:
            (B, 240) filtered guidance.
        """
        B = guidance.shape[0]
        ego   = guidance[:, :120].reshape(B, 60, 2)
        agent = guidance[:, 120:].reshape(B, 60, 2)
        g_seq = torch.cat([ego, agent], dim=-1)           # (B, 60, 4)

        G = torch.fft.rfft(g_seq, dim=1)                  # (B, 31, 4) complex
        n_freq = G.shape[1]
        cutoff = max(1, int(cutoff_ratio * n_freq))
        G_f = G.clone()
        G_f[:, cutoff:, :] = 0.0
        g_out = torch.fft.irfft(G_f, n=60, dim=1)         # (B, 60, 4)

        ego_f   = g_out[:, :, :2].reshape(B, 120)
        agent_f = g_out[:, :, 2:].reshape(B, 120)
        return torch.cat([ego_f, agent_f], dim=-1)

    def _compute_baseline_svo_v6_0(
        self,
        x0_gt: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: Optional[torch.Tensor],
        svo_config: SvoTorchConfig,
    ) -> torch.Tensor:
        """[v6.0] Compute per-sample GT baseline SVO for relative target.

        Args:
            x0_gt: (B, 240) baseline trajectory in (dx, dy) space.
            ego_history: (B, T_hist, 5) or (T_hist, 5).
            agent_history: (B, T_hist, 5) or (T_hist, 5), optional.
            svo_config: SvoTorchConfig.

        Returns:
            (B,) SVO angles in degrees (no grad).
        """
        B = x0_gt.shape[0]
        ego_dxdy   = x0_gt[:, :120].detach().reshape(B, 60, 2)
        agent_dxdy = x0_gt[:, 120:].detach().reshape(B, 60, 2)

        svo_degs = []
        for i in range(B):
            hist_i = ego_history[i] if ego_history.dim() == 3 else ego_history
            agent_pos = None
            if agent_history is not None:
                ah_i = (agent_history[i]
                        if agent_history.dim() == 3 else agent_history)
                agent_last = ah_i[-1, :2].to(
                    dtype=x0_gt.dtype, device=x0_gt.device)
                agent_pos = (agent_last.unsqueeze(0)
                             + agent_dxdy[i].cumsum(dim=0))
            svo_degs.append(
                compute_ego_svo_torch(ego_dxdy[i], hist_i, agent_pos, svo_config)
            )
        return torch.stack(svo_degs)                      # (B,) degrees

    def sdedit_with_svo_guidance_v6_0(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: Optional[torch.Tensor] = None,
        k: int = 50,
        svo_guidance_scale: float = 1.0,
        use_relative_target: bool = True,
        relative_delta_deg: float = 15.0,
        svo_target_deg: float = -45.0,
        svo_config: Optional[SvoTorchConfig] = None,
        lambda_kinematic: float = 0.3,
        time_varying_scale: bool = True,
        grad_clip_norm: float = 1.0,
        use_lowpass: bool = True,
        lowpass_cutoff_ratio: float = 0.3,
        use_bezier: bool = False,
        bezier_n_ctrl: int = 10,
        debug: bool = False,
    ) -> torch.Tensor:
        """[v6.0] SDEdit with joint SVO + kinematic DPS guidance.

        Research framing (v6.0):
        Goal — **bounded counterfactual editing near the GT trajectory**, not
        pushing SVO to an extreme absolute value.  The relative target
        (baseline_svo − delta) encodes "slightly more competitive than GT" and
        keeps generated scenarios interpretable and close to the data manifold.

        v6.0 improvements over v5.0:
        (1) Relative SVO target  — target = baseline_svo − relative_delta_deg
            (default δ=15°).  Keeps edits local to the GT; avoids OOD regime.
            Absolute target (use_relative_target=False, svo_target_deg=−45°)
            available for ablation — it exploits prior/guidance tension to
            produce near-misses but is not the primary research narrative.
        (2) Time-varying scale   — eff_scale(t) = scale × (t+1)/k; guidance
            is strongest when trajectory is malleable (high t), weak near t=0.
        (3) Joint loss           — L = L_SVO + lambda_kinematic × L_kinematic
            Kinematic realism term (ego speed + accel limits) is applied
            inside the guidance loop, not only at decode/eval time.
        (4) Gradient norm clip   — hard cap on per-step guidance magnitude.
        (5) FFT low-pass filter  — removes high-frequency gradient artefacts
            so guidance reflects smooth driving-style changes (六A).
        (6) Bezier control-point smoothing — projects ego guidance through
            n_ctrl-DoF smooth-curve space before applying (六B, optional).

        Known gap (next step — interaction realism):
        Current gradient only constrains ego kinematics.  Agent trajectory is
        passive: it does not respond to ego's modified behaviour.  A future
        L_interaction term should penalise physically implausible relative
        motion (e.g. impossible gap-closing, agent teleport) to produce
        genuine two-vehicle game scenarios rather than one-sided ego edits.

        Args:
            x_0: (B, 240) baseline trajectory in (dx, dy) space.
            condition: (B, 256) scene condition vector.
            ego_history: (B, T_hist, 5) or (T_hist, 5) [x,y,hdg,vx,vy].
            agent_history: (B, T_hist, 5) or (T_hist, 5), optional.
            k: SDEdit noise level in [1, num_diffusion_steps−1].
            svo_guidance_scale: physical step size (m/step) for guidance.
            use_relative_target: use baseline_svo − delta as SVO target.
            relative_delta_deg: SVO delta below baseline (degrees).
            svo_target_deg: absolute target (when use_relative_target=False).
            svo_config: SvoTorchConfig, uses defaults if None.
            lambda_kinematic: weight for kinematic realism loss.
            time_varying_scale: apply eff_scale(t) = scale × (t+1)/k.
            grad_clip_norm: maximum L2 norm of guidance per step (m/step).
            use_lowpass: apply FFT temporal low-pass to guidance.
            lowpass_cutoff_ratio: fraction of temporal frequencies to keep.
            use_bezier: project ego guidance through Bezier space (六B).
            bezier_n_ctrl: number of Bezier control points (default 10).

        Returns:
            (B, 240) generated trajectories in (dx, dy) space.
        """
        batch_size = x_0.size(0)
        device     = x_0.device
        k = min(k, self.config.num_diffusion_steps - 1)

        if svo_config is None:
            svo_config = SvoTorchConfig()

        # [v6.0] Per-sample SVO target from GT baseline ──────────────────────
        if use_relative_target:
            with torch.no_grad():
                baseline_svo = self._compute_baseline_svo_v6_0(
                    x_0, ego_history, agent_history, svo_config)
            target_per_sample = (baseline_svo - relative_delta_deg).to(device)
        else:
            target_per_sample = torch.full(
                (batch_size,), svo_target_deg,
                device=device, dtype=x_0.dtype)

        # SDEdit: add partial noise at level k ────────────────────────────────
        eps         = torch.randn_like(x_0)
        alpha_bar_k = self.alpha_bars[k]
        x = alpha_bar_k.sqrt() * x_0 + (1.0 - alpha_bar_k).sqrt() * eps

        # [v6.0] Denoising loop with joint SVO + kinematic guidance ──────────
        _diag_steps: list = []
        for t in reversed(range(k)):
            t_tensor    = torch.full((batch_size,), t, device=device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[t]

            with torch.no_grad():
                noise_pred = self.denoiser(x, condition, t_tensor)

            if svo_guidance_scale > 0.0:
                x_leaf = x.detach().requires_grad_(True)
                x0_approx = (
                    x_leaf - (1 - alpha_bar_t).sqrt() * noise_pred
                ) / alpha_bar_t.sqrt()
                x0_approx = torch.nan_to_num(
                    x0_approx, nan=0.0, posinf=4.0, neginf=-4.0)

                B_inner = x0_approx.shape[0]
                ego_dxdy     = x0_approx[:, :120].reshape(B_inner, 60, 2)
                agent_dxdy_d = x0_approx[:, 120:].detach().reshape(B_inner, 60, 2)

                # Per-sample SVO with relative target ─────────────────────────
                svo_list = []
                for i in range(B_inner):
                    hist_i = (ego_history[i]
                              if ego_history.dim() == 3 else ego_history)
                    agent_pos = None
                    if agent_history is not None:
                        ah_i = (agent_history[i]
                                if agent_history.dim() == 3 else agent_history)
                        agent_last = ah_i[-1, :2].to(
                            dtype=x0_approx.dtype, device=device)
                        agent_pos = (agent_last.unsqueeze(0)
                                     + agent_dxdy_d[i].cumsum(dim=0))
                    svo_list.append(
                        compute_ego_svo_torch(
                            ego_dxdy[i], hist_i, agent_pos, svo_config))
                svo_batch = torch.stack(svo_list)          # (B,)

                # SVO loss: push below per-sample target ──────────────────────
                svo_loss   = F.relu(svo_batch - target_per_sample).mean()

                # [v6.0] Kinematic realism loss (ego only) ────────────────────
                kin_loss   = self._kinematic_realism_loss_v6_0(x0_approx[:, :120])
                total_loss = svo_loss + lambda_kinematic * kin_loss

                g = torch.autograd.grad(total_loss, x_leaf)[0].detach()

                # [v6.0] FFT low-pass filter (六A) ────────────────────────────
                if use_lowpass:
                    g = self._lowpass_guidance_v6_0(
                        g, cutoff_ratio=lowpass_cutoff_ratio)

                # [v6.0] Bezier control-point smoothing on ego guidance (六B) ──
                if use_bezier:
                    ego_g_smooth = bezier_guidance_v6_0(
                        g[:, :120], n_ctrl=bezier_n_ctrl)
                    g = torch.cat([ego_g_smooth, g[:, 120:]], dim=-1)

                # Normalise to unit direction ──────────────────────────────────
                g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                g_unit = g / g_norm

                # [v6.0] Time-varying scale ───────────────────────────────────
                eff_scale = (svo_guidance_scale * (t + 1) / k
                             if time_varying_scale else svo_guidance_scale)

                guidance = eff_scale * g_unit              # (B, 240)

                # [v6.0] Per-step norm clip ───────────────────────────────────
                gout_norm = guidance.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                guidance  = guidance * (gout_norm.clamp(max=grad_clip_norm)
                                        / gout_norm)

                if debug:
                    _diag_steps.append({
                        "t": t,
                        "svo_mean": svo_batch.mean().item(),
                        "svo_target": target_per_sample.mean().item(),
                        "g_raw_norm": g_norm.mean().item(),
                        "g_clip_norm": gout_norm.mean().item(),
                        "eff_scale": eff_scale,
                        "svo_loss": svo_loss.item(),
                        "kin_loss": kin_loss.item(),
                    })

                x_corrected = x.detach() - guidance
            else:
                x_corrected = x

            # DDPM posterior step ─────────────────────────────────────────────
            x_0_pred = (
                x_corrected - (1 - alpha_bar_t).sqrt() * noise_pred
            ) / alpha_bar_t.sqrt()
            x_0_pred = torch.nan_to_num(
                x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

            mean = (self.posterior_mean_coef1[t] * x_0_pred
                    + self.posterior_mean_coef2[t] * x_corrected)

            if t > 0:
                noise = torch.randn_like(x_corrected)
                sigma = self.posterior_variance[t].sqrt()
                x     = mean + sigma * noise
            else:
                x = mean

        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)

        if debug and _diag_steps:
            print(f"\n[v6.0 diag] guidance loop (k={k}, every 5 steps shown)")
            print(f"{'t':>4}  {'svo_mean':>9}  {'target':>7}  {'g_raw':>7}  {'g_clip':>7}  "
                  f"{'eff_sc':>7}  {'L_svo':>7}  {'L_kin':>7}")
            for s in _diag_steps[::5]:
                print(f"{s['t']:>4}  {s['svo_mean']:>9.2f}°  {s['svo_target']:>6.2f}°  "
                      f"{s['g_raw_norm']:>7.4f}  {s['g_clip_norm']:>7.4f}  "
                      f"{s['eff_scale']:>7.4f}  {s['svo_loss']:>7.4f}  {s['kin_loss']:>7.4f}")
            # first vs last step SVO
            s0, sN = _diag_steps[-1], _diag_steps[0]
            print(f"\n[v6.0 diag] SVO drift: t={s0['t']} → t={sN['t']}  "
                  f"{s0['svo_mean']:.2f}° → {sN['svo_mean']:.2f}°  "
                  f"(target {sN['svo_target']:.2f}°)")
            # ego-agent min_dist in generated trajectory
            with torch.no_grad():
                ego_dxdy_out   = x[:, :120].reshape(batch_size, 60, 2)
                agent_dxdy_out = x[:, 120:].reshape(batch_size, 60, 2)
                if ego_history.dim() == 3:
                    ego_start   = ego_history[:, -1, :2].to(device=device, dtype=x.dtype)
                else:
                    ego_start   = ego_history[-1, :2].unsqueeze(0).expand(
                        batch_size, -1).to(device=device, dtype=x.dtype)
                ego_pos = ego_start.unsqueeze(1) + ego_dxdy_out.cumsum(dim=1)
                if agent_history is not None:
                    if agent_history.dim() == 3:
                        agent_start = agent_history[:, -1, :2].to(device=device, dtype=x.dtype)
                    else:
                        agent_start = agent_history[-1, :2].unsqueeze(0).expand(
                            batch_size, -1).to(device=device, dtype=x.dtype)
                    agent_pos  = agent_start.unsqueeze(1) + agent_dxdy_out.cumsum(dim=1)
                    dist       = (ego_pos - agent_pos).norm(dim=-1)   # (B, 60)
                    min_dist   = dist.min(dim=1).values                # (B,)
                    print(f"[v6.0 diag] ego-agent min_dist (generated): "
                          f"mean={min_dist.mean().item():.2f}m  "
                          f"min={min_dist.min().item():.2f}m  "
                          f"max={min_dist.max().item():.2f}m")
                else:
                    print("[v6.0 diag] agent_history=None, skip min_dist")

        return x

    def _compute_baseline_svo_pair_phase2(
        self,
        x0_gt: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: Optional[torch.Tensor],
        svo_config: SvoTorchConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[Phase2] GT baseline SVO for BOTH ego and agent (symmetric, no grad).

        ego SVO: ego trajectory is self, agent is fixed neighbour.
        agent SVO: agent trajectory is self, ego is fixed neighbour.
        Reuses compute_ego_svo_torch (which is really a "self-SVO") with the
        two roles swapped — no separate agent-SVO function needed.

        Returns:
            (ego_svo[B], agent_svo[B]) in degrees, detached.
        """
        B = x0_gt.shape[0]
        ego_dxdy   = x0_gt[:, :120].detach().reshape(B, 60, 2)
        agent_dxdy = x0_gt[:, 120:].detach().reshape(B, 60, 2)

        ego_svo_list, agent_svo_list = [], []
        for i in range(B):
            ego_hist_i = ego_history[i] if ego_history.dim() == 3 else ego_history
            ego_last   = ego_hist_i[-1, :2].to(dtype=x0_gt.dtype, device=x0_gt.device)
            ego_pos    = ego_last.unsqueeze(0) + ego_dxdy[i].cumsum(dim=0)   # (60, 2)

            agent_pos = None
            agent_hist_i = None
            if agent_history is not None:
                agent_hist_i = (agent_history[i]
                                if agent_history.dim() == 3 else agent_history)
                agent_last = agent_hist_i[-1, :2].to(
                    dtype=x0_gt.dtype, device=x0_gt.device)
                agent_pos  = agent_last.unsqueeze(0) + agent_dxdy[i].cumsum(dim=0)

            # ego self-SVO (agent neighbour)
            ego_svo_list.append(
                compute_ego_svo_torch(ego_dxdy[i], ego_hist_i, agent_pos, svo_config))
            # agent self-SVO (ego neighbour) — only if agent history available
            if agent_hist_i is not None:
                agent_svo_list.append(
                    compute_ego_svo_torch(agent_dxdy[i], agent_hist_i, ego_pos, svo_config))
            else:
                agent_svo_list.append(ego_dxdy.new_zeros(()))
        return torch.stack(ego_svo_list), torch.stack(agent_svo_list)

    @staticmethod
    def _grad_or_zeros_phase2(
        loss: torch.Tensor,
        leaf: torch.Tensor,
        retain_graph: bool,
    ) -> torch.Tensor:
        """[路线 A] 求 d(loss)/d(leaf)，loss 与 leaf 无关（或不带梯度）时返回全 0；结果 detach。

        只在 svo_step>0（SVO 梯度与其余项分开求）时使用，默认路径不经过这里。
        """
        if not (torch.is_tensor(loss) and loss.requires_grad):
            return torch.zeros_like(leaf).detach()
        grad = torch.autograd.grad(loss, leaf, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(leaf).detach()
        return grad.detach()

    def sdedit_with_full_guidance_phase2(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: Optional[torch.Tensor] = None,
        lane_polylines: Optional[torch.Tensor] = None,
        k: int = 50,
        svo_guidance_scale: float = 1.0,
        use_relative_target: bool = True,
        relative_delta_deg: float = 15.0,
        svo_target_deg: float = -45.0,
        svo_config: Optional[SvoTorchConfig] = None,
        svo_weight: float = 1.0,
        svo_timevarying: bool = True,   # True: 逐时刻时变 SVO（距离驱动避让涌现）；False: 整条标量
        proximity_weight: float = 0.5,
        proximity_tau: float = 1.0,
        near_miss_target: float = 3.0,
        lane_weight: float = 1.0,
        lane_half_width: Optional[float] = None,   # None → 用 config.road_half_width（默认 3.0m 宽松边界）
        lane_constrain_agent: bool = False,        # True → L_lane 同时约束 agent 在路面（默认只管 ego，保 S2 可复现）
        lambda_kinematic: float = 0.3,
        kin_constrain_agent: bool = False,         # True → 运动学真实性也约束 agent（默认只管 ego；与 lane_constrain_agent 配套，使 agent 与 ego 等效真实）
        time_varying_scale: bool = True,
        grad_clip_norm: float = 1.0,
        use_lowpass: bool = True,
        lowpass_cutoff_ratio: float = 0.3,
        use_bezier: bool = False,
        bezier_n_ctrl: int = 10,
        start_anchor_steps: int = 0,               # >0：生成后把前 N 步增量向历史速度混合锚定，消除起点掉头（默认关，不动 eval）
        guidance_g_ref: float = 0.0,               # >0：软归一化 g/(|g|+g_ref)，约束将满足时推力衰减→0，消 chattering 震荡（0=旧单位向量行为）
        guidance_stop_t: int = 0,                  # >0：t<此值的末段步只纯去噪、不施 guidance，让尾段把轨迹抚平（0=全程施引导）
        # ── 方式二（碰撞导向 guidance）开关：默认全 False/0 = 现行为，保 S2/eval 复现 ──
        collision_mode: bool = False,              # True：proximity 目标→collision_target（碰撞），proximity 成主驱动
        collision_target: float = 0.0,             # collision_mode 下 proximity 拉向的目标距离（0=接触）
        disable_safety_clamp: bool = False,        # True：跳过 g/g_norm 单位化+clip，让原始梯度按 eff_scale 直接驱动位移
        kinematic_projection: bool = False,        # True：每步对 x0_pred 的 ego/agent 增量做运动学硬投影（替代安全夹子守真实性）
        magnitude_cap: float = 0.0,                # >0：原始梯度宽松幅度上限（爆炸保险，远大于旧 1.0，不钉死方向）
        # ── 方式二精修①(b)：closing-speed 加权 proximity 目标（默认 'softmin' = 旧行为）──
        # 'closing' = soft-min 距离拉近（主驱动，保留 softmin 汇聚力）+ closing 修正项（罚近距停滞/发散）
        proximity_objective: str = 'softmin',      # 'softmin'：旧 soft-min 距离目标；'closing'：softmin 拉近 + closing 修正；'danger_score'：softmin 拉近 + maximize eval outcome_danger（(c) 闭合）
        closing_horizon: float = 20.0,             # closing 修正的时间权重尺度(step)：w_t=exp(-t_idx/horizon)，早段强（无距离 gate=无逃逸漏洞）
        closing_margin: float = 0.1,               # 期望最小 closing 速度(m/step)；closing<margin（停滞/发散）才罚
        closing_weight: float = 20.0,              # closing 修正项相对 soft-min 拉近项的权重（修正强度）
        closing_eps: float = 0.3,                  # closing 模式 r̂ 归一化的距离下限(m)，防接触处方向爆炸
        # ── 方式二精修③(c)：eval-closed 可微 danger_score 目标（'danger_score' 模式）──
        danger_score_weight: float = 1.0,          # maximize(-outcome_danger) 项相对 soft-min pull 的权重
        danger_dt: float = 0.1,                    # vel=dxdy/danger_dt 的 dt（与 eval DT 一致，使 TTC 单位=秒）
        danger_score_config: Optional["DangerScoreConfigTorch"] = None,  # None→DangerScoreConfigTorch()（默认即与 eval numpy 配置一致）
        # ── 方式二精修②：角色不对称 SVO（ego 激进 pursuer / agent 轻 reactor）──
        ego_relative_delta_deg: Optional[float] = None,    # None→relative_delta_deg；越大 ego 越激进（target=baseline−delta 更竞争）
        agent_relative_delta_deg: Optional[float] = None,  # None→relative_delta_deg；越小 agent 越被动（靠 joint 先验反应）
        debug: bool = False,
        # ── 路线 A（09-16）SVO guidance 修复开关：默认值 = 现行为（逐位不变），新参数一律放在 debug 之后 ──
        svo_measure: str = 'timevarying',          # 'timevarying'：旧口径（由 svo_timevarying 决定逐帧/标量）；'soft_cumulative'：可微软累加 SVO（用户 09-13 累加定义的软化版）
        svo_soft_cfg: Optional[dict] = None,       # 软累加参数；None → default_svo_soft_cfg() = dict(dth=3, ath=2, sc=1.6, acc_smooth=5, seam_skip=4)；给出时只覆盖给出的键
        svo_loss_mode: str = 'relu',               # 'relu'：旧单向 relu(α−target).mean()；'two_sided'：只用末帧 α(T)，smooth_l1(beta=1°) 双向拉向目标
        svo_step: float = 0.0,                     # >0：SVO 梯度与其余项分开求，lowpass 后按样本单位化，×svo_step×eff_scale 叠加（0 = 旧「一个总梯度」路径）
        agent_svo_target_deg: Optional[float] = None,  # 仅 use_relative_target=False 时生效；None → agent 与 ego 同用 svo_target_deg
    ) -> torch.Tensor:
        """[Phase2] SDEdit with full three-term DPS guidance.

        Extends the v6.0 SVO+kinematic guidance with three additions, for
        DIRECTION-ONLY verification on an old v5.0 checkpoint (not paper data):

        (1) Symmetric SVO — both ego AND agent are pushed toward a competitive
            (relative) SVO target.  Unlike v6.0 the agent trajectory is NOT
            detached, so gradient flows into the agent half (240D) too,
            producing a genuine two-vehicle game rather than a one-sided ego
            edit.  agent SVO reuses compute_ego_svo_torch with roles swapped.
        (2) Soft-min proximity — differentiable LogSumEx soft-min over time of
            ego-agent distance, pulled toward the near-miss band (~3m).
            Smoother gradient than hard-min argmax.
        (3) L_lane — lane deviation penalty (lane_deviation_score_torch) keeps
            the ego trajectory on-road.  min-over-segments makes lane-change
            free (penalises only when off ALL nearby lanes).

        Args mirror sdedit_with_svo_guidance_v6_0 plus:
            lane_polylines: (B, L, 4) padded ego-centric lane segments
                [x0,y0,x1,y1]; required for L_lane (else lane term skipped).
            svo_weight / proximity_weight / lane_weight: term weights.
            proximity_tau: LogSumExp temperature (smaller -> closer to hard-min).
            near_miss_target: target soft-min distance (m); proximity pulls
                ego-agent together when farther than this.
            lane_half_width: lane half-width threshold (m).

        路线 A（09-16）新增参数（默认值 = 现行为，逐位不变）：
            svo_measure: 'soft_cumulative' 时 SVO 度量改为 compute_svo_soft_cumulative_torch
                （ego 初始量取历史真实段硬计数，agent 初始 0；只依赖各自速度），
                此时 svo_timevarying / svo_config 不参与 SVO 项；use_relative_target=True 时
                基线也用同一软累加函数在 x_0 上算（逐帧 (B,T)，no_grad）再减 delta。
            svo_soft_cfg: 软累加参数（dth/ath/sc/acc_smooth/seam_skip），None 取默认。
            svo_loss_mode: 'two_sided' 时 SVO loss = smooth_l1(α(T), target(T), beta=1°) 的 batch 均值，
                ego 与 agent（若有）各一项相加；'relu' 沿用 relu(α−target).mean()。
            svo_step: >0 时 SVO 梯度 g_svo = d(svo_weight·svo_loss)/dx 单独求，其余项梯度走原路径
                （lowpass / bezier / 夹子分支逻辑不变）；g_svo 做同样的 lowpass（若开）后按样本单位化
                （范数 < 1e-8 的样本置 0），再乘 svo_step·eff_scale 叠加进 guidance。
            agent_svo_target_deg: 绝对目标模式下 agent 的目标角；None 时与 ego 相同。

        Returns:
            (B, 240) generated trajectories in (dx, dy) space.

        Note (known limitation): svo_torch lateral terms are algebraically
        ~zero-gradient (heading from atan2(dy,dx)); dominant SVO gradient comes
        from longitudinal accel and social force.  This affects ego and agent
        SVO equally — verification checks DIRECTION, not magnitude.
        """
        batch_size = x_0.size(0)
        device     = x_0.device
        k = min(k, self.config.num_diffusion_steps - 1)

        # 方式二：碰撞模式下 proximity 拉向接触（collision_target，默认 0），成为主驱动力
        if collision_mode:
            near_miss_target = collision_target

        if svo_config is None:
            svo_config = SvoTorchConfig()
        # 路线 A：新开关校验 + 软累加参数解析（纯 Python，默认值时不引入任何张量计算）
        if svo_measure not in ('timevarying', 'soft_cumulative'):
            raise ValueError(f"svo_measure 须为 'timevarying' 或 'soft_cumulative'，得到 {svo_measure!r}")
        if svo_loss_mode not in ('relu', 'two_sided'):
            raise ValueError(f"svo_loss_mode 须为 'relu' 或 'two_sided'，得到 {svo_loss_mode!r}")
        _use_soft_svo = svo_measure == 'soft_cumulative'
        _soft_cfg = default_svo_soft_cfg()
        if svo_soft_cfg is not None:
            _soft_cfg.update(svo_soft_cfg)
        # 方式二(c)：解析 danger_score 配置（默认与 eval numpy DangerScoreConfig 阈值一致 → 闭合 eval）
        _danger_cfg = danger_score_config if danger_score_config is not None else DangerScoreConfigTorch()
        if lane_half_width is None:
            lane_half_width = self.config.road_half_width   # 宽松边界，config 可设

        # [Phase2] Per-sample relative SVO targets for BOTH ego and agent ──────
        if use_relative_target:
            if _use_soft_svo:
                # 路线 A：基线与 guidance 同口径——软累加函数在 x_0 上算逐帧 α (B,T)（no_grad），再减 delta
                with torch.no_grad():
                    base_ego_svo = compute_svo_soft_cumulative_torch(
                        x_0[:, :120].detach().reshape(batch_size, 60, 2), ego_history, _soft_cfg)
                    if agent_history is not None:
                        base_agent_svo = compute_svo_soft_cumulative_torch(
                            x_0[:, 120:].detach().reshape(batch_size, 60, 2), None, _soft_cfg)
                    else:
                        base_agent_svo = torch.zeros_like(base_ego_svo)
            else:
                with torch.no_grad():
                    base_ego_svo, base_agent_svo = self._compute_baseline_svo_pair_phase2(
                        x_0, ego_history, agent_history, svo_config)
            # 角色不对称（精修②）：ego/agent 各自 delta，None 时回落到统一 relative_delta_deg（向后兼容）
            _ego_delta   = (relative_delta_deg if ego_relative_delta_deg is None
                            else ego_relative_delta_deg)
            _agent_delta = (relative_delta_deg if agent_relative_delta_deg is None
                            else agent_relative_delta_deg)
            ego_target   = (base_ego_svo   - _ego_delta).to(device)
            agent_target = (base_agent_svo - _agent_delta).to(device)
        else:
            ego_target   = torch.full((batch_size,), svo_target_deg,
                                      device=device, dtype=x_0.dtype)
            if agent_svo_target_deg is None:
                agent_target = ego_target.clone()
            else:
                # 路线 A：agent 单独的绝对目标角
                agent_target = torch.full((batch_size,), agent_svo_target_deg,
                                          device=device, dtype=x_0.dtype)

        # Batched last positions (for ego/agent absolute position reconstruction)
        if ego_history.dim() == 3:
            ego_last_xy = ego_history[:, -1, :2].to(device=device, dtype=x_0.dtype)
        else:
            ego_last_xy = ego_history[-1, :2].unsqueeze(0).expand(
                batch_size, -1).to(device=device, dtype=x_0.dtype)
        agent_last_xy = None
        if agent_history is not None:
            if agent_history.dim() == 3:
                agent_last_xy = agent_history[:, -1, :2].to(device=device, dtype=x_0.dtype)
            else:
                agent_last_xy = agent_history[-1, :2].unsqueeze(0).expand(
                    batch_size, -1).to(device=device, dtype=x_0.dtype)

        # SDEdit: add partial noise at level k ────────────────────────────────
        eps         = torch.randn_like(x_0)
        alpha_bar_k = self.alpha_bars[k]
        x = alpha_bar_k.sqrt() * x_0 + (1.0 - alpha_bar_k).sqrt() * eps

        _diag_steps: list = []
        for t in reversed(range(k)):
            t_tensor    = torch.full((batch_size,), t, device=device, dtype=torch.long)
            alpha_bar_t = self.alpha_bars[t]

            with torch.no_grad():
                noise_pred = self.denoiser(x, condition, t_tensor)

            # 末段纯去噪（t<guidance_stop_t）：尾段只让 denoiser 收敛、不施 guidance，抚平锯齿
            if svo_guidance_scale > 0.0 and t >= guidance_stop_t:
                x_leaf = x.detach().requires_grad_(True)
                x0_approx = (
                    x_leaf - (1 - alpha_bar_t).sqrt() * noise_pred
                ) / alpha_bar_t.sqrt()
                x0_approx = torch.nan_to_num(
                    x0_approx, nan=0.0, posinf=4.0, neginf=4.0 * -1)

                B_inner    = x0_approx.shape[0]
                ego_dxdy   = x0_approx[:, :120].reshape(B_inner, 60, 2)   # grad
                agent_dxdy = x0_approx[:, 120:].reshape(B_inner, 60, 2)   # grad (NOT detached)

                # Absolute positions (both retain grad) ───────────────────────
                ego_pos = ego_last_xy.unsqueeze(1) + ego_dxdy.cumsum(dim=1)   # (B,60,2)
                agent_pos = None
                if agent_last_xy is not None:
                    agent_pos = agent_last_xy.unsqueeze(1) + agent_dxdy.cumsum(dim=1)

                # (1) Symmetric SVO（时变：固定激进 target + 逐时刻 social force 避让）─
                # target 是全程固定的激进偏移（baseline−delta）= 初始激进承诺；
                # 时变 SVO 里 social force 在近距离自动推高 SVO（避让），与 target 拉锯，
                # 净效果"远激进、近避让"由 ego-agent 距离决定，无需硬分前后段。
                if _use_soft_svo:
                    # 路线 A：软累加 SVO (B,T)。ego 初始量来自历史硬计数，agent 初始 0（用户 09-13 定）；
                    # 只依赖各自速度、不需要对方位置 → ego/agent 之间不互传 SVO 梯度（与旧口径 detach 对方位置一致）。
                    ego_svo_batch = compute_svo_soft_cumulative_torch(ego_dxdy, ego_history, _soft_cfg)
                    agent_svo_batch = (compute_svo_soft_cumulative_torch(agent_dxdy, None, _soft_cfg)
                                       if agent_history is not None else None)
                else:
                    _svo_fn = (compute_ego_svo_torch_timevarying if svo_timevarying
                               else compute_ego_svo_torch)
                    ego_svo_list, agent_svo_list = [], []
                    for i in range(B_inner):
                        ego_hist_i = (ego_history[i]
                                      if ego_history.dim() == 3 else ego_history)
                        a_pos_i = (agent_pos[i].detach() if agent_pos is not None else None)
                        ego_svo_list.append(_svo_fn(ego_dxdy[i], ego_hist_i, a_pos_i, svo_config))
                        if agent_history is not None:
                            agent_hist_i = (agent_history[i]
                                            if agent_history.dim() == 3 else agent_history)
                            agent_svo_list.append(
                                _svo_fn(agent_dxdy[i], agent_hist_i, ego_pos[i].detach(), svo_config))
                    ego_svo_batch = torch.stack(ego_svo_list)            # (B,) or (B,T)
                    agent_svo_batch = torch.stack(agent_svo_list) if agent_svo_list else None
                agent_svo_mean = float('nan')
                if svo_loss_mode == 'two_sided':
                    # 路线 A：只用末帧 α(T)，smooth_l1(beta=1°) 双向拉向目标；ego 与 agent 各一项（batch 均值）相加
                    _ego_T = ego_svo_batch[:, -1] if ego_svo_batch.dim() == 2 else ego_svo_batch
                    _ego_tgt_T = ego_target[:, -1] if ego_target.dim() == 2 else ego_target
                    svo_loss = F.smooth_l1_loss(_ego_T, _ego_tgt_T, reduction='mean', beta=1.0)
                    if agent_svo_batch is not None:
                        _agent_T = (agent_svo_batch[:, -1] if agent_svo_batch.dim() == 2
                                    else agent_svo_batch)
                        _agent_tgt_T = agent_target[:, -1] if agent_target.dim() == 2 else agent_target
                        svo_loss = svo_loss + F.smooth_l1_loss(
                            _agent_T, _agent_tgt_T, reduction='mean', beta=1.0)
                        agent_svo_mean = agent_svo_batch.mean().item()
                else:
                    # 旧单向 relu（target 为 (B,) 时对逐帧序列广播；软累加相对目标本身是 (B,T)，直接逐帧比）
                    _ego_tgt = (ego_target.unsqueeze(1)
                                if ego_svo_batch.dim() == 2 and ego_target.dim() == 1 else ego_target)
                    svo_loss = F.relu(ego_svo_batch - _ego_tgt).mean()
                    if agent_svo_batch is not None:
                        _agent_tgt = (agent_target.unsqueeze(1)
                                      if agent_svo_batch.dim() == 2 and agent_target.dim() == 1
                                      else agent_target)
                        svo_loss = svo_loss + F.relu(agent_svo_batch - _agent_tgt).mean()
                        agent_svo_mean = agent_svo_batch.mean().item()

                # (2) Proximity / closing-speed objective ─────────────────────
                proximity_loss = x0_approx.new_zeros(())
                soft_min_mean  = float('nan')
                closing_mean   = float('nan')
                future_danger_mean = float('nan')
                if agent_pos is not None and proximity_weight > 0.0:
                    dist = (ego_pos - agent_pos).norm(dim=-1).clamp(min=1e-4)  # (B,60)
                    # soft-min over time = -tau * logsumexp(-dist/tau)（诊断用，始终算）
                    soft_min = -proximity_tau * torch.logsumexp(
                        -dist / proximity_tau, dim=1)                          # (B,)
                    soft_min_mean  = soft_min.mean().item()

                    if proximity_objective == 'closing':
                        # 方式二精修①(b)：soft-min 距离拉近（主驱动，保留 softmin 的强汇聚力，
                        # 即 near_miss_target=0 拉向接触）+ closing 时间加权修正项（逼真实收敛运动）。
                        # closing[t] = -(r̂·v_rel)，r=agent−ego，v_rel=agent_vel−ego_vel（每步增量 dxdy）；
                        # >0=在接近、<0=在远离。修正 = 罚"早段不接近"：w_t=exp(−t_idx/horizon)（早段强、晚段松），
                        # closing<margin（停下挤团 closing≈0 / 发散 closing<0）才罚，逼早段朝对方持续运动；
                        # 用固定时间权重而非距离 gate——距离 gate 会被"逃远→罚归零"钻空子（致发散），时间权重无此漏洞；
                        # 晚段权重低=不惩罚过最近点后的分离，保 near-miss 形态。
                        r       = agent_pos - ego_pos                          # (B,60,2) ego→agent
                        r_hat   = r / dist.unsqueeze(-1).clamp(min=closing_eps)
                        v_rel   = agent_dxdy - ego_dxdy                        # (B,60,2) 每步增量=速度·dt
                        closing = -(r_hat * v_rel).sum(dim=-1)                # (B,60) >0 接近
                        t_idx   = torch.arange(closing.shape[1], device=closing.device,
                                               dtype=closing.dtype)
                        w_t     = torch.exp(-t_idx / closing_horizon)         # (60,) 早段强（固定，无逃逸）
                        closing_pen = (w_t * F.relu(closing_margin - closing)).mean()
                        pull = F.relu(soft_min - near_miss_target).mean()
                        proximity_loss = pull + closing_weight * closing_pen
                        closing_mean   = closing.mean().item()
                    elif proximity_objective == 'danger_score':
                        # 方式二精修③(c)：soft-min 距离拉近（远场 reach，把远对拉进 5m 带）
                        # + maximize eval 的可微 outcome_danger（train/eval 完全闭合）。
                        # outcome_danger = compute_single_trajectory_danger_torch(future)['total_score']
                        # = ttc_score + distance_score(峰 3m 帐篷) + velocity_score(closing gated dist<5m)。
                        # vel=dxdy/danger_dt → m/s（TTC 单位=秒，闭合 eval）；状态构造 (B,60,4)=[x,y,vx,vy]。
                        ego_vel   = ego_dxdy  / danger_dt                     # (B,60,2)
                        agent_vel = agent_dxdy / danger_dt
                        ego_traj   = torch.cat([ego_pos,   ego_vel],   dim=-1)  # (B,60,4)
                        agent_traj = torch.cat([agent_pos, agent_vel], dim=-1)
                        fut = compute_single_trajectory_danger_torch(
                            ego_traj, agent_traj, _danger_cfg, traj_type="future")
                        future_danger = fut['total_score']                   # (B,) = eval outcome_danger
                        pull = F.relu(soft_min - near_miss_target).mean()    # 远场 reach（near_miss_target=3）
                        proximity_loss = pull + danger_score_weight * (-future_danger).mean()
                        future_danger_mean = future_danger.mean().item()
                    else:
                        # 旧 soft-min 距离目标（默认，向后兼容 S2/eval）
                        proximity_loss = F.relu(soft_min - near_miss_target).mean()

                # (3) L_lane (ego on-road；可选同时约束 agent) ─────────────────
                lane_loss = x0_approx.new_zeros(())
                if lane_polylines is not None and lane_weight > 0.0:
                    _lane_t = lane_polylines.to(device=device, dtype=x0_approx.dtype)
                    lane_loss = lane_deviation_score_torch(
                        ego_pos, _lane_t, lane_half_width).mean()
                    # benchmark 用：同时把 agent 拉回路面（默认只约束 ego；agent_pos 带 grad）
                    if lane_constrain_agent and agent_pos is not None:
                        lane_loss = lane_loss + lane_deviation_score_torch(
                            agent_pos, _lane_t, lane_half_width).mean()

                # Kinematic realism（默认只 ego；kin_constrain_agent 时同时约束 agent，使两者等效真实）
                kin_loss = self._kinematic_realism_loss_v6_0(x0_approx[:, :120])
                if kin_constrain_agent and agent_pos is not None:
                    kin_loss = kin_loss + self._kinematic_realism_loss_v6_0(x0_approx[:, 120:])

                g_svo = None
                if svo_step > 0.0:
                    # 路线 A：SVO 梯度与其余项分开求——其余项梯度 g 走下面原有路径（逻辑一字不改），
                    # SVO 梯度 g_svo 在后面单独 lowpass + 按样本单位化，不再被 danger 梯度的整体截断淹没。
                    rest_loss = (proximity_weight * proximity_loss
                                 + lane_weight      * lane_loss
                                 + lambda_kinematic * kin_loss)
                    g = self._grad_or_zeros_phase2(rest_loss, x_leaf, retain_graph=True)
                    g_svo = self._grad_or_zeros_phase2(svo_weight * svo_loss, x_leaf, retain_graph=False)
                else:
                    total_loss = (svo_weight       * svo_loss
                                  + proximity_weight * proximity_loss
                                  + lane_weight      * lane_loss
                                  + lambda_kinematic * kin_loss)

                    g = torch.autograd.grad(total_loss, x_leaf)[0].detach()

                if use_lowpass:
                    g = self._lowpass_guidance_v6_0(g, cutoff_ratio=lowpass_cutoff_ratio)
                if use_bezier:
                    ego_g_smooth = bezier_guidance_v6_0(g[:, :120], n_ctrl=bezier_n_ctrl)
                    g = torch.cat([ego_g_smooth, g[:, 120:]], dim=-1)

                g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # 原始梯度幅度（debug/诊断用，始终算）
                eff_scale = (svo_guidance_scale * (t + 1) / k
                             if time_varying_scale else svo_guidance_scale)
                if disable_safety_clamp:
                    # 方式二：去安全夹子——原始（lowpass 后）梯度直接按 eff_scale 驱动位移，
                    # 不单位化、不 clip 到固定小步长，让碰撞梯度真正推得动两车汇聚。
                    # 真实性改由 kinematic_projection 硬投影守门（见下）；magnitude_cap 仅作爆炸保险。
                    g_use = g
                    if magnitude_cap > 0.0:
                        g_use = g_use * (g_norm.clamp(max=magnitude_cap) / g_norm)
                    guidance = eff_scale * g_use
                else:
                    # 旧行为（决策树 B 口径，完全向后兼容）：单位化方向 + clip 到 grad_clip_norm。
                    # 软归一化（抗 chattering 主因 A）：g_ref>0 时，|g|>>g_ref≈单位方向（稳定），
                    # |g|<<g_ref（约束将满足）时推力→0，消除单位向量在最优点附近的等幅过冲振荡；
                    # g_ref=0 退化为旧的单位向量行为（完全向后兼容，不影响既有 eval 复现）。
                    if guidance_g_ref > 0.0:
                        g_dir = g / (g_norm + guidance_g_ref)
                    else:
                        g_dir = g / g_norm
                    guidance = eff_scale * g_dir
                    gout_norm = guidance.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    guidance  = guidance * (gout_norm.clamp(max=grad_clip_norm) / gout_norm)

                g_svo_norm_mean = float('nan')
                if g_svo is not None:
                    # 路线 A：SVO 推力 = 同样 lowpass（若开）→ 按样本单位化（范数<1e-8 的样本置 0）→ ×svo_step×eff_scale，
                    # 叠加在「其余项已按原路径截断/缩放」之后的 guidance 上
                    if use_lowpass:
                        g_svo = self._lowpass_guidance_v6_0(g_svo, cutoff_ratio=lowpass_cutoff_ratio)
                    _svo_gn = g_svo.norm(dim=-1, keepdim=True)
                    g_svo_dir = torch.where(_svo_gn < 1e-8, torch.zeros_like(g_svo),
                                            g_svo / _svo_gn.clamp(min=1e-8))
                    guidance = guidance + (eff_scale * svo_step) * g_svo_dir
                    if debug:
                        g_svo_norm_mean = _svo_gn.mean().item()

                if debug:
                    # 路线 A 诊断：软累加末帧 α(T) 均值（仅 soft_cumulative 时有值，其余为 nan）
                    _soft_ego_T = (ego_svo_batch[:, -1].mean().item() if _use_soft_svo else float('nan'))
                    _soft_agent_T = (agent_svo_batch[:, -1].mean().item()
                                     if (_use_soft_svo and agent_svo_batch is not None) else float('nan'))
                    _diag_steps.append({
                        "t": t,
                        "ego_svo": ego_svo_batch.mean().item(),
                        "agent_svo": agent_svo_mean,
                        "soft_min": soft_min_mean,
                        "closing": closing_mean,
                        "fut_danger": future_danger_mean,
                        "L_svo": svo_loss.item(),
                        "L_prox": float(proximity_loss),
                        "L_lane": float(lane_loss),
                        "L_kin": kin_loss.item(),
                        "eff_scale": eff_scale,
                        # 抗震机制证据：g_norm=原始梯度幅度（约束将满足时→小），
                        # guid_norm=clip 后实际施加的推力。g_ref>0 时 guid_norm 应随 g_norm 下降
                        # 更快衰减（软归一化生效）；g_ref=0 时 guid_norm≈eff_scale（单位向量不随 loss 衰减）。
                        "g_norm": g_norm.mean().item(),
                        "guid_norm": guidance.norm(dim=-1).mean().item(),
                        # 路线 A：软累加末帧 α(T)（ego/agent 均值）与 SVO 原始梯度范数（svo_step>0 时，lowpass 后、单位化前）
                        "svo_soft_ego_T": _soft_ego_T,
                        "svo_soft_agent_T": _soft_agent_T,
                        "g_svo_norm": g_svo_norm_mean,
                    })

                x_corrected = x.detach() - guidance
            else:
                x_corrected = x

            # DDPM posterior step ─────────────────────────────────────────────
            x_0_pred = (
                x_corrected - (1 - alpha_bar_t).sqrt() * noise_pred
            ) / alpha_bar_t.sqrt()
            x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

            # 方式二：对干净轨迹估计逐步做运动学硬投影——posterior 由可行的 x0_pred 构建，
            # 使整条去噪过程都待在物理可行域（speed≤v_max/accel≤a_max），替代安全夹子守真实性。
            if kinematic_projection:
                ego_proj = self._kinematic_project(x_0_pred[:, :120])
                agt_proj = self._kinematic_project(x_0_pred[:, 120:])
                x_0_pred = torch.cat([ego_proj, agt_proj], dim=-1)

            mean = (self.posterior_mean_coef1[t] * x_0_pred
                    + self.posterior_mean_coef2[t] * x_corrected)
            if t > 0:
                noise = torch.randn_like(x_corrected)
                sigma = self.posterior_variance[t].sqrt()
                x     = mean + sigma * noise
            else:
                x = mean

        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)

        # 起点速度锚定（修起点掉头）：去噪器为收敛后段会让 agent 从起点就朝 ego 动、
        # 可能反向。这里在生成后把**前 start_anchor_steps 步增量**向「历史末速度×dt」混合，
        # ramp 0→1（step0 完全=历史速度，到第 N 步完全=生成），保 C1 连续、必消反转。
        # 默认 0=关，不影响 eval 复现；benchmark 用 build_benchmark 显式开启。
        if start_anchor_steps > 0:
            B_ = x.shape[0]
            n = min(start_anchor_steps, 60)
            w = torch.linspace(0.0, 1.0, n + 1, device=x.device, dtype=x.dtype)[1:]  # (n,) step0→1/n
            ego_v = (ego_history[..., -1, 3:5] if ego_history.dim() == 3
                     else ego_history[-1, 3:5].unsqueeze(0).expand(B_, 2)).to(x.device, x.dtype)
            ego_t = ego_v * 0.1                                    # 历史末速度×dt(0.1s,10Hz) → 增量目标 (B,2)
            x = x.clone()
            ego = x[:, :120].reshape(B_, 60, 2).clone()   # .clone() 防 B=1 时 view 自赋值内存重叠
            ego[:, :n] = w.view(1, n, 1) * ego[:, :n] + (1 - w).view(1, n, 1) * ego_t.unsqueeze(1)
            x[:, :120] = ego.reshape(B_, 120)
            if agent_history is not None:
                ag_v = (agent_history[..., -1, 3:5] if agent_history.dim() == 3
                        else agent_history[-1, 3:5].unsqueeze(0).expand(B_, 2)).to(x.device, x.dtype)
                ag_t = ag_v * 0.1
                agt = x[:, 120:].reshape(B_, 60, 2).clone()   # 同上，防 B=1 内存重叠
                agt[:, :n] = w.view(1, n, 1) * agt[:, :n] + (1 - w).view(1, n, 1) * ag_t.unsqueeze(1)
                x[:, 120:] = agt.reshape(B_, 120)

        if debug and _diag_steps:
            print(f"\n[Phase2 diag] full guidance loop (k={k}, every 5 steps)  "
                  f"[g_ref={guidance_g_ref}, stop_t={guidance_stop_t}, "
                  f"bezier={use_bezier}, tvs={time_varying_scale}]")
            print(f"{'t':>4}  {'ego_svo':>8}  {'agt_svo':>8}  {'soft_min':>8}  {'closing':>8}  {'futDang':>8}  "
                  f"{'L_svo':>7}  {'L_prox':>7}  {'L_lane':>7}  {'L_kin':>7}  "
                  f"{'g_norm':>8}  {'guid_norm':>9}")
            for s in _diag_steps[::5]:
                print(f"{s['t']:>4}  {s['ego_svo']:>7.2f}°  {s['agent_svo']:>7.2f}°  "
                      f"{s['soft_min']:>8.2f}  {s['closing']:>8.3f}  {s['fut_danger']:>8.3f}  "
                      f"{s['L_svo']:>7.4f}  {s['L_prox']:>7.4f}  "
                      f"{s['L_lane']:>7.4f}  {s['L_kin']:>7.4f}  "
                      f"{s['g_norm']:>8.4f}  {s['guid_norm']:>9.4f}")
            s0, sN = _diag_steps[-1], _diag_steps[0]
            print(f"[Phase2 diag] ego SVO drift: {sN['ego_svo']:.2f}° → {s0['ego_svo']:.2f}°  "
                  f"(ego_target≈{ego_target.mean().item():.2f}°)")
            print(f"[Phase2 diag] agent SVO drift: {sN['agent_svo']:.2f}° → {s0['agent_svo']:.2f}°  "
                  f"(agent_target≈{agent_target.mean().item():.2f}°)")
            if _use_soft_svo or svo_step > 0.0:
                _gsn = [d['g_svo_norm'] for d in _diag_steps if d['g_svo_norm'] == d['g_svo_norm']]
                _gsn_mean = sum(_gsn) / len(_gsn) if _gsn else float('nan')
                print(f"[Phase2 diag] 路线A measure={svo_measure} loss={svo_loss_mode} svo_step={svo_step}  "
                      f"soft α(T) ego: {sN['svo_soft_ego_T']:.2f}° → {s0['svo_soft_ego_T']:.2f}°  "
                      f"agent: {sN['svo_soft_agent_T']:.2f}° → {s0['svo_soft_agent_T']:.2f}°  "
                      f"g_svo_norm(步均)={_gsn_mean:.4f}")

        return x

    def generate_with_danger_guidance(
        self,
        condition: torch.Tensor,
        danger_target: torch.Tensor,
        danger_fn: callable,
        num_samples: int = 4,
        select_best: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        带Danger引导的生成

        生成多个候选轨迹，选择最符合danger_target的。

        Args:
            condition: [batch, 256] 条件向量
            danger_target: [batch] 目标danger值
            danger_fn: 计算danger的函数
            num_samples: 每个条件生成的候选数
            select_best: 是否选择最佳候选

        Returns:
            (best_trajectory, info_dict)
        """
        batch_size = condition.size(0)
        device = condition.device

        # 生成多个候选
        candidates = []
        danger_scores = []

        for _ in range(num_samples):
            traj = self.generate(condition)  # [batch, 120]
            candidates.append(traj)

        # Stack所有候选：[batch, num_samples, 120]
        candidates = torch.stack(candidates, dim=1)

        # 计算每个候选的danger（需要danger_fn）
        if danger_fn is not None and select_best:
            # 这里简化处理，实际需要恢复轨迹并计算danger
            # 选择最接近danger_target的候选
            # 暂时随机选择
            best_idx = torch.randint(0, num_samples, (batch_size,))
            best_traj = candidates[:, best_idx]  # [batch, 120]
        else:
            best_traj = candidates[:, 0]

        info = {
            'candidates': candidates,
            'num_samples': num_samples,
            'best_idx': best_idx if select_best else None
        }

        return best_traj, info

    def forward(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        训练时的前向传播

        Args:
            x_0: [batch, 120] GT轨迹增量
            condition: [batch, 256] 条件向量

        Returns:
            (loss, info_dict)
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # 随机采样时间步
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)

        # 添加噪声
        x_t, noise = self.add_noise(x_0, t)

        # 预测噪声
        noise_pred = self.denoiser(x_t, condition, t)

        # 计算loss
        loss = F.mse_loss(noise_pred, noise)

        info = {
            'loss': loss.item(),
            't_mean': t.float().mean().item()
        }

        return loss, info


# ============================================================
# 简化版Diffusion（用于快速实验）
# ============================================================

class SimpleDiffusionGenerator(nn.Module):
    """
    简化版Diffusion生成器

    更少的去噪步数，更简单的网络，用于快速实验。
    """

    def __init__(
        self,
        trajectory_dim: int = 120,
        condition_dim: int = 256,
        num_diffusion_steps: int = 50
    ):
        super().__init__()

        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.num_diffusion_steps = num_diffusion_steps

        # 简单的去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(trajectory_dim + condition_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, trajectory_dim)
        )

        # 时间步编码（简化）
        self.time_embed = nn.Embedding(num_diffusion_steps, condition_dim)

        # 注册Diffusion参数
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        print(f"[SimpleDiffusionGenerator] 初始化完成")
        print(f"  - Trajectory dim: {trajectory_dim}")
        print(f"  - Num steps: {num_diffusion_steps}")

    def forward(self, x_0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """训练前向传播"""
        batch_size = x_0.size(0)
        device = x_0.device

        # 随机时间步
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        # 添加噪声
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x_0)
        x_t = alpha_bar.sqrt().unsqueeze(-1) * x_0 + \
              (1 - alpha_bar).sqrt().unsqueeze(-1) * noise

        # 时间嵌入
        time_emb = self.time_embed(t)  # [batch, condition_dim]

        # 组合条件
        combined = condition + time_emb

        # 预测噪声（简化：直接预测x_0）
        input_feat = torch.cat([x_t, combined], dim=-1)
        x_0_pred = self.denoiser(input_feat)

        # Loss
        loss = F.mse_loss(x_0_pred, x_0)

        return loss

    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """生成轨迹"""
        batch_size = condition.size(0)
        device = condition.device

        # 初始化噪声
        x = torch.randn(batch_size, self.trajectory_dim, device=device)

        # 逐步去噪
        for t in reversed(range(self.num_diffusion_steps)):
            # 时间嵌入
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            time_emb = self.time_embed(t_tensor)

            # 组合条件
            combined = condition + time_emb

            # 预测x_0
            input_feat = torch.cat([x, combined], dim=-1)
            x_0_pred = self.denoiser(input_feat)

            # DDPM采样
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.betas[t].sqrt()
            else:
                noise = 0
                sigma = 0

            # 简化采样：x_{t-1} = alpha.sqrt() * x_0_pred + sigma * noise
            x = alpha.sqrt() * x_0_pred + sigma * noise

        return x


# ============================================================
# 测试函数
# ============================================================

def test_diffusion():
    """测试Diffusion模型"""
    print("Testing DangerSceneDiffusion...")

    batch_size = 4
    device = 'cpu'

    # 创建配置
    config = DiffusionConfig(
        num_diffusion_steps=100,
        condition_dim=256,
        trajectory_dim=120
    )

    # 创建模型
    model = DangerSceneDiffusion(config)

    # 测试训练前向传播
    print("\nTesting training forward pass...")
    x_0 = torch.randn(batch_size, 120, device=device)
    condition = torch.randn(batch_size, 256, device=device)

    loss, info = model(x_0, condition)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Mean t: {info['t_mean']:.1f}")

    # 测试生成
    print("\nTesting generation...")
    generated = model.generate(condition)
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated mean: {generated.mean().item():.4f}")
    print(f"  Generated std: {generated.std().item():.4f}")

    # 测试带中间结果的生成
    print("\nTesting generation with intermediate...")
    generated, intermediate = model.generate(condition, return_intermediate=True)
    print(f"  Intermediate shape: {intermediate.shape}")

    print("\nTesting SimpleDiffusionGenerator...")
    simple_model = SimpleDiffusionGenerator()
    loss = simple_model(x_0, condition)
    print(f"  Simple loss: {loss.item():.4f}")

    generated_simple = simple_model.generate(condition)
    print(f"  Simple generated shape: {generated_simple.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_diffusion()