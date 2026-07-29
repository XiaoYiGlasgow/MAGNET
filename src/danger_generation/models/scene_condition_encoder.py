"""
场景条件编码器 - 用于Diffusion危险场景生成

编码生成条件：
1. 历史场景特征（来自GNN+Trans encoder的384维输出）
2. SVO角度条件（控制行为倾向）
3. Danger目标（目标危险等级）

输出：256维条件向量，用于引导Diffusion生成
"""

import os
import sys

# Avoid importing the local wheel-cache directory `nuplan-devkit/torch` as PyTorch.
sys.path = [
    p for p in sys.path
    if os.path.basename(os.path.normpath(p)) != "nuplan-devkit"
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ConditionEncoderConfig:
    """条件编码器配置"""
    scene_feat_dim: int = 384   # GNN+Trans encoder输出维度
    condition_dim: int = 256    # 条件向量维度
    svo_embed_dim: int = 64     # SVO嵌入维度
    danger_embed_dim: int = 64  # Danger嵌入维度
    agent_state_embed_dim: int = 32  # Key agent kinematic state embed dim
    # B1（v9 阶段2）：condition 是否加入 key agent 的 xy 位置（ego-centric）。
    # False = 阶段1 行为（只用 heading,vx,vy → 4 维特征）；True = 加 xy（→ 6 维特征）。
    use_agent_xy: bool = False
    # B1 量纲修正：agent xy(ego-centric)原始量纲达几十米(|x|中位~26/max~74)，远大于
    # heading[-1,1]/vx个位数；forward 里对 xy 除以此 scale 归一化到同量级，避免主导/干扰 condition。
    # 注意：train 与 eval 必须用相同 scale（默认值即一致；改了要两边都改）。
    agent_xy_scale: float = 30.0
    svo_center_deg: float = 45.0
    svo_half_range_deg: float = 90.0
    danger_scale: float = 10.0      # tent 函数最大约 6~8，用 10 让 [0,1] 范围被充分利用
    danger_gap_scale: float = 2.0   # pred_danger + 0.5 的 gap 约 0.5~2，scale=2 归一化合理
    # B1（v10 CFG）：属性 classifier-free guidance 开关。
    # Phase 0 体检证 v9 的 danger 条件被忽略（conditioning collapse）；要让 danger 成为可控旋钮，
    # 训练时须随机把 danger 分量替换为可学习 null token（属性 dropout），强制模型学到 cond/uncond 真差异，
    # 推理时 CFG 放大。True 才创建 null_danger_embed / null_danger_gap_embed 参数。
    # 【向后兼容】默认 False → 不新增参数，老 v9 checkpoint 仍可 strict load。v10 训练须置 True。
    danger_dropout_enable: bool = False


class SceneConditionEncoder(nn.Module):
    """
    场景条件编码器

    将场景特征、SVO角度和Danger目标融合为统一的条件向量，
    用于引导Diffusion模型生成更危险的轨迹。

    输入：
    - scene_feat: [batch, 384] 来自GNN+Trans encoder的场景特征
    - svo_angle: [batch, 1] SVO角度（度），范围[-45°, 135°]
      - 竞争型(-45°) → 生成更危险轨迹
      - 合作型(45°) → 生成更安全轨迹
    - inherent_danger: [batch, 1] 历史轨迹的危险指数（场景固有风险）
    - danger_target: [batch, 1] 目标危险等级（应 > inherent_danger）

    输出：
    - condition: [batch, 256] 条件向量
    """

    def __init__(self, config: ConditionEncoderConfig = None):
        super().__init__()

        if config is None:
            config = ConditionEncoderConfig()

        self.config = config

        # 历史场景编码
        self.scene_encoder = nn.Sequential(
            nn.Linear(config.scene_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, config.condition_dim)
        )

        # SVO角度编码
        # 输入：角度值（度），归一化到[-1, 1]
        self.svo_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, config.svo_embed_dim),
            nn.LayerNorm(config.svo_embed_dim)
        )

        # Danger目标编码
        # 输入：[inherent_danger, danger_target] 两个值
        self.danger_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, config.danger_embed_dim),
            nn.LayerNorm(config.danger_embed_dim)
        )

        # Danger gap编码（可选，用于强调超越幅度）
        self.danger_gap_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Key agent kinematic state encoder
        # use_agent_xy=False: [sin(heading), cos(heading), vx, vy] = 4-dim（阶段1）
        # use_agent_xy=True : [x, y, sin(heading), cos(heading), vx, vy] = 6-dim（B1）
        self.use_agent_xy = config.use_agent_xy
        self.agent_xy_scale = config.agent_xy_scale
        self._agent_state_in_dim = 6 if config.use_agent_xy else 4
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(self._agent_state_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.agent_state_embed_dim),
            nn.LayerNorm(config.agent_state_embed_dim)
        )

        # 条件融合层: scene(256) + svo(64) + danger(64) + agent_state(32) = 416
        fusion_input_dim = (config.condition_dim + config.svo_embed_dim
                            + config.danger_embed_dim + config.agent_state_embed_dim)
        self.condition_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, config.condition_dim)
        )

        # 输出层（可选的增强）
        self.output_enhance = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.condition_dim)
        )

        # B1（v10 CFG）：可学习 null-danger token，属性 dropout 时替换 danger 分量。
        # 只在 danger_dropout_enable 时创建，保证老 checkpoint 向后兼容（不引入新 key）。
        self.danger_dropout_enable = config.danger_dropout_enable
        if config.danger_dropout_enable:
            # 替换 danger_embed（64 维）与 danger_gap_embed（32 维）；[1, dim] 便于 broadcast。
            self.null_danger_embed = nn.Parameter(torch.zeros(1, config.danger_embed_dim))
            self.null_danger_gap_embed = nn.Parameter(torch.zeros(1, 32))
            nn.init.normal_(self.null_danger_embed, std=0.02)
            nn.init.normal_(self.null_danger_gap_embed, std=0.02)

        print(f"[SceneConditionEncoder] 初始化完成")
        print(f"  - Scene feat dim: {config.scene_feat_dim}")
        print(f"  - Condition dim: {config.condition_dim}")
        print(f"  - SVO embed dim: {config.svo_embed_dim}")
        print(f"  - Danger embed dim: {config.danger_embed_dim}")
        print(f"  - Agent state embed dim: {config.agent_state_embed_dim}")

    def forward(
        self,
        scene_feat: torch.Tensor,
        svo_angle: torch.Tensor,
        inherent_danger: torch.Tensor,
        danger_target: torch.Tensor,
        agent_state: Optional[torch.Tensor] = None,
        danger_dropout: float = 0.0,
        force_null_danger: bool = False,
    ) -> torch.Tensor:
        """
        编码生成条件

        Args:
            scene_feat: [batch, 384] 场景特征
            svo_angle: [batch, 1] SVO角度（度）
            inherent_danger: [batch, 1] 历史危险指数
            danger_target: [batch, 1] 目标危险指数
            agent_state: key agent last-step state in ego-centric frame.
                         use_agent_xy=False → [batch, 3] (heading, vx, vy)；
                         use_agent_xy=True  → [batch, 5] (x, y, heading, vx, vy)。
                         If None, zeros are used (backward-compatible with old checkpoints).
            danger_dropout: 训练时按此概率（逐样本）把 danger 分量替换为可学习 null token
                         （属性 CFG dropout）。仅 danger_dropout_enable=True 生效。
            force_null_danger: True 时强制全 batch 用 null danger（推理 CFG 的 uncond 路）。

        Returns:
            [batch, 256] 条件向量
        """
        batch_size = scene_feat.size(0)

        # B1（v10 CFG）：决定哪些样本用 null danger（属性 dropout）。
        #   force_null_danger → 全 null（推理 uncond 路）；
        #   training & danger_dropout>0 → 逐样本 Bernoulli(danger_dropout)。
        drop_mask = None
        if self.danger_dropout_enable:
            if force_null_danger:
                drop_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=scene_feat.device)
            elif self.training and danger_dropout > 0.0:
                drop_mask = (torch.rand(batch_size, 1, device=scene_feat.device) < danger_dropout)

        # 1. 场景特征编码
        scene_embed = self.scene_encoder(scene_feat)  # [batch, 256]

        # 2. SVO角度编码
        svo_normalized = self._normalize_svo(svo_angle)
        svo_embed = self.svo_encoder(svo_normalized)  # [batch, 64]

        # 3. Danger编码
        danger_input = self._build_danger_input(inherent_danger, danger_target)  # [batch, 2]
        danger_embed = self.danger_encoder(danger_input)  # [batch, 64]
        if drop_mask is not None:
            danger_embed = torch.where(
                drop_mask, self.null_danger_embed.expand_as(danger_embed), danger_embed)

        # 4. Key agent kinematic state encoding
        # Encode heading as (sin, cos) to avoid discontinuity at ±π
        if agent_state is not None:
            if self.use_agent_xy:
                # agent_state: [batch, 5] = (x, y, heading, vx, vy)
                # B1 量纲修正：xy 归一化到与其他特征同量级（见 agent_xy_scale）
                xy      = agent_state[:, 0:2] / self.agent_xy_scale  # [batch, 2]
                heading = agent_state[:, 2:3]          # [batch, 1]
                vx      = agent_state[:, 3:4]          # [batch, 1]
                vy      = agent_state[:, 4:5]          # [batch, 1]
                agent_state_input = torch.cat(
                    [xy, torch.sin(heading), torch.cos(heading), vx, vy], dim=-1
                )  # [batch, 6]
            else:
                # agent_state: [batch, 3] = (heading, vx, vy)
                heading = agent_state[:, 0:1]          # [batch, 1]
                vx      = agent_state[:, 1:2]          # [batch, 1]
                vy      = agent_state[:, 2:3]          # [batch, 1]
                agent_state_input = torch.cat(
                    [torch.sin(heading), torch.cos(heading), vx, vy], dim=-1
                )  # [batch, 4]
        else:
            agent_state_input = torch.zeros(
                batch_size, self._agent_state_in_dim, device=scene_feat.device)
        agent_state_embed = self.agent_state_encoder(agent_state_input)  # [batch, 32]

        # 5. Danger gap编码（超越幅度）
        danger_gap = danger_target - inherent_danger  # [batch, 1]
        danger_gap_normalized = torch.clamp(danger_gap / self.config.danger_gap_scale, 0, 1)
        danger_gap_embed = self.danger_gap_encoder(danger_gap_normalized)  # [batch, 32]
        if drop_mask is not None:
            danger_gap_embed = torch.where(
                drop_mask, self.null_danger_gap_embed.expand_as(danger_gap_embed), danger_gap_embed)

        # 6. 融合所有条件: scene(256) + svo(64) + danger(64) + agent_state(32) = 416
        combined = torch.cat([scene_embed, svo_embed, danger_embed, agent_state_embed], dim=-1)
        condition = self.condition_fusion(combined)  # [batch, 256]

        # 7. 增强输出：注入danger gap信息
        condition = condition + 0.3 * F.pad(danger_gap_embed, (0, 256 - 32))
        condition = self.output_enhance(condition)

        return condition

    def encode_svo_only(self, svo_angle: torch.Tensor) -> torch.Tensor:
        """
        仅编码SVO角度（用于推理时调整）

        Args:
            svo_angle: [batch, 1] SVO角度

        Returns:
            [batch, 64] SVO嵌入
        """
        svo_normalized = self._normalize_svo(svo_angle)
        return self.svo_encoder(svo_normalized)

    def encode_danger_only(self, inherent_danger: torch.Tensor, danger_target: torch.Tensor) -> torch.Tensor:
        """
        仅编码Danger条件

        Args:
            inherent_danger: [batch, 1]
            danger_target: [batch, 1]

        Returns:
            [batch, 64] Danger嵌入
        """
        danger_input = self._build_danger_input(inherent_danger, danger_target)
        return self.danger_encoder(danger_input)

    def _normalize_svo(self, svo_angle: torch.Tensor) -> torch.Tensor:
        """Normalize SVO angle from degrees into [-1, 1]."""
        normalized = (svo_angle - self.config.svo_center_deg) / self.config.svo_half_range_deg
        return torch.clamp(normalized, -1, 1)

    def _build_danger_input(
        self,
        inherent_danger: torch.Tensor,
        danger_target: torch.Tensor
    ) -> torch.Tensor:
        """Build the 2D danger input expected by the danger encoder."""
        inherent_normalized = torch.clamp(inherent_danger / self.config.danger_scale, 0, 1)
        target_normalized = torch.clamp(danger_target / self.config.danger_scale, 0, 1)
        return torch.cat([inherent_normalized, target_normalized], dim=-1)


class SimpleSceneEncoder(nn.Module):
    """
    简化版场景编码器

    当GNN+Trans encoder特征不可用时，直接从历史轨迹提取特征。
    用于快速实验或独立训练Diffusion模型。
    """

    def __init__(
        self,
        ego_history_dim: int = 5,
        agent_history_dim: int = 5,
        output_dim: int = 384,
        history_length: int = 20
    ):
        super().__init__()

        self.ego_history_dim = ego_history_dim
        self.agent_history_dim = agent_history_dim
        self.output_dim = output_dim
        self.history_length = history_length

        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_history_dim * history_length, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 192)
        )

        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_history_dim * history_length, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 192)
        )

        self.relative_encoder = nn.Sequential(
            nn.Linear(history_length, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # 总维度：ego(192) + agent(192) + relative(32) = 416 -> 384
        self.fusion = nn.Linear(416, output_dim)

        print(f"[SimpleSceneEncoder] 初始化完成")
        print(f"  - Output dim: {output_dim}")

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor
    ) -> torch.Tensor:
        """
        从历史轨迹提取场景特征

        Args:
            ego_history: [batch, T_hist, 5]
            agent_history: [batch, T_hist, 5]

        Returns:
            [batch, 384] 场景特征
        """
        batch_size = ego_history.size(0)
        T = ego_history.size(1)

        # 处理不同长度的历史轨迹
        # 如果长度大于history_length，取最后history_length帧
        # 如果长度小于history_length，padding到history_length
        if T > self.history_length:
            ego_history = ego_history[:, -self.history_length:]
            agent_history = agent_history[:, -self.history_length:]
            T = self.history_length
        elif T < self.history_length:
            # Padding到history_length
            ego_padding = torch.zeros(batch_size, self.history_length - T, ego_history.size(2),
                                      device=ego_history.device, dtype=ego_history.dtype)
            agent_padding = torch.zeros(batch_size, self.history_length - T, agent_history.size(2),
                                        device=agent_history.device, dtype=agent_history.dtype)
            ego_history = torch.cat([ego_padding, ego_history], dim=1)
            agent_history = torch.cat([agent_padding, agent_history], dim=1)
            T = self.history_length

        # Flatten历史轨迹
        ego_flat = ego_history.view(batch_size, -1)  # [batch, history_length*5]
        agent_flat = agent_history.view(batch_size, -1)  # [batch, history_length*5]

        # 编码
        ego_feat = self.ego_encoder(ego_flat)  # [batch, 192]
        agent_feat = self.agent_encoder(agent_flat)  # [batch, 192]

        # 相对距离特征
        distances = torch.norm(ego_history[:, :, :2] - agent_history[:, :, :2], dim=-1)  # [batch, history_length]
        relative_feat = self.relative_encoder(distances)  # [batch, 32]

        # 融合
        combined = torch.cat([ego_feat, agent_feat, relative_feat], dim=-1)  # [batch, 416]
        scene_feat = self.fusion(combined)  # [batch, 384]

        return scene_feat


class AgentPredictionHead(nn.Module):
    """Predicts key agent baseline trajectory from scene features + agent last state.

    ego trajectory is predicted by GNN+Trans output_head (pre-trained, reused directly).
    This head only predicts the key agent part, conditioned on its last observed state.

    Input:  scene_feat [batch, 384] + agent_last_state [batch, 5]  -> [batch, 389]
    Output: agent_traj [batch, 120]  (Δx/Δy increments for 60 steps)
    """

    def __init__(self, scene_feat_dim: int = 384, agent_state_dim: int = 5, traj_dim: int = 120):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(scene_feat_dim + agent_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, traj_dim),
        )

    def forward(self, scene_feat: torch.Tensor, agent_last_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_feat:       [batch, 384]
            agent_last_state: [batch, 5]  (x, y, heading, vx, vy)
        Returns:
            [batch, 120] agent baseline trajectory increments (Δx/Δy, 60 steps)
        """
        x = torch.cat([scene_feat, agent_last_state], dim=-1)  # [batch, 389]
        return self.head(x)


class MultiAgentConditionEncoder(nn.Module):
    """
    多Agent条件编码器

    当有多个关键agent时，编码所有agent的条件。
    用于生成涉及多个agent的危险场景。
    """

    def __init__(
        self,
        config: ConditionEncoderConfig = None,
        max_agents: int = 5
    ):
        super().__init__()

        if config is None:
            config = ConditionEncoderConfig()

        self.max_agents = max_agents

        # 单个agent的条件编码器
        self.single_encoder = SceneConditionEncoder(config)

        # 多agent融合层
        # 输入：max_agents * condition_dim
        self.multi_fusion = nn.Sequential(
            nn.Linear(max_agents * config.condition_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, config.condition_dim)
        )

        # Attention权重计算（选择最重要的agent）
        self.attention = nn.Sequential(
            nn.Linear(config.condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        print(f"[MultiAgentConditionEncoder] 初始化完成")
        print(f"  - Max agents: {max_agents}")

    def forward(
        self,
        scene_feat: torch.Tensor,
        svo_angles: torch.Tensor,
        inherent_dangers: torch.Tensor,
        danger_targets: torch.Tensor,
        agent_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        编码多agent条件

        Args:
            scene_feat: [batch, 384] 场景特征
            svo_angles: [batch, num_agents, 1] SVO角度
            inherent_dangers: [batch, num_agents, 1] 历史危险指数
            danger_targets: [batch, num_agents, 1] 目标危险指数
            agent_mask: [batch, num_agents] 有效agent掩码

        Returns:
            [batch, 256] 融合的条件向量
        """
        batch_size = scene_feat.size(0)
        num_agents = svo_angles.size(1)
        device = scene_feat.device

        # 编码每个agent的条件
        agent_conditions = []
        for i in range(num_agents):
            condition_i = self.single_encoder(
                scene_feat,
                svo_angles[:, i],
                inherent_dangers[:, i],
                danger_targets[:, i]
            )
            agent_conditions.append(condition_i)

        # Stack: [batch, num_agents, 256]
        agent_conditions = torch.stack(agent_conditions, dim=1)

        # 计算attention权重
        attention_scores = self.attention(agent_conditions)  # [batch, num_agents, 1]

        # 应用掩码（如果有）
        if agent_mask is not None:
            attention_scores = attention_scores.masked_fill(~agent_mask.unsqueeze(-1), -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, num_agents, 1]

        # 加权融合
        weighted_condition = (agent_conditions * attention_weights).sum(dim=1)  # [batch, 256]

        # 如果agent数小于max_agents，用零填充
        if num_agents < self.max_agents:
            padded_conditions = torch.zeros(batch_size, self.max_agents, 256, device=device)
            padded_conditions[:, :num_agents] = agent_conditions
            multi_condition = self.multi_fusion(padded_conditions.view(batch_size, -1))
            # 与加权条件融合
            final_condition = 0.7 * weighted_condition + 0.3 * multi_condition
        else:
            final_condition = weighted_condition

        return final_condition


# ============================================================
# 测试函数
# ============================================================

def test_condition_encoder():
    """测试条件编码器"""
    print("Testing SceneConditionEncoder...")

    batch_size = 4
    device = 'cpu'

    # 创建测试数据
    scene_feat = torch.randn(batch_size, 384, device=device)
    svo_angle = torch.tensor([[-30.0], [0.0], [45.0], [90.0]], device=device)
    inherent_danger = torch.tensor([[5.0], [10.0], [3.0], [8.0]], device=device)
    danger_target = torch.tensor([[10.0], [15.0], [8.0], [12.0]], device=device)

    # 创建编码器
    encoder = SceneConditionEncoder()

    # 编码
    condition = encoder(scene_feat, svo_angle, inherent_danger, danger_target)

    print(f"Input shapes:")
    print(f"  scene_feat: {scene_feat.shape}")
    print(f"  svo_angle: {svo_angle.shape}")
    print(f"  inherent_danger: {inherent_danger.shape}")
    print(f"  danger_target: {danger_target.shape}")
    print(f"Output condition shape: {condition.shape}")
    print(f"Condition mean: {condition.mean().item():.4f}")
    print(f"Condition std: {condition.std().item():.4f}")

    # 测试简单场景编码器
    print("\nTesting SimpleSceneEncoder...")

    ego_history = torch.randn(batch_size, 20, 5, device=device)
    agent_history = torch.randn(batch_size, 20, 5, device=device)

    simple_encoder = SimpleSceneEncoder()
    scene_feat_simple = simple_encoder(ego_history, agent_history)

    print(f"Simple encoder output shape: {scene_feat_simple.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_condition_encoder()
