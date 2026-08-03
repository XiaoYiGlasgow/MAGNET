"""闸④（Q50）：**切断 ego 捷径** —— ego 分支不再读历史轨迹，改读当前状态 + state dropout。
**默认关闭、flag 门控**。

诊断（2026-08-01 §0.0-b 四道闸之第④道）
--------------------------------------
`ego_is_making_progress` 实测 PlanTF 0.949 vs MAGNET calibrated 0.520 / raw 0.676；
停机是 MAGNET 最大的失分项。最省事的解释是 **ego-shortcut**：
ego 历史原样喂入且无任何抑制 ⇒ 「停住 → 预测继续停 → 自我强化」。
PlanTF `2309.10443` 用两个开关压住这条捷径：`use_ego_history=False` + `state_dropout=0.75`。

🔴 **写作口径（§7 硬约束，不得违反）**
--------------------------------------
本模块 **零 novelty** —— 它是 PlanTF `2309.10443` 已发表贡献的移植，
论文里只能写成 **baseline / 对照格**，**不得写成本文提出的方法**（Q50 默认动作已如此裁定）。
四道闸里只有闸① 是我方自己量化、文献未记录的。

与 PlanTF 的对应关系（照抄意图，不照抄代码）
--------------------------------------------
PlanTF 的 ego 分支（`external/planTF/src/models/planTF/modules/agent_encoder.py:86-89`）：
当 `use_ego_history=False` 时**整条历史编码器的输出被丢弃**，ego token 改由
`data["current_state"][:, :6]` 经 `StateAttentionEncoder` 得到。
其 `current_state = [x, y, heading, vel_x, acc_x, steering_angle, yaw_rate]`
（`nuplan_feature_builder.py:192-204`），而 `NuplanFeature.normalize` 把 **前三维置零**
（`nuplan_feature.py:81`）⇒ **前 3 个 token 恒为 0 且恒可见**，
真正携带信息的 `[vel_x, acc_x, steering_angle]` 三个 token **各自以 0.75 概率被 mask 掉**。
（前 3 个恒可见不是建模选择，而是**功能必需** —— 否则 `key_padding_mask` 可能全 True 使 attention 出 NaN。）

MAGNET 侧的差异与移植方式：

* MAGNET 的 ego 输入是 `EGO_HISTORY[20,2]` —— **车辆局部坐标系逐帧增量**，
  index 0 恒为 `[0,0]`，index 19 = 最近一帧增量（训练侧
  `scripts/preprocessing/process_nuplan_to_12d.py:120-130`、
  部署侧 `devkit_patches/.../gnn_trans_planner.py:427-443` **同一约定**，已逐行核对）。
* ⇒ 「当前状态」只能从最后两帧增量导出。`steering_angle` 无法从位置增量得到，
  用 `yaw_rate` 顶替 —— PlanTF 的 `steering_angle` 本就由 `yaw_rate` 推出
  （`calculate_additional_ego_states`），是同一个运动学原语。
* 前 3 维在 MAGNET 里**天然为 0**：局部坐标系原点就锚在当前 ego 位姿上，
  等价于 PlanTF 的 `current_state[:3] = 0`。

⇒ 本模块的 6 通道当前状态（`ego_current_state_from_history`）：

    ch0..2 = 0（ego-centric 位姿，恒可见，仅起「至少 3 个 token 不被 mask」的作用）
    ch3    = vel_x     纵向速度      [m/s]
    ch4    = acc_x     纵向加速度    [m/s²]
    ch5    = yaw_rate  横摆角速度    [rad/s]
    ch3..5 训练时**各自独立**以 `ego_state_dropout` 概率被 mask（eval 下一律不 mask）

实现约束（照 `magnet_ego_query_fusion.py` / `magnet_context_pooled.py` 的既定模式）
----------------------------------------------------------------------------------
* **不修改 devkit vendor 的 `transformer_predictor.py` 一行**；
* `use_ego_history=True`（默认）时与父类 `MagnetEgoQueryFusion` **逐字节一致**；
* 对齐不上就**抛错，不静默退化**；
* 可与闸②③（`nearby_only` + `--ego_query_fusion`）**叠加**，也可单独开。

两个必须知道的接口技巧
----------------------
1. 父类 `transformer_predictor.py:210-211` 是
   `ego_feat_seq = self.ego_transformer(ego_encoded)` 然后 `ego_feat = ego_feat_seq.mean(dim=1)`。
   本模块把 `ego_transformer` 换成包装器，**返回 `[B, 1, 128]`**（已是单 token），
   于是父类那句 `.mean(dim=1)` 退化为恒等 —— 与 `_PooledContextTransformer` 同一手法。
2. 🔴 父类 `MagnetEgoQueryFusion.__init__` 把 `_capture_ego` 挂在**内层** `ego_transformer` 上。
   门控打开后内层**根本不执行** ⇒ 那个 hook 不会触发，C2 的 ego query 会取不到值。
   故本模块把同一个 `_capture_ego` **再挂到包装器上**：
   * 门控关闭 ⇒ 内外两个 hook 先后写入**同一个张量**，逐字节一致；
   * 门控打开 ⇒ 只有外层触发，`_ego_feat` = 当前状态编码
     ⇒ **C2 的 ego query 也一并被切断了对历史的依赖**（这正是想要的：
     否则闸④ 只切了融合前的 ego 分支，捷径会从 C2 的 query 侧漏回来）。

⚠️ **已知的无害冗余**：门控打开后父类仍会跑 `ego_history_encoder`（0.1% FLOPs，判据0 实测），
其输出被包装器丢弃 ⇒ 该子模块**不再获得梯度**。保留它是为了不动父类、且让
`bn_calibration._get_bn` 的 `ego_history_encoder[1]` 路径继续成立。
🟢 **一个连带后果值得记录**：ego 分支不再经过那个
`BatchNorm1d(track_running_stats=False)` ⇒ **D0#13 的 batch 口径断层在本变体上自动消失**
（开环 batch=32 与闭环 batch=1 的 ego 分支读数一致），BN calibration 对本变体成为空操作。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parents[0]
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_ego_query_fusion import MagnetEgoQueryFusion  # noqa: E402

__all__ = [
    "EgoStateAlignmentError",
    "EgoStateAttentionEncoder",
    "MagnetEgoShortcutSuppressed",
    "ego_current_state_from_history",
    "EGO_SHORTCUT_TAG",
    "EGO_DT",
    "EGO_STATE_CHANNELS",
    "EGO_STATE_ALWAYS_VISIBLE",
]

EGO_SHORTCUT_TAG = "ego_shortcut_v1"

#: 采样间隔 [s]：`config.py:48-51` PAST_TIME_HORIZON=2 / NUM_PAST_POSES=20。
#: ⚠️ 只影响通道的物理量纲；每通道各有一个 `nn.Linear(1, dim)`，常数缩放会被权重吸收。
EGO_DT = 0.1

#: 当前状态通道数（与 PlanTF `state_channel=6` 对齐）
EGO_STATE_CHANNELS = 6
#: 前 N 个通道恒可见、不参与 dropout（PlanTF `visible_tokens = zeros((B, 3))`）
EGO_STATE_ALWAYS_VISIBLE = 3

#: 判定「车辆静止」的每帧位移阈值 [m]。低于它时增量方向由跟踪抖动决定，
#: `atan2` 会给出随机朝向 ⇒ yaw_rate 置 0，而不是把噪声当信号喂进模型。
#: 0.001 m/帧 = 0.01 m/s，与 Q32 漏斗里 `avg_speed<0.01 m/s` 的静止判据同量级。
_STATIONARY_EPS = 1e-3


class EgoStateAlignmentError(RuntimeError):
    """`ego_history` 不符合 `[B,20,2]` 契约，或样本数与 ego 状态数不符 —— 拒绝执行，不静默退化。"""


def ego_current_state_from_history(ego_history_seq: torch.Tensor,
                                   dt: float = EGO_DT) -> torch.Tensor:
    """`[B, 20, 2]` 局部逐帧增量 → `[B, 6]` 当前状态。

    通道定义见模块 docstring。**只用最后两帧**，故 index 0..17 的历史对输出无任何影响 ——
    这正是「切断捷径」的可验证含义（单测判据 C 就测这条）。

    :param ego_history_seq: `[B, 20, 2]`，index 19 = 最近一帧增量（训练/部署同约定）。
    :param dt: 采样间隔 [s]。
    """
    if ego_history_seq.dim() != 3 or ego_history_seq.size(-1) != 2:
        raise EgoStateAlignmentError(
            f"预期 ego_history 为 [B, T, 2]，收到 {tuple(ego_history_seq.shape)}")
    if ego_history_seq.size(1) < 2:
        raise EgoStateAlignmentError(
            f"当前状态需要最后两帧增量，但只有 {ego_history_seq.size(1)} 帧")

    d1 = ego_history_seq[:, -1, :]          # 最近一帧增量 [B, 2]
    d0 = ego_history_seq[:, -2, :]          # 上一帧增量   [B, 2]

    vel_x = d1[:, 0] / dt                                   # 纵向速度   [m/s]
    acc_x = (d1[:, 0] - d0[:, 0]) / (dt * dt)               # 纵向加速度 [m/s²]

    # yaw_rate：由两帧增量的方向差得到；任一帧接近静止时置 0（见 _STATIONARY_EPS）
    moving = (d1.norm(dim=-1) > _STATIONARY_EPS) & (d0.norm(dim=-1) > _STATIONARY_EPS)
    dtheta = torch.atan2(d1[:, 1], d1[:, 0]) - torch.atan2(d0[:, 1], d0[:, 0])
    dtheta = torch.remainder(dtheta + math.pi, 2 * math.pi) - math.pi   # wrap 到 (-π, π]
    yaw_rate = torch.where(moving, dtheta / dt, torch.zeros_like(dtheta))

    # 前 3 维恒为 0：MAGNET 的局部坐标系原点就锚在当前 ego 位姿上，
    # 等价于 PlanTF `NuplanFeature.normalize` 的 `current_state[:3] = 0`。
    zeros = torch.zeros(ego_history_seq.size(0), EGO_STATE_ALWAYS_VISIBLE,
                        dtype=ego_history_seq.dtype, device=ego_history_seq.device)
    return torch.cat([zeros, vel_x[:, None], acc_x[:, None], yaw_rate[:, None]], dim=1)


class EgoStateAttentionEncoder(nn.Module):
    """PlanTF `StateAttentionEncoder` 的移植（`agent_encoder.py:95-140`，`2309.10443`）。

    每个状态通道各自过一个 `Linear(1, dim)` 成一个 token，加位置嵌入后由一个可学习 query
    做 attention 汇聚。训练时前 `EGO_STATE_ALWAYS_VISIBLE` 个 token 恒可见、其余各自以
    `state_dropout` 概率被 `key_padding_mask` 屏蔽；**eval 下一律不屏蔽**。

    🔴 恒可见的 3 个 token 是**功能必需**：否则存在「全部 key 被 mask」的样本，
    `MultiheadAttention` 会输出 NaN。
    """

    def __init__(self, state_channel: int = EGO_STATE_CHANNELS, dim: int = 128,
                 state_dropout: float = 0.75, num_heads: int = 4) -> None:
        super().__init__()
        if state_channel <= EGO_STATE_ALWAYS_VISIBLE:
            raise ValueError(
                f"state_channel 必须 > {EGO_STATE_ALWAYS_VISIBLE}，否则 dropout 恒为空操作")
        if not 0.0 <= state_dropout < 1.0:
            raise ValueError(f"state_dropout 须在 [0, 1)，收到 {state_dropout}")
        self.state_channel = int(state_channel)
        self.state_dropout = float(state_dropout)
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, state_channel, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[B, C]` 当前状态 → `[B, dim]`。"""
        if x.dim() != 2 or x.size(1) != self.state_channel:
            raise EgoStateAlignmentError(
                f"预期状态为 [B, {self.state_channel}]，收到 {tuple(x.shape)}")

        x_embed = torch.stack(
            [lin(x[:, i, None]) for i, lin in enumerate(self.linears)], dim=1)  # [B, C, D]
        x_embed = x_embed + self.pos_embed

        key_padding_mask: Optional[torch.Tensor] = None
        if self.training and self.state_dropout > 0:
            visible = torch.zeros((x.size(0), EGO_STATE_ALWAYS_VISIBLE),
                                  device=x.device, dtype=torch.bool)
            dropped = torch.rand(
                (x.size(0), self.state_channel - EGO_STATE_ALWAYS_VISIBLE),
                device=x.device) < self.state_dropout
            key_padding_mask = torch.cat([visible, dropped], dim=1)

        query = self.query.expand(x.size(0), -1, -1)
        out = self.attn(query=query, key=x_embed, value=x_embed,
                        key_padding_mask=key_padding_mask, need_weights=False)[0]
        return out[:, 0]                                                    # [B, dim]


class _EgoStateTransformer(nn.Module):
    """包装父类的 `ego_transformer`：门控打开时**不执行内层**，直接返回当前状态编码。

    返回 `[B, 1, D]` ⇒ 父类 `transformer_predictor.py:211` 的 `.mean(dim=1)` 成为恒等操作。
    `reset_state(None)` ⇒ 与父类逐字节一致。
    """

    def __init__(self, inner: nn.Module, encoder: EgoStateAttentionEncoder) -> None:
        super().__init__()
        self.inner = inner
        self.encoder = encoder
        self._state: Optional[torch.Tensor] = None

    def reset_state(self, state: Optional[torch.Tensor]) -> None:
        self._state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._state is None:                 # 未启用：与父类逐字节一致
            return self.inner(x)
        if x.size(0) != self._state.size(0):
            raise EgoStateAlignmentError(
                f"ego_transformer 收到 {x.size(0)} 个样本，但本批只有 "
                f"{self._state.size(0)} 个 ego 当前状态；消费顺序与父类不一致，拒绝执行")
        return self.encoder(self._state).unsqueeze(1)                       # [B, 1, D]


def _batch_size(data) -> int:
    """复刻父类 `transformer_predictor.py:156-166` 的 batch_size 判定，**必须逐条一致**。"""
    if hasattr(data, "batch") and data.batch is not None:
        return int(data.batch.max().item()) + 1
    if hasattr(data, "y") and data.y is not None:
        if data.y.dim() == 2:
            return data.y.shape[0] // 60
        if data.y.dim() == 3:
            return data.y.shape[0]
    return 1


def _ego_history_seq(data) -> torch.Tensor:
    """取出 `[B, 20, 2]` 的 ego 历史。**不合契约就抛错**，不猜、不补零。

    ⚠️ 父类对不合契约的输入会静默退化成零填充（`:193`/`:203`）或走
    `ego_history_encoder` 直连分支（`:186-190`）。那些退化路径下「当前状态」无从定义，
    静默继续等于让实验读数失去意义 ⇒ 这里一律拒绝。
    """
    eh = getattr(data, "ego_history", None)
    if eh is None or eh.numel() == 0:
        raise EgoStateAlignmentError("data.ego_history 缺失或为空，无法构造 ego 当前状态")
    b = _batch_size(data)
    if eh.dim() == 3:
        if eh.size(0) != b or eh.size(1) != 20 or eh.size(2) != 2:
            raise EgoStateAlignmentError(
                f"ego_history 形状 {tuple(eh.shape)} 不符合 [{b}, 20, 2]")
        return eh
    if eh.dim() == 2 and eh.size(1) == 2 and eh.size(0) == b * 20:
        return eh.view(b, 20, 2)
    raise EgoStateAlignmentError(
        f"ego_history 形状 {tuple(eh.shape)} 不符合契约（batch_size={b}，"
        "期望 [B*20, 2] 或 [B, 20, 2]）；EGO_HISTORY 必须是 [20,2] 局部增量")


class MagnetEgoShortcutSuppressed(MagnetEgoQueryFusion):
    """父类（D0#7 GNN 逐样本池化 + 判据0 context 池化 + C2 ego-query 融合）+ **闸④ 切断 ego 捷径**。

    `use_ego_history=True`（默认）时与 `MagnetEgoQueryFusion` **逐字节一致**。
    """

    def __init__(
        self,
        strict_coverage: bool = True,
        context_pooling: bool = False,
        context_pool_mode: str = "max",
        ego_query_fusion: bool = False,
        ego_query_heads: int = 4,
        use_ego_history: bool = True,
        ego_state_dropout: float = 0.75,
        ego_state_channels: int = EGO_STATE_CHANNELS,
        ego_dt: float = EGO_DT,
    ) -> None:
        super().__init__(
            strict_coverage=strict_coverage,
            context_pooling=context_pooling,
            context_pool_mode=context_pool_mode,
            ego_query_fusion=ego_query_fusion,
            ego_query_heads=ego_query_heads,
        )
        self.use_ego_history = bool(use_ego_history)
        self.ego_state_dropout = float(ego_state_dropout)
        self.ego_state_channels = int(ego_state_channels)
        self.ego_dt = float(ego_dt)

        self.ego_state_emb = EgoStateAttentionEncoder(
            state_channel=ego_state_channels, dim=128, state_dropout=ego_state_dropout)
        self.ego_transformer = _EgoStateTransformer(self.ego_transformer, self.ego_state_emb)
        # 🔴 见模块 docstring 技巧 2：父类的 `_capture_ego` 挂在内层，门控打开后内层不执行
        #    ⇒ 必须把同一个 hook 再挂到包装器上，否则 C2 的 ego query 取不到值（或取到历史值）。
        self.ego_transformer.register_forward_hook(self._capture_ego)

    def _prepare_ego_state(self, data):
        if self.use_ego_history:
            self.ego_transformer.reset_state(None)
            return
        self.ego_transformer.reset_state(
            ego_current_state_from_history(_ego_history_seq(data), self.ego_dt))

    def forward(self, data):  # type: ignore[override]
        self._prepare_ego_state(data)
        return super().forward(data)

    def get_encoder_features(self, data):  # type: ignore[override]
        # ⚠️ 父类 `get_encoder_features` 的 ego 分支是 `except Exception: pass`（静默吞异常），
        #    故契约校验必须放在这里、在进父类之前完成。
        self._prepare_ego_state(data)
        return super().get_encoder_features(data)
