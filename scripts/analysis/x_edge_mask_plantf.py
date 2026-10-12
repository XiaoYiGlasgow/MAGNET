"""X —— PlanTF 侧边级干预的原语（判据 `X_跨仪器边级与逐层响应_预注册判据_20260820`）。

这个模块回答什么
----------------
F（留车切边）在 MAGNET 上测到「切断 agent↔agent 边比切等量无关边多推动轨迹 0.079 pt」。
但被测模型闭环 0.19–0.23，**不称职** ⇒ 「低承载是模型类性质」与「低承载是差模型的性质」
这两个解释混在一起。PlanTF 闭环 0.6554 > IDM 0.591，在它身上跑**同一个干预**才能拆开。

🔴 为什么以前判「PlanTF 切不开」，以及那个判定为什么不成立
--------------------------------------------------------
主控表 §7.7 原写「协议 v1 已判定边级干预不可移植到 PlanTF（单通路、切不开）」。
实现层面不成立：`TransformerEncoderLayer.forward(src, mask, key_padding_mask)` 把 `mask`
直通 `torch.nn.MultiheadAttention(attn_mask=...)`，而 self-attention 本身就是完全图上的
加权消息传递 —— `attn_mask` 就是它的邻接矩阵。`planning_model.py` 现调用只传
`key_padding_mask`，`mask` 位一直空着。

08-20 冒烟（`scratchpad/plantf_edge_mask_smoke.py`，随机权重）：
    关闭时 max|Δ| = 0.000e+00
    depth=1 切 a2a：ego Δ 恰为 0（agent↔agent 只改 agent 自身表示）
    depth=2 起传导到 ego，depth=4（PlanTF 实配）Δ = 5.5e-02
层数依赖正是机制正确的正面证据。

token 布局（已核 `nuplan_feature_builder.py:277-279` argsort + `:177-180` ego 前置）
-----------------------------------------------------------------------------------
    index 0        = ego
    index 1..A-1   = 周边 agent，**由近及远**
    index A..N-1   = map polygon

与 MAGNET 侧 `f_edge_family_audit.py` 的族定义逐条对齐
-----------------------------------------------------
| 族 | MAGNET（离散图无向半边） | PlanTF（注意力对） |
|---|---|---|
| `a2a` | 两端都是非 ego agent | `i,j ∈ [1,A)`，`i≠j` |
| `E` | 一端 ego、一端 agent | `{0,j}`，`j ∈ [1,A)` |
| `LL` | 两端都是 lane | `i,j ∈ [A,N)`，`i≠j` |

🔴 **计数口径 = 无向对**（与 MAGNET 侧 `n_cut_undirected` 直接可比）：
屏蔽一个"无向对" `{i,j}` = 同时置 `mask[i,j]` 与 `mask[j,i]` 为 True。
对角线 `i→i` 一律保留、不计入任何族。

🔴 多模态口径沿用 G18 协议 §4.3：primary 用**固定 baseline best_mode**的位移。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

PLANTF_ROOT = Path("/mnt/local/Xiao_Yi/external/planTF")
if str(PLANTF_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANTF_ROOT))

X_PROTOCOL = "X_EDGE_MASK_PLANTF_v1_20260820"
FAMILIES: Tuple[str, ...] = ("a2a", "E", "LL")
CELLS: Tuple[str, ...] = ("a2a_all", "ctrl_count", "ctrl_frac", "E_all", "EA_all")
CELL_ROLE: Dict[str, str] = {
    "a2a_all": "primary-treat：屏蔽全部 agent↔agent 注意力对（判据 §3-1）",
    "ctrl_count": "primary-control：map↔map 池随机屏蔽等量无向对（count-matched，判据 §2）",
    "ctrl_frac": "secondary：屏蔽全部 map↔map（fraction-matched 附注，判据 §3-4）",
    "E_all": "secondary：屏蔽全部 ego↔agent（判据 §3-4）",
    "EA_all": "secondary：屏蔽 E+a2a（判据 §3-4）",
}


@dataclass(frozen=True)
class TokenLayout:
    """一个样本的 token 布局与各族的无向对清单。"""

    n_total: int                       # ego + agents + maps
    n_agent: int                       # 含 ego，即 index 0..n_agent-1 属 agent 张量
    valid_agent: np.ndarray            # 有效 agent 下标（含 ego 的 0）
    valid_map: np.ndarray              # 有效 map 下标
    pairs: Dict[str, np.ndarray]       # family -> (k, 2) 无向对下标


def build_layout(agent_key_padding: torch.Tensor,
                 polygon_key_padding: torch.Tensor) -> TokenLayout:
    """从两个 key_padding_mask 推出布局与三族的无向对。

    :param agent_key_padding: `[A]`，True = padding（`~valid_mask.any(-1)`），index 0 是 ego。
    :param polygon_key_padding: `[M]`，True = padding。
    """
    a_pad = agent_key_padding.detach().cpu().numpy().astype(bool)
    m_pad = polygon_key_padding.detach().cpu().numpy().astype(bool)
    A, M = a_pad.size, m_pad.size
    valid_agent = np.flatnonzero(~a_pad)                 # 含 ego（下标 0）
    valid_map = np.flatnonzero(~m_pad) + A               # 平移到全局下标
    others = valid_agent[valid_agent != 0]               # 非 ego 的有效 agent

    def _upper_pairs(idx: np.ndarray) -> np.ndarray:
        if idx.size < 2:
            return np.zeros((0, 2), dtype=np.int64)
        iu, ju = np.triu_indices(idx.size, k=1)
        return np.stack([idx[iu], idx[ju]], axis=1).astype(np.int64)

    e_pairs = (np.stack([np.zeros(others.size, dtype=np.int64), others], axis=1)
               if others.size else np.zeros((0, 2), dtype=np.int64))
    return TokenLayout(
        n_total=A + M, n_agent=A, valid_agent=valid_agent, valid_map=valid_map,
        pairs={"a2a": _upper_pairs(others), "E": e_pairs, "LL": _upper_pairs(valid_map)})


def make_attn_mask(n_total: int, pairs: np.ndarray,
                   device: torch.device) -> Optional[torch.Tensor]:
    """把无向对清单变成 `MultiheadAttention` 的 2D bool `attn_mask`（True = 禁止 attend）。

    🔴 对称屏蔽：一个无向对 `{i,j}` ⇒ `mask[i,j] = mask[j,i] = True`。
    🔴 对角线恒 False（self-attention 一律保留）。
    空清单返回 `None` —— 判据 §1.4 的空操作门要求「干预集为空时与 base 逐位相同」，
    传 None 比传全 False 更强（连 mask 分支都不进）。
    """
    if pairs.size == 0:
        return None
    m = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
    i, j = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
    m[i, j] = True
    m[j, i] = True
    m.fill_diagonal_(False)
    return m


def forward_with_edge_mask(model, data: Dict, attn_mask: Optional[torch.Tensor],
                           collect_layers: bool = False) -> Dict:
    """复刻 `PlanningModel.forward` 但把 `attn_mask` 传进每个 encoder block。

    🔴 **不改 vendor 原文件**（`external/planTF` 保持干净）：这里按原 forward 逐行重写，
    唯一差别是 `blk(...)` 多传一个 `mask=`。原 forward 见
    `external/planTF/src/models/planTF/planning_model.py:139-181`。
    🔴 `attn_mask=None` 时逐行等价于原 forward —— 单测 V-X1 逐位验证这一点。

    :param collect_layers: 判据 §2 的逐层探针；True 时额外返回每层的 ego token 与 agent 均值。
    """
    agent_pos = data["agent"]["position"][:, :, model.history_steps - 1]
    agent_heading = data["agent"]["heading"][:, :, model.history_steps - 1]
    agent_mask = data["agent"]["valid_mask"][:, :, : model.history_steps]
    polygon_center = data["map"]["polygon_center"]
    polygon_mask = data["map"]["valid_mask"]

    bs, A = agent_pos.shape[0:2]
    position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
    angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
    pos = torch.cat([position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1)
    pos_embed = model.pos_emb(pos)

    agent_key_padding = ~(agent_mask.any(-1))
    if getattr(model, "agent_bandwidth", None) is not None:
        agent_key_padding = model._apply_agent_bandwidth(agent_key_padding)
    polygon_key_padding = ~(polygon_mask.any(-1))
    key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

    x = torch.cat([model.agent_encoder(data), model.map_encoder(data)], dim=1) + pos_embed

    layers = []
    for blk in model.encoder_blocks:
        x = blk(x, mask=attn_mask, key_padding_mask=key_padding_mask)
        if collect_layers:
            layers.append({"ego": x[:, 0].detach().clone(),
                           "agent_mean": x[:, 1:A].detach().clone().mean(dim=1)})
    x = model.norm(x)

    trajectory, probability = model.trajectory_decoder(x[:, 0])
    out = {"trajectory": trajectory, "probability": probability,
           "ego_token": x[:, 0].detach().clone()}
    if collect_layers:
        out["layers"] = layers
    return out


def pick_mode(probability: torch.Tensor) -> torch.Tensor:
    """G18 协议 §4.3：primary 用固定 baseline best_mode。"""
    return probability.argmax(dim=-1)


def trajectory_xy(trajectory: torch.Tensor, mode: torch.Tensor,
                  horizon_steps: Optional[int] = None) -> torch.Tensor:
    """取指定 mode 的 (x, y) 轨迹；`horizon_steps` 用于与 MAGNET 的 60 步对齐。"""
    bs = trajectory.shape[0]
    traj = trajectory[torch.arange(bs, device=trajectory.device), mode][..., :2]
    return traj if horizon_steps is None else traj[:, :horizon_steps]


def response(model, data: Dict, pairs: np.ndarray, device: torch.device,
             horizon_steps: Optional[int] = None,
             collect_layers: bool = False) -> Dict:
    """一个 cell 的读数：U = mean_disp / base_scale（与 MAGNET 侧同名同义）。

    🔴 `base_scale` = baseline 轨迹相对起点的平均位移幅度（与 MAGNET 侧定义同源），
    `mean_disp` = 干预前后逐帧位移的均值，**固定 baseline best_mode**。
    """
    with torch.no_grad():
        base = forward_with_edge_mask(model, data, None, collect_layers=collect_layers)
        mode = pick_mode(base["probability"])
        base_xy = trajectory_xy(base["trajectory"], mode, horizon_steps)

        mask = make_attn_mask(_n_total(data), pairs, device)
        cut = forward_with_edge_mask(model, data, mask, collect_layers=collect_layers)
        cut_xy = trajectory_xy(cut["trajectory"], mode, horizon_steps)
        free_mode = pick_mode(cut["probability"])

    disp = torch.linalg.norm(cut_xy - base_xy, dim=-1)            # [bs, T]
    scale = torch.linalg.norm(base_xy, dim=-1).mean().item()
    out = {
        "mean_disp": float(disp.mean().item()),
        "max_disp": float(disp.max().item()),
        "base_scale": float(scale),
        "usage": float(disp.mean().item() / scale) if scale > 0 else None,
        "mode_switch": bool((free_mode != mode).any().item()),
        "n_cut_undirected": int(pairs.shape[0]),
    }
    if collect_layers:
        out["layer_response"] = [
            {"ego": _rel(b["ego"], c["ego"]), "agent_mean": _rel(b["agent_mean"], c["agent_mean"])}
            for b, c in zip(base["layers"], cut["layers"])]
    return out


def _n_total(data: Dict) -> int:
    return int(data["agent"]["position"].shape[1] + data["map"]["polygon_center"].shape[1])


def _rel(base: torch.Tensor, cut: torch.Tensor) -> Optional[float]:
    """判据 §2.2 的逐层相对响应 R = ‖h_cut − h_base‖ / ‖h_base‖。

    🔴 R 与 U 不同口径（§0.4-2）：descriptive only，禁止与 U 相除当传递率。
    """
    d = torch.linalg.norm(cut - base).item()
    n = torch.linalg.norm(base).item()
    return float(d / n) if n > 0 else None
