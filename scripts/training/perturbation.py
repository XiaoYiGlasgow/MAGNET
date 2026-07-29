"""Ego state perturbation 数据增广（VT-2026-04191 阶段⑫ D2-5）。

动机（阶段9 §3.3）：`perturbation 增广缺失` 与 `route 缺失` 并列为唯二「预期收益：高」项。
PlanTF（arXiv 2309.10443）把 state perturbation 作为核心贡献 —— 纯 open-loop 回归训练出的
planner 在闭环里会因 covariate shift 复合发散（我们实测：t=0 对齐、前 3 s 贴合 <0.5 m、
之后单调发散至末端 −12.6 m，阶段5 md:33-38）。增广让模型见过「我已经偏了」的状态并学会纠偏。

机制：假装 ego 实际处在偏离位姿 `(dx, dy, dtheta)`，把整个场景变换到该偏离视角下，
而 **ground truth 仍指向原本正确的全局轨迹** —— 模型因此被迫学习「从偏离状态回到正轨」。

实现为 PyG `transform`（每次 `__getitem__` 调用），故：
  - **在线增广**：每个 epoch 采样不同扰动，等效数据量远大于一次性离线扩增；
  - **不改预处理产物**，不动 E0 冻结的 `dataset.py`；
  - 只在 train split 挂载，val/test 必须传 `transform=None`。

坐标约定（与预处理/planner 一致）：ego-centric，原点 = ego 当前位姿，x 轴 = ego heading（前），
y 轴 = 左。节点特征 12 列布局：
    col0-3 = (x_start, y_start, x_end, y_end)   ← 位置，需旋转+平移
    col4-5 = (heading_start, heading_end)        ← 朝向，需减 dtheta
    col6-8 = 对 ego/agent 是 (timestamp, v_start, v_end)；对 lane 是 route 三通道
             ← **标量语义，旋转/平移不变，一律不动**
    col9   = subgraph_ID（cluster）              ← 不动
    col10-11 = (t_start, t_end)                  ← 不动

`gnn_x` 是各子图节点特征的**均值池化**结果，本模块对它**重新池化**而不是独立变换。

  ⚠️ 早先这里写的是「旋转+平移是仿射变换，与取均值可交换（`mean(T(x)) == T(mean(x))`），
  故对 `gnn_x` 施加同一变换与先变换再池化等价」—— **该结论已被实现否定，勿再引用**。
  位置列确实可交换，但 heading 列的 `wrap_to_pi` 是**非线性**的：

      mean(wrap(h_i - dθ))  ≠  wrap(mean(h_i) - dθ)

  跨 ±π 边界的子图会给出完全不同的均值朝向。故 `_repool_gnn_x()` 按 `subgraph_ID`(col9)
  对**变换后的 x** 重新取均值，使 `gnn_x` 恒等于「真实存在该扰动场景、走完整 pipeline」
  应得的结果。只有在拿不到可对齐的 `x`（缺 col9）时才退回直接变换作为兜底
  —— 那条兜底路径位置精确、heading 近似。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_POS_COLS = [(0, 1), (2, 3)]   # (x_start,y_start), (x_end,y_end)
_HEADING_COLS = [4, 5]


def _rotate_xy(x: torch.Tensor, y: torch.Tensor, cos_t: float, sin_t: float):
    """绕原点旋转 -dtheta：R(-t) = [[cos t, sin t], [-sin t, cos t]]。"""
    return x * cos_t + y * sin_t, -x * sin_t + y * cos_t


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _apply_rigid(feat: torch.Tensor, dx: float, dy: float,
                 cos_t: float, sin_t: float, dtheta: float) -> torch.Tensor:
    """对特征矩阵施加刚体变换：位置列先平移再旋转，heading 列减 dtheta 并 wrap。

    col6 及之后（route 三通道 / subgraph_ID / 时间列）**一律不动**。
    """
    out = feat.clone()
    for cx, cy in _POS_COLS:
        px, py = out[:, cx] - dx, out[:, cy] - dy
        out[:, cx], out[:, cy] = _rotate_xy(px, py, cos_t, sin_t)
    for ch in _HEADING_COLS:
        out[:, ch] = _wrap_to_pi(out[:, ch] - dtheta)
    return out


def _repool_gnn_x(x_new: torch.Tensor, gnn_x: torch.Tensor, dx: float, dy: float,
                  cos_t: float, sin_t: float, dtheta: float) -> torch.Tensor:
    """按 subgraph_ID(col9) 对**变换后的 x** 取均值，重建 gnn_x。

    对齐方式：`subgraph_pooling` 对每个子图做 `mean(axis=0)`，而子图内 col9 恒等于该
    subgraph_ID，故 `gnn_x[:, 9]` 即每行对应的 subgraph_ID —— 用它把重算的均值映射回
    gnn_x 的行序，无需额外元数据。

    x 中不存在的 subgraph（`x_combined` 会跳过过短/无效 polyline，而池化用的是完整
    `POLYLINE_FEATURES`）保留直接变换的结果作为兜底。
    """
    fallback = _apply_rigid(gnn_x, dx, dy, cos_t, sin_t, dtheta)

    ids = x_new[:, 9].round().long()
    uniq, inv = torch.unique(ids, return_inverse=True)
    sums = torch.zeros(uniq.numel(), x_new.shape[1], dtype=x_new.dtype, device=x_new.device)
    sums.index_add_(0, inv, x_new)
    cnts = torch.zeros(uniq.numel(), dtype=x_new.dtype, device=x_new.device)
    cnts.index_add_(0, inv, torch.ones_like(ids, dtype=x_new.dtype))
    means = sums / cnts.clamp(min=1).unsqueeze(1)          # [n_subgraph, F]

    # uniq 已升序 -> 用 searchsorted 把 gnn_x 的 subgraph_ID 映射到 means 的行
    g_ids = gnn_x[:, 9].round().long()
    pos = torch.searchsorted(uniq, g_ids.clamp(min=int(uniq[0]), max=int(uniq[-1])))
    pos = pos.clamp(max=uniq.numel() - 1)
    hit = uniq[pos] == g_ids                                # 该 subgraph 在 x 中确实存在

    out = fallback.clone()
    F = min(gnn_x.shape[1], means.shape[1])
    if hit.any():
        out[hit, :F] = means[pos[hit], :F]
    return out


class EgoPerturbation:
    """随机 ego 位姿扰动 transform。

    :param max_lateral: [m] 横向（y，车体左）偏移上界。横向偏移是 drivable-area 漂移的主形态，
        给的幅度最大。
    :param max_longitudinal: [m] 纵向（x，车体前）偏移上界。给得小，因为纵向偏移主要影响
        进度而非车道保持。
    :param max_yaw: [rad] 朝向偏移上界。
    :param prob: 施加扰动的概率；其余样本保持原样，保证模型仍见得到干净分布。
    :param seed: 可选，固定采样器种子（复现/调试用；训练时通常留 None 走全局 RNG）。
    """

    def __init__(self, max_lateral: float = 1.0, max_longitudinal: float = 0.5,
                 max_yaw: float = 0.1, prob: float = 0.5, seed: Optional[int] = None) -> None:
        self.max_lateral = float(max_lateral)
        self.max_longitudinal = float(max_longitudinal)
        self.max_yaw = float(max_yaw)
        self.prob = float(prob)
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (f"EgoPerturbation(lat={self.max_lateral}, lon={self.max_longitudinal}, "
                f"yaw={self.max_yaw}, p={self.prob})")

    def _sample(self):
        dx = float(self._rng.uniform(-self.max_longitudinal, self.max_longitudinal))
        dy = float(self._rng.uniform(-self.max_lateral, self.max_lateral))
        dt = float(self._rng.uniform(-self.max_yaw, self.max_yaw))
        return dx, dy, dt

    def __call__(self, data):
        if self._rng.random() >= self.prob:
            return data

        dx, dy, dtheta = self._sample()
        cos_t, sin_t = float(np.cos(dtheta)), float(np.sin(dtheta))

        # ---- 1. 节点特征 x：位置先平移再旋转；朝向减 dtheta；col6-8 一律不动 ----
        x = getattr(data, "x", None)
        if x is not None and x.numel() > 0 and x.shape[1] >= 6:
            data.x = _apply_rigid(x, dx, dy, cos_t, sin_t, dtheta)

        # ---- 1b. gnn_x：**从变换后的 x 按 subgraph 重新池化**，不做独立近似变换 ----
        # 位置列的仿射变换与取均值可交换，但 heading 的 wrap_to_pi 是**非线性**的：
        #     mean(wrap(h_i - dθ))  ≠  wrap(mean(h_i) - dθ)
        # 故独立变换 gnn_x 会与「真实存在该扰动场景、走完整 pipeline」不一致。
        # 这里复刻 gnn_module.subgraph_pooling 的语义（按 subgraph_ID 取均值），
        # 保证 gnn_x 恒等于对变换后 x 的实际池化结果。
        gnn_x = getattr(data, "gnn_x", None)
        if gnn_x is not None and gnn_x.numel() > 0 and gnn_x.shape[1] >= 10 \
                and x is not None and x.numel() > 0 and x.shape[1] >= 10:
            data.gnn_x = _repool_gnn_x(data.x, gnn_x, dx, dy, cos_t, sin_t, dtheta)
        elif gnn_x is not None and gnn_x.numel() > 0 and gnn_x.shape[1] >= 6:
            # 兜底：拿不到可对齐的 x（缺 col9）时退回直接变换，位置精确、heading 近似
            data.gnn_x = _apply_rigid(gnn_x, dx, dy, cos_t, sin_t, dtheta)

        # ---- 2. GT：增量 → 累计位置 → 变换到偏离视角 → 回增量 ----
        # y 是相对**原** ego 的逐帧增量；cumsum 得原视角下的绝对位置序列 C_i。
        # 偏离后新原点为 (dx,dy)、新朝向多转 dtheta，故 C'_i = R(-dtheta)(C_i - delta)。
        # 新增量 y'_0 = C'_0（从偏离位置迈向第一个 GT 点），y'_i = C'_i - C'_{i-1}。
        # → GT 终点在全局中不变，模型必须学会「先纠偏、再跟上原轨迹」。
        if getattr(data, "y", None) is not None and data.y.numel() > 0:
            y = data.y.clone().reshape(-1, 2)
            c = torch.cumsum(y, dim=0)
            cx, cy = _rotate_xy(c[:, 0] - dx, c[:, 1] - dy, cos_t, sin_t)
            c_new = torch.stack([cx, cy], dim=1)
            y_new = torch.empty_like(c_new)
            y_new[0] = c_new[0]
            y_new[1:] = c_new[1:] - c_new[:-1]
            data.y = y_new.reshape(data.y.shape)

        # ---- 3. ego_history：局部逐帧增量（位移向量），只旋转不平移 ----
        if getattr(data, "ego_history", None) is not None and data.ego_history.numel() > 0:
            eh = data.ego_history.clone()
            hx, hy = _rotate_xy(eh[:, 0], eh[:, 1], cos_t, sin_t)
            data.ego_history = torch.stack([hx, hy], dim=1).reshape(eh.shape)

        return data
