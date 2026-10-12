"""路线 B 新结构 S1 第 5 件 —— **joint WTA 损失 + 多模态指标 + P5 监控**。

## 🔴 目标口径：位置域、逐帧 mask、ego 与周车同一个损失

* **位置域**（不是增量域）—— Q106 已查实增量域会**奖励恒定输出**（恒定输出增量域只差 0.40 m、
  位置域 ADE 却是 11.06 m），而路线 B 里约 80% 的周车是静止的；
* **逐帧 mask 不可省** —— `AGENT_FUTURES` 在 track 结束后逐帧补零（实测 val 上
  **17.9%** 的周车未来帧是补零、**32.26%** 的 agent 被截断）。
  🔴 **FDE 必须取「最后一个有效帧」而不是第 59 帧** —— 否则把「补零」读成「开到了原点」，
  这正是 P0 v1 栽过的坑。

## 🔴🔴 scene-level 还是 per-entity 的 WTA？（新待决 Q113，默认 scene-level）

* **scene-level（默认）**：一个模态 = **整个场景**的一个联合假设，全部实体共用同一个 `m*`。
  这才是文献里 "joint prediction" 的含义，也是 R4-1 要求的重定位所指；
* **per-entity**：每个实体各选各的 `m*` —— 那等于 M 个共享 trunk 的**边缘**预测器，
  审稿人一句「这是 marginal prediction 换了个名字」就能打掉联合性主张。

⚠️ 两者都实现了（`scene_level` 开关），但**默认 scene-level**，且指标两套都报，
以免日后被问「你报的 minADE 是哪种」时口径不清。

## 🔴🔴 指标必须分层，聚合读数会被 86% 的静止车主导（2026-08-25 实测定案）

新档 fullret 里 **86.2% 的监督实体是静止/截断车**（val 8768 个实体：ego 480 / 运动车 728 / 其余 7560）。
2026-08-25 的 P5 探针实测，**同一个 ckpt** 的模态间终点散度按层看完全是两个故事：

| 层 | 占比 | 模态间终点散度中位 | `mode_usage_max` |
|---|---|---|---|
| ego | 5.5% | **4.253 m** | 0.940 |
| agent-moving | 8.3% | **5.957 m** | 0.933 |
| agent-static | 86.2% | **0.016 m** | 0.995 |
| （聚合） | 100% | 0.019 m | 0.987 |

⇒ 🔴 **聚合的 `mode_usage_max` 0.987 会被读成「名义 6 模态、实为单模态」，那是假象** ——
预测一辆停着的车本来就不需要 6 个模态。**任何模态相关的读数不分层就不许用。**
分层判据用 `scene_strata` 的冻结口径（`is_moving` = 全 60 帧有效且 6 s 位移 > 10 m），
🔴 属 **GT 分层**，按 `scene_strata` 纪律①**只能用于诊断与训练侧**，**不得用于 test 分层**。

## 🔴 `hit_rate` 是诊断的命门（沿用 `magnet_multimodal.min_ade_fde` 的口径）

多模态把一个问题拆成两个：①生成 K 条合理候选 ②**挑对那一条**。
只看 top1 会**分不清是候选集不好还是选模没学会**，而这两种情况下一步完全相反。
随机基线 = `1/K`（K=6 ⇒ 0.167），判读时必须与它比而不是与 0 比。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from magnet_joint_decoder import JointPrediction
from magnet_joint_encoder import EntityIndex, EntityIndexError
from magnet_joint_layers import build_entity_geometry
# 🔴 分层阈值只有一个真相源：`scene_strata`（A1 冻结，2026-08-22）。不得在这里另写一个数。
from src.shared.utils.scene_strata import FULL_FRAMES as _STRATA_FULL_FRAMES
from src.shared.utils.scene_strata import MOVING_DISP_M as _STRATA_MOVING_DISP_M

#: miss rate 的终点误差阈值（m）。Argoverse / WOMD 的通行值；🔴 报数时必须连阈值一起写。
MISS_THRESHOLD_M: float = 2.0

FUTURE_FRAMES: int = 60
_EPS = 1e-6


@dataclass(frozen=True)
class JointTargets:
    """一个 batch 的联合监督目标。**行序 = `gnn_x` 行序**（与预测逐行对齐）。

    :ivar tgt: `[n_entity, 60, 2]` 绝对 ego-centric 位置；非监督实体的行为 0。
    :ivar mask: `[n_entity, 60]` bool 逐帧有效。
    :ivar is_target: `[n_entity]` bool，参与损失的实体（ego + 有 target 的周车）。
    :ivar is_ego: `[n_entity]` bool。
    :ivar is_moving: `[n_entity]` bool —— **运动周车**（`scene_strata` 冻结口径：全 60 帧有效
        且 6 s 位移 > `MOVING_DISP_M`）。🔴 只用于**分层报数**，**不进损失的加权**
        （按结果/GT 加权回归损失就是按结果筛样本）。ego 恒 False（ego 单独成层）。
    :ivar batch: `[n_entity]` 样本号（scene-level WTA 要用）。
    :ivar n_graphs: 批内样本数。
    """

    tgt: torch.Tensor
    mask: torch.Tensor
    is_target: torch.Tensor
    is_ego: torch.Tensor
    is_moving: torch.Tensor
    batch: torch.Tensor
    n_graphs: int

    def strata(self) -> Tuple[Tuple[str, torch.Tensor], ...]:
        """报数用的四层（互斥的三层 + 全部）。🔴 `agent` 层保留是为了与旧读数可比。"""
        sel = self.is_target
        return (("ego", sel & self.is_ego),
                ("agent", sel & ~self.is_ego),
                ("moving", sel & self.is_moving),
                ("static", sel & ~self.is_ego & ~self.is_moving),
                ("all", sel))


def build_joint_targets_batch(data, idx: EntityIndex, ego_only: bool = False) -> JointTargets:
    """从 batch 造联合 target：**ego 侧现算**（`cumsum(data.y)`），**周车侧读 aux 缓存**。

    :param ego_only: 🔴 **Q110 §6.2 的 A 臂** —— 结构不动、**只监督 ego**（= 单 ego 输出任务）。
        它把「B vs A」变成**单变量 = 任务**的对照：模态数、参数量、数据、锚点全部逐位相同，
        只有 `is_target` 从「ego + 周车」缩到「ego」。
        🔴 周车 target **照常构造**（`tgt` / `mask` 不变），只是不进 `is_target`
        —— 这样 A 臂的 target 张量与 B 臂逐位相同，差别只在监督集合，便于事后核对。
    :raises EntityIndexError: 缺 `AgentAuxGraphDataset` 的三个字段，或行数对不上。
        🔴 **不降级** —— 静默丢掉周车监督会让「联合预测」变成一句空话。
    """
    device = data.x.device
    n_ent, n_graphs = idx.n_entity, int(idx.entity_batch.max().item()) + 1
    tgt = torch.zeros(n_ent, FUTURE_FRAMES, 2, device=device)
    mask = torch.zeros(n_ent, FUTURE_FRAMES, dtype=torch.bool, device=device)

    # ---- ego 侧：GROUND_TRUTH 是逐帧增量，cumsum 即绝对 ego-centric 位置（首帧恒 (0,0)）----
    y = data.y.view(n_graphs, FUTURE_FRAMES, 2)
    if float(y[:, 0].abs().max()) > 1e-6:
        raise EntityIndexError("`data.y` 的首帧增量非零 —— 与 `ego_increments_to_positions` 的口径不符")
    tgt[idx.ego_index] = torch.cumsum(y, dim=1)
    mask[idx.ego_index] = True                      # ego 的 GT 恒为全帧有效

    # ---- 周车侧：aux 缓存（`AgentAuxGraphDataset`）。🔴 `gnn_is_agent` **不含 ego** ----
    for f in ("gnn_is_agent", "agent_future_tgt", "agent_future_mask"):
        if getattr(data, f, None) is None:
            raise EntityIndexError(
                f"batch 缺 `{f}` —— joint 训练必须用 `AgentAuxGraphDataset`；"
                f"缓存由 `build_agent_aux_cache.py --root <split> --adjacency_config default` 生成")
    is_ag = data.gnn_is_agent.to(torch.bool)
    if int(is_ag.numel()) != n_ent:
        raise EntityIndexError(f"`gnn_is_agent` 有 {int(is_ag.numel())} 行但 n_entity={n_ent}")
    rows = torch.nonzero(is_ag, as_tuple=True)[0]
    a_tgt, a_msk = data.agent_future_tgt.to(device), data.agent_future_mask.to(device)
    if int(a_tgt.shape[0]) != int(rows.numel()):
        raise EntityIndexError(
            f"周车 target {int(a_tgt.shape[0])} 行 != agent 节点 {int(rows.numel())} 行 —— "
            f"batch 拼接后 `gnn_is_agent` 与 `agent_future_tgt` 失去对齐，继续训练会产出系统性错误监督")
    tgt[rows] = a_tgt
    mask[rows] = a_msk

    is_ego = torch.zeros(n_ent, dtype=torch.bool, device=device)
    is_ego[idx.ego_index] = True
    # 🔴 全帧补零的周车不进损失（join 成功但整槽为零，val 实测 0.66%）：它不是「停住了」而是没数据
    is_target = (is_ego | is_ag) & idx.entity_valid & mask.any(dim=1)
    if ego_only:
        is_target = is_target & is_ego                 # A 臂：结构不动，只把监督集合缩到 ego

    # ---- 分层标志（`scene_strata` 冻结口径；只报数不进损失）--------------------------------
    #      位移锚点用 `build_entity_geometry` 的 `pos`（= 实体最后一个有效向量的终点 = t0 位置），
    #      与 decoder 的输出锚点、与 `agent_strata` 的 `agent_hist[-1, :2]` 三者同一个点。
    #      🔴 终点取**最后一个有效帧**而不是第 59 帧 —— 用 `tgt[-1]` 会把补零读成「开到了原点」（P0 v1 的坑）。
    pos0 = build_entity_geometry(data.x, idx).pos
    last = mask.float().cumsum(dim=-1).argmax(dim=-1)
    end = tgt.gather(1, last.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    disp = torch.norm(end - pos0, dim=-1)
    is_moving = (is_target & ~is_ego
                 & (mask.sum(dim=1) >= _STRATA_FULL_FRAMES)
                 & (disp > _STRATA_MOVING_DISP_M))
    return JointTargets(tgt=tgt, mask=mask, is_target=is_target, is_ego=is_ego,
                        is_moving=is_moving, batch=idx.entity_batch, n_graphs=n_graphs)


def per_mode_errors(traj: torch.Tensor, tg: JointTargets) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐实体逐模态的 (ADE, FDE)。**都按逐帧 mask 算**，FDE 取**最后一个有效帧**。

    :param traj: `[n_entity, M, 60, 2]`。
    :returns: `(ade [n_entity, M], fde [n_entity, M])`；无有效帧的实体两者均为 0。
    """
    err = torch.norm(traj - tg.tgt.unsqueeze(1), dim=-1)                  # [n, M, 60]
    m = tg.mask.unsqueeze(1).to(err.dtype)
    ade = (err * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
    # 🔴 最后一个有效帧的下标（不是 59）：cumsum().argmax() 取首个达到最大值的位置
    last = tg.mask.float().cumsum(dim=-1).argmax(dim=-1)                  # [n]
    fde = err.gather(-1, last.view(-1, 1, 1).expand(-1, err.shape[1], 1)).squeeze(-1)
    return ade, fde


def _best_mode(ade: torch.Tensor, tg: JointTargets, scene_level: bool) -> torch.Tensor:
    """选 `m*`。scene-level 时按**场景聚合**的 ADE 选一个模态给全场景实体共用。"""
    if not scene_level:
        return ade.argmin(dim=1)
    sel = tg.is_target
    n_modes = ade.shape[1]
    tot = torch.zeros(tg.n_graphs, n_modes, device=ade.device).index_add_(
        0, tg.batch[sel], ade[sel])
    cnt = torch.zeros(tg.n_graphs, device=ade.device).index_add_(
        0, tg.batch[sel], torch.ones(int(sel.sum()), device=ade.device))
    scene_best = (tot / cnt.clamp(min=1.0).unsqueeze(-1)).argmin(dim=1)    # [B]
    return scene_best[tg.batch]                                            # 广播回实体


class JointWTALoss(nn.Module):
    """联合多智能体 WTA 损失：回归只对 `m*` 回传 + 模态 CE 学置信度。

    :param mode_weight: 模态 CE 的权重。
    :param scene_level: True = 一个模态代表**整个场景**的联合假设（默认，见模块文档 Q113）。
    """

    def __init__(self, mode_weight: float = 1.0, scene_level: bool = True) -> None:
        super().__init__()
        self.mode_weight, self.scene_level = mode_weight, scene_level

    def forward(self, pred: JointPrediction, tg: JointTargets
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sel = tg.is_target
        if not bool(sel.any()):                       # 整批没有可监督实体（空图）
            zero = pred.traj.sum() * 0.0
            return zero, {"total": 0.0, "reg": 0.0, "mode_ce": 0.0, "n_target": 0}
        ade, _fde = per_mode_errors(pred.traj, tg)
        best = _best_mode(ade, tg, self.scene_level)                       # [n_entity]

        rows = torch.nonzero(sel, as_tuple=True)[0]
        chosen = pred.traj[rows, best[rows]]                               # [n_sel, 60, 2]
        fmask = tg.mask[rows]
        reg = F.smooth_l1_loss(chosen[fmask], tg.tgt[rows][fmask])
        ce = F.cross_entropy(pred.logits[rows], best[rows])
        total = reg + self.mode_weight * ce
        return total, {"total": float(total), "reg": float(reg), "mode_ce": float(ce),
                       "n_target": int(rows.numel())}


@torch.no_grad()
def joint_metrics(pred: JointPrediction, tg: JointTargets,
                  miss_threshold: float = MISS_THRESHOLD_M) -> Dict[str, float]:
    """minADE@K / minFDE@K / miss rate / top1 / hit_rate，**按 `JointTargets.strata()` 分层报**。

    🔴 分层是硬要求，两个理由：
    ① ego 侧有常速基线可比（匀速解释 ego 位置域方差 98.80%），周车侧才是交互通道有负载的一侧；
    ② **86.2% 的监督实体是静止/截断车**，合并读数被它们主导（见模块文档的分层表）——
       2026-08-25 实测：聚合 `mode_usage_max` 0.987 vs ego 0.940 / moving 0.933 / static 0.995。
    """
    out: Dict[str, float] = {}
    sel = tg.is_target
    if not bool(sel.any()):
        return {"n_target": 0.0}
    ade, fde = per_mode_errors(pred.traj, tg)
    top1 = pred.logits.argmax(dim=1)
    idx_all = torch.arange(ade.shape[0], device=ade.device)

    for name, sub in tg.strata():
        if not bool(sub.any()):
            continue
        a, f = ade[sub], fde[sub]
        out[f"{name}_minADE"] = float(a.min(dim=1).values.mean())
        out[f"{name}_minFDE"] = float(f.min(dim=1).values.mean())
        out[f"{name}_missRate"] = float((f.min(dim=1).values > miss_threshold).float().mean())
        out[f"{name}_top1ADE"] = float(a[torch.arange(a.shape[0], device=a.device),
                                         top1[sub]].mean())
        out[f"{name}_hitRate"] = float((top1[sub] == a.argmin(dim=1)).float().mean())
        out[f"{name}_n"] = float(int(sub.sum()))

    # scene-level：一个模态给全场景，报「场景最优」而不是「逐实体最优」（两者口径不同，勿混）
    scene_best = _best_mode(ade, tg, scene_level=True)
    out["scene_minADE"] = float(ade[idx_all[sel], scene_best[sel]].mean())
    out["scene_minFDE"] = float(fde[idx_all[sel], scene_best[sel]].mean())
    out["n_target"] = float(int(sel.sum()))
    out["miss_threshold_m"] = miss_threshold
    return out


class MarginFunnelMonitor:
    """P5：每 epoch 记 **margin** 与 **funnel 余弦**的 p1（阶段299 的教训）。

    * **margin** = 末个 ReLU 的**预激活最大值**（≤0 ⟺ 该样本塌陷）。
      🔴 新头的 ReLU 前置了 LayerNorm，但 **LN 的 β 可为负** ——「数学上排除塌陷」只在 β=0 时成立
      （V-H2 的教训：训练后 β 均值 −0.24~−0.28、124/128 个 ≤0）⇒ **必须实测，不得假定**。
    * **funnel 余弦** = 该层输出两两余弦的中位数。塌陷场景实测 0.961–0.988，正常 0.291–0.617。

    用法::

        mon = MarginFunnelMonitor(model.active_head())
        ...  # 跑一个 epoch
        stats = mon.summary(); mon.reset()
    """

    def __init__(self, head: nn.Sequential, pre_relu_index: int = 1,
                 max_rows: int = 4096) -> None:
        self.max_rows = max_rows
        self._buf: list = []
        self._handle = head[pre_relu_index].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out) -> None:
        if len(self._buf) * out.shape[0] < self.max_rows:
            self._buf.append(out.detach().reshape(-1, out.shape[-1]).float().cpu())

    def reset(self) -> None:
        self._buf = []

    def close(self) -> None:
        self._handle.remove()

    def summary(self) -> Dict[str, float]:
        if not self._buf:
            return {}
        z = torch.cat(self._buf)[: self.max_rows]
        margin = z.max(dim=-1).values
        zn = F.normalize(z[:512], dim=-1)
        cos = (zn @ zn.T)
        off = ~torch.eye(cos.shape[0], dtype=torch.bool)
        return {"margin_p1": float(margin.quantile(0.01)),
                "margin_median": float(margin.median()),
                "collapse_rate": float((margin <= 0).float().mean()),
                "funnel_cos_median": float(cos[off].median()),
                "funnel_cos_p99": float(cos[off].quantile(0.99))}


@torch.no_grad()
def mode_usage(pred: JointPrediction, tg: JointTargets) -> Dict[str, float]:
    """P5：模态使用率直方图（**逐层**）。

    🔴 **聚合的 `mode_usage_max` 不许单独用来判 mode collapse** —— 86.2% 的实体是静止车，
    它们本来就该把六个模态压在一起（见模块文档的分层表）。判 collapse 看 `ego_/moving_` 两层，
    并且必须与 :func:`mode_diversity` 的**轨迹散度**一起看：
    `mode_usage_max` 高而散度也高 ⇒ 塌的是**置信度头**不是模态本身（2026-08-25 实测正是这种）。
    """
    sel = tg.is_target
    if not bool(sel.any()):
        return {}
    n_modes = pred.logits.shape[1]
    out: Dict[str, float] = {}
    for name, sub in tg.strata():
        if not bool(sub.any()):
            continue
        frac = torch.bincount(pred.logits[sub].argmax(dim=1),
                              minlength=n_modes).float() / int(sub.sum())
        pre = "" if name == "all" else f"{name}_"          # all 层保持旧键名，勿动（旧读数要可比）
        out[f"{pre}mode_usage_max"] = float(frac.max())
        out[f"{pre}mode_usage_entropy"] = float(
            -(frac.clamp(min=_EPS) * frac.clamp(min=_EPS).log()).sum())
        if name == "all":
            out.update({f"mode_usage_{i}": float(v) for i, v in enumerate(frac)})
    return out


@torch.no_grad()
def mode_diversity(pred: JointPrediction, tg: JointTargets) -> Dict[str, float]:
    """P5 的第二个测点：**模态间轨迹散度**（逐层的两两终点距离均值）。

    🔴 它与 :func:`mode_usage` 缺一不可 —— `mode_usage` 测的是**置信度头**的输出分布，
    散度测的是**模态本身**是否重合。2026-08-25 实测两者背离：
    聚合 `mode_usage_max` 0.987（像塌了）但 ego 散度 4.25 m / moving 5.96 m（根本没塌），
    真正无信息的是置信度头（CE 1.773 ≥ 赢家分布熵 1.744）。
    只看其中一个会得出相反的结论。
    """
    sel = tg.is_target
    if not bool(sel.any()):
        return {}
    n_modes = pred.traj.shape[1]
    if n_modes < 2:
        # 🔴 K=1（单模态对照臂）没有「模态之间」可言 ⇒ 散度**按定义为 0**，不是 NaN。
        #    返回 NaN 会污染 `loss_record.json` 的 history 与下游聚合（08-25 实测 5 个键全 NaN）。
        return {("" if n == "all" else f"{n}_") + "mode_spread": 0.0
                for n, sub in tg.strata() if bool(sub.any())}
    last = tg.mask.float().cumsum(dim=-1).argmax(dim=-1)
    end = pred.traj.gather(2, last.view(-1, 1, 1, 1).expand(-1, n_modes, 1, 2)).squeeze(2)
    d = torch.cdist(end, end)                                        # [n, M, M]
    off = ~torch.eye(n_modes, dtype=torch.bool, device=d.device)
    spread = d[:, off].mean(dim=1)
    out: Dict[str, float] = {}
    for name, sub in tg.strata():
        if not bool(sub.any()):
            continue
        pre = "" if name == "all" else f"{name}_"
        out[f"{pre}mode_spread"] = float(spread[sub].mean())
    return out


__all__ = ["MISS_THRESHOLD_M", "JointTargets", "JointWTALoss", "MarginFunnelMonitor",
           "build_joint_targets_batch", "joint_metrics", "mode_diversity", "mode_usage",
           "per_mode_errors"]


class EdgeRankLoss:
    """S5 的边监督项 —— 与 S4 **逐字同构**的 within-target RankNet + 噪声门 + 分层 macro-average。

    🔴 **为什么 S5 必须保留它**：端到端梯度对 `w` 的唯一诉求是「降低预测误差」，
    没有任何力量把它拴在反事实语义上。去掉这一项，`EdgeScorer` 在数学上就退化成一个
    普通 attention 门（它的输入 `(h_i, h_j, r_ij, g_ij)` 与 decoder 的 QK 没有本质区别）
    ⇒ R4-2 逐字要的 "learn an appropriate topology" 答不了，且 Q109 §6.2-8
    「检查 scorer 是否退化成普通 attention」必然朝退化的方向走。

    🔴 **λ 的定法（不调参）**：第一个有效 batch 上按量级标定 `λ = L_main / L_rank`，
    此后固定。Q109 §5 的 `λ_r` 是空槽；量级标定是其中唯一不需要扫参的填法，且第一个
    batch 由 seed 决定 ⇒ 可复现。标定值写进 sidecar。

    🔴 **`scorable` / `valid` 在这里重算**（不重跑 `build_edge_context`，省一次几何前向）。
    两行定义与 :func:`magnet_joint_layers.build_edge_context` 逐字相同，由
    `tests/test_s5_edge_rank_loss.py` 的 [S3] 判据锁住 —— 那边改了这里不跟着改就会静默错。
    """

    def __init__(self, seed: int = 0, device=None) -> None:
        self.lam: Optional[float] = None
        self.gen = torch.Generator(device=device) if device is not None else torch.Generator()
        self.gen.manual_seed(seed)
        self.n_pairs_raw = 0
        self.n_pairs_kept = 0

    def __call__(self, edge_logit, data, idx, main_loss):
        """:returns: `(lam * L_rank, 诊断 dict)`；无可用对时返回 `(0-tensor, {...})`。"""
        import sys as _sys
        from pathlib import Path as _P
        _t = str(_P(__file__).resolve().parent)
        if _t not in _sys.path:
            _sys.path.insert(0, _t)
        from magnet_edge_label_dataset import STRATA                      # noqa: WPS433
        from train_edge_scorer_s4 import _pair_within_target, _ranknet_terms  # noqa: WPS433

        zero = torch.zeros((), device=edge_logit.device, dtype=edge_logit.dtype)
        U = getattr(data, "s4_U", None)
        if U is None:
            raise ValueError("S5 需要 `EdgeLabelGraphDataset` 挂的 s4_U —— 当前 batch 没有")
        src, dst = data.gnn_edge_index[0], data.gnn_edge_index[1]
        # 🔴 与 build_edge_context 逐字相同的两行（[S3] 判据锁）
        scorable = idx.entity_is_agent[src] & idx.entity_is_agent[dst]
        valid = idx.entity_valid[src] & idx.entity_valid[dst]
        m = (~torch.isnan(U)) & scorable & valid
        if int(m.sum()) < 2:
            return zero, {"rank_loss": 0.0, "rank_lam": self.lam or 0.0, "rank_pairs": 0.0}
        sel = torch.nonzero(m, as_tuple=True)[0]
        a, b = _pair_within_target(dst[sel], self.gen)
        a, b = sel[a], sel[b]
        self.n_pairs_raw += int(a.numel())
        loss_pp, _, keep = _ranknet_terms(edge_logit, U, data.s4_Usd, a, b)
        self.n_pairs_kept += int(keep.sum())
        if loss_pp.numel() == 0:
            return zero, {"rank_loss": 0.0, "rank_lam": self.lam or 0.0, "rank_pairs": 0.0}
        strat_k = data.s4_stratum[a][keep]
        terms = [loss_pp[strat_k == k].mean() for k in range(len(STRATA))
                 if bool((strat_k == k).any())]
        rank = torch.stack(terms).mean()
        if self.lam is None:                      # 量级标定，只做一次
            self.lam = float(main_loss.detach().abs() / rank.detach().clamp_min(1e-8))
        return self.lam * rank, {"rank_loss": float(rank.detach()), "rank_lam": self.lam,
                                 "rank_pairs": float(loss_pp.numel())}


__all__.append("EdgeRankLoss")


def run_epoch_joint(model: nn.Module, loader, criterion: "JointWTALoss", optimizer,
                    device: torch.device, train: bool,
                    monitor: Optional["MarginFunnelMonitor"] = None,
                    ego_only: bool = False,
                    edge_rank: Optional["EdgeRankLoss"] = None) -> Dict[str, float]:
    """跑一个 joint epoch。返回 `{loss, reg, mode_ce, <joint_metrics 的全部键>, <mode_usage>}`。

    :param ego_only: 见 :func:`build_joint_targets_batch` —— Q110 §6.2 的 A 臂。

    🔴 **聚合按 `n_target` 加权**（不是按 batch 数）—— 每批的监督实体数差很多
    （实测 4 样本批 = ego 4 + 周车 86，而 5.3% 的场景一个 agent 都没有），
    按批平均会让小场景与大场景等权，指标随 batch 组成漂。
    """
    from magnet_joint_encoder import build_entity_index                # noqa: WPS433 避免成环

    model.train(train)
    acc: Dict[str, float] = {}
    wacc: Dict[str, float] = {}
    n_tot = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for data in loader:
            data = data.to(device)
            idx = build_entity_index(data)
            tg = build_joint_targets_batch(data, idx, ego_only=ego_only)
            # E1：`oracle_gate` 是 `OracleGateGraphDataset` 挂的逐边门（`[E]`，未打标边恒 1.0）。
            # 🔴 字段不存在时传 None ⇒ 与改动前**逐位等价**（`edge_gate=None` 是模型默认值）。
            pred = model(data, edge_gate=getattr(data, "oracle_gate", None))
            loss, comp = criterion(pred, tg)
            if comp["n_target"] == 0:
                continue
            # S5：边监督项。🔴 `edge_rank` 为 None（固定臂 / 未开启）时**整段跳过**
            #    ⇒ 与改动前逐位等价。`pred.edge_logit` 在固定臂上恒为 None。
            rank_diag: Dict[str, float] = {}
            if edge_rank is not None and pred.edge_logit is not None:
                r_term, rank_diag = edge_rank(pred.edge_logit, data, idx, loss)
                loss = loss + r_term
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            w = float(comp["n_target"])
            row = {"loss": comp["total"], "reg": comp["reg"], "mode_ce": comp["mode_ce"]}
            row.update(rank_diag)
            row.update(joint_metrics(pred, tg))
            row.update(mode_usage(pred, tg))
            row.update(mode_diversity(pred, tg))
            # 🔴 逐层的读数必须按**该层的实体数**加权，不能按 batch 的 n_target ——
            #    否则 `ego_minADE` 会被「这批周车多不多」左右（ego 每批恒 = 场景数）。
            for k, v in row.items():
                if k.endswith("_n") or k in ("n_target", "miss_threshold_m"):
                    continue
                pre = k.split("_", 1)[0]
                wk = float(row.get(f"{pre}_n", w)) if f"{pre}_n" in row else w
                acc[k] = acc.get(k, 0.0) + float(v) * wk
                wacc[k] = wacc.get(k, 0.0) + wk
            n_tot += w
    out = {k: v / max(wacc.get(k, n_tot), 1.0) for k, v in acc.items()}
    out["n_target"] = n_tot
    out["miss_threshold_m"] = MISS_THRESHOLD_M
    if monitor is not None:
        out.update(monitor.summary())
        monitor.reset()
    return out


__all__.append("run_epoch_joint")
