"""闸⑥ —— agent 预测辅助监督（Q70 / 阶段75）。

**这个模块要回答什么**

C1 测出「删光全部 agent 只让轨迹移动 0.72%」。阶段73 §6 给的机制假说是**梯度饥饿**：

    agent 节点唯一的梯度来源 = ego 轨迹的 ADE/FDE loss
      ↓ 经 global_mean_pool（agent 在池化里权重仅 4.65%）
      ↓ 经 feature_fusion 的 1/3
    到达 agent 节点的梯度 ≈ 全强度的 1.5%

闸⑥ 给每个 agent 节点接一个**直接、不经池化、全强度**的监督（预测该 agent 的未来轨迹），
再复测使用度。**判据事前写死于主控表 §7「闸⑥ 的预注册判据」，赢输都能写。**

**三处必须小心的地方（都是本项目付过学费的坑）**

1. 🔴🔴 **节点 → AGENT_FUTURES 槽位不是恒等映射**（阶段75 实测）。
   图里的 agent 节点是**删掉静止车之后**剩下的（Q32：前置删除 72.9%），
   而 `AGENT_FUTURES` 恒为 20 槽且全部非零。实测一例：

       图 agent sid=1 → 槽 3 ｜ sid=2 → 槽 4 ｜ sid=3 → 槽 19

   ⇒ **若按「第 i 个 agent 节点 = 第 i 槽」接线，100% 的监督信号都是错的**，
   而错掉的结果会伪装成「闸⑥ 判据不成立 ⇒ 瓶颈不在监督」这一漂亮结论。
   本模块用**末帧位置精确 join** 恢复映射：60 样本 / 173 个 agent 节点
   **100% 唯一匹配**（最近距离 max **0.0000 m**、次近距离 min **2.03 m**）。

2. 🔴 **不得顺手打开逐样本 GNN 池化**（Q45 的混杂原样重演）。
   L1 基线 recipe 是 `TransformerPredictor` + `gnn_pooling_fix=false`
   （见 `experiments/scaling_ladder/s4b_magnet_L1/seed42/models/train_meta.json`）。
   aux 头需要逐节点嵌入与 node→sample 归属，但**主分支的池化语义必须逐位保持父类行为**，
   否则 `+aux(真)` vs `基线` 差的是两件事，归因无从谈起。
   本模块的做法：aux 路径**另算一遍节点嵌入**（复用同一套 `gnn_encoder` 权重），
   **完全不碰** `forward()` 里主分支那条路径。

3. 🔴 **必须配 `+aux(打乱)` 对照**（保容量与梯度总量、只破坏信号），
   否则「提升来自参数量还是辅助任务」无法回答。打乱 = **batch 内跨样本重新配对**
   agent 未来（见 `AgentAuxLoss.shuffle`）。

**权重兼容性**：`agent_future_head` 是新增子模块 ⇒ state_dict 多出若干键。
旧 checkpoint 用 `strict=True` 加载本类会失败，这是**有意的**（与 `build_model` 既有约定一致：
架构猜错就抛错，绝不静默建成部分权重）。消费侧须由 sidecar `train_meta.json` 的
`architecture.agent_aux` 决定建哪个类。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# --- sys.path 自包含（沿用 magnet_imitation_pooled.py 的隔离模板）---
# ⚠️ 必须把 devkit 的 `models/` 目录本身加进 path 再 `from transformer_predictor import`，
#    不能写 `from models.transformer_predictor import` —— GNN_Trans 下有同名 `models` 包会抢先命中
#    （见记忆 `project_code_duplicate_map`：本仓多套同名副本，train/eval 用 `models/` 那份）。
_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
_MODELS = _DEVKIT / "models"
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_MODELS), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformer_predictor import TransformerPredictor  # noqa: E402  父类 models/（E0 生效副本）

logger = logging.getLogger(__name__)

#: `AGENT_FUTURES` 的形状约定 —— agent 槽 × 60 帧 × (x, y, heading)。
#: 🔴 **`AGENT_SLOTS` 只是旧档（20 槽）的记录值，不是运行时约束**：
#:    ① 它在本模块**从未参与任何计算**（只出现在 `__all__` 里）；
#:    ② 真正的节点→槽位映射走 `resolve_agent_slots()` 的**位置 join**，与槽数无关。
#:    D1（2026-08-06）把新档槽数改到 **64** 之后，这个字面值就不再等于实际槽数 ——
#:    要槽数请读 `AGENT_FUTURES.shape[0]`，**不要读本常量**。
AGENT_SLOTS, FUTURE_FRAMES, FUTURE_CH = 20, 60, 3

#: aux 头只预测 (x, y)，与 ego 输出头的 60×2 口径一致（heading 不监督）。
AUX_OUT_DIM = FUTURE_FRAMES * 2

#: 位置 join 的判据（阶段75 实测：最近 max 0.0000、次近 min 2.03）。
#: 放宽到 0.05 / 0.5 仍有 40× 余量；超出即抛错，**不静默降级**。
JOIN_MAX_DIST = 0.05
JOIN_MIN_MARGIN = 0.5


class AgentSlotJoinError(RuntimeError):
    """节点 → AGENT_FUTURES 槽位的映射无法唯一恢复 —— 必须暴露而不是猜。"""


def resolve_agent_slots(agent_end_xy: np.ndarray,
                        agent_futures_last_hist_xy: np.ndarray,
                        *, strict: bool = True) -> np.ndarray:
    """用末帧位置把「图里的 agent 节点」对回「`AGENT_FUTURES` 的槽位」。

    :param agent_end_xy: `[n_agent, 2]`，图中每个 agent 子图末节点的终点 (x, y)。
    :param agent_futures_last_hist_xy: `[20, 2]`，各槽末帧历史位置
        （即 `AGENT_HISTORIES[:, -1, :2]`，与建图所用的是同一批坐标）。
    :param strict: True 时匹配不唯一就抛 `AgentSlotJoinError`；False 时该节点记 -1（不参与 loss）。
    :returns: `[n_agent]` 的 int64，元素为槽位下标；-1 表示无法匹配。

    判据（事前写死，见模块 docstring 第 1 点）：最近距离 < 0.05 m **且** 次近 > 0.5 m。
    """
    n = agent_end_xy.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out

    # [n_agent, 20] 距离矩阵 —— 规模至多 20×20，直接算比建索引更快也更好查
    d = np.linalg.norm(agent_end_xy[:, None, :] - agent_futures_last_hist_xy[None, :, :], axis=-1)
    order = np.argsort(d, axis=1)
    nearest, second = order[:, 0], order[:, 1]
    d1 = d[np.arange(n), nearest]
    d2 = d[np.arange(n), second]

    good = (d1 < JOIN_MAX_DIST) & (d2 > JOIN_MIN_MARGIN)

    # 🔴 重复槽位必须在**两种模式下都**处理：strict 抛错、非 strict 也要把冲突的全部作废。
    #    早期版本只在 strict 下查重 ⇒ 非 strict 时两个节点可映射到同一槽、静默产出重复目标。
    cand = np.where(good, nearest, -1)
    for slot in np.unique(cand[cand >= 0]):
        hit = np.where(cand == slot)[0]
        if len(hit) > 1:
            good[hit] = False                      # 冲突的全部作废，不猜哪个对
            logger.warning("槽位 %d 被 %d 个 agent 节点同时命中 ⇒ 全部作废", int(slot), len(hit))
    out[good] = nearest[good]

    if strict and not good.all():
        bad = int((~good).sum())
        raise AgentSlotJoinError(
            f"{bad}/{n} 个 agent 节点无法唯一匹配到 AGENT_FUTURES 槽位："
            f"最近距离 max={float(d1.max()):.4f}（判据 <{JOIN_MAX_DIST}）、"
            f"次近距离 min={float(d2.min()):.4f}（判据 >{JOIN_MIN_MARGIN}）。"
            f"⚠️ 按下标顺序硬接会让 100% 的监督信号错位，故此处不降级。"
        )

    used = out[out >= 0]
    assert len(used) == len(set(used.tolist())), "去重后仍有重复槽位 —— 上面的冲突作废逻辑有问题"
    return out


class MagnetAgentAux(TransformerPredictor):
    """父类 + agent 未来预测辅助头（闸⑥）。

    🔴 **主分支逐位不变**：`forward()` 直接调父类，池化语义、输出头、权重形状全不动。
    aux 头只在 `agent_aux_forward()` 里被调用，由 trainer 显式驱动
    ⇒ 推理/闭环/干预审计路径**完全不受影响**（它们只调 `forward` / `get_encoder_features`）。
    """

    def __init__(self, agent_aux: bool = True, aux_hidden: int = 128) -> None:
        super().__init__()
        self.agent_aux = agent_aux
        if agent_aux:
            self.agent_future_head = nn.Sequential(
                nn.Linear(128, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, AUX_OUT_DIM),
            )
            with torch.no_grad():
                nn.init.normal_(self.agent_future_head[-1].weight, mean=0.0, std=0.1)
                nn.init.zeros_(self.agent_future_head[-1].bias)

    # ── 逐节点 GNN 嵌入（复用父类同一套权重，但不池化）────────────────────
    def _gnn_node_embeddings(self, data) -> Optional[torch.Tensor]:
        """复算一遍 GNN 的**节点级**输出 `[N_nodes, 128]`。

        为什么是"复算"而不是从 `forward` 里截：父类的 GNN 分支在 `GNNEncoder.forward`
        内部就把节点池化掉了，而 `transformer_predictor.py` 是 `devkit_patches`
        **受管文件**（改它须同批过 `sync_devkit_patches.py --check` 4/4）。
        这里直接复用 `self.gnn_encoder.conv1/conv2` 的**同一批权重**重算，
        ⇒ **零改动受管文件**，代价是 GNN 分支多算一次（节点中位数 43，可忽略）。

        ⚠️ `dropout` 是随机的 ⇒ aux 路径看到的是另一组 dropout mask。
        这不影响归因（两格同样处理），但复现时须固定 seed。
        """
        if not (hasattr(data, "gnn_x") and hasattr(data, "gnn_edge_index")):
            return None
        if data.gnn_x.numel() == 0 or data.gnn_edge_index.numel() == 0:
            return None

        x = self._extract_core_features(data.gnn_x)
        edge_index = data.gnn_edge_index
        # 与父类 GNNEncoder 同款的越界保护（父类会静默过滤，这里保持一致）
        max_node_idx = x.size(0) - 1
        if edge_index.numel() > 0 and int(edge_index.max()) > max_node_idx:
            valid = (edge_index[0] <= max_node_idx) & (edge_index[1] <= max_node_idx)
            if valid.sum() == 0:
                return None
            edge_index = edge_index[:, valid]

        h = F.relu(self.gnn_encoder.conv1(x, edge_index))
        h = self.gnn_encoder.dropout(h)
        return self.gnn_encoder.conv2(h, edge_index)          # [N_nodes, 128]

    def agent_aux_forward(self, data) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """对图中的 agent 节点预测未来轨迹。

        :returns: `(pred, node_index)`，`pred` 为 `[n_agent_nodes, 60, 2]`，
            `node_index` 为这些节点在 `gnn_x` 里的下标；无 agent 节点时返回 None。
        """
        if not self.agent_aux:
            return None
        node_emb = self._gnn_node_embeddings(data)
        if node_emb is None:
            return None
        is_agent = getattr(data, "gnn_is_agent", None)
        if is_agent is None:
            raise RuntimeError(
                "闸⑥ 需要 `gnn_is_agent`，但 batch 里没有 —— "
                "请确认 dataset 侧已按阶段75 的接线清单产出该字段（不得回退成"
                "「按下标猜 agent」，那会让监督信号 100% 错位）"
            )
        idx = torch.nonzero(is_agent.bool(), as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        pred = self.agent_future_head(node_emb[idx])          # [n_agent, 120]
        return pred.view(-1, FUTURE_FRAMES, 2), idx


def _shuffle_perm(n: int, device: torch.device,
                  generator: Optional[torch.Generator]) -> torch.Tensor:
    """生成打乱用的置换下标。

    🔴 **不能写成 `torch.randperm(n, generator=g, device=device)`** —— 当 `g` 是 CPU generator
    而 `device` 是 cuda 时，PyTorch 抛
    ``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``。
    这个 bug **只在 GPU 上、且只在「打乱」那一格触发**，CPU 单测抓不到
    （2026-08-04 code review 实测复现）⇒ 一律先在 CPU 上生成再搬。
    """
    return torch.randperm(n, generator=generator).to(device)


def masked_aux_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
                    *, shuffle: bool = False, weight: float = 1.0,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """闸⑥ 的**唯一** aux loss —— mask 是必传参数，不提供无 mask 的重载。

    为什么强制 mask：`AGENT_FUTURES` 的尾部是零填充（track 中断），实测
    **9.6% 的帧、21% 的 agent 槽**受影响（p90 45%、最坏 98.3%）。
    漏 mask 就是在教 aux 头预测 (0,0)，会把闸⑥ 推向"不升"档，
    再被读成「瓶颈不在监督」这一**错误结论**。
    ⚠️ 本函数**取代**了早期的 `AgentAuxLoss`（那个版本没有 mask 参数，已删除）。

    :param pred: `[n_agent, 60, 2]`；:param tgt: 同形；:param mask: `[n_agent, 60]` bool。
    :param shuffle: True 即 `+aux(打乱)` 对照格 —— 跨 agent 重排目标
        （**保容量与梯度总量、只破坏信号**）。`mask` 随 `tgt` 一同重排，否则有效帧模式会串位。

    ⚠️ **判读注意**：训练初期 aux 头输出 ≈0，此时 smooth-L1 退化成 `mean(|tgt|)`，
    **对置换不变** ⇒ 「真」与「打乱」两条 loss 曲线会重合。这是预期行为不是接线错误；
    **两格分叉的时刻本身就是 aux 头开始学到 agent-specific 信息的证据**。
    若训到最后仍重合 ⇒ aux 头没学到东西，主判据无论结果如何都不可解读。
    """
    if pred.numel() == 0 or int(mask.sum()) == 0:
        return pred.sum() * 0.0
    if shuffle and pred.size(0) > 1:
        perm = _shuffle_perm(pred.size(0), tgt.device, generator)
        tgt, mask = tgt[perm], mask[perm]
    m = mask.unsqueeze(-1).expand_as(tgt)
    return weight * F.smooth_l1_loss(pred[m], tgt[m])


__all__ = [
    "AGENT_SLOTS",
    "AUX_OUT_DIM",
    "AgentSlotJoinError",
    "MagnetAgentAux",
    "masked_aux_loss",
    "resolve_agent_slots",
]
