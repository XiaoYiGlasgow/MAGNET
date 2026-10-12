"""路线 B 新结构 S1 第 2 件 —— **relation-aware message passing + edge scorer**（Q110 §3.3 / §3.4）。

## 它取代了什么

旧结构的交互通路是**完全图上的 `GCNConv`**，而 D-1 已解析 + 数值双证它**精确退化为 `MLP(mean)`**
（max|Δ| **2.608e-08**）—— 图结构零作用，每个节点收到的是同一个「全节点均值」。
本模块用**依赖 `(i, j)` 对的消息**取代它：消息内容由 `h_j`、显式几何 `g_ij`、关系类型 `r_ij` 共同决定，
完全图因此**不再**让所有节点收到同一个聚合。

这一步同时解决两件审稿人点名的事：
* **R4-4**（图里画了多类语义关系，公式和代码却是一个共享 GCN）—— 现在图、公式、代码三者一致；
* **D-1 的退化** —— 见上。

## 🔴 本模块必须守的四条硬约束

============  ==================================================================================
**P6 / D-b**  边**打分**只定义在**车辆↔车辆**边上；lane 保留为 token 且必须被 attend，但走固定权重。
              🔴 理由**只有「主张作用域」一条**：本文要卖的是「交通参与者之间的交互拓扑」，
              把 lane 并进同一个打分空间，这句话就退化成「我们学了一个通用注意力」。
              🔴 **代价**：存在二阶泄漏 `j → lane → decoder → i` ⇒ 正文只能写
              「验证**车辆↔车辆拓扑**在保留车辆的切边干预下的 faithfulness」。
**P8**        padding 实体**不进图**：任一端为 padding 的边，其 `w` 恒为 **0** ⇒ 消息恰好为 0；
              每层结束后 padding token 再次整行置零。V-11 的「max|Δ| = 0」靠这两条**逐位**成立。
**D-f**       decoder 与 MP **共用同一套 `w`**（`attn_logit += log(w+ε)`）。故本模块的 scorer
              **只前向一次**、`w` 由 :meth:`JointInteractionStack.forward` 一并返回给 decoder。
              用第二个打分头会让论文里出现**两个拓扑**，"learned topology" 指哪个说不清。
**P3**        一律 `Linear → LayerNorm → ReLU`（前置归一化）。阶段299 的塌陷下游一半就出在缺这个。
============  ==================================================================================

## 🔴 几何特征 `g_ij` 是**显式构造**的，不是学出来的（Q110 §3.3）

逐实体的参考位姿取法（**不得用 heading 的算术平均** —— D-3 实测 4.27% 的子图会出现 >π 跳变，
wrap 内 median **75.8°**、5 例近反向 max **170.5°**）：

* **agent**：取该实体**最后一个有效向量**的终点 / 终点朝向 / `v_end` —— 即「当前时刻」的位姿与速度；
* **lane**：位置取有效向量终点的**质心**，朝向取**圆周均值** `atan2(Σsin, Σcos)`，速度恒 0。

## 🔴 固定拓扑臂也实例化 scorer（Q110 §4-1）

`learn_topology=False` 时 `w ≡ 1`，但 scorer **仍被实例化**，以保证所有 topology 臂
**参数量逐位相同**（Q109 §6.1 要求同容量/同模态数/同预算）。它不接收梯度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from magnet_joint_encoder import EntityIndex

#: 几何特征 `g_ij` 的维度（见 :func:`build_edge_context` 的列表）。
G_DIM: int = 10

#: 关系类型数：veh→veh / veh→lane / lane→veh / lane→lane。
N_RELATION: int = 4
REL_VEH_VEH, REL_VEH_LANE, REL_LANE_VEH, REL_LANE_LANE = 0, 1, 2, 3
REL_NAMES = ("veh->veh", "veh->lane", "lane->veh", "lane->lane")

#: D-f 的 ε：`attn_logit += log(w + EPS_GATE)`。**decoder 必须用同一个常量**，勿各写各的。
EPS_GATE: float = 1e-6

#: 归一化尺度。位置用 `map_radius`（§1.1 现值 50 m），速度用 10 m/s，二者都是**常量不是超参**，
#: 改它会改变输入分布 ⇒ 改了必须在 §1 参数台账同步登记。
POS_SCALE: float = 50.0
VEL_SCALE: float = 10.0
#: 邻近核 `exp(-d / PROX_SCALE)` 的尺度：10 m ≈ 车辆交互的主要作用距离。
PROX_SCALE: float = 10.0

_EPS = 1e-6


@dataclass(frozen=True)
class EntityGeometry:
    """逐实体的参考位姿（显式构造，无参数）。

    :ivar pos: `[n_entity, 2]` ego-centric 位置（agent = 当前位置，lane = 质心）。
    :ivar heading: `[n_entity]` 朝向 (rad)（agent = 当前朝向，lane = **圆周均值**）。
    :ivar vel: `[n_entity, 2]` 速度向量（lane 恒 0）。
    :ivar on_route: `[n_entity]` lane 的 `is_on_route ∈ {0,1}`；agent 恒 0。
    """

    pos: torch.Tensor
    heading: torch.Tensor
    vel: torch.Tensor
    on_route: torch.Tensor


@dataclass(frozen=True)
class EdgeContext:
    """逐边的显式上下文（无参数；建一次可跨层复用）。

    :ivar edge_index: `[2, E]`，`edge_index[0] = j`（源），`edge_index[1] = i`（目标）。
    :ivar rel_type: `[E]` long ∈ [0, 4)，见 :data:`REL_NAMES`。
    :ivar g: `[E, G_DIM]` 几何特征。
    :ivar scorable: `[E]` bool —— 车辆↔车辆边（P6/D-b：只有它们参与打分）。
    :ivar valid: `[E]` bool —— 两端都非 padding（P8）。
    """

    edge_index: torch.Tensor
    rel_type: torch.Tensor
    g: torch.Tensor
    scorable: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def build_entity_geometry(x: torch.Tensor, idx: EntityIndex) -> EntityGeometry:
    """从**未池化**的 `data.x` 造逐实体参考位姿。

    :param x: `[n_vec, 12]`，列语义见 `docs/feature_schema_v2.md`
        （0-1 起点 xy / 2-3 终点 xy / 4-5 起终朝向 / 8 = agent 的 `v_end` 或 lane 的 `is_on_route`）。
    :param idx: :func:`magnet_joint_encoder.build_entity_index` 的产物。
    """
    device, n_ent = x.device, idx.n_entity
    valid_f = idx.vec_valid.to(x.dtype)

    # ---- 每个实体的「最后一个有效向量」的行号（agent 的当前时刻）----------------------
    row = torch.arange(x.shape[0], device=device)
    scored = torch.where(idx.vec_valid, row, torch.full_like(row, -1))
    last = torch.full((n_ent,), -1, dtype=torch.long, device=device)
    last = last.index_reduce_(0, idx.global_eid, scored, "amax", include_self=True)
    last = last.clamp(min=0)                      # 全 padding 的实体取 0；它随后被 entity_valid 抹掉

    # ---- lane 侧：终点质心 + 朝向圆周均值（🔴 绝不用 heading 的算术平均，见 D-3）--------
    cnt = torch.zeros(n_ent, device=device).index_add_(0, idx.global_eid, valid_f).clamp(min=1.0)
    pos_mean = torch.zeros(n_ent, 2, device=device).index_add_(
        0, idx.global_eid, x[:, 2:4] * valid_f.unsqueeze(-1)) / cnt.unsqueeze(-1)
    sin_sum = torch.zeros(n_ent, device=device).index_add_(
        0, idx.global_eid, torch.sin(x[:, 5]) * valid_f)
    cos_sum = torch.zeros(n_ent, device=device).index_add_(
        0, idx.global_eid, torch.cos(x[:, 5]) * valid_f)
    head_circ = torch.atan2(sin_sum, cos_sum)

    is_agent = idx.entity_is_agent
    pos = torch.where(is_agent.unsqueeze(-1), x[last, 2:4], pos_mean)
    heading = torch.where(is_agent, x[last, 5], head_circ)
    # 🔴 col8 的语义按 node type 分岔：agent 行 = v_end，lane 行 = is_on_route（feature_schema_v2 §1）
    speed = torch.where(is_agent, x[last, 8], torch.zeros_like(head_circ))
    vel = speed.unsqueeze(-1) * torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    on_route = torch.where(is_agent, torch.zeros_like(head_circ), x[last, 8])

    keep = idx.entity_valid.unsqueeze(-1).to(x.dtype)
    return EntityGeometry(pos=pos * keep, heading=heading * idx.entity_valid.to(x.dtype),
                          vel=vel * keep, on_route=on_route * idx.entity_valid.to(x.dtype))


@torch.no_grad()
def build_edge_context(edge_index: torch.Tensor, idx: EntityIndex,
                       geom: EntityGeometry) -> EdgeContext:
    """造逐边的关系类型、几何特征、可打分位、有效位。

    `g_ij` 的十列（**全部旋进目标 i 的朝向系**，故对整个场景的刚体旋转不变）：

    ===  ==========================  ==================================================
    列   含义                        构造
    ===  ==========================  ==================================================
    0-1  j 相对 i 的位移 (dx, dy)    旋进 i 的朝向系，除以 `POS_SCALE`
    2    距离                        `‖d‖ / POS_SCALE`
    3    邻近核                      `exp(-‖d‖ / PROX_SCALE)`
    4-5  相对朝向                    `cos Δh`, `sin Δh`（🔴 不用角度差，避开 wrap）
    6-7  相对速度 (dvx, dvy)         同样旋进 i 的朝向系，除以 `VEL_SCALE`
    8    接近速度                    `-(d·dv)/‖d‖ / VEL_SCALE`，**正 = 正在靠近**
    9    route 位                    j 的 `is_on_route`（agent 恒 0）
    ===  ==========================  ==================================================
    """
    src, dst = edge_index[0], edge_index[1]        # src = j（源），dst = i（目标）
    rel = geom.pos[src] - geom.pos[dst]
    c, s = torch.cos(geom.heading[dst]), torch.sin(geom.heading[dst])
    dx = c * rel[:, 0] + s * rel[:, 1]
    dy = -s * rel[:, 0] + c * rel[:, 1]
    dist = rel.norm(dim=-1)

    dvel = geom.vel[src] - geom.vel[dst]
    dvx = c * dvel[:, 0] + s * dvel[:, 1]
    dvy = -s * dvel[:, 0] + c * dvel[:, 1]
    closing = -(rel * dvel).sum(dim=-1) / (dist + _EPS)

    dh = geom.heading[src] - geom.heading[dst]
    g = torch.stack([dx / POS_SCALE, dy / POS_SCALE, dist / POS_SCALE,
                     torch.exp(-dist / PROX_SCALE), torch.cos(dh), torch.sin(dh),
                     dvx / VEL_SCALE, dvy / VEL_SCALE, closing / VEL_SCALE,
                     geom.on_route[src]], dim=-1)

    a_src, a_dst = idx.entity_is_agent[src], idx.entity_is_agent[dst]
    rel_type = torch.where(
        a_src & a_dst, torch.full_like(src, REL_VEH_VEH),
        torch.where(a_src & ~a_dst, torch.full_like(src, REL_VEH_LANE),
                    torch.where(~a_src & a_dst, torch.full_like(src, REL_LANE_VEH),
                                torch.full_like(src, REL_LANE_LANE))))

    valid = idx.entity_valid[src] & idx.entity_valid[dst]
    return EdgeContext(edge_index=edge_index, rel_type=rel_type, g=g * valid.unsqueeze(-1).to(g.dtype),
                       scorable=a_src & a_dst, valid=valid)


class _GeometryEncoder(nn.Module):
    """`g_ij` 的共享小编码器（P3：`Linear → LayerNorm → ReLU`）。"""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.LayerNorm(G_DIM), nn.Linear(G_DIM, dim),
                                 nn.LayerNorm(dim), nn.ReLU())

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class RelationMessagePassing(nn.Module):
    """一层 relation-aware message passing（Q110 §3.3 / Q109 §3.6 方案 A）。

    .. code-block:: text

        m_{j→i} = w_{j→i} · φ_{r_ij}(h_j, g_ij)      每个 relation type 一套小变换
        h_i'    = h_i + ψ(h_i, Σ_j m_{j→i})           ψ 内部按 P3 前置归一化

    🔴 `w` 由外部传入（scorer 只跑一次、与 decoder 共用，见 D-f），本层**不自己打分**。
    """

    def __init__(self, dim: int = 128, g_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        # φ：每个 relation type 一个线性变换（Q110 §3.3「每个 relation type 一套小变换」），
        # 共享其后的归一化与激活 —— 参数少、且让「换 r_ij 必须改变消息」这件事可测（V-6）。
        self.phi = nn.ModuleList([nn.Linear(dim + g_dim, dim) for _ in range(N_RELATION)])
        self.msg_norm = nn.Sequential(nn.LayerNorm(dim), nn.ReLU())
        self.update = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, ectx: EdgeContext, g_enc: torch.Tensor,
                w: torch.Tensor, entity_valid: torch.Tensor) -> torch.Tensor:
        """:param h: `[n_entity, dim]`；:param w: `[E]` 边权（已含 P8 与干预门）。"""
        src, dst = ectx.edge_index[0], ectx.edge_index[1]
        inp = torch.cat([h[src], g_enc], dim=-1)

        msg = torch.zeros(inp.shape[0], self.dim, dtype=h.dtype, device=h.device)
        for r in range(N_RELATION):                # 只有 4 类，循环比 gather 4 份省 4×算力
            sel = torch.nonzero(ectx.rel_type == r, as_tuple=True)[0]
            if sel.numel():
                msg = msg.index_copy(0, sel, self.phi[r](inp.index_select(0, sel)))
        msg = self.msg_norm(msg) * w.unsqueeze(-1)  # 🔴 w=0 ⇒ 消息**恰好** 0（V-11 逐位成立）

        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        h = self.out_norm(h + self.update(torch.cat([h, agg], dim=-1)))
        return h * entity_valid.unsqueeze(-1).to(h.dtype)      # P8：padding token 再次归零


def conserve_target_mass(w: torch.Tensor, ectx: EdgeContext, n_entity: int,
                         active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """逐目标把 scorable 边的权重和归一到 `d_i`（该目标的 scorable ∧ valid 入边数）。

    .. code-block:: text

        w'_{j→i} = w_{j→i} · d_i / Σ_{j'} w_{j'→i}        只在 scorable ∧ valid 的边上

    **为什么需要这一步**（三条，写在这里免得日后被当成可有可无的装饰）：

    1. MP 层是**无归一化**的加权求和（:meth:`RelationMessagePassing.forward` 的 `index_add_`）
       ⇒ 不守恒时 `w` 的整体水平直接改变交互通道增益，「换权重形状」与「把通道调弱」
       两个变量会混在一起，learned vs fixed 的差异不可归因；
    2. decoder 的 `log(w+ε)` 里 lane 边被 P6/D-b 钉死 `w≡1` 是锚 ⇒ 车↔车边整体偏低会
       **系统性地把注意力从车推向 lane**；
    3. 逐目标 min-max 那类映射强制每个目标必有一条 1.0、一条 floor ⇒ 在 87.6% 的
       static/truncated 目标上（`U` p95 0.0077 ≈ teacher 分歧 0.00228）把噪声展开成满量程。

    🔑 守恒归一的失败模式是**良态**的：scorer 无信息（`w` 组内近似均匀）时逐位退化为
    `w ≡ 1`，即恰好复现固定臂。它也是 Q109 §6.1「可比的激活边预算」的正确实现 ——
    锚是从固定臂**量出来的** `d_i`，不预设任何稀疏目标（与 §5.5 / §12.2 的稀疏性禁令不冲突）。

    🔴🔴 **必须在所有 mask（`valid`、干预 `edge_gate`）之后调用**。放在前面会让被切掉的边
    的质量仍留在分母里 ⇒ ① 改被孤立实体 j 的特征会经分母泄漏到 i≠j（V-2 实测 max|Δ| 5.3e-03）；
    ② 「切边门」与「重建 `edge_index`」不再等价（实测 max|Δ| 1.8e-02）。两者都是干预审计的地基。

    :param w: `[E]` 边权（已乘过 valid 与干预门）。
    :param n_entity: 实体总数（批内全局行号的上界），用于 `index_add_` 的目标维。
    :param active: `[E]` bool，哪些边**算进预算**。默认 `scorable ∧ valid`；
        有干预门时调用方须再 `& (gate > 0)`，否则被切的边仍占分母。
    """
    m = (active if active is not None else (ectx.scorable & ectx.valid)).to(w.dtype)
    dst = ectx.edge_index[1]
    zeros = torch.zeros(n_entity, dtype=w.dtype, device=w.device)
    tot = zeros.index_add(0, dst, w * m)          # 每个目标的 scorable 权重和
    cnt = zeros.index_add(0, dst, m)              # 每个目标的 scorable 边数 d_i
    scale = cnt / tot.clamp_min(_EPS)
    return torch.where(m.bool(), w * scale[dst], w)


class EdgeScorer(nn.Module):
    """反事实边效用的运行期打分头（Q110 §3.4）。

    .. code-block:: text

        w_{j→i} = σ( f_θ(h_i, h_j, r_ij, g_ij) )     MLP: (2·dim + r_dim + g_dim) → 64 → 1

    * 推理时**只做一次前向**：不读未来、不删车、不重跑 planner；
    * 第一版用**连续软边权**，不用 Gumbel-Softmax（Q109 §3.5 / §12.2）；
    * 🔴 **P6/D-b**：只在车辆↔车辆边上打分，其余边 `w ≡ 1`。
    """

    def __init__(self, dim: int = 128, g_dim: int = 32, r_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(N_RELATION, r_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * dim + r_dim + g_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def raw_logit(self, h: torch.Tensor, ectx: EdgeContext, g_enc: torch.Tensor) -> torch.Tensor:
        """pre-sigmoid 分数 `s_{j→i}`，`[E]`。

        🔴 S4 的排序损失算在**这一层**，不算在 :meth:`forward` 的输出上：
        ① `σ` 有界，hinge/logistic 在饱和区梯度消失两次；
        ② `forward` 里的逐目标归一会把「同一目标内的整体平移」吸收掉，排序梯度会被它扭曲。
        """
        src, dst = ectx.edge_index[0], ectx.edge_index[1]
        feat = torch.cat([h[dst], h[src], self.rel_emb(ectx.rel_type), g_enc], dim=-1)
        return self.net(feat).squeeze(-1)

    def forward(self, h: torch.Tensor, ectx: EdgeContext, g_enc: torch.Tensor) -> torch.Tensor:
        """🔴 返回的是**未归一**的 `σ(·) ∈ (0,1)`。逐目标质量守恒归一由
        :meth:`JointInteractionStack.forward` 在 `valid` 与干预门之后统一做 ——
        放在这里会破坏干预口径（见 :func:`conserve_target_mass` 的红字）。"""
        w = torch.sigmoid(self.raw_logit(h, ectx, g_enc))
        # 🔴 P6/D-b：lane 参与的边不打分，恒 1（lane 必须被 attend，只是不受学习拓扑管辖）
        return torch.where(ectx.scorable, w, torch.ones_like(w))


@dataclass(frozen=True)
class InteractionOutput:
    """:ivar tokens: `[n_entity, dim]` 关系增强后的实体 token（**没有** global pooling）。
    :ivar edge_weight: `[E]` 最终边权 —— 🔴 **decoder 必须复用这一份**（D-f），不得另打一套分。
    :ivar ectx: 建好的边上下文，供 decoder 与干预脚本复用。
    :ivar edge_logit: `[E]` scorer 的 **pre-sigmoid** 分数，`None` = 未请求或固定臂。
        🔴 S4/S5 的排序损失算在**这一层**（与 `train_edge_scorer_s4.py` 同一个量）：
        `forward` 输出的 `w` 经过逐目标归一，同目标内虽与 logit 单调等价、排序无差，
        但梯度会跨边耦合；算在 logit 上才与 S4 的优化问题逐字一致。
    """

    tokens: torch.Tensor
    edge_weight: torch.Tensor
    ectx: EdgeContext
    edge_logit: Optional[torch.Tensor] = None


class JointInteractionStack(nn.Module):
    """2 层 relation-aware MP + **单个** edge scorer（Q110 §3.3–§3.4）。

    层数 **2**：完全图上 1 层已全连通，2 层给二阶关系；再多会加参数且难归因。

    🔴 scorer 只跑一次、从**编码器输出的 token** `h⁰` 打分，两层 MP 与 decoder **共用同一份 `w`**。
    这不是实现上的偷懒，是 D-f 的要求：两套 `w` 会让 "learned topology" 指哪个说不清。
    """

    def __init__(self, dim: int = 128, n_layers: int = 2, g_hidden: int = 32,
                 r_dim: int = 16, dropout: float = 0.1, learn_topology: bool = True) -> None:
        super().__init__()
        self.dim, self.learn_topology = dim, learn_topology
        self.geo_enc = _GeometryEncoder(g_hidden)
        # 🔴 固定拓扑臂也实例化 scorer，保证各 topology 臂参数量逐位相同（Q110 §4-1）；它不收梯度。
        self.scorer = EdgeScorer(dim=dim, g_dim=g_hidden, r_dim=r_dim)
        self.layers = nn.ModuleList(
            [RelationMessagePassing(dim=dim, g_dim=g_hidden, dropout=dropout)
             for _ in range(n_layers)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """与 `magnet_joint_encoder.EntityEncoder` 同一套（xavier + zero bias）。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, h: torch.Tensor, ectx: EdgeContext, idx: EntityIndex,
                edge_gate: Optional[torch.Tensor] = None) -> InteractionOutput:
        """:param h: `[n_entity, dim]` 编码器输出的实体 token。
        :param edge_gate: `[E]` 可选的**干预门** —— 切边/隔离实验把要切的边置 0
            （与 `g1_dose_ladder_probe._isolate` 等价，只是不重建 `edge_index`）。
        :returns: :class:`InteractionOutput`；`edge_weight` 供 decoder 按 D-f 用。
        """
        g_enc = self.geo_enc(ectx.g)
        e_logit = None
        if self.learn_topology:
            e_logit = self.scorer.raw_logit(h, ectx, g_enc)
            w = torch.where(ectx.scorable, torch.sigmoid(e_logit),
                            torch.ones_like(e_logit))          # == scorer.forward，见 N5
        else:
            w = torch.ones(ectx.edge_index.shape[1], dtype=h.dtype, device=h.device)
        w = w * ectx.valid.to(w.dtype)             # P8：任一端 padding ⇒ 恒 0
        if edge_gate is not None:
            w = w * edge_gate.to(w.dtype)
        if self.learn_topology:
            # 🔴 顺序不可换：归一必须在 valid 与干预门**之后**，且被切的边不得留在分母里，
            #    否则 V-2 实体隔离与「门 ≡ 重建 edge_index」两条干预地基同时失效。
            active = ectx.scorable & ectx.valid
            if edge_gate is not None:
                active = active & (edge_gate.to(w.dtype) > 0)
            w = conserve_target_mass(w, ectx, h.shape[0], active=active)

        for layer in self.layers:
            h = layer(h, ectx, g_enc, w, idx.entity_valid)
        return InteractionOutput(tokens=h, edge_weight=w, ectx=ectx, edge_logit=e_logit)


def isolate_edge_gate(ectx: EdgeContext, nodes: torch.Tensor) -> torch.Tensor:
    """造一个把给定实体**完全孤立**的 `edge_gate`（两个方向的边全切）。

    🔴 与 `scripts/analysis/g1_dose_ladder_probe._isolate` **语义逐字对应**（切它的全部边、节点保留），
    区别只是这里用门而不是重建 `edge_index` —— 前者可微、可批量，后者是审计脚本的既有口径。
    ⚠️ 单靠它**不足以**做 V-2 的实体隔离：decoder 对该 token 的注意力也必须置零（D-b 的二阶泄漏）。
    """
    src, dst = ectx.edge_index[0], ectx.edge_index[1]
    hit = torch.isin(src, nodes) | torch.isin(dst, nodes)
    return (~hit).to(torch.float32)


__all__ = ["G_DIM", "N_RELATION", "REL_NAMES", "REL_VEH_VEH", "REL_VEH_LANE",
           "REL_LANE_VEH", "REL_LANE_LANE", "EPS_GATE", "POS_SCALE", "VEL_SCALE", "PROX_SCALE",
           "EntityGeometry", "EdgeContext", "InteractionOutput",
           "RelationMessagePassing", "EdgeScorer", "JointInteractionStack",
           "build_entity_geometry", "build_edge_context", "isolate_edge_gate",
           "conserve_target_mass"]
