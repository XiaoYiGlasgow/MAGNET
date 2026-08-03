"""D2 feasibility：Controlled Semantic Interaction-Reliance Audit 的干预与对照原语。

阶段⑳ §7.1 的 minimum D2 feasibility bundle 需要四件东西，本模块提供其中三件（推理端，
不需要重新训练）：

    2. G7-core：semantic intervention
    3. edge-count-matched random control
    4. replacement/resampling control

（`1. G6 capacity-matched GCN` 需要另训模型，不在本模块。）

primary estimand（§7.1）::

    Semantic Specificity = response(semantic intervention) - response(matched control)

若 controls 之后效应消失 ⇒ audit 立即降为 case study / negative result，
**不扩 sparsity sweep，不写进 Contributions**。

设计前提（都是从代码里读出来的事实，不是假设）
------------------------------------------------
* 训练用 `dataset_config="default"` ⇒ `DefaultBuilder` 建的是**全连接图**，
  `edge_types` 恒为 0。E0 决策 B 已删 typed-edge、只留普通 GCN。
  ⇒ **"语义"不能由边类型定义，只能由两个端点的节点类型定义**（本模块即据此分类）。
* `build_graph_data_from_feature_df` 里 GNN 分支的边是**无向补反向**存的：
  `edge_index = [directed(E) | reversed(E)]`，第 `i` 与第 `i+E` 条是同一条无向边。
  ⇒ **任何干预都必须成对增删**，否则邻接矩阵不再对称、破坏 GCN 的语义。
* 节点顺序 = `gnn_graph['nodes']`，其类型来自 `SUBGRAPH_ID_TO_TYPE`（`'ego'/'agent'/'lane'`）。

口径约束
--------
* 本模块是**推理端干预**：同一个 checkpoint、同一个 scene、同一个 seed、同一个 intervention
  budget，只改图的边集。不涉及重训。
* response 默认用**不依赖 GT 的输出偏移**（模型对干预的敏感度），同时附带 ΔADE 供参考 ——
  因为 audit 问的是「模型是否依赖该语义结构」，不是「干预后精度是否变差」。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: 语义类别 = 无向边两端节点类型的无序对。'ego' 并入 'agent'（E0 契约里 ego 也是车辆节点）。
SEMANTIC_CATEGORIES = ("agent-agent", "agent-lane", "lane-lane")

_VEHICLE_TYPES = frozenset({"ego", "agent", "vehicle", 0})
_LANE_TYPES = frozenset({"lane", 1})


class InterventionError(RuntimeError):
    """干预无法在给定图/预算下执行（调用方必须显式处理，不得静默跳过）。"""


@dataclass(frozen=True)
class InterventionResult:
    """一次干预的产物与其可审计的元数据。"""

    edge_index: torch.Tensor          # [2, 2E'] 干预后的边集（仍然成对对称）
    n_undirected_before: int
    n_undirected_after: int
    n_removed: int
    n_added: int
    kind: str                         # 'semantic' | 'random_matched' | 'rewire_matched'
    category: str | None              # semantic/rewire 才有


def _normalize_type(t) -> str:
    """把 `SUBGRAPH_ID_TO_TYPE` 的取值归一到 'agent' / 'lane'。

    :raises InterventionError: 遇到既不是车辆也不是车道的类型（宁可炸也不要静默归错类）。
    """
    if isinstance(t, str):
        key = t.strip().lower()
    else:
        key = t
    if key in _VEHICLE_TYPES:
        return "agent"
    if key in _LANE_TYPES:
        return "lane"
    raise InterventionError(f"未知节点类型 {t!r}；语义分类必须显式覆盖所有类型")


def node_types_for_graph(subgraph_id_to_type: Dict, node_order: Sequence) -> List[str]:
    """按 GNN 图的节点顺序给出每个节点的归一类型。

    ⚠️ 归一会把 `'ego'` 并进 `'agent'`（E0 契约里 ego 也是车辆节点），这对**边级**分类是对的
    （`SEMANTIC_CATEGORIES` 只有三类），但**节点级干预必须另外用**
    :func:`ego_mask_for_graph` **把 ego 排除**，否则「删掉全部 agent」会连自车节点一起删。

    :param subgraph_id_to_type: pkl 里的 `SUBGRAPH_ID_TO_TYPE`。
    :param node_order: `gnn_graph['nodes']`，即 `gnn_x` 的行顺序。
    """
    return [_normalize_type(subgraph_id_to_type[n]) for n in node_order]


def ego_mask_for_graph(subgraph_id_to_type: Dict, node_order: Sequence) -> np.ndarray:
    """按 GNN 图的节点顺序标出哪些节点是**自车**（协议 v1 §5：节点级干预必须排除 ego）。

    :returns: 长度为节点数的 bool 数组；`SUBGRAPH_ID_TO_TYPE` 用整数编码时无法识别 ego，
        此时全为 False，由调用方按 `n_ego == 0` 报警而不是静默继续。
    """
    return np.array(
        [isinstance(subgraph_id_to_type[n], str) and subgraph_id_to_type[n].strip().lower() == "ego"
         for n in node_order],
        dtype=bool,
    )


def split_directed_half(edge_index: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """取出无向边的"正向那一半"，并校验补反向的配对结构确实成立。

    :returns: `(directed[2, E], E)`
    :raises InterventionError: 边数为奇数，或后半段不是前半段的严格反向。
    """
    total = edge_index.shape[1]
    if total == 0:
        return edge_index.new_zeros((2, 0)), 0
    if total % 2 != 0:
        raise InterventionError(f"edge_index 列数 {total} 为奇数，不符合无向补反向的存储约定")
    e = total // 2
    fwd, bwd = edge_index[:, :e], edge_index[:, e:]
    if not torch.equal(fwd.flip(0), bwd):
        raise InterventionError("edge_index 后半段不是前半段的反向，配对假设不成立；"
                                "干预会破坏邻接对称性，拒绝执行")
    return fwd, e


def classify_edges(directed: torch.Tensor, node_types: Sequence[str]) -> np.ndarray:
    """给每条无向边打语义类别标签。

    :param directed: `[2, E]` 正向半边。
    :returns: 长度 E 的字符串数组，取值属于 :data:`SEMANTIC_CATEGORIES`。
    """
    if directed.shape[1] == 0:
        return np.empty(0, dtype=object)
    src = directed[0].tolist()
    dst = directed[1].tolist()
    labels = []
    for a, b in zip(src, dst):
        ta, tb = node_types[a], node_types[b]
        labels.append("-".join(sorted((ta, tb))) if ta != tb else f"{ta}-{ta}")
    return np.array(labels, dtype=object)


def _rebuild_symmetric(directed: torch.Tensor) -> torch.Tensor:
    """把正向半边重新补成对称的 `[2, 2E]`。"""
    if directed.shape[1] == 0:
        return directed.new_zeros((2, 0))
    return torch.cat([directed, directed.flip(0)], dim=1)


def _resolve_budget(n_available: int, budget: float | int) -> int:
    """把 budget 解释成"要移除的无向边条数"。

    :param budget: `0 < budget <= 1` 视为比例；`>= 1` 的整数视为绝对条数。
    """
    if budget <= 0:
        raise InterventionError(f"budget 必须为正，收到 {budget}")
    k = int(round(n_available * budget)) if budget <= 1 else int(budget)
    return max(0, min(k, n_available))


def intervene_semantic(edge_index: torch.Tensor, node_types: Sequence[str],
                       category: str, budget: float | int,
                       rng: np.random.Generator) -> InterventionResult:
    """G7-core：只移除指定语义类别的边。

    :param category: :data:`SEMANTIC_CATEGORIES` 之一。
    :param budget: 移除量，比例或绝对条数；**按该类别内的可用边数**解释。
    :raises InterventionError: 该类别在此图上没有边（调用方必须跳过该 scene 并记录，
        不得当成"干预了但没效果"）。
    """
    if category not in SEMANTIC_CATEGORIES:
        raise InterventionError(f"未知语义类别 {category!r}")
    directed, e = split_directed_half(edge_index)
    labels = classify_edges(directed, node_types)
    idx = np.flatnonzero(labels == category)
    if idx.size == 0:
        raise InterventionError(f"该图没有 {category} 边，无法施加语义干预")

    k = _resolve_budget(idx.size, budget)
    drop = rng.choice(idx, size=k, replace=False) if k > 0 else np.empty(0, dtype=int)
    keep = np.setdiff1d(np.arange(e), drop, assume_unique=False)
    return InterventionResult(
        edge_index=_rebuild_symmetric(directed[:, torch.as_tensor(keep, dtype=torch.long)]),
        n_undirected_before=e, n_undirected_after=int(keep.size),
        n_removed=int(k), n_added=0, kind="semantic", category=category,
    )


def intervene_random_matched(edge_index: torch.Tensor, n_removed: int,
                             rng: np.random.Generator) -> InterventionResult:
    """edge-count-matched random control：从**全图**随机移除等量的边。

    与 :func:`intervene_semantic` 的差别只有"移除哪些"，移除的条数严格相同 ——
    这正是 primary estimand 里 `matched control` 的定义。
    """
    directed, e = split_directed_half(edge_index)
    if n_removed > e:
        raise InterventionError(f"要移除 {n_removed} 条，但图只有 {e} 条无向边")
    drop = rng.choice(e, size=n_removed, replace=False) if n_removed > 0 else np.empty(0, dtype=int)
    keep = np.setdiff1d(np.arange(e), drop, assume_unique=False)
    return InterventionResult(
        edge_index=_rebuild_symmetric(directed[:, torch.as_tensor(keep, dtype=torch.long)]),
        n_undirected_before=e, n_undirected_after=int(keep.size),
        n_removed=int(n_removed), n_added=0, kind="random_matched", category=None,
    )


def intervene_rewire_matched(edge_index: torch.Tensor, node_types: Sequence[str],
                             category: str, n_removed: int,
                             rng: np.random.Generator) -> InterventionResult:
    """replacement/resampling control：移除 k 条语义边后，再随机接回 k 条不存在的边。

    作用是把"少了 k 条边（capacity 下降）"这个混杂因素扣掉 —— 边数与干预前**完全相同**，
    变的只有连接是否语义合理。若效应在此对照下消失，说明模型依赖的是连通度而非语义结构。

    :raises InterventionError: 图已接近完全图、补不回等量的新边（此时该 scene 不可用）。
    """
    base = intervene_semantic(edge_index, node_types, category, n_removed, rng)
    directed, e = split_directed_half(edge_index)
    n_nodes = int(directed.max().item()) + 1 if e > 0 else 0

    existing = {(min(a, b), max(a, b)) for a, b in zip(directed[0].tolist(), directed[1].tolist())}
    kept = base.edge_index
    kept_dir, _ = split_directed_half(kept)
    pool_max = n_nodes * (n_nodes - 1) // 2
    if pool_max - len(existing) < base.n_removed:
        raise InterventionError("图接近完全图，无法补回等量的新边；该 scene 不适用 rewire 对照")

    added: List[Tuple[int, int]] = []
    guard = 0
    while len(added) < base.n_removed:
        guard += 1
        if guard > 100 * max(1, base.n_removed) + 1000:
            raise InterventionError("重连采样超过重试上限，放弃该 scene")
        a, b = int(rng.integers(n_nodes)), int(rng.integers(n_nodes))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in existing:
            continue
        existing.add(key)
        added.append(key)

    add_t = torch.tensor(np.array(added).T, dtype=torch.long) if added else kept_dir.new_zeros((2, 0))
    new_dir = torch.cat([kept_dir, add_t], dim=1)
    return InterventionResult(
        edge_index=_rebuild_symmetric(new_dir),
        n_undirected_before=e, n_undirected_after=new_dir.shape[1],
        n_removed=base.n_removed, n_added=len(added),
        kind="rewire_matched", category=category,
    )


def zero_nodes(gnn_x: torch.Tensor, drop: np.ndarray) -> torch.Tensor:
    """节点特征层面的干预（Q20 路线 (c)）：把**给定下标**的节点特征置零。

    为什么用它替代边层面干预：训练图是**完全图**（实测 E=C(n,2)），边集只是节点数的函数、
    不携带场景信息，故边层面的 semantic/matched-control 对比在本模型上不成立
    （且 replacement control 数学上无法构造）。节点层面则保留了干净的 matched design：
    「置零 k 个 agent 节点」对「置零 k 个 lane 节点」，节点数严格匹配、图结构完全不变。

    这也正是审稿意见要求的干预：R3-M2 的 vehicle-removal perturbation、
    R1-8 的可解释性实证、R1-5 的图结构重要性量化。

    ⚠️ **只收显式下标，不再自己解释 budget**（协议 v1 §3）：旧版接受 `budget`，而
    :func:`_resolve_budget` 把 `1` 当成「比例 100%」而不是「1 个节点」——
    协议里的 `count_1` 剂量正好落在这个歧义上。选点（含 ego 排除、count/fraction 两种口径）
    一律由调用方完成，本函数只负责置零。

    :param drop: 要置零的节点下标。
    :returns: 新的 `gnn_x`；原张量不被修改。
    """
    out = gnn_x.clone()
    if drop.size:
        out[torch.as_tensor(drop, dtype=torch.long)] = 0.0
    return out


def intervene_remove_nodes(gnn_x: torch.Tensor, edge_index: torch.Tensor,
                           drop: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """真正移除节点及其所有关联边，并把剩余节点重新连续编号。

    对应 R3-M2 字面要求的 "vehicle removal"。与 :func:`intervene_node_features` 的
    置零版互为稳健性对照：置零保持节点数不变（图容量不变），移除则同时改变节点数。

    :param drop: 要移除的节点下标。
    :returns: `(新 gnn_x, 新 edge_index)`，edge_index 仍保持成对对称。
    """
    n = gnn_x.shape[0]
    keep = np.setdiff1d(np.arange(n), drop, assume_unique=False)
    if keep.size == 0:
        raise InterventionError("移除后没有剩余节点")
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(keep.size)

    directed, _ = split_directed_half(edge_index)
    src, dst = directed[0].numpy(), directed[1].numpy()
    mask = np.isin(src, keep) & np.isin(dst, keep)
    new_dir = torch.tensor(np.stack([remap[src[mask]], remap[dst[mask]]]), dtype=torch.long)
    return gnn_x[torch.as_tensor(keep, dtype=torch.long)], _rebuild_symmetric(new_dir)


@torch.no_grad()
def response(model, data_base, data_interv, device: torch.device,
             horizon_steps: int | None = None) -> Dict[str, float]:
    """度量一次干预引起的模型输出改变量。

    默认口径**不依赖 GT**：audit 问的是"模型是否依赖该结构"，不是"干预后精度变差多少"。
    ΔADE 一并返回，仅供参考，不作 primary。

    :param horizon_steps: 截断到前 N 步再比较（协议 v1 §4.2 的公共 horizon）。
        MAGNET 原生就是 60 步，故默认 None 时行为不变；PlanTF 侧（80 步）必须显式传 60。
    :returns: `mean_disp` 主度量（每步位置偏移的样本均值，米）、`max_disp`、
        `mean_abs_increment_diff`、`delta_ade`、`base_scale`（基线轨迹尺度，用于相对效应量）。
    """
    from torch_geometric.data import Batch

    def _fwd(d):
        inc = model(Batch.from_data_list([d]).to(device))[:1].cpu().reshape(-1, 2)
        return inc if horizon_steps is None else inc[:horizon_steps]

    inc_a, inc_b = _fwd(data_base), _fwd(data_interv)
    pos_a, pos_b = inc_a.cumsum(0), inc_b.cumsum(0)
    disp = (pos_a - pos_b).norm(dim=1)

    out = {
        "mean_disp": float(disp.mean()),
        "max_disp": float(disp.max()),
        "mean_abs_increment_diff": float((inc_a - inc_b).abs().mean()),
        "base_scale": float(pos_a.norm(dim=1).mean()),
    }
    y = getattr(data_base, "y", None)
    if y is not None:
        gt = y.reshape(-1, 2).cumsum(0)
        out["delta_ade"] = float((pos_b - gt).norm(dim=1).mean() - (pos_a - gt).norm(dim=1).mean())
    return out
