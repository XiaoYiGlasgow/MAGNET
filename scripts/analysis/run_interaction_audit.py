"""Controlled Interaction-Reliance Audit —— 干预审计的可复跑驱动（纯 eval，零训练）。

本脚本把 `d2_semantic_intervention.py` 的干预原语接成一个**逐 token 可配对**的实验，
产出的估计量是：

    Semantic Specificity = response(目标干预) − response(matched control)

即「模型对交互输入的依赖，扣掉『干预了多少东西』这个混杂之后还剩多少」。
这正是 R3-M2 亲自设计的实验（*"remove individual surrounding vehicles and measure how the
planned ego trajectory changes"*），也是 R1-8 / R4-3 / R1-2 / R2-2 要的干预与归因证据。

为什么必须从 pkl 重建而不是直接读 `dataset_*.pt`
------------------------------------------------
`build_graph_data_from_feature_df` **没有把节点类型写进 `GraphData`**
（字段只有 `x/y/cluster/edge_index/.../gnn_x/gnn_edge_index/gnn_edge_attr/ego_history`），
而语义分类必须知道每个 GNN 节点是 ego/agent/lane。三条出路里：

* 改 `GraphData` 加 `gnn_node_type` ⇒ 要重做全部预处理，且会动 D0#4 冻结的
  feature schema hash `31bce53e29f9f0bd` —— **不可接受**；
* 从 `gnn_edge_attr` 反推 ⇒ **无效**，`DefaultBuilder` 的 `edge_types` 恒为 0；
* **从 `features_<token>.pkl` 重建**（pkl 里有 `SUBGRAPH_ID_TO_TYPE`）⇒ 不动任何冻结契约，
  且能顺带拿到 token，与官方闭环产物逐 token 配对。**本脚本走这条。**

口径约束
--------
* 推理端干预：同一 checkpoint、同一 scene、同一 seed、同一 budget，只改图；不涉及重训。
* forward 一律 **batch=1**（planner 部署口径）。batch=1 下 GNN 池化 fix ON/OFF 逐位相同
  （主控表 Q13 实测），故本脚本对 fix=False 的 calibrated ckpt 同样成立。
* response 主度量**不依赖 GT**：audit 问「模型是否依赖该结构」，不是「干预后精度差多少」。
* 任何无法施加的干预都**显式记录 skip 原因**，绝不静默当成「干预了但没效果」。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/run_interaction_audit.py \
        --ckpt experiments/calibration/calibrated/B_seed42/B_seed42_calibrated.pth \
        --raw_dir data/processed_features_magnet_e1_none_v3_split3/val \
        --adjacency_config default --limit 200 --seeds 1000,1001,1002 \
        --out experiments/interaction_audit/B_seed42_default.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent), str(_GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.d2_semantic_intervention import (  # noqa: E402
    SEMANTIC_CATEGORIES,
    InterventionError,
    classify_edges,
    ego_mask_for_graph,
    intervene_random_matched,
    intervene_remove_nodes,
    intervene_rewire_matched,
    intervene_semantic,
    node_types_for_graph,
    response,
    split_directed_half,
    zero_nodes,
)
from scripts.analysis.temporal_intervention import (  # noqa: E402
    TemporalInterventionError,
    attach_temporal_graph,
    num_entities,
    remove_temporal_entities,
    verify_entity_alignment,
    zero_temporal_entities,
)
from src.shared.utils.dataset import (  # noqa: E402
    build_graph_data_from_feature_df,
    subgraph_pooling,
)

logger = logging.getLogger("interaction_audit")

#: 协议版本 —— 与 `docs/work_records/paper/MAGNET/VT-2026-04191_修改工作记录/
#: 干预协议_INTERVENTION_PROTOCOL_v1_20260731.md` 绑定，写进每个产物用于事后对账。
PROTOCOL_VERSION = "INTERVENTION_PROTOCOL_v1_20260731"

#: 目标类型 → matched control 类型（协议 §1：实体级，两个方向都跑，读数配对相减）。
TARGET_TYPE, CONTROL_TYPE = "agent", "lane"


# Q47（08-01 收口）：剂量口径**只有一份定义**，两侧共用 `audit_dose_contract`。
#   之前 MAGNET 侧与 PlanTF 侧各有一份等价副本 —— PlanTF 的 `plantf` env 没有
#   `torch_geometric`，无法 import 本文件，故当时先落了零依赖的共享模块、暂留副本，
#   由 `tests/test_audit_dose_contract.py`（162 个 (剂量,n_t,n_c) 组合逐点比对）保证不分叉。
#   G19 四 config 已于 08-01 08:52 全部跑完、无进程再加载本文件 ⇒ 现删副本改 import。
#   🔴 `unit="节点"` 不是措辞偏好：`skip_reason` 会被写进 shard JSON
#   （见下方 `{"skipped": skip}`），已落盘的 G19 产物用的就是「节点」。
#   ⚠️ 用包路径 import（与本文件其余 import 一致）—— 裸 `import audit_dose_contract`
#   只在「直接执行本脚本」时有效，被 `scripts.analysis.run_interaction_audit` 方式导入
#   （单测就是这么导的）时会 ModuleNotFoundError。
from scripts.analysis.audit_dose_contract import (  # noqa: E402
    DOSES, RATIO_DOSES, Dose, resolve_dose,
)

#: `--doses` 的白名单 = 冻结的 primary 三个 + Q83 的剂量-反应扫描网格。
#: 🔴 默认值仍然只有冻结的三个 —— 扫描网格必须**显式指定**才会跑，
#: 避免有人无意中把 exploratory 的格子混进 primary family。
_ALL_DOSES = DOSES + RATIO_DOSES

#: `skip_reason` 里的量词，锁死为「节点」以逐位复现已落盘的 G19 shard。
DOSE_UNIT = "节点"

#: 🔴 与旧架构读数的可比性说明 —— 直接写进产物 provenance，判读时不必再去翻主控表。
#: R1 的 ΔS 要与 `s4b_magnet_L1` 配对相减，先决条件是**两侧测的是同一个 estimand**。
_COMPARABILITY_SHARED = {
    "estimand_level": "通路级（GNN 分支），非模型级",
    "why": "两侧的 `_clone_with` 都只覆盖 GNN 分支的输入，`data.x` 原封不动 ⇒ 被『删掉』的 "
           "agent 其 vector-level 表示仍由 context 通路完整送达（阶段85 实测 "
           "context_feat max|Δ| = 0.000e+00，20/20 样本）",
    "not_comparable_with": "PlanTF 的 13.80% —— 那是把 valid_mask 整行置 False、"
                           "agent 从**整个模型**消失（单通路），干预覆盖范围不同（§8）",
    "dose_unit": "实体（agent/lane 各一个 subgraph），两侧逐字同义",
    "metric": "mean_disp / base_scale，median-of-ratios，非塌陷子集（§8 主口径，不得换）",
}

_COMPARABILITY_LEGACY = {
    **_COMPARABILITY_SHARED,
    "architecture": "旧架构（建图前 `subgraph_pooling` 沿时间轴 np.mean，1 实体 = 1 节点）",
    "intervened_fields": ["gnn_x", "gnn_edge_index", "gnn_edge_attr"],
}

_COMPARABILITY_TEMPORAL = {
    **_COMPARABILITY_SHARED,
    "architecture": "R1 逐帧场景图（1 实体 = T=19 个节点，每帧一个）+ 跨帧时序 Transformer",
    "intervened_fields": ["tg_x", "tg_edge_index", "tg_frame"],
    "same_as_legacy": [
        "干预覆盖范围相同：两侧都只切 GNN 分支，context 分支两侧都完整保留",
        "干预单位相同：删/置零的是**实体**，R1 内部展开到该实体的全部 19 帧节点",
        "选点相同：ego 已从两侧干预池剔除（协议 §5），剂量合同与 seed 均未改",
        "度量相同：response 与 base_scale 的定义、batch=1、horizon 均未改",
    ],
    "differs_from_legacy": [
        "🔴 GNN 分支内部的图规模不同（N vs T*N 个节点）⇒ 任何按图规模归一的量不可并列，"
        "ΔS 用的 mean_disp/base_scale 不属于此类（base_scale 是**轨迹**尺度不是图规模）",
        "🔴 两侧是不同的 ckpt 家族，绝对使用度受各自预测尺度影响 ⇒ "
        "只能报**配对差 ΔS**（同 token 配对、log-cluster CI），不得并列绝对值",
        "🔴 边层面补充分析在 R1 侧不可用（打在 gnn_edge_index 上，R1 不读）⇒ 强制 --skip_edge",
    ],
    "known_limitation": "R1 的 GNN 分支同时引入了跨帧时序 Transformer（新增模块、随机初始化后训练）"
                        "⇒ ΔS 度量的是「逐帧建图 + 时序聚合」这一**整包**的效应，"
                        "不能拆成「逐帧建图单独贡献多少」（预注册判据 §1 已把 GNN 分支整体作为唯一变量）。"
                        "另：`count_sat` 是 k = min(n_agent, n_lane)，只有 n_lane ≥ n_agent 时才等于"
                        "「删光全部 agent」⇒ 判读一律看每条记录里实际落盘的 k_target / frac_target",
}


def _load_feature_df(pkl_path: Path) -> Optional[pd.DataFrame]:
    """读单个 `features_<token>.pkl`；读不出来返回 None（调用方记 skip）。"""
    try:
        df = pd.read_pickle(pkl_path)
    except (OSError, EOFError, ValueError) as exc:
        logger.warning("读取失败 %s: %s", pkl_path.name, exc)
        return None
    return df if isinstance(df, pd.DataFrame) and len(df) > 0 else None


def _node_types_from_df(df: pd.DataFrame) -> Optional[Tuple[List[str], np.ndarray, List]]:
    """按 `gnn_x` 的行顺序还原每个 GNN 节点的归一类型，并单独给出 ego 掩码。

    节点顺序 = `subgraph_pooling()` 产出 dict 的键顺序 —— 与
    `build_graph_data_from_feature_df` 内部 `gnn_graph['nodes']` 的来源完全相同。

    :returns: `(node_types, is_ego, node_order)`；`node_types` 里 ego 已被归一成 `'agent'`
        （边级分类要这样），节点级干预必须再用 `is_ego` 把自车剔除（协议 §5）。
        `node_order` 是这些行对应的 subgraph_ID，R1 侧用它校验与 `tg_x` 的实体顺序一致。
    """
    if "SUBGRAPH_ID_TO_NODE_INDICES" not in df.columns:
        return None
    features = df["POLYLINE_FEATURES"].values[0].astype(np.float32)
    pooled = subgraph_pooling(features, df["SUBGRAPH_ID_TO_NODE_INDICES"].values[0])
    order = list(pooled.keys())
    id_to_type = df["SUBGRAPH_ID_TO_TYPE"].values[0]
    try:
        return (node_types_for_graph(id_to_type, order),
                ego_mask_for_graph(id_to_type, order), order)
    except InterventionError as exc:
        logger.warning("节点类型归一失败: %s", exc)
        return None


def _clone_with(data, **overrides):
    """浅拷贝一个 `GraphData` 并替换指定字段（原对象不变）。"""
    out = data.clone()
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def _last_relu(model) -> Optional[torch.nn.Module]:
    """取**实际参与前向**的输出头里最后一个 ReLU（塌陷探针挂在它上面）。

    🔴 不能写死 `output_head`：`MagnetImitationMultiModal` 走 `traj_head`，
    父类的 `output_head` 在它里面是死参数 —— 挂上去 hook 永不触发、`collapsed` 恒为 None，
    而「非塌陷子集」是 §8 使用度的主口径组成部分 ⇒ 主判据被静默污染。
    ⇒ 模型自报挂点（`active_head()`），没有该方法的才回退到 `output_head`。
    """
    head = model.active_head() if hasattr(model, "active_head") else getattr(model, "output_head", None)
    if head is None:
        return None
    relus = [m for m in head if isinstance(m, torch.nn.ReLU)]
    return relus[-1] if relus else None


@torch.no_grad()
def _probe_collapsed_head(model, data, device: torch.device) -> Optional[bool]:
    """探测本场景是否触发**输出头塌陷**。

    2026-07-31 实测：`output_head` 的第二个 `ReLU` 在部分场景上输出**恒为 0**，
    于是末层 `Linear(128,120)` 的输入是零向量、输出退化为**纯 bias** ——
    该场景的预测与 ego / context / GNN **全部三条分支的输入无关**。
    20 样本抽样：B_seed42_calibrated 7/20、B_seed42_raw 5/20、B_seed1_raw 7/20、B_seed2_raw 1/20。

    🔴 **为什么审计必须记录它**：塌陷场景上任何干预的 response 都恰好是 0.0。
    若不标记，会被误读成「模型不依赖交互输入」—— 而真相是**它此刻什么输入都不依赖**。
    这两件事必须在统计里分开，否则 primary estimand 被稀释成一个假的零效应。
    """
    relu = _last_relu(model)
    if relu is None:
        return None
    from torch_geometric.data import Batch

    cap: Dict[str, torch.Tensor] = {}
    handle = relu.register_forward_hook(lambda mod, inp, out: cap.__setitem__("o", out.detach()))
    try:
        model(Batch.from_data_list([data]).to(device))
    finally:
        handle.remove()
    return None if "o" not in cap else bool(float(cap["o"].abs().max()) == 0.0)


def _resolve_dose(dose: Dose, n_target: int, n_control: int) -> Tuple[int, int, Optional[str]]:
    """把剂量解析成 `(k_target, k_control, skip_reason)`（协议 §3）。

    Q47 后本函数只是 `audit_dose_contract.resolve_dose` 的**薄绑定** ——
    绑住 MAGNET 侧的实体命名（agent/lane）与量词（节点），逻辑不再有第二份。
    """
    return resolve_dose(dose, n_target, n_control,
                        target_name=TARGET_TYPE, control_name=CONTROL_TYPE,
                        unit=DOSE_UNIT)


def _apply_node_ops(model, data, drop: np.ndarray,
                    device: torch.device, horizon: Optional[int],
                    temporal: Optional[str] = None) -> Dict[str, Dict]:
    """对**同一批被选中的节点**施加 `remove` 与 `zero` 两个算子并各取一次 response。

    两个算子共用同一个 `drop` 集合，故两者之差只反映算子本身（实体消失 vs 实体没信息），
    不掺入「选中了不同实体」的随机性。

    :param temporal: None = 旧架构（干预打在 `gnn_x`/`gnn_edge_index`）；
        给出拓扑名（`full`/`knn30`）= R1 逐帧图（干预打在 `tg_x`/`tg_edge_index`）。
        🔴 **两条路必须走同一个 `drop`（实体下标）**，否则两侧的 ΔS 不可配对，见
        `temporal_intervention` 模块 docstring 的「干预单位是实体」。
    """
    out: Dict[str, Dict] = {}
    if temporal is None:
        out["zero"] = response(model, data,
                               _clone_with(data, gnn_x=zero_nodes(data.gnn_x, drop)),
                               device, horizon_steps=horizon)
        try:
            gnn_x, edge_index = intervene_remove_nodes(data.gnn_x, data.gnn_edge_index, drop)
        except InterventionError as exc:
            out["remove"] = {"skipped": str(exc)}
            return out
        interv = _clone_with(data, gnn_x=gnn_x, gnn_edge_index=edge_index,
                             gnn_edge_attr=data.gnn_edge_attr.new_zeros(edge_index.shape[1]))
        out["remove"] = response(model, data, interv, device, horizon_steps=horizon)
        return out

    n_ent = num_entities(data.tg_x)
    out["zero"] = response(model, data,
                           _clone_with(data, tg_x=zero_temporal_entities(data.tg_x, drop, n_ent)),
                           device, horizon_steps=horizon)
    try:
        tg_x, tg_edge_index, tg_frame = remove_temporal_entities(
            data.tg_x, drop, n_ent, topology=temporal)
    except TemporalInterventionError as exc:
        out["remove"] = {"skipped": str(exc)}
        return out
    interv = _clone_with(data, tg_x=tg_x, tg_edge_index=tg_edge_index, tg_frame=tg_frame)
    out["remove"] = response(model, data, interv, device, horizon_steps=horizon)
    return out


def _temporal_noop_disp(model, data, device: torch.device,
                        horizon: Optional[int], topology: str) -> float:
    """🔴 空操作校验：**走完整条干预流水线但一个实体都不删**，位移必须恰好 0.0。

    与已有的 `noop_disp`（同一张图进出两次）互补 —— 那条只证明模型是确定性的，
    本条证明**干预原语本身**（reshape → 删列 → 重算边 → 重新摊平）是恒等的。

    🔴 **非 0 时直接抛错，不是记一个字段了事**：空操作非 0 说明重建路径把图改坏了，
    而这一轮里**每一条** response 都会被同一份偏移污染 —— 产物照样写出来、看上去完全正常。
    这正是「不报错但出假结论」那一类，必须让它停在这里。
    判据是**严格等于 0**：空操作下重建结果与输入逐位相同（`test_temporal_intervention::test_a`
    用 `torch.equal` 锁死），模型确定性 ⇒ 位移必然恰好 0，不存在浮点容差问题。

    :raises RuntimeError: 空操作位移非 0，或两个算子都无法施加（无读数可判）。
    """
    empty = np.zeros(0, dtype=np.int64)
    ops = _apply_node_ops(model, data, empty, device, horizon, temporal=topology)
    usable = [v for v in ops.values() if "skipped" not in v]
    if not usable:
        raise RuntimeError(
            f"空操作校验拿不到任何读数（两个算子都被跳过：{ops}）—— "
            "删 0 个实体本不该失败，说明干预原语的前置条件坏了")
    worst = max(abs(float(v["mean_disp"])) for v in usable)
    if worst != 0.0:
        raise RuntimeError(
            f"🔴 空操作校验失败：不删任何实体时位移 {worst:.6e} ≠ 0 —— "
            "逐帧图重建路径改动了图，本轮所有 response 都已被污染，拒绝继续产出读数")
    return worst


def _node_interventions(model, data, node_types: Sequence[str], is_ego: np.ndarray,
                        doses: Sequence[Dose], rng: np.random.Generator,
                        device: torch.device, horizon: Optional[int],
                        temporal: Optional[str] = None) -> Dict[str, Dict]:
    """节点层面：剂量网格 × {`remove`, `zero`} × {target=agent, matched control=lane}。

    协议 §5：`is_ego` 为真的节点**从两侧的干预池里剔除** —— 旧实现把 ego 归一成 agent，
    「删掉全部 agent」实际上连自车节点一起删了，读数不可用。
    """
    types = np.array(node_types)
    pool_t = np.flatnonzero((types == TARGET_TYPE) & ~is_ego)
    pool_c = np.flatnonzero((types == CONTROL_TYPE) & ~is_ego)

    out: Dict[str, Dict] = {}
    for dose in doses:
        k_t, k_c, skip = _resolve_dose(dose, pool_t.size, pool_c.size)
        if skip is not None:
            out[dose.name] = {"mode": dose.mode, "skipped": skip}
            continue
        pick_t = rng.choice(pool_t, size=k_t, replace=False)
        pick_c = rng.choice(pool_c, size=k_c, replace=False)
        rec: Dict = {
            "mode": dose.mode,
            "k_target": k_t, "k_control": k_c,
            "frac_target": round(k_t / pool_t.size, 4),
            "frac_control": round(k_c / pool_c.size, 4),
        }
        for op, res in _apply_node_ops(model, data, pick_t, device, horizon, temporal).items():
            rec[f"{op}_{TARGET_TYPE}"] = res
        for op, res in _apply_node_ops(model, data, pick_c, device, horizon, temporal).items():
            rec[f"{op}_{CONTROL_TYPE}"] = res
        out[dose.name] = rec
    return out


def _edge_interventions(model, data, node_types: Sequence[str], budget: float | int,
                        rng: np.random.Generator, device: torch.device,
                        horizon: Optional[int] = None) -> Dict[str, Dict]:
    """边层面：语义删边 + 同边数随机对照 + rewire 对照。

    🔴 **协议 v1 §1 把这一族降级为 MAGNET 内部补充分析** —— PlanTF 无图无边，
    边层面干预不可移植，而外部复现是 claim 成立的必要条件。读数**不得作任何 claim 的主证据**。

    ⚠️ 在**完全图**（`adjacency_config=default`，实测 E=C(n,2)）上：
    `rewire` 数学上不可构造，会被显式记为 skip；`semantic vs random_matched` 仍可算，
    但边集只是节点数的函数、不携带场景信息。非完全图（`nearby_only`）上这一族才有信息量。
    """
    out: Dict[str, Dict] = {}
    try:
        directed, n_edges = split_directed_half(data.gnn_edge_index)
    except InterventionError as exc:
        return {"edge_all": {"skipped": str(exc)}}
    if n_edges == 0:
        return {"edge_all": {"skipped": "图无边"}}

    labels = classify_edges(directed, node_types)
    for category in SEMANTIC_CATEGORIES:
        avail = int((labels == category).sum())
        if avail == 0:
            out[f"edge_{category}"] = {"skipped": f"该图无 {category} 边"}
            continue
        try:
            sem = intervene_semantic(data.gnn_edge_index, node_types, category, budget, rng)
        except InterventionError as exc:
            out[f"edge_{category}"] = {"skipped": str(exc)}
            continue
        out[f"edge_{category}"] = {
            "n_removed": sem.n_removed, "n_available": avail,
            **response(model, data, _rewire_data(data, sem.edge_index), device, horizon),
        }

        rnd = intervene_random_matched(data.gnn_edge_index, sem.n_removed, rng)
        out[f"edge_randmatched_{category}"] = {
            "n_removed": rnd.n_removed,
            **response(model, data, _rewire_data(data, rnd.edge_index), device, horizon),
        }

        try:
            rw = intervene_rewire_matched(data.gnn_edge_index, node_types, category,
                                          sem.n_removed, rng)
        except InterventionError as exc:
            out[f"edge_rewire_{category}"] = {"skipped": str(exc)}
            continue
        out[f"edge_rewire_{category}"] = {
            "n_removed": rw.n_removed, "n_added": rw.n_added,
            **response(model, data, _rewire_data(data, rw.edge_index), device, horizon),
        }
    return out


def _rewire_data(data, edge_index: torch.Tensor):
    """换掉 `gnn_edge_index` 并同步 `gnn_edge_attr` 的长度（builder 的 edge_types 恒为 0）。"""
    return _clone_with(data, gnn_edge_index=edge_index,
                       gnn_edge_attr=data.gnn_edge_attr.new_zeros(edge_index.shape[1]))


def _load_model(ckpt_path: Path, gnn_pooling_fix: bool, device: torch.device,
                temporal_graph: bool = False):
    """载入 checkpoint 并置 eval —— **必须走 S4-b 的统一入口**。

    🔴 `--temporal_graph` 必须作为**显式 override** 传下去，不能只靠 sidecar：
    `train_meta.json` 是训练**跑完**才写的，R1 训练中途落盘的 `best_model_default.pth`
    旁边根本没有 sidecar ⇒ 只认 sidecar 会退回老两态、建成父类、strict load 崩。

    ⚠️ 不要自己 `TransformerPredictor()` + `load_state_dict`：默认构造的
    `ego_history_encoder` BatchNorm 是 `track_running_stats=False`，calibrated ckpt 多出来的
    `running_mean/running_var/num_batches_tracked` 三个键会变成 unexpected keys 被丢弃 ——
    加载到的其实是 **raw 权重而不是 calibrated 权重**（本脚本首次冒烟就踩了这个坑）。
    `load_model_for_inference` 会按 state_dict 里有没有 running stats 自动决定是否先换 BN。
    """
    from src.shared.utils.bn_calibration import load_model_for_inference

    # 只在开关为真时传 —— `load_model_for_inference` 用 `is not None` 过滤 override，
    # 传 False 会把 sidecar 里的真值覆盖掉。
    extra = {"temporal_graph": True} if temporal_graph else {}
    return load_model_for_inference(ckpt_path, gnn_pooling_fix=gnn_pooling_fix,
                                    device=device, **extra)


def _check_model_matches_mode(model, temporal: Optional[str], ckpt: Path) -> None:
    """🔴 建出来的模型类必须与 `--temporal_graph` 一致，否则干预打不中被测通路。

    两个方向都会静默出错，故两个方向都拦：
    * 忘了给 `--temporal_graph` 去审 R1 ckpt ⇒ `load_state_dict` 会先崩（strict），
      但万一将来 sidecar 补全了别的路径，这里再拦一道；
    * 给了 `--temporal_graph` 却指向旧架构 ckpt ⇒ 模型压根不读 `tg_x`，
      所有 response 恒为 0，会被读成「模型不依赖交互」。
    """
    is_temporal_model = type(model).__name__ == "TemporalPredictor"
    if is_temporal_model and temporal is None:
        raise SystemExit(f"{ckpt} 建出的是 TemporalPredictor，必须加 --temporal_graph "
                         "（否则干预打在 gnn_x 上，而该模型读的是 tg_x ⇒ response 恒 0）")
    if not is_temporal_model and temporal is not None:
        raise SystemExit(f"{ckpt} 建出的是 {type(model).__name__}，它不读 tg_x；"
                         "去掉 --temporal_graph，否则所有 response 恒为 0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled Interaction-Reliance Audit")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--raw_dir", required=True, type=Path,
                   help="含 features_<token>.pkl 的目录。⚠️ 实测是 split3 的 `<split>/` **本身**"
                        "（如 .../val），PyG 约定的 `<split>/raw/` 是空目录")
    p.add_argument("--adjacency_config", default="default",
                   help="必须与 checkpoint 的训练配置一致（default / nearby_only / ...）")
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="D0#7：必须与 checkpoint 的训练设置一致；G1'/G16 的 ckpt 均为 False")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 个 token（排序后取，可复现）")
    p.add_argument("--tokens_file", type=Path, default=None,
                   help="每行一个 token 的清单；给了它就忽略 --limit")
    p.add_argument("--doses", default=",".join(d.name for d in DOSES),
                   help=f"逗号分隔的剂量名。**默认只有协议 §3.1 冻结的** "
                        f"{[d.name for d in DOSES]}；另可显式指定 Q83 剂量-反应扫描网格 "
                        f"{[d.name for d in RATIO_DOSES]}（固定 k_target=1、k_control=R，"
                        f"用于估计交换率 R*）。🔴 扫描网格是 exploratory，不进 primary family")
    p.add_argument("--edge_budget", type=float, default=1.0,
                   help="**仅边层面补充分析**用的预算；节点层面走 --doses，不受它影响")
    p.add_argument("--seeds", default="1000,1001,1002",
                   help="干预随机性的 seed 列表（逗号分隔）；协议 §7 冻结为 1000,1001,1002")
    p.add_argument("--horizon_steps", type=int, default=None,
                   help="协议 §4.2 公共 horizon；MAGNET 原生 60 步，默认 None 即全长")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8,
                   help="torch CPU 线程数；默认 8，避免与 GPU 训练的 dataloader 抢核")
    p.add_argument("--skip_edge", action="store_true",
                   help="只跑节点层面（协议 §1 已把边层面降级为补充分析，全量跑建议开）")
    p.add_argument("--temporal_graph", action="store_true",
                   help="R1（Q76）：被审 ckpt 是 TemporalPredictor，干预改打在 tg_x/tg_edge_index 上。"
                        "不给这个开关去审 R1 ckpt 会加载失败（strict load 报 unexpected keys）")
    p.add_argument("--temporal_topology", default="full", choices=["full", "knn30"],
                   help="🔴 必须与训练时的 --temporal_topology 一致，否则审的是另一张图")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    a = parse_args()
    device = torch.device(a.device)
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    by_name = {d.name: d for d in _ALL_DOSES}
    unknown = [n for n in a.doses.split(",") if n.strip() and n.strip() not in by_name]
    if unknown:
        raise SystemExit(f"未知剂量 {unknown}；可用取值 = 协议 §3.1 冻结的 "
                         f"{[d.name for d in DOSES]} + Q83 扫描网格 "
                         f"{[d.name for d in RATIO_DOSES]}")
    doses = tuple(by_name[n.strip()] for n in a.doses.split(",") if n.strip())

    from src.shared.utils.dataset import get_adjacency_config
    get_adjacency_config(a.adjacency_config)   # 与训练配置对齐（同 GraphDataset 的做法）

    if a.tokens_file is not None:
        tokens = [t.strip() for t in a.tokens_file.read_text().splitlines() if t.strip()]
        pkls = [a.raw_dir / f"features_{t}.pkl" for t in tokens]
    else:
        pkls = sorted(a.raw_dir.glob("features_*.pkl"))
        if a.limit:
            pkls = pkls[: a.limit]
    logger.info("待审计 token 数=%d，adjacency_config=%s，剂量=%s",
                len(pkls), a.adjacency_config, [d.name for d in doses])

    model = _load_model(a.ckpt, a.gnn_pooling_fix, device, temporal_graph=a.temporal_graph)
    # 旋钮 A（判据 20260810）：建图聚合必须与被审 ckpt 的训练所见一致 —— 按 sidecar 设全局，
    # 缺键（旧 ckpt）回退 mean = 旧行为逐位不变。产物记 subgraph_agg 供 provenance 核对。
    from src.shared.utils.bn_calibration import read_architecture
    from src.shared.utils.dataset import set_subgraph_agg
    _sidecar = read_architecture(Path(a.ckpt)) or {}
    subgraph_agg = _sidecar.get("subgraph_agg") or "mean"
    set_subgraph_agg(subgraph_agg)
    logger.info("建图 subgraph_agg 取自 sidecar: %s", subgraph_agg)
    temporal = a.temporal_topology if a.temporal_graph else None
    _check_model_matches_mode(model, temporal, a.ckpt)
    if temporal is not None and not a.skip_edge:
        # 边层面干预改的是 `gnn_edge_index` —— R1 的 GNN 分支不读它 ⇒ response 恒为 0，
        # 那是「干预没打中通路」不是「模型不依赖边」。宁可拒跑也不产出会被误读的 0。
        raise SystemExit("--temporal_graph 必须与 --skip_edge 同用："
                         "边层面干预打在 gnn_edge_index 上，R1 的 GNN 分支读的是 tg_edge_index，"
                         "跑出来会是恒 0 的假读数（协议 §1 已把边层面降级为补充分析）")

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    records: List[Dict] = []
    skipped: Dict[str, int] = {}
    n_collapsed = 0
    n_no_ego = 0
    max_noop = 0.0
    max_temporal_noop = 0.0
    t0 = time.time()
    for i, pkl in enumerate(pkls):
        token = pkl.stem.replace("features_", "")
        df = _load_feature_df(pkl)
        if df is None:
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
            continue
        parsed = _node_types_from_df(df)
        if parsed is None:
            skipped["no_node_types"] = skipped.get("no_node_types", 0) + 1
            continue
        node_types, is_ego, node_order = parsed
        data, reason = build_graph_data_from_feature_df(df, a.adjacency_config)
        if data is None:
            skipped[reason or "build_failed"] = skipped.get(reason or "build_failed", 0) + 1
            continue
        if data.gnn_x.shape[0] != len(node_types):
            skipped["node_count_mismatch"] = skipped.get("node_count_mismatch", 0) + 1
            logger.warning("%s: gnn_x 行数 %d ≠ node_types %d，跳过（不猜对应关系）",
                           token, data.gnn_x.shape[0], len(node_types))
            continue
        if temporal is not None:
            # 🔴 先校验实体顺序再挂图：`drop` 是按 node_types 的下标选的，
            #    错位的后果是「干预打到别的实体上」且全程不报错。
            try:
                verify_entity_alignment(data.x, node_order)
                attach_temporal_graph(data, temporal)
            except TemporalInterventionError as exc:
                skipped["temporal_misaligned"] = skipped.get("temporal_misaligned", 0) + 1
                logger.warning("%s: %s，跳过（不猜对应关系）", token, exc)
                continue
            if num_entities(data.tg_x) != len(node_types):
                skipped["temporal_count_mismatch"] = skipped.get("temporal_count_mismatch", 0) + 1
                logger.warning("%s: tg 实体数 %d ≠ node_types %d，跳过",
                               token, num_entities(data.tg_x), len(node_types))
                continue
        if not is_ego.any():
            n_no_ego += 1   # 识别不出 ego ⇒ 该 token 的 agent 池可能混入自车，显式计数不静默

        collapsed = _probe_collapsed_head(model, data, device)
        if collapsed:
            n_collapsed += 1
        # no-op 自检（协议 §7）：同一张图进出两次，位移必须恰好为 0
        noop = response(model, data, data.clone(), device,
                        horizon_steps=a.horizon_steps)["mean_disp"]
        max_noop = max(max_noop, abs(noop))
        temporal_noop = None
        if temporal is not None:
            # 🔴 更强的一条：走完整条干预流水线但一个实体都不删，位移仍须恰好 0
            temporal_noop = _temporal_noop_disp(model, data, device, a.horizon_steps, temporal)
            max_temporal_noop = max(max_temporal_noop, temporal_noop)

        for seed in seeds:
            rng = np.random.default_rng(seed)
            rec: Dict = {"token": token, "seed": seed,
                         "collapsed_head": collapsed,
                         "noop_disp": noop,
                         "n_nodes": int(data.gnn_x.shape[0]),
                         "n_ego": int(is_ego.sum()),
                         "n_agent_noego": int(((np.array(node_types) == TARGET_TYPE) & ~is_ego).sum()),
                         "n_lane": int(sum(t == CONTROL_TYPE for t in node_types)),
                         "n_undirected_edges": int(data.gnn_edge_index.shape[1] // 2)}
            if temporal is not None:
                # 只在 R1 侧写，旧 config 复跑时逐字段与已落盘 shard 保持一致
                rec["temporal_noop_disp"] = temporal_noop
                rec["n_tg_rows"] = int(data.tg_x.shape[0])
                rec["n_tg_undirected_edges"] = int(data.tg_edge_index.shape[1] // 2)
            rec["doses"] = _node_interventions(model, data, node_types, is_ego, doses,
                                               rng, device, a.horizon_steps, temporal)
            if not a.skip_edge:
                rec["edge_supplementary"] = _edge_interventions(
                    model, data, node_types, a.edge_budget, rng, device, a.horizon_steps)
            records.append(rec)

        if (i + 1) % 50 == 0:
            logger.info("进度 %d/%d，用时 %.1f min（塌陷 %d）",
                        i + 1, len(pkls), (time.time() - t0) / 60, n_collapsed)

    payload = {
        "protocol": PROTOCOL_VERSION,
        "ckpt": str(a.ckpt), "ckpt_exists": a.ckpt.exists(),
        "raw_dir": str(a.raw_dir), "adjacency_config": a.adjacency_config,
        "doses": [{"name": d.name, "mode": d.mode, "value": d.value, "absolute": d.absolute}
                  for d in doses],
        "edge_budget": a.edge_budget, "seeds": seeds, "device": a.device,
        "horizon_steps": a.horizon_steps,
        "gnn_pooling_fix": a.gnn_pooling_fix,
        "subgraph_agg": subgraph_agg,
        "skip_edge": a.skip_edge,
        "temporal_graph": bool(a.temporal_graph),
        "temporal_topology": a.temporal_topology if a.temporal_graph else None,
        "model_class": type(model).__name__,
        "n_tokens_requested": len(pkls), "n_records": len(records),
        "n_collapsed_head": n_collapsed,
        "n_tokens_without_ego": n_no_ego,
        "max_noop_disp": max_noop,
        "max_temporal_noop_disp": max_temporal_noop if a.temporal_graph else None,
        "comparability": _COMPARABILITY_TEMPORAL if a.temporal_graph else _COMPARABILITY_LEGACY,
        "collapsed_head_note": "output_head 末个 ReLU 输出恒为 0 ⇒ 该场景预测退化为与全部输入无关的常数；这些 token 的 response 恒为 0.0，统计时必须与「不依赖交互」分开",
        "skipped": skipped,
        "estimand": "semantic_specificity = response(移除 k_a 个 agent) - response(移除 k_m 个 lane)；"
                    "response 主度量为 mean_disp（不依赖 GT）；ego 已从两侧干预池剔除；"
                    "count-matched 与 fraction-matched 并列 primary（协议 v1 §3）",
        "wall_minutes": round((time.time() - t0) / 60, 2),
        "records": records,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("写入 %s（%d 条记录，skip=%s，用时 %.1f min）",
                a.out, len(records), skipped, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
