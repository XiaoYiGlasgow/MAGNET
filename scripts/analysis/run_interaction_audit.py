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
from scripts.analysis.audit_dose_contract import DOSES, Dose, resolve_dose  # noqa: E402

#: `skip_reason` 里的量词，锁死为「节点」以逐位复现已落盘的 G19 shard。
DOSE_UNIT = "节点"


def _load_feature_df(pkl_path: Path) -> Optional[pd.DataFrame]:
    """读单个 `features_<token>.pkl`；读不出来返回 None（调用方记 skip）。"""
    try:
        df = pd.read_pickle(pkl_path)
    except (OSError, EOFError, ValueError) as exc:
        logger.warning("读取失败 %s: %s", pkl_path.name, exc)
        return None
    return df if isinstance(df, pd.DataFrame) and len(df) > 0 else None


def _node_types_from_df(df: pd.DataFrame) -> Optional[Tuple[List[str], np.ndarray]]:
    """按 `gnn_x` 的行顺序还原每个 GNN 节点的归一类型，并单独给出 ego 掩码。

    节点顺序 = `subgraph_pooling()` 产出 dict 的键顺序 —— 与
    `build_graph_data_from_feature_df` 内部 `gnn_graph['nodes']` 的来源完全相同。

    :returns: `(node_types, is_ego)`；`node_types` 里 ego 已被归一成 `'agent'`（边级分类要这样），
        节点级干预必须再用 `is_ego` 把自车剔除（协议 §5）。
    """
    if "SUBGRAPH_ID_TO_NODE_INDICES" not in df.columns:
        return None
    features = df["POLYLINE_FEATURES"].values[0].astype(np.float32)
    pooled = subgraph_pooling(features, df["SUBGRAPH_ID_TO_NODE_INDICES"].values[0])
    order = list(pooled.keys())
    id_to_type = df["SUBGRAPH_ID_TO_TYPE"].values[0]
    try:
        return node_types_for_graph(id_to_type, order), ego_mask_for_graph(id_to_type, order)
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
    """取 `output_head` 里最后一个 ReLU（塌陷探针挂在它上面）。"""
    head = getattr(model, "output_head", None)
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
                    device: torch.device, horizon: Optional[int]) -> Dict[str, Dict]:
    """对**同一批被选中的节点**施加 `remove` 与 `zero` 两个算子并各取一次 response。

    两个算子共用同一个 `drop` 集合，故两者之差只反映算子本身（实体消失 vs 实体没信息），
    不掺入「选中了不同实体」的随机性。
    """
    out: Dict[str, Dict] = {}
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


def _node_interventions(model, data, node_types: Sequence[str], is_ego: np.ndarray,
                        doses: Sequence[Dose], rng: np.random.Generator,
                        device: torch.device, horizon: Optional[int]) -> Dict[str, Dict]:
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
        for op, res in _apply_node_ops(model, data, pick_t, device, horizon).items():
            rec[f"{op}_{TARGET_TYPE}"] = res
        for op, res in _apply_node_ops(model, data, pick_c, device, horizon).items():
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


def _load_model(ckpt_path: Path, gnn_pooling_fix: bool, device: torch.device):
    """载入 checkpoint 并置 eval —— **必须走 S4-b 的统一入口**。

    ⚠️ 不要自己 `TransformerPredictor()` + `load_state_dict`：默认构造的
    `ego_history_encoder` BatchNorm 是 `track_running_stats=False`，calibrated ckpt 多出来的
    `running_mean/running_var/num_batches_tracked` 三个键会变成 unexpected keys 被丢弃 ——
    加载到的其实是 **raw 权重而不是 calibrated 权重**（本脚本首次冒烟就踩了这个坑）。
    `load_model_for_inference` 会按 state_dict 里有没有 running stats 自动决定是否先换 BN。
    """
    from src.shared.utils.bn_calibration import load_model_for_inference

    return load_model_for_inference(ckpt_path, gnn_pooling_fix=gnn_pooling_fix, device=device)


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
                   help=f"逗号分隔的剂量名，取值属于 {[d.name for d in DOSES]}（协议 §3.1 冻结）")
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
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    a = parse_args()
    device = torch.device(a.device)
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    by_name = {d.name: d for d in DOSES}
    unknown = [n for n in a.doses.split(",") if n.strip() and n.strip() not in by_name]
    if unknown:
        raise SystemExit(f"未知剂量 {unknown}；协议 §3.1 冻结的取值是 {list(by_name)}")
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

    model = _load_model(a.ckpt, a.gnn_pooling_fix, device)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    records: List[Dict] = []
    skipped: Dict[str, int] = {}
    n_collapsed = 0
    n_no_ego = 0
    max_noop = 0.0
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
        node_types, is_ego = parsed
        data, reason = build_graph_data_from_feature_df(df, a.adjacency_config)
        if data is None:
            skipped[reason or "build_failed"] = skipped.get(reason or "build_failed", 0) + 1
            continue
        if data.gnn_x.shape[0] != len(node_types):
            skipped["node_count_mismatch"] = skipped.get("node_count_mismatch", 0) + 1
            logger.warning("%s: gnn_x 行数 %d ≠ node_types %d，跳过（不猜对应关系）",
                           token, data.gnn_x.shape[0], len(node_types))
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
            rec["doses"] = _node_interventions(model, data, node_types, is_ego, doses,
                                               rng, device, a.horizon_steps)
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
        "skip_edge": a.skip_edge,
        "n_tokens_requested": len(pkls), "n_records": len(records),
        "n_collapsed_head": n_collapsed,
        "n_tokens_without_ego": n_no_ego,
        "max_noop_disp": max_noop,
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
