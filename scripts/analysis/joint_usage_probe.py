"""§6.2 判别实验的读数件 —— 在**新结构**上量交互使用度 `U`。

## 为什么要新写一个

§6.2 比的是 **使用度 U**（旧 = **0.72%**，Q74 `remove_agent`），**不是 ADE**。
但 Q74 的 `response()` 假定 `model(Batch)` 返回 **ego 增量张量**；
`MagnetJointAdaptive` 返回 `JointPrediction`（**位置域**、逐实体逐模态）⇒ 直接复用会静默读错行。

## 口径（逐条对齐 Q74，差一条都不能比）

| 项 | Q74（旧模型） | 本脚本（新结构） |
|---|---|---|
| batch | **1**（planner 部署口径） | **1** |
| 干预 | `remove_agent` = **真移除节点及其全部边** | **同**（`remove_entities` 连 `data.x` 的向量行一并删并重编号） |
| 度量 | `mean_disp / base_scale` | **同** |
| 轨迹 | `cumsum(增量)` | **已是位置域**，直接用（不再 cumsum） |
| 选模 | 单输出，无此问题 | 🔴 **D-i = `argmax π`**（部署口径），**绝不做模态平均** |
| 剂量 | `count_sat` = `min(n_agent_noego, n_lane)` | **同** |

🔴 **空操作门**（Q74 `noop_gate` 的等价物）：空删除集下输出必须与 base **逐位相同**
（`torch.equal`）。不过就抛错 —— 删除流水线把图改坏时产物照样写得出来且看上去正常，
必须让它停在这里。

🔴🔴 **必须跑 CPU**（2026-08-25 实测）：`MagnetJointAdaptive` 在 **GPU 上前向不可复现** ——
**同一份输入连跑两次** max|Δ| = **3.815e-06**（`index_add_` 的 atomic 求和顺序不定），
于是空操作门必然失败，且每一条 response 都会被同量级噪声污染。
**CPU 上逐位可复现**（同输入两次 `torch.equal` 为真、空删除亦为真）⇒ 图手术本身是干净的。
这与本项目既有的 canonical 规则同源（chunk=1 + batch=1 + CPU 才做逐样本精确主张）。
⚠️ 使用度是 **1e-2 m 量级**的量而噪声是 1e-6 ⇒ GPU 噪声不会淹没信号，
但空操作门是**逐位**判据，且「门形同虚设」比「读数略糙」更危险 ⇒ 一律 CPU。

🔴 **跨档纪律**：A（新结构）vs 旧的绝对使用度**不得并列**（§8.2 红线：两侧是不同 ckpt 家族，
绝对使用度受各自预测尺度影响）⇒ 只能报**同 token 配对的 ΔS + log-cluster CI**。
本脚本负责产出逐 token 的 `mean_disp`/`base_scale`，配对与 CI 交给 `paired_usage_delta.py`。

## 🔴🔴🔴 §6.2 表里的「旧 = 0.72%」是**口径错误**，本脚本按模型级重测（2026-08-25 查实）

**0.72% 是 Q74 的 G2 ＝ GNN 通路级**，不是模型级。原因（§8.2 已登记）：旧 audit 的
`run_interaction_audit.py:202/209` 的 `_clone_with` **只覆盖 `gnn_x`/`gnn_edge_index`/`gnn_edge_attr`，
`data.x` 原封不动** ⇒ 被「删掉」的 agent 其 vector-level 表示**仍由 context 通路完整送达**
（实测 `context_feat` 的 max|Δ| = 0.000e+00，20/20 样本）。**模型级是 G4 = 6.8697%**
（三 seed pooled，CI[5.0242, 8.6858]）。

本脚本的 `remove_entities()` **同时删 `gnn_x` 行与 `data.x` 里的向量行** ⇒ 量的是**模型级**。
⇒ 🔴 **拿它与 0.72% 比会把改进虚报约 9.6 倍**，把「A ≫ 旧?」变成平凡为真。

**且新结构没有通路级的对应物** —— Q110 已把 ego/context/gnn 三条分支合并成**一套** per-entity token
⇒ 模型级是**唯一**合法口径，这不是可选项。

**因此本脚本支持 `--model_kind auto`，旧模型也走同一条代码路径**：同一份图手术、同一个
`mean_disp/base_scale`、同一批样本下标、同一批 seed ⇒ A vs 旧变成**同码同档同 token 的配对**，
跨档/跨实现/跨干预覆盖范围三种口径风险一次性消掉。
唯一的分岔在**前向**：旧模型返回 ego 增量 `[60,2]`（需 `cumsum`），新模型返回 `JointPrediction`
（位置域、取 ego 行、`argmax π` 选模）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

_THIS = Path(__file__).resolve()
GNN_ROOT = _THIS.parents[2]
for _p in (str(GNN_ROOT), str(GNN_ROOT / "scripts" / "training"),
           str(GNN_ROOT.parent), str(GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402,F401
from src.shared.utils.bn_calibration import load_model_for_inference, read_architecture  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config, set_subgraph_agg  # noqa: E402
from magnet_joint_encoder import SUBGRAPH_ID_COL, build_entity_index  # noqa: E402

logger = logging.getLogger("joint_usage")

TYPE_AGENT, TYPE_LANE = 0, 1


class ProbeError(RuntimeError):
    """图手术或空操作门失败 —— 必须暴露，不得降级。"""


def remove_entities(data, drop: np.ndarray):
    """**真移除**实体：删 `gnn_x` 行、删其在 `data.x` 里的全部向量行、重编号 subgraph_id、删边。

    🔴 三处必须同时改，少一处就破契约 A（`build_entity_index` 会响亮报错，不会静默）：
    ① `gnn_x` 少一行；② `data.x` 里属于它的向量行全删；③ 其余实体的 `x[:, 9]` 重编号。

    :param drop: `gnn_x` 的行号（单样本 ⇒ 全局行号即局部行号）。
    """
    d = data.clone()
    n_ent = int(d.gnn_x.shape[0])
    keep = np.setdiff1d(np.arange(n_ent), np.asarray(drop, dtype=np.int64))
    if keep.size == 0:
        raise ProbeError("移除后没有剩余实体")
    remap = -np.ones(n_ent, dtype=np.int64)
    remap[keep] = np.arange(keep.size)

    local = d.x[:, SUBGRAPH_ID_COL].round().long().cpu().numpy()
    keep_vec = np.isin(local, keep)
    x_new = d.x[torch.as_tensor(np.flatnonzero(keep_vec), dtype=torch.long)].clone()
    x_new[:, SUBGRAPH_ID_COL] = torch.as_tensor(remap[local[keep_vec]],
                                                dtype=x_new.dtype, device=x_new.device)
    d.x = x_new
    d.gnn_x = d.gnn_x[torch.as_tensor(keep, dtype=torch.long)]

    ei = d.gnn_edge_index.cpu().numpy()
    m = np.isin(ei[0], keep) & np.isin(ei[1], keep)
    d.gnn_edge_index = torch.as_tensor(np.stack([remap[ei[0][m]], remap[ei[1][m]]]),
                                       dtype=torch.long, device=data.gnn_edge_index.device)
    if getattr(d, "cluster", None) is not None:
        cl = d.cluster.cpu().numpy()
        d.cluster = torch.as_tensor(remap[cl[keep_vec]], dtype=torch.long, device=d.x.device)
    return d


@torch.no_grad()
def _ego_all_modes(model, data, device: torch.device, joint: bool = True):
    """返回 `(traj_all [M, 60, 2], argmax_mode)`；旧模型 M=1、mode 恒 0。

    🔴 为什么要把「全部模态」拿出来：见 :func:`usage` 的**模态翻转**说明。
    """
    from torch_geometric.data import Batch
    b = Batch.from_data_list([data]).to(device)
    out = model(b)
    if not joint:
        return out[:1].detach().cpu().reshape(1, -1, 2).cumsum(1), 0
    idx = build_entity_index(b)
    row = int(idx.ego_index[0])
    return out.traj[row].detach().cpu(), int(out.logits[row].argmax())


@torch.no_grad()
def _ego_traj(model, data, device: torch.device, joint: bool = True) -> torch.Tensor:
    """base/干预两侧共用的前向：返回 **ego 轨迹** `[60, 2]`，**位置域**。

    :param joint: True = `MagnetJointAdaptive`（返回 `JointPrediction`，已是位置域，
        取 ego 行并按 🔴 **D-i `argmax π`** 选模，**绝不做模态平均**）；
        False = 旧模型（返回 ego **增量**张量，需 `cumsum` —— 与
        `d2_semantic_intervention.response` 逐字同口径）。
    """
    from torch_geometric.data import Batch
    b = Batch.from_data_list([data]).to(device)
    out = model(b)
    if not joint:
        return out[:1].detach().cpu().reshape(-1, 2).cumsum(0)
    idx = build_entity_index(b)
    row = int(idx.ego_index[0])
    return out.traj[row, int(out.logits[row].argmax())].detach().cpu()


def usage(model, base, interv, device: torch.device, joint: bool = True,
          base_pred=None) -> Dict[str, float]:
    """一次干预的 response。字段名与 `d2_semantic_intervention.response` 同名同义。

    🔴🔴 **多模态模型必须同时报两种口径**（2026-08-25 实测发现，旧模型无此问题）：

    * **`mean_disp`（argmax 口径，主口径）** —— 干预前后各自取 `argmax π`（= D-i 部署口径）。
      它包含**模态翻转**：删掉车之后置信度头可能跳到另一个模态，位移里混进**离散跳变**。
    * **`mean_disp_fixed`（固定模态口径）** —— 干预后**沿用 base 选中的那个模态**，
      隔离出**连续敏感度**。
    * **`mode_flip`** —— 两者是否选了不同模态（0/1）。
    * **`mean_disp_modes`（全模态口径，**完全不依赖选模**）** —— 对 **M 个模态逐一自比**
      （模态 m 的干预前 vs 干预后）再取平均。多模态模型的「ego 轨迹」本来就**不是输入的单值函数**，
      这个口径把选模整个拿掉，是三者里**最不受置信度头污染**的。

    **为什么必须分开**：S2 实测置信度头**弱且不稳定**（test 上 mode CE 1.6809 仅略低于赢家熵 1.6948、
    `best_fixed` 反而优于 top1）⇒ argmax 口径的 U 会被翻转**系统性抬高**
    （B_seed42 实测 U **均值 122.09%** / 中位 13.51% —— 均值 >100% 意味着「轨迹变化超过自身尺度」，
    那是跳变不是敏感度）。
    🔴 **只报 argmax 口径会把「选模不稳」冒充成「交互使用度高」**，而 §6.2 要问的是后者。
    """
    # 🔴 base 前向可复用：同一个样本的 base 在 (2 侧 × N seed) 次调用里**完全相同**，
    #    重算纯属浪费。旧模型的 context 分支是 O(M²)、M 中位 2036 ⇒ 这一项占大头
    #    （实测 legacy 三格 300 token 要 1 小时量级）。复用后每样本前向数 14 → 8，省约 40%。
    ta, ma = base_pred if base_pred is not None else _ego_all_modes(model, base, device, joint)
    tb, mb = _ego_all_modes(model, interv, device, joint)
    a = ta[ma]
    disp = (a - tb[mb]).norm(dim=1)
    disp_fix = (a - tb[ma]).norm(dim=1)
    # 全模态口径：逐模态自比后平均（M=1 时与前两者恒等 ⇒ legacy 侧三口径同值）
    disp_modes = (ta - tb).norm(dim=-1).mean(dim=0)
    return {"mean_disp": float(disp.mean()), "max_disp": float(disp.max()),
            "mean_disp_fixed": float(disp_fix.mean()), "max_disp_fixed": float(disp_fix.max()),
            "mean_disp_modes": float(disp_modes.mean()),
            "mode_flip": float(ma != mb), "base_mode": int(ma), "interv_mode": int(mb),
            "base_scale": float(a.norm(dim=1).mean()),
            # 🔴 全模态口径的分母也要用全模态尺度，否则分子分母不同口径
            "base_scale_modes": float(ta.norm(dim=-1).mean())}


def noop_gate(model, data, device: torch.device, joint: bool = True, base_traj=None) -> None:
    """空删除集下必须逐位恒等。不过就抛错（Q74 `noop_gate` 的等价物）。"""
    a = base_traj if base_traj is not None else _ego_traj(model, data, device, joint)
    b = _ego_traj(model, remove_entities(data, np.zeros(0, dtype=np.int64)), device, joint)
    if not torch.equal(a, b):
        raise ProbeError(
            f"🔴 空操作门失败：空删除集下 ego 轨迹与 base 不逐位相同"
            f"（max|Δ|={float((a - b).abs().max()):.3e}）—— 图手术把图改坏了，全部读数会被同一份"
            f"偏移污染，拒绝继续")


def entity_types(data) -> Dict[str, np.ndarray]:
    """按 `build_entity_index` 的**同一套**判据分实体类型（不另写一份，避免两处分叉）。"""
    from torch_geometric.data import Batch
    idx = build_entity_index(Batch.from_data_list([data]))
    is_agent = idx.entity_is_agent.cpu().numpy()
    valid = idx.entity_valid.cpu().numpy()
    ego = np.zeros_like(is_agent)
    ego[int(idx.ego_index[0])] = True
    return {"agent_noego": np.flatnonzero(is_agent & valid & ~ego),
            "lane": np.flatnonzero(~is_agent & valid),
            "ego": np.flatnonzero(ego)}


def probe_sample(model, data, seeds: Sequence[int], device: torch.device,
                 joint: bool = True) -> Optional[Dict]:
    """一个样本的 `count_sat` 剂量读数（target=agent、control=lane，逐 seed）。"""
    t = entity_types(data)
    n_t, n_c = t["agent_noego"].size, t["lane"].size
    k = min(n_t, n_c)                                    # count_sat：功效最大的剂量
    if k == 0:
        return None
    base_pred = _ego_all_modes(model, data, device, joint)          # 全程复用这一份
    noop_gate(model, data, device, joint, base_traj=base_pred[0][base_pred[1]])
    recs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        pick_t = rng.choice(t["agent_noego"], size=k, replace=False)
        pick_c = rng.choice(t["lane"], size=k, replace=False)
        recs.append({"seed": int(s),
                     "remove_agent": usage(model, data, remove_entities(data, pick_t), device,
                                           joint, base_pred=base_pred),
                     "remove_lane": usage(model, data, remove_entities(data, pick_c), device,
                                          joint, base_pred=base_pred)})
    return {"k": int(k), "n_agent_noego": int(n_t), "n_lane": int(n_c),
            "n_entity": int(data.gnn_x.shape[0]), "records": recs}


def resolve_tokens(root: str, n_dataset: int) -> List[str]:
    """把样本下标映射回 **scenario token**（`paired_usage_delta.py` 以 token 为配对键）。

    `GraphDataset.process()` 的顺序 = `os.walk` 收集全部 `.pkl` 后 **`sorted()`**，
    但它会 **skip** 掉 `skipped_empty_traj` / `skipped_no_polylines` 的样本
    ⇒ 「第 i 个样本」与「排序后第 i 个文件」**未必**对齐。

    🔴 因此这里**不猜**：文件数与数据集长度不等就直接抛错。
    （新档 fullret 实测 skip 全为 0：train 20936/20936、val 2400/2400、test 5943/5943，
    故此路径可用；旧档若有 skip，必须先补一份显式 token 清单再来配对。）
    """
    files = sorted(str(f) for f in Path(root).rglob("*.pkl"))
    if len(files) != n_dataset:
        raise ProbeError(
            f"{root}: 目录下 {len(files)} 个 pkl 但 GraphDataset 有 {n_dataset} 个样本 —— "
            f"`process()` skip 过样本，下标↔token 的位置映射**不成立**。"
            f"按位置硬接会让 100% 的配对错位，此处拒绝降级。")
    return [Path(f).stem.replace("features_", "") for f in files]


def to_shard(res: Dict) -> Dict:
    """转成 `paired_usage_delta.py` 消费的 shard 格式（`{"records": [...]}`）。

    键名与 `q74_2x2_audit` / `run_interaction_audit` 的产物**同名同义**
    （`doses.<dose>.<op_side>.mean_disp/base_scale`），以便直接
    `--audit_dir <目录> --treat <config名>` 复用既有的四层聚合与 log-cluster bootstrap。
    """
    recs = []
    for smp in res["samples"]:
        for rec in smp["records"]:
            recs.append({
                "token": smp["token"], "seed": rec["seed"],
                # 🔴 新结构不做塌陷剔除：它的 margin 监控是另一套（P5），
                #    且 `collapsed_head=False` 会让 paired 的「两侧都非塌陷」过滤退化成恒真。
                #    旧模型侧若要剔除塌陷，须另跑 `collapse_rate_probe.py` 并在此填真值。
                "collapsed_head": smp.get("collapsed_head", False),
                "n_agent_noego": smp["n_agent_noego"], "n_lane": smp["n_lane"],
                "doses": {res["dose"]: {"remove_agent": rec["remove_agent"],
                                        "remove_lane": rec["remove_lane"]}},
            })
    return {"protocol": "joint_usage_probe/模型级真移除", "config": res.get("config_name"),
            "checkpoint": res["checkpoint"], "model_kind": res["model_kind"],
            "caliber": res["caliber"], "records": recs}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="新结构的交互使用度 U（§6.2 判别实验）")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True, help="split leaf root")
    p.add_argument("--out", required=True)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--shard_dir", default=None,
                   help="额外把结果写成 `paired_usage_delta.py` 的 shard 格式到 "
                        "<shard_dir>/<config_name>/shard_000.json")
    p.add_argument("--config_name", default=None, help="shard 的 config 目录名（配对时的臂名）")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--model_kind", choices=("auto", "joint", "legacy"), default="auto",
                   help="🔴 两侧走同一条代码路径才能配对；auto = 按 sidecar 的 joint_adaptive 判定")
    p.add_argument("--device", default="cpu",
                   help="🔴 默认且应当保持 cpu —— GPU 前向不可复现（同输入两次差 3.8e-6），"
                        "空操作门会必然失败，见模块文档")
    a = p.parse_args()

    ckpt = Path(a.checkpoint).resolve()
    sidecar = read_architecture(ckpt) or {}
    joint = bool(sidecar.get("joint_adaptive"))
    if a.model_kind != "auto":
        want = a.model_kind == "joint"
        if want != joint:
            raise SystemExit(f"--model_kind={a.model_kind} 与 sidecar 的 joint_adaptive={joint} 不符 ⇒ 拒绝猜")
    logger.info("模型类型 = %s（sidecar joint_adaptive=%s）", "joint" if joint else "旧模型", joint)
    set_subgraph_agg(sidecar.get("subgraph_agg") or "mean")
    meta = json.loads((ckpt.parent / "train_meta.json").read_text())
    get_adjacency_config(meta.get("dataset_config") or "default")

    ds = GraphDataset(a.root, transform=None)
    tokens = resolve_tokens(a.root, len(ds))
    dev = torch.device(a.device)
    if dev.type != "cpu":
        logger.warning("🔴 device=%s：GPU 前向不可复现（实测同输入两次 max|Δ|=3.8e-06）⇒ "
                       "空操作门会失败。除非在做敏感性分析，否则请用 cpu。", a.device)
    model = load_model_for_inference(ckpt, device=dev).to(dev).eval()

    out: List[Dict] = []
    for i in range(min(a.n_samples, len(ds))):
        r = probe_sample(model, ds[i], a.seeds, dev, joint)
        if r is not None:
            r["sample_index"] = i
            r["token"] = tokens[i]
            out.append(r)
        if (i + 1) % 50 == 0:
            logger.info("… %d/%d（有效 %d）", i + 1, a.n_samples, len(out))

    def agg(side: str) -> Dict[str, float]:
        recs = [rec[side] for s in out for rec in s["records"]]
        u = [r["mean_disp"] / max(r["base_scale"], 1e-9) for r in recs]
        uf = [r["mean_disp_fixed"] / max(r["base_scale"], 1e-9) for r in recs]
        um = [r["mean_disp_modes"] / max(r["base_scale_modes"], 1e-9) for r in recs]
        return {"U_mean": float(np.mean(u)), "U_median": float(np.median(u)),
                "U_fixed_mean": float(np.mean(uf)), "U_fixed_median": float(np.median(uf)),
                "U_modes_mean": float(np.mean(um)), "U_modes_median": float(np.median(um)),
                "mean_disp_mean": float(np.mean([r["mean_disp"] for r in recs])),
                "mode_flip_rate": float(np.mean([r["mode_flip"] for r in recs])), "n": len(u)}

    res = {"checkpoint": str(ckpt), "root": a.root, "model_kind": "joint" if joint else "legacy",
           "n_samples_probed": len(out),
           "seeds": list(a.seeds), "dose": "count_sat",
           "learn_topology": sidecar.get("learn_topology"),
           "joint_ego_only": meta.get("architecture", {}).get("joint_ego_only"),
           "remove_agent": agg("remove_agent"), "remove_lane": agg("remove_lane"),
           "config_name": a.config_name,
           "caliber": "batch=1；真移除节点+全部边+其向量行；U = mean_disp/base_scale；D-i argmax π",
           "cross_tier_warning": "🔴 与旧档绝对使用度不得并列，只能报同 token 配对 ΔS（§8.2）",
           "samples": out}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    if a.shard_dir:
        if not a.config_name:
            raise SystemExit("--shard_dir 必须与 --config_name 同时给（config 目录名 = 配对时的臂名）")
        sd = Path(a.shard_dir) / a.config_name
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "shard_000.json").write_text(json.dumps(to_shard(res), ensure_ascii=False, indent=2))
        logger.info("shard 已写出：%s", sd / "shard_000.json")

    ra, rl = res["remove_agent"], res["remove_lane"]
    print(f"\n=== 使用度 | {'joint' if joint else 'legacy'} | {len(out)} 样本 × "
          f"{len(a.seeds)} seed | dose=count_sat ===")
    print(f"  {'':13s} {'U 中位':>9s} {'U_fix 中位':>11s} {'U_mode 中位':>12s} "
          f"{'U 均值':>10s} {'U_mode 均值':>12s} {'翻转率':>8s}")
    for nm, r in (("remove_agent", ra), ("remove_lane", rl)):
        print(f"  {nm:13s} {r['U_median'] * 100:8.3f}% {r['U_fixed_median'] * 100:10.3f}% "
              f"{r['U_modes_median'] * 100:11.3f}% {r['U_mean'] * 100:9.3f}% "
              f"{r['U_modes_mean'] * 100:11.3f}% {r['mode_flip_rate'] * 100:7.2f}%")
    print(f"  🔑 中位数是主读数（分布右偏，均值被尾部主导）；"
          f"`U_mode` 完全不依赖选模，是三者里最干净的")
    if joint and ra["mode_flip_rate"] > 0.05:
        print(f"  🔴 模态翻转率 {ra['mode_flip_rate'] * 100:.1f}% ⇒ **argmax 口径的 U 含离散跳变**，"
              f"§6.2 判读须以 `U_fixed` 为主口径、argmax 口径同报")
    print(f"\n🔴 对照的**只能是模型级 G4 = 6.8697%**（Q74 三 seed，CI[5.0242,8.6858]），"
          f"**不是 0.72%** —— 那是 GNN **通路级**（G2），旧 audit 的 `_clone_with` 不动 `data.x`。")
    print(f"🔴 且跨档不得并列绝对值 ⇒ 用 `--model_kind legacy` 在**同一档同一批样本**上重跑旧 ckpt，"
          f"再交 `paired_usage_delta.py` 出配对 ΔS。")
    print(f"落盘 {a.out}")


if __name__ == "__main__":
    main()
