#!/usr/bin/env python
"""S3 —— **GT 锚定的反事实边效用标签**（Q109 §3.3.2 / Q110 §6 的 S3 步）。

以 S2 的 ckpt 为 **teacher**，在**训练/验证 split** 上离线生成 `I_{j→i}`，供 S4 的 edge scorer 拟合。
teacher ckpt 的 sha256 写进 `meta.json` 并**冻结**（Q110 §6：「teacher ckpt 的 sha 进 sidecar，冻结」）。

## 标签定义（Q109 §3.3.2 逐字）

    U_{j→i} = err(Ŷ_i^{cut(j)}, Y_i) − err(Ŷ_i, Y_i)
    I_{j→i} = [ U_{j→i} − E_{c} U_{c→i} ]_+

🔴 **为什么必须 GT 锚定**（Q108 §B.3）：模型锚定的标签量的是「模型**当前用了** j 多少」，
而旧 MAGNET 几乎不用交互（F 门A ΔU_F +0.0787 pt、Q74 G2 0.72%）⇒ 标签接近 0、学出空图，
是自我实现的预言。GT 锚定量的是「**j 对预测 i 值多少**」，判据直接从任务目标推出
—— 正是 R1-1 要的 *principled optimization criterion*。
模型锚定量 `S_{j→i}`（Q109 §3.3.3）**同时**算出来并落盘，两者的一致性是**诊断量**，
🔴 不得预设必然一致，也**不得**拿 `S` 当监督。

## 三个必须显式记录的口径选择（都不是唯一解，都写进 `meta.json`）

**① 标签粒度 vs 干预粒度不同 —— 用一次 D2c 前向给全部目标打标签。**
`D2c`（主控表定的剂量）是**逐源 j** 的干预：切 j 与**全部车辆节点**之间的边，**保留 j–lane**
（`g1_dose_ladder_probe` 第 28 行的定义，与那里的 `_cut_edges_between(ei, j, others_veh)` 同义）。
而 scorer 要的是**逐有向边** `w_{j→i}`。本脚本对每个 j 只做**一次**干预，然后在**每个**目标 i
上读误差变化 ⇒ 每场景前向数 `1 + n_veh + n_placebo` 而不是 `O(n_veh²)`。
🔴 **代价（不得隐瞒）**：`U_{j→i}` 因此包含 `j→k→i` 的二阶通路，不是纯 `j→i` 的边际效用。
D2c 保留 j–lane ⇒ `j→lane→i` 的通路也仍在（D-b 已登记的同一件事）。

**② `err` 用 minADE@K，不用 top1。**
S2 实测**模态翻转率 53–57%**、置信度头是净负担（`best_fixed` 1.7635 < top1 1.8505）
⇒ top1 口径会把**选模抖动**记成边效用，这正是 §6.2 主口径改用 `U_modes` 的同一个理由。
minADE 由 **GT** 选模 ⇒ 与「GT 锚定」自洽，且与 `ckpt_metric=val_ade` 同源。
`--err_metric {minade,top1,meanmode}` 三选一，默认 `minade`，落进 meta。

**③ placebo 用 P2c（随机 lane 节点对全部车辆切边）。**
Q109 §3.3.2 写的是「count-matched 控制边」，`g1_dose_ladder_probe` 的配对表写的是
`D2c ↔ P2c`。本脚本按**既有配对表**走（避免两处分叉），并把两侧的 `n_edges_cut`
都落盘 ⇒ 计数匹配度**可事后审计**（D2c 切 `n_veh−1` 条无向边、P2c 切 `n_veh` 条）。

## 标签范围（Q109 §3.3.4）

只对 **scorable 边**（车↔车、两端非 padding）落标签，且目标 i 必须是监督实体（`is_target`）。
🔴 **未生成标签的边必须靠 mask 排除监督，不得当作零影响边** —— 产物里只有出现过的 `(j,i)` 对，
S4 侧按 `(src,dst)` join，join 不上的边不进损失。

## 跑法

    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/s3_edge_utility_labels.py \
        --teacher experiments/joint_s2_fixed/seed1/models/checkpoint_best.pt \
        --root data/processed_features_fullret_L1_logsplit/train \
        --out data/edge_labels_s3_20260825/train --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
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

import config  # noqa: E402,F401  （建图参数的单一真相源；与训练侧同一份）
from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402
from magnet_joint_encoder import build_entity_index  # noqa: E402
from magnet_joint_layers import build_edge_context, build_entity_geometry  # noqa: E402
from magnet_joint_loss import build_joint_targets_batch, per_mode_errors  # noqa: E402

logger = logging.getLogger("s3_labels")

#: 产物格式版本。改动字段语义必须同步升版，否则 S4 会静默读到旧口径。
S3_VERSION = "S3_EDGE_UTILITY_v1_20260825"
#: 冻结标记文件名（照 `build_fullret_test_split.py` 的 `.TEST_SPLIT_FROZEN` 做法）。
FROZEN_MARK = ".S3_LABELS_FROZEN"

#: GPU 上空操作门的容差。🔴 **由故障注入实测定，不是拍的**（2026-08-25，val，n=11 场景）：
#:
#: * **最小可能的结构损坏**（`edge_gate` 里只切 **1 条**车↔车边）⇒ noop 读数
#:   中位 **0.838 m**、min **0.222 m**、max 7.875 m，**CPU 与 GPU 逐位相同**；
#: * GPU 侧的**浮点噪声**（`index_add_` 的 atomic 求和顺序，§8.2）实测 191 场景
#:   中位 2.67e-05 / p99 6.52e-05 / max 7.63e-05，全量跑里撞到过 **1.53e-04**。
#:
#: ⇒ 1e-3 比噪声尾部高 **6.5×**、比最小真实损坏低 **222×**（分离度 ≥1450×）。
#: 🔴 CPU 上仍然是 **0.0（逐位）** —— 单测 [C][H] 用的就是 CPU 那一档。
GPU_NOOP_TOL = 1e-3


class S3Error(RuntimeError):
    """契约或门失败 —— 一律抛出，不降级（标签错了整条 S4/S5 都建在流沙上）。"""


# ────────────────────────────────────────────────────────────────────────────
# 干预门（都只改门、不重建 edge_index —— 与 `isolate_edge_gate` 同一路数）
# ────────────────────────────────────────────────────────────────────────────
def d2c_gate(ectx, src: int, veh_rows: torch.Tensor) -> torch.Tensor:
    """**D2c**：切 `src` 与**其他车辆节点**之间的边（两个方向），**保留 src–lane 边**。

    与 `g1_dose_ladder_probe._cut_edges_between(ei, j, others_veh)` 语义逐字对应。
    """
    ei = ectx.edge_index
    others = veh_rows[veh_rows != src]
    hit = ((ei[0] == src) & torch.isin(ei[1], others)) | \
          ((ei[1] == src) & torch.isin(ei[0], others))
    return (~hit).to(torch.float32)


def p2c_gate(ectx, lane: int, veh_rows: torch.Tensor) -> torch.Tensor:
    """**P2c** placebo：切随机 lane 节点与**全部车辆节点**之间的边（两个方向）。"""
    ei = ectx.edge_index
    hit = ((ei[0] == lane) & torch.isin(ei[1], veh_rows)) | \
          ((ei[1] == lane) & torch.isin(ei[0], veh_rows))
    return (~hit).to(torch.float32)


# ────────────────────────────────────────────────────────────────────────────
# 复制批：把 K 个干预塞进一次前向
# ────────────────────────────────────────────────────────────────────────────
def _replicate(data, k: int, device: torch.device):
    """把同一个场景复制 `k` 份成一个 `Batch`（`follow_batch=['gnn_x']` 是契约 B 的硬要求）。"""
    from torch_geometric.data import Batch
    return Batch.from_data_list([data] * k, follow_batch=["gnn_x"]).to(device)


@torch.no_grad()
def _forward_replicas(model, data, gates: Sequence[torch.Tensor], device: torch.device):
    """一次前向跑完 `len(gates)` 个干预。

    :param gates: 每个元素 `[E]`（**单副本**的边空间），第 0 个约定是 base（全 1）。
    :returns: `(err [K, n_ent], traj [K, n_ent, M, 60, 2], tg)`；`err` 由 `_ERR` 选定的口径给出。
    """
    k = len(gates)
    b = _replicate(data, k, device)
    idx = build_entity_index(b)
    n_ent = idx.n_entity // k
    if idx.n_entity != n_ent * k:
        raise S3Error(f"复制批的实体数 {idx.n_entity} 不是单份 {n_ent} 的 {k} 倍")
    # 行序必须是 [副本0 的 n_ent 行, 副本1 的 n_ent 行, ...]，否则下面的 reshape 全错
    want = torch.arange(k, device=idx.entity_batch.device).repeat_interleave(n_ent)
    if not torch.equal(idx.entity_batch, want):
        raise S3Error("复制批的实体样本号不是 repeat_interleave 形态 —— reshape 会错位")

    e_single = int(gates[0].numel())
    gate = torch.cat([g.to(device) for g in gates])           # [k*E]
    # 🔴 门是在**单副本**边空间里造的，靠 `gnn_edge_index` 逐副本等长才能直接 cat。
    #    `GraphData.__inc__` 对 `gnn_edge_index` 按 `gnn_x.size(0)` 偏移 ⇒ 边**序**逐副本相同。
    if int(b.gnn_edge_index.shape[1]) != e_single * k:
        raise S3Error(f"复制批的边数 {int(b.gnn_edge_index.shape[1])} != 单份 {e_single} × {k}")

    pred = model(b, edge_gate=gate)
    tg = build_joint_targets_batch(b, idx)
    ade, _ = per_mode_errors(pred.traj, tg)                   # [k*n_ent, M]
    err = _ERR_FN(ade, pred.logits).view(k, n_ent)
    traj = pred.traj.view(k, n_ent, pred.traj.shape[1], pred.traj.shape[2], 2)
    return err, traj, tg


def _err_minade(ade: torch.Tensor, _logits: torch.Tensor) -> torch.Tensor:
    return ade.min(dim=1).values


def _err_top1(ade: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return ade.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)


def _err_meanmode(ade: torch.Tensor, _logits: torch.Tensor) -> torch.Tensor:
    return ade.mean(dim=1)


ERR_FNS = {"minade": _err_minade, "top1": _err_top1, "meanmode": _err_meanmode}
_ERR_FN = ERR_FNS["minade"]          # 由 `main()` 按 --err_metric 改写


# ────────────────────────────────────────────────────────────────────────────
# 单场景打标
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _teacher_pass(model, data, iv: List[torch.Tensor], n_src: int, ones: torch.Tensor,
                  device: torch.device, chunk: int, noop_tol: float):
    """**一个 teacher** 在一个场景上的完整读数。

    :param iv: 干预门列表，前 `n_src` 个是 D2c、其余是 P2c。🔴 由调用方**在 teacher 循环之前**
        造好并共用 —— 三个 teacher 必须吃同一批对照边，否则跨 teacher 取中位数是在比不同的东西。
    :returns: `(u_src, s_src, p_mean, p_sd, err_base, noop)`

    分块跑，且 🔴 **每块自带一个 base 副本、差值在块内作差** —— 批规模不同会让同一次前向差
    ~1e-5（BLAS 按张量形状选 kernel；实测 `err_base` 逐位不变但干预副本会动）。块内作差 ⇒
    两侧经历**完全相同**的数值路径。块 0 额外带一个全 1 副本作空操作门。
    """
    d_err, d_traj, err_base, mask0, noop = [], [], None, None, 0.0
    for st in range(0, len(iv), chunk):
        part = iv[st:st + chunk]
        gs = ([ones, ones] + part) if st == 0 else ([ones] + part)
        off = 2 if st == 0 else 1
        err, traj, tg = _forward_replicas(model, data, gs, device)
        if st == 0:
            err_base = err[0]
            mask0 = tg.mask.view(err.shape[0], -1, tg.mask.shape[-1])[0]
            noop = float((traj[0] - traj[1]).abs().max())
            if noop > noop_tol:
                raise S3Error(
                    f"🔴 空操作门失败：两个全 1 副本的轨迹 max|Δ|={noop:.3e} > 容差 {noop_tol:.1e}"
                    f" —— 复制批把图改坏了，全部标签会被同一份偏移污染，拒绝继续")
        d_err.append(err[off:] - err[0].unsqueeze(0))                    # [k_i, n_ent]
        # S 只要位移大小，先在块内求好逐实体标量，省下 [k,n,M,60] 的显存
        m = mask0.to(traj.dtype)
        dm = (traj[0].unsqueeze(0) - traj[off:]).norm(dim=-1)            # [k_i,n_ent,M,60]
        d_traj.append((dm * m.view(1, -1, 1, m.shape[-1])).sum(-1).mean(-1)
                      / m.sum(-1).clamp(min=1.0).view(1, -1))            # [k_i, n_ent]
    delta, s_all = torch.cat(d_err), torch.cat(d_traj)
    pl = delta[n_src:]
    p_mean = pl.mean(dim=0) if pl.shape[0] else torch.zeros_like(err_base)
    p_sd = pl.std(dim=0, unbiased=False) if pl.shape[0] else torch.zeros_like(err_base)
    return delta[:n_src], s_all[:n_src], p_mean, p_sd, err_base, noop


@torch.no_grad()
def label_scene(models, data, device: torch.device, rng: np.random.Generator,
                n_placebo: int = 4, noop_tol: float = 0.0, chunk: int = 32) -> Optional[Dict]:
    """给一个场景生成全部 `(j, i)` 标签。返回 `{"skip": ...}` 表示该场景不可用。

    :param models: teacher 列表。🔴 **多 teacher 按跨 seed 中位数聚合** —— 与项目
        median-of-ratios 纪律同源。Q110 §6 字面写的是单个 teacher，但本项目已反复自证单 seed
        会得出错误结论（修A 塌陷率 2.17 / 0.00 / 22.17%），⇒ 单 seed 的 teacher 会让整条
        S4/S5 隐含 seed 依赖。跨 teacher 的离散度落盘为 `U_sd_teacher`，使该选择可事后审计。
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    from torch_geometric.data import Batch
    b1 = Batch.from_data_list([data], follow_batch=["gnn_x"]).to(device)
    idx = build_entity_index(b1)
    ectx = build_edge_context(b1.gnn_edge_index, idx, build_entity_geometry(b1.x, idx))
    tg1 = build_joint_targets_batch(b1, idx)

    # 🔑 实测（val 首样本）：`idx.entity_is_agent[ego] = True` ⇒ `ectx.scorable` **含 ego 边**
    #    （该场景 650 条 scorable 里 50 条一端是 ego）⇒ `j→ego` 标签生得出来。
    #    ⚠️ 别与 aux 的 `gnn_is_agent` 混淆 —— 那个**不含 ego**（26 vs 25，两处口径不同）。
    veh_rows = torch.nonzero(idx.entity_is_agent & idx.entity_valid, as_tuple=True)[0]
    veh_rows = torch.unique(torch.cat([veh_rows, idx.ego_index]))     # ego 一定算车辆节点
    lane_rows = torch.nonzero(~idx.entity_is_agent & idx.entity_valid, as_tuple=True)[0]
    tgt_rows = torch.nonzero(tg1.is_target, as_tuple=True)[0]
    if veh_rows.numel() < 2 or lane_rows.numel() == 0 or tgt_rows.numel() == 0:
        return {"skip": f"veh={int(veh_rows.numel())} lane={int(lane_rows.numel())} "
                        f"tgt={int(tgt_rows.numel())}"}

    e = int(ectx.edge_index.shape[1])
    ones = torch.ones(e, dtype=torch.float32, device=device)
    src_list = [int(v) for v in veh_rows.tolist()]
    # 🔴 lane 抽样在 teacher 循环**之前**，三个 teacher 共用同一批 placebo 对照边
    lanes = rng.choice(lane_rows.cpu().numpy(), size=min(n_placebo, int(lane_rows.numel())),
                       replace=False)
    n_src, n_pl = len(src_list), int(lanes.size)
    iv = [d2c_gate(ectx, s_, veh_rows) for s_ in src_list] + \
         [p2c_gate(ectx, int(c), veh_rows) for c in lanes]

    per = [_teacher_pass(m, data, iv, n_src, ones, device, chunk, noop_tol) for m in models]
    st_u = torch.stack([t[0] for t in per])
    u_src = st_u.median(dim=0).values
    u_sd = st_u.std(dim=0, unbiased=False) if len(per) > 1 else torch.zeros_like(u_src)
    s_src = torch.stack([t[1] for t in per]).median(dim=0).values
    p_mean = torch.stack([t[2] for t in per]).median(dim=0).values
    p_sd = torch.stack([t[3] for t in per]).median(dim=0).values
    err_base = torch.stack([t[4] for t in per]).median(dim=0).values
    noop = max(t[5] for t in per)
    # ---- 只落 scorable 边（车↔车、两端非 padding），且目标必须是监督实体 ----
    ei = ectx.edge_index
    keep = ectx.scorable & ectx.valid & torch.isin(ei[1], tgt_rows) & torch.isin(ei[0], veh_rows)
    cols = torch.nonzero(keep, as_tuple=True)[0]
    if cols.numel() == 0:
        return {"skip": "无可打标的车↔车边"}
    # 🔴 分层标志（`scene_strata` 冻结口径，来自 `build_joint_targets_batch`）：
    #    0=ego｜1=运动周车｜2=静止/截断周车。§8.2 红线要求分层报数，事后补要重跑 ⇒ 现在就落盘。
    strat = torch.full((idx.n_entity,), 2, dtype=torch.uint8, device=device)
    strat[tg1.is_moving] = 1
    strat[tg1.is_ego] = 0
    strat_np = strat.cpu().numpy()
    src_pos = {v: k for k, v in enumerate(src_list)}
    j_rows = ei[0, cols].cpu().numpy()
    i_rows = ei[1, cols].cpu().numpy()
    sel = np.array([src_pos[int(j)] for j in j_rows], dtype=np.int64)
    u = u_src.cpu().numpy()[sel, i_rows]
    p = p_mean.cpu().numpy()[i_rows]
    cut_per_src = np.array([int((1.0 - g).sum()) for g in iv[:n_src]], dtype=np.int32)
    cut_placebo = [int((1.0 - g).sum()) for g in iv[n_src:]]
    return {
        "src": j_rows.astype(np.int32), "dst": i_rows.astype(np.int32),
        "U": u.astype(np.float32), "I": np.maximum(u - p, 0.0).astype(np.float32),
        "S": s_src.cpu().numpy()[sel, i_rows].astype(np.float32),
        "placebo": p.astype(np.float32),
        "placebo_sd": p_sd.cpu().numpy()[i_rows].astype(np.float32),
        "U_sd_teacher": u_sd.cpu().numpy()[sel, i_rows].astype(np.float32),
        "err_base_dst": err_base.cpu().numpy()[i_rows].astype(np.float32),
        "n_edges_cut": cut_per_src[sel],
        "dst_stratum": strat_np[i_rows], "src_stratum": strat_np[j_rows],
        "n_entity": int(idx.n_entity), "n_veh": len(src_list), "n_lane": int(lane_rows.numel()),
        "n_placebo": n_pl, "noop_max": noop, "n_teachers": len(models),
        "cut_d2c_mean": float(cut_per_src.mean()),
        "cut_p2c_mean": float(np.mean(cut_placebo)) if cut_placebo else 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
def _flush(shard: List[Dict], tokens: List[str], out: Path, k: int) -> None:
    """把一批场景写成一个 `.npz`（扁平数组 + `scene_ptr` 边界）。"""
    ptr = np.cumsum([0] + [r["src"].size for r in shard]).astype(np.int64)
    np.savez_compressed(
        out / f"labels_{k:04d}.npz", scene_ptr=ptr, token=np.array(tokens, dtype="U64"),
        **{f: np.concatenate([r[f] for r in shard]) for f in
           ("src", "dst", "U", "I", "S", "placebo", "placebo_sd",
            "U_sd_teacher", "err_base_dst", "n_edges_cut",
            "dst_stratum", "src_stratum")},
        **{f: np.array([r[f] for r in shard], dtype=np.int64) for f in
           ("n_entity", "n_veh", "n_lane", "n_placebo")},
        noop_max=np.array([r["noop_max"] for r in shard], dtype=np.float64),
        cut_d2c_mean=np.array([r["cut_d2c_mean"] for r in shard], dtype=np.float32),
        cut_p2c_mean=np.array([r["cut_p2c_mean"] for r in shard], dtype=np.float32))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="S3：GT 锚定反事实边效用标签（teacher = S2 ckpt）")
    p.add_argument("--teacher", required=True, nargs="+",
                   help="S2 的 checkpoint，可给多个（每个 seed 一个）。🔴 多个时按**跨 seed "
                        "中位数**聚合 U/S/placebo；全部 sha 一起冻结进 meta")
    p.add_argument("--root", required=True, help="split leaf root（含 aux 缓存）")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--err_metric", choices=tuple(ERR_FNS), default="minade")
    p.add_argument("--n_placebo", type=int, default=4,
                   help="P2c 抽样数。🔴 默认 4 是实测收敛点：placebo 的跨对标准差 "
                        "2→4 从 0.0481 降到 0.0240，4→8 不再变（0.0241）")
    p.add_argument("--start", type=int, default=0,
                   help="从第几个场景开始（多进程切分用）。与 --limit 一起界定 [start, start+limit)")
    p.add_argument("--limit", type=int, default=None, help="最多跑几个场景（冒烟 / 切分用）")
    p.add_argument("--shard_size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--noop_tol", type=float, default=None,
                   help="空操作门容差；缺省 = CPU 0.0（逐位）/ GPU 1e-3（见 GPU_NOOP_TOL：由故障注入实测定）")
    p.add_argument("--interv_chunk", type=int, default=32,
                   help="一次前向最多塞几个干预副本。实测 n_veh 中位 26 / max 44 ⇒ "
                        "不分块时一批可到 50 个副本；分块把显存峰值钉住，且**块内作差**更干净")
    p.add_argument("--force", action="store_true", help="覆盖已冻结的产物目录")
    a = p.parse_args()

    global _ERR_FN
    _ERR_FN = ERR_FNS[a.err_metric]
    dev = torch.device(a.device)
    tol = a.noop_tol if a.noop_tol is not None else (0.0 if dev.type == "cpu" else GPU_NOOP_TOL)
    out = Path(a.out)
    if (out / FROZEN_MARK).exists() and not a.force:
        raise S3Error(f"{out} 已冻结（{FROZEN_MARK} 存在）—— 要重生成请显式 --force")
    out.mkdir(parents=True, exist_ok=True)

    shas, models = [], []
    for t in a.teacher:
        shas.append(hashlib.sha256(Path(t).read_bytes()).hexdigest())
        m = load_model_for_inference(Path(t), device=dev)
        if type(m).__name__ != "MagnetJointAdaptive":
            raise S3Error(f"teacher {t} 是 {type(m).__name__}，S3 只接受 MagnetJointAdaptive")
        models.append(m)
    if len(set(shas)) != len(shas):
        raise S3Error("teacher 里有重复的 ckpt —— 跨 seed 中位数会被同一份权重灌水，拒绝继续")

    from magnet_agent_aux_dataset import AgentAuxGraphDataset
    from joint_usage_probe import resolve_tokens
    from src.shared.utils.dataset import set_subgraph_agg
    set_subgraph_agg("mean")
    ds = AgentAuxGraphDataset(a.root, adjacency_config=a.adjacency_config)
    lo = max(0, a.start)
    hi = len(ds) if a.limit is None else min(lo + a.limit, len(ds))
    if lo >= hi:
        raise S3Error(f"场景区间为空：[{lo}, {hi})（数据集 {len(ds)} 个）")
    n = hi - lo
    logger.info("teacher×%d sha16=%s | %d 场景 | err=%s | placebo=%d | dev=%s | noop_tol=%.1e",
                len(models), ",".join(h[:16] for h in shas), n, a.err_metric, a.n_placebo, dev, tol)

    toks = resolve_tokens(a.root, len(ds))        # 真 scenario token（配对/聚类用）
    if len(toks) != len(ds):
        raise S3Error(f"token 数 {len(toks)} != 样本数 {len(ds)} —— 无法给标签配 provenance")

    # 🔴 **逐场景派生 RNG**，不是全局顺序消耗 —— 否则多进程切分时第 k 段的 placebo
    #    lane 抽样与不切分时不同，`I` 会变（`U` 不受影响，因为它不依赖 lane 抽样）。
    #    `default_rng([seed, i])` 让「第 i 个场景抽哪几条 lane」只与 (seed, i) 有关。
    def _rng_for(i: int) -> np.random.Generator:
        return np.random.default_rng([a.seed, i])
    shard: List[Dict] = []
    tokens: List[str] = []
    skipped, k, t0, noop_seen = [], 0, time.time(), 0.0
    for i in range(lo, hi):
        r = label_scene(models, ds[i], dev, _rng_for(i), a.n_placebo, tol, a.interv_chunk)
        if r is None or "skip" in r:
            skipped.append({"i": i, "why": r["skip"] if r else "None"})
        else:
            shard.append(r)
            tokens.append(toks[i])
            noop_seen = max(noop_seen, float(r["noop_max"]))
        if len(shard) >= a.shard_size:
            _flush(shard, tokens, out, k); k += 1; shard, tokens = [], []
        if (i - lo + 1) % 200 == 0:
            logger.info("  %d/%d  %.1f 场景/s  skip=%d", i - lo + 1, n,
                        (i - lo + 1) / (time.time() - t0), len(skipped))
    if shard:
        _flush(shard, tokens, out, k); k += 1

    meta = {
        "version": S3_VERSION, "n_teachers": len(models),
        "teachers": [str(Path(t).resolve()) for t in a.teacher], "teacher_sha256": shas,
        "teacher_aggregation": "median over teachers (U / S / placebo / err_base)" if len(models) > 1
        else "single teacher",
        "root": str(Path(a.root).resolve()), "adjacency_config": a.adjacency_config,
        "scene_range": [lo, hi], "n_dataset": len(ds),
        "n_scenes_requested": n, "n_scenes_labelled": n - len(skipped), "n_shards": k,
        "err_metric": a.err_metric, "dose": "D2c", "placebo": "P2c", "n_placebo": a.n_placebo,
        "seed": a.seed, "rng": "default_rng([seed, scene_index]) 逐场景派生 ⇒ 切分不变",
        "device": str(dev), "noop_tol": tol,
        "noop_max_observed": float(noop_seen), "interv_chunk": a.interv_chunk,
        "script_sha256": hashlib.sha256(_THIS.read_bytes()).hexdigest(),
        "caveats": [
            "U_{j→i} 来自**逐源 j** 的 D2c 干预（切 j 与全部车辆节点的边，保留 j–lane）"
            "⇒ 含 j→k→i 与 j→lane→i 的二阶通路，不是纯 j→i 的边际效用",
            "err = %s；GT 锚定（Q109 §3.3.2）。top1 口径会把 53–57%% 的模态翻转记成边效用" % a.err_metric,
            "placebo = P2c（随机 lane 节点对全部车辆切边），按 g1_dose_ladder_probe 的配对表；"
            "两侧 n_edges_cut 均已落盘，计数匹配度可事后审计",
            "placebo 是**逐场景**抽的（同一批 lane 对全部 j 共用）"
            "⇒ 实现的是 E_c U_{c→i} 而非 Q109 写的逐 (i,j) 控制池 E_{c~C_ij}；两者在本剂量下同义，"
            "因为 P2c 的控制池本来就不依赖 j",
            "空操作门容差由故障注入实测定：只切 1 条边的最小结构损坏读 0.222–7.875 m，"
            "GPU 浮点噪声 ≤1.53e-04 ⇒ 分离度 ≥1450×，GPU 档取 1e-3、CPU 档取 0.0（逐位）",
            "S（模型锚定，全模态口径）只作诊断，🔴 不得当监督",
            "多 teacher 时 U/S/placebo/err_base 全部取**跨 seed 中位数**；跨 teacher 离散度落在 "
            "`U_sd_teacher`。🔴 Q110 §6 字面写的是单个 teacher —— 改成三个是为了不让 S4/S5 隐含 seed 依赖",
            "🔴 标签重测信度（val 5 场景 2026 对冒烟实测）：placebo 抽样数不同 ⇒ "
            "rho(I) np4-vs-np8 = 0.87、I>0 符号一致 91.7%%。`I` 的**绝对值与 clamp 位置**受 "
            "placebo 抽样影响；但 placebo 是**逐目标的公共偏移** ⇒ **同一目标 i 内部 j 的排序不受影响**。"
            "S4 若只学 within-target 排序则不受此噪声支配",
            "数值噪底：复制批规模不同会让 U 差 7.2e-07（0.8%% 的对；err_base 逐位不变）—— "
            "是 BLAS 按形状选 kernel，不是批依赖；实测落在 clamp 边界 1e-6 内的边 **0 条** ⇒ 不翻任何标签",
            "未出现在产物里的边**不是零影响边** —— S4 必须靠 join 失败来 mask 掉，不得填 0",
        ],
        "skipped": skipped[:200], "n_skipped": len(skipped),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / FROZEN_MARK).write_text("\n".join(shas) + "\n", encoding="utf-8")
    if tol > 0 and noop_seen > 0.2 * tol:
        logger.warning("⚠️ 空操作门余量不足：实测 max %.2e vs 容差 %.1e（<5×）—— "
                       "GPU 前向不可复现（§8.2）；逐位完整性以 CPU 单测 [C][H] 为准", noop_seen, tol)
    logger.info("✅ 落盘 %s：%d 场景 / %d 分片 / skip %d / 空操作门实测 max %.2e / 用时 %.1f min",
                out, n - len(skipped), k, len(skipped), noop_seen, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
