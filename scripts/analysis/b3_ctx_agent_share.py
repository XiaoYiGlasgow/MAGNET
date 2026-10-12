#!/usr/bin/env python3
"""B3 —— 旧档 context 分支 75% 贡献方差里，agent（他车）信息占多少（欠账清单 §10.1 B3 / Q-d）。

估计量（描述性，不设判据）
--------------------------
在阶段113/A2′ 的旧档口径（s4b_magnet_L1/seed42 final、L1 val、batch=1 CPU、
gnn_pooling_fix=False、adjacency=default）上做「删光他车」干预：

* 干预 = 按 Q74 审计的实体映射（两道自检 + 空操作门）删掉 `data.x` 里全部
  type=='agent' 实体的行（ego 与 lane 保留；`gnn_x` 原封不动 ⇒ 只动 context 通路）；
* 读数 1（幅度）：逐场景 ‖ctx' − ctx‖ / ‖ctx‖ —— 与阶段114 §四 gnn 分支「删光 agent 变
  17.9%」同型，可并列；
* 读数 2（方差）：删光他车后 context 贡献（c_ctx = ctx·W_ctxᵀ）的场景间总方差变化
  1 − Var'/Var —— 即「ctx 那 75% 里有多大比例是 agent 行驱动的」；同报三分支占比重分解。

🔴 口径红线（写作时必须带）
---------------------------
- 这是「删光」剂量（一次删全部他车），与 Q74 G3 的 usage 6.46%（逐 agent、轨迹尺度）
  **不同 estimand 不同尺度，不得并列**；
- 单 ckpt（seed42）、旧档（default 契约 / L1 val），描述性读数不做区间；
- 方差分解忽略非线性与分支协方差，只作描述。

三道硬门（任一不过 exit 2）
---------------------------
1. base 配对门：前 50 token 重算 base，ctx/gnn 必须与 A2′ npz **逐位相等**；
   ego 容差 1e-4（已知 ego BatchNorm(track_running_stats=False) 有跨进程浮点抖动，全量实测最大 ~1e-5，
   冒烟实测 ctx/gnn=0.0、ego=4.8e-07；ego 最大偏差落盘）；
2. 空操作门：前 5 token 空删除集，ctx 必须与 base 逐位相等；
3. 不变性门：每个 token 删他车后 gnn 分支逐位相等、ego 容差 1e-4
   （删 data.x 行只应影响 context 通路）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/b3_ctx_agent_share.py \
        [--limit 0] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
BASE_NPZ = GNN / "experiments/branch_probe_20260818/oldtier_s4b_seed42.npz"
CKPT = GNN / "experiments/scaling_ladder/s4b_magnet_L1/seed42/models/final_model_default.pth"
RAW = GNN / "data/processed_features_scaling_L1_logsplit/val"
OUT_DIR = GNN / "experiments/b3_ctx_agent_20260819"

logger = logging.getLogger("b3")


def _agent_drop_idx(sample) -> np.ndarray:
    """待删集合 = 全部非 ego 的 agent/traj 节点（node_order 坐标系）。"""
    idx = [i for i, (t, e) in enumerate(zip(sample.node_types, sample.is_ego))
           if (not e) and str(t).lower().startswith(("agent", "traj"))]
    return np.asarray(idx, dtype=np.int64)


@torch.no_grad()
def _encoder_feats(model, data, device) -> np.ndarray:
    from torch_geometric.data import Batch
    return model.get_encoder_features(Batch.from_data_list([data]).to(device))[0].cpu().numpy()


def dist(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 token；0=全 2400")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")

    from scripts.analysis.q74_2x2_audit import (
        load_token_sample, _context_removal_overrides, configure_subgraph_agg)
    from scripts.analysis.run_interaction_audit import _clone_with
    from scripts.analysis.fusion_contribution_share import _fusion_weight, BLOCK
    from src.shared.utils.bn_calibration import load_model_for_inference

    base = np.load(BASE_NPZ, allow_pickle=True)
    tokens = [str(t) for t in base["tokens"]]
    if a.limit:
        tokens = tokens[: a.limit]
    ego_b, ctx_b, gnn_b = base["ego"], base["ctx"], base["gnn"]

    agg = configure_subgraph_agg(CKPT)
    model = load_model_for_inference(CKPT, gnn_pooling_fix=False, device=device)
    model.eval()
    logger.info("ckpt=%s  agg=%s  tokens=%d", CKPT.name, agg, len(tokens))

    w = _fusion_weight(CKPT)
    w_ctx = w[:, BLOCK["ctx"]]

    ctx_no = np.full_like(ctx_b[: len(tokens)], np.nan)
    n_agents = np.zeros(len(tokens), dtype=np.int64)
    skipped: dict = {}
    ego_max_dev = 0.0        # ego BatchNorm 抖动的实测上界（透明记录）
    EGO_ATOL = 1e-4
    t0 = time.time()

    for i, tok in enumerate(tokens):
        sample, reason = load_token_sample(RAW, tok, "default")
        if sample is None:
            skipped[tok] = reason
            continue

        # 门1：前 50 token base 配对 npz（ctx/gnn 逐位；ego 容差，见 docstring）
        if i < 50:
            f = _encoder_feats(model, sample.data, device)
            dev = float(np.abs(f[:128] - ego_b[i]).max())
            ego_max_dev = max(ego_max_dev, dev)
            if not (np.array_equal(f[128:256], ctx_b[i]) and np.array_equal(f[256:], gnn_b[i])
                    and dev < EGO_ATOL):
                logger.error("🔴 门1失败：token %s base 特征与 A2′ npz 配对不过（ego dev=%.2e）",
                             tok, dev)
                return 2
            # 门2：前 5 token 空删除集恒等
            if i < 5:
                ov0 = _context_removal_overrides(
                    sample.data, np.zeros(0, dtype=np.int64), sample.node_order, sample.ent_rows)
                f0 = _encoder_feats(model, _clone_with(sample.data, **ov0), device)
                if not np.array_equal(f0, f):
                    logger.error("🔴 门2失败：token %s 空删除集输出与 base 不逐位相等", tok)
                    return 2

        drop = _agent_drop_idx(sample)
        n_agents[i] = drop.size
        if drop.size == 0:
            ctx_no[i] = ctx_b[i]        # 无他车 ⇒ 干预是恒等（已由门2证明）
            continue
        ov = _context_removal_overrides(sample.data, drop, sample.node_order, sample.ent_rows)
        f = _encoder_feats(model, _clone_with(sample.data, **ov), device)
        # 门3：gnn 逐位不变、ego 容差内（删 data.x 行只应影响 context 通路）
        dev = float(np.abs(f[:128] - ego_b[i]).max())
        ego_max_dev = max(ego_max_dev, dev)
        if not (np.array_equal(f[256:], gnn_b[i]) and dev < EGO_ATOL):
            logger.error("🔴 门3失败：token %s 删他车后 gnn/ego 分支变了（ego dev=%.2e）", tok, dev)
            return 2
        ctx_no[i] = f[128:256]

        if (i + 1) % 300 == 0:
            logger.info("  %d/%d  用时 %.1f min", i + 1, len(tokens), (time.time() - t0) / 60)

    valid = ~np.isnan(ctx_no).any(axis=1)
    logger.info("有效 %d / %d（skip %d），用时 %.1f min",
                int(valid.sum()), len(tokens), len(skipped), (time.time() - t0) / 60)

    def readout(mask: np.ndarray, tag: str) -> dict:
        cb, cn, na = ctx_b[: len(tokens)][mask], ctx_no[mask], n_agents[mask]
        rel = np.linalg.norm(cn - cb, axis=1) / np.maximum(np.linalg.norm(cb, axis=1), 1e-12)
        c_base, c_no = cb @ w_ctx.T, cn @ w_ctx.T
        var_b = float(np.var(c_base, axis=0, ddof=1).sum())
        var_n = float(np.var(c_no, axis=0, ddof=1).sum())
        # 三分支贡献占比重分解（ctx 换成删他车版；ego/gnn 用 base）
        eb, gb = ego_b[: len(tokens)][mask], gnn_b[: len(tokens)][mask]
        var3 = {"ego": float(np.var(eb @ w[:, BLOCK["ego"]].T, axis=0, ddof=1).sum()),
                "ctx_noagent": var_n,
                "gnn": float(np.var(gb @ w[:, BLOCK["gnn"]].T, axis=0, ddof=1).sum())}
        s3 = sum(var3.values())
        return {
            "tag": tag, "n": int(mask.sum()),
            "n_agents_in_graph": {"median": float(np.median(na)), "mean": float(na.mean()),
                                  "zero_share": float((na == 0).mean())},
            "rel_change": {"all": dist(rel), "with_agents": dist(rel[na > 0]),
                           "by_count": {"0": dist(rel[na == 0]), "1": dist(rel[na == 1]),
                                        "2-3": dist(rel[(na >= 2) & (na <= 3)]),
                                        ">=4": dist(rel[na >= 4])}},
            "ctx_contribution_var": {"base": var_b, "noagent": var_n,
                                     "removed_share": 1.0 - var_n / var_b},
            "var_share_pct_noagent_redecomp": {k: 100.0 * v / s3 for k, v in var3.items()},
        }

    full = readout(valid, "limit0")
    m300 = valid.copy(); m300[300:] = False
    r300 = readout(m300, "limit300")

    out = {
        "protocol": {
            "ckpt": str(CKPT), "raw_dir": str(RAW), "base_npz": str(BASE_NPZ),
            "adjacency_config": "default", "gnn_pooling_fix": False, "batch": 1,
            "device": "cpu", "subgraph_agg": agg,
            "intervention": "删光全部非 ego 的 agent 实体的 data.x 行（context 通路侧；gnn_x 不动）",
            "gates": "门1 base 配对(前50, ctx/gnn 逐位+ego 1e-4)｜门2 空操作恒等(前5)｜"
                     "门3 gnn 逐位+ego 1e-4 不变性(逐 token)",
            "ego_bn_max_dev_observed": None,   # 运行结束时回填
        },
        "readouts": {"limit300": r300, "limit0": full},
        "ego_bn_max_dev_observed": ego_max_dev,
        "skipped": skipped,
        "caveats": [
            "「删光」剂量：与 Q74 G3 usage 6.46%（逐 agent、轨迹尺度）不同 estimand，不得并列",
            "与阶段114 §四 gnn 分支删光口径同型（gnn 输出变 17.945%），可并列",
            "单 ckpt seed42、旧档 default 契约、描述性读数不做区间",
            "方差分解忽略非线性与分支协方差；removed_share 只作描述",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "b3_ctx_agent_share_20260819.json"
    np.savez_compressed(OUT_DIR / "ctx_noagent_feats.npz",
                        tokens=np.array(tokens), ctx_noagent=ctx_no.astype(np.float32),
                        n_agents=n_agents, valid=valid, ckpt=str(CKPT))
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for tag, r in (("limit300", r300), ("limit0", full)):
        logger.info("[%s] rel_change(with_agents) median=%.3f%%  removed_var_share=%.2f%%  重分解=%s",
                    tag, 100 * r["rel_change"]["with_agents"]["median"],
                    100 * r["ctx_contribution_var"]["removed_share"],
                    {k: round(v, 2) for k, v in r["var_share_pct_noagent_redecomp"].items()})
    logger.info("✅ -> %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
