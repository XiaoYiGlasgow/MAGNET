#!/usr/bin/env python3
"""旋钮 A §4 描述性读数（🔴 只报不判；判据 = 旋钮A_时间池化聚合对照_预注册判据_20260810.md §4）。

双格共用（a_max / a_sum，各 vs fr_base）：

(a) 训练后 `gnn_feat` 场景间方差 tr(Cov)——口径同阶段74 `branch_feature_probe.py`：
    `get_encoder_features` 输出 [1,384] 的第 256:384 段跨场景方差之和；同报 ctx 段
    （128:256）作参照分母，连接「gnn 方差 0.557 = ctx 的 1/154 ⇒ 近似常数」诊断线。
(b) 建图侧 agent 节点坐标维幅度分布（max vs mean，64 val 样本，model-free）——
    坐标维 = 12 维池化向量的 col0-3（x_start/y_start/x_end/y_end，
    `data_utils.py:838-841`）；量化「保峰值」的实际幅度；sum 仅作附注。

事前声明（见任何数字之前写死，0812 提示词 ③）：
- 只报不判；若 max/sum 抬方差而使用度不动，只能写「瓶颈进一步指向读出/融合/训练信号」，
  不得升级判定；
- 口径 = 新档 `processed_features_fullret_L1_logsplit/val` 前 64（sorted）、
  **batch=1 逐样本前向**（未校准 batch=1 口径，不得与 batch=8 训练读数并列）、
  `best_model_default.pth`（与使用度审计同一 ckpt 口径）；
- 建图 `subgraph_agg` 必须与被读 ckpt 训练所见一致（sidecar 为准，缺键回退 mean，
  同 `q74_2x2_audit.configure_subgraph_agg`）；期望值硬断言（fr_base=mean/a_max=max/
  a_sum=sum），防拿错 ckpt 静默产出。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python \
        scripts/analysis/a_knob_descriptive_readout_20260812.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN), str(_GNN.parent), str(_GNN.parent / "models"),
           str(_GNN / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.run_interaction_audit import (  # noqa: E402
    TARGET_TYPE,
    _load_feature_df,
    _node_types_from_df,
)
from src.shared.utils.bn_calibration import (  # noqa: E402
    load_model_for_inference,
    read_architecture,
)
from src.shared.utils.dataset import (  # noqa: E402
    build_graph_data_from_feature_df,
    set_subgraph_agg,
)

logger = logging.getLogger("a_knob_descriptive")

VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
OUT = _GNN / "scratchpad/a_knob_descriptive_readout_20260812.json"
N_SAMPLES = 64
COORD_DIMS = (0, 1, 2, 3)          # x_start, y_start, x_end, y_end
SEEDS = (42, 1, 2)

#: 三 config 及其期望 agg（sidecar 自证的硬断言值）与 ckpt 根。
CONFIGS: Dict[str, Tuple[str, Path]] = {
    "fr_base": ("mean", _GNN / "experiments/knobs/fr_base_L1"),
    "a_max": ("max", _GNN / "experiments/knobs/a_max_L1"),
    "a_sum": ("sum", _GNN / "experiments/knobs/a_sum_L1"),
}


def load_samples(agg: str) -> List[Tuple[str, object, np.ndarray]]:
    """按指定 agg 重建 val 前 64 个样本；返回 (token, data, agent非ego行掩码) 列表。

    掩码在 GNN 节点顺序上（`gnn_x` 行序 = node_order，行数不匹配即跳过——不猜对应）。
    """
    set_subgraph_agg(agg)
    pkls = sorted(VAL.glob("features_*.pkl"))[:N_SAMPLES]
    if len(pkls) < N_SAMPLES:
        raise RuntimeError(f"val 只有 {len(pkls)} 个 pkl，读数要求前 {N_SAMPLES}")
    out: List[Tuple[str, object, np.ndarray]] = []
    for pkl in pkls:
        df = _load_feature_df(pkl)
        if df is None:
            continue
        parsed = _node_types_from_df(df)
        if parsed is None:
            continue
        node_types, is_ego, _order = parsed
        data, _reason = build_graph_data_from_feature_df(df, "default")
        if data is None or data.gnn_x.shape[0] != len(node_types):
            continue
        mask = (np.array(node_types) == TARGET_TYPE) & ~is_ego
        out.append((pkl.stem.replace("features_", ""), data, mask))
    if len(out) < N_SAMPLES:
        logger.warning("可用样本 %d/%d（其余建图失败被跳过，三种 agg 同一跳过集）",
                       len(out), N_SAMPLES)
    return out


@torch.no_grad()
def gnn_feat_variance(ckpt: Path, expect_agg: str,
                      device: torch.device) -> Dict[str, float]:
    """(a) 单 ckpt：batch=1 逐样本取 encoder 三段，报 gnn/ctx 段跨场景 tr(Cov)。"""
    from torch_geometric.data import Batch

    arch = read_architecture(ckpt) or {}
    agg = arch.get("subgraph_agg") or "mean"
    if agg != expect_agg:
        raise RuntimeError(f"{ckpt} sidecar agg={agg!r}，期望 {expect_agg!r} —— 拿错 ckpt")
    fix = bool(arch.get("gnn_pooling_fix", False))
    model = load_model_for_inference(ckpt, gnn_pooling_fix=fix, device=device)
    model.eval()

    samples = load_samples(agg)
    gnn_rows, ctx_rows = [], []
    for _token, data, _mask in samples:
        batch = Batch.from_data_list([data]).to(device)
        feats = model.get_encoder_features(batch)[0].cpu().numpy()   # [384]
        ctx_rows.append(feats[128:256])
        gnn_rows.append(feats[256:])
    gnn_mat = np.stack(gnn_rows)     # [n, 128]
    ctx_mat = np.stack(ctx_rows)
    var_gnn = float(gnn_mat.var(axis=0, ddof=1).sum())
    var_ctx = float(ctx_mat.var(axis=0, ddof=1).sum())
    return {
        "model_class": type(model).__name__,
        "gnn_pooling_fix(sidecar)": fix,
        "subgraph_agg(sidecar)": agg,
        "n_scenes": len(samples),
        "trCov_gnn_feat": var_gnn,
        "trCov_ctx_feat": var_ctx,
        "ctx_over_gnn": var_ctx / var_gnn if var_gnn > 0 else float("inf"),
    }


def coord_amplitude(agg: str) -> Dict[str, object]:
    """(b) 单 agg：agent（非 ego）节点坐标维 |值| 分布统计（model-free）。"""
    vals: List[np.ndarray] = []
    n_nodes = 0
    for _token, data, mask in load_samples(agg):
        rows = data.gnn_x[torch.as_tensor(mask)][:, list(COORD_DIMS)]
        if rows.numel() == 0:
            continue
        n_nodes += int(rows.shape[0])
        vals.append(rows.abs().flatten().numpy())
    v = np.concatenate(vals)
    return {
        "n_agent_nodes": n_nodes,
        "n_values(节点×4维)": int(v.size),
        "abs_median": float(np.median(v)),
        "abs_mean": float(v.mean()),
        "abs_p90": float(np.percentile(v, 90)),
        "abs_max": float(v.max()),
        "raw_values": v,               # 落盘前删掉，只用于配对比值
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="旋钮 A §4 描述性读数（只报不判）")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)

    # ---- (a) gnn_feat 场景间方差：3 config × 3 seed ----
    var_report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cfg, (agg, root) in CONFIGS.items():
        var_report[cfg] = {}
        for sd in SEEDS:
            ckpt = root / f"seed{sd}" / "models" / "best_model_default.pth"
            var_report[cfg][f"seed{sd}"] = gnn_feat_variance(ckpt, agg, device)
            logger.info("(a) %s seed%s: trCov_gnn=%.4g ctx/gnn=%.1f", cfg, sd,
                        var_report[cfg][f"seed{sd}"]["trCov_gnn_feat"],
                        var_report[cfg][f"seed{sd}"]["ctx_over_gnn"])

    # 同 seed 比值（config / fr_base）+ 三 seed pooled（几何均值，方差为正数天然对数尺度）
    ratios: Dict[str, Dict[str, float]] = {}
    for cfg in ("a_max", "a_sum"):
        per_seed = {
            f"seed{sd}": var_report[cfg][f"seed{sd}"]["trCov_gnn_feat"]
            / var_report["fr_base"][f"seed{sd}"]["trCov_gnn_feat"]
            for sd in SEEDS
        }
        vals = np.array(list(per_seed.values()))
        ratios[cfg] = {**per_seed, "pooled(geomean)": float(np.exp(np.log(vals).mean()))}

    # ---- (b) 坐标维幅度分布：mean / max（判据本体）+ sum（附注）----
    amp = {agg: coord_amplitude(agg) for agg in ("mean", "max", "sum")}
    # 逐值配对（三种 agg 的节点集与顺序相同——agg 只改池化值不改节点簿记）
    m_raw, x_raw = amp["mean"].pop("raw_values"), amp["max"].pop("raw_values")
    amp["sum"].pop("raw_values")
    ok = np.abs(m_raw) > 1e-6
    paired = {
        "median_of_ratios(|max|/|mean|)": float(np.median(x_raw[ok] / m_raw[ok])),
        "n_paired": int(ok.sum()),
        "n_excluded(|mean|<=1e-6)": int((~ok).sum()),
    }

    report = {
        "判据": "旋钮A_时间池化聚合对照_预注册判据_20260810.md §4（只报不判）",
        "口径": "新档 fullret_L1_logsplit val 前 64（sorted）；batch=1 逐样本（未校准口径，"
              "不得与 batch=8 训练读数并列）；best_model_default.pth；建图 agg=sidecar；"
              f"device={a.device}",
        "诊断线锚": "阶段74：B_seed42 gnn_feat 场景间方差 0.557 = ctx 的 1/154（旧档、旧特征，"
                 "只作方向参照不并列绝对值）",
        "(a)_gnn_feat场景间方差": var_report,
        "(a)_方差比_vs_fr_base(同seed配对)": ratios,
        "(b)_agent节点坐标维幅度(|col0-3|)": amp,
        "(b)_配对倍数(max_vs_mean)": paired,
        "解释限制": "若 max/sum 抬方差而使用度不动，只能写「瓶颈进一步指向读出/融合/训练信号」，"
                "不得升级判定；sum 幅度仅作附注（判据本体只写 max vs mean）",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n===== (a) gnn_feat 场景间方差 tr(Cov)（只报不判）=====")
    for cfg in CONFIGS:
        row = " / ".join(f"seed{sd} {var_report[cfg][f'seed{sd}']['trCov_gnn_feat']:.4g}"
                         f"(ctx/gnn {var_report[cfg][f'seed{sd}']['ctx_over_gnn']:.0f}×)"
                         for sd in SEEDS)
        print(f"  {cfg:8s}: {row}")
    for cfg, r in ratios.items():
        print(f"  {cfg}/fr_base 方差比: " + " / ".join(
            f"{k} {v:.3f}" for k, v in r.items()))
    print("===== (b) agent 节点坐标维 |值| 分布 =====")
    for agg, s in amp.items():
        print(f"  {agg:5s}: median {s['abs_median']:.3f} | mean {s['abs_mean']:.3f} | "
              f"p90 {s['abs_p90']:.3f} | max {s['abs_max']:.1f}"
              + ("（附注）" if agg == "sum" else ""))
    print(f"  配对 median(|max|/|mean|) = {paired['median_of_ratios(|max|/|mean|)']:.3f}"
          f"（n={paired['n_paired']}，剔除 |mean|≤1e-6 共 {paired['n_excluded(|mean|<=1e-6)']}）")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
