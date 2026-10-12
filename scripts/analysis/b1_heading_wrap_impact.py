#!/usr/bin/env python3
"""B1 —— heading 环绕错误（D-3, 4.27%）的影响量：sin/cos（circular mean）重算对比。

背景（欠账清单 §10.1 B1；出处 = progress_20260806 阶段114）
----------------------------------------------------------
`utils/dataset.py::subgraph_pooling` 用 `np.mean` 对 12 维特征逐列平均，
其中 col4/col5 是 heading_start/heading_end（弧度，(-π,π]）。角度算术平均在
环绕点附近是错的（359° 与 1° 平均得 180°）。阶段114 只登记了受影响占比
**4.27%（35/819 agent 子图，300 场景）**，影响量「尚未评估」。

本脚本做的事（纯描述性诊断，不设判据、不下结论）
------------------------------------------------
1. 同口径复现阶段114 的分母：L1 default 档 val（sorted）前 300 个 pkl 的全部
   type=='agent' 子图。🔴 回归门：agent 子图数必须 = 819 且 wrap 子图数 = 35，
   不符则打印差异并 exit 2（不得静默换口径读数）。
2. 对每个 agent 子图、col4 与 col5 各算：
   - 算术平均（= 现行 pipeline 落盘值）；
   - circular mean = atan2(mean(sin), mean(cos))；
   - 角误差 e = |wrap_to_pi(算术 − circular)|。
3. 落盘分布读数（全体 / wrap 子集），单位弧度+角度，含描述性分箱。

口径
----
- wrap 判定沿用阶段114：子图内相邻行 heading 跳变 |diff| > π（col4）。
- ego 子图同缺陷但不在 4.27% 分母内，单独一节 `extra_ego` 只报不并。
- 本读数在旧档（default 契约 / L1 val）上，与阶段114 同口径；不外推到 fullret。

用法：/home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/b1_heading_wrap_impact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
VAL = GNN / "data/processed_features_scaling_L1_logsplit/val"
OUT = GNN / "scratchpad/b1_heading_wrap_impact_20260819.json"

N_SCENES = 300           # 阶段114 口径
EXPECT_AGENT_SUBGRAPHS = 819
EXPECT_WRAPPED = 35
COL_H_START, COL_H_END = 4, 5


def wrap_to_pi(a: np.ndarray | float) -> np.ndarray | float:
    """把角差折回 (-π, π]。"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def circular_mean(h: np.ndarray) -> float:
    return float(np.arctan2(np.sin(h).mean(), np.cos(h).mean()))


def dist_stats(errs: np.ndarray) -> dict:
    """弧度误差数组 → 描述性分布（弧度 + 角度 + 分箱计数）。"""
    if errs.size == 0:
        return {"n": 0}
    deg = np.degrees(errs)
    return {
        "n": int(errs.size),
        "median_rad": float(np.median(errs)),
        "p90_rad": float(np.percentile(errs, 90)),
        "p99_rad": float(np.percentile(errs, 99)),
        "max_rad": float(errs.max()),
        "mean_rad": float(errs.mean()),
        "median_deg": float(np.median(deg)),
        "p90_deg": float(np.percentile(deg, 90)),
        "max_deg": float(deg.max()),
        "count_gt_10deg": int((deg > 10).sum()),
        "count_gt_45deg": int((deg > 45).sum()),
        "count_gt_90deg": int((deg > 90).sum()),
        "count_gt_170deg": int((deg > 170).sum()),
    }


def main() -> int:
    pkls = sorted(VAL.glob("*.pkl"))[:N_SCENES]
    if len(pkls) < N_SCENES:
        print(f"❌ val pkl 不足 {N_SCENES}：{len(pkls)}")
        return 1

    rows = []          # 每个 agent 子图一行
    ego_rows = []      # ego 子图（分母外，另报）
    for p in pkls:
        df = pd.read_pickle(p)
        r = df.iloc[0]
        feat = r["POLYLINE_FEATURES"]
        idx = r["SUBGRAPH_ID_TO_NODE_INDICES"]
        typ = r["SUBGRAPH_ID_TO_TYPE"]
        for sid, (s, e) in idx.items():
            t = typ.get(sid)
            if t not in ("agent", "ego") or e < s or s >= len(feat):
                continue
            seg = feat[s : e + 1]
            if len(seg) == 0:
                continue
            rec = {"token": p.stem, "sid": int(sid)}
            for name, col in (("h4", COL_H_START), ("h5", COL_H_END)):
                h = seg[:, col].astype(np.float64)
                am = float(h.mean())                       # 现行 pipeline 的落盘值
                cm = circular_mean(h)
                rec[f"{name}_err"] = float(abs(wrap_to_pi(am - cm)))
                rec[f"{name}_wrapped"] = bool(np.any(np.abs(np.diff(h)) > np.pi))
                rec[f"{name}_span"] = float(h.max() - h.min())
            (ego_rows if t == "ego" else rows).append(rec)

    n_agent = len(rows)
    n_wrapped = sum(1 for r in rows if r["h4_wrapped"])
    print(f"agent 子图 {n_agent}（期望 {EXPECT_AGENT_SUBGRAPHS}）；"
          f"wrap(col4) {n_wrapped}（期望 {EXPECT_WRAPPED}）")

    # 🔴 回归门：分母与 wrap 数必须复现阶段114，否则不读数
    if n_agent != EXPECT_AGENT_SUBGRAPHS or n_wrapped != EXPECT_WRAPPED:
        alt = {
            "wrap_col5": sum(1 for r in rows if r["h5_wrapped"]),
            "wrap_col4_or_col5": sum(1 for r in rows if r["h4_wrapped"] or r["h5_wrapped"]),
        }
        print(f"❌ 回归门不过，阶段114 口径未复现；其它 wrap 变体计数 = {alt}")
        return 2

    e4_all = np.array([r["h4_err"] for r in rows])
    e5_all = np.array([r["h5_err"] for r in rows])
    e4_wrap = np.array([r["h4_err"] for r in rows if r["h4_wrapped"]])
    e5_wrap = np.array([r["h5_err"] for r in rows if r["h4_wrapped"]])
    e4_clean = np.array([r["h4_err"] for r in rows if not r["h4_wrapped"]])

    out = {
        "protocol": {
            "data": str(VAL),
            "n_scenes": N_SCENES,
            "scene_selection": "sorted(*.pkl)[:300]（阶段114 口径）",
            "heading_cols": {"h4": "heading_start(col4)", "h5": "heading_end(col5)"},
            "wrap_criterion": "子图内相邻行 |diff(heading_col4)| > π（阶段114 同款）",
            "error_metric": "|wrap_to_pi(算术平均 − circular_mean)|",
            "regression_gate": f"{EXPECT_AGENT_SUBGRAPHS} agent 子图 / {EXPECT_WRAPPED} wrapped = PASS",
        },
        "counts": {
            "agent_subgraphs": n_agent,
            "wrapped_col4": n_wrapped,
            "wrapped_share": n_wrapped / n_agent,
            "wrapped_col5": int(sum(1 for r in rows if r["h5_wrapped"])),
            "wrapped_col4_or_col5": int(sum(1 for r in rows if r["h4_wrapped"] or r["h5_wrapped"])),
        },
        "heading_start_col4": {
            "all_agent": dist_stats(e4_all),
            "wrapped_subset": dist_stats(e4_wrap),
            "non_wrapped_subset": dist_stats(e4_clean),
        },
        "heading_end_col5": {
            "all_agent": dist_stats(e5_all),
            "wrapped_subset_by_col4_flag": dist_stats(e5_wrap),
        },
        "extra_ego": {
            "note": "ego 子图不在 4.27% 分母内，仅描述性另报",
            "n": len(ego_rows),
            "wrapped_col4": int(sum(1 for r in ego_rows if r["h4_wrapped"])),
            "err_col4": dist_stats(np.array([r["h4_err"] for r in ego_rows])),
        },
        "wrapped_cases_top": sorted(
            ({k: r[k] for k in ("token", "sid", "h4_err", "h4_span")}
             for r in rows if r["h4_wrapped"]),
            key=lambda x: -x["h4_err"])[:10],
        "caveats": [
            "纯描述性诊断：不设阈值判据，不作『显著/不显著』宣称",
            "旧档口径（default 契约 / L1 val / 300 场景），不外推 full_retention",
            "误差是特征层角误差，不是模型输出层影响；后者未测",
            "wrap 子集外的子图也可能有小角误差（算术平均对分散角度本就有偏），已分列",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"✅ 回归门 PASS；已落盘 {OUT}")
    for sec in ("heading_start_col4",):
        s = out[sec]
        print(f"[{sec}] wrap 子集 n={s['wrapped_subset']['n']}: "
              f"median {s['wrapped_subset']['median_deg']:.1f}° / max {s['wrapped_subset']['max_deg']:.1f}°；"
              f"全体 median {s['all_agent']['median_deg']:.3f}° / >45° 共 {s['all_agent']['count_gt_45deg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
