"""闭环「吸收态」三张论文图（0 GPU，只读已有闭环产物）。

图 1  速度剖面      —— 逐帧中位数速度（含 IQR 带），显示模型先正常行驶、随后永久停机
图 2  停机时刻 ECDF —— 「到 t 秒为止已永久停机的场景比例」，曲线终点 = 永久停机率
图 3  按 scenario_type 的停机率 —— dumbbell（calibrated vs IDM），按差值排序

数据源与口径（阶段㉙ / 预注册判读表）
------------------------------------
`ego_progress_along_expert_route.parquet` 的 `time_series_values` 是**每帧沿 expert route
的位移增量（米）**（求和 = ego_total_progress），**不是累计值**；速度 = 增量 / 0.1 s。
⚠️ 该字段是到 expert route 上的投影，个别帧会出现投影跳变（实测单场景最大值可达 663 m/s），
因此**只用中位数类统计量**，且图 1 的纵轴显式裁剪并在图注声明。

永久停机定义：存在帧 i，使从 i 到 rollout 结束速度始终 < 0.1 m/s。

用法：
    python3 scripts/analysis/plot_absorbing_state_20260731.py [--outdir <dir>]
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
CL = GNN / "experiments/closed_loop"

# 四条曲线：提交版模型 / BN 校准版（dev 诊断） / IDM / 专家回放
RUNS: Dict[str, Path] = {
    "MAGNET (submitted)": CL / "pittsburgh_heldout_N20/pittsburgh_heldout_gnn_trans_nonreactive_N20",
    "MAGNET (BN-calibrated)": CL / "g13a_20260730/calibrated/pittsburgh_heldout_gnn_trans_nonreactive_N20",
    "IDM": CL / "pittsburgh_heldout_N20/pittsburgh_heldout_idm_nonreactive_N20",
    "Log replay": CL / "pittsburgh_heldout_N20/pittsburgh_heldout_log_future_nonreactive_N20",
}

# Okabe-Ito 色盲安全配色 + 线型区分（灰度可读）
STYLE: Dict[str, Dict[str, object]] = {
    "MAGNET (submitted)":     {"color": "#D55E00", "ls": "-",  "lw": 1.6, "z": 4},
    "MAGNET (BN-calibrated)": {"color": "#E69F00", "ls": "--", "lw": 1.4, "z": 3},
    "IDM":                    {"color": "#0072B2", "ls": "-.", "lw": 1.4, "z": 2},
    "Log replay":             {"color": "#333333", "ls": ":",  "lw": 1.4, "z": 1},
}

# 图 3 的紧凑类型名（保持与 nuPlan 官方 scenario_type 一一对应，只缩短显示）
SHORT_TYPE: Dict[str, str] = {
    "starting_straight_traffic_light_intersection_traversal": "start straight at signal",
    "waiting_for_pedestrian_to_cross": "waiting for pedestrian",
    "high_lateral_acceleration": "high lateral accel.",
    "near_multiple_vehicles": "near multiple vehicles",
    "following_lane_with_lead": "following lane w/ lead",
    "stationary_in_traffic": "stationary in traffic",
    "high_magnitude_speed": "high speed",
    "low_magnitude_speed": "low speed",
    "behind_long_vehicle": "behind long vehicle",
    "stopping_with_lead": "stopping with lead",
    "starting_left_turn": "starting left turn",
    "starting_right_turn": "starting right turn",
    "changing_lane": "changing lane",
}

DT = 0.1
STOP_THR = 0.1
T_MAX = 15.0          # 画到 15 s（场景时长中位数 14.9 s）
Y_CLIP = 8.0          # 图 1 纵轴上限（投影跳变导致的尖峰不入图，图注说明）


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load(run_dir: Path) -> List[Tuple[str, np.ndarray]]:
    hits = sorted(glob.glob(f"{run_dir}/**/metrics/ego_progress_along_expert_route.parquet",
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"找不到闭环产物：{run_dir}")
    df = pd.read_parquet(hits[-1])
    out: List[Tuple[str, np.ndarray]] = []
    for _, r in df.iterrows():
        ts = r["time_series_values"]
        if ts is None or len(ts) < 100:
            continue
        out.append((r["scenario_type"], np.asarray(ts, dtype=float) / DT))
    return out


def stop_time(v: np.ndarray, thr: float = STOP_THR) -> Optional[float]:
    for i in range(len(v)):
        if np.all(np.abs(v[i:]) < thr):
            return i * DT
    return None


def frame_quantiles(rows: List[Tuple[str, np.ndarray]], q: float) -> Tuple[np.ndarray, np.ndarray]:
    """逐帧分位数；只保留仍有 ≥50% 场景在线的帧。"""
    n_max = int(T_MAX / DT)
    n_tot = len(rows)
    vals, times = [], []
    for i in range(n_max):
        col = [v[i] for _, v in rows if len(v) > i]
        if len(col) < 0.5 * n_tot:
            break
        vals.append(float(np.quantile(col, q)))
        times.append(i * DT)
    return np.asarray(times), np.asarray(vals)


def smooth(v: np.ndarray, w: int = 5) -> np.ndarray:
    """0.5 s 居中滑动中位数，仅用于可读性；不改变任何结论，图注须声明。"""
    if len(v) < w:
        return v
    pad = w // 2
    ext = np.concatenate([np.repeat(v[0], pad), v, np.repeat(v[-1], pad)])
    return np.asarray([float(np.median(ext[i:i + w])) for i in range(len(v))])


def fig_speed_profile(data: Dict[str, List[Tuple[str, np.ndarray]]], out: Path) -> None:
    # 只画逐帧中位数：IQR 带在四条曲线上互相遮挡（实测覆盖近满幅），信息量不敌噪声。
    # 分散度改在 README 与正文里以停机时刻分布（图 2）表达。
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for name, rows in data.items():
        s = STYLE[name]
        t, med = frame_quantiles(rows, 0.50)
        ax.plot(t, smooth(med), color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
                zorder=s["z"], label=name)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Speed along expert route (m/s)")
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, Y_CLIP)
    ax.axhline(STOP_THR, color="0.55", linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
    ax.text(T_MAX - 0.2, STOP_THR + 0.12, "standstill threshold 0.1 m/s",
            ha="right", va="bottom", fontsize=6, color="0.35")
    # 图例放到画布上方两列：曲线一直画到 14.9 s，任何框内图例都会压住 log replay 的尾段。
    ax.legend(frameon=False, ncol=2, handlelength=2.2, columnspacing=1.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.16), mode="expand")
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    logger.info("图 1 已写出：%s.{pdf,png}", out)


def fig_stop_time_ecdf(data: Dict[str, List[Tuple[str, np.ndarray]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for name, rows in data.items():
        s = STYLE[name]
        st = np.asarray([t for t in (stop_time(v) for _, v in rows) if t is not None])
        n = len(rows)
        grid = np.arange(0, T_MAX + DT, DT)
        frac = np.asarray([100.0 * np.sum(st <= g) / n for g in grid])
        ax.step(grid, frac, where="post", color=s["color"], linestyle=s["ls"],
                linewidth=s["lw"], zorder=s["z"], label=name)
        # 终值（= 永久停机率）标在曲线右端，而不是塞进图例标签里（会撑爆两列图例）
        ax.annotate(f"{frac[-1]:.1f}%", xy=(T_MAX, frac[-1]), xytext=(2.5, 0),
                    textcoords="offset points", va="center", fontsize=7, color=s["color"])
    ax.set_xlabel("Time of permanent stop (s)")
    ax.set_ylabel("Permanently stopped (%)")
    ax.set_xlim(0, T_MAX * 1.13)
    ax.set_ylim(0, 100)
    # 同图 1：图例出画布，否则白底图例会盖住提交版模型 5–12 s 的曲线段
    ax.legend(frameon=False, ncol=2, handlelength=2.2, columnspacing=1.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.16), mode="expand")
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    logger.info("图 2 已写出：%s.{pdf,png}", out)


def fig_per_type(data: Dict[str, List[Tuple[str, np.ndarray]]], out: Path) -> None:
    def rate_by_type(rows: List[Tuple[str, np.ndarray]]) -> Dict[str, float]:
        types = sorted({t for t, _ in rows})
        return {t: 100.0 * float(np.mean([stop_time(v) is not None
                                          for tt, v in rows if tt == t])) for t in types}

    cal = rate_by_type(data["MAGNET (BN-calibrated)"])
    idm = rate_by_type(data["IDM"])
    types = sorted(cal, key=lambda t: cal[t] - idm.get(t, np.nan))
    y = np.arange(len(types))

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    for i, t in enumerate(types):
        ax.plot([idm[t], cal[t]], [i, i], color="0.75", linewidth=1.0, zorder=1)
    ax.scatter([cal[t] for t in types], y, s=15, color=STYLE["MAGNET (BN-calibrated)"]["color"],
               marker="D", zorder=3, label="MAGNET (BN-calibrated)")
    # IDM 画空心圆并置于最上层：两者相等时（changing_lane 75% vs 75%）实心点会被完全遮住
    ax.scatter([idm[t] for t in types], y, s=17, facecolors="none", linewidths=1.0,
               edgecolors=STYLE["IDM"]["color"], marker="o", zorder=4, label="IDM")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_TYPE.get(t, t.replace("_", " ")) for t in types])
    ax.set_xlabel("Permanently stopped scenarios (%)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.7, len(types) - 0.3)
    ax.grid(axis="x", color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", scatterpoints=1, borderaxespad=0.3,
                    framealpha=0.9, edgecolor="none", facecolor="white")
    leg.get_frame().set_linewidth(0.0)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    logger.info("图 3 已写出：%s.{pdf,png}", out)


def main() -> int:
    ap = argparse.ArgumentParser(description="闭环吸收态三张论文图")
    ap.add_argument("--outdir", type=Path,
                    default=GNN / "docs/work_records/paper/MAGNET/absorbing_state_figures")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_style()
    data = {k: load(v) for k, v in RUNS.items()}
    for k, rows in data.items():
        st = [t for t in (stop_time(v) for _, v in rows) if t is not None]
        logger.info("%-24s n=%d  停机率=%.1f%%  停机时刻中位数=%.1fs",
                    k, len(rows), 100.0 * len(st) / len(rows),
                    float(np.median(st)) if st else float("nan"))

    fig_speed_profile(data, args.outdir / "fig_absorbing_a_speed_profile")
    fig_stop_time_ecdf(data, args.outdir / "fig_absorbing_b_stop_time_ecdf")
    fig_per_type(data, args.outdir / "fig_absorbing_c_stop_rate_by_type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
