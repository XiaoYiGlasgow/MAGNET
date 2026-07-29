"""benchmark 网格可视化：把一个 benchmark 目录里的全部场景画成网格图，并打印数值复核。

每格（一个保留场景）：ego/agent 的 GT(虚线) + 生成(实线) + 历史(点线) + 3m 路缓冲 + 车道中心线
+ agent 起点(★) + ego 起点(●)。碰撞场景（gen 两车最近 ≤2m）显示端截断于碰撞帧，撞击点画红 X + 2m 碰撞圈，
标题转红。标题附 ego/agent offroad@3m、seed 的相对朝向与最小侧向距离，用于核对「同向 + 同道走廊」是否生效。

用法：
  python scripts/analysis/viz_benchmark_grid.py \
      --benchmark_dir experiments/danger_generation/benchmark/v7_speed_corridor \
      --out docs/figures/benchmark/v7_speed_corridor_grid.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import LineString
from shapely.ops import unary_union

ROAD_HALF_WIDTH = 3.0
COLLISION_DISTANCE = 2.0   # 与 danger_score / benchmark 一致


def _draw_road(ax, lane):
    lines = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in lane
             if np.abs(s).sum() > 1e-6 and (s[0] != s[2] or s[1] != s[3])]
    if lines:
        road = unary_union([ln.buffer(ROAD_HALF_WIDTH, cap_style=2) for ln in lines])
        for g in (road.geoms if road.geom_type == "MultiPolygon" else [road]):
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color="#bfe3c0", alpha=0.4, ec="#7fb98a", lw=0.4, zorder=1)
    for s in lane:
        if np.abs(s).sum() > 1e-6:
            ax.plot([s[0], s[2]], [s[1], s[3]], color="#9e9e9e", lw=0.4, zorder=2)


def _nearest_lane_dist(pts, segs):
    if len(segs) == 0:
        return np.full(len(pts), np.nan)
    a = segs[None, :, :2]; b = segs[None, :, 2:]; p = pts[:, None, :]; ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clip(1e-8), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1).min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_dir", required=True)
    ap.add_argument("--out", default="/tmp/benchmark_grid.png")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    fs = sorted(glob.glob(f"{args.benchmark_dir}/*.pkl"))
    if not fs:
        print(f"[空] {args.benchmark_dir} 无 pkl"); return
    recs = [pd.read_pickle(f) for f in fs]
    recs.sort(key=lambda r: r["metrics"]["min_dist"])

    n = len(recs); cols = args.cols; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows), squeeze=False)
    off_ego, off_ag, heads, lats = [], [], [], []
    for i, d in enumerate(recs):
        ax = axes[i // cols, i % cols]
        lane = np.asarray(d["lane_polylines"]); segs = lane[np.abs(lane).sum(1) > 0]
        eg = d["ego_future_gt"][:, :2]; ag = d["agent_future_gt"][:, :2]
        ge = d["ego_future_gen"]; ga = d["agent_future_gen"]
        eh = np.asarray(d["ego_history"])[:, :2]; ah = np.asarray(d["agent_history"])[:, :2]
        _draw_road(ax, segs)
        ax.plot(eg[:, 0], eg[:, 1], "--", color="#08306b", lw=1.6, label="ego GT", zorder=4)
        ax.plot(ag[:, 0], ag[:, 1], "--", color="#7f2704", lw=1.6, label="agent GT", zorder=4)
        # 碰撞判别：gen 两车最近点 ≤2m → 碰撞，显示端截断于碰撞帧并标记撞击点
        # （兼容 v2 全长 pkl 与 v3 已截断 pkl：已截断时 argmin=末帧，截断为 no-op）
        ge = np.asarray(ge); ga = np.asarray(ga)
        gdist = np.linalg.norm(ge[:, :2] - ga[:, :2], axis=1)
        is_coll = float(gdist.min()) <= COLLISION_DISTANCE
        cf = int(gdist.argmin())
        ge_d = ge[:cf + 1] if is_coll else ge
        ga_d = ga[:cf + 1] if is_coll else ga
        ax.plot(ge_d[:, 0], ge_d[:, 1], color="#2c7fb8", lw=1.3, alpha=.8, label="ego gen", zorder=3)
        ax.plot(ga_d[:, 0], ga_d[:, 1], color="#f16913", lw=1.3, alpha=.8, label="agent gen", zorder=3)
        ax.plot(eh[:, 0], eh[:, 1], ":", color="#08306b", lw=1.0, alpha=.6, zorder=3)  # ego 历史
        ax.plot(ah[:, 0], ah[:, 1], ":", color="#7f2704", lw=1.0, alpha=.6, zorder=3)  # agent 历史
        ax.scatter(*eh[-1], c="#08306b", s=50, marker="o", ec="white", lw=.8, zorder=6)
        ax.scatter(*ah[-1], c="#7f2704", s=70, marker="*", ec="white", lw=.8, zorder=6, label="agent start")
        if is_coll:
            cp = (ge[cf, :2] + ga[cf, :2]) / 2          # 撞击点（两车质心中点）
            ax.add_patch(Circle(cp, COLLISION_DISTANCE, color="red", alpha=0.12, zorder=2))
            ax.scatter(*cp, c="red", marker="X", s=190, ec="black", lw=1.2, zorder=10, label="collision")
        eo = (_nearest_lane_dist(eg, segs) > 3).mean() * 100
        ao = (_nearest_lane_dist(ag, segs) > 3).mean() * 100
        off_ego.append(eo); off_ag.append(ao)
        sg = d.get("seed_geom", {})
        rh = sg.get("rel_heading_deg", float("nan")); lat = sg.get("min_lateral_m", float("nan"))
        heads.append(rh); lats.append(lat)
        allp = np.vstack([eg, ag, ge_d[:, :2], ga_d[:, :2], eh, ah]); cx, cy = allp.mean(0)
        rad = max(allp.ptp(0).max() / 2, 12) + 4
        ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad); ax.set_aspect("equal")
        tag = "COLLISION" if is_coll else "near-miss"
        ax.set_title(f"{d['scenario_name'][:14]} | {tag} {d['metrics']['min_dist']:.1f}m\n"
                     f"agent off {ao:.0f}% | head {rh:.0f}deg | lat {lat:.1f}m",
                     fontsize=8, color="darkred" if is_coll else "black")
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6.5, loc="best")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    n_coll = sum(1 for d in recs if d["metrics"]["min_dist"] <= COLLISION_DISTANCE)
    fig.suptitle(f"benchmark [{Path(args.benchmark_dir).name}] | N={n} (collision {n_coll}) | "
                 f"GT(dashed)+gen(solid)+hist(dotted) | red X=collision (gen truncated at impact)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight"); plt.close()

    print(f"[saved] {args.out}  (N={n})")
    print(f"数值复核：agent offroad@3m mean={np.nanmean(off_ag):.0f}% (旧=64%) | "
          f"ego offroad mean={np.nanmean(off_ego):.0f}%")
    print(f"  seed 相对朝向 max={np.nanmax(heads):.0f}° (应≤45) | seed 最小侧向 max={np.nanmax(lats):.1f}m (应≤3.5)")
    print(f"  agent off>50% 的场景数: {sum(1 for x in off_ag if x > 50)}/{n} (旧=8/12)")
    print("VIZ_GRID_DONE")


if __name__ == "__main__":
    main()
