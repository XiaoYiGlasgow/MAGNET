"""C4 判据 [J-real]：在**真实 nuPlan 地图**上核对可行驶区域闭包与官方 `is_in_layer` 的一致性。

为什么必须做这一步
------------------
`DrivableAreaClosure` 为了性能把官方的「逐点 `vector_layer.contains(Point)`」换成了
「取 patch → `union_all` → `shapely.contains_xy` 向量化」。语义上应当等价，但
**「应当等价」不是证据** —— 官方判的是「点落在某一个多边形内」，闭包判的是
「点落在并集内」，两者在相邻多边形的公共边界上不同（零测度集，方向偏宽松）。

本脚本在真地图上按网格采样，直接测两者的**逐点不一致率**与**加速比**，
把这条近似从「假设」变成「有数字的已知偏差」（写作时要报的就是这个数字）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python \\
        scripts/eval_closed_loop/validate_drivable_closure.py --map us-pa-pittsburgh-hazelwood
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/local/Xiao_Yi/nuplan-devkit")
sys.path.insert(0, "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")

from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

from src.planning_layer.nuplan_adapter import DrivableAreaClosure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="us-pa-pittsburgh-hazelwood")
    ap.add_argument("--version", default="nuplan-maps-v1.0")
    ap.add_argument("--grid", type=int, default=60, help="每边采样点数（总点数 = grid^2）")
    ap.add_argument("--span", type=float, default=100.0, help="采样正方形半边长（m）")
    args = ap.parse_args()

    maps_root = os.environ["NUPLAN_MAPS_ROOT"]
    map_api = get_maps_api(maps_root, args.version, args.map)

    layer = map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
    # 🔴 采样中心必须落在**真有 drivable 几何**的地方。
    #    第一版取图层 bbox 中心，结果落在空白区：官方全判 False、闭包按「patch 空 ⇒ 未知不罚」
    #    全判 True，得到 100% 不一致 —— 那测的是空白区兜底，不是两种判定的一致性。
    #    改成取**面积最大的那块** drivable 多边形的形心（一定是主干道/路口）。
    biggest = layer.geometry.iloc[int(np.argmax(layer.geometry.area.values))]
    cx, cy = float(biggest.centroid.x), float(biggest.centroid.y)
    print(f"地图 {args.map}：drivable 多边形 {len(layer)} 个，采样中心 ({cx:.1f}, {cy:.1f})")

    g = np.linspace(-args.span, args.span, args.grid)
    xx, yy = np.meshgrid(cx + g, cy + g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    closure = DrivableAreaClosure(map_api, patch_radius_m=args.span + 50.0)
    t0 = time.time()
    closure.update((cx, cy))
    t_build = time.time() - t0
    t0 = time.time()
    mine = closure(pts)
    t_mine = time.time() - t0

    t0 = time.time()
    official = np.array([map_api.is_in_layer(Point2D(x, y), SemanticMapLayer.DRIVABLE_AREA)
                         for x, y in pts], dtype=bool)
    t_official = time.time() - t0

    n = len(pts)
    mismatch = int(np.sum(mine != official))
    only_mine = int(np.sum(mine & ~official))
    only_off = int(np.sum(~mine & official))
    print(f"采样点 {n} 个｜官方判为可行驶 {int(official.sum())}｜闭包判为可行驶 {int(mine.sum())}")
    print(f"不一致 {mismatch} 点（{100.0 * mismatch / n:.4f}%）"
          f"：闭包宽松 {only_mine}、闭包严格 {only_off}")
    print(f"耗时：闭包建 patch {t_build * 1e3:.1f} ms + 查询 {t_mine * 1e3:.1f} ms"
          f"｜官方逐点 {t_official * 1e3:.1f} ms ⇒ 查询加速 {t_official / max(t_mine, 1e-9):.0f}×")

    # 判据：闭包**不得比官方更严格**（那会制造假的越界惩罚）；宽松侧只允许零测度级别
    ok = only_off == 0 and (only_mine / n) < 0.01
    print("判据 [J-real]", "PASS" if ok else "FAIL",
          "—— 闭包不得比官方严格；宽松侧 < 1%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
