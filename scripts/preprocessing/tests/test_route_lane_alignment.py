"""route ↔ lane 顺序对齐测试（VT-2026-04191 阶段⑬ P0 回归）。

事故背景：`build_route_channels()` 曾按**原始 lane_ids 顺序**产出 route 三通道，而生效副本
`GNN_Trans/src/shared/utils/data_utils.py::convert_feature_layer_to_fixed_size()` 随后按
**lane 到 ego 的最小距离重排并截断**，两者未共享 permutation → route 标签写到错误 lane。
实测 20 scenario 中 14 个重排、479 lane 行中 38 行 `is_on_route` 错位。

本测试是该 bug 的永久回归护栏，包含两部分：

  A. synthetic：构造原始顺序与距离顺序**故意不同**的 lane 集合，验证重排后
     lane ID / 几何 / 三个 route channel 完全对应。
  B. real：在 ≥100 个真实 nuPlan scenario 上验证
     `最终第 i 条 lane 的 route 标签 == (该 lane 的 id ∈ route_lane_progress)`，
     **允许的 mismatch 数为 0**。

运行：
  cd /mnt/local/Xiao_Yi/nuplan-devkit
  PYTHONPATH=GNN_Trans:. /home/team/anaconda3/envs/nuplan/bin/python \
      GNN_Trans/scripts/preprocessing/tests/test_route_lane_alignment.py [--n_scenarios 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parents[3]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from src.shared.utils.data_utils import (  # noqa: E402
    convert_feature_layer_to_fixed_size,
    build_route_channels,
    reorder_lane_ids,
    get_route_lane_progress,
    get_neighbor_vector_set_map,
    map_process,
)

DATASET_ROOT = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset"
DATA = f"{DATASET_ROOT}/nuplan-v1.1/splits/train_pittsburgh"


# ---------------------------------------------------------------- A. synthetic
def test_synthetic_misorder() -> bool:
    """原始顺序 != 距离顺序，验证 selected_indices 能把三通道正确对齐。"""
    print("=" * 70)
    print("A. synthetic：原始顺序与距离顺序故意不同")
    print("=" * 70)

    # ego 在原点。5 条 lane，**原始顺序刻意按距离倒序排列**（最远的排最前）
    dists = [40.0, 30.0, 20.0, 10.0, 5.0]
    lane_ids = [f"lane_{i}" for i in range(5)]
    feature_coords = [
        torch.tensor([[d, 0.0], [d + 1.0, 0.0], [d + 2.0, 0.0]], dtype=torch.float32)
        for d in dists
    ]
    ego_pose = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    max_elements, max_points = 4, 3   # 故意 < 5，同时触发截断
    coords_t, _tl, avails, sel = convert_feature_layer_to_fixed_size(
        ego_pose, feature_coords, None, max_elements, max_points,
        traffic_light_encoding_dim=4, interpolation="linear",
        return_selected_indices=True,
    )

    # 期望：按距离升序选前 4 条 -> 原始索引 [4, 3, 2, 1]
    expect_sel = [4, 3, 2, 1]
    ok_sel = sel == expect_sel
    print(f"[A1] selected_indices={sel} 期望={expect_sel} -> {'✅' if ok_sel else '❌'}")
    assert sel != list(range(len(sel))), "构造失败：本例必须发生重排，否则测不到 bug"

    ordered_ids = reorder_lane_ids(lane_ids, sel)
    expect_ids = ["lane_4", "lane_3", "lane_2", "lane_1"]
    ok_ids = ordered_ids == expect_ids
    print(f"[A2] 重排后 lane_ids={ordered_ids} -> {'✅' if ok_ids else '❌'}")

    # 几何一致性：coords_t[i] 的首点 x 应等于 dists[sel[i]]
    ok_geo = True
    for i, si in enumerate(sel):
        if abs(float(coords_t[i, 0, 0]) - dists[si]) > 1e-4:
            ok_geo = False
    print(f"[A3] 几何与 selected_indices 一致 -> {'✅' if ok_geo else '❌'}")

    # route 只含 lane_3（原始 idx=3，重排后应落在位置 1）
    progress = {"lane_3": 0.5}
    ch = build_route_channels(ordered_ids, coords_t.numpy(), progress,
                              max_lanes=max_elements, map_radius=50.0)
    on_route_pos = np.flatnonzero(ch[:, 2] > 0.5).tolist()
    ok_pos = on_route_pos == [1]
    print(f"[A4] is_on_route 落点={on_route_pos} 期望=[1] -> {'✅' if ok_pos else '❌'}")

    ok_prog = abs(float(ch[1, 1]) - 0.5) < 1e-6 and float(ch[0, 1]) == 0.0
    print(f"[A5] progress 写对位置（ch[1,1]={ch[1,1]:.2f}, ch[0,1]={ch[0,1]:.2f}）"
          f" -> {'✅' if ok_prog else '❌'}")

    ok_prox = abs(float(ch[1, 0]) - 1.0) < 1e-5
    print(f"[A6] on-route lane 邻近度=1（ch[1,0]={ch[1,0]:.3f}） -> {'✅' if ok_prox else '❌'}")

    # 反例：若沿用**原始顺序**（bug 行为），is_on_route 会落在位置 3 而非 1
    ch_bug = build_route_channels(lane_ids[:max_elements], coords_t.numpy(), progress,
                                  max_lanes=max_elements, map_radius=50.0)
    bug_pos = np.flatnonzero(ch_bug[:, 2] > 0.5).tolist()
    ok_detect = bug_pos != on_route_pos
    print(f"[A7] 反例（原始顺序）落点={bug_pos} != 正确落点 -> "
          f"{'✅ 本测试确实能抓到该 bug' if ok_detect else '❌ 测试无效'}")

    # 向后兼容：不传 return_selected_indices 时仍返回三元组
    ret = convert_feature_layer_to_fixed_size(
        ego_pose, feature_coords, None, max_elements, max_points, 4, "linear")
    ok_compat = isinstance(ret, tuple) and len(ret) == 3
    print(f"[A8] 默认调用仍返回三元组 -> {'✅' if ok_compat else '❌'}")

    return all([ok_sel, ok_ids, ok_geo, ok_pos, ok_prog, ok_prox, ok_detect, ok_compat])


# ---------------------------------------------------------------- B. real data
def test_real_scenarios(n_scenarios: int) -> bool:
    """≥N 个真实 scenario：最终第 i 条 lane 的 route 标签必须等于其 id 是否在 route 上。"""
    print()
    print("=" * 70)
    print(f"B. 真实数据：{n_scenarios} 个 scenario 的 route semantic alignment")
    print("=" * 70)

    import glob
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    dbs = sorted(glob.glob(f"{DATA}/*.db"))[:40]
    builder = NuPlanScenarioBuilder(
        data_root=DATA, map_root=f"{DATASET_ROOT}/maps",
        sensor_root=f"{DATASET_ROOT}/nuplan-v1.1/sensor_blobs",
        db_files=dbs, map_version="nuplan-maps-v1.0",
    )
    sf = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=n_scenarios,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
    )
    scenarios = builder.get_scenarios(sf, Sequential())
    print(f"取得 {len(scenarios)} 个 scenario")

    n_checked = n_lane_rows = n_mismatch = n_reordered = n_with_route = 0

    for sc in scenarios:
        try:
            ego_states = list(sc.get_ego_past_trajectory(
                iteration=0, num_samples=config.NUM_PAST_POSES,
                time_horizon=config.PAST_TIME_HORIZON))
            if not ego_states:
                continue
            cur = ego_states[-1]
            pos = Point2D(cur.rear_axle.x, cur.rear_axle.y)
            rb = sc.get_route_roadblock_ids() or []

            coords, tl, lane_ids = get_neighbor_vector_set_map(
                sc.map_api, config.MAP_FEATURES, pos, config.MAP_RADIUS, rb, [],
                return_lane_ids=True)
            progress = get_route_lane_progress(sc.map_api, pos, config.MAP_RADIUS, rb)
            vector_map, sel_idx = map_process(
                cur.rear_axle, coords, tl, config.MAP_FEATURES, config.MAX_ELEMENTS,
                config.MAX_POINTS, config.INTERPOLATION, return_selected_indices=True)

            sel = sel_idx.get("LANE", [])
            if sel != sorted(sel):
                n_reordered += 1

            ordered_ids = reorder_lane_ids(lane_ids, sel)
            ch = build_route_channels(ordered_ids, vector_map["lanes"], progress,
                                      config.LANE_NUM, config.MAP_RADIUS)
            if progress:
                n_with_route += 1

            # ★ 核心判据：第 i 条 lane 的 is_on_route 必须等于 (ordered_ids[i] ∈ progress)
            for i, lid in enumerate(ordered_ids):
                n_lane_rows += 1
                expect = 1.0 if lid in progress else 0.0
                if abs(float(ch[i, 2]) - expect) > 1e-6:
                    n_mismatch += 1
                # progress 通道同样校验
                exp_p = progress.get(lid, 0.0) if progress else 0.0
                if abs(float(ch[i, 1]) - exp_p) > 1e-5:
                    n_mismatch += 1
            n_checked += 1
        except Exception as e:  # noqa: BLE001
            print(f"  跳过 {sc.token[:10]}: {type(e).__name__}: {e}")

    print(f"检查 scenario 数      : {n_checked}")
    print(f"其中发生 lane 重排    : {n_reordered}  ({n_reordered / max(n_checked,1):.0%})")
    print(f"其中含 route 信号     : {n_with_route}")
    print(f"检查 lane 行数        : {n_lane_rows}")
    print(f"★ mismatch 数         : {n_mismatch}  （要求 = 0）")

    ok = (n_mismatch == 0) and (n_checked >= min(n_scenarios, 100)) and (n_reordered > 0)
    if n_reordered == 0:
        print("  ⚠️ 无任何重排样本 —— 测试未覆盖 bug 场景，判为不通过")
    print(f"B 结论: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenarios", type=int, default=120)
    a = ap.parse_args()

    ok_a = test_synthetic_misorder()
    ok_b = test_real_scenarios(a.n_scenarios)
    print()
    print("=" * 70)
    print(f"route ↔ lane 对齐总判定: {'✅ 全部通过' if (ok_a and ok_b) else '❌ 有失败'}")
    sys.exit(0 if (ok_a and ok_b) else 1)


if __name__ == "__main__":
    main()
