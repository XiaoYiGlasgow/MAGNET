"""Urban Driver 对齐版的单元测试（nuplan 环境，CPU，冻结 test 前几个 token，不训练）。

验证对象：GNN_Trans/external_baselines/urban_driver/（方案文档 补外部基线_方案_20260911.md §00 的 D8）

检查项
------
U1 时域与形状：自车历史 20 帧、周车 [20, N, 8]、target 60 帧，模型 yaml 的时域 60 帧 / 6 s
U2 车辆集合与我方相同：只有 VEHICLE，辆数相等，逐辆按顺序对应，20 帧历史位置逐位相同
U3 自车历史与我方折线自车行逐位相同（两者都走 float32 路径；不补精确 EGO_HISTORY，U9）
U4 车道「来源相同」（集合与精度用我们的，点数照 Urban Driver 的 20）：
   只有 LANE 层、红绿灯全 UNKNOWN、车道条数与顺序与我方相同、首尾端点逐位相同、
   两边的点都落在同一条 float32 取整源折线上（我方 50 点、对齐版 20 点都是在 float32 全局坐标上按下标插值，
   偏离不超过半个 float32 网格对角线）
U5 自车 GT 与我方 GROUND_TRUTH 同口径逐点吻合（E8 同一道闸，0.05 m）
U6 模型从 yaml 实例化、CPU 前向输出 [B, 60, 3]
U7 历史运动学增强（dt=0.1，D7 待裁）在 20 帧自车历史上能求解且形状不变（只查自洽，不代表 D7 结论）

用法：
  /home/team/anaconda3/envs/nuplan/bin/python tests/test_urban_driver_aligned.py [--n_tokens 4]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for p in (str(GNN.parent), str(GNN / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_plantf_aligned_builder as base  # noqa: E402  复用取场景、读我方特征、点到折线距离

MODEL_YAML = GNN / "external_baselines/config/model/urban_driver_aligned.yaml"
IDENTITY_TOL_M = 1e-3
GT_PARITY_TOL_M = 0.05
UNKNOWN_TL = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _build_model():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.load(MODEL_YAML))


def _our_ego_points(df) -> np.ndarray:
    """我方折线里的自车行还原成 20 个点：19 段的起点 + 最后一段的终点。"""
    poly = np.asarray(df["POLYLINE_FEATURES"].iloc[0])
    types = df["SUBGRAPH_ID_TO_TYPE"].iloc[0]
    ego_sid = [int(s) for s, t in types.items() if t == "ego"][0]
    rows = poly[poly[:, 9].astype(int) == ego_sid]
    return np.vstack([rows[:, 0:2], rows[-1:, 2:4]])


def check_token(sc, model, fails: List[str]) -> Dict:
    import torch
    from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame

    from GNN_Trans.external_baselines.magnet_alignment import select_magnet_lanes

    tok = sc.token
    df = base._our_df(tok)
    stat: Dict = {"token": tok}
    map_builder, agent_builder = model.feature_builders
    vmap = map_builder.get_features_from_scenario(sc)
    agents = agent_builder.get_features_from_scenario(sc)
    target = model.target_builders[0].get_targets(sc)

    # U1 形状
    ego, veh = agents.ego[0], agents.agents["VEHICLE"][0]
    if ego.shape[0] != 20 or veh.shape[0] != 20 or target.data.shape != (60, 3):
        fails.append(f"{tok} U1 形状不符 ego {ego.shape} agents {veh.shape} target {target.data.shape}")
    if list(agents.agents) != ["VEHICLE"]:
        fails.append(f"{tok} U1 周车类型 {list(agents.agents)}")

    # U2 车辆集合、顺序与 20 帧历史位置
    ah = np.asarray(df["AGENT_HISTORIES"].iloc[0])
    our_valid = np.abs(ah[:, -1, :]).sum(-1) != 0
    n_our = int(our_valid.sum())
    stat["n_agent_our"], stat["n_agent_aligned"] = n_our, int(veh.shape[1])
    if n_our != veh.shape[1] or not our_valid[:n_our].all():
        fails.append(f"{tok} U2 车辆数不等或我方槽位不连续：我方 {n_our} vs 对齐版 {veh.shape[1]}")
    else:
        gap = float(np.abs(veh[:, :, :2].transpose(1, 0, 2) - ah[:n_our, :, :2]).max()) if n_our else 0.0
        stat["agent_hist_gap_max_m"] = gap
        if gap > IDENTITY_TOL_M:
            fails.append(f"{tok} U2 周车历史位置不恒等（逐辆按顺序对应）：最大差 {gap:.4f} m")

    # U3 自车历史
    ego_gap = float(np.abs(ego[:, :2] - _our_ego_points(df)).max())
    stat["ego_hist_gap_max_m"] = ego_gap
    if ego_gap > IDENTITY_TOL_M:
        fails.append(f"{tok} U3 自车历史与我方折线自车行不恒等：最大差 {ego_gap:.4f} m")

    # U4 车道
    if set(vmap.coords) != {"LANE"}:
        fails.append(f"{tok} U4 地图层 {set(vmap.coords)}，应只有 LANE")
    coords, avails = vmap.coords["LANE"][0], vmap.availabilities["LANE"][0]
    tl = vmap.traffic_light_data["LANE"][0]
    live = avails.any(-1)
    mine = [coords[i] for i in np.where(live)[0]]
    ours = base._our_lanes(df)
    stat["n_lane_our"], stat["n_lane_aligned"] = len(ours), len(mine)
    if coords.shape[1] != 20 or not avails[live].all():
        fails.append(f"{tok} U4 每条车道应为 20 个有效点，实为 {coords.shape}")
    if not np.all(tl[live] == UNKNOWN_TL):
        fails.append(f"{tok} U4 红绿灯不全是 UNKNOWN")
    if len(ours) != len(mine) or not live[: len(mine)].all():
        fails.append(f"{tok} U4 车道条数不等或槽位不连续：我方 {len(ours)} vs 对齐版 {len(mine)}")
    elif mine:
        end_gap = max(float(np.abs(np.concatenate([m[0] - o[0], m[-1] - o[-1]])).max()) for m, o in zip(mine, ours))
        stat["lane_endpoint_gap_max_m"] = end_gap
        if end_gap > IDENTITY_TOL_M:
            fails.append(f"{tok} U4 车道按顺序配对后端点不恒等：最大差 {end_gap:.4f} m（集合或顺序不同）")
        rear = sc.initial_ego_state.rear_axle
        lanes = select_magnet_lanes(sc.map_api, rear)
        raw32 = [coordinates_to_local_frame(c.double(), lanes.anchor.double(), precision=torch.float64).numpy()
                 for c in lanes.centerlines]
        a32 = np.array([rear.x, rear.y], dtype=np.float32)
        half_diag = 0.5 * float(np.hypot(*np.spacing(a32)))
        a_gap = max(float(base._dist_to_polyline(m, r).max()) for m, r in zip(mine, raw32))
        b_gap = max(float(base._dist_to_polyline(o, r).max()) for o, r in zip(ours, raw32))
        stat["lane_aligned_on_raw32_max_m"], stat["lane_our_on_raw32_max_m"] = a_gap, b_gap
        if max(a_gap, b_gap) > half_diag + IDENTITY_TOL_M:
            fails.append(f"{tok} U4 车道偏离同一源折线：对齐版 {a_gap:.4f} / 我方 {b_gap:.4f} m > 半网格 {half_diag:.4f} m")

    # U5 自车 GT
    gt_gap = float(np.linalg.norm(base._load_e8().load_our_gt(tok)
                                  - (target.data[:, :2] - target.data[0:1, :2]), axis=1).max())
    stat["gt_parity_max_m"] = gt_gap
    if gt_gap > GT_PARITY_TOL_M:
        fails.append(f"{tok} U5 GT 不吻合 max|Δ| = {gt_gap:.4f} m")
    return {**stat, "_features": {"vector_set_map": vmap, "generic_agents": agents}}


def check_forward(model, rows: List[Dict], fails: List[str]) -> None:
    """U6：批量前向输出 [B, 60, 3]。"""
    import torch

    from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
    from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap

    batch = {
        "vector_set_map": VectorSetMap.collate([r["_features"]["vector_set_map"].to_feature_tensor() for r in rows]),
        "generic_agents": GenericAgents.collate([r["_features"]["generic_agents"].to_feature_tensor() for r in rows]),
    }
    with torch.no_grad():
        out = model.eval()(batch)
    shape = tuple(out["trajectory"].data.shape)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"U6 前向输出 {shape}，参数量 {n_param / 1e6:.3f} M")
    if shape != (len(rows), 60, 3):
        fails.append(f"U6 trajectory 形状 {shape}，应为 ({len(rows)}, 60, 3)")


def check_augmentation(row: Dict, fails: List[str]) -> None:
    """U7：历史运动学增强在 20 帧自车历史上可求解、形状不变。"""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(GNN / "external_baselines/config/data_augmentation/"
                                "kinematic_history_generic_agent_augmentation_aligned.yaml")
    cfg = next(iter(cfg.values()))
    cfg.augment_prob = 1.0
    aug = instantiate(cfg)
    feats = copy.deepcopy(row["_features"])
    before = feats["generic_agents"].ego[0].copy()
    np.random.seed(0)
    t0 = time.time()
    out, _ = aug.augment(feats, {})
    after = out["generic_agents"].ego[0]
    print(f"U7 增强一次耗时 {1000 * (time.time() - t0):.1f} ms，自车历史是否被改写：{not np.array_equal(before, after)}")
    if after.shape != before.shape:
        fails.append(f"U7 增强后自车历史形状 {after.shape} ≠ {before.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=4)
    args = ap.parse_args()

    e8 = base._load_e8()
    tokens = sorted(p.stem.replace("features_", "") for p in base.TEST_FEATURES.glob("features_*.pkl"))[: args.n_tokens]
    model = _build_model()
    fails: List[str] = []
    fts = model.future_trajectory_sampling
    if fts.num_poses != 60 or fts.time_horizon != 6.0:
        fails.append(f"U1 模型时域 {fts}")
    rows = [check_token(sc, model, fails) for sc in base._scenarios(e8, tokens)]
    if len(rows) != len(tokens):
        fails.append(f"只取到 {len(rows)} / {len(tokens)} 个场景")
    if rows:
        check_forward(model, rows, fails)
        check_augmentation(rows[0], fails)

    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "_features"})
    if fails:
        print(f"\n❌ {len(fails)} 项失败：")
        for f in fails:
            print("  -", f)
        return 1
    print(f"\n✅ U1–U7 全部通过（{len(rows)} 个 token）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
