"""LaneGCN 对齐版的单元测试（nuplan 环境，CPU，冻结 test 前几个 token，不训练）。

验证对象：GNN_Trans/external_baselines/lanegcn/（方案文档 补外部基线_方案_20260911.md §00 的 D8）

检查项
------
L1 时域与形状：自车 [20, 3]、周车 [20, N, 8]、target [60, 3]
L2 车辆集合与我方相同：辆数相等、逐辆按顺序对应、20 帧历史位置逐位相同
L3 自车历史与我方折线自车行逐位相同
L4 车道来源与裁剪：
   - 所用车道集合就是我方的（逐条首尾端点与我方逐位相同）
   - 每组线段恰为对应车道 float32 原始节点按 LaneGCN 规则裁剪后的连续节点
     （首个与末个距锚点 ≤ 50 m 的节点之间全部保留），半径内节点 ≤ 2 个的车道不建线段
L5 连接只在车道内部：第 k 尺度连接恰为同一车道内相距 k 段的 (i, i+k)，没有任何跨车道连接
L6 红绿灯全 UNKNOWN、on_route 全 UNKNOWN [0, 0]
L7 自车 GT 与我方 GROUND_TRUTH 同口径逐点吻合（0.05 m）
L8 模型从 yaml 实例化、CPU 前向输出 [B, 60, 3]，报前向耗时
L9 运动学增强（60 点 / 0.1 s，D7 待裁）可求解、形状不变，报耗时（只查自洽，不代表 D7 结论）

用法：
  /home/team/anaconda3/envs/nuplan/bin/python tests/test_lanegcn_aligned.py [--n_tokens 4]
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

import test_plantf_aligned_builder as base  # noqa: E402
from test_urban_driver_aligned import _our_ego_points  # noqa: E402

MODEL_YAML = GNN / "external_baselines/config/model/lanegcn_aligned.yaml"
AUG_YAML = GNN / "external_baselines/config/data_augmentation/kinematic_agent_augmentation_aligned.yaml"
IDENTITY_TOL_M = 1e-3
NODE_TOL_M = 1e-4          # 自车系 float32 输出（≤ 60 m 量级）的存储误差约 4e-6 m
GT_PARITY_TOL_M = 0.05
RADIUS_M = 50.0


def _build_model():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.load(MODEL_YAML))


def _expected_trim(centerline32: np.ndarray, anchor32: np.ndarray) -> np.ndarray:
    """LaneGCN ``trim_lane_nodes`` 的规则（独立写一遍作参照）：保留首个到末个在半径内的节点之间的全部节点。"""
    d2 = ((centerline32.astype(np.float64) - anchor32.astype(np.float64)) ** 2).sum(-1)
    inside = np.where(d2 <= RADIUS_M ** 2)[0]
    return centerline32[inside[0]: inside[-1] + 1] if len(inside) else centerline32[:0]


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

    # L1
    ego, veh = agents.ego[0], agents.agents[0]
    if ego.shape != (20, 3) or veh.shape[0] != 20 or veh.shape[2] != 8 or target.data.shape != (60, 3):
        fails.append(f"{tok} L1 形状不符 ego {ego.shape} agents {veh.shape} target {target.data.shape}")

    # L2
    ah = np.asarray(df["AGENT_HISTORIES"].iloc[0])
    our_valid = np.abs(ah[:, -1, :]).sum(-1) != 0
    n_our = int(our_valid.sum())
    stat["n_agent_our"], stat["n_agent_aligned"] = n_our, int(veh.shape[1])
    if n_our != veh.shape[1] or not our_valid[:n_our].all():
        fails.append(f"{tok} L2 车辆数不等或我方槽位不连续：我方 {n_our} vs 对齐版 {veh.shape[1]}")
    elif n_our:
        gap = float(np.abs(veh[:, :, :2].transpose(1, 0, 2) - ah[:n_our, :, :2]).max())
        stat["agent_hist_gap_max_m"] = gap
        if gap > IDENTITY_TOL_M:
            fails.append(f"{tok} L2 周车历史位置不恒等：最大差 {gap:.4f} m")

    # L3
    ego_gap = float(np.abs(ego[:, :2] - _our_ego_points(df)).max())
    stat["ego_hist_gap_max_m"] = ego_gap
    if ego_gap > IDENTITY_TOL_M:
        fails.append(f"{tok} L3 自车历史与我方折线自车行不恒等：最大差 {ego_gap:.4f} m")

    # L4 车道集合 = 我方
    rear = sc.initial_ego_state.rear_axle
    lanes = select_magnet_lanes(sc.map_api, rear)
    to_ego = lambda g: coordinates_to_local_frame(  # noqa: E731
        torch.as_tensor(g).double(), lanes.anchor.double(), precision=torch.float64).numpy()
    ours = base._our_lanes(df)
    stat["n_lane_our"], stat["n_lane_magnet"] = len(ours), len(lanes.centerlines)
    if len(ours) != len(lanes.centerlines):
        fails.append(f"{tok} L4 车道集合条数不等：我方 {len(ours)} vs 对齐版取数 {len(lanes.centerlines)}")
        return {**stat, "_features": {"vector_map": vmap, "agents": agents}, "_target": target}
    raw_local = [to_ego(c) for c in lanes.centerlines]
    end_gap = max((float(np.abs(np.concatenate([r[0] - o[0], r[-1] - o[-1]])).max())
                   for r, o in zip(raw_local, ours)), default=0.0)
    stat["lane_endpoint_gap_max_m"] = end_gap
    if end_gap > IDENTITY_TOL_M:
        fails.append(f"{tok} L4 车道集合与我方端点不恒等：最大差 {end_gap:.4f} m")

    # L4 裁剪与线段：线段组按车道顺序对应「裁剪后节点 > 2 个」的车道
    a32 = lanes.anchor[:2].numpy()
    expected = [to_ego(t) for t in (_expected_trim(c.numpy(), a32) for c in lanes.centerlines) if len(t) > 2]
    coords, groups = np.asarray(vmap.coords[0]), [np.asarray(g) for g in vmap.lane_groupings[0]]
    stat["n_group"], stat["n_lane_dropped"], stat["n_segment"] = len(groups), len(ours) - len(groups), len(coords)
    if len(groups) != len(expected):
        fails.append(f"{tok} L4 线段组数 {len(groups)} ≠ 裁剪规则下应建线段的车道数 {len(expected)}")
    else:
        node_gap = 0.0
        for g, exp in zip(groups, expected):
            nodes = np.vstack([coords[g, 0], coords[g[-1]][1:2]])
            if len(nodes) != len(exp):
                fails.append(f"{tok} L4 某组节点数 {len(nodes)} ≠ 裁剪规则 {len(exp)}")
                break
            node_gap = max(node_gap, float(np.abs(nodes - exp).max()))
        stat["segment_node_gap_max_m"] = node_gap
        if node_gap > NODE_TOL_M:
            fails.append(f"{tok} L4 线段节点与裁剪后原始节点不符：最大差 {node_gap:.2e} m")
        chain = max((float(np.abs(coords[g[1:], 0] - coords[g[:-1], 1]).max()) for g in groups if len(g) > 1), default=0.0)
        if chain > 0.0:
            fails.append(f"{tok} L4 同组线段首尾不相接：最大差 {chain:.2e} m")

    # L5 连接只在车道内部
    msc = vmap.multi_scale_connections[0]
    for k, conn in sorted(msc.items()):
        want = {(int(g[a]), int(g[a + k])) for g in groups for a in range(len(g) - k)}
        got = {tuple(map(int, c)) for c in np.asarray(conn).reshape(-1, 2)}
        if got != want:
            fails.append(f"{tok} L5 第 {k} 尺度连接与「车道内相距 {k} 段」不一致：多 {len(got - want)}、少 {len(want - got)}")
    stat["n_conn_scale1"] = int(np.asarray(msc[1]).reshape(-1, 2).shape[0])

    # L6
    if not np.all(np.asarray(vmap.traffic_light_data[0]) == np.array([0, 0, 0, 1])):
        fails.append(f"{tok} L6 红绿灯不全是 UNKNOWN")
    if np.any(np.asarray(vmap.on_route_status[0]) != 0):
        fails.append(f"{tok} L6 on_route 不全是 UNKNOWN [0, 0]")

    # L7
    gt_gap = float(np.linalg.norm(base._load_e8().load_our_gt(tok)
                                  - (target.data[:, :2] - target.data[0:1, :2]), axis=1).max())
    stat["gt_parity_max_m"] = gt_gap
    if gt_gap > GT_PARITY_TOL_M:
        fails.append(f"{tok} L7 GT 不吻合 max|Δ| = {gt_gap:.4f} m")
    return {**stat, "_features": {"vector_map": vmap, "agents": agents}, "_target": target}


def check_forward(model, rows: List[Dict], fails: List[str]) -> None:
    """L8：批量前向输出 [B, 60, 3]，报 CPU 前向耗时（LaneGCN 前向有逐样本 python 循环）。"""
    import torch

    from nuplan.planning.training.preprocessing.features.agents import Agents
    from nuplan.planning.training.preprocessing.features.vector_map import VectorMap

    batch = {
        "vector_map": VectorMap.collate([r["_features"]["vector_map"].to_feature_tensor() for r in rows]),
        "agents": Agents.collate([r["_features"]["agents"].to_feature_tensor() for r in rows]),
    }
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model(batch)
        dt = time.time() - t0
    shape = tuple(out["trajectory"].data.shape)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"L8 前向输出 {shape}，参数量 {n_param / 1e6:.3f} M，CPU 前向 {1000 * dt:.0f} ms / {len(rows)} 样本")
    if shape != (len(rows), 60, 3):
        fails.append(f"L8 trajectory 形状 {shape}，应为 ({len(rows)}, 60, 3)")


def check_augmentation(row: Dict, fails: List[str]) -> None:
    """L9：自车未来运动学增强（60 点 / 0.1 s）可求解、形状不变。"""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = next(iter(OmegaConf.load(AUG_YAML).values()))
    cfg.augment_prob = 1.0
    aug = instantiate(cfg)
    feats = {"agents": copy.deepcopy(row["_features"]["agents"])}
    targets = {"trajectory": copy.deepcopy(row["_target"])}
    before = targets["trajectory"].data.copy()
    np.random.seed(0)
    t0 = time.time()
    feats, targets = aug.augment(feats, targets)
    print(f"L9 增强一次耗时 {1000 * (time.time() - t0):.1f} ms，GT 是否被改写：{not np.array_equal(before, targets['trajectory'].data)}")
    if targets["trajectory"].data.shape != before.shape or feats["agents"].ego[0].shape != (20, 3):
        fails.append(f"L9 增强后形状不符：target {targets['trajectory'].data.shape} ego {feats['agents'].ego[0].shape}")


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
        fails.append(f"L1 模型时域 {fts}")
    rows = [check_token(sc, model, fails) for sc in base._scenarios(e8, tokens)]
    if len(rows) != len(tokens):
        fails.append(f"只取到 {len(rows)} / {len(tokens)} 个场景")
    if rows:
        check_forward(model, rows, fails)
        check_augmentation(rows[0], fails)

    for r in rows:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")})
    if fails:
        print(f"\n❌ {len(fails)} 项失败：")
        for f in fails:
            print("  -", f)
        return 1
    print(f"\n✅ L1–L9 全部通过（{len(rows)} 个 token）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
