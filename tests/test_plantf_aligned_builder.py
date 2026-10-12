"""PlanTF 对齐版的单元测试（CPU，几个冻结 test token，不训练）。

验证对象（方案文档 `补外部基线_方案_20260911.md` §00 的 U2 / U5）：
  external/planTF/src/feature_builders/nuplan_feature_builder_aligned.py
  external/planTF/src/data_augmentation/state_perturbation_aligned.py
  external/planTF/src/models/planTF/planning_model_aligned.py

检查项
------
T1 官方 builder 输出与 E8 已落盘特征逐位相同 ⇒ 父类路径未被改动
T2 对齐版形状：20 帧历史 + 60 帧未来，target 为 60 帧
T3 对齐版周车只有 EGO / VEHICLE，且与我方 AGENT_HISTORIES 的车辆逐辆配对、t0 位置逐位相同
T3b 配对车辆的未来 60 帧位置与我方 AGENT_FUTURES 逐帧相同（周车 GT 同一份）
T4 对齐版地图：只有 LANE 类型、红绿灯全 UNKNOWN、on_route 全 False、无限速；
   车道「来源相同」（方案 D8：车道集合与精度用我们的，每条车道的表示照 PlanTF 弧长 20 段 + 50 m 方框掩点）：
   - 掩点前车道条数与我方非零车道数相等，逐条一一配对，首尾端点与我方逐位相同（float32 路径）
   - 每条的 21 个弧长采样点都落在我方对应 50 点折线上（误差 = 我方按下标插值的弦差 + float32 取整）
   - 掩点后保留的车道恰为方框内至少有一个采样点的那些，valid_mask 与方框判据一致
   （「逐位相同」依赖 replicate_float32_utm=True：MAGNET 的地图与周车带 float32 全局坐标取整）
T5 自车 GT 与我方 GROUND_TRUTH 同口径逐点吻合（E8 同一道闸，0.05 m）
T6 对齐版增强扰动后 target 与 position 自洽（按 20 帧切）
T7 对齐版模型在 CPU 上前向，输出 60 帧

用法（必须用 plantf 环境，nuplan 环境缺 numba）：
  /home/team/anaconda3/envs/plantf/bin/python tests/test_plantf_aligned_builder.py [--n_tokens 4]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
PLANTF = Path("/mnt/local/Xiao_Yi/external/planTF")
for p in (str(PLANTF), str(GNN.parent), str(GNN)):
    if p not in sys.path:
        sys.path.insert(0, p)

OURANCHOR = GNN / "data/plantf_openloop_ouranchor"
TEST_FEATURES = GNN / "data/processed_features_fullret_L1_logsplit/test"
MATCH_TOL_M = 1.0          # 周车 t0 几何配对阈值（09-11 实测残差中位 0.140 m）
GT_PARITY_TOL_M = 0.05     # 与 E8 相同的 GT 一致性闸
IDENTITY_TOL_M = 1e-3      # 恒等闸：车道端点、周车 t0 与未来位置都应只剩 float32 存储误差
#: 「来源相同」拆成两条闸，都以同一条源折线为参照：raw32 = float32 取整后的原始车道节点折线（ego 系）
#:   对齐版 21 个弧长采样点 → raw32：弧长插值精确落在折线上，只剩 float64 舍入
#:   我方 50 点 → raw32：我方在 float32 全局坐标上按下标插值，结果再被取整到 float32 网格，
#:   偏离不超过半个网格对角线（每个 token 按锚点处 np.spacing 计算）
#: 注：对齐版点到「我方 50 点折线」的距离不作为闸（09-13 实测最大 0.35 m）——float32 北向 0.5 m 网格
#:   把原始节点变成锯齿，弧长点贴着锯齿走，我方按下标抽 50 点的弦跨过锯齿，两者差是取整形状而非来源不同
LANE_ON_RAW_TOL_M = 1e-6
LANE_RADIUS_M = 50.0
CAT_EGO, CAT_VEHICLE = 0, 1


def _load_e8():
    """按文件路径加载 E8 脚本，复用它的取场景与 GT 口径函数（不改 E8 文件）。"""
    spec = importlib.util.spec_from_file_location(
        "e8_plantf_openloop", GNN / "scripts/analysis/e8_plantf_openloop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scenarios(e8, tokens: List[str]):
    """与 E8 相同的配方取场景：不传 scenario_mapping、scenario_types=None。"""
    import os

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    by_log: Dict[str, List[str]] = {}
    for tk, lg in e8.token_to_log(tokens).items():
        by_log.setdefault(lg, []).append(tk)
    out = []
    for lg, tks in by_log.items():
        db = e8.db_path(lg)
        builder = NuPlanScenarioBuilder(data_root=os.path.dirname(db), map_root=e8.MAP_ROOT, sensor_root=None,
                                        db_files=[db], map_version=e8.MAP_VERSION)
        sf = ScenarioFilter(scenario_types=None, scenario_tokens=tks, log_names=None, map_names=None,
                            num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                            ego_displacement_minimum_m=None, expand_scenarios=False,
                            remove_invalid_goals=False, shuffle=False)
        out.extend(builder.get_scenarios(sf, Sequential()))
    return out


def _our_df(token: str):
    with open(TEST_FEATURES / f"features_{token}.pkl", "rb") as f:
        return pickle.load(f)


def _our_vehicle_t0(df) -> np.ndarray:
    """我方该 token 在 t0 有效的周车位置（ego 系），与 09-11 配对实测用的判据相同。"""
    ah = np.asarray(df["AGENT_HISTORIES"].iloc[0])
    valid = np.abs(ah[:, -1, :]).sum(-1) != 0
    return ah[valid, -1, :2]


def _our_lanes(df) -> List[np.ndarray]:
    """我方非零车道，每条还原成 50 个点：49 段的起点 + 最后一段的终点。"""
    poly = np.asarray(df["POLYLINE_FEATURES"].iloc[0])
    types = df["SUBGRAPH_ID_TO_TYPE"].iloc[0]
    sid = poly[:, 9].astype(int)
    lanes = []
    for s, t in types.items():
        if t != "lane":
            continue
        rows = poly[sid == int(s)]
        if len(rows) == 0 or np.abs(rows[:, :4]).sum() == 0:
            continue
        lanes.append(np.vstack([rows[:, 0:2], rows[-1:, 2:4]]))
    return lanes


def _aligned_lanes(mp: Dict) -> List[np.ndarray]:
    """对齐版中心线还原成点：20 个起点 + 最后一个向量的终点。"""
    pos, vec = mp["point_position"][:, 0], mp["point_vector"][:, 0]
    return [np.vstack([pos[m], pos[m, -1:] + vec[m, -1:]]) for m in range(len(pos))]


def _dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """每个点到折线（逐段线段）的最短距离。"""
    a, ab = poly[:-1], poly[1:] - poly[:-1]
    rel = pts[:, None, :] - a[None]
    t = np.clip((rel * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0.0, 1.0)
    return np.linalg.norm(rel - t[..., None] * ab[None], axis=-1).min(axis=1)


def _premask_lanes(sc, feat: Dict):
    """重建 normalize 掩点之前的全部车道，与 normalize 同一个平移与旋转（ego 系）。

    返回 (对齐版每条 21 点, 对应的 raw32 源折线, float32 半网格对角线 m)。
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from src.feature_builders.nuplan_feature_builder_aligned import NuplanFeatureBuilderAligned, float32_utm

    rear = sc.initial_ego_state.rear_axle
    anchor = np.array([rear.x, rear.y], dtype=np.float64)
    builder = NuplanFeatureBuilderAligned()
    pre = builder._get_lane_features(sc.map_api, Point2D(rear.x, rear.y), anchor)
    raw = [float32_utm(mid, anchor) for mid, _, _ in builder._select_lane_paths(sc.map_api, Point2D(rear.x, rear.y), anchor)]
    ang = feat["angle"]
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float64)
    to_ego = lambda g: np.matmul(g - feat["origin"], rot)  # noqa: E731
    half_diag = 0.5 * float(np.hypot(np.spacing(np.float32(anchor[0])), np.spacing(np.float32(anchor[1]))))
    return [to_ego(l) for l in _aligned_lanes(pre)], [to_ego(r) for r in raw], half_diag


def check_token(e8, sc, fails: List[str]) -> Dict[str, float]:
    from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
    from src.feature_builders.nuplan_feature_builder_aligned import NuplanFeatureBuilderAligned

    tok = sc.token
    stat: Dict[str, float] = {}
    df = _our_df(tok)

    # T1 官方路径未被改动：与 E8 用默认 builder 落盘的特征逐位相同
    ref_path = OURANCHOR / f"{tok}.pkl"
    if ref_path.exists():
        off = NuplanFeatureBuilder().get_features_from_scenario(sc).data
        with open(ref_path, "rb") as f:
            ref = pickle.load(f)["feature"]
        for grp in ("agent", "map"):
            for k, v in ref[grp].items():
                if not np.array_equal(np.asarray(off[grp][k]), np.asarray(v)):
                    fails.append(f"{tok} T1 官方 builder 的 {grp}.{k} 与 E8 落盘不同")
    else:
        fails.append(f"{tok} T1 缺 E8 参照特征 {ref_path}")

    feat = NuplanFeatureBuilderAligned().get_features_from_scenario(sc).data
    ag, mp = feat["agent"], feat["map"]

    # T2 形状
    if ag["position"].shape[1] != 80 or ag["target"].shape[1] != 60:
        fails.append(f"{tok} T2 形状不符 position {ag['position'].shape} target {ag['target'].shape}")

    # T3 周车类型与车辆逐辆配对
    if not set(np.unique(ag["category"]).tolist()) <= {CAT_EGO, CAT_VEHICLE} or ag["category"][0] != CAT_EGO:
        fails.append(f"{tok} T3 category 出现非 EGO/VEHICLE：{np.unique(ag['category'])}")
    present = ag["valid_mask"][1:, 19]
    al_t0 = ag["position"][1:, 19][present]
    our_t0 = _our_vehicle_t0(df)
    stat["n_our"], stat["n_aligned"] = len(our_t0), len(al_t0)
    if len(our_t0) != len(al_t0):
        fails.append(f"{tok} T3 车辆数不等：我方 {len(our_t0)} vs 对齐版 {len(al_t0)}")
    if len(our_t0) and len(al_t0):
        d = np.linalg.norm(our_t0[:, None] - al_t0[None], axis=-1).min(axis=1)
        stat["match_rate"] = float((d < MATCH_TOL_M).mean())
        stat["agent_t0_gap_max_m"] = float(d.max())
        if stat["match_rate"] < 1.0:
            fails.append(f"{tok} T3 几何配对率 {stat['match_rate']:.3f} < 1")
        if stat["agent_t0_gap_max_m"] > IDENTITY_TOL_M:
            fails.append(f"{tok} T3 周车 t0 位置不恒等：最大差 {stat['agent_t0_gap_max_m']:.4f} m")

        # T3b 未来位置：我方 AGENT_FUTURES[k] ↔ 对齐版 position[20 + k]，都对应 t0 + (k+1)·0.1 s
        ah = np.asarray(df["AGENT_HISTORIES"].iloc[0]); af = np.asarray(df["AGENT_FUTURES"].iloc[0])
        our_slots = np.where(np.abs(ah[:, -1, :]).sum(-1) != 0)[0]
        al_rows = np.where(present)[0] + 1                      # +1 跳过 ego
        pair = np.linalg.norm(our_t0[:, None] - al_t0[None], axis=-1).argmin(axis=1)
        worst, n_cmp = 0.0, 0
        for a, b in zip(our_slots, al_rows[pair]):
            ours_f = af[a, :, :2]
            mine_f = ag["position"][b, 20:80]
            both = (np.abs(ours_f).sum(-1) != 0) & ag["valid_mask"][b, 20:80]
            if both.any():
                worst = max(worst, float(np.linalg.norm(ours_f[both] - mine_f[both], axis=1).max()))
                n_cmp += int(both.sum())
        stat["agent_future_gap_max_m"], stat["agent_future_frames"] = worst, n_cmp
        if worst > IDENTITY_TOL_M:
            fails.append(f"{tok} T3b 周车未来位置不恒等：最大差 {worst:.4f} m（比较 {n_cmp} 帧）")

    # T4 地图语义
    if np.any(mp["polygon_type"] != 0):
        fails.append(f"{tok} T4 仍有非 LANE 多边形：{np.unique(mp['polygon_type'])}")
    if np.any(mp["polygon_tl_status"] != 3):
        fails.append(f"{tok} T4 红绿灯不全是 UNKNOWN")
    if np.any(mp["polygon_on_route"]):
        fails.append(f"{tok} T4 on_route 不全是 False")
    if np.any(mp["polygon_has_speed_limit"]) or np.any(mp["polygon_speed_limit"] != 0):
        fails.append(f"{tok} T4 仍带限速信息")
    c = mp["point_position"][:, 0]
    in_box = (np.abs(c[..., 0]) < LANE_RADIUS_M) & (np.abs(c[..., 1]) < LANE_RADIUS_M)
    if mp["point_position"].shape[2] != 20:
        fails.append(f"{tok} T4 每条车道应为 20 个向量（PlanTF 原方法），实为 {mp['point_position'].shape[2]}")
    if not np.array_equal(mp["valid_mask"], in_box) or not np.all(in_box.any(-1)):
        fails.append(f"{tok} T4 valid_mask 与 ±{LANE_RADIUS_M:.0f} m 方框判据不一致")

    # T4 车道来源：掩点前全集与我方逐条配对
    ours = _our_lanes(df)
    pre, raw32, half_diag = _premask_lanes(sc, feat)
    stat["n_lane_our"], stat["n_lane_premask"] = len(ours), len(pre)
    if len(ours) != len(pre):
        fails.append(f"{tok} T4 掩点前车道条数不等：我方 {len(ours)} vs 对齐版 {len(pre)}")
    if ours and pre:
        ends_o = np.stack([np.concatenate([o[0], o[-1]]) for o in ours])
        ends_p = np.stack([np.concatenate([p[0], p[-1]]) for p in pre])
        end_gap = np.abs(ends_p[:, None] - ends_o[None]).max(-1)       # [对齐版, 我方]
        pair = end_gap.argmin(axis=1)
        stat["lane_endpoint_gap_max_m"] = float(end_gap.min(axis=1).max())
        stat["lane_order_same"] = bool(np.array_equal(pair, np.arange(len(pre))))
        if len(set(pair.tolist())) != len(pre):
            fails.append(f"{tok} T4 车道配对不是一一对应")
        if stat["lane_endpoint_gap_max_m"] > IDENTITY_TOL_M:
            fails.append(f"{tok} T4 车道端点不恒等：最大差 {stat['lane_endpoint_gap_max_m']:.4f} m")
        # 两条表示都落在同一条 raw32 源折线上 ⇒ 来源相同
        a_gap = max(float(_dist_to_polyline(p, r).max()) for p, r in zip(pre, raw32))
        b_gap = max(float(_dist_to_polyline(ours[j], r).max()) for r, j in zip(raw32, pair))
        stat["lane_aligned_on_raw32_max_m"], stat["lane_our_on_raw32_max_m"] = a_gap, b_gap
        stat["lane_aligned_to_our_polyline_max_m"] = max(
            float(_dist_to_polyline(p, ours[j]).max()) for p, j in zip(pre, pair))   # 只报不设闸，见常量注释
        if a_gap > LANE_ON_RAW_TOL_M:
            fails.append(f"{tok} T4 对齐版弧长采样点不在源折线上：最大 {a_gap:.2e} m")
        if b_gap > half_diag + IDENTITY_TOL_M:
            fails.append(f"{tok} T4 我方车道偏离同一源折线 {b_gap:.4f} m > float32 半网格 {half_diag:.4f} m")

        # 掩点后保留集合 = 方框内至少有一个起点（normalize 只看 20 个起点）的车道，几何不变
        kept = [p for p in pre if ((np.abs(p[:-1]) < LANE_RADIUS_M).all(-1)).any()]
        mine = _aligned_lanes(mp)
        stat["n_lane_kept"] = len(mine)
        if len(kept) != len(mine):
            fails.append(f"{tok} T4 掩点后车道条数 {len(mine)} ≠ 方框判据 {len(kept)}")
        elif mine and max(float(np.abs(k - m).max()) for k, m in zip(kept, mine)) > 1e-6:
            fails.append(f"{tok} T4 掩点后保留车道的几何与掩点前不同")

    # T5 自车 GT 同口径逐点吻合
    pt_gt = e8.to_our_convention(np.asarray(ag["target"][0][:, :2], dtype=np.float64))
    gap = float(np.linalg.norm(e8.load_our_gt(tok) - pt_gt, axis=1).max())
    stat["gt_parity_max_m"] = gap
    if gap > GT_PARITY_TOL_M:
        fails.append(f"{tok} T5 GT 不吻合 max|Δ| = {gap:.4f} m")
    return {"token": tok, **stat, "_feat": feat}


def check_augmentation(feat: Dict, fails: List[str]) -> None:
    """T6：强制施加扰动，检查重归一化后 target 按 20 帧切。"""
    from src.data_augmentation.state_perturbation_aligned import StatePerturbationAligned
    from src.features.nuplan_feature import NuplanFeature

    aug = StatePerturbationAligned(dt=0.1, hist_len=20, low=[-1.0, -0.75, -0.35, -1, -0.5, -0.2, -0.1],
                                   high=[1.0, 0.75, 0.35, 1, 0.5, 0.2, 0.1], augment_prob=1.0, normalize=True)
    for seed in range(20):
        np.random.seed(seed)
        out, _ = aug.augment({"feature": NuplanFeature(data=copy.deepcopy(feat))})
        data = out["feature"].data
        if not np.allclose(data["current_state"], feat["current_state"]):
            ag = data["agent"]
            expect = ag["position"][:, 20:] - ag["position"][:, 19][:, None]
            expect[~ag["valid_mask"][:, 20:]] = 0
            if not np.allclose(ag["target"][..., :2], expect):
                fails.append("T6 扰动后 target 与 position 不自洽（hist_steps 未按 20 切）")
            return
    fails.append("T6 20 个随机种子都没通过安全检查，未能施加扰动")


def check_model_forward(feats: List[Dict], fails: List[str]) -> None:
    """T7：对齐版模型 CPU 前向，输出 60 帧。"""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.features.nuplan_feature import NuplanFeature

    cfg = OmegaConf.load(PLANTF / "config/model/planTF_aligned.yaml")
    model = instantiate(cfg).eval()
    fts = model.future_trajectory_sampling
    if fts.num_poses != 60 or fts.time_horizon != 6.0:
        fails.append(f"T7 target builder 时域不是 60 帧 / 6 s：{fts}")
    batch = NuplanFeature.collate([NuplanFeature(data=copy.deepcopy(f)).to_feature_tensor() for f in feats])
    with torch.no_grad():
        out = model(batch.data)
    b = len(feats)
    if tuple(out["trajectory"].shape) != (b, 6, 60, 4):
        fails.append(f"T7 trajectory 形状 {tuple(out['trajectory'].shape)}，应为 ({b}, 6, 60, 4)")
    if tuple(out["prediction"].shape[2:]) != (60, 2):
        fails.append(f"T7 prediction 形状 {tuple(out['prediction'].shape)}，末两维应为 (60, 2)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=4)
    args = ap.parse_args()

    e8 = _load_e8()
    tokens = sorted(p.stem.replace("features_", "") for p in TEST_FEATURES.glob("features_*.pkl"))[: args.n_tokens]
    fails: List[str] = []
    rows = [check_token(e8, sc, fails) for sc in _scenarios(e8, tokens)]
    if len(rows) != len(tokens):
        fails.append(f"只取到 {len(rows)} / {len(tokens)} 个场景")
    if rows:
        check_augmentation(rows[0]["_feat"], fails)
        check_model_forward([r["_feat"] for r in rows], fails)

    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "_feat"})
    if fails:
        print(f"\n❌ {len(fails)} 项失败：")
        for f in fails:
            print("  -", f)
        return 1
    print(f"\n✅ T1–T7 全部通过（{len(rows)} 个 token）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
