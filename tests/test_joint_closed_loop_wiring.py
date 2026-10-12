#!/usr/bin/env python
"""路线 B 联合模型 —— **闭环 planner 接线**等价单测（§10 A 表第 10 项，2026-08-25）。

被测对象：受管文件 `nuplan/planning/simulation/planner/gnn_trans_planner.py` 新增的
`_increments_to_local_pos` / `_joint_ego_local_pos` / `_is_joint` 分支，
以及 `_output_to_trajectory` 抬高后的入口口径。

为什么要接：逐时域曲线给出了**事前可证伪的预测** —— 新结构在 0.9 s 处已反超常速、
相对旧模型好 2.09 倍，而 planner 靠的正是短时域 ⇒ 闭环分数应显著高于 0.2288。
Q23 的定位（诊断 → 修复 → **官方闭环验证**）在这一步才闭合。

判据
----
[A] **旧路径逐位不变**：`_increments_to_local_pos` == 抽出前的 `cumsum(reshape(-1,2))`；
    点数失配当场 raise（不静默截断/补齐）。
[B] **ego 行取法**：`_joint_ego_local_pos` 逐位等于 `predict_closed_loop(batch)[ego_index[0]]`，
    且 `ego_index[0] == 0`（契约 D：ego 是每样本节点 0）；形状 `[60,2]`。
[C] **D-i：argmax π，绝不模态平均**：与 `select_mode_argmax(model(batch))` 逐位相同，
    且与模态平均**不同**（反证 —— 否则这条判据是平凡为真的）。
[D] **时间基同构（防「原点滞后」旧坑复发）**：joint 的 `local_pos[0]` 就在原点附近，
    与旧路径 `cumsum[0]=[0,0]` 同口径；喂进 `_output_to_trajectory` 后
    `trajectory[0]` = 当前 ego、`trajectory[1]` 的位移 == `‖local_pos[1]-local_pos[0]‖`。
[E] **加载分支是必需的不是冗余的**：joint 模型**没有** `ego_history_encoder`
    （P4 已把那条 BN 换成 LayerNorm）⇒ 旧日志行必 AttributeError；
    它**有** `active_head()` ⇒ Q46 塌陷探针照样挂得上。
[F] **三道 sidecar 门在真 ckpt 上的行为**：route_mode/dataset_config 通过，
    feature_contract 由 `train_root` 推断为 `full_retention`（配 default 必 raise），
    temporal 判定为「非 R1」。`_is_joint` 由真正建出的类名置真。

🔴 全程 CPU —— `MagnetJointAdaptive` 在 GPU 上前向不可复现（同输入两次 max|Δ| 3.8e-6，
`index_add_` 的 atomic 顺序），逐位判据在 GPU 上必然假失败（§8.2 红线）。

跑法::

    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    /home/team/anaconda3/envs/nuplan/bin/python tests/test_joint_closed_loop_wiring.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
_DEVKIT = _GNN.parent
for _p in (str(_GNN), str(_GNN / "scripts" / "training"), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
CKPT = _GNN / "experiments/joint_s2_fixed/seed1/models/checkpoint_best.pt"
NUM_POINTS = 60
_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def _planner_module():
    from nuplan.planning.simulation.planner import gnn_trans_planner as m  # noqa: E402
    return m


def _one_batch():
    """取 fullret val 第一个可用样本 → 单样本 `Batch`（与 planner 在线路径同一 helper）。"""
    from torch_geometric.data import Batch  # noqa: E402
    from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
    for p in sorted(VAL.glob("features_*.pkl"))[:60]:
        df = pd.read_pickle(p)
        data, reason = build_graph_data_from_feature_df(df, "default")
        if data is None or getattr(data, "gnn_x", None) is None or not data.gnn_x.numel():
            continue
        if getattr(data, "ego_history", None) is None or not data.ego_history.numel():
            continue
        return Batch.from_data_list([data])
    raise RuntimeError(f"{VAL} 前 60 个样本里没有可用的（df/建图/ego_history 三者齐全）")


def _model():
    from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402
    return load_model_for_inference(CKPT, device=torch.device("cpu"))


# ── [A] 旧路径逐位不变 ─────────────────────────────────────────────────────
def test_a_legacy_bitwise_identical() -> None:
    print("[A 旧路径逐位不变]")
    m = _planner_module()
    g = torch.Generator().manual_seed(0)
    inc = torch.zeros(60, 2)
    inc[1:, 0] = 0.6
    inc[1:, 1] = 0.02 * torch.randn(59, generator=g)
    out = inc.reshape(1, -1)

    golden = np.cumsum(out.detach().cpu().numpy().reshape(-1, 2), axis=0)  # 抽出前的原算法
    got = m._increments_to_local_pos(out, NUM_POINTS)
    check(np.array_equal(got, golden), f"cumsum 口径逐位相同（max|Δ|={np.max(np.abs(got - golden)):.3e}）")
    check(got.dtype == golden.dtype, f"dtype 不变：{got.dtype}")
    check(bool(np.all(got[0] == 0.0)), "首点恒为原点（首帧增量 [0,0] 的必然结果）")

    try:
        m._increments_to_local_pos(torch.zeros(1, 100), NUM_POINTS)
        check(False, "点数失配应当 raise")
    except RuntimeError as e:
        check("口径失配即失败" in str(e), "点数失配当场 raise，不静默截断/补齐")


# ── [B][C][D] 真 ckpt + 真样本 ─────────────────────────────────────────────
def test_bcd_joint_ego_row() -> None:
    print("[B/C/D 联合模型的 ego 行、模态选择、时间基]")
    from magnet_joint_decoder import select_mode_argmax  # noqa: E402
    from magnet_joint_encoder import build_entity_index  # noqa: E402
    m = _planner_module()
    model, batch = _model(), _one_batch()

    idx = build_entity_index(batch)
    ego_row = int(idx.ego_index[0])
    check(int(idx.ego_index.numel()) == 1, f"单样本 ⇒ 只有一个 ego 行（实测 {idx.ego_index.tolist()}）")
    check(ego_row == 0, f"契约 D：ego 是节点 0（实测 {ego_row}）")

    got = m._joint_ego_local_pos(model, batch, NUM_POINTS)
    check(got.shape == (NUM_POINTS, 2), f"形状 [60,2]（实测 {got.shape}）")

    with torch.no_grad():
        pred = model.predict_closed_loop(batch)                 # [n_entity,60,2]
        full = model(batch)
        arg = select_mode_argmax(full)[0]                       # 同上，但走公开 API
        avg = full.traj.mean(dim=1)                     # [n_entity,60,2] 模态平均（反例）
    check(np.array_equal(got, pred[ego_row].numpy()), "[B] 与 predict_closed_loop 的 ego 行逐位相同")
    check(np.array_equal(got, arg[ego_row].numpy()), "[C] 与 select_mode_argmax 逐位相同（D-i）")
    d_avg = float(np.max(np.abs(got - avg[ego_row].numpy())))
    check(d_avg > 1e-6, f"[C 反证] 与模态平均**不同**（max|Δ|={d_avg:.4f} m）—— 判据非平凡")

    r0 = float(np.hypot(*got[0]))
    step = float(np.hypot(*(got[1] - got[0])))
    check(r0 < 0.5, f"[D] 首点在原点附近（‖local_pos[0]‖={r0:.4f} m）⇒ 与旧路径 cumsum[0]=[0,0] 同口径")
    check(step > 1e-4, f"[D] 第二点相对首点有真实位移（{step:.4f} m）")


def test_d_trajectory_anchor() -> None:
    """[D 续] 喂进真 `_output_to_trajectory`：首点=当前 ego，第二点位移=局部一步位移。"""
    print("[D 原点锚定（防 2026-07 那个滞后 bug 复发）]")
    m = _planner_module()
    from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402

    st = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(10.0, -5.0, 0.7), rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0,
        time_point=TimePoint(1_000_000), vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True, angular_vel=0.0, angular_accel=0.0)

    class _FakeHistory:
        ego_states = [st]

    class _FakeInput:
        history = _FakeHistory()

    for k in ("MAGNET_PLANNING_LAYER", "MAGNET_REFERENCE_SOURCE"):
        os.environ.pop(k, None)
    p = m.GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu")
    local_pos = np.cumsum(np.concatenate([np.zeros((1, 2)), np.tile([0.6, 0.0], (59, 1))]), axis=0)
    traj = p._output_to_trajectory(local_pos.astype(np.float32), _FakeInput())
    pts = traj.get_sampled_trajectory()
    check(len(pts) == NUM_POINTS, f"输出 {len(pts)} 个点")
    check(abs(pts[0].rear_axle.x - 10.0) < 1e-9 and abs(pts[0].rear_axle.y + 5.0) < 1e-9,
          "trajectory[0] == 当前 ego（cumsum[0] 放 t0，不是 t0+dt）")
    d1 = float(np.hypot(pts[1].rear_axle.x - pts[0].rear_axle.x, pts[1].rear_axle.y - pts[0].rear_axle.y))
    check(abs(d1 - 0.6) < 1e-5, f"trajectory[1] 位移 = local_pos[1]-local_pos[0] = 0.6 m（实测 {d1:.6f}）")


# ── [G] frame-0 口径：**不做去偏**，且把偏移量级钉在案 ──────────────────────
def test_g_no_frame0_debias() -> None:
    """[G] joint 的 `local_pos[0]` 不在原点（旧模型恒为 [0,0]），本接线**不做 frame-0 去偏**。

    为什么这样选（🔴 是选择不是默认，改动它等于改部署口径）：
      * 开环 ADE/FDE 就是拿这套绝对 ego-centric 位置直接与 GT 比的（GT 侧 `cumsum(GT)[0]`
        逐位在原点，§8.2），**没有**任何去偏 ⇒ 闭环若去偏，用的就不是我们报的那个量；
      * 去偏是只在部署侧存在的后处理，属于「悄悄改口径」，正是 §8.2 红线要防的。
    ⇒ `local_pos[0]` 被 `_output_to_trajectory` **整个丢弃**，首个未来点 = 当前 ego + rot(local_pos[1])。
    本判据把这个行为钉死，并把偏移量级记录在案（数字进日志，不进正文）。
    """
    print("[G frame-0 不去偏（口径钉死 + 量级在案）]")
    import numpy as _np
    m = _planner_module()
    model = _model()

    # (1) 接线**不**减 local_pos[0]：造一个首点非零的 local_pos，看第一步位移取哪个
    from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # noqa: E402
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402
    st = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0), rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0,
        time_point=TimePoint(1_000_000), vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True, angular_vel=0.0, angular_accel=0.0)

    class _H:
        ego_states = [st]

    class _I:
        history = _H()

    for k in ("MAGNET_PLANNING_LAYER", "MAGNET_REFERENCE_SOURCE"):
        os.environ.pop(k, None)
    p_ = m.GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu")
    lp = _np.zeros((NUM_POINTS, 2), dtype=_np.float32)
    lp[0] = [0.3, 0.0]                                  # 首点故意非零
    lp[1:, 0] = 0.3 + 0.6 * _np.arange(1, NUM_POINTS)   # 逐步 0.6 m
    pts = p_._output_to_trajectory(lp, _I()).get_sampled_trajectory()
    d1 = float(_np.hypot(pts[1].rear_axle.x - pts[0].rear_axle.x, pts[1].rear_axle.y - pts[0].rear_axle.y))
    check(abs(d1 - 0.9) < 1e-5,
          f"第一步 = ‖local_pos[1]‖ = 0.9 m（**不是** ‖local_pos[1]-local_pos[0]‖ = 0.6）；实测 {d1:.6f}")

    # (2) 量级在案：20 个 val 样本上 ‖local_pos[0]‖ 的分布
    from torch_geometric.data import Batch  # noqa: E402
    from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
    r0 = []
    for f in sorted(VAL.glob("features_*.pkl"))[:60]:
        if len(r0) >= 20:
            break
        d, _ = build_graph_data_from_feature_df(pd.read_pickle(f), "default")
        if d is None or getattr(d, "gnn_x", None) is None or not d.gnn_x.numel():
            continue
        if getattr(d, "ego_history", None) is None or not d.ego_history.numel():
            continue
        pos = m._joint_ego_local_pos(model, Batch.from_data_list([d]), NUM_POINTS)
        r0.append(float(_np.hypot(*pos[0])))
    a = _np.asarray(r0)
    print(f"   ℹ️ ‖local_pos[0]‖ over n={a.size}: median={_np.median(a):.4f} "
          f"p90={_np.percentile(a, 90):.4f} max={a.max():.4f} m")
    check(a.size >= 10, f"至少取到 10 个样本（实测 {a.size}）")
    check(float(_np.median(a)) > 1e-3,
          "joint 的首点确实不在原点 ⇒ [G] 这条判据非平凡（旧模型恒为 0）")


# ── [E] 加载分支必需性 ────────────────────────────────────────────────────
def test_e_joint_has_no_bn_branch() -> None:
    print("[E 加载分支不是防御性冗余]")
    model = _model()
    check(type(model).__name__ == "MagnetJointAdaptive", f"建出的类 = {type(model).__name__}")
    check(not hasattr(model, "ego_history_encoder"),
          "joint 模型没有 ego_history_encoder ⇒ 旧的 BN 日志行必 AttributeError（P4 换成了 LayerNorm）")
    check(hasattr(model, "active_head") and len(list(model.active_head())) > 0,
          "有 active_head() ⇒ Q46 塌陷探针挂得上")
    check(hasattr(model, "predict_closed_loop"), "有 predict_closed_loop（D-i 的唯一入口）")


# ── [F] 三道 sidecar 门 ────────────────────────────────────────────────────
def test_f_sidecar_gates_on_real_ckpt() -> None:
    print("[F 三道 sidecar 门在真 ckpt 上的行为]")
    m = _planner_module()
    for k in ("MAGNET_FEATURE_CONTRACT", "MAGNET_EDGE_MASK", "MAGNET_PLANNING_LAYER"):
        os.environ.pop(k, None)

    p = m.GNNTransPlanner(checkpoint_path=str(CKPT), device="cpu", feature_contract="full_retention")
    p._load_model()
    check(p._is_joint is True, "_is_joint 由真正建出的类名置真")
    check(p._needs_temporal is False, "temporal 判定为非 R1（sidecar temporal_graph=false）")
    check(p._contract["name"] == "full_retention" and p._contract["num_agents"] == 64,
          "契约 = full_retention / 64 agents（S2 训练在 fullret 上）")

    bad = m.GNNTransPlanner(checkpoint_path=str(CKPT), device="cpu", feature_contract="default")
    try:
        bad._load_model()
        check(False, "配 default 契约应当 raise")
    except ValueError as e:
        check("full_retention" in str(e), "配 default 契约 ⇒ raise 且点名 full_retention（train_root 含 fullret）")


if __name__ == "__main__":
    if not CKPT.exists():
        print(f"🔴 缺 ckpt：{CKPT}")
        sys.exit(1)
    test_a_legacy_bitwise_identical()
    test_bcd_joint_ego_row()
    test_d_trajectory_anchor()
    test_g_no_frame0_debias()
    test_e_joint_has_no_bn_branch()
    test_f_sidecar_gates_on_real_ckpt()
    if _FAILED:
        print(f"\n🔴 {len(_FAILED)} 项失败:")
        for f in _FAILED:
            print("   -", f)
        sys.exit(1)
    print("\n✅ test_joint_closed_loop_wiring 6/6 通过")
