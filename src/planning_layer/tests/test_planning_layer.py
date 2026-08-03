"""C4 规划层单测 —— 判据 [A]–[G] 逐条对应设计文档 §3.3 第 4 条。

判据在写实现之前就已写死（`C4_规划形式化_设计_20260801.md` §3.3），
不得为了让测试通过而放宽判据；要改判据须走干预协议 §11。

跑法：
    /home/team/anaconda3/envs/nuplan/bin/python -m pytest \\
        src/planning_layer/tests/test_planning_layer.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.planning_layer import (  # noqa: E402  (必须在 sys.path 设好之后)
    EgoSnapshot,
    PlanningLayerConfig,
    SceneContext,
    available_presets,
    constant_velocity_reference,
    generate,
    preset,
    preset_from_env,
    refine,
    score_breakdown,
    select,
)
from src.planning_layer import config as C  # noqa: E402

DT = 0.1
T = 60


def straight_reference(ego: EgoSnapshot, speed: float = 5.0, n: int = T) -> np.ndarray:
    """沿 ego 朝向、匀速 `speed` 的直线参考轨迹。"""
    t = np.arange(n, dtype=float) * DT
    d = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    return np.array([ego.x, ego.y])[None, :] + (speed * t)[:, None] * d[None, :]


# ─────────────────────────────────────────────────────────────────────
# [A] 命门：关闭时输出逐位不变
# ─────────────────────────────────────────────────────────────────────
def test_A_disabled_is_bitwise_identity():
    ego = EgoSnapshot(x=10.0, y=-3.0, heading=0.4, speed=6.0)
    ref = straight_reference(ego)
    out = refine(ref, ego, cfg=PlanningLayerConfig(enabled=False))
    assert np.max(np.abs(out - ref)) == 0.0, "关闭时必须逐位相同"
    assert out is ref, "关闭时应原样返回同一对象，杜绝任何拷贝期间的数值漂移"


def test_A_env_default_is_disabled(monkeypatch):
    monkeypatch.delenv("MAGNET_PLANNING_LAYER", raising=False)
    assert preset_from_env().enabled is False
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)
    ref = straight_reference(ego)
    assert np.max(np.abs(refine(ref, ego) - ref)) == 0.0


def test_A_unknown_preset_raises_not_silently_degrades():
    with pytest.raises(ValueError):
        preset("no_such_preset")
    os.environ["MAGNET_PLANNING_LAYER"] = "typo_full"
    try:
        with pytest.raises(ValueError):
            preset_from_env()
    finally:
        del os.environ["MAGNET_PLANNING_LAYER"]


# ─────────────────────────────────────────────────────────────────────
# [B] dynamics 开 ⇒ 所有候选满足运动学硬界
# ─────────────────────────────────────────────────────────────────────
def test_B_dynamics_on_all_candidates_within_bounds():
    """🔴 判据范围 = **网格候选**（08-02 修订）。

    identity 候选（= 上游原轨迹本身）**有意豁免**动力学硬界：它代表「不干预」，
    上游给什么样就是什么样，可能违反 `a_lon`/`jerk`/`a_lat`。其动力学质量由
    `comfort_term` 在**打分**时评估，而不是靠构造排除 —— 否则它就不再是
    「原样」候选，argmax 也就失去了「层最差与层 OFF 打平」的下界保证。
    豁免本身由 `test_I3_identity_is_exempt_from_construction_bounds` 明写。
    """
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=8.0)
    ref = straight_reference(ego, speed=8.0)
    cfg = preset("full")
    all_cands = generate(ref, ego, cfg)
    assert len(all_cands) == cfg.n_candidates == 36      # 35 网格 + 1 identity
    cands = [c for c in all_cands if not c.is_identity]
    assert len(cands) == 35

    tol = 1e-6
    for c in cands:
        # 速度剖面由前推生成，逐步施界 ⇒ 逐步 a 必然在界内
        a_step = np.diff(c.speed) / DT
        assert a_step.min() >= C.A_LON_MIN - tol, f"减速度越界 {a_step.min()}"
        assert a_step.max() <= C.A_LON_MAX + tol, f"加速度越界 {a_step.max()}"
        j_step = np.diff(a_step) / DT
        assert np.abs(j_step).max() <= C.JERK_MAX + 1e-3, f"jerk 越界 {np.abs(j_step).max()}"
        assert c.speed.min() >= -tol, "速度不得为负"
        assert c.speed.max() <= C.V_MAX + tol


def test_B_dynamics_off_does_not_enforce_bounds():
    """消融格必须**真的**关掉约束，否则 leave-one-out 测不出件① 的边际贡献。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=0.0)
    ref = straight_reference(ego, speed=20.0)
    off = generate(ref, ego, preset("no_dynamics"))
    on = generate(ref, ego, preset("full"))
    a_off = max(np.abs(np.diff(c.speed) / DT).max() for c in off)
    a_on = max(np.abs(np.diff(c.speed) / DT).max() for c in on)
    assert a_off > C.A_LON_MAX, "关掉 dynamics 后应出现越界加速度（否则开关没生效）"
    assert a_on <= C.A_LON_MAX + 1e-6


def _arc_reference(radius: float = 20.0, n: int = T) -> np.ndarray:
    theta = np.linspace(0.0, np.pi / 2, n)
    return np.stack([radius * np.sin(theta), radius * (1 - np.cos(theta))], axis=1)


def test_B_curve_speed_limited_by_lateral_accel_from_feasible_state():
    """弯道限速：v^2 * kappa <= a_lat_max。

    ⚠️ **判据细化（08-01，实现过程中记录，不是事后放宽）**：
    原判据写「任意候选全程在界内」。实测发现当**初始状态本身就越界**时
    （ego 已经以 15 m/s 冲进 R=20 m 的弯，a_lat 起步就是 10.6），
    任何满足 `A_LON_MIN` 的减速都不可能让首帧合规 —— 这是初值不可行，不是规划层的选择。
    ⇒ 拆成两条判据：本条查「初值可行时全程在界内」，
    下一条查「初值不可行时必须单调把它压回界内」。**合起来比原判据更强，不更弱。**
    """
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)   # 5^2 * 0.05 = 1.25 < 4，初值可行
    cands = generate(_arc_reference(), ego, preset("full"))
    for c in cands:
        # 容差 0.1% = **离散化残差**，已用网格收敛验证过它不是真违规：
        #   路径 60 点（生产分辨率）a_lat_max=4.8930（+0.003）；
        #   加密到 120 / 240 / 480 点后降到 3.14 / 3.18 / 3.19 并收敛。
        assert c.lat_accel.max() <= C.A_LAT_MAX * 1.001, (
            f"初值可行却越界：a_lat={c.lat_accel.max():.3f} > {C.A_LAT_MAX}"
        )


def test_B_curve_infeasible_initial_state_is_actively_reduced():
    """初值越界时，规划层必须**主动**把 a_lat 压回界内，而不是听之任之。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=15.0)  # 15^2 * 0.05 = 11.25 > 4
    # identity 候选有意豁免构造期硬界（见 `test_B_dynamics_on_all_candidates_within_bounds`）
    cands = [c for c in generate(_arc_reference(), ego, preset("full")) if not c.is_identity]
    for c in cands:
        assert c.lat_accel[0] > C.A_LAT_MAX, "构造的初值本应越界"
        over = np.nonzero(c.lat_accel > C.A_LAT_MAX)[0]
        first_ok = int(over[-1]) + 1
        # ⚠️ **判据改对了对象**（08-01 复核）：原来断言 `a_lat` 单调不增是错的 ——
        #    `a_lat = v²·kappa`，而车沿路径前进时 kappa 本身在上升（横移爬升段 + 弯道），
        #    即使车在全力减速 a_lat 也会回升。实测 [5.31, 9.69, 8.73, …, 8.09, 8.16, 8.31, …]
        #    的那几次回升全部来自 kappa，不是规划层不作为。
        #    「主动减速」的正确判据是**速度**单调不增。
        assert np.all(np.diff(c.speed[:first_ok]) <= 1e-6), (
            "越界区间内速度必须单调不增（说明在主动减速）"
        )
        # ⚠️ 窗口从 2 s 放到 3 s，理由是**物理下界**而非迁就实现：
        #    官方界 a_min=-4.05 / jerk=4.13 下，15 -> 9.89 m/s 至少需要
        #    5.11/4.05 + 4.05/4.13 ≈ 2.25 s（含 jerk 爬升段），2 s 无解。
        assert first_ok <= 30, f"用了 {first_ok * DT:.1f} s 才压回界内，超过 3 s"
        assert c.lat_accel[first_ok:].max() <= C.A_LAT_MAX + 1e-6, "回到界内后不得再越界"


# ─────────────────────────────────────────────────────────────────────
# [C] collision 开 ⇒ 选中的候选不碰；关 ⇒ 会碰
# ─────────────────────────────────────────────────────────────────────
def _blocking_scene():
    """正前方 25 m 处一辆静止车，直行必撞。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=10.0)
    ref = straight_reference(ego, speed=10.0)          # 6 s 跑 60 m ⇒ 必然穿过
    blocker = np.tile(np.array([[25.0, 0.0]]), (T, 1))[:, None, :]  # [T,1,2]
    ctx = SceneContext(agents_xy=blocker, origin=np.array([0.0, 0.0]),
                       progress_direction=np.array([1.0, 0.0]))
    return ego, ref, ctx


def test_C_collision_on_avoids_and_off_hits():
    ego, ref, ctx = _blocking_scene()

    out_on = refine(ref, ego, ctx=ctx, cfg=preset("full"))
    d_on = np.min(np.hypot(out_on[:, 0] - 25.0, out_on[:, 1] - 0.0))
    assert d_on >= C.COLLISION_RADIUS_M, f"collision 开启时仍碰撞（最小间距 {d_on:.2f}）"

    out_off = refine(ref, ego, ctx=ctx, cfg=preset("no_collision"))
    d_off = np.min(np.hypot(out_off[:, 0] - 25.0, out_off[:, 1] - 0.0))
    assert d_off < C.COLLISION_RADIUS_M, (
        f"关掉 collision 后仍未碰撞（{d_off:.2f}）⇒ 开关没生效，消融测不出件② 的贡献"
    )


# ─────────────────────────────────────────────────────────────────────
# [D] rules 开 ⇒ 不超速 / 不驶出可行驶区域
# ─────────────────────────────────────────────────────────────────────
def test_D_rules_on_respects_speed_limit():
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)
    ref = straight_reference(ego, speed=20.0)          # 参考远超限速
    ctx = SceneContext(speed_limit=8.0, origin=np.array([0.0, 0.0]),
                       progress_direction=np.array([1.0, 0.0]))
    cands = generate(ref, ego, preset("full"))
    best_on, _ = select(cands, ctx, preset("full"), ego.heading)
    best_off, _ = select(cands, ctx, preset("no_rules"), ego.heading)
    # 官方 speed_limit_compliance 的容差是 max_overspeed_value_threshold=2.23 m/s
    limit_with_tol = 8.0 + C.OVERSPEED_TOLERANCE_MPS
    assert best_on.speed.max() <= limit_with_tol + 1.0, (
        f"限速 8 m/s（官方容差 {C.OVERSPEED_TOLERANCE_MPS}），选中候选峰值 {best_on.speed.max():.2f}"
    )
    assert best_off.speed.max() > best_on.speed.max(), "关掉 rules 后应选更快的候选"


def test_D_rules_on_avoids_offroad():
    """可行驶区域只有 |y| <= 1.2 m 的走廊 ⇒ 不应选 ±1.0 之外的横向偏移。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=6.0)
    ref = straight_reference(ego, speed=6.0)

    def drivable(pts: np.ndarray) -> np.ndarray:
        return np.abs(pts[:, 1]) <= 0.4

    ctx = SceneContext(drivable_fn=drivable, origin=np.array([0.0, 0.0]),
                       progress_direction=np.array([1.0, 0.0]))
    out = refine(ref, ego, ctx=ctx, cfg=preset("full"))
    assert np.abs(out[:, 1]).max() <= 0.4 + 1e-6, "rules 开启时驶出了可行驶区域"


# ─────────────────────────────────────────────────────────────────────
# [E] 上游塌陷（输出恒零）时，progress 开 ⇒ 车仍然前进
#     这是针对「不作为失效」的靶：基准闭环 59.4% 场景出现输出头塌陷
# ─────────────────────────────────────────────────────────────────────
def test_E_collapsed_upstream_still_makes_progress():
    ego = EgoSnapshot(x=100.0, y=50.0, heading=0.3, speed=4.0)
    collapsed = np.tile(np.array([[ego.x, ego.y]]), (T, 1))   # 恒 bias ⇒ 完全不动

    out_on = refine(collapsed, ego, cfg=preset("full"))
    advanced = np.hypot(out_on[-1, 0] - ego.x, out_on[-1, 1] - ego.y)
    assert advanced > 5.0, f"塌陷上游 + progress 开启，6 s 只前进了 {advanced:.2f} m"

    out_off = refine(collapsed, ego, cfg=preset("no_progress"))
    advanced_off = np.hypot(out_off[-1, 0] - ego.x, out_off[-1, 1] - ego.y)
    assert advanced_off < advanced, (
        "关掉 progress 后前进距离应显著变小（否则测不出件④ 的贡献）"
    )


def test_E_collapsed_upstream_uses_route_geometry_when_given():
    """给了 route 中心线时，塌陷上游应沿 route 走，而不是沿 ego 朝向直线。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)
    collapsed = np.tile(np.array([[0.0, 0.0]]), (T, 1))
    # route 向左弯
    s = np.linspace(0.0, 60.0, 40)
    route = np.stack([s, 0.15 * s], axis=1)
    out = refine(collapsed, ego, cfg=preset("full"), route_xy=route)
    assert out[-1, 1] > 1.0, "未沿 route 的横向走势前进"


# ─────────────────────────────────────────────────────────────────────
# [F] 预置与 provenance 自证
# ─────────────────────────────────────────────────────────────────────
def test_F_presets_cover_the_preregistered_grid():
    names = set(available_presets())
    assert {"off", "full", "no_dynamics", "no_collision", "no_rules", "no_progress"} <= names
    for n in ("no_dynamics", "no_collision", "no_rules", "no_progress"):
        cfg = preset(n)
        flags = [cfg.dynamics, cfg.collision, cfg.rules, cfg.progress]
        assert sum(flags) == 3, f"{n} 应当且只当关掉一件缺失件"
        assert cfg.enabled is True


def test_F_provenance_is_self_describing():
    p = preset("no_collision").provenance()
    assert p["enabled"] is True and p["collision"] is False
    assert p["n_candidates"] == 36                    # 35 网格 + 1 identity（08-02）
    assert p["lon_ratios"] and p["lat_offsets"]
    # 🔴 provenance 必须自证用了哪种候选集 —— 08-01 那批七格是 include_identity=False，
    #    08-02 之后是 True，两批不可混比，靠这一项区分。
    assert p["include_identity"] is True


# ─────────────────────────────────────────────────────────────────────
# [G] 三个上游走同一份代码；CV 上游可用；确定性
# ─────────────────────────────────────────────────────────────────────
def test_G_same_module_object_via_package_path():
    """坑#18：同一文件两条 import 路径 = 两个模块对象，isinstance 会静默失败。"""
    import src.planning_layer.candidates as m1
    from src.planning_layer import candidates as m2

    assert m1 is m2
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=3.0)
    assert isinstance(ego, m1.EgoSnapshot) and isinstance(ego, m2.EgoSnapshot)


def test_G_constant_velocity_upstream_is_usable():
    ego = EgoSnapshot(x=5.0, y=5.0, heading=np.pi / 4, speed=7.0)
    cfg = preset("full")
    cv = constant_velocity_reference(ego, cfg)
    assert cv.shape == (cfg.horizon, 2)
    travelled = np.hypot(cv[-1, 0] - ego.x, cv[-1, 1] - ego.y)
    assert abs(travelled - 7.0 * (cfg.horizon - 1) * cfg.dt) < 1e-6
    out = refine(cv, ego, cfg=cfg)
    assert out.shape == (cfg.horizon, 2)


def test_G_selection_is_deterministic():
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=6.0)
    ref = straight_reference(ego, speed=6.0)
    ctx = SceneContext(origin=np.array([0.0, 0.0]), progress_direction=np.array([1.0, 0.0]))
    outs = [refine(ref, ego, ctx=ctx, cfg=preset("full")) for _ in range(3)]
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) == 0.0


def test_G_breakdown_matches_total():
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=6.0)
    ref = straight_reference(ego, speed=6.0)
    cfg = preset("full")
    ctx = SceneContext(origin=np.array([0.0, 0.0]), progress_direction=np.array([1.0, 0.0]))
    c = generate(ref, ego, cfg)[0]
    b = score_breakdown(c, ctx, cfg, ego.heading)
    manual = (
        C.W_PROGRESS * b["progress"]
        - C.W_COLLISION * b["collision"]
        - C.W_RULES * b["rules"]
        - C.W_COMFORT * b["comfort"]
    )
    assert abs(manual - b["total"]) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# [H] 08-01 复核（对照 nuPlan 官方 metric 实现）新增的回归判据
#     四条都对应一个已复现的真 bug，不是假想
# ─────────────────────────────────────────────────────────────────────
def test_H_bounds_match_official_comfort_thresholds():
    """运动学界必须**不越过**官方 comfort 判据，否则「可行」候选会被判 uncomfortable。"""
    assert C.A_LON_MAX <= 2.40 + 1e-9      # ego_lon_acceleration_statistics.yaml
    assert C.A_LON_MIN >= -4.05 - 1e-9
    assert C.JERK_MAX <= 4.13 + 1e-9       # ego_lon_jerk_statistics.yaml
    assert C.A_LAT_MAX <= 4.89 + 1e-9      # ego_lat_acceleration_statistics.yaml
    assert C.YAW_RATE_MAX <= 0.95 + 1e-9   # ego_yaw_rate_statistics.yaml


def test_H_trajectory_starts_at_ego_not_one_step_ahead():
    """轨迹首点必须≈ego 当前位置；第一版 `cumsum(v)*dt` 让整条轨迹超前一步（0.6 m）。"""
    ego = EgoSnapshot(x=12.0, y=-4.0, heading=0.7, speed=6.0)
    ref = straight_reference(ego, speed=6.0)
    for c in generate(ref, ego, preset("full")):
        d = np.hypot(c.xy[0, 0] - ego.x, c.xy[0, 1] - ego.y)
        assert d < 1e-6, f"候选首点偏离 ego {d:.4f} m（应为 0）"


def test_H_route_fallback_does_not_teleport_ego_backwards():
    """上游塌陷 + route 起点在 ego 后方时，绝不能把 ego 瞬移回 route 起点。"""
    ego = EgoSnapshot(x=50.0, y=0.0, heading=0.0, speed=5.0)
    collapsed = np.tile(np.array([[50.0, 0.0]]), (T, 1))
    route = np.stack([np.linspace(0.0, 200.0, 50), np.zeros(50)], axis=1)  # 起于 x=0
    out = refine(collapsed, ego, cfg=preset("full"), route_xy=route)
    assert abs(out[0, 0] - ego.x) < 1e-6, f"首点 x={out[0,0]:.3f}，ego 在 x=50 —— 发生瞬移"
    assert out[-1, 0] > ego.x, "应沿 route 前进而不是后退"


def test_H_backward_reference_does_not_earn_progress():
    """参考朝后时，进展项必须为负 —— 不得让每个候选按自己方向计分。"""
    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=0.0)
    t = np.arange(T, dtype=float) * DT
    ref_back = np.stack([-6.0 * t, 0.0 * t], axis=1)
    ctx = SceneContext()                      # 故意不给 progress_direction，走兜底
    vals = [
        __import__("src.planning_layer.scoring", fromlist=["x"]).progress_term(c, ctx, ego.heading)
        for c in generate(ref_back, ego, preset("full"))
    ]
    assert min(vals) < -1.0, "倒退参考应产生负进展"
    assert max(vals) <= 1e-6, f"倒退参考却拿到正进展 {max(vals):.2f}"


def test_H_agents_shape_is_validated_with_clear_message():
    """agent 前推步数不匹配时必须报清楚的错，而不是让 numpy 在别处抛广播错。"""
    from src.planning_layer.scoring import collision_term

    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)
    c = generate(straight_reference(ego), ego, preset("full"))[0]
    with pytest.raises(ValueError, match="agents_xy"):
        collision_term(c, SceneContext(agents_xy=np.zeros((30, 1, 2))))


def test_H_ttc_term_matches_official_window():
    """TTC 项只看官方的 3 s 窗口：窗口外的接近不应被罚。"""
    from src.planning_layer.scoring import ttc_term

    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=10.0)
    c = generate(straight_reference(ego, speed=10.0), ego, preset("full"))[0]
    # 迎面而来、3 s 内就贴到 ego：应被罚
    near = np.stack([np.stack([np.full(T, 12.0) - 0.0 * np.arange(T),
                               np.zeros(T)], axis=1)], axis=1)
    # 远在 150 m 外静止：3 s 窗口内够不着，不应被罚
    far = np.stack([np.stack([np.full(T, 150.0), np.zeros(T)], axis=1)], axis=1)
    assert ttc_term(c, SceneContext(agents_xy=near)) > 0.0
    assert ttc_term(c, SceneContext(agents_xy=far)) == 0.0


def test_H_drivable_uses_four_corners_when_dims_given():
    """给了车身尺寸时，可行驶区域必须按四角查（官方口径），不是只查质心。"""
    from src.planning_layer.scoring import rules_term

    ego = EgoSnapshot(x=0.0, y=0.0, heading=0.0, speed=5.0)
    # 必须取**不带横向偏移**的候选，否则候选自身就偏出走廊，测的就不是四角了
    c = [x for x in generate(straight_reference(ego), ego, preset("full"))
         if x.lat_offset == 0.0][0]
    # 走廊只有 |y| <= 0.5，车半宽 1.0 ⇒ 质心合规但车角必然出界
    def drivable(pts: np.ndarray) -> np.ndarray:
        return np.abs(pts[:, 1]) <= 0.5

    center_only = rules_term(c, SceneContext(drivable_fn=drivable), ego.heading)
    with_corners = rules_term(
        c,
        SceneContext(drivable_fn=drivable, ego_half_width=1.0, ego_front_m=3.0, ego_rear_m=1.0),
        ego.heading,
    )
    assert center_only == 0.0, "质心确实在走廊内"
    assert with_corners > 0.0, "按四角查应当检出越界"


# ─────────────────────────────────────────────────────────────────────
# [I] identity 候选 —— argmax 的下界保证（08-02 新增，见阶段52 根因分析）
#
# 背景：网格里的 `(lon_ratio=1.0, lat_offset=0.0)` **不是**上游原轨迹 ——
# 它只保留路径形状，速度剖面被 `_rollout_speed_profile` 从头重新生成。
# 实测只有上游恰好匀速时两者才相等：减速停车差 4.24 m、起步加速差 8.65 m、
# **塌陷差 15.03 m**（塌陷时还有 `v_ref := max(ego.speed, 3.0)` 兜底）。
# ⇒ 候选集里没有「什么都不做」这个选项，argmax 可以选出比上游更差的轨迹。
# ─────────────────────────────────────────────────────────────────────
def _collapsed_reference(n: int = 60):
    """MAGNET 的主导失效模式：输出恒 bias ⇒ 参考轨迹几乎不动。"""
    return np.stack([np.linspace(0.0, 0.2, n), np.zeros(n)], axis=1)


def test_I1_identity_candidate_is_bitwise_the_input():
    """🔴 命门：identity 候选必须与输入**逐位相同**，否则它不是「不干预」。"""
    cfg = preset("full")
    for ref, ego in [
        (_collapsed_reference(), EgoSnapshot(0.0, 0.0, 0.0, 0.0)),
        (straight_reference(EgoSnapshot(0.0, 0.0, 0.0, 8.0), speed=8.0),
         EgoSnapshot(0.0, 0.0, 0.0, 8.0)),
    ]:
        ident = [c for c in generate(ref, ego, cfg) if c.is_identity]
        assert len(ident) == 1, "identity 候选必须恰好一个"
        assert float(np.max(np.abs(ident[0].xy - ref))) == 0.0, "identity 必须逐位等于输入"


def test_I2_identity_wins_when_all_grid_candidates_collide():
    """identity 获胜时，`refine()` 必须**逐位返回输入** ⇒ 层与层 OFF 打平。

    构造：ego 静止 + 参考塌陷 + **正前方近处有 agent**。
    网格候选一律前冲（塌陷时 `v_ref` 兜底 3 m/s）⇒ 全部撞上；
    identity 原地不动 ⇒ 不撞。碰撞权重 4.0 足以压过进展权重 1.0。
    """
    cfg = preset("full")
    ref = _collapsed_reference()
    ego = EgoSnapshot(0.0, 0.0, 0.0, 0.0)
    n = len(ref)
    # agent 静止停在正前方 6 m（网格候选推进 4~21 m ⇒ 必然穿过它）
    agents = np.tile(np.array([[6.0, 0.0]]), (n, 1, 1))
    ctx = SceneContext(
        agents_xy=agents,
        origin=np.array([0.0, 0.0]),
        progress_direction=np.array([1.0, 0.0]),
    )
    cands = generate(ref, ego, cfg)
    best, _ = select(cands, ctx, cfg, ego.heading)
    assert best.is_identity, (
        "所有网格候选都撞上时必须回落到 identity —— 否则规划层会主动选择碰撞"
    )
    out = refine(ref, ego, ctx=ctx, cfg=cfg)
    assert float(np.max(np.abs(out - ref))) == 0.0, "identity 获胜 ⇒ 输出必须逐位等于输入"


def test_I3_identity_is_exempt_from_construction_bounds():
    """identity **有意**豁免构造期动力学硬界 —— 它就是上游给的那条。

    这一条把豁免写死成判据：如果哪天有人给 identity 也加上构造期投影，
    它就不再是「原样」候选，argmax 的下界保证随之失效，本测试会失败。
    """
    cfg = preset("full")
    # 违反 a_lon 的参考：前 30 帧静止，之后每帧位移 3 m ⇒ 速度一步从 0 跳到 30 m/s
    # （⚠️ 等步长的轨迹速度恒定、加速度为 0，测不到这条判据 —— 第一版就写错成那样）
    x = np.concatenate([np.zeros(30), np.arange(1, 31, dtype=float) * 3.0])
    ref = np.stack([x, np.zeros(60)], axis=1)
    ego = EgoSnapshot(0.0, 0.0, 0.0, 0.0)
    ident = [c for c in generate(ref, ego, cfg) if c.is_identity][0]
    a_step = np.diff(ident.speed) / DT
    assert float(np.max(np.abs(a_step))) > C.A_LON_MAX, (
        "本例的 identity 应当越界（说明它确实没被投影）"
    )
    assert float(np.max(np.abs(ident.xy - ref))) == 0.0, "越界也必须原样保留"


def test_I4_include_identity_false_reproduces_pre_fix_behaviour():
    """`include_identity=False` 必须**逐位复现** 08-01 那批七格的行为。

    没有这条就无法做「修前 / 修后」对照 —— 两批分数会失去可比性。
    """
    import dataclasses

    cfg_new = preset("full")
    cfg_old = dataclasses.replace(cfg_new, include_identity=False)
    ref = _collapsed_reference()
    ego = EgoSnapshot(0.0, 0.0, 0.0, 0.0)
    ctx = SceneContext(origin=np.array([0.0, 0.0]), progress_direction=np.array([1.0, 0.0]))

    old = generate(ref, ego, cfg_old)
    new = generate(ref, ego, cfg_new)
    assert len(old) == 35 and len(new) == 36
    assert cfg_old.n_candidates == 35 and cfg_new.n_candidates == 36
    for a, b in zip(old, new):                       # 网格候选逐位不变
        assert float(np.max(np.abs(a.xy - b.xy))) == 0.0
    # 关掉时 `refine()` 与修复前完全一致（该场景修复前选 ratio=1.5）
    out_old = refine(ref, ego, ctx=ctx, cfg=cfg_old)
    assert float(np.max(np.abs(out_old - ref))) > 1.0, "修前行为：会大幅偏离输入"
