"""C4 规划层 → planner 接线单测（判据 [L]）—— 核心是「不开就等于没接」。

为什么这条判据最重要
--------------------
C4 主表的「层 OFF」两格（MAGNET 0.198 / PlanTF 0.356）是**直接复用已有结果**、不重跑的。
只要接线动了默认路径哪怕一个 bit，这两格就不再是同一个实验，整张主表作废。
所以判据 [L2] 直接把 `_output_to_trajectory` 的输出与**接线前的算法**逐位比对，
要求 `max|Δ| == 0`（照抄 Q46 探针判据 [A] 的写法）。

判据清单（先于实现写死，见 `C4_规划形式化_设计_20260801.md` §3.3 第 4 条）：

| 判据 | 内容 |
|---|---|
| [L1] | 默认关闭：不设环境变量 / 不传 yaml 参数时，规划层与 CV 上游都不生效 |
| [L2] | **命门**：关闭时轨迹与接线前算法逐位相同 |
| [L3] | 环境变量可开启；yaml 参数优先；未知 preset 直接 raise（不静默降级） |
| [L4] | 开启后输出确实变了（证明开关真的接上了），且首点仍严格等于 ego |
| [L5] | `MAGNET_REFERENCE_SOURCE=cv` 真的旁路模型（换模型输出，轨迹不变） |
| [L6] | 官方 `build_planners` 路径可用 —— C4 起跑前的唯一防线（Q46 已吃过一次亏） |
| [L7] | 规划层时间栅格与 planner 失配时 raise，不静默对齐 |

用法::

    /home/team/anaconda3/envs/nuplan/bin/python \\
        scripts/eval_closed_loop/tests/test_planning_layer_wiring.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/local/Xiao_Yi/nuplan-devkit")
sys.path.insert(0, "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")

from nuplan.common.actor_state.ego_state import EgoState  # noqa: E402
from nuplan.common.actor_state.state_representation import (  # noqa: E402
    StateSE2, StateVector2D, TimePoint,
)
from nuplan.common.actor_state.tracked_objects import TrackedObjects  # noqa: E402
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # noqa: E402
from nuplan.planning.simulation.planner.gnn_trans_planner import (  # noqa: E402
    GNNTransPlanner, _increments_to_local_pos)

_ENV_VARS = ("MAGNET_PLANNING_LAYER", "MAGNET_REFERENCE_SOURCE")


class _FakeMapApi:
    """最小地图：没有 lane、没有 roadblock ⇒ 限速 None、route None。

    接线本身与地图内容无关；地图内容的正确性由 `test_nuplan_adapter.py` 的 [J]/[K] 判据管。
    """

    def get_all_map_objects(self, point, layer):  # noqa: ANN001
        return []

    def get_proximal_map_objects(self, point, radius, layers):  # noqa: ANN001
        return {layer: [] for layer in layers}


class _FakeHistory:
    def __init__(self, ego_state: EgoState):
        self.ego_states = [ego_state]
        self.observations = [DetectionsTracks(TrackedObjects([]))]


class _FakeInput:
    def __init__(self, ego_state: EgoState):
        self.history = _FakeHistory(ego_state)


def _ego_state(x: float = 100.0, y: float = -50.0, heading: float = 0.7,
               speed: float = 6.0) -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(1_600_000_000_000_000),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True,
        angular_vel=0.0,
        angular_accel=0.0,
    )


def _model_output(seed: int = 0, step: float = 0.6) -> torch.Tensor:
    """[1,120] 局部逐帧增量，首帧 [0,0]（与训练契约一致）。"""
    g = torch.Generator().manual_seed(seed)
    inc = torch.zeros(60, 2)
    inc[1:, 0] = step
    inc[1:, 1] = 0.02 * torch.randn(59, generator=g)
    return inc.reshape(1, -1)


def _local_pos(seed: int = 0, step: float = 0.6) -> np.ndarray:
    """[60,2] 局部累计位移 —— `_output_to_trajectory` 在 2026-08-25 之后的入口口径。

    🔴 `_golden_trajectory` 仍吃**原始增量**（它复刻的是接线前的算法），所以判据 [L2]
    比的依然是「接线前 vs 现在」，抬口径这件事本身也被这条判据罩住。
    """
    return _increments_to_local_pos(_model_output(seed=seed, step=step), 60)


def _make_planner(**kwargs) -> GNNTransPlanner:
    p = GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu", **kwargs)
    p._map_api = _FakeMapApi()
    p._route_roadblock_ids = []
    return p


def _golden_trajectory(model_output: torch.Tensor, current_state: EgoState) -> np.ndarray:
    """**接线前**的算法，逐字照抄（局部增量 → cumsum → 旋转回全局）。

    刻意不复用生产代码：golden 必须是独立第二实现，否则「逐位一致」只证明了自己等于自己。
    """
    inc = model_output.detach().cpu().numpy().reshape(-1, 2)
    local = np.cumsum(inc, axis=0)
    h = current_state.rear_axle.heading
    cos_h, sin_h = np.cos(h), np.sin(h)
    ox, oy = current_state.rear_axle.x, current_state.rear_axle.y
    gx = local[:, 0] * cos_h - local[:, 1] * sin_h + ox
    gy = local[:, 0] * sin_h + local[:, 1] * cos_h + oy

    # t0 那一项是 `current_state` **本身**（不是重算的），故速度取它自己的值，
    # 不是 0 —— 第一版 golden 写死 0.0，抓出 max|Δ|=6.0 的假失败。
    out = [(ox, oy, h, float(current_state.dynamic_car_state.rear_axle_velocity_2d.x))]
    prev_x, prev_y, prev_h = ox, oy, h
    for i in range(1, 60):
        x, y = float(gx[i]), float(gy[i])
        dx, dy = x - prev_x, y - prev_y
        dist = float(np.hypot(dx, dy))
        heading = float(np.arctan2(dy, dx)) if dist > 1e-3 else prev_h
        out.append((x, y, heading, dist / 0.1))
        prev_x, prev_y, prev_h = x, y, heading
    return np.asarray(out, dtype=float)


def _as_array(traj) -> np.ndarray:
    states = traj.get_sampled_trajectory()
    return np.asarray([
        (s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading,
         s.dynamic_car_state.rear_axle_velocity_2d.x)
        for s in states
    ], dtype=float)


def _clear_env() -> dict:
    old = {k: os.environ.pop(k, None) for k in _ENV_VARS}
    return old


def _restore_env(old: dict) -> None:
    for k, v in old.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────
def test_L1_default_is_off() -> None:
    """判据 [L1]：默认必须完全关闭 —— 这是「层 OFF 两格可复用」的前提。"""
    old = _clear_env()
    try:
        p = _make_planner()
        assert p._planning_layer_cfg.enabled is False, "规划层默认必须关闭"
        assert p._reference_source == "model", "上游默认必须是模型"
        assert p._drivable is None and p._route_centerline is None, "关闭时不得碰地图"
    finally:
        _restore_env(old)
    print("[L1] 默认关闭                      PASS")


def test_L2_disabled_path_is_bitwise_identical() -> None:
    """判据 [L2]（命门）：关闭时输出与接线前算法**逐位相同**。"""
    old = _clear_env()
    try:
        p = _make_planner()
        st = _ego_state()
        out = _as_array(p._output_to_trajectory(_local_pos(), _FakeInput(st)))
        ref = _golden_trajectory(_model_output(), st)
        assert out.shape == ref.shape, f"点数变了：{out.shape} vs {ref.shape}"
        delta = float(np.max(np.abs(out - ref)))
        assert delta == 0.0, f"接线改变了默认路径 max|Δ|={delta:.3e}"
    finally:
        _restore_env(old)
    print("[L2] 关闭时逐位不变               PASS  max|Δ|=0.0")


def test_L3_env_var_and_yaml_precedence() -> None:
    """判据 [L3]：环境变量可开启；yaml 参数优先；未知 preset **直接 raise**。"""
    old = _clear_env()
    try:
        os.environ["MAGNET_PLANNING_LAYER"] = "full"
        p = _make_planner()
        assert p._planning_layer_cfg.enabled is True
        assert p._planning_layer_cfg.progress is True
        p2 = _make_planner(planning_layer="no_progress")
        assert p2._planning_layer_cfg.progress is False, "yaml 参数必须优先于环境变量"

        os.environ["MAGNET_PLANNING_LAYER"] = "definitely_not_a_preset"
        try:
            _make_planner()
        except ValueError:
            pass
        else:
            raise AssertionError("未知 preset 必须 raise，不得静默降级成关闭")
    finally:
        _restore_env(old)
    print("[L3] env 可开 + yaml 优先 + 未知即抛 PASS")


def test_L4_enabled_changes_output_and_keeps_first_point() -> None:
    """判据 [L4]：开启后输出确实变了，且首点仍严格等于 ego（控制器在 t0 不得见阶跃）。"""
    old = _clear_env()
    try:
        st = _ego_state()
        base = _as_array(_make_planner()._output_to_trajectory(_local_pos(), _FakeInput(st)))
        os.environ["MAGNET_PLANNING_LAYER"] = "full"
        on = _as_array(_make_planner()._output_to_trajectory(_local_pos(), _FakeInput(st)))
        assert float(np.max(np.abs(on - base))) > 0.0, "开了层却什么都没变 ⇒ 开关没接上"
        assert on[0, 0] == base[0, 0] and on[0, 1] == base[0, 1], "首点必须仍是 ego 当前位姿"
        assert np.all(np.isfinite(on)), "开层后轨迹里不得出现 NaN/Inf"
    finally:
        _restore_env(old)
    print("[L4] 开启后确实改变输出           PASS")


def test_L5_cv_reference_bypasses_the_model() -> None:
    """判据 [L5]：`reference_source=cv` 必须真的旁路模型 —— 换模型输出，轨迹不变。

    🔴 这是 C4 的诚实性支点（设计文档 §4.1）：CV 那一列若与 MAGNET 不可区分，
    结论只能写「学习组件无可测量的额外贡献」。前提是 CV 真的没用到模型。
    """
    old = _clear_env()
    try:
        os.environ["MAGNET_REFERENCE_SOURCE"] = "cv"
        st = _ego_state()
        a = _as_array(_make_planner()._output_to_trajectory(_local_pos(seed=0), _FakeInput(st)))
        b = _as_array(_make_planner()._output_to_trajectory(_local_pos(seed=7, step=2.0),
                                                           _FakeInput(st)))
        delta = float(np.max(np.abs(a - b)))
        assert delta == 0.0, f"CV 上游仍受模型输出影响 max|Δ|={delta:.3e}"
        # 常速外推：6 m/s × 5.9 s ≈ 35.4 m，沿 ego 朝向
        span = float(np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1]))
        assert 30.0 < span < 40.0, f"CV 外推距离不合理：{span:.2f} m"

        os.environ["MAGNET_REFERENCE_SOURCE"] = "not_a_source"
        try:
            _make_planner()
        except ValueError:
            pass
        else:
            raise AssertionError("非法 reference_source 必须 raise")
    finally:
        _restore_env(old)
    print("[L5] CV 上游真的旁路模型          PASS")


def test_L6_official_build_planners_path() -> None:
    """判据 [L6]：官方 `build_planners` 路径可用 —— C4 起跑前的唯一防线。"""
    old = _clear_env()
    try:
        from omegaconf import OmegaConf
        from nuplan.planning.script.builders.planner_builder import build_planners

        cfg = OmegaConf.load(
            "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/planning/script/config/"
            "simulation/planner/gnn_trans_planner.yaml"
        )

        class FakeScenario:
            token = "deadbeef12345678"
            log_name = "some_log"
            scenario_type = "following_lane_with_lead"

        p = build_planners(cfg, FakeScenario())[0]
        assert type(p).__name__ == "GNNTransPlanner"
        assert p._planning_layer_cfg.enabled is False, "官方路径下默认也必须关闭"
        assert p._reference_source == "model"
    finally:
        _restore_env(old)
    print("[L6] 官方 build_planners 路径可用 PASS")


def test_L7_grid_mismatch_raises() -> None:
    """判据 [L7]：规划层与 planner 的时间栅格失配必须 raise，不静默对齐。"""
    old = _clear_env()
    try:
        os.environ["MAGNET_PLANNING_LAYER"] = "full"
        try:
            _make_planner(horizon_seconds=3.0)   # 30 点 ≠ 层的 60 点
        except ValueError as exc:
            assert "时间栅格失配" in str(exc)
        else:
            raise AssertionError("栅格失配必须 raise")
        # 关闭时不校验（默认路径一行都不执行）
        os.environ.pop("MAGNET_PLANNING_LAYER")
        _make_planner(horizon_seconds=3.0)
    finally:
        _restore_env(old)
    print("[L7] 栅格失配即抛                 PASS")


if __name__ == "__main__":
    tests = [
        test_L1_default_is_off,
        test_L2_disabled_path_is_bitwise_identical,
        test_L3_env_var_and_yaml_precedence,
        test_L4_enabled_changes_output_and_keeps_first_point,
        test_L5_cv_reference_bypasses_the_model,
        test_L6_official_build_planners_path,
        test_L7_grid_mismatch_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[{t.__name__}] FAIL  {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} PASS")
    sys.exit(1 if failed else 0)
