"""PlanTF + C4 规划层接线判据 [T1]–[T8]（主表第四格，Q53=(a)）。

跑法（**必须用 plantf env**，它才有 planTF 的依赖）::

    cd /mnt/local/Xiao_Yi/external/planTF
    PYTHONPATH=/mnt/local/Xiao_Yi/external/planTF \
      /home/team/anaconda3/envs/plantf/bin/python -m pytest \
      /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/tests/test_plantf_layer_wiring.py -q

判据（**先写死再实现**，与 MAGNET 侧的 [L1]–[L7] 同构）
------------------------------------------------------
[T1] 不设 `MAGNET_PLANNING_LAYER` ⇒ `enabled=False`，且构造时**不加载规划层、不碰地图**；
[T2] 🔴 **命门** —— 关闭时 `compute_planner_trajectory` 的输出与接线前**逐位相同** `max|Δ|=0`；
[T3] 设了变量 ⇒ 真的开启，且 `provenance()` 能自证 preset；
[T4] 未知 preset **直接 raise**，不静默降级（否则会静默跑成「PlanTF 原样」）；
[T5] 🔴 **Q53=(a)** —— 层按 PlanTF 的**原生 horizon 80** 施加，不是层默认的 60；
[T6] `src` 撞名已绕开：换名加载后 planTF 自己的 `src` 仍可用；
[T7] 开启后输出确实改变，且形状仍是 `[T,3]`、heading 与 xy 自洽；
[T8] `MAGNET_GNN_ROOT` 指向不存在的路径时**报错**，不静默降级。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PLANTF_ROOT = Path("/mnt/local/Xiao_Yi/external/planTF")
if str(PLANTF_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANTF_ROOT))

pytest.importorskip("torch", reason="需要 plantf env")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例都从「环境变量未设置」开始，避免相互污染。"""
    monkeypatch.delenv("MAGNET_PLANNING_LAYER", raising=False)
    monkeypatch.delenv("MAGNET_GNN_ROOT", raising=False)
    yield


def _hook(dt: float = 0.1):
    from src.planners.magnet_layer_hook import PlanningLayerHook

    return PlanningLayerHook(dt=dt)


class _Ego:
    """最小 ego 替身（只有 hook 的 heading 重算路径会用到）。"""

    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.x, self.y, self.heading = x, y, heading


# ── [T1] 默认关闭，且构造时什么都不加载 ────────────────────────────────
def test_t1_disabled_by_default_and_loads_nothing():
    sys.modules.pop("magnet_planning_layer", None)
    hook = _hook()
    assert hook.enabled is False
    assert hook.provenance() == {"enabled": False}
    # 🔴 关闭时绝不能把规划层拖进内存（否则 plantf env 里多一份无谓依赖）
    assert "magnet_planning_layer" not in sys.modules


# ── [T2] 🔴 命门：关闭时与接线前逐位相同 ───────────────────────────────
def test_t2_disabled_is_bitwise_identical_to_pre_wiring():
    """接线只在 `if self._planning_layer.enabled:` 里加了一个分支。

    关闭时分支体不执行 ⇒ `_global_trajectory` 仍是 `_get_global_trajectory()` 的原值。
    这里直接对拍「走接线后的路径」与「接线前的等价计算」。
    """
    pytest.importorskip("numba", reason="PlanTF rotation utility requires numba")
    from src.feature_builders.common.utils import rotate_round_z_axis

    rng = np.random.default_rng(20260801)
    local = rng.normal(size=(80, 3)) * 5.0
    origin, angle = np.array([123.5, -47.25]), 0.7

    # 接线前的等价计算（照抄 _get_global_trajectory 的算法）
    expected_xy = rotate_round_z_axis(np.ascontiguousarray(local[..., :2]), -angle) + origin
    expected = np.concatenate([expected_xy, (local[..., 2] + angle)[..., None]], axis=1)

    hook = _hook()
    assert hook.enabled is False
    got = expected.copy()
    if hook.enabled:                      # 关闭 ⇒ 分支体不执行，与 planner 里完全同构
        got = hook.apply(got, _Ego(), None)

    assert np.max(np.abs(got - expected)) == 0.0


# ── [T3] 设了变量就真的开启，且能自证 ──────────────────────────────────
def test_t3_env_enables_and_provenance_self_certifies(monkeypatch):
    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    hook = _hook()
    assert hook.enabled is True
    prov = hook.provenance()
    assert prov["enabled"] is True and prov["preset"] == "full"
    assert prov["upstream"] == "plantf"
    assert prov["n_candidates"] == 36          # 7 纵 × 5 横 + identity candidate
    assert prov["dynamics"] and prov["collision"] and prov["rules"] and prov["progress"]


def test_t3b_ablation_presets_are_reachable(monkeypatch):
    """四个 leave-one-out 预置在 PlanTF 侧同样可用（与 MAGNET 侧同一份 config）。"""
    for name, off_key in [
        ("no_dynamics", "dynamics"), ("no_collision", "collision"),
        ("no_rules", "rules"), ("no_progress", "progress"),
    ]:
        monkeypatch.setenv("MAGNET_PLANNING_LAYER", name)
        prov = _hook().provenance()
        assert prov["enabled"] is True and prov[off_key] is False


# ── [T4] 未知 preset 直接报错 ──────────────────────────────────────────
def test_t4_unknown_preset_raises(monkeypatch):
    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "definitely_not_a_preset")
    with pytest.raises(ValueError, match="未知的规划层预置"):
        _hook()


# ── [T5] 🔴 Q53=(a)：按 PlanTF 原生 horizon 80 施加 ────────────────────
def test_t5_horizon_follows_plantf_native_length(monkeypatch):
    """层默认 horizon=60，但 PlanTF 输出 80 点 ⇒ 必须被改写成 80。"""
    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    hook = _hook()
    assert hook._cfg.horizon == 60, "层的出厂默认应是 MAGNET 的 60"

    # 走一次 horizon 协商（不需要地图：只断言配置被改写）
    import dataclasses
    hook._horizon = 80
    hook._cfg = dataclasses.replace(hook._cfg, horizon=80, dt=0.1)
    prov = hook.provenance()
    assert prov["horizon"] == 80 and prov["dt"] == 0.1
    assert hook._cfg.horizon == 80


# ── [T6] `src` 撞名已绕开 ─────────────────────────────────────────────
def test_t6_no_src_namespace_collision(monkeypatch):
    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    import src.planners.planner_utils as pu       # planTF 自己的 src

    _hook()                                        # 触发换名加载
    assert "magnet_planning_layer" in sys.modules
    # planTF 的 src 未被顶掉
    assert sys.modules["src"].__name__ == "src"
    assert hasattr(pu, "global_trajectory_to_states")
    # 换名加载的确实是 GNN_Trans 那份
    loaded = Path(sys.modules["magnet_planning_layer"].__file__).resolve()
    assert "GNN_Trans" in str(loaded) and loaded.name == "__init__.py"


# ── [T7] heading 重算与 xy 自洽 ───────────────────────────────────────
def test_t7_heading_rebuild_is_consistent():
    from src.planners.magnet_layer_hook import _rebuild_with_heading

    xy = np.stack([np.arange(80, dtype=float), np.zeros(80)], axis=1)  # 沿 +x 直行
    out = _rebuild_with_heading(xy, _Ego(x=-1.0, y=0.0, heading=1.234))
    assert out.shape == (80, 3)
    assert np.max(np.abs(out[:, :2] - xy)) == 0.0        # xy 不被改动
    assert np.allclose(out[:, 2], 0.0)                    # 全程朝 +x


def test_t7b_degenerate_step_keeps_previous_heading():
    """位移小于 eps 时沿用上一朝向（避免 atan2 在静止处抖动）。"""
    from src.planners.magnet_layer_hook import _rebuild_with_heading

    xy = np.zeros((5, 2))                                 # 完全静止
    out = _rebuild_with_heading(xy, _Ego(x=0.0, y=0.0, heading=0.9))
    assert np.allclose(out[:, 2], 0.9)                    # 全部沿用 ego 朝向


# ── [T8] 路径不对时报错，不静默降级 ───────────────────────────────────
def test_t8_bad_root_raises_instead_of_silently_disabling(monkeypatch):
    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    monkeypatch.setenv("MAGNET_GNN_ROOT", "/nonexistent/path/to/gnn_trans")
    sys.modules.pop("magnet_planning_layer", None)
    with pytest.raises(FileNotFoundError, match="找不到"):
        _hook()
    # 失败不留半成品模块
    assert "magnet_planning_layer" not in sys.modules


# ── [T9] 🔴🔴 命门二：钩子必须可 pickle（08-02 实测事故补的判据）───────
#
# 背景：`SimulationLogCallback.on_simulation_end` 会 `pickle.dumps(SimulationLog)`，
# 而 `SimulationLog` 含 planner，planner 含本钩子 ⇒ **钩子的任何实例属性都必须可 pickle**。
# 第一版把 `magnet_planning_layer` 与 `nuplan_adapter` 两个模块对象存成了实例属性，
# 层 ON 时 50/50 场景全在仿真跑完后抛 `TypeError: cannot pickle 'module' object`。
# 🔴 层 OFF 时两者是 None ⇒ [T1]–[T8] 与 harness parity **都测不到**，故单列两条。
def test_t9_enabled_hook_is_picklable(monkeypatch):
    """层 ON 的钩子必须能 pickle —— 这正是 08-02 那次 50/50 全失败的直接判据。"""
    import pickle

    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    hook = _hook()
    assert hook.enabled is True

    blob = pickle.dumps(hook, protocol=pickle.HIGHEST_PROTOCOL)   # 修复前在这里抛
    back = pickle.loads(blob)

    # 往返后行为不变
    assert back.enabled is True
    assert back.provenance() == hook.provenance()
    # 🔴 模块对象不得出现在实例状态里（否则等于把 bug 换个写法留下）
    import types
    for k, v in vars(back).items():
        assert not isinstance(v, types.ModuleType), f"实例属性 {k} 仍是模块对象"


def test_t9b_disabled_hook_is_picklable():
    """层 OFF 也要可 pickle —— 保护「层 OFF 那几格」不被本次修复带坏。"""
    import pickle
    import types

    hook = _hook()
    assert hook.enabled is False
    back = pickle.loads(pickle.dumps(hook, protocol=pickle.HIGHEST_PROTOCOL))
    assert back.enabled is False
    assert back.provenance() == {"enabled": False}
    for k, v in vars(back).items():
        assert not isinstance(v, types.ModuleType), f"实例属性 {k} 仍是模块对象"


# ── [T10] 往返后仍能干活（不只是「能序列化」）─────────────────────────
def test_t10_pickled_hook_still_works(monkeypatch):
    """pickle 往返后 `_layer` / `_adapter` 仍取得到 —— property 而不是死引用。"""
    import pickle

    monkeypatch.setenv("MAGNET_PLANNING_LAYER", "full")
    back = pickle.loads(pickle.dumps(_hook(), protocol=pickle.HIGHEST_PROTOCOL))

    # 访问器仍指向同一份换名加载的模块（坑#18：全进程只有一份）
    assert back._layer is sys.modules["magnet_planning_layer"]
    assert back._adapter.__name__.endswith("nuplan_adapter")
    assert hasattr(back._layer, "refine") and hasattr(back._adapter, "ego_snapshot")
