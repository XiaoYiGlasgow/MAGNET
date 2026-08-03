"""C4 判据 [M]：候选生成的**性能优化必须数值等价**。

背景（08-01 接线时实测）：`refine()` 单步 **109.6 ms** ⇒ 一格闭环从 80 min 涨到约 150 min，
8 格主表+消融会从预算的 ≈13 h 涨到 ≈20 h。于是对 `candidates.py` 做了三处纯提速改写：

  1. 前瞻窗口的 6 次标量 `np.interp` → 1 次向量化 `np.interp`（查询点完全相同）；
  2. 逐点曲率的 60 次标量 `np.interp` → 1 次向量化（同上）；
  3. 滚动内部的标量 `np.sqrt`/`np.clip` → `math.sqrt`/`min`+`max`
     （IEEE 双精度下语义相同，只是不再走 numpy 往返）。

🔴 **候选网格与所有阈值一个都没动** —— 那些是事前定死的，改动须走干预协议 §11。
本测试把优化前的 `candidates.py`（冻结副本）与现版本在多组场景上**逐位对拍**，
要求 `max|Δ| == 0`。若哪天要再提速，先让这条判据继续过。

跑法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest \\
        src/planning_layer/tests/test_candidates_opt_parity.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.planning_layer import EgoSnapshot, preset  # noqa: E402
from src.planning_layer import candidates as new_mod  # noqa: E402

#: 优化前的冻结副本。放在 tests/ 下随版本库一起走，不依赖 scratchpad。
_FROZEN = Path(__file__).parent / "_frozen_candidates_20260801.py"


def _load_frozen():
    """按文件路径加载冻结副本。

    ⚠️ 它与 `src.planning_layer.candidates` 是**两个模块对象**（坑#18）——
    这里正是有意为之：要的就是两份独立实现的对拍。故只比数组，不做 `isinstance`。
    """
    name = "_frozen_candidates_20260801"
    spec = importlib.util.spec_from_file_location(name, _FROZEN)
    mod = importlib.util.module_from_spec(spec)
    # 🔴 必须先进 `sys.modules` 再 exec —— `@dataclass` 解析注解时会去
    #    `sys.modules[cls.__module__]` 找命名空间，不注册会抛 `NoneType has no __dict__`。
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scenarios():
    """覆盖直线 / 弯道 / 塌陷 / 低速 / 高速五类，且横纵档位全网格都会被走到。"""
    t = np.arange(60, dtype=float) * 0.1
    straight = np.stack([t * 8.0, np.zeros(60)], axis=1)
    ang = t * 8.0 / 25.0
    curve = np.stack([25.0 * np.sin(ang), 25.0 * (1.0 - np.cos(ang))], axis=1)
    collapsed = np.zeros((60, 2))
    slow = np.stack([t * 1.0, np.zeros(60)], axis=1)
    fast = np.stack([t * 18.0, np.zeros(60)], axis=1)
    route = np.stack([np.linspace(-30.0, 250.0, 400), np.zeros(400)], axis=1)
    return [
        ("直线", straight, EgoSnapshot(0.0, 0.0, 0.0, 8.0), None),
        ("弯道", curve, EgoSnapshot(0.0, 0.0, 0.0, 12.0), None),
        ("塌陷+route", collapsed, EgoSnapshot(50.0, 0.0, 0.0, 6.0), route),
        ("低速", slow, EgoSnapshot(0.0, 0.0, 1.2, 0.3), None),
        ("高速", fast, EgoSnapshot(-10.0, 5.0, -0.6, 20.0), None),
    ]


@pytest.mark.skipif(not _FROZEN.exists(), reason="缺少优化前冻结副本")
@pytest.mark.parametrize("preset_name", ["full", "no_dynamics"])
def test_M_candidate_generation_is_bitwise_identical(preset_name):
    frozen = _load_frozen()
    cfg = preset(preset_name)
    for name, ref, ego, route in _scenarios():
        old = frozen.generate(ref, frozen.EgoSnapshot(ego.x, ego.y, ego.heading, ego.speed),
                              cfg, route_xy=route)
        new_all = new_mod.generate(ref, ego, cfg, route_xy=route)

        # 🔴 08-02：候选集新增了 identity 候选（= 上游原轨迹本身，见阶段52 根因分析）。
        #    它**不在**冻结副本里，也**不是**性能优化的产物 ⇒ 对拍时单独摘出来。
        #    这样本判据保护的东西**一点没变**：35 个网格候选必须逐位不变。
        new = [c for c in new_all if not c.is_identity]
        ident = [c for c in new_all if c.is_identity]
        assert len(ident) == (1 if cfg.include_identity else 0), "identity 候选数目不对"
        assert len(old) == len(new) == len(cfg.lon_ratios) * len(cfg.lat_offsets)
        assert len(new_all) == cfg.n_candidates
        for i, (a, b) in enumerate(zip(old, new)):
            for field in ("xy", "speed", "accel", "jerk", "lat_accel"):
                d = float(np.max(np.abs(getattr(a, field) - getattr(b, field))))
                assert d == 0.0, (
                    f"[{preset_name}/{name}] 候选 {i} 的 {field} 被优化改变了 max|Δ|={d:.3e}"
                )
