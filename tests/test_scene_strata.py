"""`scene_strata.py` 的单测（A1 冻结定义的回归门）。

重点锁死两件**已经踩过的坑**：
① 位移必须用**最后一个有效帧**（P0 v1 用 `af[-1]` 把补零读成「开到原点」）；
② GT 分层与 t0 可观测分层**互不混用**（test 侧只许用后者）。
"""
from __future__ import annotations

import glob, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.scene_strata import (ENCOUNTER_M, FULL_FRAMES, MOVING_DISP_M,  # noqa: E402
                                           NEAR_EGO_M, T0_SPEED_MPS, agent_strata,
                                           scene_strata)

VAL = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "val"


def _hist(x, y, vx=0.0, vy=0.0, T=20):
    h = np.zeros((T, 11), dtype=np.float32)
    h[:, 0], h[:, 1], h[:, 3], h[:, 4] = x, y, vx, vy
    return h


def _fut(pts, n_valid=None):
    f = np.zeros((FULL_FRAMES, 3), dtype=np.float32)
    k = len(pts) if n_valid is None else n_valid
    f[:k, :2] = np.asarray(pts)[:k]
    return f


# ── 🔴 核心：补零尾巴不得被当成终点 ──────────────────────────────────────
def test_截断未来必须用最后有效帧():
    """agent 停在 (50, 0) 不动，track 第 10 帧结束 ⇒ 位移必须是 0，不是 50。"""
    h = _hist(50.0, 0.0)
    f = _fut([(50.0, 0.0)] * 10, n_valid=10)          # 后 50 帧补零
    s = agent_strata(h, f)
    assert s["n_valid_frames"] == 10
    assert not s["is_full60"]
    assert abs(s["disp"]) < 1e-5, f"位移应为 0，实得 {s['disp']} —— P0 v1 的 bug 复现了"
    assert not s["is_moving"], "被截断的 agent 不得判为 moving（is_full60 前提）"


def test_full60_真在动():
    # 🔴 起点不得放在 (0,0)：补零判据 |x|+|y|>0 区分不了「真在原点」与「补零」
    h = _hist(5.0, 0.0, vx=6.0)
    f = _fut([(5.0 + 0.6 * i, 0.0) for i in range(FULL_FRAMES)])
    s = agent_strata(h, f)
    assert s["is_full60"] and s["is_moving"]
    assert s["disp"] > MOVING_DISP_M
    assert s["is_near"], "t0 距 ego 5 m，应判 near"


def test_原点歧义是已知局限():
    """补零判据无法区分「真的在 ego 原点」与「补零」—— 锁死这条已知局限，防止有人误改判据。"""
    h = _hist(0.0, 0.0, vx=6.0)
    f = _fut([(0.0, 0.0)] + [(0.6 * i, 0.0) for i in range(1, FULL_FRAMES)])
    s = agent_strata(h, f)
    assert s["n_valid_frames"] == FULL_FRAMES - 1, "首帧恰在原点会被判成补零 —— 这是已知且可接受的局限"


def test_停车不算moving():
    h = _hist(20.0, 3.0)
    f = _fut([(20.0, 3.0)] * FULL_FRAMES)
    s = agent_strata(h, f)
    assert s["is_full60"] and not s["is_moving"] and not s["is_encounter"]


# ── t0 可观测判据：只读观测，不碰未来 ────────────────────────────────────
def test_t0判据不依赖未来():
    h = _hist(10.0, 0.0, vx=3.0)
    f_move = _fut([(10.0 + 0.3 * i, 0.0) for i in range(FULL_FRAMES)])
    f_zero = np.zeros((FULL_FRAMES, 3), dtype=np.float32)
    assert agent_strata(h, f_move)["is_t0_observable"] == agent_strata(h, f_zero)["is_t0_observable"] is True


def test_t0判据的两个阈值都生效():
    assert not agent_strata(_hist(10.0, 0.0, vx=0.1), _fut([(10.0, 0.0)]))["is_t0_observable"], "速度不足"
    assert not agent_strata(_hist(NEAR_EGO_M + 5, 0.0, vx=5.0), _fut([(35.0, 0.0)]))["is_t0_observable"], "距离超限"


def test_遭遇判据用同时刻距离():
    ego = np.stack([np.linspace(0, 60, FULL_FRAMES), np.zeros(FULL_FRAMES)], axis=1)
    h = _hist(0.0, 100.0, vx=6.0)
    # 与 ego 同向同速但横向偏 100 m ⇒ 同时刻始终远，不算遭遇
    f_far = _fut([(60.0 * i / FULL_FRAMES, 100.0) for i in range(FULL_FRAMES)])
    assert not agent_strata(h, f_far, ego)["is_encounter"]
    # 横向偏 1 m ⇒ 同时刻很近，算遭遇
    h2 = _hist(0.0, 1.0, vx=6.0)
    f_near = _fut([(60.0 * i / FULL_FRAMES, 1.0) for i in range(FULL_FRAMES)])
    assert agent_strata(h2, f_near, ego)["is_encounter"]


# ── 真实数据上的一致性 ─────────────────────────────────────────────────
def test_真实场景_计数单调且比例落在冻结区间():
    fs = sorted(glob.glob(os.path.join(str(VAL), "**", "*.pkl"), recursive=True))
    random.Random(42).shuffle(fs)
    rows = [scene_strata(pd.read_pickle(f).iloc[0]) for f in fs[:300]]
    for c in rows:
        # 🔴 near 与 encounter 是 moving 的两个**平行**子集，互不包含（实测存在 near=0 而 encounter=2）
        assert c["n_agents"] >= c["n_full60"] >= c["n_moving"], f"主链必须单调收缩：{c}"
        assert c["n_moving"] >= c["n_moving_near"] and c["n_moving"] >= c["n_encounter"], \
            f"near / encounter 必须各自 <= moving：{c}"
    frac_enc0 = sum(c["n_encounter"] == 0 for c in rows) / len(rows)
    # 全量 val 实测 86.9%，CI [82.4, 91.2]；300 抽样放宽到 ±10 pt
    assert 0.75 <= frac_enc0 <= 0.97, f"无遭遇场景比例 {frac_enc0:.3f} 偏离全量实测 0.869 过多"
    frac_mov2 = sum(c["subset_moving_ge2"] for c in rows) / len(rows)
    assert 0.20 <= frac_mov2 <= 0.55, f"moving>=2 子集比例 {frac_mov2:.3f} 偏离全量实测 0.352 过多"


if __name__ == "__main__":
    import traceback
    fails = 0
    names = [k for k in list(globals()) if k.startswith("test_")]
    for name in sorted(names):
        try:
            globals()[name]()
            print("  PASS  %s" % name)
        except Exception:
            fails += 1
            print("  FAIL  %s" % name)
            traceback.print_exc()
    print("\n%d passed, %d failed" % (len(names) - fails, fails))
    sys.exit(1 if fails else 0)
