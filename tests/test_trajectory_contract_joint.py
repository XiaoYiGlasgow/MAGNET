"""`src/shared/utils/trajectory_contract_joint.py` 的单测（路线 B 第 1 步）。

覆盖三类：① 纯函数的口径与断言；② 掩码损失的边界；③ **真实 pkl 上的端到端不变量**。
"""
from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN / "scripts" / "training"), str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.trajectory_contract_joint import (  # noqa: E402
    FUTURE_FRAMES, JointTargetError, build_joint_targets, ego_increments_to_positions,
    masked_position_l1)

VAL_ROOT = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "val"


def _pkls(n: int = 25):
    fs = sorted(glob.glob(os.path.join(str(VAL_ROOT), "**", "*.pkl"), recursive=True))
    random.Random(42).shuffle(fs)
    return fs[:n]


# ── ① 纯函数 ────────────────────────────────────────────────────────────
def test_ego_cumsum_基本():
    y = np.zeros((FUTURE_FRAMES, 2), dtype=np.float32)
    y[1:, 0] = 0.5                                    # 每帧前进 0.5 m
    p = ego_increments_to_positions(y)
    assert p.shape == (FUTURE_FRAMES, 2)
    assert np.allclose(p[0], 0.0), "首帧必须落在原点（t0 的 ego 位置）"
    assert np.isclose(p[-1, 0], 0.5 * (FUTURE_FRAMES - 1))


def test_ego_接受_1x60x2():
    y = np.zeros((1, FUTURE_FRAMES, 2), dtype=np.float32)
    assert ego_increments_to_positions(y).shape == (FUTURE_FRAMES, 2)


def test_ego_首帧非零必须抛错():
    y = np.zeros((FUTURE_FRAMES, 2), dtype=np.float32)
    y[0, 0] = 1.0
    try:
        ego_increments_to_positions(y)
    except JointTargetError:
        return
    raise AssertionError("首帧增量非零时必须抛 JointTargetError —— 否则两侧同系的结论被静默破坏")


def test_ego_形状不符必须抛错():
    for bad in (np.zeros((30, 2), np.float32), np.zeros((FUTURE_FRAMES, 3), np.float32)):
        try:
            ego_increments_to_positions(bad)
        except JointTargetError:
            continue
        raise AssertionError(f"形状 {bad.shape} 应抛错")


# ── ② 掩码损失 ──────────────────────────────────────────────────────────
def test_masked_loss_全掩码返回零而非nan():
    pred = torch.randn(3, FUTURE_FRAMES, 2, requires_grad=True)
    tgt = torch.randn(3, FUTURE_FRAMES, 2)
    msk = torch.zeros(3, FUTURE_FRAMES, dtype=torch.bool)
    loss = masked_position_l1(pred, tgt, msk)
    assert torch.isfinite(loss) and float(loss) == 0.0
    loss.backward()                                   # 必须可反传且不产生 nan 梯度
    assert torch.isfinite(pred.grad).all()


def test_masked_loss_只算有效帧():
    pred = torch.zeros(1, 4, 2)
    tgt = torch.zeros(1, 4, 2)
    tgt[0, 2:] = 100.0                                # 后两帧差很大
    msk = torch.tensor([[True, True, False, False]])
    assert float(masked_position_l1(pred, tgt, msk)) == 0.0, "被 mask 的帧不得进损失"


# ── ③ 真实 pkl 上的不变量 ───────────────────────────────────────────────
def test_真实样本_口径与形状():
    n_ok = 0
    for f in _pkls():
        row = pd.read_pickle(f).iloc[0]
        out = build_joint_targets(row, "default", strict=False)
        if out is None:
            continue
        n_ok += 1
        assert out["ego_tgt"].shape == (FUTURE_FRAMES, 2)
        assert bool(out["ego_mask"].all())
        na = int(out["agent_tgt"].shape[0])
        assert out["agent_mask"].shape == (na, FUTURE_FRAMES)
        assert int(out["is_agent"].sum()) == na, "agent 节点数必须等于 target 行数"
        # 🔴 ego 首帧必须在原点 —— 两侧同坐标系的支点
        assert torch.allclose(out["ego_tgt"][0], torch.zeros(2), atol=1e-5)
    assert n_ok >= 15, f"真实样本可用数 {n_ok} 过少，可能是数据路径或建图配置不对"


def test_真实样本_掩码确实标出了补零尾巴():
    """P0 v1 的 bug 就是没有这一层 —— 这个测试锁死它不再发生。"""
    seen_truncated = False
    for f in _pkls(40):
        row = pd.read_pickle(f).iloc[0]
        out = build_joint_targets(row, "default", strict=False)
        if out is None or out["agent_tgt"].shape[0] == 0:
            continue
        m = out["agent_mask"]
        # 有效帧必须是**前缀连续**（P0-c 实测 100% 如此）
        for i in range(m.shape[0]):
            k = int(m[i].sum())
            assert bool(m[i, :k].all()), "有效帧不是前缀连续 —— 与 P0-c 实测冲突，须重查"
            if 0 < k < FUTURE_FRAMES:
                seen_truncated = True
                # 被 mask 的帧在 target 里就是补零
                assert torch.allclose(out["agent_tgt"][i, k:], torch.zeros(FUTURE_FRAMES - k, 2))
    assert seen_truncated, "40 个样本里一个截断 agent 都没有 —— 与 P0-c 的 32.26% 冲突"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("  PASS  %s" % name)
            except Exception:
                fails += 1
                print("  FAIL  %s" % name)
                traceback.print_exc()
    print("\n%d passed, %d failed" % (sum(1 for k in globals() if k.startswith("test_")) - fails, fails))
    sys.exit(1 if fails else 0)
