"""A2 —— `resolve_agent_slots` 在**新档 fullret 真数据**上的回归门（2026-08-22）。

## 为什么必须有这个门

`resolve_agent_slots` 一旦错位，**100% 的监督信号都是错的，而且不会报错** ——
它只会安静地给出一批系统性错误的标签，然后下游判「没信号」，
于是得出「交互没用」这个漂亮但错误的结论并放弃整条路线。
（`magnet_agent_aux.py` docstring 记录了阶段75 已经踩过一次。）

## 为什么现在必须重测

判据 `JOIN_MAX_DIST=0.05 且 JOIN_MIN_MARGIN=0.5` 是在**旧档**上定的 ——
旧档每场景中位 **2** 辆入图 agent，次近距离实测 min **2.03 m**，余量 4×。
新档 fullret 全量保留静止车，每场景中位 **15** 辆，次近距离本来可能塌到阈值以下。

## 2026-08-22 实测基线（train 800 场景 / 13 773 个 agent 节点）

===================  ==========  ==========
量                   实测        判据
===================  ==========  ==========
最近距离 d1 max      0.000000    < 0.05
次近距离 d2 中位     5.874       > 0.5
次近距离 d2 p1       2.343       —
**join 失败率**      **0.26%**   闸门 1%
===================  ==========  ==========

失败的那 0.26% 是**位置完全重合**的真歧义（d2 = 0.000000），
现有实现把冲突的全部作废（而不是猜哪个对）是正确处置。
"""
from __future__ import annotations

import glob, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd

_GNN = Path(__file__).resolve().parents[1]
for _p in (str(_GNN / "scripts" / "training"), str(_GNN), str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_agent_aux import JOIN_MAX_DIST, JOIN_MIN_MARGIN, resolve_agent_slots  # noqa: E402
from magnet_agent_aux_wiring import build_agent_aux_fields  # noqa: E402

TRAIN = _GNN / "data" / "processed_features_fullret_L1_logsplit" / "train"

#: 闸门：join 失败率上限。2026-08-22 实测 0.26%，留 ~4× 余量。
MAX_FAIL_RATE = 0.01
#: 冻结基线（用于检出「判据被人悄悄改宽/改窄」）。
BASELINE_D1_MAX = 0.05
BASELINE_D2_MEDIAN_MIN = 3.0


def _pkls(n: int):
    fs = sorted(glob.glob(os.path.join(str(TRAIN), "**", "*.pkl"), recursive=True))
    random.Random(42).shuffle(fs)
    return fs[:n]


def test_判据常量未被改动():
    assert JOIN_MAX_DIST == 0.05, "判据被改了 —— 改它会同时改变全部历史标签，须重新预注册"
    assert JOIN_MIN_MARGIN == 0.5


def test_槽位距离分布_新档():
    """直接量 `AGENT_HISTORIES[:, -1, :2]` 内部的最近/次近距离分布。"""
    d1s, d2s = [], []
    for f in _pkls(200):
        ah = np.asarray(pd.read_pickle(f).iloc[0]["AGENT_HISTORIES"])
        live = np.where(np.abs(ah).sum(axis=(1, 2)) > 0)[0]
        if len(live) < 2:
            continue
        end = ah[:, -1, :2].astype(np.float64)
        D = np.linalg.norm(end[live][:, None, :] - end[None, :, :], axis=-1)
        o = np.argsort(D, axis=1)
        idx = np.arange(len(live))
        d1s.append(D[idx, o[:, 0]])
        d2s.append(D[idx, o[:, 1]])
    d1, d2 = np.concatenate(d1s), np.concatenate(d2s)
    assert len(d1) > 2000, f"样本量 {len(d1)} 过少"
    assert d1.max() < BASELINE_D1_MAX, f"最近距离 max {d1.max():.6f} 已触判据上限 —— 位置 join 前提失效"
    assert np.median(d2) > BASELINE_D2_MEDIAN_MIN, f"次近距离中位 {np.median(d2):.3f} 塌到基线以下"
    fail = float((d2 <= JOIN_MIN_MARGIN).mean())
    assert fail <= MAX_FAIL_RATE, f"次近余量不足的比例 {fail:.4%} > 闸门 {MAX_FAIL_RATE:.0%}"


def test_端到端_join失败率在闸门内():
    """走真实路径（建图 → 节点 → 位置 join），统计 `slot < 0` 的比例。"""
    n_node = n_fail = n_scene = 0
    for f in _pkls(150):
        row = pd.read_pickle(f).iloc[0]
        out = build_agent_aux_fields(row, "default", strict=False, return_slots=True)
        if out is None:
            continue
        n_scene += 1
        slots = out[3]
        n_node += int(slots.size)
        n_fail += int((slots < 0).sum())
    assert n_scene >= 100, f"可用场景 {n_scene} 过少 —— 建图配置或数据路径可能不对"
    assert n_node > 1000, f"agent 节点数 {n_node} 过少"
    rate = n_fail / n_node
    assert rate <= MAX_FAIL_RATE, f"端到端 join 失败率 {rate:.4%} > 闸门 {MAX_FAIL_RATE:.0%}（{n_fail}/{n_node}）"


def test_槽位互斥_无重复映射():
    for f in _pkls(60):
        row = pd.read_pickle(f).iloc[0]
        out = build_agent_aux_fields(row, "default", strict=False, return_slots=True)
        if out is None:
            continue
        used = out[3][out[3] >= 0]
        assert len(used) == len(set(used.tolist())), "同一槽位被多个节点命中 —— 冲突作废逻辑失效"


def test_人造歧义必须被作废而非猜():
    """两个槽位位置完全重合 ⇒ 次近 = 0 ⇒ 必须记 -1，不得挑一个。"""
    end = np.array([[10.0, 0.0], [10.0, 0.0], [50.0, 0.0]], dtype=np.float64)
    got = resolve_agent_slots(np.array([[10.0, 0.0]]), end, strict=False)
    assert got[0] == -1, f"重合槽位应作废，实得 {got[0]}"


if __name__ == "__main__":
    import traceback
    names = [k for k in list(globals()) if k.startswith("test_")]
    fails = 0
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
