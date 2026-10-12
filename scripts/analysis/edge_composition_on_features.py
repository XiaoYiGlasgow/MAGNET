#!/usr/bin/env python
"""在**已落盘特征**上精确核算边构成 —— L-2 可行性判断的直接依据。

为什么不能只看探针
------------------
`probe_entity_capacity.py` 从共享清单分层抽样（覆盖 train+val 全部 log），
而 **L-2 的干预审计实际跑在 `processed_features_scaling_L1_logsplit/val` 上**
（`r1_audit_worker_20260805.sh:35` / `q86_worker_20260805.sh:30`）。
两者车密度不同（08-06 实测：清单抽样闸1后 13.37 辆 vs L1 val 10.17 辆），
⇒ **可行性必须按真正被干预的那份数据判**，否则会高估还剩几条边可切。

能还原什么、不能还原什么
------------------------
落盘 `AGENT_HISTORIES` 是 **20 槽、按到 ego 距离升序、静止过滤之前**的原始数组
（Q49 已用逐字节等价门证过）⇒

* ✅ **`current` 能精确还原** —— 这是 L-2 可行性的直接依据；
* 🔴🔴 **`swap_order` 还原不了，且失败方式很隐蔽**：落盘的 20 槽**已经**是「取最近 20」之后的结果，
  第 21 辆及以后的车没落盘 ⇒ 在落盘数据上「先删静止再取 20」与「先取 20 再删静止」
  **恒等**（08-06 冒烟实测两列逐位相同）。
  那不是「调换顺序零收益」，而是**这份数据测不出收益** —— 差别只存在于被截断掉的那部分车里。
  ⇒ 本脚本**拒绝输出 `swap_order`**，避免读者把退化当读数（第九十条同类：判据必须能穿透）。
  调换顺序的收益只能由 `probe_entity_capacity.py`（`probe_cap=200`，不截断）估计。
* 🔴 `planB_*` 同理还原不了。但探针已证明方案B 对「运动×运动」边**零增益**
  （运动车 max 19 < 20，恒填得下 20 槽），所以缺这两档不影响 L-2 的结论。

口径逐字复用 `audit_agent_filter_funnel.py`（Q32）：同一个 `_avg_speed`、同一个空槽判据。

用法
----
    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/edge_composition_on_features.py \
        --val_root data/processed_features_scaling_L1_logsplit/val \
        --out experiments/interaction_audit/edge_composition_L1val_20260806.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))

from scripts.analysis.graph_edge_accounting import (  # noqa: E402
    RetentionPolicy, apply_policy, describe, format_policy_table, summarize_counts,
)

#: 与 `utils/data_utils.py:850` 及 Q32 脚本保持一致；改动必须三处同步
SPEED_THRESHOLD_MPS = 0.01
COL_VX, COL_VY = 3, 4

#: 🔴 落盘数据只能还原 `current` 一档 —— 其余策略在这份数据上会退化成它（见模块 docstring）
POLICIES = [
    RetentionPolicy("current", agent_cap=20, lane_cap=40,
                    keep_static=False, filter_static_before_cap=False),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _avg_speed(agent_hist: np.ndarray) -> float:
    """逐字复刻 `encoding_vectornet_features:862-864` 的平均速度（逐段中点平均）。"""
    speed = np.sqrt(agent_hist[:, COL_VX] ** 2 + agent_hist[:, COL_VY] ** 2)
    if speed.shape[0] < 2:
        return 0.0
    return float(np.mean((speed[:-1] + speed[1:]) / 2))


def read_scene(pkl_path: Path) -> Optional[Tuple[List[bool], int, int]]:
    """从一个落盘场景还原 `(slot_moving, n_lane, n_graph_agents)`。

    Returns:
        `slot_moving` 按到 ego 距离升序（落盘顺序即该顺序，见 `agent_past_process`）；
        `n_lane` 为图中 lane 子图数；`n_graph_agents` 为图中 agent 子图数（用于交叉校验）。
        文件不可用时返回 None —— **不静默计入分母**。
    """
    try:
        df = pickle.load(open(pkl_path, "rb"))
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        logger.warning("读不了 %s: %s", pkl_path.name, exc)
        return None

    row = df.iloc[0]
    hist = np.asarray(row["AGENT_HISTORIES"])          # [20, T, 11]
    slot_moving: List[bool] = []
    for i in range(hist.shape[0]):
        a = hist[i]
        if a.shape[0] < 2 or not np.any(a):            # 闸2：全零占位 ⇒ 真实对象不足 20
            continue
        slot_moving.append(_avg_speed(a) >= SPEED_THRESHOLD_MPS)

    type_map = row["SUBGRAPH_ID_TO_TYPE"]
    n_lane = sum(1 for v in type_map.values() if v == "lane")
    n_graph_agents = sum(1 for v in type_map.values() if v == "agent")
    return slot_moving, n_lane, n_graph_agents


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="落盘特征上的精确边构成核算")
    ap.add_argument("--val_root", required=True, help="含 features_*.pkl 的目录（绝对或相对 GNN_ROOT）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个，0=全量")
    args = ap.parse_args(argv)

    root = Path(args.val_root)
    if not root.is_absolute():
        root = GNN_ROOT / root
    pkls = sorted(root.glob("features_*.pkl"))
    if args.limit:
        pkls = pkls[: args.limit]
    if not pkls:
        logger.error("🔴 %s 下没有 features_*.pkl", root)
        return 1
    logger.info("读 %d 个落盘场景：%s", len(pkls), root)

    per_scene: List[Dict[str, Any]] = []
    n_bad = n_mismatch = 0
    for p in pkls:
        rec = read_scene(p)
        if rec is None:
            n_bad += 1
            continue
        slot_moving, n_lane, n_graph_agents = rec
        # 🔑 交叉校验：图里实际的 agent 子图数必须等于「现状策略」留下的数量。
        #    不等 ⇒ 我对落盘语义的理解错了，读数不可信（第八十四条：闸门也会有 bug）。
        if sum(slot_moving) != n_graph_agents:
            n_mismatch += 1
        per_scene.append({"token": p.stem.replace("features_", ""),
                          "slot_moving": slot_moving, "n_lane": n_lane})

    if not per_scene:
        logger.error("🔴 一个场景都没读到 ⇒ 不出读数")
        return 1
    if n_mismatch:
        logger.error("🔴 %d/%d 个场景的 agent 子图数与现状策略不符 ⇒ 落盘语义理解有误，读数不可信",
                     n_mismatch, len(per_scene))
        return 1

    summaries: Dict[str, Any] = {}
    for policy in POLICIES:
        # 落盘的 lane 数**已经**被 `max_elements` 截断过 ⇒ 这里把 lane_cap 放到无穷大，
        # 避免二次截断（否则 40 会被再截一次，虽然结果相同但语义是错的）
        eff = RetentionPolicy(policy.name, policy.agent_cap, 10 ** 9,
                              policy.keep_static, policy.filter_static_before_cap)
        counts = [apply_policy(r["slot_moving"], r["n_lane"], eff) for r in per_scene]
        summaries[policy.name] = summarize_counts(counts)

    # 🔑 顶格比例 = 「调换顺序**可能**产生差别」的场景上界。
    #    落盘只有 20 槽 ⇒ 非空槽数恰好 20 的场景才被截断过；其余场景两策略必然相同。
    n_slots_full = sum(1 for r in per_scene if len(r["slot_moving"]) >= 20)
    res: Dict[str, Any] = {
        "provenance": {
            "val_root": str(root), "n_scenes": len(per_scene), "n_unreadable": n_bad,
            "speed_threshold_mps": SPEED_THRESHOLD_MPS,
            "note": ("只出 current 一档：落盘 20 槽已是「取最近 20」之后的结果，"
                     "swap_order/planB_* 在这份数据上会退化成 current，不是零收益而是测不出"),
        },
        "truncation": {
            "n_scenes_hitting_20_slots": n_slots_full,
            "pct_hitting_20_slots": round(100.0 * n_slots_full / len(per_scene), 2),
            "meaning": "调换顺序只可能在这部分场景里改变入图集合；其余场景两策略必然逐位相同",
        },
        "lane_in_graph": describe([r["n_lane"] for r in per_scene]),
        "edge_accounting": summaries,
        "per_scene": per_scene,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = GNN_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"落盘特征上的精确边构成（{len(per_scene)} 场景，{root.name}）")
    print("=" * 78)
    print(format_policy_table(summaries))
    cm = summaries["current"]["edges_moving_moving"]
    print(f"\n🎯 L-2 可行性（在真正被干预的这份数据上）：")
    print(f"   运动×运动边  中位 {cm['median']:.0f}｜均值 {cm['mean']:.2f}｜p90 {cm['p90']:.0f}｜max {cm['max']:.0f}")
    print(f"   一条可切的边都没有的场景：{summaries['current']['zero_moving_edge_pct']}%")
    print(f"   他车↔他车边全无的场景：{summaries['current']['zero_agent_edge_pct']}%")
    print(f"\n⚠️ 20 槽顶格场景 {res['truncation']['pct_hitting_20_slots']}% "
          f"—— 调换过滤顺序的收益只可能落在这部分，且**本数据测不出**（见脚本 docstring）")
    print(f"\n已写 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
