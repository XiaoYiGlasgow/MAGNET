"""闸①（Q49）：**把被静止过滤删掉的 73% 近邻车辆放回图里** —— 只重建 subgraph，不回 nuPlan DB。

背景（Q32 全量实测，08-01）
---------------------------
`utils/data_utils.py::encoding_vectornet_features:850,867` 有一道
``avg_speed < 0.01 m/s`` 的静止过滤，**发生在建图之前**。val 全量 3000 场景实测：
真实近邻车辆均值 **10.17** → 入图 agent 均值 **2.75**（中位数 2），
**72.91% 的近邻车辆根本没进过图**；空图率 **29.53%**。

⇒ C1 的「删掉全部 agent 轨迹只移动 0.72%」这个读数，分母是**削掉 73% 之后**的图。
Q49 就是把这道闸打开（阈值 0.01 → 0）重训一格，**两种结果都有价值**：

* 动了 ⇒ 我方自己量化出的、文献未记录的缺陷被修好，链条闭合；
* 不动 ⇒ 反而**强化 C1** —— 排除掉「模型没用 agent 是因为压根没给它」这个 confound。

为什么可以不回 nuPlan DB
------------------------
过滤发生在建图前，`features_*.pkl` 的 `POLYLINE_FEATURES` 里确实已经没有那些车；
但同一个 pkl 里还存着 **`AGENT_HISTORIES`（过滤前的 20 槽原始数组，shape `[20, T, 11]`）**
—— Q32 审计就是拿它做的，且与图内 agent 子图数 **3000/3000 逐场景零误差**。
⇒ 重建 trajectory subgraph 所需的一切都在 pkl 内：

======================  ==================================================
需要的量                 从哪里恢复
======================  ==================================================
`agents_past`           `AGENT_HISTORIES`（原样，float32）
`ego_subgraph`          `POLYLINE_FEATURES` 的 ego 行（**原样搬运，不重算**）
`ego_timestamps`        ego 行的 col10/col11（写入时就是它，可逐位恢复）
`timestamp_list`        `np.linspace(0, 1, N-1, endpoint=False)`，N 由 ego 行数定
`lane_subgraph`         `POLYLINE_FEATURES` 的 lane 行（原样搬运，**只改 col9 = 子图 ID**）
======================  ==================================================

🔴 **逐字节等价门（本项目 E0 用过的模式，必须先过）**
------------------------------------------------------
用**现阈值 0.01** 重建，断言与已落盘 pkl **逐字节完全一致**；过了才敢把阈值翻成 0。
不过门就说明这条便宜路不成立，老老实实重跑预处理。

    python rebuild_agent_subgraph.py --src <val> --speed_threshold 0.01 --verify_only

⚠️ **dtype 陷阱（等价门的命门）**：原实现里
``speed_a = np.sqrt(vx_a**2 + vy_a**2)`` 是在 **float32** 上算的（`AGENT_HISTORIES` 是
float32），之后才在 `np.hstack` 里被 timestamp 的 float64 顺带提升。
若这里先把 `agent_hist` 转成 float64 再开方，**低位会不一样**，等价门必然挂 ——
且 `avg_speed` 恰好落在阈值附近的 agent 还会被判到另一边。故本脚本全程保持 float32 输入。

⚠️ **不改的东西**：`GROUND_TRUTH` / `EGO_LAST_POS` / `EGO_HISTORY` /
`AGENT_HISTORIES` / `AGENT_FUTURES` 原样透传 —— 闸① 只改「哪些车进图」，不动标签与 ego 输入。

⚠️ **槽位与半径不在本脚本范围内**：最近 **20** 辆 / 半径 **50 m** 由
`agent_past_process:181-188` 与 `config.py` 决定，改它们**必须回 nuPlan DB 重跑预处理**，
且半径与 `MAP_RADIUS` 绑定会连带改地图 ⇒ **另开一格，别混**（主控表 §0.0-b）。
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_GNN_ROOT = Path(__file__).resolve().parents[2]
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from utils.gnn_module import subgraph_pooling  # noqa: E402  生效副本（data_utils.py:10 用的就是它）

logger = logging.getLogger(__name__)

#: 现行阈值（`utils/data_utils.py:850`）。等价门用它，正式重建用 0.0。
CURRENT_SPEED_THRESHOLD = 0.01

#: `AGENT_HISTORIES` 的列语义（`agent_past_process` 打包顺序），与
#: `scripts/analysis/audit_agent_filter_funnel.py` 保持同一份定义。
COL_VX, COL_VY = 3, 4

#: 输出列顺序必须与 `encoding_vectornet_features` 的产出完全一致
COLUMNS = ["POLYLINE_FEATURES", "GROUND_TRUTH", "TRAJ_ID_TO_INDICES", "LANE_ID_TO_INDICES",
           "TARJ_SIZE", "LANE_SIZE", "SUBGRAPH_ID_TO_NODE_INDICES", "SUBGRAPH_ID_TO_TYPE",
           "SUBGRAPH_FEATURES", "EGO_LAST_POS", "EGO_HISTORY", "AGENT_HISTORIES",
           "AGENT_FUTURES"]


class RebuildContractError(RuntimeError):
    """pkl 不符合重建所需的契约 —— 拒绝执行，不静默产出半对的特征。"""


def _recover_ego_timestamps(ego_rows: np.ndarray) -> np.ndarray:
    """从 ego 子图的 col10/col11 逐位恢复 `ego_timestamps`（长度 N = 行数 + 1）。

    写入时 `col10 = ego_timestamps[:-1]`、`col11 = ego_timestamps[1:]`
    （`data_utils.py:836-837`）⇒ 相邻两行必须首尾相接，这里显式校验，
    对不上说明该 pkl 不是本管线产出的，直接抛错。
    """
    if ego_rows.shape[0] < 2:
        raise RebuildContractError(f"ego 子图只有 {ego_rows.shape[0]} 行，无法恢复时间戳")
    if not np.array_equal(ego_rows[1:, 10], ego_rows[:-1, 11]):
        raise RebuildContractError("ego 子图 col10/col11 首尾不相接，时间戳恢复不可靠")
    return np.concatenate([ego_rows[:, 10], ego_rows[-1:, 11]])


def _agent_subgraph(agent_hist: np.ndarray, timestamp_list: np.ndarray,
                    ego_timestamps: np.ndarray, subgraph_id: int) -> np.ndarray:
    """逐字复刻 `data_utils.py:870-881` 的 agent 子图打包（**保持 float32 输入**）。"""
    m = agent_hist.shape[0]
    agent_ts = timestamp_list[: m - 1]
    agent_seg_start = ego_timestamps[:m][:-1]
    agent_seg_end = ego_timestamps[:m][1:]
    vx_a, vy_a = agent_hist[:, COL_VX], agent_hist[:, COL_VY]
    speed_a = np.sqrt(vx_a ** 2 + vy_a ** 2)
    return np.hstack((
        agent_hist[:-1, :2],
        agent_hist[1:, :2],
        agent_hist[:-1, 2].reshape(-1, 1),
        agent_hist[1:, 2].reshape(-1, 1),
        agent_ts.reshape(-1, 1),
        speed_a[:-1].reshape(-1, 1),
        speed_a[1:].reshape(-1, 1),
        np.ones((m - 1, 1)) * subgraph_id,
        agent_seg_start.reshape(-1, 1),
        agent_seg_end.reshape(-1, 1),
    ))


def _avg_speed(agent_hist: np.ndarray) -> float:
    """逐字复刻 `data_utils.py:861-864`。**必须在 float32 上算**（见模块 docstring 的 dtype 陷阱）。"""
    speed_a = np.sqrt(agent_hist[:, COL_VX] ** 2 + agent_hist[:, COL_VY] ** 2)
    return np.mean((speed_a[:-1] + speed_a[1:]) / 2)


def rebuild_row(row, speed_threshold: float) -> Dict[str, Any]:
    """按给定阈值重建单个场景的 subgraph 相关字段，返回可直接组 DataFrame 的字典。

    :param row: `pd.Series`，来自 `features_*.pkl` 的第 0 行。
    :param speed_threshold: 静止过滤阈值 [m/s]。`0.01` = 现行值（等价门），`0.0` = 闸① 打开。
    """
    poly = np.asarray(row["POLYLINE_FEATURES"])
    traj_idx: Dict[int, Tuple[int, int]] = row["TRAJ_ID_TO_INDICES"]
    lane_idx: Dict[int, Tuple[int, int]] = row["LANE_ID_TO_INDICES"]
    hist = np.asarray(row["AGENT_HISTORIES"])
    traj_size, lane_size = int(row["TARJ_SIZE"]), int(row["LANE_SIZE"])

    if poly.shape[0] != traj_size + lane_size:
        raise RebuildContractError(
            f"polyline 行数 {poly.shape[0]} != TARJ_SIZE+LANE_SIZE {traj_size}+{lane_size}")
    if 0 not in traj_idx or traj_idx[0][0] != 0:
        raise RebuildContractError(f"ego 子图必须是 ID 0 且从第 0 行开始，实得 {traj_idx.get(0)}")
    if hist.dtype != np.float32:
        raise RebuildContractError(f"AGENT_HISTORIES 应为 float32，实得 {hist.dtype}")

    # ── ego：原样搬运（不重算，避免任何浮点重现问题）──
    ego_end = int(traj_idx[0][1])
    ego_rows = poly[: ego_end + 1]
    n = ego_rows.shape[0] + 1                       # = len(ego_past)
    ego_timestamps = _recover_ego_timestamps(ego_rows)
    timestamp_list = np.linspace(0, 1, num=(n - 1), endpoint=False)

    # ── agents：按新阈值重建 ──
    trajectory_subgraph = ego_rows
    new_traj_idx: Dict[int, Tuple[int, int]] = {0: (0, n - 2)}
    new_types: Dict[int, str] = {0: "ego"}
    subgraph_id = 1
    n_dropped = 0
    for agent_hist in hist:
        m = agent_hist.shape[0]
        if m < 2 or np.all(agent_hist == 0):        # 闸2：全零占位
            continue
        if _avg_speed(agent_hist) < speed_threshold:  # 闸3：静止过滤（本脚本要打开的就是它）
            n_dropped += 1
            continue
        block = _agent_subgraph(agent_hist, timestamp_list, ego_timestamps, subgraph_id)
        start_id = trajectory_subgraph.shape[0]
        trajectory_subgraph = np.vstack((trajectory_subgraph, block))
        new_traj_idx[subgraph_id] = (start_id, trajectory_subgraph.shape[0] - 1)
        new_types[subgraph_id] = "agent"
        subgraph_id += 1

    # ── lanes：原样搬运，只改 col9（子图 ID）；LANE_ID_TO_INDICES 的值是 map 内偏移，不变 ──
    map_subgraph = poly[traj_size:].copy()
    new_lane_idx: Dict[int, Tuple[int, int]] = {}
    for _, (start, end) in lane_idx.items():
        map_subgraph[int(start): int(end) + 1, 9] = subgraph_id
        new_lane_idx[subgraph_id] = (int(start), int(end))
        new_types[subgraph_id] = "lane"
        subgraph_id += 1

    polyline = np.vstack((trajectory_subgraph, map_subgraph))

    # ── 逐字复刻 data_utils.py:941-950（注意值是 list 不是 tuple）──
    node_indices: Dict[int, List[int]] = {}
    for cur, r in enumerate(polyline):
        sid = int(r[9])
        if sid not in node_indices:
            node_indices[sid] = [cur, cur]
        else:
            node_indices[sid][1] = cur

    return {
        "POLYLINE_FEATURES": polyline,
        "GROUND_TRUTH": row["GROUND_TRUTH"],
        "TRAJ_ID_TO_INDICES": new_traj_idx,
        "LANE_ID_TO_INDICES": new_lane_idx,
        "TARJ_SIZE": trajectory_subgraph.shape[0],
        "LANE_SIZE": map_subgraph.shape[0],
        "SUBGRAPH_ID_TO_NODE_INDICES": node_indices,
        "SUBGRAPH_ID_TO_TYPE": new_types,
        "SUBGRAPH_FEATURES": subgraph_pooling(polyline, node_indices),
        "EGO_LAST_POS": row["EGO_LAST_POS"],
        "EGO_HISTORY": row["EGO_HISTORY"],
        "AGENT_HISTORIES": row["AGENT_HISTORIES"],
        "AGENT_FUTURES": row["AGENT_FUTURES"],
        "_n_agents": len(new_traj_idx) - 1,
        "_n_dropped_static": n_dropped,
    }


# ------------------------------------------------------------------ 逐字节等价门
def _bytes_equal(a: Any, b: Any) -> bool:
    """数组按 shape+dtype+原始字节比较；标量/容器按值比较。**不做任何容差**。"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return bool(a == b)


def diff_row(old, new: Dict[str, Any]) -> List[str]:
    """返回不一致的字段名列表；空列表 = 逐字节一致。**类型也要一致**（tuple ≠ list）。"""
    bad: List[str] = []
    for col in COLUMNS:
        o, v = old[col], new[col]
        if isinstance(o, dict):
            if set(o) != set(v):
                bad.append(f"{col}(键集合 {len(o)}->{len(v)})")
                continue
            for k in o:
                if type(o[k]) is not type(v[k]) or not _bytes_equal(o[k], v[k]):
                    bad.append(f"{col}[{k}]")
                    break
        elif not _bytes_equal(o, v):
            bad.append(col)
    return bad


# ------------------------------------------------------------------ 主流程
def _process(src: Path, dst: Optional[Path], threshold: float,
             verify: bool, limit: int, log_every: int) -> Dict[str, Any]:
    files = sorted(src.glob("features_*.pkl"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"{src} 下没有 features_*.pkl")
    logger.info("场景 %d 个 <- %s（阈值 %.4g，%s）", len(files), src, threshold,
                "等价门" if verify else f"重建 -> {dst}")

    if dst is not None:
        dst.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    failures: List[Tuple[str, List[str]]] = []
    agents_before: List[int] = []
    agents_after: List[int] = []
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            df = pickle.load(fh)
        row = df.iloc[0]
        new = rebuild_row(row, threshold)
        agents_before.append(len(row["TRAJ_ID_TO_INDICES"]) - 1)
        agents_after.append(new["_n_agents"])

        if verify:
            bad = diff_row(row, new)
            if bad:
                failures.append((f.stem, bad))
            else:
                n_ok += 1
        else:
            out = pd.DataFrame([[new[c] for c in COLUMNS]], columns=COLUMNS)
            with open(dst / f.name, "wb") as fh:
                pickle.dump(out, fh)
            n_ok += 1

        if log_every and (i + 1) % log_every == 0:
            logger.info("  已处理 %d/%d（失败 %d）", i + 1, len(files), len(failures))

    before, after = np.array(agents_before), np.array(agents_after)
    return {
        "src": str(src), "dst": str(dst) if dst else None,
        "speed_threshold": threshold, "mode": "verify" if verify else "rebuild",
        "n_scenes": len(files), "n_ok": n_ok, "n_mismatch": len(failures),
        "mismatch_samples": [{"token": t, "fields": b} for t, b in failures[:10]],
        "agents_in_graph": {
            "before_mean": round(float(before.mean()), 3),
            "before_median": float(np.median(before)),
            "after_mean": round(float(after.mean()), 3),
            "after_median": float(np.median(after)),
            "empty_graph_pct_before": round(100.0 * float((before == 0).mean()), 2),
            "empty_graph_pct_after": round(100.0 * float((after == 0).mean()), 2),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Q49 闸①：按新静止阈值只重建 subgraph（不回 nuPlan DB）")
    p.add_argument("--src", required=True, help="含 features_*.pkl 的源目录（绝对路径）")
    p.add_argument("--dst", default=None, help="输出目录（绝对路径）；--verify_only 时不需要")
    p.add_argument("--speed_threshold", type=float, default=CURRENT_SPEED_THRESHOLD,
                   help=f"静止过滤阈值 [m/s]；{CURRENT_SPEED_THRESHOLD}=现行值（等价门），0=闸①打开")
    p.add_argument("--verify_only", action="store_true",
                   help="只跑逐字节等价门，不写任何文件")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个，0=全量")
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--out_json", default=None, help="汇总 JSON 落盘路径（绝对路径）")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not a.verify_only and not a.dst:
        raise SystemExit("重建模式必须给 --dst（或加 --verify_only 只跑等价门）")
    if a.verify_only and a.speed_threshold != CURRENT_SPEED_THRESHOLD:
        raise SystemExit(
            f"等价门必须用现行阈值 {CURRENT_SPEED_THRESHOLD} 才有意义，"
            f"收到 {a.speed_threshold}")

    summary = _process(Path(a.src), Path(a.dst) if a.dst else None,
                       a.speed_threshold, a.verify_only, a.limit, a.log_every)

    import json
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(text, encoding="utf-8")
        logger.info("写出 %s", a.out_json)
    print(text)

    if summary["n_mismatch"]:
        raise SystemExit(
            f"❌ 等价门未通过：{summary['n_mismatch']}/{summary['n_scenes']} 个场景不一致 "
            "⇒ 这条便宜路不成立，必须回 nuPlan DB 重跑预处理")


if __name__ == "__main__":
    main()
