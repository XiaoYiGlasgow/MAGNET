"""场景与 agent 的**分层定义**（A1，2026-08-22 冻结）—— **唯一可执行定义源**。

为什么必须冻结成代码而不是散文
--------------------------------
新档 fullret 里 **~80% 的 agent 是停着的车**（路边停车，实测见下）。
在停车上做边级干预，响应本来就恒为 0。⇒
- 混进聚合指标 ⇒ 信号被稀释（假阴性）；
- **看到读数之后**再挑子集 ⇒ 按结果筛样本（纪律 56，闸⑥ 已栽过一次）。

所以分层必须在**任何 G1/G2 读数产生之前**冻结，且必须是**可执行的**——散文定义会漂移。
配套判据文件：`预注册判据/分层与子集定义_20260822_冻结.md`。

🔴 三条使用纪律
---------------
1. **GT 分层（`is_moving` / `is_encounter`）只能用于诊断与训练侧筛选**，
   **不得用于 test 分层** —— 那是按结果筛样本。
2. **test 分层一律用 `is_t0_observable`**（只读 t0 观测，不碰未来）。
3. 🔴 **`is_near` 与 `is_encounter` 是 `is_moving` 的两个平行子集，互不包含。**
   一辆车可以 t0 时距 ego 60 m（不 near）却在 6 s 内开过来与 ego 相遇（encounter）——
   实测存在 `n_moving_near=0` 而 `n_encounter=2` 的场景。
   ⇒ 计数的单调链只有 `n_agents >= n_full60 >= n_moving >= max(n_moving_near, n_encounter)`。
4. ⚠️ **已知局限**：补零判据是 `|x|+|y| > 1e-9`，**区分不了「真的位于 ego 原点」与「补零」**。
   真实数据里 agent 不可能与 ego 同点，故无害；但合成用例不得把 agent 放在 (0, 0)。
   该判据与 `magnet_agent_aux_wiring` 逐字同源，**不得单方面改**（会破坏与既有标签的可比性）。
5. 位移一律用**最后一个有效帧**算。`AGENT_FUTURES` 在 track 结束后逐帧补零
   （实测 32.26% 的 agent 未来被截断，有效帧前缀连续）；
   用 `af[-1]` 会把「补零」读成「开到了原点」，位移退化为「该车到 ego 的距离」
   —— P0 v1 正是栽在这里（伪造出 22.41% 的「起步事件」，真值 ≈1%）。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# ── 冻结阈值（改任何一个都必须重新预注册并注明生效日期）────────────────────
#: agent「在动」判据：6 s 位移（米）。
MOVING_DISP_M: float = 10.0
#: agent「近距」判据：t0 时刻到 ego 的距离（米）。
NEAR_EGO_M: float = 30.0
#: 「与 ego 遭遇」判据：agent 未来与 ego 未来**同时刻**最近距离（米）。
ENCOUNTER_M: float = 5.0
#: t0 可观测「运动」判据：速度列模长（m/s）。🔴 不读未来。
T0_SPEED_MPS: float = 0.5
#: 未来帧数；少于此值即视为 track 被截断。
FULL_FRAMES: int = 60
#: `AGENT_HISTORIES` 的速度列下标（[x, y, heading, vx, vy, yaw_rate, 长, 宽, one-hot×3]）。
VX_COL, VY_COL = 3, 4


def _un(x):
    try:
        return x.iloc[0]
    except Exception:
        return x


def agent_strata(agent_hist: np.ndarray, agent_fut: np.ndarray,
                 ego_fut: Optional[np.ndarray] = None) -> Dict:
    """单个 agent 的分层标签。

    :param agent_hist: (T_h, >=5) 历史，列序见 `VX_COL` 注释；绝对 ego-centric 位置。
    :param agent_fut: (60, >=2) 未来绝对 ego-centric 位置；track 结束后**逐帧补零**。
    :param ego_fut: (60, 2) ego 未来位置（位置域）；给了才算 `is_encounter`。
    :returns: dict，键见下；`n_valid == 0` 时位移相关项为 `None`。
    """
    xy = np.asarray(agent_fut)[:, :2]
    valid = np.abs(xy).sum(axis=-1) > 1e-9
    n_valid = int(valid.sum())
    p0 = np.asarray(agent_hist)[-1, :2].astype(np.float64)
    t0_speed = float(np.hypot(agent_hist[-1, VX_COL], agent_hist[-1, VY_COL]))
    dist_t0 = float(np.linalg.norm(p0))

    out: Dict = {
        "n_valid_frames": n_valid,
        "is_full60": n_valid >= FULL_FRAMES,
        "t0_speed": t0_speed,
        "dist_t0": dist_t0,
        # 🔴 test 侧唯一可用的判据：只读 t0 观测
        "is_t0_observable": bool(t0_speed > T0_SPEED_MPS and dist_t0 <= NEAR_EGO_M),
        "disp": None, "is_moving": False, "is_near": False, "is_encounter": False,
    }
    if n_valid == 0:
        return out

    last = int(np.where(valid)[0][-1])                 # 🔴 最后一个**有效**帧，不是 xy[-1]
    disp = float(np.linalg.norm(xy[last] - p0))
    out["disp"] = disp
    out["is_moving"] = bool(out["is_full60"] and disp > MOVING_DISP_M)
    out["is_near"] = bool(out["is_moving"] and dist_t0 <= NEAR_EGO_M)
    if ego_fut is not None and out["is_moving"]:
        d = np.linalg.norm(xy[:FULL_FRAMES] - np.asarray(ego_fut)[:FULL_FRAMES], axis=-1)
        out["is_encounter"] = bool(float(d[valid[:FULL_FRAMES]].min()) <= ENCOUNTER_M)
    return out


def scene_strata(row) -> Dict:
    """整个场景的分层计数与标志。

    :param row: `pd.read_pickle(pkl).iloc[0]`，需含 `AGENT_HISTORIES` / `AGENT_FUTURES` / `GROUND_TRUTH`。
    :returns: dict，含五个计数、ego 位移、以及三个冻结子集标志。
    """
    ah = np.asarray(_un(row["AGENT_HISTORIES"]))
    af = np.asarray(_un(row["AGENT_FUTURES"]))
    ego_fut = np.cumsum(np.asarray(_un(row["GROUND_TRUTH"]))[0], axis=0)

    c = {"n_agents": 0, "n_full60": 0, "n_moving": 0, "n_moving_near": 0,
         "n_encounter": 0, "n_t0_observable": 0}
    for j in range(ah.shape[0]):
        if np.all(ah[j] == 0):
            continue
        c["n_agents"] += 1
        s = agent_strata(ah[j], af[j], ego_fut)
        c["n_full60"] += int(s["is_full60"])
        c["n_moving"] += int(s["is_moving"])
        c["n_moving_near"] += int(s["is_near"])
        c["n_encounter"] += int(s["is_encounter"])
        c["n_t0_observable"] += int(s["is_t0_observable"])

    c["ego_disp"] = float(np.linalg.norm(ego_fut[-1]))
    c["ego_static"] = bool(c["ego_disp"] < 1.0)
    # ── 三个冻结子集（🔴 前两个只许诊断/训练侧用，第三个才是 test 侧）────────
    c["subset_moving_ge2"] = bool(c["n_moving"] >= 2)          # 主判据子集（GT 分层）
    c["subset_high_interaction"] = bool(c["n_encounter"] >= 1)  # 高交互子集（GT 分层）
    c["subset_t0_ge2"] = bool(c["n_t0_observable"] >= 2)        # test 侧分层（无 GT）
    return c


def summarize(scenes: Sequence[Dict]) -> Dict[str, float]:
    """把一批 `scene_strata` 汇总成比例（**不含区间** —— 区间必须用 log-cluster bootstrap 另算）。"""
    n = max(len(scenes), 1)
    keys = ("subset_moving_ge2", "subset_high_interaction", "subset_t0_ge2", "ego_static")
    out = {k: float(sum(bool(s[k]) for s in scenes) / n) for k in keys}
    for k in ("n_moving", "n_encounter", "n_t0_observable"):
        v = np.array([s[k] for s in scenes], dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_frac_zero"] = float((v == 0).mean())
    return out


__all__ = ["ENCOUNTER_M", "FULL_FRAMES", "MOVING_DISP_M", "NEAR_EGO_M", "T0_SPEED_MPS",
           "agent_strata", "scene_strata", "summarize"]
