"""生成结果筛选器：从模型生成的危险场景里挑出「物理意义上真危险」的样本，构建 benchmark。

背景：模型在物理工作点生成大量场景，但混有退化样本（两车远离 / 一方静止 / 只在 t=0 擦过，
看上去完全不危险）。本模块在**生成结果输出端**做筛选，只保留 near-miss 或碰撞的有效危险场景，
顺带挡掉初始数据筛选层漏进来的退化对，无需重训。

⚠️ 关键：训练用的 danger_score「帐篷函数」把碰撞算 0 分（视碰撞为无用 terminal），
不能直接当 benchmark 判据——benchmark 要保留碰撞。所以本模块用**独立的、基于 min_dist 的判据**。

⚠️ min_dist 用「生成 ego vs 生成 agent」两条生成轨迹（不是 GT agent）。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# 让本模块能 import 到 src/shared（src/danger_generation/core/ → 上三级是 src/）
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from shared.utils.danger_score import DangerScoreConfig  # noqa: E402

__all__ = [
    "BenchmarkFilterConfig", "compute_scenario_metrics", "is_dangerous_scenario",
    "SpeedSeedConfig", "is_valid_speed_seed", "start_direction_cos",
    "truncate_at_collision",
]

DT = 0.1  # 时间步长（秒），与 NuPlan 10Hz 一致
# 碰撞距离从 danger_score 取，不重复定义（保持与训练度量一致）
COLLISION_DISTANCE = DangerScoreConfig().collision_distance  # 2.0 m


@dataclass(frozen=True)
class BenchmarkFilterConfig:
    """benchmark 筛选阈值（全部 config 可调）。四个设计决策对应的参数都收在这里。"""

    # —— 点1：核心危险距离 ——
    # 生成 ego 与生成 agent 的最小距离 ≤ 此值才算「危险」。
    # 独立于 DangerScoreConfig.near_miss_distance：那是训练 tent 的峰值，这是 benchmark 判危门槛。
    # 取 5.0m（= danger_score tent 支撑区上界 min_safe）：min_dist 是质心距离，SUV/大型车
    # 车身从质心外延 ~2-2.5m，质心 3-5m 对大车已是擦碰/极近；且 3-5m 在 tent 里仍有非零 danger。
    danger_distance_threshold: float = 5.0

    # —— 点2：碰撞是否纳入 ——
    # True：碰撞（min_dist ≤ COLLISION_DISTANCE）也算危险，保留。
    # False：只保留 near-miss（COLLISION_DISTANCE < min_dist ≤ danger_distance_threshold）。
    include_collision: bool = True

    # —— 点3：反退化（a/b/c/d，边界放宽，挡掉「看似有 min_dist 但其实没危险」的退化样本）——
    min_approach_frame: int = 3      # (a) argmin(dist) ≥ 此帧——排除「开局擦过即分开」
    min_approach_drop: float = 1.0   # (b) dist[0] − min_dist ≥ 此值——两车至少真实接近过 1m
    min_vehicle_speed: float = 0.5   # (c) 两车各自平均速度 ≥ 此值——排除一方静止（沿用 ttc_filter 静止阈值）
    min_close_frames: int = 2        # (d) dist ≤ 危险阈值 的帧数 ≥ 此值——贴近不能只是瞬时单帧

    # —— 点4：物理有效性 gate（兜底剔除「飞出路面 / 加速度不物理」的无效驾驶场景）——
    enable_validity_gate: bool = True  # 总开关
    offroad_half_width: float = 3.0    # 道路软边界半宽（同 guidance，不用 1.75m 单车道严判）
    max_offroad_frac: float = 0.3      # offroad@offroad_half_width 的点比例 ≤ 此值
    # 平均加速度 ≤ 此值（m/s²）。放宽到 1.5g：急加速本身是事故成因，不该当噪声滤掉；
    # 只剔明显不物理（均值破 1.5g）。卡均值不卡瞬时峰值——允许 sudden 急加速。
    max_accel_mean: float = 15.0

    # —— 点5：起点连续性 gate（剔除生成 ego/agent 起点反向的物理不可能样本）——
    # 生成轨迹前几步方向若与历史朝向夹角 cos < 此值，视为瞬时倒车（proximity 把 agent 往
    # 后方 ego 拽过头）→ 剔除。start_anchor 已减少此类，gate 兜底保证 benchmark 无反向。
    min_start_cos: float = -0.3
    start_check_steps: int = 5         # 取前 N 步平均方向判起点连续性


def _offroad_fraction(xy: np.ndarray, lane_segs: np.ndarray, half_width: float) -> float:
    """轨迹点落在所有车道 half_width 之外的比例（镜像 eval_v5_svo._lane_metrics，自包含避免反向依赖）。

    对每个点求到最近车道线段的投影距离，> half_width 即算 offroad。变道靠 min-over-segs
    天然不罚（就近车道）。无有效车道时返回 nan（上层按「未知」放行）。
    """
    valid = np.abs(lane_segs).sum(axis=-1) > 0     # (K,) 过滤零填充段
    segs = lane_segs[valid]
    if len(segs) == 0 or len(xy) == 0:
        return float("nan")
    p = xy[:, None, :]            # (T, 1, 2)
    a = segs[None, :, :2]         # (1, K, 2) 段起点
    b = segs[None, :, 2:]         # (1, K, 2) 段终点
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(-1).clip(min=1e-8)
    t = np.clip((ap * ab).sum(-1) / denom, 0.0, 1.0)   # 投影参数 clamp 到线段内
    proj = a + t[..., None] * ab
    dist = np.linalg.norm(p - proj, axis=-1)           # (T, K)
    min_dist = dist.min(axis=1)                        # (T,) 每点到最近车道距离
    return float((min_dist > half_width).mean())


def _speed_mean(xy: np.ndarray) -> float:
    """轨迹绝对位置 (T,2) 的平均速度幅值（m/s）= 平均逐步位移 / dt。"""
    if len(xy) < 2:
        return 0.0
    vel = np.diff(xy, axis=0) / DT
    return float(np.linalg.norm(vel, axis=-1).mean())


def start_direction_cos(xy: np.ndarray, hist_heading: float, n: int = 5) -> float:
    """生成轨迹 (T,2) 前 n 步平均方向 与 历史朝向 的 cos。

    cos≈1=顺历史方向；cos<0=起点反向（物理不可能的瞬时倒车）。
    用绝对轨迹的逐步增量均值定方向（增量本身即速度方向，勿再差分）。
    """
    inc = np.diff(np.asarray(xy), axis=0)[:n]   # 前 n 步位移增量
    v = inc.mean(axis=0)
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return 1.0                              # 几乎不动：不判反向
    hdir = np.array([np.cos(hist_heading), np.sin(hist_heading)])
    return float((v / nv) @ hdir)


def _accel_mean(xy: np.ndarray) -> float:
    """轨迹绝对位置 (T,2) 的平均加速度幅值（m/s²）= 位置二阶差分 / dt²。

    与 eval_synergy_report._accel_mean 同一物理量（那里输入是增量，等价 diff(增量)/dt²）。
    """
    if len(xy) < 3:
        return 0.0
    acc = np.diff(xy, n=2, axis=0) / (DT * DT)
    return float(np.linalg.norm(acc, axis=-1).mean())


def compute_scenario_metrics(
    ego_xy: np.ndarray,        # (T, 2) 生成 ego 绝对轨迹
    agent_xy: np.ndarray,      # (T, 2) 生成 agent 绝对轨迹
    lane_segs: np.ndarray,     # (K, 4) [x0,y0,x1,y1]，允许零填充
    cfg: BenchmarkFilterConfig,
) -> Dict[str, float]:
    """对单个生成场景计算筛选所需的全部指标。"""
    dist = np.linalg.norm(ego_xy - agent_xy, axis=-1)   # (T,) 逐帧两车距离
    min_dist = float(dist.min())
    return {
        "min_dist": min_dist,
        "argmin": int(dist.argmin()),                    # 最近点出现在第几帧
        "dist0": float(dist[0]),                         # 初始距离
        "approach_drop": float(dist[0] - min_dist),      # 真实接近量
        "close_frames": int((dist <= cfg.danger_distance_threshold).sum()),
        "ego_speed": _speed_mean(ego_xy),
        "agent_speed": _speed_mean(agent_xy),
        "ego_offroad": _offroad_fraction(ego_xy, lane_segs, cfg.offroad_half_width),
        "agent_offroad": _offroad_fraction(agent_xy, lane_segs, cfg.offroad_half_width),
        "ego_accel": _accel_mean(ego_xy),
        "agent_accel": _accel_mean(agent_xy),
    }


def is_dangerous_scenario(
    metrics: Dict[str, float],
    cfg: BenchmarkFilterConfig,
) -> Tuple[bool, str]:
    """按四类判据决定生成场景是否保留进 benchmark。

    返回 (是否保留, 原因)。原因字符串用于统计被剔分布——保留时为 'kept'，否则为具体拒绝项。
    判据顺序：核心危险距离 → 反退化 a/b/c/d → 物理有效性 gate。
    """
    m = metrics

    # —— 核心：必须在危险距离内 ——
    if m["min_dist"] > cfg.danger_distance_threshold:
        return False, "too_far"                       # 两车始终太远，不危险
    if (not cfg.include_collision) and m["min_dist"] <= COLLISION_DISTANCE:
        return False, "collision_excluded"            # 配置为排除碰撞时，剔掉碰撞场景

    # —— 反退化 ——
    if m["argmin"] < cfg.min_approach_frame:
        return False, "approach_too_early"            # (a) 开局擦过即分开
    if m["approach_drop"] < cfg.min_approach_drop:
        return False, "no_approach"                   # (b) 距离没真正下降（典型退化，如 sample46）
    if (m["ego_speed"] < cfg.min_vehicle_speed) or (m["agent_speed"] < cfg.min_vehicle_speed):
        return False, "vehicle_static"                # (c) 一方静止（如 sample46 的 ego 停着）
    if m["close_frames"] < cfg.min_close_frames:
        return False, "close_too_brief"               # (d) 贴近只是瞬时单帧

    # —— 物理有效性 gate（ego 与 agent 都查；offroad 为 nan 时视为未知、放行）——
    if cfg.enable_validity_gate:
        for who, off, acc in (
            ("ego", m["ego_offroad"], m["ego_accel"]),
            ("agent", m["agent_offroad"], m["agent_accel"]),
        ):
            if (not np.isnan(off)) and (off > cfg.max_offroad_frac):
                return False, f"{who}_offroad"        # 大段飞出路面
            if acc > cfg.max_accel_mean:
                return False, f"{who}_accel"          # 加速度明显不物理

    return True, "kept"


def truncate_at_collision(
    ego_xy: np.ndarray,        # (T, D) 生成 ego 绝对轨迹（前 2 列为 x,y）
    agent_xy: np.ndarray,      # (T, D) 生成 agent 绝对轨迹
    collision_distance: float = COLLISION_DISTANCE,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """碰撞场景把生成轨迹截断于碰撞帧。

    模型联合一次性出 60 帧，碰撞后两车会穿过彼此继续走（物理不可能）。本函数对
    **碰撞场景**（质心最近距 min_dist ≤ collision_distance）截断于**最近接近帧**
    （argmin，= impact 高潮，截断后末点距 = min_dist，与 metrics 一致），丢弃碰后帧。
    非碰撞场景（min_dist > 阈值）原样返回、collision_frame 为 None。

    返回 (ego_截断, agent_截断, collision_frame)。GT 是真实物理不穿模，调用方不应截断 GT。
    """
    ego = np.asarray(ego_xy)
    agent = np.asarray(agent_xy)
    dist = np.linalg.norm(ego[:, :2] - agent[:, :2], axis=1)
    if float(dist.min()) > collision_distance:
        return ego_xy, agent_xy, None
    cf = int(dist.argmin())
    return ego[:cf + 1], agent[:cf + 1], cf


# ============================================================
# speed split 种子判据（同向 + 同道走廊）
# ============================================================
# 背景：speed 危险机制是「同道纵向」（急加速→追尾/抢道 near-miss）。但 ttc_filter 的
# TTC 选择会混入反向车、以及隔 2 条车道、侧向 ~7m 的平行流车（GT 欧氏近距也能 <8m，
# 但根本不在同一条纵向冲突走廊里）。本判据在**生成种子阶段**挑出真正的同道纵向冲突对，
# 用「agent GT 到 ego GT 路径的几何垂距」判同道/紧邻道——不依赖（不可靠的）车道采集。


@dataclass(frozen=True)
class SpeedSeedConfig:
    """speed benchmark 的种子筛选阈值（全部 config 可调）。"""

    # 同向：|agent 相对 ego 朝向| ≤ 此值（剔反向 ~27% + 横向 ~16%）
    max_rel_heading_deg: float = 45.0
    # 同道/紧邻道：agent GT 到 ego GT 路径的最小侧向（垂向）距离 ≤ 此值。
    # 3.5m ≈ 一条车道宽：同道(侧向~0)与直接相邻道/抢道都保留，隔 2 条道(~7m)剔除。
    max_lateral_offset_m: float = 3.5
    # 在「走廊帧」（侧向已达标）里的最小纵向 gap ≤ 此值——保证 near-miss 物理可达
    # （6s @ 高速可闭合 ~40m；不从相隔太远硬拽）。
    max_longitudinal_m: float = 40.0
    # agent GT 平均速度 ≥ 此值才算「在动」——剔停车退化对（沿用 ttc_filter/benchmark 静止阈值）。
    min_agent_speed: float = 0.5
    enable: bool = True

    # —— agent GT 车道成员：剔「离道车被选为 key agent」（初始 TTC 选择无 map，泄漏 ~11%）——
    # 用 ego 相对信任：仅当 ego GT 在道（本地车道覆盖可信）但 agent GT 大段离道时才剔，
    # 覆盖差（ego 也离道）或无覆盖（nan）时不信任地图、放行，避免误伤稀疏覆盖区。
    enable_lane_check: bool = True
    offroad_half_width: float = 3.0    # 软边界半宽（与 guidance road_half_width / gate 一致）
    max_agent_offroad: float = 0.3     # agent GT 离道帧比例 > 此值算离道车
    ego_on_thr: float = 0.2            # ego GT 离道比例 < 此值才信任本地地图覆盖


def _wrap_deg(rad: float) -> float:
    """把弧度角差包到 (-180, 180]，返回绝对角度（度）。"""
    return float(abs(np.degrees(np.arctan2(np.sin(rad), np.cos(rad)))))


def is_valid_speed_seed(
    ego_gt_xy: np.ndarray,     # (T, 2) ego GT 未来绝对轨迹
    agent_gt_xy: np.ndarray,   # (T, 2) agent GT 未来绝对轨迹
    ego_hist: np.ndarray,      # (T_hist, 5) ego 历史，末帧取朝向
    agent_hist: np.ndarray,    # (T_hist, 5) agent 历史，末帧取朝向
    cfg: SpeedSeedConfig,
    lane_segs: Optional[np.ndarray] = None,   # (K, 4) ego-centric 车道段，做 agent 车道成员判据
) -> Tuple[bool, str]:
    """判断一个 GT 对是否为合格的 speed 纵向冲突种子。

    返回 (是否保留, 原因)。保留时为 'kept'，否则为具体拒绝项（用于统计被剔分布）。
    判据顺序：agent 在动 → agent 在车道 → 同向 → 同道走廊（侧向）→ 纵向可达。
    """
    if not cfg.enable:
        return True, "kept"

    # —— agent 在动：剔停车退化对（路口停车等）——
    if _speed_mean(np.asarray(agent_gt_xy)) < cfg.min_agent_speed:
        return False, "agent_static"

    # —— agent 在车道：剔离道车被选为 key agent（ego 在道但 agent GT 大段离道）——
    if cfg.enable_lane_check and lane_segs is not None:
        agent_off = _offroad_fraction(np.asarray(agent_gt_xy), np.asarray(lane_segs),
                                      cfg.offroad_half_width)
        ego_off = _offroad_fraction(np.asarray(ego_gt_xy), np.asarray(lane_segs),
                                    cfg.offroad_half_width)
        if (not np.isnan(agent_off)) and (not np.isnan(ego_off)) \
                and ego_off < cfg.ego_on_thr and agent_off > cfg.max_agent_offroad:
            return False, "agent_offroad_gt"

    # —— 同向：agent 相对 ego 的朝向差 ——
    rel_deg = _wrap_deg(float(agent_hist[-1, 2]) - float(ego_hist[-1, 2]))
    if rel_deg > cfg.max_rel_heading_deg:
        return False, "opposing_or_crossing"

    # —— 同道走廊：把 (agent-ego) 分解到 ego 路径的切向/法向 ——
    ego = np.asarray(ego_gt_xy, dtype=np.float64)
    agent = np.asarray(agent_gt_xy, dtype=np.float64)
    tang = np.gradient(ego, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True).clip(min=1e-6)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)   # 左法向
    rel = agent - ego
    lateral = np.abs((rel * normal).sum(axis=1))           # (T,) 逐帧侧向距离
    lon = (rel * tang).sum(axis=1)                         # (T,) 逐帧纵向（带符号）

    # 走廊帧 = 侧向已达标（agent 进入 ego 同道/紧邻道）的那些帧
    corridor = lateral <= cfg.max_lateral_offset_m
    if not corridor.any():
        return False, "lateral_too_far"                    # 全程隔车道平行流，不在纵向冲突走廊
    # 走廊帧里最小纵向 gap：agent 进入车道时是否曾纵向靠近到可闭合范围
    if np.abs(lon[corridor]).min() > cfg.max_longitudinal_m:
        return False, "longitudinal_too_far"               # 纵向始终太远，near-miss 不物理

    return True, "kept"
