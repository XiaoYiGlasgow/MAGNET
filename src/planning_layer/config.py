"""C4 规划层配置（R4-1 四件缺失件的开关 + 事前定死的候选网格与权重）。

🔴 设计纪律：本文件里的**任何数值都在见到闭环分数之前定死**，改动须走
`干预协议_INTERVENTION_PROTOCOL_v1_20260731.md` §11。
理由见 `C4_规划形式化_设计_20260801.md` §3.2 / §4.4。

🔴 依赖纪律：本包**只允许依赖标准库与 numpy** —— `plantf` env 没有 `torch_geometric`，
而这一层要在 MAGNET / PlanTF 两个 env 里共用同一份代码（坑#7）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

# ── 候选网格（事前定死，见设计文档 §3.2）────────────────────────────────
# 纵向：目标速度相对参考速度的比例，最后一档 0.0 = 紧急制动
LON_SPEED_RATIOS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
# 横向：沿参考路径法向的偏移量（m），正 = 左
LAT_OFFSETS_M: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

# ── 运动学硬界（R4-1 件①）────────────────────────────────────────────
# 🔴🔴 **数值全部取自 nuPlan 官方 comfort 判据配置，不是自己拍的**（08-01 复核修正）。
# 出处（相对 `/mnt/local/Xiao_Yi/nuplan-devkit/`）：
#   nuplan/planning/script/config/common/simulation_metric/low_level/
#     ego_lon_acceleration_statistics.yaml : min_lon_accel=-4.05 / max_lon_accel=2.40
#     ego_lon_jerk_statistics.yaml         : max_abs_lon_jerk=4.13
#     ego_lat_acceleration_statistics.yaml : max_abs_lat_accel=4.89
#     ego_yaw_rate_statistics.yaml         : max_abs_yaw_rate=0.95
#     ego_jerk_statistics.yaml             : max_abs_mag_jerk=8.37（合成 jerk，此处不单独约束）
# ⚠️ 第一版自己拍的 3.0 / −5.0 / 8.0 **全部越过官方 comfort 界** ——
#    也就是说「运动学可行」的候选会被官方判为 **uncomfortable**，而 comfort 是总分的一项。
#    这正是「调用公开实现做参照」抓出来的错。
A_LON_MAX = 2.40     # m/s^2 纵向加速度上界（官方 max_lon_accel）
A_LON_MIN = -4.05    # m/s^2 纵向减速度下界（官方 min_lon_accel）
A_LAT_MAX = 4.89     # m/s^2 侧向加速度上界（官方 max_abs_lat_accel）；决定弯道限速
JERK_MAX = 4.13      # m/s^3 纵向 jerk 上界（官方 max_abs_lon_jerk）
YAW_RATE_MAX = 0.95  # rad/s 偏航率上界（官方 max_abs_yaw_rate）；决定另一条弯道限速 v <= w/kappa
V_MAX = 30.0         # m/s 速度硬上界（无限速信息时的兜底）
# 速度剖面的前瞻窗口（秒）与采样点数：见 `candidates._rollout_speed_profile` 的说明。
# 0.5 s 由实测扫描选出（0/0.5/1.0/1.5/2.0 五档），是唯一能把 a_lat 压到界内且不损失推进的一档。
LOOKAHEAD_S = 0.5
LOOKAHEAD_SAMPLES = 6

# ── 避障（R4-1 件②）──────────────────────────────────────────────────
# ⚠️ 官方 `no_ego_at_fault_collisions` 判的是**包围盒相交**（阈值 0 m），不是质心距离。
#    这里用质心距离做**保守代理**：半径取「ego 半对角 + 一般车半对角」的量级。
#    该近似必须写进论文（设计文档 §7）。
COLLISION_RADIUS_M = 2.5    # 小于此间距判为碰撞（质心距离代理）
NEAR_MISS_RADIUS_M = 5.0    # 软惩罚起作用的间距

# ── TTC（对齐官方 time_to_collision_within_bound）─────────────────────
# 出处：simulation_metric/high_level/time_to_collision_within_bound_statistics.yaml
TTC_TIME_HORIZON_S = 3.0    # 官方 time_horizon
TTC_LEAST_MIN_S = 0.95      # 官方 least_min_ttc；低于它即判违规

# ── 交规（R4-1 件③）──────────────────────────────────────────────────
# 出处：high_level/speed_limit_compliance_statistics.yaml : max_overspeed_value_threshold=2.23
#      high_level/drivable_area_compliance_statistics.yaml: max_violation_threshold=0.3
#      high_level/driving_direction_compliance_statistics.yaml: 阈值 2 m（容忍）/ 6 m（不容忍）
OVERSPEED_TOLERANCE_MPS = 2.23
DRIVABLE_TOLERANCE_M = 0.3
BACKWARD_TOLERANCE_M = 2.0
BACKWARD_VIOLATION_M = 6.0

# ── 目标函数权重（事前定死）──────────────────────────────────────────
# 🔴 §7 硬约束：这些项与官方 score 的分项同源 ⇒ 论文必须声明「部分是同义反复」。
# 🔴 前提：**所有分项都已无量纲化到 O(1)**（见 `scoring.py`），权重才是真正的权衡系数；
#    第一版进展项用「米」，直接压倒限速项，被单测判据 [D] 抓到。
W_PROGRESS = 1.0
W_COLLISION = 4.0
W_TTC = 2.0        # 与 collision 同属件②（`collision` 开关一起控制）
W_RULES = 2.0
W_COMFORT = 0.5

DT = 0.1             # s 与训练/评测 horizon 一致
HORIZON = 60         # 步（6 s）


@dataclass(frozen=True)
class PlanningLayerConfig:
    """规划层配置。四个 bool 就是 R4-1 点名的四件缺失件，可独立开关做 leave-one-out。"""

    enabled: bool = False
    # R4-1 的四件缺失件
    dynamics: bool = True      # ① 车辆动力学约束
    collision: bool = True     # ② 避障
    rules: bool = True         # ③ 交规（限速 / 可行驶区域 / 行驶方向）
    progress: bool = True      # ④ 显式规划目标（进展项）
    # 候选网格
    lon_ratios: Tuple[float, ...] = LON_SPEED_RATIOS
    lat_offsets: Tuple[float, ...] = LAT_OFFSETS_M
    dt: float = DT
    horizon: int = HORIZON
    # 🔴 identity 候选（08-02 新增，见根因分析 progress_20260802.md 阶段52）
    #
    # 网格里的 `(lon_ratio=1.0, lat_offset=0.0)` **不是**上游原轨迹 ——
    # 它只保留路径形状，速度剖面被 `_rollout_speed_profile` 从头重新生成
    # （向「参考轨迹的**平均**速度」收敛）。实测只有上游恰好匀速时两者才相等：
    # 减速停车差 4.24 m、起步加速差 8.65 m、**塌陷差 15.03 m**。
    # ⇒ 候选集里没有「什么都不做」这个选项，argmax **没有下界保证**，
    #    层可以比上游更差（实测 MAGNET 0.1978→0.1248）。
    #
    # 补上上游原轨迹本身作为候选，argmax 的下界就是上游 ⇒ 层最差打平。
    #
    # 🔴 **依据是框架完备性，不是文献先例**（08-02 核查后更正）：
    #   「候选生成 → 打分 → argmax」若候选集里没有「不干预」这一项，
    #   argmax 在数学上就没有下界保证，层可以输出比上游更差的结果。
    #   这是该框架**自身的完备性要求**，不需要文献背书。
    #
    # ⚠️ **不得再写「这是 PDM 2306.07962 的标准做法」** —— 该 claim 已被原文证伪：
    #   PDM-Closed 的候选集是 **15 个 IDM policy 变体**
    #   （5 个目标速度 {20,40,60,80,100}% 限速 × 3 个横向中心线偏移 {−1, 0, +1} m），
    #   **全部由 IDM 生成，不含任何「上游原轨迹」候选**
    #   （PDM-Closed 是规则方法，压根没有 learned 上游这个概念；
    #    其 0 m 横向偏移只是「横向居中」，不是「保留原轨迹」）。
    #
    # ⚠️ 诚实性：本改动是**见到负分数之后**做的。写作时必须
    #   ①修前/修后两组数字都报；②说明依据是框架完备性而非先例；
    #   ③同报配对 bootstrap 的 CI（见 §8）。
    #
    # 留成开关而不是写死：`False` 可**逐位复现 08-01 那批七格的行为**，
    # 供「修前 / 修后」对照，也让 provenance 能自证这一格用的是哪种。
    include_identity: bool = True

    @property
    def n_candidates(self) -> int:
        return len(self.lon_ratios) * len(self.lat_offsets) + int(self.include_identity)

    def provenance(self) -> Dict[str, object]:
        """落进 provenance 的最小描述 —— 任何一格闭环都必须能自证用了哪个配置。"""
        return {
            "enabled": self.enabled,
            "dynamics": self.dynamics,
            "collision": self.collision,
            "rules": self.rules,
            "progress": self.progress,
            "n_candidates": self.n_candidates,
            "lon_ratios": list(self.lon_ratios),
            "lat_offsets": list(self.lat_offsets),
            "include_identity": self.include_identity,
        }


#: 预置档位 —— 主表与消融各一个名字，闭环用 `MAGNET_PLANNING_LAYER=<name>` 选。
_PRESETS: Dict[str, Dict[str, bool]] = {
    "off": dict(enabled=False),
    "full": dict(enabled=True, dynamics=True, collision=True, rules=True, progress=True),
    "no_dynamics": dict(enabled=True, dynamics=False, collision=True, rules=True, progress=True),
    "no_collision": dict(enabled=True, dynamics=True, collision=False, rules=True, progress=True),
    "no_rules": dict(enabled=True, dynamics=True, collision=True, rules=False, progress=True),
    "no_progress": dict(enabled=True, dynamics=True, collision=True, rules=True, progress=False),
}


def available_presets() -> Tuple[str, ...]:
    return tuple(_PRESETS.keys())


def preset(name: str) -> PlanningLayerConfig:
    """按名字取配置；名字不认识就**报错**，不静默降级（否则会静默跑错格）。"""
    if name not in _PRESETS:
        raise ValueError(
            f"未知的规划层预置 '{name}'；可选：{', '.join(sorted(_PRESETS))}"
        )
    return PlanningLayerConfig(**_PRESETS[name])


def preset_from_env(env_var: str = "MAGNET_PLANNING_LAYER") -> PlanningLayerConfig:
    """从环境变量取配置。**未设置 ⇒ 关闭**（默认关闭是判据 [A] 的前提）。

    走环境变量而不给 `run_closed_loop.sh` 加 flag 的理由见设计文档 §3.3 第 2 条：
    改 harness 会变更 sha ⇒ 触发规则基线四格重跑并让两组 provenance 分叉。
    """
    name = os.environ.get(env_var)
    if not name:
        return PlanningLayerConfig(enabled=False)
    return preset(name.strip())
