"""C4 规划层 —— 主入口 `refine()`。

接口刻意做成**纯几何 + 纯数据**：输入上游轨迹的全局 xy，输出精化后的全局 xy。
这样同一份代码可以接三个上游（MAGNET / PlanTF / CV 常速外推），
且不依赖 devkit 类型 ⇒ 在 `nuplan` 与 `plantf` 两个 env 里都能 import（坑#7）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from . import config as C
from . import geometry as G
from .candidates import Candidate, EgoSnapshot, generate
from .scoring import SceneContext, progress_scale, score


def refine(
    reference_xy: np.ndarray,
    ego: EgoSnapshot,
    ctx: Optional[SceneContext] = None,
    cfg: Optional[C.PlanningLayerConfig] = None,
    route_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """把上游参考轨迹精化成一条满足 R4-1 四件缺失件的可执行轨迹。

    :param reference_xy: ``[T, 2]`` 上游 planner 的全局轨迹
    :param ego: 当前 ego 快照（后轴中心）
    :param ctx: 场景信息（agent 前推位置 / 限速 / 可行驶区域 / 进展方向）
    :param cfg: 规划层配置；None ⇒ 从环境变量取（默认关闭）
    :param route_xy: 可选 route 中心线，上游塌陷时兜底几何
    :returns: ``[T, 2]`` 精化后的全局轨迹

    🔴 **判据 [A]（命门）**：`cfg.enabled=False` 时**原样返回输入**（`max|Δ| = 0`），
    因此接进 planner 后默认行为逐位不变，两组 run 只差一个变量。
    """
    cfg = cfg if cfg is not None else C.preset_from_env()
    if not cfg.enabled:
        return reference_xy

    ref = np.asarray(reference_xy, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"reference_xy 形状应为 [T,2]，实际 {ref.shape}")

    ctx = ctx if ctx is not None else SceneContext()
    if ctx.origin is None:
        ctx = _with_origin(ctx, np.array([ego.x, ego.y]))
    if ctx.progress_direction is None and route_xy is not None and len(route_xy) >= 2:
        ctx = _with_direction(ctx, G.path_direction(np.asarray(route_xy, float), ego.heading))

    cands = generate(ref, ego, cfg, route_xy=route_xy)
    best, _ = select(cands, ctx, cfg, ego.heading)
    return best.xy


def select(
    cands, ctx: SceneContext, cfg: C.PlanningLayerConfig, ego_heading: float
) -> Tuple[Candidate, np.ndarray]:
    """在候选集中取分数最高者；同分取**索引最小**（确定性，便于复现）。

    进展项的归一化尺度在**本候选集上**算（见 `scoring.progress_scale`），
    因此必须先扫一遍候选再打分。
    """
    scale = progress_scale(cands, ctx, ego_heading)
    scores = np.array([score(c, ctx, cfg, ego_heading, scale) for c in cands], dtype=float)
    return cands[int(np.argmax(scores))], scores


def _with_origin(ctx: SceneContext, origin: np.ndarray) -> SceneContext:
    # 🔴 必须用 `replace` 而不是手写字段列表（08-01 接线时抓到的第 10 个实现错）：
    #    手写版漏掉了 `ego_half_width` / `ego_front_m` / `ego_rear_m` 三个字段，
    #    而 `refine()` 在 `ctx.origin is None` 时**必然**走这条路径
    #    ⇒ 调用方即使传了车身尺寸也会被静默丢掉，可行驶区域退化成只查质心，
    #    与官方 `drivable_area_compliance`（判四角）不一致。
    #    `replace` 让新增字段自动透传，杜绝同类漏字段。
    return replace(ctx, origin=origin)


def _with_direction(ctx: SceneContext, direction: np.ndarray) -> SceneContext:
    return replace(ctx, progress_direction=direction)


def constant_velocity_reference(ego: EgoSnapshot, cfg: C.PlanningLayerConfig) -> np.ndarray:
    """CV 上游 —— 沿当前朝向按当前速度匀速外推。

    🔴 这是 C4 的**诚实性支点**（设计文档 §4.1）：如果「CV + 层」与「MAGNET + 层」
    不可区分，就说明学习组件在这一规划形式化下没有可测量的额外贡献。
    """
    t = np.arange(cfg.horizon, dtype=float) * cfg.dt
    direction = np.array([np.cos(ego.heading), np.sin(ego.heading)])
    return np.array([ego.x, ego.y])[None, :] + (ego.speed * t)[:, None] * direction[None, :]
