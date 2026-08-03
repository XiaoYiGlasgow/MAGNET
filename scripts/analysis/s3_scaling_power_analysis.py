"""S3 起跑前的必答题 —— 开环 ADE 这条 scaling 曲线到底有没有分辨率。

来由（阶段63 §四「仍然未做的」）：
    第二轮已把 scaling 曲线的主指标从闭环 final_score 改成开环 ADE/FDE
    （MAGNET 闭环 78.5% 场景得 0、平局占 184–217/256，闭环没有分辨率）。
    但**换指标 ≠ 有分辨率**（第十三条同类）。开环噪声底 σ_rep = 0.283 m，
    而三级只从 25 k 涨到 245.5 k（9.82×），预期增益能否超过噪声底，从没算过。

本脚本不猜 α（真实 scaling 指数），而是回答两个可判定的问题：

  Q-a  给定每级 k 个 seed，**需要多大的 α** 才能让 log-log 斜率的 95% CI 排除 0？
  Q-b  给定预算（MAGNET 三级 k=1 合计已 83 h 单卡），**k 该取几**？

判读方式按实际会用的那种：三级各 k 个 seed，拟合 log(ADE) ~ log(N) 的斜率，
看斜率的 95% CI 含不含 0（第四十九条：阈值判据不如直接用 CI）。
两点差值的问法会低估检出力 —— 三个点的回归比一次比较更有力。

🔴 噪声底本身有两个互相矛盾的估计，脚本对两个都算，因为选哪个会改变结论：
  * σ_rep = **0.283 m** —— `ablation_2x2_f1_rep1/rep2` 的 B_seed42（1.9257 vs 2.2085）
    同 seed 重复的极差。**n=2，且两次的 best epoch 不同（ep36 vs ep32）**，
    所以它混进了 checkpoint 选取的变异，多半是**高估**。
  * σ_seed(nearby) = **0.1228 m** —— `ablation_2x2_f1_nearby` 三 seed 的 sd。
  * ⚠️ 反常：default 三 seed 的 sd 只有 **0.0163**，是 σ_rep 的 1/17。
    同 seed 重复比不同 seed 之间还散 17 倍，在统计上讲不通 ⇒ σ_rep 不可当作
    纯训练噪声引用，必须同报这条。

用法：
    python s3_scaling_power_analysis.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 阶梯（Q58 两城裁定后的实测值，L4 已删除）
TIER_SIZES = np.array([25_000, 75_000, 245_500], dtype=float)

# 基线 ADE：ablation_2x2_f1_nearby 三 seed 均值（C2/闸④ 系列的共同对照）
ADE_BASE = 2.0154

# 两个互相矛盾的噪声估计，见模块 docstring
SIGMAS: Dict[str, float] = {
    "σ_rep=0.283（同 seed 重复极差，n=2，疑高估）": 0.283,
    "σ_epoch=0.169（实测末段 epoch sd 中位数，n=10 run）": 0.169,
    "σ_seed=0.1228（nearby 三 seed sd）": 0.1228,
    "σ_avg5=0.076（改口径：末 5 epoch 均值，σ_epoch/√5）": 0.076,
    "σ_seed=0.0163（default 三 seed sd，疑为巧合）": 0.0163,
}

ALPHAS = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
N_SEEDS = [1, 2, 3, 6]
N_TRIALS = 4000
RNG_SEED = 20260802


def simulate_power(alpha: float, sigma: float, k: int, rng: np.random.Generator) -> float:
    """蒙特卡洛：三级各 k 个 seed，log-log 斜率的 95% CI 排除 0 的比例。

    Args:
        alpha: 真实 scaling 指数，ADE ∝ N^(-alpha)。
        sigma: 单次训练的 ADE 标准差（加性）。
        k: 每级的 seed 数。
        rng: 随机数发生器。

    Returns:
        统计功效（0–1）。
    """
    if k * len(TIER_SIZES) - 2 <= 0:
        return float("nan")  # 自由度不足，无法给斜率置信区间

    truth = ADE_BASE * (TIER_SIZES / TIER_SIZES[0]) ** (-alpha)
    log_n = np.repeat(np.log(TIER_SIZES), k)
    n_obs = log_n.size
    dof = n_obs - 2

    # 一次生成全部 trial 的观测：shape (N_TRIALS, n_obs)
    mean = np.repeat(truth, k)
    obs = mean[None, :] + rng.normal(0.0, sigma, size=(N_TRIALS, n_obs))
    # ADE 不可能为负；截断到一个小正数，避免 log 爆掉
    obs = np.maximum(obs, 1e-3)
    log_y = np.log(obs)

    x_centered = log_n - log_n.mean()
    sxx = (x_centered ** 2).sum()
    slope = (log_y * x_centered).sum(axis=1) / sxx
    intercept = log_y.mean(axis=1) - slope * log_n.mean()
    resid = log_y - (intercept[:, None] + slope[:, None] * log_n[None, :])
    se_slope = np.sqrt((resid ** 2).sum(axis=1) / dof / sxx)

    # 双侧 95%，用 t 分布临界值
    from scipy import stats  # 局部导入：只有本函数需要

    t_crit = stats.t.ppf(0.975, dof)
    ci_lo = slope - t_crit * se_slope
    ci_hi = slope + t_crit * se_slope
    return float(np.mean((ci_lo > 0) | (ci_hi < 0)))


def relative_gain(alpha: float) -> float:
    """L1 → L3 的相对 ADE 改善。"""
    return 1.0 - (TIER_SIZES[-1] / TIER_SIZES[0]) ** (-alpha)


def main() -> None:
    """跑功效表并给结论。"""
    rng = np.random.default_rng(RNG_SEED)
    ratio = TIER_SIZES[-1] / TIER_SIZES[0]

    logger.info("===== S3 scaling 曲线分辨率（开环 ADE）=====")
    logger.info("阶梯 %s（L1→L3 = %.2f×）｜基线 ADE %.4f",
                [int(x) for x in TIER_SIZES], ratio, ADE_BASE)
    logger.info("\n【第一问】若真实 α 是某值，L1→L3 的 ADE 改善有多大（点估计，无噪声）：")
    logger.info("  %8s %12s %12s", "α", "相对改善", "绝对改善 m")
    for alpha in ALPHAS:
        gain = relative_gain(alpha)
        logger.info("  %8.3f %11.1f%% %12.4f", alpha, 100 * gain, gain * ADE_BASE)

    logger.info("\n【第二问】统计功效 —— 三级各 k 个 seed，log-log 斜率 95%% CI 排除 0 的概率")
    logger.info("  （%d 次蒙特卡洛；k=1 时自由度 3-2=1，CI 极宽）", N_TRIALS)
    for label, sigma in SIGMAS.items():
        logger.info("\n  噪声 = %s", label)
        header = "  %8s" + " %9s" * len(N_SEEDS)
        logger.info(header, "α", *[f"k={k}" for k in N_SEEDS])
        for alpha in ALPHAS:
            row = [simulate_power(alpha, sigma, k, rng) for k in N_SEEDS]
            logger.info("  %8.3f" + " %8.0f%%" * len(row), alpha, *[100 * r for r in row])

    logger.info("\n【第三问】达到 80%% 功效所需的最小 α：")
    logger.info("  %46s %9s %9s %9s %9s", "噪声", *[f"k={k}" for k in N_SEEDS])
    for label, sigma in SIGMAS.items():
        mins: List[str] = []
        for k in N_SEEDS:
            found = "—"
            for alpha in np.arange(0.01, 0.61, 0.005):
                if simulate_power(float(alpha), sigma, k, rng) >= 0.80:
                    found = f"{alpha:.3f}"
                    break
            mins.append(found)
        logger.info("  %46s %9s %9s %9s %9s", label[:46], *mins)

    logger.info("\n【成本对照】MAGNET 三级 k=1 已是 83 h 单卡（L1 6 + L2 18 + L3 59）")
    for k in N_SEEDS:
        logger.info("  k=%d ⇒ %3d h 单卡 ≈ %.1f 天三卡并行", k, 83 * k, 83 * k / 24 / 3)


if __name__ == "__main__":
    main()
