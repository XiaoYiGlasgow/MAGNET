"""S3 决策图 —— scaling 曲线的统计功效，按 seed 数与噪声底两个轴。

这张图存在的理由（`results-analysis` 的图必须能答第三问）：
    它直接决定 §2 的 S4 预算要不要从 83 h 单卡改成 249 h。
    读者要注意的是：**k=1 那条线（现有排期）在整个 α 取值范围里都贴着地板**。
    这个观察改变的决策 = Q60 选 (a) 而不是 (c)。

数据来源：`s3_scaling_power_analysis.py` 的同一套蒙特卡洛（同 seed，结果可复现）；
噪声底来自 `s3_epoch_jitter_noise_floor.py` 对 10 个训练日志的实测。

用法：
    python s3_plot_power_curves.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s3_scaling_power_analysis import (  # noqa: E402
    ADE_BASE, N_TRIALS, RNG_SEED, TIER_SIZES, relative_gain, simulate_power,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/scaling_ladder/figures")
ALPHA_GRID = np.arange(0.01, 0.31, 0.01)
PANELS = [
    ("$\\sigma_{epoch}$ = 0.169  (measured: median late-epoch sd over 10 runs)", 0.169),
    ("$\\sigma_{avg5}$ = 0.069  (proposed: mean of last 5 epochs, measured ratio 0.410)", 0.169 * 0.410),
]
K_STYLES = {1: ("#c0392b", "-", 2.4), 2: ("#e67e22", "--", 1.8), 3: ("#2980b9", "-", 2.2), 6: ("#27ae60", ":", 1.8)}


def main() -> None:
    """画两栏功效曲线。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (title, sigma) in zip(axes, PANELS):
        for k, (color, ls, lw) in K_STYLES.items():
            power = [simulate_power(float(a), sigma, k, rng) for a in ALPHA_GRID]
            ax.plot(ALPHA_GRID, power, color=color, linestyle=ls, linewidth=lw,
                    label=f"k={k} seed" + ("  (current plan)" if k == 1 else ""))
        ax.axhline(0.80, color="0.35", linewidth=1.0, linestyle="-.")
        ax.text(0.012, 0.815, "80% power", fontsize=8, color="0.35")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"True scaling exponent $\alpha$  (ADE $\propto N^{-\alpha}$)")
        ax.set_xlim(0.01, 0.30)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Statistical power\n(P[95% CI of log-log slope excludes 0])")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.92)

    # 第二个 x 轴的含义：α 对应的实际 ADE 改善，便于判断「多大 α 才现实」
    for ax in axes:
        sec = ax.secondary_xaxis(
            "top",
            functions=(lambda a: 100 * relative_gain(np.asarray(a)),
                       lambda g: -np.log1p(-np.asarray(g) / 100) / np.log(TIER_SIZES[-1] / TIER_SIZES[0])),
        )
        sec.set_xlabel("Relative ADE improvement, L1$\\to$L3 (%)", fontsize=8)
        sec.tick_params(labelsize=7)

    fig.suptitle(
        f"Resolution of the scaling ladder: 25k$\\to$75k$\\to$245.5k ({TIER_SIZES[-1]/TIER_SIZES[0]:.1f}$\\times$), "
        f"baseline ADE {ADE_BASE:.3f} m, {N_TRIALS} Monte-Carlo trials",
        fontsize=10, y=1.06,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"figure-s3-power-curves.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info("已写 %s", path)


if __name__ == "__main__":
    main()
