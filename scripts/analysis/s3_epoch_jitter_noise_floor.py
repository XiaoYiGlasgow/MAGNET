"""S3 前置 —— 用「末段 epoch 间抖动」重估开环 ADE 的噪声底。

为什么要重估（08-02 功效分析里暴露的矛盾）：
    现有两个噪声估计差 **17 倍**，而 S3/S4 的 seed 预算完全取决于选哪个：
      * σ_rep      = 0.283 m —— rep1/rep2 同 seed 重复的极差，但 **n=2**；
      * σ_seed(def)= 0.0163 m —— `ablation_2x2_f1` 三 seed 的 sd。
    「同一个 seed 重复两次比三个不同 seed 之间还散 17 倍」在统计上讲不通，
    必须先判定哪个是伪的，否则功效结论会在「完全不可行」和「k=2 就够」之间摇摆。

领先假说：**两个都不是纯训练噪声，真正的噪声源是 checkpoint 选取**。
    gate4_seed2 的日志里，相邻 epoch 的 val ADE 就在 1.863↔2.249 之间跳（差 0.386），
    比 σ_rep 还大。`checkpoint_best` = 在这样一条抖动序列上取 min ⇒
      ① 引入与抖动同量级的方差；
      ② 引入**向下偏倚**（取 min 系统性低估）。
    若为真，则 σ_rep 0.283 大体是对的，而 0.0163 是三个 seed 恰好落得近的巧合。

本脚本从训练日志直接量化末段抖动，样本量比 n=2 大得多。

用法：
    python s3_epoch_jitter_noise_floor.py [--tail 10]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
LINE_RE = re.compile(r"epoch\s+(\d+)/(\d+).*?ADE=([\d.]+)m\s+FDE=([\d.]+)m")


def parse_log(path: Path) -> List[tuple[int, float, float]]:
    """从训练日志解析 (epoch, ADE, FDE)。"""
    rows: List[tuple[int, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = LINE_RE.search(line)
        if match:
            rows.append((int(match.group(1)), float(match.group(3)), float(match.group(4))))
    return rows


def main() -> None:
    """逐日志报末段抖动，并汇总成噪声底估计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=10, help="取末 N 个 epoch")
    args = parser.parse_args()

    logs = sorted(GNN.glob("data/*train*.log"))
    per_run_sd: List[float] = []
    per_run_range: List[float] = []
    min_vs_mean: List[float] = []

    logger.info("===== 末 %d 个 epoch 的 val ADE 抖动 =====", args.tail)
    logger.info("  %-42s %5s %8s %8s %8s %8s", "run", "n", "mean", "sd", "极差", "min")
    for path in logs:
        rows = parse_log(path)
        if len(rows) < args.tail:
            continue
        ade = np.array([r[1] for r in rows[-args.tail:]])
        sd = float(ade.std(ddof=1))
        rng = float(ade.max() - ade.min())
        per_run_sd.append(sd)
        per_run_range.append(rng)
        # 「取 min」相对「取均值」低多少 —— 这就是 checkpoint_best 的向下偏倚
        min_vs_mean.append(float(ade.mean() - ade.min()))
        logger.info("  %-42s %5d %8.4f %8.4f %8.4f %8.4f",
                    path.name[:42], len(ade), ade.mean(), sd, rng, ade.min())

    if not per_run_sd:
        logger.error("没有解析到任何训练日志")
        return

    sd_arr = np.array(per_run_sd)
    bias_arr = np.array(min_vs_mean)
    logger.info("\n===== 汇总（n=%d 个 run）=====", len(sd_arr))
    logger.info("  末段 epoch 间 sd：中位数 %.4f｜均值 %.4f｜范围 %.4f–%.4f",
                float(np.median(sd_arr)), float(sd_arr.mean()), float(sd_arr.min()), float(sd_arr.max()))
    logger.info("  末段极差：中位数 %.4f｜均值 %.4f",
                float(np.median(per_run_range)), float(np.mean(per_run_range)))
    logger.info("  🔴 checkpoint_best 的向下偏倚（末段均值 − 末段 min）：中位数 %.4f｜均值 %.4f",
                float(np.median(bias_arr)), float(bias_arr.mean()))

    logger.info("\n===== 与两个既有噪声估计对照 =====")
    logger.info("  σ_rep（同 seed 重复极差，n=2）        = 0.2830")
    logger.info("  σ_seed（nearby 三 seed sd）           = 0.1228")
    logger.info("  σ_seed（default 三 seed sd）          = 0.0163")
    logger.info("  本脚本：单 run 末段 epoch 间 sd 中位数 = %.4f", float(np.median(sd_arr)))
    logger.info("\n  判读：若末段 sd 与 σ_rep 同量级，则 σ_rep 主要是 checkpoint 选取噪声、")
    logger.info("        不是 seed 噪声，而 default 三 seed 的 0.0163 是巧合 ⇒ 功效分析")
    logger.info("        必须用大的那个，k=1 一定不可行。")


if __name__ == "__main__":
    main()
