#!/usr/bin/env python3
"""BT-S1 / BT-S2 功能层判据判读（预注册 `旋钮B_gatv2算子对照_预注册判据_20260809.md（BT 版）` §3）。

判据（冻结）：
- BT-S1（非劣化）：三 seed 配对 Δ末5均值ADE（b_transformer − fr_base）的均值 ≤ **+0.145**（δ_main^new）；
- BT-S2（改善声明门）：三 seed Δ 同号为负 且 |Δ均值| > **0.059**（δ_strict^new）才可写「观察到改善」。

算法 = Q60 口径（`q77_readout_last5.py` 同款）：per-seed 末 5 epoch `val_ade` 均值，
ddof=1 样本 sd；配对按同 seed。K0-S1 先例同款（配对 Δ 均值比阈）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
BASE = _GNN / "experiments/knobs"
OUT = _GNN / "scratchpad/b_transformer_s1s2_readout_20260809.json"
SEEDS = (42, 1, 2)
D_MAIN, D_STRICT = 0.145, 0.059


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _last5(cell: str, seed: int) -> float:
    rec = json.loads((BASE / cell / f"seed{seed}" / "loss_record.json").read_text())
    va = rec["history"]["val_ade"]
    if len(va) != 45:
        raise ValueError(f"{cell}/seed{seed}: {len(va)}/45 epoch —— 训练未跑满，拒绝读数")
    return _mean(va[-5:])


def main() -> int:
    fr = [_last5("fr_base_L1", s) for s in SEEDS]
    bg = [_last5("b_transformer_L1", s) for s in SEEDS]
    delta = [b - a for a, b in zip(fr, bg)]
    dm = _mean(delta)
    s1 = dm <= D_MAIN
    all_neg = all(d < 0 for d in delta)
    s2 = all_neg and abs(dm) > D_STRICT

    report = {
        "fr_base_末5均值": dict(zip(map(str, SEEDS), fr)),
        "b_transformer_末5均值": dict(zip(map(str, SEEDS), bg)),
        "fr_base_三seed": {"mean": _mean(fr), "sd": _sd(fr)},
        "b_transformer_三seed": {"mean": _mean(bg), "sd": _sd(bg)},
        "配对Δ_per_seed": dict(zip(map(str, SEEDS), delta)),
        "Δ均值": dm,
        "BT-S1": {"判据": f"Δ均值 ≤ +{D_MAIN}", "PASS": s1},
        "BT-S2": {"判据": f"三seed同号为负 且 |Δ均值| > {D_STRICT}",
                 "同号为负": all_neg, "触发": s2,
                 "措辞": "观察到 ADE 改善" if s2 else "未观察到 ADE 改善"},
        "口径": "Q60 末5均值；batch=8+fix=True+高agent密度档；配对按同 seed",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"fr_base  {_mean(fr):.4f} ± {_sd(fr):.4f} | b_transformer {_mean(bg):.4f} ± {_sd(bg):.4f}")
    print(f"配对 Δ: {['%+.4f' % d for d in delta]} | Δ均值 {dm:+.4f}")
    print(f"BT-S1 (≤ +{D_MAIN}): {'✅ PASS' if s1 else '🔴 FAIL'}")
    print(f"BT-S2: {'触发 ⇒ 观察到改善' if s2 else '未触发 ⇒ 「未观察到 ADE 改善」'}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
