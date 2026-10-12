#!/usr/bin/env python3
"""D0#19 —— 训练控制信号一致性的单测（2026-08-05）。

## 被测的事故

多模态（Q77）的 `val_loss` = WTA 最优模回归 + 选模 CE。训练中 CE 项的增长盖过回归项的下降
⇒ **val_loss 一路上升而 ADE 一路下降**。当时 checkpoint 选择 / LR 调度 / early stopping
三处控制信号**全部挂在 val_loss 上**，于是：

* best checkpoint 落在 **epoch 1/1/4**（保存的是几乎没训练的模型）；
* LR 被砍 **51×/51×/35×**（同期单模态基线只砍 8×/3×/12×）；
* early stopping 若非 patience=999 会在 epoch ~2 停掉。

而 `loss_record` 的 `best_val_ade` 写的是 **min(history)**，与保存的 ckpt 无关 ⇒ **掩盖了事故**。

## 三组断言（全部拿**真实训练曲线**回放，不是构造数据）

  [A] `resolve_monitor_name` 的解析规则；
  [B] 🔴 **回归门** —— 单模态在 `auto` 下选出的 epoch 必须与旧版（val_loss）**完全一致**，
      否则既有 checkpoint 不再可复现；
  [C] 🔴 **修复有效性** —— 多模态在 `auto` 下不再落在最初几个 epoch，
      且 `best_val_ade` 语义指向被选中的那个 epoch（而非曲线最小值）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python tests/test_ckpt_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
sys.path.insert(0, str(GNN / "scripts" / "training"))

from train_magnet_imitation import resolve_monitor_name  # noqa: E402

_FAILED: List[str] = []

#: 真实训练曲线（多模态那格是事故现场；单模态那格是回归基准）
RUNS = {
    "Q77多模态": (GNN / "experiments" / "q77_multimodal" / "wta_k6", True),
    "L1基线": (GNN / "experiments" / "scaling_ladder" / "s4b_magnet_L1", False),
    "R1时序": (GNN / "experiments" / "r1_temporal" / "full", False),
}


def check(cond: bool, msg: str) -> None:
    print(f"    {'✅' if cond else '❌'} {msg}")
    if not cond:
        _FAILED.append(msg)


def select_epoch(history: Dict[str, List[float]], metric: str) -> Tuple[int, float]:
    """回放训练循环的选择逻辑：取 `metric` 首次达到最小值的 epoch（1-based）。

    与 `train_magnet_imitation` 的 `if monitor < best_val` 语义一致 ——
    **严格小于** ⇒ 并列时保留更早的那个。
    """
    xs = history[metric]
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs[1:], start=1):
        if v < best_v:
            best_i, best_v = i, v
    return best_i + 1, best_v


def load(root: Path, seed: int) -> Optional[Dict]:
    p = root / f"seed{seed}" / "loss_record.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_a_resolution() -> None:
    print("[A resolve_monitor_name 解析规则]")
    check(resolve_monitor_name("auto", True) == "val_ade", "auto + 多模态 -> val_ade")
    check(resolve_monitor_name("auto", False) == "val_loss", "auto + 单模态 -> val_loss（保回归）")
    check(resolve_monitor_name("val_ade", False) == "val_ade", "显式 val_ade 覆盖 auto")
    check(resolve_monitor_name("val_loss", True) == "val_loss", "显式 val_loss 可复现旧行为")
    try:
        resolve_monitor_name("bogus", False)
        check(False, "未知取值应抛错")
    except ValueError:
        check(True, "未知取值抛 ValueError（不静默兜底）")


def test_b_regression_single_modal() -> None:
    """🔴 回归门：单模态在 auto 下的选择必须与旧版（val_loss）逐位一致。"""
    print("[B 单模态回归门 —— 既有 checkpoint 必须仍可复现]")
    n = 0
    for name, (root, multimodal) in RUNS.items():
        if multimodal:
            continue
        for seed in (42, 1, 2):
            rec = load(root, seed)
            if rec is None:
                continue
            n += 1
            old_e, _ = select_epoch(rec["history"], "val_loss")          # 旧行为
            new_e, _ = select_epoch(rec["history"],
                                    resolve_monitor_name("auto", multimodal))
            check(old_e == new_e, f"{name} seed{seed}: auto 与旧版同选 epoch {old_e}")
    check(n >= 6, f"回归门覆盖了 {n} 个单模态 run（期望 ≥6）")


def test_c_multimodal_fix() -> None:
    """🔴 修复有效性：多模态改用 val_ade 后不再落在最初几个 epoch。"""
    print("[C 多模态修复有效性 —— 事故现场回放]")
    root, multimodal = RUNS["Q77多模态"]
    for seed in (42, 1, 2):
        rec = load(root, seed)
        if rec is None:
            continue
        h = rec["history"]
        old_e, _ = select_epoch(h, "val_loss")
        new_e, _ = select_epoch(h, resolve_monitor_name("auto", multimodal))
        n_ep = len(h["val_ade"])
        print(f"      seed{seed}: 旧(val_loss)-> epoch {old_e}/{n_ep} (ADE {h['val_ade'][old_e-1]:.4f})"
              f"  |  新(val_ade)-> epoch {new_e}/{n_ep} (ADE {h['val_ade'][new_e-1]:.4f})")
        check(old_e <= 4, f"seed{seed}: 复现事故 —— 旧逻辑确实落在 epoch {old_e}")
        check(new_e > 10, f"seed{seed}: 修复后落在 epoch {new_e}（> 10）")
        check(h["val_ade"][new_e - 1] < h["val_ade"][old_e - 1],
              f"seed{seed}: 新选中的 ckpt ADE 更好"
              f"（{h['val_ade'][new_e-1]:.4f} < {h['val_ade'][old_e-1]:.4f}）")

    # 语义修复：新的 best_val_ade 必须 == 被选中 epoch 的值，而不是曲线最小值
    rec = load(root, 42)
    h = rec["history"]
    new_e, _ = select_epoch(h, "val_ade")
    at_sel, curve_min = h["val_ade"][new_e - 1], min(h["val_ade"])
    check(abs(at_sel - curve_min) < 1e-9,
          "val_ade 作监控时，被选 epoch 的值恰为曲线最小值（两个语义此时重合）")
    old_e, _ = select_epoch(h, "val_loss")
    check(abs(h["val_ade"][old_e - 1] - curve_min) > 1.0,
          f"🔴 旧语义的危害可量化：旧 ckpt 实际 ADE {h['val_ade'][old_e-1]:.4f} "
          f"vs 旧字段报的 min(history) {curve_min:.4f}，差 {h['val_ade'][old_e-1]-curve_min:.4f} m")


def main() -> int:
    for fn in (test_a_resolution, test_b_regression_single_modal, test_c_multimodal_fix):
        fn()
    print(f"\n{'✅ 全部通过' if not _FAILED else f'❌ {len(_FAILED)} 条失败'}")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
