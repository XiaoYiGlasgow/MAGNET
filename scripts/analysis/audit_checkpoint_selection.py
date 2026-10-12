#!/usr/bin/env python3
"""D0#19 存量审计 —— 扫全部训练产物，找出 best checkpoint 落点异常的 run（2026-08-05）。

## 为什么需要

08-05 查实：训练循环用 `val_loss` 同时驱动 **checkpoint 选择 / LR 调度 / early stopping**，
而多模态的 `val_loss` 含选模 CE 项、与 ADE **反向** ⇒ best checkpoint 落在 epoch 1。
`loss_record` 的旧字段 `best_val_ade` 写的是 `min(history)`、与保存的 ckpt 无关
⇒ **从记录上看不出来**，必须重算。

本脚本对每个 run 回放两种选择逻辑，报告：

* `best@loss` —— 旧逻辑实际选中的 epoch 及其 val_ade（＝磁盘上 `best_model_default.pth` 的水平）；
* `best@ade` —— 修复后会选中的 epoch 及其 val_ade；
* `代价` —— 两者的 ADE 差（这就是用错 ckpt 的直接损失）。

🔴 判为**受影响**的条件（任一）：
  1. `best@loss` 的 epoch ≤ 3 且总 epoch > 10（选在训练最开头）；
  2. 两种逻辑的 ADE 差 > 0.30 m（≈ 本项目 σ_rep 的量级）。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/audit_checkpoint_selection.py
    $PY scripts/analysis/audit_checkpoint_selection.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
EXP = GNN / "experiments"

#: 受影响判据（见 docstring）
EARLY_EPOCH_MAX = 3
MIN_EPOCHS_FOR_FLAG = 10
ADE_COST_THRESHOLD = 0.30


def select_epoch(xs: List[float]) -> Tuple[int, float]:
    """首次达到最小值的 epoch（1-based）—— 与训练循环的 `严格小于` 语义一致。"""
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs[1:], start=1):
        if v < best_v:
            best_i, best_v = i, v
    return best_i + 1, best_v


def audit_run(rec_path: Path) -> Optional[Dict]:
    try:
        rec = json.loads(rec_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    h = rec.get("history") or {}
    if not h.get("val_ade") or not h.get("val_loss"):
        return None
    n_ep = len(h["val_ade"])
    e_loss, _ = select_epoch(h["val_loss"])
    e_ade, _ = select_epoch(h["val_ade"])
    ade_at_loss, ade_at_ade = h["val_ade"][e_loss - 1], h["val_ade"][e_ade - 1]
    cost = ade_at_loss - ade_at_ade
    affected = ((e_loss <= EARLY_EPOCH_MAX and n_ep > MIN_EPOCHS_FOR_FLAG)
                or cost > ADE_COST_THRESHOLD)
    return {
        "run": str(rec_path.parent.relative_to(EXP)),
        "n_epochs": n_ep,
        "multimodal": bool(rec.get("multimodal", False)),
        "ckpt_metric_recorded": rec.get("ckpt_metric"),     # 新字段；旧记录为 None
        "best_epoch_at_loss": e_loss, "ade_at_loss": ade_at_loss,
        "best_epoch_at_ade": e_ade, "ade_at_ade": ade_at_ade,
        "ade_cost": cost,
        # 旧字段报的值（＝ min(history)）与磁盘 ckpt 的真实水平之差 —— 「记录掩盖了多少」
        "recorded_best_val_ade": rec.get("best_val_ade"),
        "affected": affected,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="D0#19 存量审计：best checkpoint 落点")
    p.add_argument("--root", type=Path, default=EXP)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()

    rows = [r for rec in sorted(a.root.rglob("loss_record.json"))
            if (r := audit_run(rec)) is not None]
    if not rows:
        raise SystemExit(f"{a.root} 下没有可读的 loss_record.json")

    rows.sort(key=lambda r: (-r["ade_cost"], r["run"]))
    aff = [r for r in rows if r["affected"]]

    print(f"\n=== D0#19 存量审计：{len(rows)} 个 run，其中 🔴 受影响 {len(aff)} 个 ===")
    print(f"{'run':<46}{'ep':>4}{'MM':>4}{'best@loss':>11}{'ADE':>8}{'best@ade':>10}{'ADE':>8}{'代价':>8}")
    for r in rows:
        mark = "🔴" if r["affected"] else "  "
        print(f"{mark}{r['run']:<44}{r['n_epochs']:>4}{'✔' if r['multimodal'] else '-':>4}"
              f"{r['best_epoch_at_loss']:>11}{r['ade_at_loss']:>8.3f}"
              f"{r['best_epoch_at_ade']:>10}{r['ade_at_ade']:>8.3f}{r['ade_cost']:>+8.3f}")

    if aff:
        print(f"\n🔴 受影响的 {len(aff)} 个 run —— 它们磁盘上的 `best_model_default.pth` 不可用：")
        for r in aff:
            print(f"   · {r['run']}：ckpt 实际 ADE **{r['ade_at_loss']:.4f}**，"
                  f"而 loss_record 的 best_val_ade 报 {r['recorded_best_val_ade']:.4f}"
                  f" ⇒ 记录掩盖了 {r['ade_at_loss'] - (r['recorded_best_val_ade'] or 0):.4f} m")
        print("\n   处置：改用 `final_model_default.pth`，或用修复后的 "
              "`--ckpt_metric val_ade` 重训；**任何用过这些 ckpt 的下游产物都要复核**"
              "（闭环 / BN calibration / 干预审计）。")
    else:
        print("\n🟢 未发现受影响的 run。")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"n_runs": len(rows), "n_affected": len(aff),
                                      "criteria": {"early_epoch_max": EARLY_EPOCH_MAX,
                                                   "min_epochs_for_flag": MIN_EPOCHS_FOR_FLAG,
                                                   "ade_cost_threshold": ADE_COST_THRESHOLD},
                                      "rows": rows}, indent=2, ensure_ascii=False))
        print(f"\n已写 {a.json}")


if __name__ == "__main__":
    main()
