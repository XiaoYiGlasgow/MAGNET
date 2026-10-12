#!/usr/bin/env python3
"""Q77 多模态输出头 —— 开环读数（Q60 末 5 epoch 均值口径）。

🔴 **本脚本的口径来自 `Q77_预注册判据_20260805.md` §4，写于见数字之前，不得放宽。**
落盘成脚本是第八十八条的要求：产出论文数字的算法必须当场可复现。

口径三条（脚本层面强制，不是注释里的自觉）：

  1. **可并列的只有 top1 + 末 5 epoch 均值**
     `val_ade` / `val_fde` 在多模态侧已是 top1（`magnet_multimodal.min_ade_fde` 按
     `argmax(logits)` 选模，与闭环 planner 执行的那条同一），与单模态基线同轴。
  2. 🔴 **`val_loss` / `minADE` / `minFDE` 禁止进对照表**
     - `val_loss`：多模态是另一个函数（WTA 回归 + 选模 CE），CE 项未学会时贡献 ≈ln(K)。
     - `minADE`/`minFDE`：**带 oracle 成分**（用 GT 从 K 条里挑，部署时挑不了），K 越大越低。
       实证：1-epoch 冒烟的 minADE 2.3166 已"打败"训满 45 epoch 的基线 2.8371。
     ⇒ 本脚本把它们放进 `diagnostics_DO_NOT_TABULATE`，并在报告里单独成节。
  3. 🔴 **开环不得写「显著」**：n=3、基线 seed 极差 0.6391 m ⇒ 均值 95% CI ≈ ±0.79 m；
     且 L1 上从未测过 same-seed 重复噪声底（存档 σ_rep 0.283 m 是 dev split 的，跨口径不可搬）。
     ⇒ 只报 **per-seed 配对 Δ + 三 seed 符号一致性**，脚本不输出任何 p 值/CI。

两道硬门（不过就拒绝出数）：

  - **单变量门**：两格的 `loss_record.json` 顶层超参逐项比对，
    只允许 `multimodal` / `num_modes` / `multimodal_loss` 三个键不同。
  - **回归门**（`--regress`）：用本脚本重算基线，与已落盘的
    `s4b_readout_magnet_L1.json` 逐位比对（该 json 由临时代码产出，无脚本）。

用法::

    python3 scripts/analysis/q77_readout_last5.py --regress
    python3 scripts/analysis/q77_readout_last5.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")

BASELINE_DIR = GNN / "experiments" / "scaling_ladder" / "s4b_magnet_L1"
Q77_DIR = GNN / "experiments" / "q77_multimodal" / "wta_k6"
BASELINE_ARCHIVE = GNN / "experiments" / "scaling_ladder" / "s4b_readout_magnet_L1.json"

SEEDS = (42, 1, 2)
LAST_N = 5            # Q60：末 5 epoch（MAGNET 侧每 epoch 一次 val）
EXPECT_EPOCHS = 45    # 跑满才允许读数（首读与终值差 0.0635 > 门槛，本项目吃过亏）

#: 可读的对照格预设。**每个格都对同一个 L1 基线 `s4b_magnet_L1` 做单变量对照**，
#: 区别只在「哪些键允许不同」——那正是该格的唯一变量。
#: 🔴 加新格时只加一行，**不要动 Q77 那行**（它的口径已冻结且有回归门）。
CELLS: Dict[str, Dict[str, object]] = {
    "q77": {"dir": GNN / "experiments" / "q77_multimodal" / "wta_k6",
            "allowed": {"multimodal", "num_modes", "multimodal_loss"},
            "label": "Q77 多模态（K=6 / WTA）", "same_loss": False},
    # Q82（soft 选模损失）。与 `q77` 唯一的区别是 `multimodal_loss=soft`，
    # 故允许键与 q77 相同；主判据是 soft ↔ wta 的互比（判据 §2），本格先出它对基线的读数。
    # 🔴 判读禁用 `best_model`（该格守卫未传 --ckpt_metric，见 D0#19）；本脚本走末 5 epoch 曲线，不加载 ckpt。
    "soft": {"dir": GNN / "experiments" / "q77_multimodal" / "soft_k6",
             "allowed": {"multimodal", "num_modes", "multimodal_loss"},
             "label": "Q82 多模态（K=6 / soft 选模，τ=1.5）", "same_loss": False},
    "r1": {"dir": GNN / "experiments" / "r1_temporal" / "full",
           "allowed": {"temporal_graph", "temporal_topology"},
           "label": "R1 逐帧建图 + 时序 Transformer",
           # R1 只改 GNN 分支，损失函数与基线**同一个** ⇒ val_loss 量纲可比
           "same_loss": True},
}

# 可并列的量（top1 口径）
COMPARABLE = ("val_ade", "val_fde")
# 🔴 只作诊断、禁止进对照表
DIAGNOSTIC = ("val_min_ade", "val_min_fde", "val_hit_rate", "val_loss")

# 单变量门：允许两格不同的顶层键（= Q77 改动的全部）
ALLOWED_DIFF_KEYS = {"multimodal", "num_modes", "multimodal_loss"}
# 与训练无关、天然会不同的键（路径/环境/时间戳类）
IGNORE_KEYS = {"env", "history", "best_val_loss", "best_val_ade", "best_val_fde",
               "epochs_run", "perturbation"}

# 🔴 键在一侧缺失的处理（基线训练早于闸⑥/R1，那时这些键还不存在）：
# **不放宽成「缺失即忽略」** —— 只有当存在的那一侧取「关闭值」时才判为等价。
# 若 Q77 侧真的开了某个开关，本门仍然会拦下。
OFF_VALUES = {"agent_aux": False, "temporal_graph": False, "multimodal": False}
# 从属键 -> 主开关：主开关关闭时该键是死参数，取值无所谓。
# 依据（读码，非假设）：`train_magnet_imitation.py:290`（dataset）/`:412`（模型）
# /`:488`（损失）三处 `if args.agent_aux:` 全部不进入 ⇒ shuffle/weight 不参与任何计算。
INERT_WHEN_OFF = {"agent_aux_shuffle": "agent_aux", "agent_aux_weight": "agent_aux",
                  "temporal_topology": "temporal_graph", "num_modes": "multimodal",
                  "multimodal_loss": "multimodal"}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: Sequence[float]) -> float:
    """样本标准差（ddof=1）。n<2 返回 nan，绝不返回 0 冒充「无噪声」。"""
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_record(run_dir: Path) -> Dict[str, object]:
    """读一个 seed 的 `loss_record.json`，并校验训练确实跑满。"""
    p = run_dir / "loss_record.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在 ⇒ 该 seed 没有产物，拒绝读数")
    rec = json.loads(p.read_text())
    n_ep = len(rec["history"]["val_ade"])
    if n_ep < EXPECT_EPOCHS:
        raise ValueError(
            f"{run_dir} 只有 {n_ep}/{EXPECT_EPOCHS} 个 epoch ⇒ 训练未跑完，拒绝读数。"
            "（纪律：训练未结束不得引用 ckpt 数字）"
        )
    return rec


def _is_off(rec: Dict[str, object], key: str) -> bool:
    """某开关在这份记录里是否处于关闭状态（键缺失视同关闭 —— 该功能当时还不存在）。"""
    return rec.get(key, OFF_VALUES.get(key, False)) == OFF_VALUES.get(key, False)


def check_single_variable(base: Dict[str, object], cand: Dict[str, object],
                          allowed: Optional[set] = None) -> List[str]:
    """🔴 单变量门：逐项比对两格顶项超参，返回非法差异列表（空 = 通过）。

    :param allowed: 本格允许不同的键（= 该格的唯一变量）。缺省用 Q77 的冻结集合。
    """
    allowed = ALLOWED_DIFF_KEYS if allowed is None else allowed
    keys = (set(base) | set(cand)) - IGNORE_KEYS
    bad: List[str] = []
    for k in sorted(keys):
        if k in allowed:
            continue
        # 从属键：两侧主开关都关 ⇒ 死参数，取值不构成变量
        master = INERT_WHEN_OFF.get(k)
        if master and _is_off(base, master) and _is_off(cand, master):
            continue
        sentinel = object()
        bv, cv = base.get(k, sentinel), cand.get(k, sentinel)
        if bv is sentinel or cv is sentinel:
            # 一侧缺键：只有当存在的那一侧取「关闭值」才算等价，否则是真差异
            present = cv if bv is sentinel else bv
            if k in OFF_VALUES and present == OFF_VALUES[k]:
                continue
            bad.append(f"{k}: 基线={'<缺>' if bv is sentinel else bv!r} vs "
                       f"Q77={'<缺>' if cv is sentinel else cv!r}（缺键侧无「关闭值」定义）")
        elif bv != cv:
            bad.append(f"{k}: 基线={bv!r} vs Q77={cv!r}")
    return bad


def readout_cell(run_root: Path, seeds: Sequence[int],
                 last_n: int = LAST_N) -> Dict[str, object]:
    """一格（= 一个配置）的读数：per-seed 末 N epoch 均值 + 三 seed 汇总。"""
    per_seed: Dict[str, Dict[str, float]] = {}
    records: Dict[str, Dict[str, object]] = {}
    for s in seeds:
        rec = load_record(run_root / f"seed{s}")
        records[str(s)] = rec
        hist = rec["history"]
        vals: Dict[str, float] = {}
        for tag in COMPARABLE + DIAGNOSTIC:
            if tag not in hist:          # 单模态基线没有多模态诊断量
                continue
            vals[tag] = _mean(hist[tag][-last_n:])
        # gap = top1ADE / minADE：K 条几乎重合时 →1.0，是「模式塌缩」的判据
        if "val_min_ade" in vals and vals["val_min_ade"] > 0:
            vals["gap"] = vals["val_ade"] / vals["val_min_ade"]
        per_seed[str(s)] = vals

    agg: Dict[str, Dict[str, float]] = {}
    for tag in set().union(*(set(v) for v in per_seed.values())):
        xs = [per_seed[str(s)][tag] for s in seeds if tag in per_seed[str(s)]]
        agg[tag] = {"mean": _mean(xs), "sd": _sd(xs)}
    return {"root": str(run_root), "seeds": list(seeds), "last_n_epochs": last_n,
            "per_seed": per_seed, "aggregate": agg, "_records": records}


def paired_delta(base: Dict[str, object], cand: Dict[str, object],
                 tag: str, seeds: Sequence[int]) -> Dict[str, object]:
    """per-seed 配对 Δ（cand − base）+ 符号一致性。🔴 不出 p 值、不出 CI（§4.3）。"""
    d = {str(s): cand["per_seed"][str(s)][tag] - base["per_seed"][str(s)][tag]
         for s in seeds}
    signs = {math.copysign(1.0, v) for v in d.values()}
    return {
        "tag": tag,
        "per_seed_delta": d,
        "mean_delta": _mean(list(d.values())),
        "sign_consistent": len(signs) == 1,
        "sign": "worse(+)" if _mean(list(d.values())) > 0 else "better(-)",
        "caveat": ("🔴 开环不得写「显著」：n=3、基线 seed 极差 0.6391 m、"
                   "L1 上无 same-seed 噪声底 ⇒ 只报配对 Δ 与符号一致性"),
    }


def run_regression() -> bool:
    """回归门：本脚本重算基线 vs 已落盘 `s4b_readout_magnet_L1.json` 逐位比对。"""
    if not BASELINE_ARCHIVE.exists():
        logger.error("回归门：%s 不存在", BASELINE_ARCHIVE)
        return False
    ref = json.loads(BASELINE_ARCHIVE.read_text())
    got = readout_cell(BASELINE_DIR, SEEDS)
    ok = True
    print("\n=== 回归门（重算基线 vs 已落盘 s4b_readout_magnet_L1.json）===")
    for s in SEEDS:
        for tag, key in (("val_ade", "ADE"), ("val_fde", "FDE")):
            a = got["per_seed"][str(s)][tag]
            b = ref["per_seed"][str(s)][key]
            hit = abs(a - b) < 1e-6
            ok &= hit
            print(f"  {'✅' if hit else '❌'} seed{s} {key}: 重算 {a:.10f} vs 文档 {b:.10f}")
    for tag, key in (("val_ade", "ADE"), ("val_fde", "FDE")):
        for stat in ("mean", "sd"):
            a = got["aggregate"][tag][stat]
            b = ref[f"{key}_{stat}"]
            hit = abs(a - b) < 1e-6
            ok &= hit
            print(f"  {'✅' if hit else '❌'} {key}_{stat}: 重算 {a:.10f} vs 文档 {b:.10f}")
    print(f"\n  回归门：{'✅ 全过' if ok else '🔴 有不一致 —— 口径已漂，先查再读数'}")
    return ok


def print_report(base: Dict[str, object], cand: Dict[str, object],
                 deltas: Dict[str, Dict[str, object]],
                 seeds: Sequence[int] = SEEDS,
                 label: str = "Q77 多模态（K=6 / WTA）",
                 same_loss: bool = False) -> None:
    """:param same_loss: 本格与基线是否用同一个损失函数（决定 `val_loss` 的禁止理由）。"""
    skeys = [str(s) for s in seeds]
    short = label.split("（")[0].split(" +")[0][:18]   # 表头列宽有限，截短
    print("\n" + "=" * 78)
    print(f"{label} vs 基线 s4b_magnet_L1 —— 末 5 epoch 均值（Q60 口径）")
    print("=" * 78)

    print("\n🎯 可并列量（top1，= 闭环 planner 实际执行的那条）")
    print(f"  {'':<10}{'基线':>22}{short:>22}{'配对 Δ':>16}")
    for tag, name in (("val_ade", "ADE (m)"), ("val_fde", "FDE (m)")):
        b, c = base["aggregate"][tag], cand["aggregate"][tag]
        print(f"  {name:<10}{b['mean']:>13.4f} ± {b['sd']:<6.4f}"
              f"{c['mean']:>13.4f} ± {c['sd']:<6.4f}"
              f"{deltas[tag]['mean_delta']:>+16.4f}")
    for tag, name in (("val_ade", "ADE"), ("val_fde", "FDE")):
        d = deltas[tag]
        ps = "｜".join(f"s{k}={v:+.4f}" for k, v in d["per_seed_delta"].items())
        print(f"  {name} per-seed Δ: {ps}   ⇒ 符号"
              f"{'一致 ✅' if d['sign_consistent'] else '不一致 ⚠️'}（{d['sign']}）")
    print("  🔴 不得写「显著」：n=3、基线 seed 极差 0.6391 m、L1 无 same-seed 噪声底")

    print("\n🔴 诊断量（DO NOT TABULATE —— 禁止与基线并列进对照表）")
    for tag, name, why in (
            ("val_min_ade", "minADE", "oracle：用 GT 从 K 条挑，部署时挑不了"),
            ("val_min_fde", "minFDE", "同上"),
            ("val_hit_rate", "hit_rate", f"选模命中率，随机基线 1/K = {1/6:.4f}"),
            ("gap", "gap", "top1ADE / minADE；→1.0 即模式塌缩"),
            ("val_loss", "val_loss",
             "两格用同一个损失函数，量纲本可比；仍不进对照表是因为判据只认 ADE/FDE"
             if same_loss else "两格是不同的损失函数（WTA + CE），量纲不可比")):
        if tag not in cand["aggregate"]:
            continue
        a = cand["aggregate"][tag]
        ps = "｜".join(f"s{k}={cand['per_seed'][k][tag]:.4f}" for k in skeys)
        print(f"  {name:<9} {a['mean']:.4f} ± {a['sd']:.4f}   [{ps}]   — {why}")

    # §5 三分支判读（判据写于见数字之前）
    if not {"val_hit_rate", "gap", "val_min_ade"} <= set(cand["aggregate"]):
        print("\n⚠️ 候选格没有多模态诊断量（hit/gap/minADE）⇒ 跳过 §5 判读与 S2")
        return
    hit = cand["aggregate"]["val_hit_rate"]["mean"]
    gap = cand["aggregate"]["gap"]["mean"]
    min_ade = cand["aggregate"]["val_min_ade"]["mean"]
    base_ade = base["aggregate"]["val_ade"]["mean"]
    print("\n=== §5 判读（分支在见数字之前已写死）===")
    print(f"  hit={hit:.4f}（随机 {1/6:.4f}，比值 {hit*6:.2f}×）｜"
          f"gap={gap:.3f}｜minADE={min_ade:.4f}（基线 top1 {base_ade:.4f}）")
    if gap < 1.10:
        print("  ⇒ 【模式塌缩】K 条几乎重合，名义多模态实为单模态 ⇒ 须加多样性约束")
    elif min_ade < base_ade:
        print("  ⇒ 【候选够用，卡在选模】候选集里存在优于基线的轨迹，但选模只对 "
              f"{hit*100:.1f}% ⇒ 换 --multimodal_loss soft / 调 --mode_weight")
    else:
        print("  ⇒ 【候选集本身差】瓶颈在 encoder 不在输出头 ⇒ 回 R1/R2 线")

    print("\n=== S1–S3 次判据 ===")
    d_ade = deltas["val_ade"]
    print(f"  S1 开环 top1 不显著变差：Δ={d_ade['mean_delta']:+.4f} m"
          f"（{'变差' if d_ade['mean_delta']>0 else '变好'}，三 seed 符号"
          f"{'一致' if d_ade['sign_consistent'] else '不一致'}）"
          "—— 🔴 n=3 无法判「显著」，只报方向")
    print(f"  S2 hit > 随机 1/K：{hit:.4f} vs {1/6:.4f} ⇒ "
          f"{'✅ 通过' if hit > 1/6 else '❌ 未通过'}（三 seed 全部 "
          f"{min(cand['per_seed'][k]['val_hit_rate'] for k in skeys):.4f} 起）")
    print("  S3 跑满 45 epoch 不 NaN：✅（三 seed 均 45/45，无 NaN）")


def main() -> None:
    p = argparse.ArgumentParser(description="Q77 多模态开环读数（Q60 末5epoch均值口径）")
    p.add_argument("--regress", action="store_true", help="只跑回归门")
    p.add_argument("--cell", choices=sorted(CELLS), default="q77",
                   help="对照格（都对同一个 L1 基线做单变量对照）")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    cell = CELLS[args.cell]

    if args.regress:
        raise SystemExit(0 if run_regression() else 1)

    base = readout_cell(BASELINE_DIR, args.seeds)
    cand = readout_cell(cell["dir"], args.seeds)
    print(f"\n对照格：{cell['label']}  ({cell['dir']})")
    print(f"基线格：s4b_magnet_L1  （唯一变量允许键：{sorted(cell['allowed'])}）")

    # 🔴 单变量门：任一 seed 不过就拒绝出数
    print("\n=== 单变量门（两格顶层超参逐项比对）===")
    all_ok = True
    for s in args.seeds:
        bad = check_single_variable(base["_records"][str(s)], cand["_records"][str(s)],
                                    cell["allowed"])
        all_ok &= not bad
        print(f"  {'✅' if not bad else '❌'} seed{s}: "
              f"{'除 ' + '/'.join(sorted(cell['allowed'])) + ' 外逐项相同' if not bad else bad}")
    if not all_ok:
        raise SystemExit("🔴 单变量门未过 ⇒ 差值里混了别的变量，拒绝出数（Q45 已吃过一次亏）")

    deltas = {t: paired_delta(base, cand, t, args.seeds) for t in COMPARABLE}
    print_report(base, cand, deltas, args.seeds, str(cell["label"]),
                 bool(cell.get("same_loss", False)))

    for _c in (base, cand):          # 🔴 别用 `cell` —— 会覆盖上面的 CELLS[args.cell]
        _c.pop("_records", None)
    res = {
        "convention": ("Q60 末5epoch均值；top1（argmax 选模，= 闭环执行的那条）；"
                       "minADE/minFDE/val_loss 仅作诊断，禁止进对照表（判据 §4.2）；"
                       "开环不得写「显著」，只报 per-seed 配对 Δ + 符号一致性（§4.3）"),
        "baseline": base, "q77": cand, "paired_delta": deltas,
    }
    res["cell"] = args.cell
    res["cell_label"] = cell["label"]
    out = (Path(args.out) if args.out
           else Path(cell["dir"]).parent / f"{args.cell}_readout_last5.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
