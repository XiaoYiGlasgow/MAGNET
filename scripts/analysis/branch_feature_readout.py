"""分支特征判读 —— 阶段74 实验 A（判别力稀释）与实验 B（容量竞争）的统计。

🔴 判据在见到任何数字之前写死（本文件先于探针产物落盘）：

**实验 A —— 机制 1「mean pooling 的判别力随节点数下降」**

对比 `base_on_base`（N≈43）与 `base_on_gate1`（N≈51）——**同一份权重**，只有图变了。
统计量 = `gnn_feat` 在**场景之间**的总方差 `tr(Cov)`（128 维方差之和）。

* 若 `Var[pool] ~ 1/N` 严格成立且节点特征独立同分布，比值应 ≈ 43/51 = **0.843**（降 15.7%）。
* 真实节点特征相关 ⇒ 效应会更弱。**判据：**
  * 比值 **< 0.95** ⇒ 机制 1 **成立**（判别力确实被稀释）
  * 比值 **∈ [0.95, 1.05]** ⇒ **未观察到**稀释，机制 1 不是主因
  * 比值 **> 1.05** ⇒ 与预测**反向**，须重新解释

对照 `base_on_gate1` vs `gate1_on_gate1` ⇒ 重训是否补偿了稀释。

**实验 B —— 机制 3「容量竞争：交互只在高密度场景有用」**

按**运动车数量**分层（= 基线图里的 agent 数，因为基线已删静止车）比较逐场景 ADE。

* 若机制 3 成立 ⇒ 闸① 的改善（若有）应**集中在高密度层**，低密度层应持平或变差。
* **判据：** 高密度层（运动车 ≥4）的 ΔADE 与低密度层（≤1）的 ΔADE **符号相反**，
  且高密度层的配对 CI 排除 0 ⇒ 机制 3 成立；否则未观察到。

🔴 所有配对比较用 **bootstrap CI**，n 报 token 数；ADE 分布右偏 ⇒ **点估计用中位数**，
并同报均值（第十八条：两种统计量同报）。

⚠️ **自检**：`base_on_base` 的**均值** ADE 对照 B_seed42 ckpt 内嵌的 `val_ade` **2.1180**
（§8 per-seed 42 = 2.118；此前锚的 2.136 是 seed1 的值 / 三 seed 均值 2.135，**已订正**）。

🔴 **但这个自检必定不过，且不是脚本 bug —— 事前写死于阶段75 §8（见任何数字之前）**：
内嵌 2.1180 是训练时 **batch=32** 的读数，而本脚本逐样本 **batch=1**；
且 `ablation_2x2_f1/dev/B_seed42/models/checkpoint_best.pt` **不含 BN running stats**
（实测 `running_*` 键为空），`bn_calibration.py:285` 只在 ckpt 已有 running stats 时才换 BN
⇒ 本脚本跑的是 **batch-stat 模式的未校准模型**，正是 **D0#13** 那条断层
（开环 batch=32 ADE 2.194 vs planner batch=1 **4.495**）。

⇒ **判读时按此处理**：

* **实验 A 不受影响** —— 统计量是 `tr(Cov[gnn_feat])` 的**同权重两格比值**，
  BN 只在 ego 分支（`ego_history_encoder.1`），两格施加完全相同 ⇒ 比值不变。
* **实验 B 受影响** —— 用 ADE 分层配对，判读与写作**必须标明「未校准 batch=1 口径」**，
  **不得**与 §8 的 batch=32 读数并列。

若偏差**没有**出现（即 ADE 确实 ≈2.118），反而说明上述机制理解有误，
**先查脚本再解释数据**（第七十条同源纪律）。

用法::

    python3 scripts/analysis/branch_feature_readout.py --dir <abs probe dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

RATIO_LO, RATIO_HI = 0.95, 1.05          # 实验 A 判据
DENSE_MIN, SPARSE_MAX = 4, 1             # 实验 B 分层门槛
N_BOOT, BOOT_SEED = 10000, 20260803
# B_seed42 ckpt 内嵌 val_ade（batch=32 训练口径）。2.136 是 seed1 的值，已订正为 seed42 自身值。
RECORDED_VAL_ADE = 2.1180
# 本脚本是 batch=1 且 ckpt 无 BN running stats ⇒ 预期显著偏高（阶段75 §8 事前写死），非脚本 bug。
SELFCHECK_EXPECTED_TO_FAIL = True


def _boot_ci(x: np.ndarray, stat, n_boot: int, rng: np.random.Generator) -> tuple:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _load(d: Path, name: str) -> Optional[dict]:
    f = d / f"{name}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    rng = np.random.default_rng(BOOT_SEED)
    cells = {n: _load(a.dir, n) for n in ("base_on_base", "base_on_gate1", "gate1_on_gate1")}
    missing = [k for k, v in cells.items() if v is None]
    if missing:
        print(f"🔴 缺格：{missing} ⇒ 无法判读")
        return 1

    out: dict = {"criteria": {"ratio_lo": RATIO_LO, "ratio_hi": RATIO_HI,
                              "dense_min": DENSE_MIN, "sparse_max": SPARSE_MAX,
                              "n_boot": N_BOOT, "boot_seed": BOOT_SEED}}

    # ── 自检 ────────────────────────────────────────────────────────
    bb = cells["base_on_base"]
    ade_mean = float(np.nanmean(bb["ade"]))
    dev = abs(ade_mean - RECORDED_VAL_ADE) / RECORDED_VAL_ADE
    out["selfcheck"] = {"ade_mean": ade_mean, "recorded": RECORDED_VAL_ADE,
                        "rel_dev": dev, "pass": bool(dev <= 0.05),
                        "batch_protocol": "batch=1, BN 未校准（ckpt 无 running stats）",
                        "recorded_protocol": "batch=32, 训练时写入 ckpt",
                        "expected_to_fail": SELFCHECK_EXPECTED_TO_FAIL,
                        "note": "偏差属已知 D0#13 batch 口径断层，非脚本 bug；"
                                "实验 A（同权重比值）不受影响，实验 B（ADE 分层）须标 batch=1 未校准口径"}
    print("=== 自检：ADE 口径 ===")
    print(f"  base_on_base 均值 ADE = {ade_mean:.4f}（batch=1，BN 未校准）"
          f"｜ckpt 内嵌 {RECORDED_VAL_ADE}（batch=32）｜相对偏差 {dev*100:.2f}%")
    if dev <= 0.05:
        print("  ✅ 与 batch=32 读数一致 —— 🔴 但这与阶段75 §8 的事前预测相反，先查脚本再解释数据")
    else:
        print("  ⚠️ 偏差 >5%：属**事前已预测**的 D0#13 batch 口径断层，非脚本 bug")
        print("     ⇒ 实验 A（tr(Cov) 同权重两格比值）不受影响；"
              "实验 B（ADE 分层配对）判读与写作必须标明「未校准 batch=1 口径」")

    # ── 实验 A：判别力（tr(Cov) of gnn_feat across scenes）────────────
    print("\n=== 实验 A：判别力稀释（机制 1）===")
    tr = {k: float(v["gnn"].var(axis=0).sum()) for k, v in cells.items()}
    nnode = {k: float(np.median(v["n_node"])) for k, v in cells.items()}
    ratio = tr["base_on_gate1"] / tr["base_on_base"]
    pred = nnode["base_on_base"] / nnode["base_on_gate1"]
    verdict = ("成立（判别力被稀释）" if ratio < RATIO_LO
               else "未观察到" if ratio <= RATIO_HI else "反向，须重新解释")
    for k in cells:
        print(f"  {k:16s} tr(Cov[gnn]) = {tr[k]:10.4f}   节点中位数 {nnode[k]:.0f}")
    print(f"  比值 base_on_gate1 / base_on_base = {ratio:.4f}"
          f"   （1/N 预测 {pred:.4f}）⇒ **{verdict}**")
    print(f"  重训补偿：gate1_on_gate1 / base_on_gate1 = "
          f"{tr['gate1_on_gate1']/tr['base_on_gate1']:.4f}")
    out["exp_A"] = {"trace_cov": tr, "median_nodes": nnode, "ratio": ratio,
                    "ratio_pred_1overN": pred, "verdict": verdict,
                    "retrain_compensation": tr["gate1_on_gate1"] / tr["base_on_gate1"]}

    # ── 实验 B：按运动车数分层的 ADE（机制 3）────────────────────────
    print("\n=== 实验 B：按运动车数分层（机制 3）===")
    # 运动车数 = 基线图里的 agent 数（基线已删静止车）；按 token 对齐
    tok2mov = {t: int(n) for t, n in zip(bb["tokens"], bb["n_traj"])}
    gg = cells["gate1_on_gate1"]
    common = [i for i, t in enumerate(gg["tokens"]) if t in tok2mov]
    jb = {t: i for i, t in enumerate(bb["tokens"])}
    pairs = [(jb[gg["tokens"][i]], i) for i in common]
    mov = np.array([tok2mov[gg["tokens"][i]] for _, i in pairs])
    ade_b = np.array([bb["ade"][j] for j, _ in pairs], dtype=float)
    ade_g = np.array([gg["ade"][i] for _, i in pairs], dtype=float)
    ok = ~(np.isnan(ade_b) | np.isnan(ade_g))
    mov, ade_b, ade_g = mov[ok], ade_b[ok], ade_g[ok]
    print(f"  配对 token = {len(mov)}（运动车中位数 {np.median(mov):.0f}）")

    strata = {"sparse(<=%d)" % SPARSE_MAX: mov <= SPARSE_MAX,
              "mid": (mov > SPARSE_MAX) & (mov < DENSE_MIN),
              "dense(>=%d)" % DENSE_MIN: mov >= DENSE_MIN}
    out["exp_B"] = {"n_paired": int(len(mov))}
    signs = {}
    for name, m in strata.items():
        if m.sum() < 30:
            print(f"  {name:14s} n={m.sum():4d}  样本太少，跳过")
            continue
        d = ade_g[m] - ade_b[m]                       # >0 = 闸① 更差
        med, mean = float(np.median(d)), float(np.mean(d))
        lo, hi = _boot_ci(d, np.median, N_BOOT, rng)
        excl = lo > 0 or hi < 0
        signs[name] = np.sign(med)
        print(f"  {name:14s} n={m.sum():4d}  ΔADE 中位 {med:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  均值 {mean:+.4f}  {'CI排除0' if excl else 'CI含0'}")
        out["exp_B"][name] = {"n": int(m.sum()), "median": med, "mean": mean,
                              "ci95": [lo, hi], "ci_excludes_zero": bool(excl)}
    sp, dn = f"sparse(<={SPARSE_MAX})", f"dense(>={DENSE_MIN})"
    if sp in signs and dn in signs:
        opposite = signs[sp] * signs[dn] < 0
        dense_sig = out["exp_B"].get(dn, {}).get("ci_excludes_zero", False)
        vb = "成立" if (opposite and dense_sig) else "未观察到"
        print(f"  ⇒ 稀疏层与稠密层符号{'相反' if opposite else '相同'}"
              f"，稠密层 CI {'排除' if dense_sig else '含'} 0 ⇒ **机制 3 {vb}**")
        out["exp_B"]["verdict"] = vb

    dst = a.out or (a.dir / "readout.json")
    json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
