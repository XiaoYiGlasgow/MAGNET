"""S4 —— 用 S3 的反事实边效用标签训 `EdgeScorer`（Q110 §6 的 S4 步，判据 G3）。

## 口径（起跑前定死）

===========  ====================================================================
项            选择
===========  ====================================================================
backbone      **冻结**（S2 teacher，`joint_s2_fixed/seed*`），全模型 `eval()`，
              只有 `stack.scorer` 的参数 `requires_grad=True`
              ⇒ `h` 是确定性的，G3 读数可归因到 scorer 而不是端到端梯度
标签          `U`（**不是** `I`）
损失          **逐目标** pairwise RankNet：`softplus(-(s_hi - s_lo))`，算在
              `raw_logit`（pre-sigmoid）上
噪声门        只对 `|U_a - U_b| > max(sd_a, sd_b)` 的对计损失
作用域        只在 `scorable`（车↔车）∧ `valid` ∧ 已打标（`~isnan`）的边上
===========  ====================================================================

**为什么是 within-target 排序、标签为什么用 `U`**（三条实测，非引述）：

1. `I = relu(U - c_i)`，`c_i` 是**逐目标常数**（实测 43,646 组 `ptp` = 0.0e+00）
   ⇒ within-target 排序时 `U` 与 `I` 同序，差别只在 relu 造的 ties；
   实测逐目标 `rho(U, I)` 中位 **+0.9560**。用 `U` 是把 relu 抹掉的序捡回来：
   目标全零率 4.0% → 0.48%（ego 层 13.2% → …）。
2. S3 的 `meta.json` caveat 逐字：「`I` 的**绝对值与 clamp 位置**受 placebo 抽样影响
   （`rho(I) np4-vs-np8 = 0.87`）；但 placebo 是逐目标的公共偏移 ⇒ 同一目标 i 内部
   j 的排序**不受影响**。S4 若只学 within-target 排序则不受此噪声支配。」
3. `U` 来自**逐源 j** 的 D2c 干预（切 j 与全部车辆的边）⇒ 含二阶通路，
   **不是纯 `j→i` 的边际效用**。回归它的**绝对值**到单条有向边上语义就是错的；
   「j 相对 j' 对 i 更重要」这个**序**的读法才站得住。

🔴 **绝对量级不在这里学** —— 它由 `EdgeScorer.forward` 的逐目标质量守恒归一
（`conserve_target_mass`）钉住，见那个函数的 docstring。S4 只负责序。

## 判据 G3（Q110 §6：「冻结 test 上能否复现 teacher 排序」）

本脚本每个 epoch 在 val 上出**分层** pairwise accuracy（= (Kendall τ + 1) / 2 的配对估计）。
🔴 **必须分层报** —— `U_sd` 逐层差一个量级（ego 0.0241 / moving 0.0508 / static 0.00203）。
NDCG / top-B recall / reliability diagram 走独立的 `g3_scorer_readout.py`（不占卡）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_THIS.parent), str(_GNN_ROOT), str(_DEVKIT), str(_DEVKIT / "models"),
           str(_GNN_ROOT / "scripts" / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_edge_label_dataset import STRATA, EdgeLabelGraphDataset  # noqa: E402
from magnet_joint_encoder import build_entity_index  # noqa: E402
from magnet_joint_layers import build_edge_context, build_entity_geometry  # noqa: E402
from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402

logger = logging.getLogger(__name__)

S4_VERSION = "MAGNET_S4_EDGE_SCORER_v1_20260826"


def _forward_to_tokens(model, data):
    """跑 teacher 的前半段，拿到 scorer 的三个输入 `(h, ectx, g_enc)`。

    🔴 与 `MagnetJointAdaptive.forward` 的前 6 行**逐字同构**；改那边必须同步改这里，
    否则 S4 学的是一套 `h`、S5 用的是另一套。
    """
    idx = build_entity_index(data)
    geom = build_entity_geometry(data.x, idx)
    ectx = build_edge_context(data.gnn_edge_index, idx, geom)
    n_graphs = int(idx.entity_batch.max().item()) + 1
    h = model.encoder(data.x, idx)
    h = model._fuse_ego(h, data, idx, n_graphs)          # noqa: SLF001  同项目内的既有私有方法
    g_enc = model.stack.geo_enc(ectx.g)
    return h, ectx, g_enc, idx


def _pair_within_target(dst: torch.Tensor, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """给每条边在**同一目标内**随机配一个伙伴。返回两个等长下标（已去掉自配）。

    O(E) 而不是 O(Σ d_i²)：按 `dst` 排序后每条边只抽一个组内偏移。
    每个 epoch 重抽 ⇒ 多个 epoch 累计覆盖到组内的大部分对。
    """
    order = torch.argsort(dst)
    _, counts = torch.unique_consecutive(dst[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    grp_start = torch.repeat_interleave(starts, counts)
    grp_size = torch.repeat_interleave(counts, counts)
    off = (torch.rand(dst.numel(), generator=gen, device=dst.device) * grp_size).long()
    off = torch.minimum(off, grp_size - 1)
    a = order
    b = order[grp_start + off]
    keep = a != b
    return a[keep], b[keep]


def _ranknet_terms(s: torch.Tensor, U: torch.Tensor, sd: torch.Tensor,
                   a: torch.Tensor, b: torch.Tensor):
    """套噪声门、定 hi/lo，返回 `(loss_per_pair, correct_per_pair, kept_mask)`。"""
    du = U[a] - U[b]
    noise = torch.maximum(sd[a], sd[b])
    keep = du.abs() > noise                     # 🔴 teacher 自己都分不清的对，不学
    hi = torch.where(du > 0, a, b)
    lo = torch.where(du > 0, b, a)
    diff = s[hi] - s[lo]
    return F.softplus(-diff)[keep], (diff > 0)[keep], keep


@torch.no_grad()
def evaluate(model, loader, device, gen, limit: int = 0) -> Dict[str, float]:
    """val 上的**分层** pairwise accuracy + 噪声门保留率。"""
    model.eval()
    n_ok = {s: 0.0 for s in STRATA}
    n_tot = {s: 0.0 for s in STRATA}
    n_pair_raw = n_pair_kept = 0
    for bi, data in enumerate(loader):
        if limit and bi >= limit:
            break
        data = data.to(device)
        h, ectx, g_enc, _ = _forward_to_tokens(model, data)
        s = model.stack.scorer.raw_logit(h, ectx, g_enc)
        m = (~torch.isnan(data.s4_U)) & ectx.scorable & ectx.valid
        if int(m.sum()) < 2:
            continue
        sel = torch.nonzero(m, as_tuple=True)[0]
        a, b = _pair_within_target(ectx.edge_index[1][sel], gen)
        a, b = sel[a], sel[b]
        n_pair_raw += a.numel()
        _, correct, keep = _ranknet_terms(s, data.s4_U, data.s4_Usd, a, b)
        n_pair_kept += int(keep.sum())
        strat = data.s4_stratum[a][keep]
        for k, name in enumerate(STRATA):
            sk = strat == k
            n_ok[name] += float(correct[sk].sum())
            n_tot[name] += float(sk.sum())
    out = {f"pairacc_{s}": (n_ok[s] / n_tot[s] if n_tot[s] else float("nan")) for s in STRATA}
    out["pairacc_all"] = (sum(n_ok.values()) / sum(n_tot.values())) if sum(n_tot.values()) else float("nan")
    out["n_pairs_kept"] = float(n_pair_kept)
    out["gate_keep_rate"] = float(n_pair_kept) / max(n_pair_raw, 1)
    for s in STRATA:
        out[f"n_pairs_{s}"] = n_tot[s]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="S4：训 EdgeScorer 拟合 S3 边效用排序")
    p.add_argument("--teacher", required=True, help="冻结的 S2 teacher checkpoint")
    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root", required=True)
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit_batches", type=int, default=0, help=">0 时冒烟用")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    model = load_model_for_inference(Path(a.teacher), device=dev).to(dev)
    model.eval()                                  # 🔴 冻结 teacher ⇒ h 确定性，dropout 关
    for prm in model.parameters():
        prm.requires_grad_(False)
    for prm in model.stack.scorer.parameters():
        prm.requires_grad_(True)
    n_train = sum(p_.numel() for p_ in model.stack.scorer.parameters())
    logger.info("scorer 可训参数 %d｜其余全部冻结（teacher=%s）", n_train, a.teacher)

    tr = DataLoader(EdgeLabelGraphDataset(a.train_root), batch_size=a.batch_size,
                    shuffle=True, follow_batch=["gnn_x"])
    va = DataLoader(EdgeLabelGraphDataset(a.val_root), batch_size=a.batch_size,
                    shuffle=False, follow_batch=["gnn_x"])
    opt = torch.optim.AdamW(model.stack.scorer.parameters(), lr=a.lr, weight_decay=1e-4)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0, tot, nb, kept, raw = time.time(), 0.0, 0, 0, 0
        for i, data in enumerate(tr):
            if a.limit_batches and i >= a.limit_batches:
                break
            data = data.to(dev)
            h, ectx, g_enc, _ = _forward_to_tokens(model, data)
            s = model.stack.scorer.raw_logit(h, ectx, g_enc)
            m = (~torch.isnan(data.s4_U)) & ectx.scorable & ectx.valid
            if int(m.sum()) < 2:
                continue
            sel = torch.nonzero(m, as_tuple=True)[0]
            pa, pb = _pair_within_target(ectx.edge_index[1][sel], gen)
            pa, pb = sel[pa], sel[pb]
            raw += pa.numel()
            loss_pp, _, keep = _ranknet_terms(s, data.s4_U, data.s4_Usd, pa, pb)
            kept += int(keep.sum())
            if loss_pp.numel() == 0:
                continue
            # 🔴 **分层 macro-average**，不是对全部对取平均。实测：噪声门滤掉 79% 的对，
            #    但逐层占比几乎不动（3.7/7.1/89.2% → 4.5/8.6/86.9%）⇒ micro-average 下
            #    87% 的梯度来自 static_or_truncated 层，而那层 `U` p95 仅 0.0068（≈3× 噪声）。
            #    M-13 已证**运动周车侧才是拓扑有负载的地方** ⇒ 让 static 层主导损失，就是
            #    阶段316「回归静止车附近的零」那个坑的排序版。
            #    权重 1/K 由层数定，不是调出来的 ⇒ 这是 Q109 §5.3 空槽 `ω_ij` 的**无超参**填法。
            strat_k = data.s4_stratum[pa][keep]
            terms = [loss_pp[strat_k == k].mean() for k in range(len(STRATA))
                     if bool((strat_k == k).any())]
            loss = torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        met = evaluate(model, va, dev, gen, limit=a.limit_batches)
        met.update({"epoch": ep, "train_loss": tot / max(nb, 1), "sec": time.time() - t0,
                    "train_gate_keep_rate": kept / max(raw, 1)})
        hist.append(met)
        logger.info("ep %02d｜loss %.4f｜val pairacc all %.4f (ego %.4f / moving %.4f / static %.4f)"
                    "｜噪声门保留 train %.1f%% val %.1f%%｜%.0f s",
                    ep, met["train_loss"], met["pairacc_all"], met["pairacc_ego"],
                    met["pairacc_moving_agent"], met["pairacc_static_or_truncated"],
                    100 * met["train_gate_keep_rate"], 100 * met["gate_keep_rate"], met["sec"])
        if met["pairacc_all"] > best:
            best = met["pairacc_all"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "metrics": met},
                       out / "scorer_best.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=1, ensure_ascii=False), encoding="utf-8")

    (out / "s4_meta.json").write_text(json.dumps({
        "version": S4_VERSION, "teacher": str(a.teacher), "seed": a.seed, "lr": a.lr,
        "epochs": a.epochs, "batch_size": a.batch_size,
        "label": "U", "loss": "within-target pairwise RankNet on raw_logit",
        "reduction": "分层 macro-average（每层先各自取均值再等权平均；Q109 §5.3 的 ω_ij 无超参填法）",
        "noise_gate": "|U_a - U_b| > max(sd_a, sd_b)",
        "gate_keep_rate_smoke": "val 实测 21.1%，逐层占比几乎不变 ⇒ 噪声门**不能**替代分层加权",
        "scope": "scorable(veh-veh) & valid & labelled",
        "normalization": "conserve_target_mass (在 EdgeScorer.forward 内，S4 不学绝对量级)",
        "best_pairacc_all": best,
        "caveats": [
            "🔴 pairacc 是**噪声门之后**的读数 —— 分母不是全部对，是 teacher 能分清的那些对",
            "🔴 必须分层引用：U_sd 逐层差一个量级（ego 0.0241 / moving 0.0508 / static 0.00203）",
            "🔴 U 含二阶通路（D2c 逐源干预），不是纯 j→i 的边际效用",
            "🔴 backbone 冻结 ⇒ 本读数回答「序学不学得出来」，不回答「用它预测有没有用」（那是 S5/G5）"],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("✅ S4 完成｜best val pairacc(all) %.4f｜产物 %s", best, out)


if __name__ == "__main__":
    main()
