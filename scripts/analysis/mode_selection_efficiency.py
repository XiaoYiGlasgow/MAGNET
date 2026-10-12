#!/usr/bin/env python3
"""Q84 —— MAGNET 多模态的**选模效率 η** 与逐模态统计（2026-08-05）。

## 为什么需要这个脚本（08-05 查出的两个口径缺口）

比较「MAGNET 与 PlanTF 谁的选模更好」时，之前只有 `gap = top1ADE / minADE` 一个量。
它有**两个独立的缺陷**：

1. **跨 horizon 不可比** —— MAGNET 6 s 原生、PlanTF 8 s 原生。
   （已由 `s3_truncated_ade_6s.py` 的 Q84 扩展在 PlanTF 侧补上 6 s 的 oracle。）
2. 🔴 **`gap` 混淆两件事** —— 低 gap 可能是「选得准」，也可能是
   「候选彼此更像、挑错代价小」。极端情形 K 条完全重合时 `gap ≡ 1.0`，
   看着完美其实选模毫无意义。**这两者的下一步完全相反**，不能混。

## 选模效率 η

::

    η = (ADE_mean_modes − ADE_top1) / (ADE_mean_modes − ADE_oracle)

* `ADE_mean_modes` = K 条候选误差的平均 ＝ **随机挑一条**的期望误差（选模的零信息基线）；
* `ADE_top1` = 按置信度 argmax 实际挑的那条 ＝ 部署时执行的那条；
* `ADE_oracle` = K 条里最好的 ＝ 选模的上限。

η = 1 ⇒ 选模完美；η = 0 ⇒ 等同随机。
分子分母同为「相对随机基线的改进量」⇒ **候选离散度与绝对尺度都被约掉**，可跨模型直接比。

🔴 **用聚合值之比**（先各自在 val 上取均值再相除），不是逐样本 η 的均值 ——
逐样本分母在 K 条重合时趋 0 会炸。两种算法的差异属第八十三条，此处显式声明选择。

## 口径对齐

逐模态误差的定义**逐字抄自** `magnet_multimodal.min_ade_fde`：
轨迹是**增量**，先 `cumsum` 成位置再算 L2（`pred_pos = cumsum(trajs.view(B,K,60,2), dim=2)`）。
`hit_rate` 同样用 **ADE** 定 best（与 `MultiModalWTALoss` 的 `best_k` 一致），不用 FDE。

**自校验**：本脚本算出的 `top1_ade` / `min_ade` / `hit_rate` 必须与该 ckpt 的
`loss_record.json` 末 epoch 值吻合（同一 val 集、同一定义）。不吻合就是取模态/取 target 错了，
**必须先修，不得出数**。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/mode_selection_efficiency.py --cell q77 --seeds 42 1 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
sys.path.insert(0, str(GNN))
sys.path.insert(0, str(GNN / "scripts" / "training"))
sys.path.insert(0, str(GNN / "src" / "shared" / "utils"))

VAL_ROOT = GNN / "data" / "processed_features_scaling_L1_logsplit" / "val"
NUM_FUTURE_POSES = 60          # 与 train_magnet_imitation.py:52 同源

CELLS = {
    "q77": GNN / "experiments" / "q77_multimodal" / "wta_k6",
    "soft": GNN / "experiments" / "q77_multimodal" / "soft_k6",
}


@torch.no_grad()
def eval_modes(ckpt: Path, loader, device: torch.device) -> Dict[str, float]:
    """一遍 val，算逐模态统计。返回 top1 / oracle / meanmode / hit / gap / η。"""
    from magnet_multimodal import MagnetImitationMultiModal        # noqa: WPS433
    from train_magnet_imitation import move_batch_to_device        # noqa: WPS433

    model = MagnetImitationMultiModal(num_modes=6).to(device)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)      # strict：架构不符即抛错，不静默部分加载
    model.eval()

    tot = {"n": 0, "top1": 0.0, "orc": 0.0, "mean": 0.0, "hit": 0.0,
           "top1_fde": 0.0, "orc_fde": 0.0}
    for data in loader:
        data = move_batch_to_device(data, device)
        trajs, logits = model.forward_modes(data)      # [B,K,120] / [B,K]
        bs = int(data.num_graphs)
        if trajs.shape[0] != bs:
            raise RuntimeError(f"多模态输出 batch 维 {trajs.shape[0]} ≠ {bs}，拒绝猜")
        y = data.y.reshape(bs, -1).to(device)          # [B,120]

        # 🔴 逐字对齐 magnet_multimodal.min_ade_fde：增量 -> cumsum -> L2
        k = trajs.shape[1]
        pred_pos = torch.cumsum(trajs.view(bs, k, NUM_FUTURE_POSES, 2), dim=2)
        true_pos = torch.cumsum(y.view(bs, 1, NUM_FUTURE_POSES, 2), dim=2)
        err = torch.norm(pred_pos - true_pos, dim=3)   # [B,K,60]
        ade_k = err.mean(dim=2)                        # [B,K]
        fde_k = err[:, :, -1]

        top1 = logits.argmax(dim=1)
        idx = torch.arange(bs, device=trajs.device)
        tot["n"] += bs
        tot["top1"] += float(ade_k[idx, top1].sum())
        tot["orc"] += float(ade_k.min(dim=1).values.sum())
        tot["mean"] += float(ade_k.mean(dim=1).sum())
        tot["hit"] += float((top1 == ade_k.argmin(dim=1)).float().sum())
        tot["top1_fde"] += float(fde_k[idx, top1].sum())
        tot["orc_fde"] += float(fde_k.min(dim=1).values.sum())

    n = tot["n"]
    if n == 0:
        raise RuntimeError("val loader 一个样本都没给出")
    top1, orc, mean = tot["top1"] / n, tot["orc"] / n, tot["mean"] / n
    den = mean - orc
    return {
        "n_samples": n,
        "top1_ADE": top1, "oracle_ADE": orc, "meanmode_ADE": mean,
        "top1_FDE": tot["top1_fde"] / n, "oracle_FDE": tot["orc_fde"] / n,
        "hit_rate": tot["hit"] / n,
        "gap": top1 / orc,
        # η；分母为 0（K 条完全重合）返回 nan，绝不返回 1.0 冒充「完美」
        "eta": (mean - top1) / den if den > 0 else float("nan"),
    }


def build_val_loader(batch_size: int, num_workers: int):
    """建 val loader —— 逐字对齐 `train_magnet_imitation.py:274/298/314` 的建法。

    🔴 三处必须一致，否则测的不是同一个 val：
      1. `get_adjacency_config("default")` **设全局**，`GraphDataset` 据此命名 `.pt` 并选 edge builder；
      2. val 侧 `transform=None`（**绝不增广**）；
      3. `follow_batch=None` —— Q77 的 `gnn_pooling_fix=False`，与训练时同路。
    """
    from torch_geometric.loader import DataLoader                  # noqa: WPS433
    from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: WPS433

    get_adjacency_config("default")          # 设全局；顺序不可与下一行交换
    ds = GraphDataset(str(VAL_ROOT), transform=None)
    logger.info("val 样本数 = %d（期望 2400）", len(ds))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      follow_batch=None, num_workers=num_workers)


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def self_check(res: Dict[str, float], run_dir: Path) -> List[str]:
    """自校验：与该 ckpt 的 loss_record.json 末 epoch 值比对（同 val 集、同定义）。"""
    p = run_dir / "loss_record.json"
    if not p.exists():
        return [f"⚠️ {p} 不存在，跳过自校验"]
    h = json.loads(p.read_text())["history"]
    msgs = []
    for key, tag in (("top1_ADE", "val_ade"), ("oracle_ADE", "val_min_ade"),
                     ("hit_rate", "val_hit_rate")):
        if tag not in h:
            continue
        ref = h[tag][-1]
        ok = abs(res[key] - ref) < 5e-3
        msgs.append(f"{'✅' if ok else '🔴'} {key} {res[key]:.4f} vs loss_record 末 epoch {tag} {ref:.4f}")
    return msgs


def main() -> None:
    p = argparse.ArgumentParser(description="多模态选模效率 η（Q84）")
    p.add_argument("--cell", choices=sorted(CELLS), default="q77")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--ckpt_name", default="best_model_default.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    device = torch.device(a.device)
    loader = build_val_loader(a.batch_size, a.num_workers)
    root = CELLS[a.cell]

    rows, checks = [], {}
    for s in a.seeds:
        run_dir = root / f"seed{s}"
        ckpt = run_dir / "models" / a.ckpt_name
        if not ckpt.exists():
            logger.warning("跳过 seed %d：%s 不存在", s, ckpt)
            continue
        r = eval_modes(ckpt, loader, device)
        r["seed"] = s
        rows.append(r)
        checks[str(s)] = self_check(r, run_dir)
        logger.info("seed %d｜top1 %.4f｜oracle %.4f｜meanmode %.4f｜gap %.3f｜hit %.4f｜η %.4f",
                    s, r["top1_ADE"], r["oracle_ADE"], r["meanmode_ADE"],
                    r["gap"], r["hit_rate"], r["eta"])
        for m in checks[str(s)]:
            logger.info("    %s", m)

    if not rows:
        raise SystemExit("没有可读的 checkpoint")

    agg = {}
    for k in ("top1_ADE", "oracle_ADE", "meanmode_ADE", "hit_rate", "gap", "eta"):
        vals = [r[k] for r in rows]
        agg[k] = {"mean": sum(vals) / len(vals), "sd": _sd(vals),
                  "per_seed": {str(r["seed"]): r[k] for r in rows}}

    print(f"\n=== {a.cell} 逐模态统计（6 s 原生，n={rows[0]['n_samples']}/seed）===")
    print(f"  {'量':<14}{'均值':>10}{'sd':>10}   per-seed")
    for k in ("top1_ADE", "oracle_ADE", "meanmode_ADE", "hit_rate", "gap", "eta"):
        ps = "｜".join(f"s{kk}={vv:.4f}" for kk, vv in agg[k]["per_seed"].items())
        print(f"  {k:<14}{agg[k]['mean']:>10.4f}{agg[k]['sd']:>10.4f}   {ps}")
    print("\n  η = (随机选的期望 − 实际选的) / (随机选的期望 − oracle)")
    print("  η=1 选模完美｜η=0 等同随机；候选离散度与绝对尺度均已约掉，可跨模型比较")

    res = {"cell": a.cell, "val_root": str(VAL_ROOT), "num_modes": 6,
           "convention": ("逐字对齐 magnet_multimodal.min_ade_fde：增量 cumsum 成位置后算 L2；"
                          "hit 用 ADE 定 best（与 WTA 的 best_k 一致）；"
                          "η 用聚合值之比（非逐样本均值，见 docstring）"),
           "rows": rows, "aggregate": agg, "self_check": checks}
    out = Path(a.out) if a.out else root.parent / f"{a.cell}_mode_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
