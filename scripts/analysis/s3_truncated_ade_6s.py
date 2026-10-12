#!/usr/bin/env python3
"""Q65 (b) —— PlanTF 的 **6 s 截断**开环读数，用于与 MAGNET 绝对同轴比较。

背景
----
两条管线的 horizon 不同：PlanTF ``future_horizon=8`` / ``sample_interval=0.1`` ⇒ 80 步 8 s；
MAGNET ``NUM_FUTURE_POSES=60`` ⇒ 60 步 6 s（两者采样率同为 10 Hz，**前 60 步精确对应 6.0 s**）。
ADE 随时间单调增长 ⇒ PlanTF 的 8 s ADE 与 MAGNET 的 6 s ADE **绝对值不可直接比较**。

本脚本做什么
-----------
把 PlanTF 的输出截到前 60 步重算 ADE/FDE，得到与 MAGNET 同 horizon 的绝对值。

🔴 **它产出的主结果不是绝对值，而是「截断比 r」**::

    r_ADE = ADE_6s / ADE_8s      （同一 checkpoint、同一 val 集上实测）

理由：Q60/Q65 签字口径是「**末 5 epoch 均值**」，而末 5 epoch 的 checkpoint **没有全部落盘**
（lightning 只留 top-k，实测每 seed 只有 5–6 个且 epoch 不齐）⇒ 无法对末 5 epoch 逐个重算。
⇒ 正确做法是把 horizon 转换**因子化**：

    ADE_6s(末5均值) ≈ r_ADE × minADE1(末5均值)

并**实测 r 在不同 checkpoint 间是否稳定**（脚本用 ≥2 个 ckpt 检验；不稳定就不许用这个近似）。

内建自检
-------
本脚本算的 ``ADE_8s`` 与 PlanTF 自己的 ``val/minADE1`` 是**同一个定义**
（top-1-by-probability 的 80 步平均位移误差）⇒ 应与该 ckpt 对应 epoch 的 tfevents 值吻合。
不吻合就说明本脚本的取模态/取 target 方式错了，**必须先修，不得出数**。

用法::

    /home/team/anaconda3/envs/plantf/bin/python scripts/analysis/s3_truncated_ade_6s.py \\
        --tier L1 --seeds 42 1 2 --device cuda:0
    # 冒烟（只跑 4 个 batch，验管线不验数字）
    ... --tier L1 --seeds 42 --limit_batches 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
PLANTF = Path("/mnt/local/Xiao_Yi/external/planTF")
CACHE = GNN / "data" / "plantf_cache_L3"

# 时间口径（已从源码核实，勿凭记忆改）
#   PlanTF : nuplan_feature_builder.py:38-39  future_horizon=8, sample_interval=0.1 ⇒ 80 步
#   MAGNET : train_magnet_imitation.py:52     NUM_FUTURE_POSES=60 @10Hz            ⇒ 60 步
MAGNET_STEPS = 60
PLANTF_STEPS = 80


def build_val_loader(tier: str, seed: int, batch_size: int, num_workers: int):
    """用 PlanTF 自己的 hydra 配置搭 val dataloader（保证与训练期逐位同源）。"""
    sys.path.insert(0, str(PLANTF))
    from hydra import compose, initialize_config_dir
    from nuplan.planning.script.builders.worker_pool_builder import build_worker
    from src.custom_training.custom_training_builder import (  # noqa: F401
        build_training_engine, update_config_for_training)

    with initialize_config_dir(config_dir=str(PLANTF / "config"), version_base=None):
        cfg = compose(config_name="default_training", overrides=[
            "py_func=train", "+training=train_planTF",
            "worker=single_machine_thread_pool", f"worker.max_workers={num_workers}",
            "scenario_builder=nuplan", f"splitter=gnn_scaling_{tier}",
            f"cache.cache_path={CACHE}", "cache.use_cache_without_dataset=true",
            f"data_loader.params.batch_size={batch_size}",
            f"data_loader.params.num_workers={num_workers}",
            "lr=1e-3", "epochs=25", "warmup_epochs=3",
            "lightning.trainer.params.devices=1",
            f"seed={seed}", f"group={GNN}/experiments/scaling_ladder/_trunc_tmp",
            "wandb.mode=disable",
        ])
    update_config_for_training(cfg)
    worker = build_worker(cfg)
    engine = build_training_engine(cfg, worker)
    engine.datamodule.setup("fit")
    return engine.model, engine.datamodule.val_dataloader()


def load_ckpt_into(model, ckpt_path: Path):
    """把 lightning ckpt 的权重灌进模型（只取 state_dict，不恢复 trainer 状态）。"""
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{ckpt_path.name} 缺少 {len(missing)} 个权重，例：{missing[:3]}")
    if unexpected:
        logger.warning("%s 有 %d 个多余键（通常是 metrics buffer）", ckpt_path.name, len(unexpected))
    return model


@torch.no_grad()
def eval_ckpt(model, loader, device: str, limit_batches: Optional[int] = None) -> Dict[str, float]:
    """一遍 val，同时算 6 s 与 8 s 两个 horizon 的 ADE/FDE。

    🆕 **Q84 扩展（08-05）**：除 top-1-by-probability 外，另算**逐模态**统计 ——
    `oracle`（min over K）、`mean`（K 条平均，＝随机选模的期望）、`hit`（选对率）。

    为什么需要（08-05 发现的两个口径缺口）：

    1. **`gap = top1/oracle` 之前只在 8 s 上有**，而 MAGNET 是 6 s 原生
       ⇒ 两边的 gap 从来没在同一 horizon 上比过。此前把 8 s 的 gap 与 6 s 的并列是错的：
       horizon 越短各模态分岔越少 ⇒ oracle 优势越小 ⇒ gap 越小，方向可推但幅度未知。
       本函数直接给出 **6 s 上的 oracle**，gap 才可比。
    2. 🔴 **`gap` 本身混淆两件事**：低 gap 可能是「选得准」，也可能是「候选彼此更像
       （挑错代价小）」。极端情形 K 条完全重合时 gap ≡ 1.0，看着完美其实选模无意义。
       ⇒ 引入 **选模效率 η**（见下），它把候选离散度与绝对尺度都消掉。

    **选模效率**::

        η = (ADE_mean_modes − ADE_top1) / (ADE_mean_modes − ADE_oracle)

    η = 1 ⇒ 选模完美（总挑到最优那条）；η = 0 ⇒ 等同随机挑。
    分子分母同为「相对随机基线的改进量」⇒ **候选离散度与量纲都被约掉**，可跨模型直接比。
    🔴 用**聚合值之比**（先各自在 val 上取均值再相除），不是逐样本 η 的均值 ——
    逐样本分母在 K 条重合时趋 0，会炸。两种算法的差异属第八十三条，已在此声明选择。
    """
    model = model.to(device).eval()
    tot = {"n": 0, "ade60": 0.0, "fde60": 0.0, "ade80": 0.0, "fde80": 0.0,
           "orc_ade60": 0.0, "orc_ade80": 0.0, "orc_fde60": 0.0, "orc_fde80": 0.0,
           "mean_ade60": 0.0, "mean_ade80": 0.0, "hit60": 0.0, "hit80": 0.0}
    for i, batch in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        features, targets = batch[0], batch[1]
        data = features["feature"].data
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
        # nested dict（agent / map 等）逐层搬
        for k, v in list(data.items()):
            if isinstance(v, dict):
                data[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv)
                           for kk, vv in v.items()}

        out = model(data)
        traj, prob = out["trajectory"], out["probability"]     # (B,K,T,4), (B,K)
        # 🔴 与闭环 planner 同源：planning_model.py:135 best_mode = probability.argmax(-1)
        best = prob.argmax(dim=-1)
        pred = traj[torch.arange(traj.size(0), device=traj.device), best]   # (B,T,4)

        tgt = data["agent"]["target"][:, 0]                     # (B,T,3) = [x,y,heading]
        if pred.size(1) != PLANTF_STEPS or tgt.size(1) != PLANTF_STEPS:
            raise ValueError(f"步数不是 {PLANTF_STEPS}：pred {pred.size(1)}, target {tgt.size(1)}"
                             " ⇒ 时间口径与源码核实值不符，拒绝出数")

        d = torch.norm(pred[..., :2] - tgt[..., :2], dim=-1)    # (B,T)
        b = d.size(0)
        tot["n"] += b
        tot["ade80"] += d.mean(-1).sum().item()
        tot["fde80"] += d[:, -1].sum().item()
        tot["ade60"] += d[:, :MAGNET_STEPS].mean(-1).sum().item()
        tot["fde60"] += d[:, MAGNET_STEPS - 1].sum().item()

        # 🆕 Q84：逐模态误差（B,K,T）—— top-1 那几行**原封未动**，保证 r_ADE/r_FDE 逐位可复现
        d_all = torch.norm(traj[..., :2] - tgt[:, None, :, :2], dim=-1)      # (B,K,T)
        ade80_k = d_all.mean(-1)                                             # (B,K)
        ade60_k = d_all[:, :, :MAGNET_STEPS].mean(-1)                        # (B,K)
        tot["orc_ade80"] += ade80_k.min(dim=1).values.sum().item()
        tot["orc_ade60"] += ade60_k.min(dim=1).values.sum().item()
        tot["orc_fde80"] += d_all[:, :, -1].min(dim=1).values.sum().item()
        tot["orc_fde60"] += d_all[:, :, MAGNET_STEPS - 1].min(dim=1).values.sum().item()
        tot["mean_ade80"] += ade80_k.mean(dim=1).sum().item()
        tot["mean_ade60"] += ade60_k.mean(dim=1).sum().item()
        # 选对率：argmax(prob) 是否命中该 horizon 上误差最小的那条
        tot["hit80"] += (best == ade80_k.argmin(dim=1)).float().sum().item()
        tot["hit60"] += (best == ade60_k.argmin(dim=1)).float().sum().item()

    n = tot["n"]
    if n == 0:
        raise RuntimeError("val dataloader 一个样本都没给出")
    def _eta(mean_v: float, top1_v: float, orc_v: float) -> float:
        """选模效率 η；分母为 0（K 条完全重合）时返回 nan，绝不返回 1.0 冒充「完美」。"""
        den = mean_v - orc_v
        return (mean_v - top1_v) / den if den > 0 else float("nan")

    res = {
        "n_samples": n,
        "ADE_8s": tot["ade80"] / n, "FDE_8s": tot["fde80"] / n,
        "ADE_6s": tot["ade60"] / n, "FDE_6s": tot["fde60"] / n,
        "r_ADE": (tot["ade60"] / n) / (tot["ade80"] / n),
        "r_FDE": (tot["fde60"] / n) / (tot["fde80"] / n),
        # 🆕 Q84：逐模态统计
        "oracle_ADE_8s": tot["orc_ade80"] / n, "oracle_ADE_6s": tot["orc_ade60"] / n,
        "oracle_FDE_8s": tot["orc_fde80"] / n, "oracle_FDE_6s": tot["orc_fde60"] / n,
        "meanmode_ADE_8s": tot["mean_ade80"] / n, "meanmode_ADE_6s": tot["mean_ade60"] / n,
        "hit_rate_8s": tot["hit80"] / n, "hit_rate_6s": tot["hit60"] / n,
    }
    res["gap_8s"] = res["ADE_8s"] / res["oracle_ADE_8s"]
    res["gap_6s"] = res["ADE_6s"] / res["oracle_ADE_6s"]
    res["eta_8s"] = _eta(res["meanmode_ADE_8s"], res["ADE_8s"], res["oracle_ADE_8s"])
    res["eta_6s"] = _eta(res["meanmode_ADE_6s"], res["ADE_6s"], res["oracle_ADE_6s"])
    return res


def pick_ckpts(tier: str, seed: int, k: int) -> List[Path]:
    """取该 seed 最新的 k 个 checkpoint（含 last.ckpt），用于检验 r 的稳定性。"""
    root = GNN / "experiments" / "scaling_ladder" / f"s3_plantf_{tier}" / f"seed{seed}"
    cks = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise FileNotFoundError(f"{root} 下没有 ckpt")
    return cks[-k:]


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
    p = argparse.ArgumentParser(description="PlanTF 6s 截断读数（Q65 (b)）")
    p.add_argument("--tier", required=True, choices=["L1", "L2", "L3"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--n_ckpt", type=int, default=2,
                   help="每 seed 评几个 checkpoint（≥2 才能检验 r 的稳定性）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit_batches", type=int, default=None, help="冒烟用")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model, loader = build_val_loader(args.tier, args.seeds[0], args.batch_size, args.num_workers)

    rows: List[Dict[str, object]] = []
    for s in args.seeds:
        for ck in pick_ckpts(args.tier, s, args.n_ckpt):
            load_ckpt_into(model, ck)
            r = eval_ckpt(model, loader, args.device, args.limit_batches)
            r.update({"seed": s, "ckpt": ck.name})
            rows.append(r)
            logger.info("seed %d｜%s｜ADE_8s %.4f｜ADE_6s %.4f｜r_ADE %.4f (n=%d)",
                        s, ck.name, r["ADE_8s"], r["ADE_6s"], r["r_ADE"], r["n_samples"])

    r_ades = [x["r_ADE"] for x in rows]
    r_fdes = [x["r_FDE"] for x in rows]
    res = {
        "tier": args.tier, "seeds": args.seeds, "rows": rows,
        "r_ADE_mean": sum(r_ades) / len(r_ades), "r_ADE_sd": _sd(r_ades),
        "r_FDE_mean": sum(r_fdes) / len(r_fdes), "r_FDE_sd": _sd(r_fdes),
        "convention": (
            "top-1-by-probability（= 闭环 planner 执行的那条，planning_model.py:135）；"
            "前 60 步 = 6.0 s（sample_interval=0.1），与 MAGNET NUM_FUTURE_POSES=60 同 horizon"
        ),
        "caveat": (
            "模型仍在 8 s horizon 上训练，本读数只是评测截断，不是同任务比较；"
            "论文引用时必须写明这一点。"
        ),
    }

    print(f"\n=== {args.tier} 6s 截断读数 ===")
    for x in rows:
        print(f"  seed {x['seed']:>2}｜{x['ckpt']:<34}｜ADE 8s {x['ADE_8s']:.4f} → 6s {x['ADE_6s']:.4f}"
              f"｜r={x['r_ADE']:.4f}｜n={x['n_samples']}")
    print(f"\n  r_ADE = {res['r_ADE_mean']:.4f} ± {res['r_ADE_sd']:.4f}"
          f"｜r_FDE = {res['r_FDE_mean']:.4f} ± {res['r_FDE_sd']:.4f}")
    if not math.isnan(res["r_ADE_sd"]):
        if res["r_ADE_sd"] < 0.01:
            print("  ⇒ 🟢 r 在不同 checkpoint 间稳定（sd < 0.01）⇒ 可用于把末5均值换算到 6 s")
        else:
            print(f"  ⇒ 🔴 r 的 sd = {res['r_ADE_sd']:.4f} ≥ 0.01，**不稳定** ⇒ "
                  "不得用单一 r 换算末5均值；只能报 last.ckpt 的绝对值并标注口径不同")
    print(f"\n  ⚠️ {res['caveat']}")

    out = Path(args.out) if args.out else (
        GNN / "experiments" / "scaling_ladder" / f"s3_trunc6s_{args.tier}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
