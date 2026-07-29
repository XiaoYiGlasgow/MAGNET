"""2×2 factorial 消融编排（VT-2026-04191 阶段⑬ §6）。

四组固定为：

    配置   Route            Perturbation
    B      none             off
    R      three_channel    off
    P      none             on
    RP     three_channel    on

只比较「旧模型 vs route+perturbation」无法判断提升来自哪一项，故 headline 必须来自本 2×2。

**受控变量（脚本强制，不靠自觉）**：
  - 同一 30000 token：R/RP 用的 E2 数据由 `run_e2_preprocess.py` 按 E1 的 token 精确重建；
  - 同一 480/60/60 log split：两套数据用同 seed 跑 `split_log_disjoint3.py`，
    并由 `check_split_consistency.py` 硬校验；
  - 同一 seed 集、同模型结构/optimizer/epochs/patience/checkpoint selection：
    本脚本对四组传入完全相同的训练超参，仅切 `--route_mode` 与 `--use_perturbation`；
  - test 只在方法冻结后统一评估（`--run_test` 显式开启，默认不跑）。

推荐执行顺序（§6）：
  1) `--stage smoke`   小数据单 seed，确认四组都能跑通；
  2) `--stage dev`     3 seeds 全量 train/val，用于方向判断与参数冻结；
  3) `--stage final`   冻结后 B/RP 至少 5 seeds（R/P 若要解释独立作用也需 5 seeds）。

单卡串行（当前仅 GPU0 可用）。每组跑完立即落盘，可断点续跑（已存在 loss_record.json 则跳过）。

运行示例：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/run_2x2_ablation.py \
      --stage smoke --none_root_base /abs/.../processed_features_magnet_e1_full_split3 \
      --route_root_base /abs/.../processed_features_magnet_e2_route_v2_split3 \
      --out_base /abs/.../experiments/ablation_2x2 --device cuda:0 --dry_run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_2x2_ablation")

PY = "/home/team/anaconda3/envs/nuplan/bin/python"

# 四组：(名称, route_mode, use_perturbation)
CELLS = [
    ("B",  "none",          False),
    ("R",  "three_channel", False),
    ("P",  "none",          True),
    ("RP", "three_channel", True),
]

STAGE_PRESETS = {
    # smoke：只验证四组能跑通，不看指标
    "smoke": dict(seeds=[42], epochs=2, patience=2),
    # dev：方向判断与参数冻结
    "dev":   dict(seeds=[42, 1, 2], epochs=None, patience=15),
    # final：冻结后正式跑
    "final": dict(seeds=[42, 1, 2, 3, 4], epochs=None, patience=15),
}

# ⚠️ 这些 preset 是**声明式默认值，任何调用方都不得在运行时改写本文件**。
# 需要偏离默认值时一律用 CLI 覆盖：`--epochs / --patience / --seeds`。
# 背景（阶段15 D0#2）：`scratchpad/pipeline_orchestrator.sh` 曾用 Python `write_text`
# 把 dev preset 就地改成 epochs=45/patience=12，导致「命令 + git commit」无法还原实验。
# 实际生效值一律由 `resolve_preset()` 解析并写进 `<stage>_summary.json` 的 `resolved_preset`。


def resolve_preset(stage: str, a: argparse.Namespace) -> Dict:
    """声明式 preset + CLI 覆盖 = 本次实际生效配置（唯一真相源）。"""
    preset = dict(STAGE_PRESETS[stage])
    overrides: Dict[str, object] = {}
    if a.seeds:
        preset["seeds"] = [int(s) for s in a.seeds.split(",") if s.strip()]
        overrides["seeds"] = preset["seeds"]
    if a.epochs is not None:
        preset["epochs"] = a.epochs
        overrides["epochs"] = a.epochs
    if a.patience is not None:
        preset["patience"] = a.patience
        overrides["patience"] = a.patience
    preset["_declared"] = dict(STAGE_PRESETS[stage])
    preset["_cli_overrides"] = overrides
    logger.info("stage=%s 生效配置: seeds=%s epochs=%s patience=%s%s",
                stage, preset["seeds"], preset["epochs"], preset["patience"],
                f"  (CLI 覆盖: {overrides})" if overrides else "  (全部为声明默认值)")
    return preset


def build_cmd(cell: str, route_mode: str, use_perturb: bool, seed: int,
              a: argparse.Namespace, epochs, patience, out_dir: Path) -> List[str]:
    base = Path(a.route_root_base if route_mode == "three_channel" else a.none_root_base)
    cmd = [
        PY, str(_TRAIN_DIR / "train_magnet_imitation.py"),
        "--train_root", str(base / "train"),
        "--val_root", str(base / "val"),
        "--output_dir", str(out_dir),
        "--dataset_config", a.dataset_config,
        "--seed", str(seed),
        "--device", a.device,
        "--batch_size", str(a.batch_size),
        "--lr", str(a.lr),
        "--patience", str(patience),
        "--route_mode", route_mode,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if use_perturb:
        # 扰动 seed 显式绑定实验 seed，保证同 seed 完全可复现
        cmd += ["--use_perturbation",
                "--perturb_prob", str(a.perturb_prob),
                "--max_lateral", str(a.max_lateral),
                "--max_longitudinal", str(a.max_longitudinal),
                "--max_yaw", str(a.max_yaw),
                "--perturb_seed", str(seed)]
    if getattr(a, "gnn_pooling_fix", False):
        cmd += ["--gnn_pooling_fix"]   # D0#7：默认关闭；不给此标志时命令行与旧版逐字节相同
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2×2 factorial 消融编排（B/R/P/RP）")
    p.add_argument("--stage", required=True, choices=list(STAGE_PRESETS))
    p.add_argument("--none_root_base", required=True, type=str,
                   help="route_mode=none 数据的 split3 根（含 train/val/test）")
    p.add_argument("--route_root_base", required=True, type=str,
                   help="route_mode=three_channel 数据的 split3 根")
    p.add_argument("--out_base", required=True, type=str)
    p.add_argument("--cells", default="B,R,P,RP", help="逗号分隔，子集调试用")
    p.add_argument("--dataset_config", default="default")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--perturb_prob", type=float, default=0.5)
    p.add_argument("--max_lateral", type=float, default=1.0)
    p.add_argument("--max_longitudinal", type=float, default=0.5)
    p.add_argument("--max_yaw", type=float, default=0.1)
    p.add_argument("--run_test", action="store_true",
                   help="训练后跑独立 test（**仅在方法冻结后**开启）")
    p.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="D0#7：透传给 train_magnet_imitation.py（默认关闭 = 旧行为）")
    # --- preset 覆盖（唯一合法的偏离方式；禁止运行时改写本文件，见 STAGE_PRESETS 注释）---
    p.add_argument("--epochs", type=int, default=None,
                   help="覆盖 stage preset 的 epochs（不给则用声明默认值）")
    p.add_argument("--patience", type=int, default=None,
                   help="覆盖 stage preset 的 patience")
    p.add_argument("--seeds", type=str, default=None,
                   help="覆盖 stage preset 的 seed 列表，逗号分隔，例：42,1,2")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    preset = resolve_preset(a.stage, a)
    want = [c.strip() for c in a.cells.split(",") if c.strip()]
    cells = [c for c in CELLS if c[0] in want]
    out_base = Path(a.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # 受控前提：两套数据的 split 必须完全一致
    ck = [PY, str(_TRAIN_DIR / "check_split_consistency.py"),
          "--a", str(Path(a.none_root_base) / "split3_manifest.json"),
          "--b", str(Path(a.route_root_base) / "split3_manifest.json")]
    logger.info("前置校验 split 一致性: %s", " ".join(ck))
    if not a.dry_run:
        r = subprocess.run(ck)
        if r.returncode != 0:
            logger.error("split 不一致，拒绝启动消融（否则效应与划分差异混淆）")
            sys.exit(1)

    runs: List[Dict] = []
    t_all = time.time()
    for cell, route_mode, use_perturb in cells:
        for seed in preset["seeds"]:
            tag = f"{cell}_seed{seed}"
            out_dir = out_base / a.stage / tag
            if (out_dir / "loss_record.json").exists():
                logger.info("跳过已完成: %s", tag)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd(cell, route_mode, use_perturb, seed, a,
                            preset["epochs"], preset["patience"], out_dir)
            logger.info("[%s] %s", tag, " ".join(cmd))
            if a.dry_run:
                continue
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            runs.append({"tag": tag, "cell": cell, "route_mode": route_mode,
                         "perturbation": use_perturb, "seed": seed,
                         "returncode": rc, "minutes": round((time.time() - t0) / 60, 1),
                         "output_dir": str(out_dir)})
            if rc != 0:
                logger.error("[%s] 训练失败 rc=%d", tag, rc)
                continue

            if a.run_test:
                ckpt = out_dir / "models" / "best_model_default.pth"
                test_cmd = [PY, str(_TRAIN_DIR / "eval_test_set.py"),
                            "--checkpoint", str(ckpt),
                            "--test_root", str(Path(a.route_root_base if route_mode == "three_channel"
                                                    else a.none_root_base) / "test"),
                            "--output_dir", str(out_dir / "test_eval"),
                            "--dataset_config", a.dataset_config,
                            "--device", a.device, "--tag", tag]
                logger.info("[%s] test: %s", tag, " ".join(test_cmd))
                subprocess.run(test_cmd)

    if not a.dry_run:
        summary = {"stage": a.stage,
                   "resolved_preset": {k: v for k, v in preset.items() if not k.startswith("_")},
                   "declared_preset": preset["_declared"],
                   "cli_overrides": preset["_cli_overrides"],
                   "argv": sys.argv,
                   "runs": runs,
                   "total_minutes": round((time.time() - t_all) / 60, 1),
                   "none_root_base": a.none_root_base, "route_root_base": a.route_root_base,
                   "run_test": a.run_test,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(out_base / f"{a.stage}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("阶段 %s 完成：%d 个 run，%.1f 分钟", a.stage, len(runs), summary["total_minutes"])


if __name__ == "__main__":
    main()
