"""Q25：给 calibrated checkpoint 目录补 ``train_meta.json`` sidecar。

背景
----
`run_closed_loop.sh` 的 provenance 采集会去 checkpoint 同级目录找 ``train_meta.json``
读取 ``route_mode``；找不到就退回 ``inferred``（P7 警告）。BN calibration 产出的
calibrated checkpoint 落在 ``experiments/calibration/calibrated/<run>/`` 下，
那个目录没有 sidecar，于是每一次 calibrated 闭环 run 都恒报 P7。

本脚本从**源 checkpoint 的 sidecar** 派生 calibrated 侧的 sidecar，并把 calibration
这一步本身的 provenance 一并写进去，使 ``route_mode`` 的来源从「推断」变成「记录」。

用法::

    python3 scripts/analysis/write_calibrated_train_meta.py \\
        --calibration_json experiments/calibration/calibrated/B_seed42/B_seed42_calibration.json

不覆盖已存在的文件，除非显式加 ``--force``。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DERIVED_KEYS = ("bn_calibration_derived", "source_train_meta")


def _load_json(path: Path) -> Dict[str, Any]:
    """读 JSON，失败时给出可定位的错误信息。"""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error("文件不存在: %s", path)
        raise
    except json.JSONDecodeError:
        logger.error("JSON 解析失败: %s", path)
        raise


def build_meta(calib: Dict[str, Any], src_meta: Dict[str, Any]) -> Dict[str, Any]:
    """由源 sidecar + calibration 记录派生 calibrated 侧 sidecar。

    Args:
        calib: ``<run>_calibration.json`` 的内容。
        src_meta: 源 checkpoint 目录下 ``train_meta.json`` 的内容。

    Returns:
        待写盘的 sidecar 字典。
    """
    meta = dict(src_meta)
    meta["source_train_meta"] = calib["source_checkpoint"]
    meta["bn_calibration_derived"] = {
        "note": "本 sidecar 由源 checkpoint 的 train_meta.json 派生；"
                "BN calibration 只改 running stats，不改训练配置",
        "contract": calib.get("contract"),
        "created_utc": calib.get("created_utc"),
        "source_checkpoint_sha256_file": calib.get("source_checkpoint_sha256_file"),
        "calibrated_checkpoint_sha256_file": calib.get("calibrated_checkpoint_sha256_file"),
        "calibration_manifest_hash_sha256": calib.get("calibration_manifest_hash_sha256"),
        "calibration_n_samples": calib.get("calibration_n_samples"),
        "calibration_source_split": calib.get("calibration_source_split"),
        "gnn_pooling_fix": calib.get("gnn_pooling_fix"),
    }
    return meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration_json", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 sidecar")
    args = ap.parse_args()

    calib = _load_json(args.calibration_json)
    src_ckpt = Path(calib["source_checkpoint"])
    src_meta_path = src_ckpt.parent / "train_meta.json"
    src_meta = _load_json(src_meta_path)

    out = Path(calib["calibrated_checkpoint"]).parent / "train_meta.json"
    if out.exists() and not args.force:
        logger.warning("已存在，跳过（要覆盖请加 --force）: %s", out)
        return 0

    meta = build_meta(calib, src_meta)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    logger.info("written: %s  (route_mode=%s)", out, meta.get("route_mode"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
