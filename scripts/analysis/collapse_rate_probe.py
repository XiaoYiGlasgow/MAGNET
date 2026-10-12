"""Q105 第③步 —— **输出头塌陷率复测**（修A 的主判据），逐 token 配对。

## 为什么要新写而不是读旧 shard

08-21 的全库普查（阶段271）是**直接读已落盘的 audit shard**，零算力。修A（`MagnetHeadNorm`）
没有 shard，且旧普查各格的 token 集不完全相同 ⇒ 直接并列会混进「不同场景」的差异。

本脚本把**同一批 token 一次性载入内存**，再对每个 ckpt 逐一探针 ⇒
所有格共用同一场景集合，可做**逐 token 配对**，而不是两个独立比例相减。

## 判据

修A 的机制门（`tests/test_head_norm.py` 的 V-H2）已在数学上排除塌陷：加 LayerNorm 后
`Linear[4]` 的预激活每样本均值恒 0 ⇒ 共模门 `w̄·x` 项被消掉。
**本脚本是它在真实数据上的兑现检查** —— 单测证明的是「构造上不可能」，
这里量的是「实际有没有」。若修A 仍有非零塌陷率，说明还有第二条路径，必须先查清再进 G1。

🔴 塌陷探针挂在**实际参与前向**的输出头最后一个 ReLU 上（`_last_relu` 走 `active_head()`），
`MagnetHeadNorm` 已实现该方法，不会误挂到父类的死参数上。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent), str(_GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.g1_degeneracy_probe import (  # noqa: E402
    ProbeError,
    _cluster_bootstrap_median,
    _load_token_to_log,
)
from scripts.analysis.run_interaction_audit import (  # noqa: E402
    _load_feature_df,
    _load_model,
    _probe_collapsed_head,
)
from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402

logger = logging.getLogger("collapse_rate_probe")

PROBE_VERSION = "COLLAPSE_RATE_PROBE_v1_20260822"


def _load_scenes(data_root: Path, n_scenes: int, seed: int,
                 adjacency_config: str) -> List[Tuple[str, object]]:
    """一次性载入场景 —— 所有 ckpt 共用同一批，才能做逐 token 配对。"""
    files = sorted(data_root.glob("**/features_*.pkl"))
    if not files:
        raise ProbeError(f"{data_root} 下没有 features_*.pkl")
    random.Random(seed).shuffle(files)
    out: List[Tuple[str, object]] = []
    for f in files[:n_scenes]:
        df = _load_feature_df(f)
        if df is None:
            continue
        built = build_graph_data_from_feature_df(df, adjacency_config)
        data = built[0] if isinstance(built, tuple) else built
        if data is None:
            continue
        out.append((f.stem.replace("features_", ""), data))
    return out


@torch.no_grad()
def _probe_ckpt(ckpt: Path, scenes: List[Tuple[str, object]],
                gnn_pooling_fix: bool, device: torch.device) -> Dict[str, Optional[bool]]:
    model = _load_model(ckpt, gnn_pooling_fix, device)
    model.eval()
    return {tok: _probe_collapsed_head(model, data, device) for tok, data in scenes}


def _rate(flags: Dict[str, Optional[bool]]) -> Tuple[float, int, int]:
    vals = [v for v in flags.values() if v is not None]
    n = len(vals)
    k = sum(1 for v in vals if v)
    return (k / n if n else float("nan")), k, n


def _paired_delta(a: Dict[str, Optional[bool]], b: Dict[str, Optional[bool]],
                  token_to_log: Dict[str, str], n_boot: int, boot_seed: int) -> Dict:
    """逐 token 配对的塌陷率差（b − a），CI 按 log-cluster 重抽。

    统计量用**均值**（比例）而不是中位数 —— 逐 token 的值是 0/1，中位数无意义。
    """
    toks = [t for t in a if a[t] is not None and b.get(t) is not None]
    if not toks:
        return {"n": 0}
    diff = np.array([float(bool(b[t])) - float(bool(a[t])) for t in toks], dtype=float)
    logs = np.array([token_to_log.get(t, f"__unknown_{t}") for t in toks])
    uniq = np.unique(logs)
    rng = np.random.default_rng(boot_seed)
    idx = {c: np.flatnonzero(logs == c) for c in uniq}
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        stats[i] = diff[np.concatenate([idx[c] for c in pick])].mean()
    return {
        "n": len(toks), "n_logs": int(uniq.size),
        "delta_rate": float(diff.mean()),
        "ci95_log_cluster": [float(np.percentile(stats, 2.5)),
                             float(np.percentile(stats, 97.5))],
        "b_only": int(sum(1 for t in toks if b[t] and not a[t])),
        "a_only": int(sum(1 for t in toks if a[t] and not b[t])),
        "both": int(sum(1 for t in toks if a[t] and b[t])),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="输出头塌陷率复测（逐 token 配对）")
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                   help="可重复；如 fr_base_seed42=experiments/.../seed42_calibrated.pth")
    p.add_argument("--pair", action="append", default=[], metavar="BASE:TREAT",
                   help="可重复；给出要做配对差的两个 NAME，如 fr_base_seed42:headnorm_seed42")
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n_scenes", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boot_seed", type=int, default=20260822)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--gnn_pooling_fix", action="store_true", default=True)
    p.add_argument("--no_gnn_pooling_fix", dest="gnn_pooling_fix", action="store_false")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    token_to_log = _load_token_to_log()

    t0 = time.time()
    scenes = _load_scenes(args.data_root, args.n_scenes, args.seed, args.adjacency_config)
    logger.info("载入 %d 个场景（用时 %.0f s）—— 所有 ckpt 共用同一批", len(scenes), time.time() - t0)

    specs: Dict[str, Path] = {}
    for item in args.ckpt:
        if "=" not in item:
            raise ProbeError(f"--ckpt 必须是 NAME=PATH，收到 {item}")
        name, path = item.split("=", 1)
        specs[name] = Path(path)

    flags: Dict[str, Dict[str, Optional[bool]]] = {}
    rates: Dict[str, Dict] = {}
    for name, path in specs.items():
        f = _probe_ckpt(path, scenes, args.gnn_pooling_fix, device)
        flags[name] = f
        rate, k, n = _rate(f)
        rates[name] = {"collapse_rate": rate, "n_collapsed": k, "n_probed": n,
                       "ckpt": str(path)}
        logger.info("  %-22s 塌陷率 %6.2f%%  (%d/%d)", name, 100 * rate, k, n)

    pairs: Dict[str, Dict] = {}
    for item in args.pair:
        if ":" not in item:
            raise ProbeError(f"--pair 必须是 BASE:TREAT，收到 {item}")
        base, treat = item.split(":", 1)
        if base not in flags or treat not in flags:
            raise ProbeError(f"--pair 里的名字未在 --ckpt 中定义：{item}")
        d = _paired_delta(flags[base], flags[treat], token_to_log, args.n_boot, args.boot_seed)
        pairs[item] = d
        if d.get("n"):
            logger.info("  配对 %-38s Δ塌陷率 %+.4f  CI [%+.4f, %+.4f]  "
                        "(仅 treat 塌 %d / 仅 base 塌 %d / 都塌 %d)",
                        item, d["delta_rate"], d["ci95_log_cluster"][0],
                        d["ci95_log_cluster"][1], d["b_only"], d["a_only"], d["both"])

    payload = {
        "contract": PROBE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "data_root": str(args.data_root),
        "n_scenes_loaded": len(scenes),
        "sampling": {"n_requested": args.n_scenes, "seed": args.seed},
        "note": "所有 ckpt 共用同一批 token ⇒ 配对差有效；比例的 CI 按 log-cluster 重抽",
        "rates": rates,
        "pairs": pairs,
        "per_token": {name: {t: v for t, v in f.items()} for name, f in flags.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("写出 %s（总用时 %.0f s）", args.out, time.time() - t0)


if __name__ == "__main__":
    main()
