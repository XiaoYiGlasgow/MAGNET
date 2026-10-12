"""外部基线 val 读数的共用部分：我方 GT、周车总体与分层、几何配对、误差与汇总。

只依赖 numpy 与标准库，plantf 与 nuplan 两个环境都能导入。
口径全部取自 MAGNET 读数的唯一定义源，不另写数：

==================  =====================================================================
项                  来源
==================  =====================================================================
自车 GT             ``cumsum(GROUND_TRUTH)``，index i ↔ t0+(i+1)·0.1 s，锚点 pos(0.1)（E8 docstring §三）
预测搬进我方口径     预测减**自己的**首点（E8 ``to_our_convention``，无 GT 泄露）
周车总体             历史非全零（``agent_labels.is_usable_agent``）且至少一帧有效未来
                     （``magnet_joint_loss``：全帧补零的周车不进损失）
逐帧有效             ``|x| + |y| > 1e-9``（``scene_strata.agent_strata``）
moving / static     ``scene_strata.agent_strata`` 的 ``is_moving``（全 60 帧有效且 6 s 位移 > 10 m）
ADE / FDE           按逐帧 mask 平均；FDE 取最后一个有效帧（``magnet_joint_loss.per_mode_errors``）
missRate 阈值        2.0 m（``magnet_joint_loss.MISS_THRESHOLD_M``）
层内平均             按实体微平均（``eval_joint_test.add`` 按 n 加权）
==================  =====================================================================
"""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
VAL_ROOT = GNN / "data/processed_features_fullret_L1_logsplit/val"
TOKEN_LOG_MANIFEST = GNN / "experiments/scaling_ladder/s4_log_manifest_from_s1.json"

N_WINDOW = 60
MISS_THRESHOLD_M = 2.0
#: 周车 t0 几何配对阈值（09-11 实测 40 token 配对率 100%、残差中位 0.140 m）
AGENT_MATCH_TOL_M = 1.0
#: 自车 GT 一致性闸（与 E8 相同）
GT_PARITY_TOL_M = 0.05


def load_module(name: str, path: Path):
    """按文件路径加载模块（PlanTF 与 GNN_Trans 都有名为 ``src`` 的包，不能走常规 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_strata = load_module("scene_strata_frozen", GNN / "src/shared/utils/scene_strata.py")


def val_tokens(limit: int = 0) -> List[str]:
    toks = sorted(p.stem.replace("features_", "") for p in VAL_ROOT.glob("features_*.pkl"))
    return toks[:limit] if limit else toks


def token_to_log(tokens: Sequence[str]) -> Dict[str, str]:
    want = set(tokens)
    out = {e["token"]: e["log_name"] for e in json.loads(TOKEN_LOG_MANIFEST.read_text()) if e["token"] in want}
    missing = want - set(out)
    if missing:
        raise KeyError(f"{len(missing)} 个 token 在 {TOKEN_LOG_MANIFEST.name} 里找不到 log，例如 {sorted(missing)[:3]}")
    return out


def load_our_sample(token: str, root: Path = VAL_ROOT) -> Dict[str, np.ndarray]:
    """我方该 token 的自车 GT（位置域 [60, 2]）与周车历史 / 未来。"""
    with open(root / f"features_{token}.pkl", "rb") as f:
        df = pickle.load(f)
    gt = np.asarray(df["GROUND_TRUTH"].iloc[0], dtype=np.float64).reshape(-1, 2)
    if gt.shape[0] != N_WINDOW or np.abs(gt[0]).max() > 1e-9:
        raise ValueError(f"{token}: GROUND_TRUTH 形状 {gt.shape} 或首帧增量非零，口径假设不成立")
    return {
        "ego_gt": np.cumsum(gt, axis=0),
        "agent_hist": np.asarray(df["AGENT_HISTORIES"].iloc[0], dtype=np.float64),
        "agent_fut": np.asarray(df["AGENT_FUTURES"].iloc[0], dtype=np.float64),
    }


def to_our_convention(traj: np.ndarray) -> np.ndarray:
    """``[..., T, D]`` → 前 60 帧 xy 减自己的首点。"""
    xy = np.asarray(traj, dtype=np.float64)[..., :N_WINDOW, :2]
    return xy - xy[..., 0:1, :]


def our_agent_population(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """MAGNET 周车读数的总体：返回槽位、t0 位置、未来 GT、逐帧 mask、is_moving。"""
    slots, p0, fut, mask, moving = [], [], [], [], []
    for s, (hist, f) in enumerate(zip(sample["agent_hist"], sample["agent_fut"])):
        if np.all(hist == 0):                      # 补零空槽
            continue
        st = _strata.agent_strata(hist, f)
        if st["n_valid_frames"] == 0:              # 全帧补零的周车不进读数
            continue
        slots.append(s)
        p0.append(hist[-1, :2])
        fut.append(f[:, :2])
        mask.append(np.abs(f[:, :2]).sum(axis=-1) > 1e-9)
        moving.append(bool(st["is_moving"]))
    n = len(slots)
    return {
        "slot": np.asarray(slots, dtype=np.int64),
        "p0": np.asarray(p0, dtype=np.float64).reshape(n, 2),
        "fut": np.asarray(fut, dtype=np.float64).reshape(n, N_WINDOW, 2),
        "mask": np.asarray(mask, dtype=bool).reshape(n, N_WINDOW),
        "is_moving": np.asarray(moving, dtype=bool),
    }


def masked_errors(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray):
    """逐实体 (ADE, FDE)：逐帧 mask 平均，FDE 取最后一个有效帧。"""
    err = np.linalg.norm(pred - tgt, axis=-1)
    m = mask.astype(np.float64)
    ade = (err * m).sum(axis=-1) / np.maximum(m.sum(axis=-1), 1.0)
    last = np.cumsum(m, axis=-1).argmax(axis=-1)
    fde = err[np.arange(err.shape[0]), last] if err.shape[0] else np.zeros(0)
    return ade, fde


def pair_agents(ours_p0: np.ndarray, theirs_p0: np.ndarray, tol: float = AGENT_MATCH_TOL_M) -> np.ndarray:
    """t0 位置最近邻一对一配对（按距离从小到大贪心）；未配上的记 -1。

    🔴 必须按几何配对、不能按下标：两边距离排序的基准点不同（09-11 实测排序不一致）。
    """
    out = np.full(len(ours_p0), -1, dtype=np.int64)
    if len(ours_p0) == 0 or len(theirs_p0) == 0:
        return out
    d = np.linalg.norm(ours_p0[:, None] - theirs_p0[None], axis=-1)
    used_t = set()
    for flat in np.argsort(d, axis=None):
        i, j = divmod(int(flat), d.shape[1])
        if d[i, j] > tol:
            break
        if out[i] >= 0 or j in used_t:
            continue
        out[i] = j
        used_t.add(j)
    return out


def _mean(x: np.ndarray) -> Optional[float]:
    return float(np.mean(x)) if len(x) else None


def summarize_ego(ade: np.ndarray, fde: np.ndarray) -> Dict[str, float]:
    return {"ego.top1_ADE": _mean(ade), "ego.top1_FDE": _mean(fde),
            "ego.top1_missRate": _mean((fde > MISS_THRESHOLD_M).astype(np.float64)), "ego.n": int(len(ade))}


def summarize_agents(ade: np.ndarray, fde: np.ndarray, is_moving: np.ndarray, paired: np.ndarray,
                     prefix: str = "") -> Dict[str, object]:
    """agent / moving / static 三层（只在已配对实体上算），附总体与配对率。"""
    out: Dict[str, object] = {f"{prefix}n_population": int(len(paired)), f"{prefix}n_paired": int(paired.sum()),
                              f"{prefix}pairing_rate": _mean(paired.astype(np.float64))}
    for name, sel in (("agent", paired), ("moving", paired & is_moving), ("static", paired & ~is_moving)):
        out[f"{prefix}{name}.top1_ADE"] = _mean(ade[sel])
        out[f"{prefix}{name}.top1_FDE"] = _mean(fde[sel])
        out[f"{prefix}{name}.n"] = int(sel.sum())
    return out


def aggregate_seeds(rows: List[Dict], keys: Sequence[str]) -> Dict[str, Dict]:
    agg = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            agg[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                      "per_seed": v}
    return agg


# =========================================================================== D4（2026-09-16 追加，只增不改）
# 冻结 test 一次性读数用。上面的函数一律不动，val 读数逐位可复现。

TEST_ROOT = GNN / "data/processed_features_fullret_L1_logsplit/test"
SPLIT_ROOTS = {"val": VAL_ROOT, "test": TEST_ROOT}
FROZEN_FILE = TEST_ROOT.parent / ".TEST_SPLIT_FROZEN"
SPLIT_MANIFEST = TEST_ROOT.parent / "split_manifest.json"


def split_tokens(split: str, limit: int = 0) -> List[str]:
    """``val`` / ``test`` 的 token 列表（按字典序，与 val_tokens 同规则）。"""
    root = SPLIT_ROOTS[split]
    toks = sorted(p.stem.replace("features_", "") for p in root.glob("features_*.pkl"))
    return toks[:limit] if limit else toks


def verify_test_firewall() -> Dict[str, object]:
    """冻结 test 的身份核对：token 数、log 数、log 清单 sha256[:16]、与 val 不相交。

    sha 算法逐字照 ``scripts/training/build_fullret_test_split.py:127``：
    ``sha256("\\n".join(sorted(test_logs)))[:16]``。任一不符直接抛错。
    """
    import hashlib

    frozen = json.loads(FROZEN_FILE.read_text())
    toks = split_tokens("test")
    logs = sorted(set(token_to_log(toks).values()))
    sha = hashlib.sha256("\n".join(logs).encode()).hexdigest()[:16]
    val_logs = set(json.loads(SPLIT_MANIFEST.read_text())["val_logs"])
    rec = {"frozen_file": str(FROZEN_FILE), "n_tokens": len(toks), "n_logs": len(logs), "logs_sha256_16": sha,
           "expected": {k: frozen[k] for k in ("n_tokens", "n_logs", "logs_sha256_16")},
           "overlap_with_val_logs": len(set(logs) & val_logs)}
    bad = []
    if len(toks) != frozen["n_tokens"]:
        bad.append(f"token 数 {len(toks)} ≠ {frozen['n_tokens']}")
    if len(logs) != frozen["n_logs"]:
        bad.append(f"log 数 {len(logs)} ≠ {frozen['n_logs']}")
    if sha != frozen["logs_sha256_16"]:
        bad.append(f"log sha {sha} ≠ {frozen['logs_sha256_16']}")
    if rec["overlap_with_val_logs"]:
        bad.append(f"与 val 共享 {rec['overlap_with_val_logs']} 条 log")
    if bad:
        raise RuntimeError("冻结 test 身份核对失败：" + "；".join(bad))
    rec["pass"] = True
    return rec


def agent_population_meta(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """与 :func:`our_agent_population` **同一筛选、同一行序**的逐车元数据（落盘后做任意分层用）。

    返回 slot、t0 速度、t0 到自车距离、is_t0_observable（只读 t0）、有效未来帧数、6 s 位移（GT，仅诊断用）。
    调用方须断言 ``slot`` 与 ``our_agent_population(sample)["slot"]`` 逐元素相等。
    """
    slots, spd, dist, obs, nvf, disp = [], [], [], [], [], []
    for s, (hist, f) in enumerate(zip(sample["agent_hist"], sample["agent_fut"])):
        if np.all(hist == 0):
            continue
        st = _strata.agent_strata(hist, f)
        if st["n_valid_frames"] == 0:
            continue
        slots.append(s)
        spd.append(st["t0_speed"])
        dist.append(st["dist_t0"])
        obs.append(bool(st["is_t0_observable"]))
        nvf.append(int(st["n_valid_frames"]))
        disp.append(np.nan if st["disp"] is None else float(st["disp"]))
    return {"slot": np.asarray(slots, dtype=np.int64), "t0_speed": np.asarray(spd, dtype=np.float64),
            "dist_t0": np.asarray(dist, dtype=np.float64), "is_t0_observable": np.asarray(obs, dtype=bool),
            "n_valid_frames": np.asarray(nvf, dtype=np.int64), "disp_gt": np.asarray(disp, dtype=np.float64)}
