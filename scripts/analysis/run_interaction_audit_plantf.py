"""G18 —— **PlanTF 侧**受控干预审计（干预协议 v1，两侧共用同一份协议）。

这个脚本回答什么
----------------
C1 那条腿要证明的是「交互机制**是否真的被模型使用**」。MAGNET 侧（G19）已出读数，
但**单一架构上的读数无法区分「这个模型的毛病」与「这类模型的通病」**。
PlanTF 是无图 / 多模态 / 80 步的另一种架构 ⇒ 同一协议在它身上跑一遍，
才能说清协议本身可移植，以及 MAGNET 的读数有多少是架构特有的。

⚠️ 写作硬约束（协议 §10）：复刻的是**架构与训练配置、不是发布权重**
（官方 ckpt 三链接全 404，`natten` 由等价实现替代，且我方没有 nuPlan test split）
⇒ **不得引用 PlanTF 的 Test14 分数作为我方数字，也不得声称复现了 PlanTF。**

两侧实现对照（协议 §9，逐行对齐）
--------------------------------
| 协议项 | MAGNET | 本脚本（PlanTF） |
|---|---|---|
| 实体池 agent | `SUBGRAPH_ID_TO_TYPE=='agent'` 且非 ego | `agent` 张量 index `1..A-1` 且 `valid_mask.any(-1)` |
| 实体池 map | `SUBGRAPH_ID_TO_TYPE=='lane'` | `map` polygon 且 `valid_mask.any(-1)` |
| `remove` | 删节点 + 关联边 + 重编号 | `valid_mask[:, i, :] = False` |
| `zero` | `gnn_x[idx] = 0` | 该 agent/polygon 的全部特征张量置零 |
| horizon | 原生 60 步 | 原生 80 步，**一律截到前 60 步** |
| 塌陷探针 | `output_head` 末个 ReLU 恒零 | `trajectory_decoder.loc[2]`（ReLU）恒零 |
| 剂量口径 | `audit_dose_contract.DOSES` | **同一个模块**（见该文件说明） |

ego 必须排除（协议 §5）：PlanTF 的 ego 是 `agent` 张量 **index 0**
（`trajectory_decoder(x[:, 0])`），干预只作用于 `1..A-1`。

多模态口径（协议 §4.3，PlanTF 专有）
-----------------------------------
`num_modes=6`，推理取 `probability.argmax`。干预可能引起 **best_mode 跳变**，
产生巨大位移，但那**不是连续敏感度**。⇒ 三个量必须同时报：

1. `mean_disp_fixed_mode` —— 固定 **baseline 的 best_mode**（**primary**，与 MAGNET 定义可比）；
2. `mode_switch` —— 干预后 argmax 是否改变（单独一列，跨 token 汇总成跳变率）；
3. `mean_disp_free_mode` —— 自由 argmax（secondary，会被跳变污染，只作参考）。

为什么干预掩码是干净的（已核实，不是假设）
------------------------------------------
`agent_encoder.forward` 把 `agent_feature` reshape 成 `[bs*A, T, C]` **逐 agent 独立**编码，
被 mask 的 agent 走不进 `history_encoder`（`x_agent[valid_agent_mask] = ...`），
且 `key_padding_mask = ~valid_mask.any(-1)` 会在每个 encoder block 的注意力里把它排除
⇒ 它对 ego 的 `x[:, 0]` 没有任何通路。map 侧同理。
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[2]
for _p in (str(_GNN_ROOT), str(_THIS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ⚠️ 用**包路径**（`_GNN_ROOT` 已在上面进了 sys.path）。两侧必须用同一条路径 ——
#   裸 `audit_dose_contract` 与 `scripts.analysis.audit_dose_contract` 会把同一个文件
#   加载成**两个不同的模块对象**，`Dose` 因而有两个不相等的类，`isinstance` 会静默失败。
#   `scripts/` 与 `scripts/analysis/` 均无 `__init__.py`（namespace package），
#   共享模块本身零依赖 ⇒ `plantf` env 走包路径已实测可行。
from scripts.analysis.audit_dose_contract import DOSES, Dose, resolve_dose  # noqa: E402

logger = logging.getLogger(__name__)

#: 协议 §7：三个干预 seed，同 token 内目标侧与对照侧共用同一个 rng 流。
INTERVENTION_SEEDS: Tuple[int, ...] = (1000, 1001, 1002)
#: 协议 §4.2：跨模型公共 horizon。PlanTF 原生 80 步，必须截到 60。
DEFAULT_HORIZON_STEPS = 60

TARGET_KEY, CONTROL_KEY = "agent", "map"


# ---------------------------------------------------------------- 数据装载

def load_feature(path: Path) -> Optional[Dict]:
    """读一个 PlanTF cache 的 `feature.gz`（`{'agent': {...}, 'map': {...}, ...}`）。"""
    try:
        with gzip.open(path, "rb") as fh:
            obj = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        logger.warning("读取失败 %s: %s", path, exc)
        return None
    data = obj.data if hasattr(obj, "data") else obj
    if not isinstance(data, dict) or TARGET_KEY not in data or CONTROL_KEY not in data:
        logger.warning("结构不符 %s", path)
        return None
    return data


def to_batch(data: Dict, device: torch.device) -> Dict:
    """numpy 特征 dict → batch=1 的 torch dict，**走 PlanTF 官方的 `NuplanFeature` 路径**。

    不自己搓 dtype/形状转换：训练与在线推理都走这条路，审计必须喂给模型**完全一样**的东西，
    否则读数对应的不是模型的真实行为。官方语义是「`bool` 保 bool、其余一律 float」
    （`src/utils/conversion.py:to_tensor`），`collate` 对单样本等价于 `unsqueeze(0)`。

    ⚠️ `collate` 只保留 `agent`/`map`/`current_state`/`origin`/`angle` 五个顶层键 ——
    正好是缓存里的全部键，没有信息被丢掉（2026-08-01 逐张量核对过：键/形状/dtype/数值全等）。
    """
    from src.features.nuplan_feature import NuplanFeature  # noqa: WPS433 需 planTF 在 PYTHONPATH 上

    feat = NuplanFeature.deserialize(data).to_feature_tensor()
    return NuplanFeature.collate([feat]).to_device(device).data


def deep_clone(batch: Dict) -> Dict:
    """深拷贝 batch（干预必须不污染 baseline）。"""
    return {k: ({kk: vv.clone() for kk, vv in v.items()} if isinstance(v, dict) else v.clone())
            for k, v in batch.items()}


# ---------------------------------------------------------------- 干预算子

def entity_pool(batch: Dict, key: str, exclude_ego: bool) -> np.ndarray:
    """有效实体下标。协议 §5：agent 侧必须剔除 ego（index 0）。"""
    valid = batch[key]["valid_mask"][0]                     # [A, T] / [M, P]
    idx = torch.nonzero(valid.any(-1), as_tuple=False).flatten().cpu().numpy()
    return idx[idx != 0] if exclude_ego else idx


def apply_remove(batch: Dict, key: str, drop: np.ndarray) -> Dict:
    """`remove` = 把这些实体的 `valid_mask` 整行置 False（协议 §9）。"""
    out = deep_clone(batch)
    if drop.size:
        out[key]["valid_mask"][0, torch.as_tensor(drop, dtype=torch.long)] = False
    return out


def apply_zero(batch: Dict, key: str, drop: np.ndarray) -> Dict:
    """`zero` = 保留实体（`valid_mask` 不动）但把它的全部特征张量置零。

    与 `remove` 互为稳健性对照：`zero` 保持 token 数不变（模型容量不变），
    `remove` 则同时改变有效实体数。
    """
    out = deep_clone(batch)
    if not drop.size:
        return out
    sel = torch.as_tensor(drop, dtype=torch.long)
    for name, tensor in out[key].items():
        if name == "valid_mask" or not torch.is_floating_point(tensor):
            continue
        tensor[0, sel] = 0.0
    return out


# ---------------------------------------------------------------- 前向与度量

def _last_relu(model) -> Optional[torch.nn.Module]:
    """塌陷探针挂点 = `trajectory_decoder.loc` 里最后一个 ReLU。

    该 ReLU 恒零 ⇒ 末层 `Linear(hidden, future_steps*out_channels)` 输入是零向量、
    输出退化为**纯 bias** ⇒ 该场景预测与全部输入无关（协议 §6，与 MAGNET 侧同义）。
    """
    dec = getattr(model, "trajectory_decoder", None)
    loc = getattr(dec, "loc", None) if dec is not None else None
    if loc is None:
        return None
    relus = [m for m in loc if isinstance(m, torch.nn.ReLU)]
    return relus[-1] if relus else None


@torch.no_grad()
def forward_once(model, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 `(trajectory[num_modes, T, C], probability[num_modes])`（batch=1）。"""
    out = model(batch)
    return out["trajectory"][0].detach().cpu(), out["probability"][0].detach().cpu()


@torch.no_grad()
def probe_collapsed(model, batch: Dict) -> Optional[bool]:
    """本场景是否触发输出头塌陷。`None` = 探针挂不上（结构不符预期）。"""
    relu = _last_relu(model)
    if relu is None:
        return None
    cap: Dict[str, torch.Tensor] = {}
    handle = relu.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    try:
        model(batch)
    finally:
        handle.remove()
    return None if "o" not in cap else bool(float(cap["o"].abs().max()) == 0.0)


def response(traj_base: torch.Tensor, prob_base: torch.Tensor,
             traj_interv: torch.Tensor, prob_interv: torch.Tensor,
             horizon_steps: int) -> Dict[str, float]:
    """协议 §4.1 + §4.3 的三个量。

    PlanTF 的 `trajectory_decoder` 输出**已经是位置**（不像 MAGNET 输出增量需 cumsum），
    故这里直接取 `[..., :2]` 比较，不做 cumsum。
    """
    best_base = int(prob_base.argmax())
    best_interv = int(prob_interv.argmax())

    pa = traj_base[best_base, :horizon_steps, :2]
    # ① primary：固定 baseline 的 best_mode ⇒ 与 MAGNET 单模态定义可比
    pb_fixed = traj_interv[best_base, :horizon_steps, :2]
    # ③ secondary：自由 argmax，会被跳变污染
    pb_free = traj_interv[best_interv, :horizon_steps, :2]

    disp_fixed = (pa - pb_fixed).norm(dim=1)
    disp_free = (pa - pb_free).norm(dim=1)
    return {
        "mean_disp_fixed_mode": float(disp_fixed.mean()),
        "max_disp_fixed_mode": float(disp_fixed.max()),
        "mean_disp_free_mode": float(disp_free.mean()),
        "mode_switch": bool(best_interv != best_base),      # ② 跳变率的逐 token 原料
        "base_scale": float(pa.norm(dim=1).mean()),
        "best_mode_base": best_base,
        "best_mode_interv": best_interv,
    }


def apply_ops(model, batch: Dict, key: str, drop: np.ndarray,
              traj_base: torch.Tensor, prob_base: torch.Tensor,
              horizon: int) -> Dict[str, Dict]:
    """对**同一批被选中的实体**施加 `remove` 与 `zero`，各取一次 response。

    两个算子共用同一个 `drop` 集合 ⇒ 两者之差只反映算子本身
    （实体消失 vs 实体没信息），不掺入「选中了不同实体」的随机性。
    """
    out: Dict[str, Dict] = {}
    for op, fn in (("remove", apply_remove), ("zero", apply_zero)):
        t, p = forward_once(model, fn(batch, key, drop))
        out[op] = response(traj_base, prob_base, t, p, horizon)
    return out


def interventions(model, batch: Dict, doses: Sequence[Dose], rng: np.random.Generator,
                  horizon: int) -> Dict[str, Dict]:
    """剂量网格 × {remove, zero} × {target=agent(非 ego), matched control=map polygon}。"""
    pool_t = entity_pool(batch, TARGET_KEY, exclude_ego=True)
    pool_c = entity_pool(batch, CONTROL_KEY, exclude_ego=False)
    traj_base, prob_base = forward_once(model, batch)

    out: Dict[str, Dict] = {"n_target": int(pool_t.size), "n_control": int(pool_c.size)}
    for dose in doses:
        k_t, k_c, skip = resolve_dose(dose, pool_t.size, pool_c.size, TARGET_KEY, CONTROL_KEY)
        if skip is not None:
            out[dose.name] = {"mode": dose.mode, "skipped": skip}
            continue
        pick_t = rng.choice(pool_t, size=k_t, replace=False)
        pick_c = rng.choice(pool_c, size=k_c, replace=False)
        rec: Dict = {
            "mode": dose.mode,
            "k_target": k_t, "k_control": k_c,
            # 🔴 协议 §10：每个数字都要能查到两侧**实际删除比例** ——
            #    MAGNET 侧就是在这里发现 count_sat 下 agent 删 100%、lane 只删 8.3%
            "frac_target": round(k_t / pool_t.size, 4),
            "frac_control": round(k_c / pool_c.size, 4),
        }
        for op, res in apply_ops(model, batch, TARGET_KEY, pick_t, traj_base, prob_base, horizon).items():
            rec[f"{op}_{TARGET_KEY}"] = res
        for op, res in apply_ops(model, batch, CONTROL_KEY, pick_c, traj_base, prob_base, horizon).items():
            rec[f"{op}_{CONTROL_KEY}"] = res
        out[dose.name] = rec
    return out


@torch.no_grad()
def noop_check(model, batch: Dict, horizon: int) -> float:
    """协议 §7 的 no-op 自检：`response(base, base)` 必须**恰好为 0**。

    返回实测的 `mean_disp_fixed_mode`；调用方记录全 shard 的最大值。
    非零即说明前向不确定（dropout 没关 / 非确定性 kernel），读数不可用。
    """
    t1, p1 = forward_once(model, batch)
    t2, p2 = forward_once(model, deep_clone(batch))
    return response(t1, p1, t2, p2, horizon)["mean_disp_fixed_mode"]


# ---------------------------------------------------------------- 模型装载

def load_model(ckpt_path: Path, device: torch.device):
    """从 lightning checkpoint 还原 PlanTF `PlanningModel` 并置 eval。

    ⚠️ 不自己 `PlanningModel()` + `load_state_dict` 硬凑 —— 走 lightning 模块的还原路径，
    键名前缀（`model.`）由它负责剥离，避免静默加载成随机权重（那正是 gate⑤ 冒烟 0.0 分的成因）。
    """
    # 只需要 PlanningModel —— 实测不 import LightningTrainer 也能完整加载（无注册副作用），
    # 少一个 import 就少拖一层 lightning 依赖。
    from src.models.planTF.planning_model import PlanningModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    if not stripped:
        raise RuntimeError(f"checkpoint 里没有 `model.` 前缀的权重，键样例: {list(state)[:5]}")

    model = PlanningModel()
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        # 不静默：缺键 = 加载成了部分随机权重，读数没有意义
        raise RuntimeError(f"权重不匹配 missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return model.to(device).eval()


# ---------------------------------------------------------------- 主流程

def audit_token(model, feat_path: Path, horizon: int, device: torch.device) -> Optional[Dict]:
    """单个 token：塌陷探针 + 三 seed 干预。"""
    data = load_feature(feat_path)
    if data is None:
        return None
    batch = to_batch(data, device)
    rec: Dict = {
        "token": feat_path.parent.name,
        "scenario_type": feat_path.parent.parent.name,
        "log_name": feat_path.parent.parent.parent.name,
        "collapsed_head": probe_collapsed(model, batch),
        "noop_disp": noop_check(model, batch, horizon),
        "seeds": {},
    }
    for seed in INTERVENTION_SEEDS:
        rng = np.random.default_rng(seed)
        rec["seeds"][str(seed)] = interventions(model, batch, DOSES, rng, horizon)
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G18：PlanTF 侧受控干预审计（协议 v1）")
    p.add_argument("--ckpt", required=True, help="PlanTF lightning checkpoint（绝对路径）")
    p.add_argument("--cache_dir", required=True, help="PlanTF feature cache 根目录（绝对路径）")
    p.add_argument("--tokens_file", default=None,
                   help="每行一个 token 的清单；不给则扫 cache_dir 全量")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个 token，0=不限")
    p.add_argument("--horizon_steps", type=int, default=DEFAULT_HORIZON_STEPS,
                   help="截断步数。🔴 跨模型比较必须是 60（协议 §4.2）")
    p.add_argument("--device", default="cpu", help="cpu 或 cuda:N")
    p.add_argument("--threads", type=int, default=2, help="torch 线程数（CPU 长任务必须限）")
    p.add_argument("--out", required=True, help="输出 JSON（绝对路径）")
    return p.parse_args()


def collect_features(cache_dir: Path, tokens_file: Optional[str], limit: int) -> List[Path]:
    """列出待审计的 `feature.gz`。token 清单优先，保证与 MAGNET 侧一样可分片。"""
    if tokens_file:
        wanted = {t.strip() for t in Path(tokens_file).read_text().splitlines() if t.strip()}
        files = [f for f in cache_dir.glob("*/*/*/feature.gz") if f.parent.name in wanted]
    else:
        files = sorted(cache_dir.glob("*/*/*/feature.gz"))
    return files[:limit] if limit else files


def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, a.threads))

    if a.horizon_steps != DEFAULT_HORIZON_STEPS:
        logger.warning("🔴 horizon_steps=%d ≠ 协议要求的 %d —— 该读数不可与 MAGNET 侧并列",
                       a.horizon_steps, DEFAULT_HORIZON_STEPS)

    device = torch.device(a.device)
    model = load_model(Path(a.ckpt), device)
    files = collect_features(Path(a.cache_dir), a.tokens_file, a.limit)
    logger.info("待审计 token %d 个 | ckpt=%s | device=%s | horizon=%d",
                len(files), a.ckpt, device, a.horizon_steps)

    records, skipped, t0 = [], 0, time.time()
    for i, f in enumerate(files):
        rec = audit_token(model, f, a.horizon_steps, device)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d  用时 %.1f min", i + 1, len(files), (time.time() - t0) / 60)

    if not records:
        raise SystemExit("没有任何可用 token")

    noop_max = max(r["noop_disp"] for r in records)
    n_collapsed = sum(1 for r in records if r["collapsed_head"] is True)
    out = {
        "model": "planTF",
        "ckpt": str(a.ckpt),
        "horizon_steps": a.horizon_steps,
        "seeds": list(INTERVENTION_SEEDS),
        "doses": [d.name for d in DOSES],
        "n_tokens": len(records),
        "n_skipped": skipped,
        # 协议 §7：非零即前向不确定，读数不可用
        "noop_max": noop_max,
        # 协议 §6：塌陷率是独立诊断量，primary 只在非塌陷子集上算
        "collapse_rate": round(n_collapsed / len(records), 4),
        "records": records,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    logger.info("写出 %s | n=%d skipped=%d noop_max=%.3e 塌陷率=%.4f",
                a.out, len(records), skipped, noop_max, out["collapse_rate"])


if __name__ == "__main__":
    main()
