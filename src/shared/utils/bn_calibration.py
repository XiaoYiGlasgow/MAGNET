"""S4-b：给 `ego_history_encoder` 的 BatchNorm 补 running statistics（post-training calibration）。

背景（D0#13）
-------------
`models/transformer_predictor.py:69` 建的是 ``BatchNorm1d(64, track_running_stats=False)``，
它**没有** running_mean/running_var，因此 ``model.eval()`` 下仍然用**当前 batch** 的统计量归一化。
输入是 ``[B*20, 2]``（:208），于是 open-loop（batch=32）与闭环 planner（batch=1）**不是同一个函数**：
同一 ckpt、n=400 val 样本、CPU 实测 batch=32 ADE 2.194 / batch=1 ADE 4.495。

阶段⑳ S4-b 裁定：用固定的 **train-only** manifest 估计 running statistics 并固化进 calibrated
checkpoint，open/closed-loop 共用同一个 inference function。

本模块只做三件事，不改父类 `transformer_predictor.py` 一个字：
  1. `swap_bn_with_running_stats()` —— 原地把那一层换成同权重、带 running stats 的 BN；
  2. `calibrate_bn()` —— 只让该 BN 处于 train 模式累计统计量，其余模块保持 eval；
  3. `load_model_for_inference()` —— 原始 ckpt 与 calibrated ckpt 的统一加载入口。

⚠️ GNN 分支的 batch 依赖是**另一个**来源（D0#7）：`GNNEncoder.forward` 在 ``batch is None`` 时走
``x.mean(dim=0)``，整个 batch 共享一个向量。故 `gnn_pooling_fix=False` 的权重在 batch>1 时即使补了
BN 也**不可能**满足 1e-5 严格 invariance。Q13 已裁定 (ii)：G3 训练时就开 `--gnn_pooling_fix`，
S4-b 的正式验收跑在 fix=True 的 checkpoint 上。batch=1 时两种池化语义逐字节相同。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[3]                 # .../GNN_Trans
_DEVKIT = _GNN_ROOT.parent                   # .../nuplan-devkit
for _p in (str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT),
           str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformer_predictor import TransformerPredictor  # noqa: E402  E0 生效副本

logger = logging.getLogger(__name__)

__all__ = [
    "ARCH_BUILD_KEYS",
    "ARCH_KEYS",
    "BN_MODULE_PATH",
    "CalibrationError",
    "build_model",
    "calibrate_bn",
    "collate_fixed_batches",
    "file_sha256",
    "load_model_for_inference",
    "read_architecture",
    "state_dict_sha256",
    "swap_bn_with_running_stats",
]

#: 被 calibrate 的那一层在 `TransformerPredictor` 里的位置（写进 provenance，防止以后换层不自知）
BN_MODULE_PATH = "ego_history_encoder.1"


class CalibrationError(RuntimeError):
    """calibration 前置条件不成立 —— 必须暴露而不是静默降级。"""


def file_sha256(path: Path) -> str:
    """算文件 SHA256（provenance 用）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """算 state_dict 的内容哈希（按键名排序，逐张量取 bytes）。

    与 `file_sha256` 互补：文件哈希受 torch 序列化格式影响，内容哈希只看权重本身。
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        t = v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


#: 可直接透传给 `build_model` 的架构字段（即 sidecar 里叫什么、build_model 的形参就叫什么）。
#: 🔴 加新架构开关时**只改这里** —— 08-02 闸④ calibration 就是因为
#: `bn_calibrate_checkpoint.py` 各自维护了一份漏掉 `ego_shortcut_suppress` 的清单，
#: 建成父类 `MagnetEgoQueryFusion`，`ego_state_emb.*` 无处安放，strict load 直接崩。
ARCH_BUILD_KEYS = ("context_pooling", "context_pool_mode", "ego_query_fusion",
                   "ego_query_heads", "ego_shortcut_suppress", "ego_state_dropout")

#: 训练侧写进 `train_meta.json["architecture"]` 的字段名（改动要与 `train_magnet_imitation.py` 同步）。
#: 比 `ARCH_BUILD_KEYS` 多一个 `gnn_pooling_fix_effective` —— 它在消费侧要映射成
#: `build_model(gnn_pooling_fix=...)`，名字不同，故不能直接透传。
ARCH_KEYS = ("gnn_pooling_fix_effective",) + ARCH_BUILD_KEYS


def build_model(gnn_pooling_fix: bool = False, context_pooling: bool = False,
                context_pool_mode: str = "max", ego_query_fusion: bool = False,
                ego_query_heads: int = 4, ego_shortcut_suppress: bool = False,
                ego_state_dropout: float = 0.75) -> nn.Module:
    """按架构开关重建模型。**选择顺序必须与 `train_magnet_imitation.py` 一致**，否则推理侧建错类。

    :param gnn_pooling_fix: True 时 GNN 分支逐图池化（`MagnetImitationPooled`），
        False 时用父类 `TransformerPredictor`（整 batch 共享 GNN 向量 = 旧语义）。
    :param context_pooling: 判据0 的 context 分支 polyline 池化（`MagnetContextPooled`）。
    :param ego_query_fusion: C2 的 ego-query attention 融合（`MagnetEgoQueryFusion`）。
    :param ego_query_heads: C2 注意力头数。⚠️ **无法从权重形状反推**
        （`MultiheadAttention.in_proj_weight` 是 `[3D, D]`，与 heads 无关）⇒ 必须由调用方给对，
        给错时 `load_state_dict` 依然会成功但模型算的是另一回事。
    :param ego_shortcut_suppress: 闸④（Q50）切断 ego 捷径（`MagnetEgoShortcutSuppressed`）。
        它是 `MagnetEgoQueryFusion` 的子类 ⇒ **必须排在 ego_query_fusion 之前判断**，
        否则闸④ 的 ckpt 会被建成父类、`ego_state_emb.*` 权重无处安放而 strict 加载报错。
    :param ego_state_dropout: 闸④ 的 state dropout 概率。⚠️ 与 heads 同理**无法从权重形状反推**
        （它只影响训练期采样，不改任何张量形状），推理时不生效但必须给出以保持构造签名一致。
    """
    if ego_shortcut_suppress:
        from magnet_ego_shortcut import MagnetEgoShortcutSuppressed  # noqa: WPS433
        return MagnetEgoShortcutSuppressed(context_pooling=context_pooling,
                                           context_pool_mode=context_pool_mode,
                                           ego_query_fusion=ego_query_fusion,
                                           ego_query_heads=ego_query_heads,
                                           use_ego_history=False,
                                           ego_state_dropout=ego_state_dropout)
    if ego_query_fusion:
        from magnet_ego_query_fusion import MagnetEgoQueryFusion  # noqa: WPS433
        return MagnetEgoQueryFusion(context_pooling=context_pooling,
                                    context_pool_mode=context_pool_mode,
                                    ego_query_fusion=True,
                                    ego_query_heads=ego_query_heads)
    if context_pooling:
        from magnet_context_pooled import MagnetContextPooled  # noqa: WPS433
        return MagnetContextPooled(context_pooling=True, context_pool_mode=context_pool_mode)
    if gnn_pooling_fix:
        from magnet_imitation_pooled import MagnetImitationPooled  # noqa: WPS433 局部导入避免无谓依赖
        return MagnetImitationPooled()
    return TransformerPredictor()


def read_architecture(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """从 checkpoint 同目录的 `train_meta.json` 读 `architecture` 块。

    为什么要有这一步：**calibrated 产物是 bare state_dict**，架构信息不在权重里，
    而 `ego_query_heads` 又无法从形状反推 ⇒ 只能靠 sidecar。
    没有 sidecar 或没有 `architecture` 块时返回 None（老 checkpoint 的正常情况），
    由调用方决定用显式参数还是报错 —— 本函数**不猜**。
    """
    meta = Path(ckpt_path).parent / "train_meta.json"
    if not meta.exists():
        return None
    try:
        arch = json.loads(meta.read_text(encoding="utf-8")).get("architecture")
    except (OSError, ValueError) as exc:
        logger.warning("读取 %s 失败: %s", meta, exc)
        return None
    return arch if isinstance(arch, dict) else None


def _get_bn(model: nn.Module) -> nn.BatchNorm1d:
    """取出 `ego_history_encoder` 里的 BatchNorm1d。"""
    encoder = getattr(model, "ego_history_encoder", None)
    if encoder is None:
        raise CalibrationError("模型没有 ego_history_encoder，S4-b 的前提不成立")
    bn = encoder[1]
    if not isinstance(bn, nn.BatchNorm1d):
        raise CalibrationError(f"{BN_MODULE_PATH} 不是 BatchNorm1d，而是 {type(bn).__name__}")
    return bn


def swap_bn_with_running_stats(model: nn.Module) -> nn.BatchNorm1d:
    """把 `ego_history_encoder[1]` 换成同权重、带 running stats 的 BN，返回新层。

    ``momentum=None`` ⇒ 累积移动平均（cumulative moving average），等价于对所有 calibration
    batch 求均值；配合**等大小 batch** 时该平均是精确的样本均值（见 `calibrate_bn` 的断言）。

    :raises CalibrationError: 原层已经带 running stats（说明 ckpt 已 calibrated，重复操作会清空统计量）。
    """
    old = _get_bn(model)
    if old.track_running_stats:
        raise CalibrationError(
            f"{BN_MODULE_PATH} 已带 running stats（track_running_stats=True），"
            "不要对 calibrated checkpoint 重复 calibration")
    new = nn.BatchNorm1d(old.num_features, eps=old.eps, momentum=None,
                         affine=old.affine, track_running_stats=True)
    if old.affine:
        new.weight.data.copy_(old.weight.data)
        new.bias.data.copy_(old.bias.data)
    new.to(next(model.parameters()).device)
    model.ego_history_encoder[1] = new
    return new


@torch.no_grad()
def calibrate_bn(model: nn.Module, batches: Sequence[Any], device: torch.device,
                 move_batch_fn) -> Dict[str, Any]:
    """在固定的 calibration batch 序列上累计 running statistics。

    只把目标 BN 切到 train 模式，其余模块（含 dropout）保持 eval —— 与 `bn_calibration_probe.py`
    的做法一致，但数据来源必须是 **train split 的固定 manifest**。

    :param batches: 已经 collate 好的 batch 序列（顺序固定，全部等大小）。
    :param move_batch_fn: `train_magnet_imitation.move_batch_to_device`，搬自定义属性用。
    :returns: 统计量摘要（写进 provenance）。
    :raises CalibrationError: batch 大小不一致（累积平均会被小 batch 拉偏）。
    """
    bn = _get_bn(model)
    if not bn.track_running_stats:
        raise CalibrationError("请先调用 swap_bn_with_running_stats()")

    sizes = [int(b.num_graphs) for b in batches]
    if len(set(sizes)) != 1:
        raise CalibrationError(
            f"calibration batch 大小不一致 {sorted(set(sizes))}；momentum=None 的累积平均按 batch "
            "等权，必须保证 N 能被 batch_size 整除")

    model.eval()
    bn.train()
    bn.reset_running_stats()
    for batch in batches:
        model(move_batch_fn(batch, device))
    bn.eval()

    n_samples = sum(sizes)
    logger.info("BN calibration 完成：%d batch × %d 样本 = %d；running_mean[:3]=%s",
                len(batches), sizes[0], n_samples,
                [round(v, 6) for v in bn.running_mean.tolist()[:3]])
    return {
        "bn_module": BN_MODULE_PATH,
        "n_batches": len(batches),
        "batch_size": sizes[0],
        "n_samples": n_samples,
        "num_batches_tracked": int(bn.num_batches_tracked.item()),
        "running_mean": [float(v) for v in bn.running_mean.tolist()],
        "running_var": [float(v) for v in bn.running_var.tolist()],
        "eps": float(bn.eps),
    }


def load_model_for_inference(ckpt_path: Path, gnn_pooling_fix: bool = False,
                             device: Optional[torch.device] = None,
                             **arch_overrides) -> nn.Module:
    """统一加载入口：原始 ckpt 与 calibrated ckpt 都走这里。

    calibrated 权重比原始权重多 ``ego_history_encoder.1.running_{mean,var}`` 与
    ``num_batches_tracked`` 三个键；据此自动决定是否先换 BN 再 load，避免
    ``load_state_dict`` 报 unexpected keys。

    **架构解析顺序**（C2 之后模型不再只有两种，必须显式定义）：

    1. 调用方显式传入的 `arch_overrides`（最高优先）；
    2. checkpoint 同目录 `train_meta.json` 的 `architecture` 块；
    3. 都没有 ⇒ 退回 `gnn_pooling_fix` 参数描述的老两态。

    第 3 档对老 checkpoint 是正确的（它们本来就只有两态）；对 C2 这类新架构，
    缺 sidecar 时 `load_state_dict(strict=True)` 会**直接抛错**而不是静默建错模型 —— 这是有意的。

    :returns: 已 `eval()` 的模型。
    """
    device = device or torch.device("cpu")
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj

    kwargs: Dict[str, Any] = {"gnn_pooling_fix": gnn_pooling_fix}
    sidecar = read_architecture(Path(ckpt_path))
    if sidecar:
        kwargs["gnn_pooling_fix"] = bool(sidecar.get("gnn_pooling_fix_effective",
                                                     sidecar.get("gnn_pooling_fix", gnn_pooling_fix)))
        for k in ARCH_BUILD_KEYS:
            if k in sidecar:
                kwargs[k] = sidecar[k]
        logger.info("架构取自 sidecar train_meta.json: %s", kwargs)
    kwargs.update({k: v for k, v in arch_overrides.items() if v is not None})

    model = build_model(**kwargs).to(device)
    if f"{BN_MODULE_PATH}.running_mean" in state:
        swap_bn_with_running_stats(model)
    # strict=True（默认）：架构猜错就抛错，绝不静默加载成部分权重
    model.load_state_dict(state)
    model.eval()
    return model


def collate_fixed_batches(data_list: List[Any], batch_size: int,
                          follow_batch: Optional[List[str]] = None) -> List[Any]:
    """按固定顺序、固定大小 collate（不 shuffle、不 drop 尾巴，尾巴不整除时报错）。"""
    from torch_geometric.loader import DataLoader  # noqa: WPS433

    if len(data_list) % batch_size != 0:
        raise CalibrationError(
            f"样本数 {len(data_list)} 不能被 batch_size {batch_size} 整除；S4-b 要求等大小 batch")
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, follow_batch=follow_batch)
    return list(loader)
