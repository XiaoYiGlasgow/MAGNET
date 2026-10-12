"""devkit 运动学增强「求解失败回退 G.T.」修正子类的单元测试（nuplan 环境，CPU，读全量 L1 cache 真实样本，不训练）。

验证对象：GNN_Trans/external_baselines/data_augmentation/kinematic_fallback.py（09-14 用户选定方案 A）

检查项
------
K1 求解成功路径与父类逐位一致：同一批真实 cache 样本、同一随机数状态、augment_prob=1（强制走求解），
   父类与子类输出的 features / targets 的 serialize() 逐位相同；并要求至少一个样本确实被增强改动（排除平凡相等）。
   LaneGCN 对应 KinematicAgentAugmentor（未来 60 点 / 0.1 s），Urban Driver 对应 KinematicHistoryGenericAgentAugmentor（dt 0.1）
K2 强制求解器抛 RuntimeError：父类抛 UnboundLocalError（复现 devkit bug）；子类不抛，写出 FALLBACK_MARK，
   且输出与「父类走 success=False 官方回退分支」逐位相同（同一随机数状态）
K3 我方两个对齐版增强 yaml 经 hydra 实例化得到的是修正子类，关键参数与 yaml 一致

用法：
  CUDA_VISIBLE_DEVICES="" /home/team/anaconda3/envs/nuplan/bin/python tests/test_kinematic_fallback_augmentation.py [--n_samples 8]
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(GNN.parent) not in sys.path:
    sys.path.insert(0, str(GNN.parent))

AUG_DIR = GNN / "external_baselines/config/data_augmentation"
SPEC: Dict[str, Dict[str, Any]] = {
    "lanegcn": {"model_yaml": GNN / "external_baselines/config/model/lanegcn_aligned.yaml",
                "cache": GNN / "data/cache_L1_lanegcn",
                "aug_key": "kinematic_agent_augmentation_aligned"},
    "urban_driver": {"model_yaml": GNN / "external_baselines/config/model/urban_driver_aligned.yaml",
                     "cache": GNN / "data/cache_L1_urban_driver",
                     "aug_key": "kinematic_history_generic_agent_augmentation_aligned"},
}


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _FakeFailedSol:
    """模拟求解器返回 success=False（父类官方回退分支）。"""

    @staticmethod
    def stats() -> Dict[str, Any]:
        return {"success": False, "return_status": "Fake_Failure"}


def _same(a: Any, b: Any) -> bool:
    """serialize() 结果逐位相同（浮点含 NaN 位置一致）。"""
    import torch

    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return type(a) is type(b) and a == b


def _serialize(objs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.serialize() for k, v in objs.items()}


def load_samples(model: str, n: int) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """从全量 L1 cache 均匀抽 n 个 token，按模型 builder 反序列化出 (token, features, targets)。"""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle

    net = instantiate(OmegaConf.load(SPEC[model]["model_yaml"]))
    feat_builders = list(net.get_list_of_required_feature())
    targ_builders = list(net.get_list_of_computed_target())
    store = FeatureCachePickle()
    dirs = sorted(d for d in SPEC[model]["cache"].glob("*/*/*") if d.is_dir() and d.parts[-3] != "metadata")
    picks = dirs[:: max(len(dirs) // n, 1)][:n]
    out = []
    for d in picks:
        feats = {b.get_feature_unique_name(): store.load_computed_feature_from_folder(d / b.get_feature_unique_name(),
                                                                                    b.get_feature_type())
                 for b in feat_builders}
        targs = {b.get_feature_unique_name(): store.load_computed_feature_from_folder(d / b.get_feature_unique_name(),
                                                                                    b.get_feature_type())
                 for b in targ_builders}
        out.append((d.name, feats, targs))
    return out


def aug_kwargs(model: str) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(AUG_DIR / f"{SPEC[model]['aug_key']}.yaml")[SPEC[model]["aug_key"]])
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def classes(model: str):
    from nuplan.planning.training.data_augmentation.kinematic_agent_augmentation import KinematicAgentAugmentor
    from nuplan.planning.training.data_augmentation.kinematic_history_generic_agent_augmentation import (
        KinematicHistoryGenericAgentAugmentor,
    )
    from GNN_Trans.external_baselines.data_augmentation import (
        KinematicAgentAugmentorFallback,
        KinematicHistoryGenericAgentAugmentorFallback,
    )

    if model == "lanegcn":
        return KinematicAgentAugmentor, KinematicAgentAugmentorFallback
    return KinematicHistoryGenericAgentAugmentor, KinematicHistoryGenericAgentAugmentorFallback


def _run(cls, kwargs: Dict[str, Any], feats: Dict[str, Any], targs: Dict[str, Any], seed: int):
    """新建增强器（噪声生成器状态一致），固定全局随机数后增强一次，返回 serialize() 结果。"""
    aug = cls(**kwargs)
    f, t = copy.deepcopy(feats), copy.deepcopy(targs)
    np.random.seed(seed)
    f2, t2 = aug.augment(f, t, None)
    return _serialize(f2), _serialize(t2)


def check_k1(model: str, samples, fails: List[str]) -> Dict[str, Any]:
    parent, child = classes(model)
    kw = {**aug_kwargs(model), "augment_prob": 1.0}
    changed = 0
    for i, (tok, feats, targs) in enumerate(samples):
        p = _run(parent, kw, feats, targs, seed=1000 + i)
        c = _run(child, kw, feats, targs, seed=1000 + i)
        if not (_same(p[0], c[0]) and _same(p[1], c[1])):
            fails.append(f"K1 {model} {tok}：子类与父类输出不逐位相同")
        if not (_same(p[0], _serialize(feats)) and _same(p[1], _serialize(targs))):
            changed += 1
    if changed == 0:
        fails.append(f"K1 {model}：{len(samples)} 个样本全部未被增强改动，逐位相同不能说明求解路径一致")
    return {"n": len(samples), "changed_by_augmentation": changed}


def check_k2(model: str, samples, fails: List[str]) -> Dict[str, Any]:
    import nuplan.planning.training.data_augmentation.data_augmentation_util as util
    from GNN_Trans.external_baselines.data_augmentation import FALLBACK_MARK

    parent, child = classes(model)
    kw = {**aug_kwargs(model), "augment_prob": 1.0}
    handler = _ListHandler()
    fb_logger = logging.getLogger("GNN_Trans.external_baselines.data_augmentation.kinematic_fallback")
    fb_logger.addHandler(handler)
    fb_logger.setLevel(logging.WARNING)   # main 里 basicConfig 把根 logger 设成 ERROR，warning 会在 logger 层被滤掉
    original: Callable = util.ConstrainedNonlinearSmoother.solve
    res = {"parent_raises": [], "child_marks": 0}
    try:
        for i, (tok, feats, targs) in enumerate(samples[:3]):
            def _raise(self):  # noqa: ANN001
                raise RuntimeError("Error in Opti::solve [OptiNode] 强制失败（单测）")

            util.ConstrainedNonlinearSmoother.solve = _raise
            try:
                _run(parent, kw, feats, targs, seed=2000 + i)
                fails.append(f"K2 {model} {tok}：父类在求解器抛 RuntimeError 时没有抛异常（与 bug 描述不符）")
            except UnboundLocalError:
                res["parent_raises"].append("UnboundLocalError")
            except Exception as e:  # noqa: BLE001  记录父类实际抛出的异常类型
                res["parent_raises"].append(type(e).__name__)
                fails.append(f"K2 {model} {tok}：父类抛的是 {type(e).__name__} 而非 UnboundLocalError")
            before = len(handler.messages)
            got = _run(child, kw, feats, targs, seed=2000 + i)
            if not any(FALLBACK_MARK in m for m in handler.messages[before:]):
                fails.append(f"K2 {model} {tok}：子类回退时没有写出 FALLBACK_MARK")
            res["child_marks"] += sum(FALLBACK_MARK in m for m in handler.messages[before:])

            util.ConstrainedNonlinearSmoother.solve = lambda self: _FakeFailedSol()
            want = _run(parent, kw, feats, targs, seed=2000 + i)
            if not (_same(got[0], want[0]) and _same(got[1], want[1])):
                fails.append(f"K2 {model} {tok}：子类回退输出与父类 success=False 官方分支不逐位相同")
    finally:
        util.ConstrainedNonlinearSmoother.solve = original
    return res


def check_k3(fails: List[str]) -> Dict[str, Any]:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    out = {}
    for model in SPEC:
        key = SPEC[model]["aug_key"]
        cfg = OmegaConf.load(AUG_DIR / f"{key}.yaml")[key]
        aug = instantiate(cfg)
        _, child = classes(model)
        out[model] = type(aug).__name__
        if type(aug) is not child:
            fails.append(f"K3 {model}：yaml 实例化得到 {type(aug).__name__}，不是 {child.__name__}")
        if abs(aug._augment_prob - 0.5) > 1e-12:
            fails.append(f"K3 {model}：augment_prob {aug._augment_prob} ≠ 0.5")
    kw_l, kw_u = aug_kwargs("lanegcn"), aug_kwargs("urban_driver")
    if (kw_l["trajectory_length"], kw_l["dt"]) != (60, 0.1):
        fails.append(f"K3 lanegcn：trajectory_length / dt = {kw_l['trajectory_length']} / {kw_l['dt']}，应为 60 / 0.1")
    if kw_u["dt"] != 0.1:
        fails.append(f"K3 urban_driver：dt = {kw_u['dt']}，应为 0.1")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    fails: List[str] = []
    report: Dict[str, Any] = {}
    for model in SPEC:
        samples = load_samples(model, args.n_samples)
        report[f"K1_{model}"] = check_k1(model, samples, fails)
        report[f"K2_{model}"] = check_k2(model, samples, fails)
    report["K3"] = check_k3(fails)
    for k, v in report.items():
        print(k, v)
    if fails:
        print(f"❌ {len(fails)} 项未通过：")
        for f in fails:
            print("  -", f)
        return 1
    print("✅ K1–K3 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
