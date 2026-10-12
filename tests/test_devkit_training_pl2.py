"""devkit 两个对齐版基线的 PL 2 训练管线单元测试（nuplan 环境，CPU，几个 val token，不产出可用模型）。

验证对象：GNN_Trans/external_baselines/training/pl2_compat.py（run_devkit_training.py 只是把它接到 hydra 上）

检查项
------
P1 trainer 参数映射：devkit 真实的 default_lightning.yaml 里每个键都被登记（直通或写明理由丢弃），没有未知键
P2 ``DataModulePL2.transfer_batch_to_device`` 接受 PL 2 的三参调用，结果与 devkit 两参调用相同
P3 两个模型各用 devkit 的 DataModule / LightningModuleWrapper / 官方 objective 与 metric / 对齐版增强，
   在 4 个 val token 上由 PL 2 Trainer 真训 1 epoch（CPU、precision 32、batch 2、2 个训练 step、1 个验证 batch），
   loss 有限；checkpoints/ 下存出 ``epoch=0.ckpt`` 与 ``last.ckpt``，best_model/ 下存出按 monitor 选的 ckpt
P4 ``last.ckpt`` 经读数脚本的 ``DevkitAdapter.load_weights`` strict 加载成功，前向输出 [B, 60, 3]

用法：
  /home/team/anaconda3/envs/nuplan/bin/python tests/test_devkit_training_pl2.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
DEVKIT_CFG = GNN.parent / "nuplan/planning/script/config/training"
for p in (str(GNN.parent), str(GNN / "scripts/analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

TRAIN = {
    "urban_driver": ("trajectory_weight_decay_imitation_objective",
                     ["kinematic_history_generic_agent_augmentation_aligned", "generic_agent_dropout_augmentation"]),
    "lanegcn": ("imitation_objective", ["kinematic_agent_augmentation_aligned", "agent_dropout_augmentation"]),
}
METRICS = ("avg_displacement_error", "avg_heading_error", "final_displacement_error", "final_heading_error")


class _AllSplitter:
    """测试用：训练集与验证集都返回全部场景。"""

    def get_train_samples(self, scenarios, worker):  # noqa: D401
        return scenarios

    def get_val_samples(self, scenarios, worker):
        return scenarios

    def get_test_samples(self, scenarios, worker):
        return scenarios


def _aug_yaml(name: str) -> Path:
    local = GNN / f"external_baselines/config/data_augmentation/{name}.yaml"
    return local if local.exists() else DEVKIT_CFG / f"data_augmentation/{name}.yaml"


def check_trainer_mapping(fails: List[str]) -> None:
    from omegaconf import OmegaConf

    from GNN_Trans.external_baselines.training.pl2_compat import trainer_kwargs_from_cfg

    params = OmegaConf.load(DEVKIT_CFG / "lightning/default_lightning.yaml").trainer.params
    try:
        kw = trainer_kwargs_from_cfg(params, use_gpu=True)
        print(f"P1 映射后 Trainer 参数：{kw}")
    except ValueError as e:
        fails.append(f"P1 {e}")


def check_transfer(fails: List[str]) -> None:
    import torch

    from nuplan.planning.training.data_loader.datamodule import DataModule
    from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
    from GNN_Trans.external_baselines.training.pl2_compat import DataModulePL2

    # devkit 的 move_features_type_to_device 要求值是带 to_device 的 feature 对象
    batch = ({"trajectory": Trajectory(data=torch.zeros(2, 60, 3))},
             {"trajectory": Trajectory(data=torch.ones(2, 60, 3))}, ["scenario"])
    # 该方法不读实例字段；用 __new__ 取实例（不跑 __init__），子类里的零参 super() 需要真实实例
    a = DataModulePL2.__new__(DataModulePL2).transfer_batch_to_device(batch, torch.device("cpu"), 0)
    b = DataModule.__new__(DataModule).transfer_batch_to_device(batch, torch.device("cpu"))
    if not (torch.equal(a[0]["trajectory"].data, b[0]["trajectory"].data)
            and torch.equal(a[1]["trajectory"].data, b[1]["trajectory"].data) and a[2] == b[2]):
        fails.append("P2 三参调用结果与 devkit 两参调用不同")


def check_train(name: str, scenarios, fails: List[str]) -> None:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import external_baselines_val_readout as readout
    from nuplan.planning.training.data_loader.datamodule import DataModule
    from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
    from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    from GNN_Trans.external_baselines.training.pl2_compat import adapt_datamodule, build_trainer_pl2

    adapter = readout.DevkitAdapter(name)
    model = adapter.build_model()
    obj_name, aug_names = TRAIN[name]
    obj_cfg = next(iter(OmegaConf.load(DEVKIT_CFG / f"objective/{obj_name}.yaml").values()))
    objectives = [instantiate(obj_cfg, scenario_type_loss_weighting={})]
    metrics = [instantiate(next(iter(OmegaConf.load(DEVKIT_CFG / f"training_metric/{m}.yaml").values())))
               for m in METRICS]
    augmentors = [instantiate(next(iter(OmegaConf.load(_aug_yaml(a)).values()))) for a in aug_names]

    dm = adapt_datamodule(DataModule(
        feature_preprocessor=FeaturePreprocessor(cache_path=None, force_feature_computation=True,
                                                 feature_builders=model.get_list_of_required_feature(),
                                                 target_builders=model.get_list_of_computed_target()),
        splitter=_AllSplitter(), all_scenarios=list(scenarios), train_fraction=1.0, val_fraction=1.0,
        test_fraction=1.0, dataloader_params={"batch_size": 2, "num_workers": 0, "pin_memory": False,
                                              "drop_last": True},
        scenario_type_sampling_weights=OmegaConf.create({"enable": False, "scenario_type_weights": {}}),
        worker=Sequential(), augmentors=augmentors))
    lm = LightningModuleWrapper(model=model, objectives=objectives, metrics=metrics, batch_size=2,
                                optimizer=OmegaConf.load(DEVKIT_CFG / "optimizer/adam.yaml"),
                                objective_aggregate_mode="mean")

    with tempfile.TemporaryDirectory() as tmp:
        lightning = OmegaConf.load(DEVKIT_CFG / "lightning/default_lightning.yaml")
        lightning.trainer.params.update({"max_epochs": 1, "precision": 32, "limit_train_batches": 2,
                                         "limit_val_batches": 1, "num_sanity_val_steps": 0})
        cfg = OmegaConf.create({"gpu": False, "group": tmp, "experiment": "unit", "output_dir": tmp,
                                "lightning": lightning})
        trainer = build_trainer_pl2(cfg)
        trainer.fit(model=lm, datamodule=dm)
        loss = trainer.callback_metrics.get("loss/train_loss")
        ck = Path(tmp) / "checkpoints"
        best = list((Path(tmp) / "best_model").glob("*.ckpt"))
        print(f"P3 {name}：global_step {trainer.global_step}，train loss {loss}，"
              f"checkpoints {sorted(p.name for p in ck.glob('*.ckpt'))}，best_model {[p.name for p in best]}")
        if trainer.global_step != 2:
            fails.append(f"P3 {name} 训练 step 数 {trainer.global_step} ≠ 2")
        if loss is None or not bool(torch.isfinite(loss)):
            fails.append(f"P3 {name} train loss 非有限或未记录：{loss}")
        if not (ck / "epoch=0.ckpt").exists() or not (ck / "last.ckpt").exists() or not best:
            fails.append(f"P3 {name} checkpoint 缺失")
            return

        # P4 读数脚本加载 last.ckpt 并前向
        fresh = adapter.build_model()
        adapter.load_weights(fresh, str(ck / "last.ckpt"))
        feats = [{"features": {b.get_feature_unique_name(): b.get_features_from_scenario(sc)
                               for b in fresh.feature_builders}} for sc in scenarios[:2]]
        with torch.no_grad():
            out = adapter.predict(fresh.eval(), feats, torch.device("cpu"))
        shape = (len(out),) + out[0]["ego_top1"].shape
        print(f"P4 {name}：strict 加载 last.ckpt 成功，读数前向 {shape}")
        if shape != (2, 60, 2):
            fails.append(f"P4 {name} 读数输出形状 {shape}")


def main() -> int:
    import external_baselines_readout_lib as lib
    import external_baselines_val_readout as readout

    fails: List[str] = []
    check_trainer_mapping(fails)
    check_transfer(fails)
    tokens = lib.val_tokens(4)
    scenarios = []
    by_log = {}
    for tk, lg in lib.token_to_log(tokens).items():
        by_log.setdefault(lg, []).append(tk)
    for lg, tks in by_log.items():
        scenarios += readout._scenarios_for_log(lg, tks)
    for name in TRAIN:
        check_train(name, scenarios, fails)

    if fails:
        print(f"\n❌ {len(fails)} 项失败：")
        for f in fails:
            print("  -", f)
        return 1
    print("\n✅ P1–P4 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
