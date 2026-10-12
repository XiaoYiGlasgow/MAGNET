"""外部基线在 val 2400 上的开环读数（方案文档 补外部基线_方案_20260911.md §00.5 的 S4）。

模型
----
* ``plantf_official``：L1 三 seed（``s3_plantf_L1``），特征按 E8 同配方重建（默认 ``NuplanFeatureBuilder()``）
* ``plantf_aligned``：``NuplanFeatureBuilderAligned``（external/planTF）
* ``urban_driver`` / ``lanegcn``：``GNN_Trans/external_baselines`` 对齐版

读数
----
* 自车 top1 ADE / FDE / missRate：窗口 60 点（t0+0.1 … t0+6.0 s），预测减自己的首点搬进我方口径
  （PlanTF top1 = ``argmax probability``；devkit 两个模型单输出）。
* PlanTF 两个版本另读周车辅助头：只取 ``category == 1``、t0 有效的行，按 t0 几何最近邻（≤ 1.0 m、一对一）
  配对到我方周车总体；误差用**我方** ``AGENT_FUTURES`` 与我方逐帧 mask；分 agent / moving / static 三层。
  同一总体上附 ``stay_still`` 参照（预测恒为 t0 位置）。口径定义见 ``external_baselines_readout_lib``。

隔离
----
* 不改 ``e8_plantf_openloop.py``，只借它的 ``db_path`` / ``MAP_ROOT`` / L1 ckpt 定位 / 官方模型构造。
* 🔴 只读 val；冻结 test 待 D4 裁定后一次性批量开，本脚本不接受 test。
* 场景提取与 MAGNET 相同：``NuPlanScenarioBuilder`` 不传 ``scenario_mapping``、``scenario_types=None``。

环境：``plantf_*`` 用 plantf 环境；``urban_driver`` / ``lanegcn`` 用 nuplan 环境。

用法
----
  python scripts/analysis/external_baselines_val_readout.py --model lanegcn --stage features \\
      [--log_shard_index 0 --log_shard_total 4]
  python scripts/analysis/external_baselines_val_readout.py --model lanegcn --stage eval \\
      --ckpts 1:/abs/seed1.ckpt,2:/abs/seed2.ckpt,42:/abs/seed42.ckpt [--device cpu]
  冒烟（几个 token、随机权重，须显式给 --feature_dir 与 --out）：--n_tokens 3 --random_init
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
PLANTF = Path("/mnt/local/Xiao_Yi/external/planTF")
sys.path.insert(0, str(GNN / "scripts/analysis"))
import external_baselines_readout_lib as lib  # noqa: E402

logger = logging.getLogger("external_baselines_val")
MODELS = ("plantf_official", "plantf_aligned", "urban_driver", "lanegcn")
FEATURE_ROOT = GNN / "data/external_baselines_val_features"
OUT_ROOT = GNN / "experiments/external_baselines_val"
DEVKIT_MODEL_YAML = {
    "urban_driver": GNN / "external_baselines/config/model/urban_driver_aligned.yaml",
    "lanegcn": GNN / "external_baselines/config/model/lanegcn_aligned.yaml",
}
E8 = lib.load_module("e8_plantf_openloop", GNN / "scripts/analysis/e8_plantf_openloop.py")


# --------------------------------------------------------------------------- 模型适配


class PlanTFAdapter:
    """PlanTF 官方配置版 / 对齐版：自车 6 模态 + 周车辅助头。"""

    has_agents = True

    def __init__(self, aligned: bool) -> None:
        sys.path.insert(0, str(PLANTF))
        self.aligned = aligned
        self.hist_idx = 19 if aligned else 20          # 当前帧下标：对齐版 20 帧历史，官方 21 帧

    def prepare_features(self) -> Callable:
        if self.aligned:
            from src.feature_builders.nuplan_feature_builder_aligned import NuplanFeatureBuilderAligned
            builder = NuplanFeatureBuilderAligned()
        else:
            from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
            builder = NuplanFeatureBuilder()               # E8 同配方：默认参数（max_agents 64，见 D3）

        def build(sc) -> Tuple[Dict, np.ndarray]:
            feat = builder.get_features_from_scenario(sc).data
            return {"feature": feat}, lib.to_our_convention(feat["agent"]["target"][0])
        return build

    def build_model(self):
        if not self.aligned:
            return E8._build_plantf_model("cpu")
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        return instantiate(OmegaConf.load(PLANTF / "config/model/planTF_aligned.yaml"))

    @staticmethod
    def load_weights(model, ckpt: str) -> None:
        from src.planners.planner_utils import load_checkpoint
        model.load_state_dict(load_checkpoint(ckpt), strict=True)

    def predict(self, model, payloads: List[Dict], device) -> List[Dict]:
        import torch
        from src.features.nuplan_feature import NuplanFeature

        feats = [p["feature"] for p in payloads]
        batch = NuplanFeature.collate([NuplanFeature(data=d).to_feature_tensor() for d in feats]).to_device(device)
        out = model(batch.data)
        traj = out["trajectory"][..., :2][:, :, : lib.N_WINDOW]
        traj = (traj - traj[:, :, 0:1]).cpu().numpy()                     # [B, K, 60, 2]
        top1 = out["probability"].argmax(dim=-1).cpu().numpy()
        pred = out["prediction"][:, :, : lib.N_WINDOW].cpu().numpy()      # [B, A_max-1, 60, 2]，相对各自 t0
        res = []
        for b, d in enumerate(feats):
            ag = d["agent"]
            n = ag["position"].shape[0] - 1
            keep = ag["valid_mask"][1:, self.hist_idx] & (ag["category"][1:] == 1)
            p0 = np.asarray(ag["position"][1:, self.hist_idx], dtype=np.float64)[keep]
            fut_valid = ag["valid_mask"][1:, self.hist_idx + 1: self.hist_idx + 1 + lib.N_WINDOW][keep]
            res.append({"ego_top1": traj[b, top1[b]], "agent_p0": p0,
                        "agent_pred": p0[:, None] + pred[b, :n][keep], "agent_valid": fut_valid})
        return res


class DevkitAdapter:
    """Urban Driver / LaneGCN 对齐版：自车单输出，无周车头。"""

    has_agents = False

    def __init__(self, name: str) -> None:
        sys.path.insert(0, str(GNN.parent))
        self.yaml = DEVKIT_MODEL_YAML[name]

    def build_model(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        return instantiate(OmegaConf.load(self.yaml))

    def prepare_features(self) -> Callable:
        model = self.build_model()                         # 只借它的 builder，不用权重

        def build(sc) -> Tuple[Dict, np.ndarray]:
            feats = {b.get_feature_unique_name(): b.get_features_from_scenario(sc) for b in model.feature_builders}
            target = model.target_builders[0].get_targets(sc)
            return {"features": feats}, lib.to_our_convention(target.data)
        return build

    @staticmethod
    def load_weights(model, ckpt: str) -> None:
        import torch
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        # LightningModuleWrapper 把网络挂在 self.model 下，键带 "model." 前缀
        model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)

    @staticmethod
    def predict(model, payloads: List[Dict], device) -> List[Dict]:
        names = list(payloads[0]["features"])
        batch = {n: type(payloads[0]["features"][n]).collate(
            [p["features"][n].to_feature_tensor() for p in payloads]).to_device(device) for n in names}
        xy = model(batch)["trajectory"].data[..., :2]
        xy = (xy - xy[:, 0:1]).cpu().numpy()
        return [{"ego_top1": xy[b]} for b in range(len(payloads))]


def make_adapter(name: str):
    return PlanTFAdapter(aligned=name == "plantf_aligned") if name.startswith("plantf") else DevkitAdapter(name)


# --------------------------------------------------------------------------- stage: features


def _scenarios_for_log(log_name: str, tokens: List[str]):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    db = E8.db_path(log_name)
    builder = NuPlanScenarioBuilder(data_root=os.path.dirname(db), map_root=E8.MAP_ROOT, sensor_root=None,
                                    db_files=[db], map_version=E8.MAP_VERSION)
    sf = ScenarioFilter(scenario_types=None, scenario_tokens=list(tokens), log_names=None, map_names=None,
                        num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                        ego_displacement_minimum_m=None, expand_scenarios=False,
                        remove_invalid_goals=False, shuffle=False)
    return builder.get_scenarios(sf, Sequential())


def stage_features(args, adapter) -> None:
    out_dir = Path(args.feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = lib.val_tokens(args.n_tokens)
    by_log: Dict[str, List[str]] = {}
    for tk, lg in lib.token_to_log(tokens).items():
        by_log.setdefault(lg, []).append(tk)
    logs = [lg for i, lg in enumerate(sorted(by_log)) if i % args.log_shard_total == args.log_shard_index]
    logger.info("val token %d 个 / log %d 条；本分片 %d 条（shard %d/%d）", len(tokens), len(by_log), len(logs),
                args.log_shard_index, args.log_shard_total)

    build = adapter.prepare_features()
    stat = {"n_req": 0, "n_scenario": 0, "ok": 0, "gt_fail": 0, "err": 0}
    t0 = time.time()
    for i, lg in enumerate(logs, 1):
        scenarios = _scenarios_for_log(lg, by_log[lg])
        stat["n_req"] += len(by_log[lg])
        stat["n_scenario"] += len(scenarios)
        for sc in scenarios:
            dst = out_dir / f"{sc.token}.pkl"
            if dst.exists():
                stat["ok"] += 1
                continue
            try:
                payload, gt_theirs = build(sc)
                gap = float(np.linalg.norm(lib.load_our_sample(sc.token)["ego_gt"] - gt_theirs, axis=1).max())
                if gap > lib.GT_PARITY_TOL_M:
                    stat["gt_fail"] += 1
                    logger.warning("  %s GT 不吻合 max|Δ| = %.4f m ⇒ 丢弃", sc.token, gap)
                    continue
                with open(dst, "wb") as f:
                    pickle.dump({"payload": payload, "gt_parity_max_m": gap, "scenario_type": sc.scenario_type}, f)
                stat["ok"] += 1
            except Exception as e:                           # 单个场景坏掉不炸整条 log
                stat["err"] += 1
                logger.warning("  %s 失败：%s: %s", sc.token, type(e).__name__, e)
        logger.info("[%d/%d] %s 累计 %s（%.0f s）", i, len(logs), lg, stat, time.time() - t0)
    (out_dir / f"_stage_features_stat_shard{args.log_shard_index}.json").write_text(
        json.dumps({**stat, "model": args.model, "sec": time.time() - t0}, indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- stage: eval


def _parse_ckpts(args) -> Dict[int, str]:
    if args.random_init:
        return {0: ""}
    if args.ckpts:
        return {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in args.ckpts.split(",")}
    if args.model == "plantf_official":
        return {s: str(E8._l1_ckpt(s)) for s in (1, 2, 42)}
    raise SystemExit(f"{args.model} 须用 --ckpts 给出 seed:绝对路径")


def stage_eval(args, adapter) -> None:
    import torch

    files = sorted(Path(args.feature_dir).glob("*.pkl"))[: args.n_tokens or None]
    toks = [f.stem for f in files]
    payloads, stypes, gts, pops = [], [], [], []
    for f in files:
        o = pickle.load(open(f, "rb"))
        payloads.append(o["payload"])
        stypes.append(o["scenario_type"])
        s = lib.load_our_sample(f.stem)
        gts.append(s["ego_gt"])
        pops.append(lib.our_agent_population(s))
    GT = np.stack(gts)
    logger.info("装载 %d 个 token（val 共 %d）", len(toks), len(lib.val_tokens()))

    # 周车 stay_still 参照：与模型无关，同一总体（全部我方周车，不经配对）
    ss_ade, ss_fde = zip(*[lib.masked_errors(np.repeat(p["p0"][:, None], lib.N_WINDOW, 1), p["fut"], p["mask"])
                           for p in pops]) if pops else ((), ())
    all_moving = np.concatenate([p["is_moving"] for p in pops]) if pops else np.zeros(0, bool)
    stay_still = lib.summarize_agents(np.concatenate(ss_ade), np.concatenate(ss_fde), all_moving,
                                      np.ones(len(all_moving), bool), prefix="stay_still.")

    dev = torch.device(args.device)
    rows, per_token = [], {}
    for seed, ck in _parse_ckpts(args).items():
        model = adapter.build_model()
        if ck:
            adapter.load_weights(model, ck)
        model = model.to(dev).eval()
        t0 = time.time()
        preds: List[Dict] = []
        with torch.no_grad():
            for i in range(0, len(payloads), args.batch_size):
                preds += adapter.predict(model, payloads[i: i + args.batch_size], dev)
        ego_ade, ego_fde = lib.masked_errors(np.stack([p["ego_top1"] for p in preds]), GT,
                                             np.ones(GT.shape[:2], bool))
        row = {"seed": seed, "checkpoint": ck, **lib.summarize_ego(ego_ade, ego_fde)}
        rec = {"ego_ade": ego_ade, "ego_fde": ego_fde}
        if adapter.has_agents:
            a_tok, a_mov, a_pair, a_ade, a_fde, disagree = [], [], [], [], [], 0
            for t, (pop, pr) in enumerate(zip(pops, preds)):
                pair = lib.pair_agents(pop["p0"], pr["agent_p0"])
                ok = pair >= 0
                pred = np.zeros_like(pop["fut"])
                pred[ok] = pr["agent_pred"][pair[ok]]
                ade, fde = lib.masked_errors(pred, pop["fut"], pop["mask"])
                disagree += int((pop["mask"][ok] & ~pr["agent_valid"][pair[ok]]).sum())
                a_tok += [t] * len(ok); a_mov.append(pop["is_moving"]); a_pair.append(ok)
                a_ade.append(ade); a_fde.append(fde)
            a_mov, a_pair = np.concatenate(a_mov), np.concatenate(a_pair)
            a_ade, a_fde = np.concatenate(a_ade), np.concatenate(a_fde)
            row.update(lib.summarize_agents(a_ade, a_fde, a_mov, a_pair))
            row["agent_mask_disagree_frames"] = disagree
            rec.update({"agent_token_idx": np.asarray(a_tok), "agent_is_moving": a_mov, "agent_paired": a_pair,
                        "agent_ade": a_ade, "agent_fde": a_fde})
        row["sec"] = time.time() - t0
        logger.info("seed %s｜%s", seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)
        per_token[seed] = rec
        del model

    keys = [k for k in rows[0] if k.endswith(("ADE", "FDE", "missRate", "pairing_rate"))]
    out = {
        "model": args.model, "split": "val", "n_tokens": len(toks), "n_val_tokens": len(lib.val_tokens()),
        "smoke_random_init": bool(args.random_init), "feature_dir": str(args.feature_dir),
        "per_seed_rows": rows, "agg": lib.aggregate_seeds(rows, keys), "reference": stay_still,
        "scenario_type_hist": {t: stypes.count(t) for t in sorted(set(stypes))},
        "caliber": [
            "窗口 60 点 / t0+0.1 … t0+6.0 s；我方口径 = 预测减自己的首点（E8 docstring §三）",
            "val 2400（log-split 与 MAGNET 逐位相同），不得与冻结 test 同表",
            "周车：PlanTF 辅助头，category==1、t0 几何一对一配对（≤1.0 m），我方 AGENT_FUTURES 与我方 mask；"
            "moving/static 为 scene_strata 冻结口径；stay_still 在全部我方周车上算（不经配对）",
            "周车绝对误差含锚点 float32 平移地板（≤0.25 m，逐 token 恒定，输入不可见），基线与 MAGNET 同一地板",
            "plantf_official：训练 yaml max_agents=32、特征按 E8 默认 64（待 D3）；原生 80 步截到 60",
        ],
    }
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "per_token.npz", tokens=np.array(toks), scenario_type=np.array(stypes),
                        **{f"{k}_seed{s}": v for s, rec in per_token.items() for k, v in rec.items()})
    (dst / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    logger.info("✅ 产物 %s", dst / "summary.json")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--stage", choices=("features", "eval"), required=True)
    ap.add_argument("--feature_dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpts", default="", help="seed:绝对路径,…；plantf_official 缺省用 L1 三 seed")
    ap.add_argument("--n_tokens", type=int, default=0, help="只取前 N 个 val token（冒烟用）")
    ap.add_argument("--random_init", action="store_true", help="随机权重冒烟；须显式给 --feature_dir 与 --out")
    ap.add_argument("--log_shard_index", type=int, default=0)
    ap.add_argument("--log_shard_total", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if (args.random_init or args.n_tokens) and not (args.feature_dir and args.out or args.stage == "features"
                                                    and args.feature_dir):
        raise SystemExit("冒烟（--random_init / --n_tokens）不得写默认产物目录，须显式给 --feature_dir（eval 另需 --out）")
    args.feature_dir = args.feature_dir or str(FEATURE_ROOT / args.model)
    args.out = args.out or str(OUT_ROOT / args.model)
    adapter = make_adapter(args.model)
    (stage_features if args.stage == "features" else stage_eval)(args, adapter)


if __name__ == "__main__":
    main()
