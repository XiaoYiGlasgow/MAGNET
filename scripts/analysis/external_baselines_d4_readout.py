"""D4 · 外部基线在冻结 test 上的**一次性**全量读数（方案文档 §00.6 步 D；兼容 val 用于回归）。

与 ``external_baselines_val_readout.py`` 的关系
------------------------------------------------
* 复用其模型适配器与场景提取（按文件路径加载），**不修改** val 脚本，09-15/16 的 val 产物逐位可复现。
* 新增 ``plantf_official32``：官方配置版按训练口径 ``max_agents=32`` 建特征（D3 = (a)，09-16 用户裁定）；
  其余构造参数与 ``config/model/planTF.yaml`` 的 feature_builder 及 ``e8_plantf_openloop._build_plantf_model``
  逐项相同（radius 100 / history 2 s / future 8 s / sample 0.1 s）。
* 读数口径与 val 脚本逐字相同（窗口 60 点、预测减自己首点、周车 t0 几何配对 ≤1.0 m、我方 GT 与 mask、行微平均）。

一次性全量落盘（用户 09-16：「D4 做完再设计读数」⇒ 冻结 test 只开一次，出表所需一律从落盘数据算）
-------------------------------------------------------------------------------------------------
``per_token.npz``（模型无关的量只存一份，模型相关的量按 seed 存）：

* 模型无关：``tokens`` / ``scenario_type`` / ``ego_gt`` [N,60,2]；周车总体（与 MAGNET 读数同一筛选、同一行序）的
  ``agent_token_idx`` / ``agent_slot`` / ``agent_p0`` / ``agent_fut`` [R,60,2] / ``agent_mask`` [R,60] /
  ``agent_is_moving``（GT 分层，仅诊断）/ ``agent_t0_speed`` / ``agent_dist_t0`` / ``agent_is_t0_observable``（只读 t0）/
  ``agent_n_valid_frames`` / ``agent_disp_gt``；以及 ``stay_still_agent_ade`` / ``stay_still_agent_fde``。
* 每个 seed：``ego_ade`` / ``ego_fde``（top-1）、``ego_all`` [N,K,60,2]、``ego_prob`` [N,K]、``ego_top1_idx`` [N]、
  ``ego_ade_all`` / ``ego_fde_all`` [N,K]；有周车头的模型另存 ``agent_paired`` / ``agent_pred`` [R,60,2] /
  ``agent_ade`` / ``agent_fde``。

冻结 test 防火墙
----------------
* ``--split test`` 必须同时给 ``--open_test_once``，并当场通过 ``lib.verify_test_firewall()``（token 数 / log 数 /
  log 清单 sha256[:16] / 与 val 不相交）。features 阶段也算开 test（要读 test 标签做 GT 一致性闸）。
* 任何产物目录已有 ``summary.json`` 即拒绝写入（不覆盖）；ckpt 必须显式给 ``--ckpts``（D10 = A 的 best 路径）。
* eval 阶段在读任何特征与标签之前，先逐个 ckpt 建模型并试加载权重（``_preflight_ckpts``）；任一失败即退出，
  不会因 ckpt 出错而白开一次 test（09-16 开 test 前审查第⑤条）。

产物
----
* 特征：``data/external_baselines_{val,test}_features/<模型>/``（val 侧只新增 ``plantf_official32``，其余复用已有目录只读）
* 读数：``experiments/external_baselines_d4/<split>/<模型>/{summary.json,per_token.npz}``

用法
----
  # val 回归（只读已有特征，写 scratch）
  python scripts/analysis/external_baselines_d4_readout.py --model plantf_aligned --stage eval --split val \\
      --feature_dir data/external_baselines_val_features/plantf_aligned --out <scratch> --n_tokens 40 --ckpts 1:/abs.ckpt
  # 冻结 test（由 D4 守卫调用）
  python scripts/analysis/external_baselines_d4_readout.py --model lanegcn --stage features --split test --open_test_once
  python scripts/analysis/external_baselines_d4_readout.py --model lanegcn --stage eval --split test --open_test_once \\
      --ckpts 1:/abs1,2:/abs2,42:/abs42
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
sys.path.insert(0, str(GNN / "scripts/analysis"))
import external_baselines_readout_lib as lib  # noqa: E402

VR = lib.load_module("external_baselines_val_readout", GNN / "scripts/analysis/external_baselines_val_readout.py")

logger = logging.getLogger("external_baselines_d4")
MODELS = ("plantf_official32", "plantf_aligned", "urban_driver", "lanegcn")
FEATURE_ROOT = {"val": GNN / "data/external_baselines_val_features",
                "test": GNN / "data/external_baselines_test_features"}
OUT_ROOT = GNN / "experiments/external_baselines_d4"
OFFICIAL_BUILDER_KW = dict(radius=100, history_horizon=2, future_horizon=8, sample_interval=0.1, max_agents=32)


# --------------------------------------------------------------------------- 适配器（只加输出，不改读数）


class PlanTFFullAdapter(VR.PlanTFAdapter):
    """在父类读数之外，多返回全部自车模态、概率与 top-1 下标。"""

    def __init__(self, aligned: bool, official_builder_kw: Dict | None = None) -> None:
        super().__init__(aligned=aligned)
        self.official_builder_kw = official_builder_kw

    def prepare_features(self) -> Callable:
        if self.aligned or self.official_builder_kw is None:
            return super().prepare_features()
        from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
        builder = NuplanFeatureBuilder(**self.official_builder_kw)

        def build(sc) -> Tuple[Dict, np.ndarray]:
            feat = builder.get_features_from_scenario(sc).data
            return {"feature": feat}, lib.to_our_convention(feat["agent"]["target"][0])
        return build

    def predict(self, model, payloads: List[Dict], device) -> List[Dict]:
        import torch  # noqa: F401
        from src.features.nuplan_feature import NuplanFeature

        feats = [p["feature"] for p in payloads]
        batch = NuplanFeature.collate([NuplanFeature(data=d).to_feature_tensor() for d in feats]).to_device(device)
        out = model(batch.data)
        traj = out["trajectory"][..., :2][:, :, : lib.N_WINDOW]
        traj = (traj - traj[:, :, 0:1]).cpu().numpy()                     # [B, K, 60, 2]（与父类逐字相同）
        prob = out["probability"].cpu().numpy()                            # [B, K]
        top1 = out["probability"].argmax(dim=-1).cpu().numpy()
        pred = out["prediction"][:, :, : lib.N_WINDOW].cpu().numpy()
        res = []
        for b, d in enumerate(feats):
            ag = d["agent"]
            n = ag["position"].shape[0] - 1
            keep = ag["valid_mask"][1:, self.hist_idx] & (ag["category"][1:] == 1)
            p0 = np.asarray(ag["position"][1:, self.hist_idx], dtype=np.float64)[keep]
            fut_valid = ag["valid_mask"][1:, self.hist_idx + 1: self.hist_idx + 1 + lib.N_WINDOW][keep]
            res.append({"ego_top1": traj[b, top1[b]], "agent_p0": p0,
                        "agent_pred": p0[:, None] + pred[b, :n][keep], "agent_valid": fut_valid,
                        "ego_all": traj[b], "ego_prob": prob[b], "ego_top1_idx": int(top1[b])})
        return res


class DevkitFullAdapter(VR.DevkitAdapter):
    """devkit 两模型单输出：K = 1，概率恒 1。"""

    @staticmethod
    def predict(model, payloads: List[Dict], device) -> List[Dict]:
        res = VR.DevkitAdapter.predict(model, payloads, device)
        for r in res:
            r["ego_all"] = r["ego_top1"][None]
            r["ego_prob"] = np.ones(1, dtype=np.float32)
            r["ego_top1_idx"] = 0
        return res


def make_adapter(name: str):
    if name == "plantf_official32":
        return PlanTFFullAdapter(aligned=False, official_builder_kw=OFFICIAL_BUILDER_KW)
    if name == "plantf_aligned":
        return PlanTFFullAdapter(aligned=True)
    return DevkitFullAdapter(name)


def build_model_for(name: str, adapter):
    """官方 32 版的模型构造与 E8 逐字相同（E8 的 builder 也是 max_agents=32）。"""
    return adapter.build_model()


# --------------------------------------------------------------------------- 防火墙与不覆盖


def guard_split(args) -> Dict:
    if args.split != "test":
        return {"split": args.split}
    if not args.open_test_once:
        raise SystemExit("🔴 --split test 须显式给 --open_test_once（冻结 test 只开一次，由 D4 守卫调用）")
    rec = lib.verify_test_firewall()
    logger.info("冻结 test 身份核对通过：%s", rec)
    return {"split": "test", "firewall": rec}


def refuse_overwrite(dst: Path) -> None:
    if (dst / "summary.json").exists():
        raise SystemExit(f"🔴 {dst}/summary.json 已存在，拒绝覆盖")


# --------------------------------------------------------------------------- stage: features


def stage_features(args, adapter) -> None:
    fw = guard_split(args)
    out_dir = Path(args.feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = lib.SPLIT_ROOTS[args.split]
    tokens = lib.split_tokens(args.split, args.n_tokens)
    by_log: Dict[str, List[str]] = {}
    for tk, lg in lib.token_to_log(tokens).items():
        by_log.setdefault(lg, []).append(tk)
    logs = [lg for i, lg in enumerate(sorted(by_log)) if i % args.log_shard_total == args.log_shard_index]
    logger.info("%s token %d 个 / log %d 条；本分片 %d 条（shard %d/%d）", args.split, len(tokens), len(by_log),
                len(logs), args.log_shard_index, args.log_shard_total)
    build = adapter.prepare_features()
    stat = {"n_req": 0, "n_scenario": 0, "ok": 0, "gt_fail": 0, "err": 0}
    t0 = time.time()
    for i, lg in enumerate(logs, 1):
        scenarios = VR._scenarios_for_log(lg, by_log[lg])
        stat["n_req"] += len(by_log[lg])
        stat["n_scenario"] += len(scenarios)
        for sc in scenarios:
            dst = out_dir / f"{sc.token}.pkl"
            if dst.exists():
                stat["ok"] += 1
                continue
            try:
                payload, gt_theirs = build(sc)
                gap = float(np.linalg.norm(lib.load_our_sample(sc.token, root)["ego_gt"] - gt_theirs, axis=1).max())
                if gap > lib.GT_PARITY_TOL_M:
                    stat["gt_fail"] += 1
                    logger.warning("  %s GT 不吻合 max|Δ| = %.4f m ⇒ 丢弃", sc.token, gap)
                    continue
                with open(dst, "wb") as f:
                    pickle.dump({"payload": payload, "gt_parity_max_m": gap, "scenario_type": sc.scenario_type}, f)
                stat["ok"] += 1
            except Exception as e:
                stat["err"] += 1
                logger.warning("  %s 失败：%s: %s", sc.token, type(e).__name__, e)
        logger.info("[%d/%d] %s 累计 %s（%.0f s）", i, len(logs), lg, stat, time.time() - t0)
    (out_dir / f"_stage_features_stat_shard{args.log_shard_index}.json").write_text(json.dumps(
        {**stat, "model": args.model, "split": args.split, "sec": time.time() - t0,
         "builder_kw": OFFICIAL_BUILDER_KW if args.model == "plantf_official32" else None, **fw},
        indent=1, ensure_ascii=False))


# --------------------------------------------------------------------------- stage: eval


def _parse_ckpts(args) -> Dict[int, str]:
    if args.random_init:
        return {0: ""}
    if not args.ckpts:
        raise SystemExit("须用 --ckpts 给出 seed:绝对路径（D10 = A 的 best，不设缺省）")
    out = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in args.ckpts.split(",")}
    for s, c in out.items():
        if not Path(c).is_file():
            raise SystemExit(f"seed {s} 的 ckpt 不存在：{c}")
    return out


def _preflight_ckpts(args, adapter) -> Dict[int, str]:
    """读任何特征与标签之前，逐个 ckpt 建模型 + 试加载权重后释放；任一失败即退出（不白开 test）。"""
    ckpts = _parse_ckpts(args)
    for seed, ck in ckpts.items():
        if not ck:                                         # --random_init 冒烟没有权重可验
            continue
        model = build_model_for(args.model, adapter)
        try:
            adapter.load_weights(model, ck)
        except Exception as e:
            raise SystemExit(f"🔴 seed {seed} 的 ckpt 试加载失败（尚未读取任何 {args.split} 特征与标签）："
                             f"{type(e).__name__}: {e}") from e
        finally:
            del model
        logger.info("ckpt 预检通过：seed %s ← %s", seed, ck)
    return ckpts


def stage_eval(args, adapter) -> None:
    import torch

    fw = guard_split(args)
    dst = Path(args.out)
    refuse_overwrite(dst)
    ckpts = _preflight_ckpts(args, adapter)
    root = lib.SPLIT_ROOTS[args.split]
    files = sorted(Path(args.feature_dir).glob("*.pkl"))[: args.n_tokens or None]
    toks = [f.stem for f in files]
    if not args.n_tokens and toks != lib.split_tokens(args.split):
        raise SystemExit(f"特征目录 token 集合与 {args.split} 不一致：{len(toks)} vs {len(lib.split_tokens(args.split))}")

    payloads, stypes, gts, pops, metas = [], [], [], [], []
    for f in files:
        o = pickle.load(open(f, "rb"))
        payloads.append(o["payload"])
        stypes.append(o["scenario_type"])
        s = lib.load_our_sample(f.stem, root)
        gts.append(s["ego_gt"])
        pop, meta = lib.our_agent_population(s), lib.agent_population_meta(s)
        if not np.array_equal(pop["slot"], meta["slot"]):
            raise RuntimeError(f"{f.stem}: 周车总体与元数据行序不一致")
        pops.append(pop)
        metas.append(meta)
    GT = np.stack(gts)
    logger.info("装载 %d 个 token（%s 共 %d）", len(toks), args.split, len(lib.split_tokens(args.split)))

    # 模型无关的周车总体与 stay_still
    tok_idx = np.concatenate([np.full(len(p["slot"]), t, np.int64) for t, p in enumerate(pops)]) if pops else np.zeros(0, np.int64)
    cat = lambda key, src: np.concatenate([x[key] for x in src]) if src else np.zeros(0)  # noqa: E731
    ss_ade, ss_fde = zip(*[lib.masked_errors(np.repeat(p["p0"][:, None], lib.N_WINDOW, 1), p["fut"], p["mask"])
                           for p in pops])
    ss_ade, ss_fde = np.concatenate(ss_ade), np.concatenate(ss_fde)
    all_moving = cat("is_moving", pops).astype(bool)
    stay_still = lib.summarize_agents(ss_ade, ss_fde, all_moving, np.ones(len(all_moving), bool), prefix="stay_still.")
    common = {
        "tokens": np.array(toks), "scenario_type": np.array(stypes), "ego_gt": GT.astype(np.float32),
        "agent_token_idx": tok_idx, "agent_slot": cat("slot", pops).astype(np.int64),
        "agent_p0": cat("p0", pops).astype(np.float64), "agent_fut": cat("fut", pops).astype(np.float32),
        "agent_mask": cat("mask", pops).astype(bool), "agent_is_moving": all_moving,
        "agent_t0_speed": cat("t0_speed", metas), "agent_dist_t0": cat("dist_t0", metas),
        "agent_is_t0_observable": cat("is_t0_observable", metas).astype(bool),
        "agent_n_valid_frames": cat("n_valid_frames", metas).astype(np.int64),
        "agent_disp_gt": cat("disp_gt", metas), "stay_still_agent_ade": ss_ade, "stay_still_agent_fde": ss_fde,
    }

    dev = torch.device(args.device)
    rows, per_seed = [], {}
    for seed, ck in ckpts.items():
        model = build_model_for(args.model, adapter)
        if ck:
            adapter.load_weights(model, ck)
        model = model.to(dev).eval()
        t0 = time.time()
        preds: List[Dict] = []
        with torch.no_grad():
            for i in range(0, len(payloads), args.batch_size):
                preds += adapter.predict(model, payloads[i: i + args.batch_size], dev)
        ego_ade, ego_fde = lib.masked_errors(np.stack([p["ego_top1"] for p in preds]), GT, np.ones(GT.shape[:2], bool))
        ego_all = np.stack([p["ego_all"] for p in preds]).astype(np.float64)          # [N, K, 60, 2]
        K = ego_all.shape[1]
        ade_all = np.stack([lib.masked_errors(ego_all[:, k], GT, np.ones(GT.shape[:2], bool))[0] for k in range(K)], 1)
        fde_all = np.stack([lib.masked_errors(ego_all[:, k], GT, np.ones(GT.shape[:2], bool))[1] for k in range(K)], 1)
        row = {"seed": seed, "checkpoint": ck, **lib.summarize_ego(ego_ade, ego_fde),
               "ego.num_modes": int(K), "ego.minADE": float(ade_all.min(1).mean()), "ego.minFDE": float(fde_all.min(1).mean())}
        rec = {"ego_ade": ego_ade, "ego_fde": ego_fde, "ego_all": ego_all.astype(np.float32),
               "ego_prob": np.stack([p["ego_prob"] for p in preds]).astype(np.float32),
               "ego_top1_idx": np.array([p["ego_top1_idx"] for p in preds], np.int64),
               "ego_ade_all": ade_all, "ego_fde_all": fde_all}
        if adapter.has_agents:
            a_pair, a_ade, a_fde, a_pred, disagree = [], [], [], [], 0
            for pop, pr in zip(pops, preds):
                pair = lib.pair_agents(pop["p0"], pr["agent_p0"])
                ok = pair >= 0
                pred = np.zeros_like(pop["fut"])
                pred[ok] = pr["agent_pred"][pair[ok]]
                ade, fde = lib.masked_errors(pred, pop["fut"], pop["mask"])
                disagree += int((pop["mask"][ok] & ~pr["agent_valid"][pair[ok]]).sum())
                a_pair.append(ok); a_ade.append(ade); a_fde.append(fde); a_pred.append(pred)
            a_pair, a_ade, a_fde = np.concatenate(a_pair), np.concatenate(a_ade), np.concatenate(a_fde)
            row.update(lib.summarize_agents(a_ade, a_fde, all_moving, a_pair))
            row["agent_mask_disagree_frames"] = disagree
            rec.update({"agent_paired": a_pair, "agent_ade": a_ade, "agent_fde": a_fde,
                        "agent_pred": np.concatenate(a_pred).astype(np.float32)})
        row["sec"] = time.time() - t0
        logger.info("seed %s｜%s", seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)
        per_seed[seed] = rec
        del model

    keys = [k for k in rows[0] if k.endswith(("ADE", "FDE", "missRate", "pairing_rate"))]
    out = {
        "model": args.model, **fw, "n_tokens": len(toks), "n_split_tokens": len(lib.split_tokens(args.split)),
        "smoke_random_init": bool(args.random_init), "feature_dir": str(args.feature_dir), "device": args.device,
        "batch_size": args.batch_size, "per_seed_rows": rows, "agg": lib.aggregate_seeds(rows, keys),
        "reference": stay_still, "scenario_type_hist": {t: stypes.count(t) for t in sorted(set(stypes))},
        "builder_kw": OFFICIAL_BUILDER_KW if args.model == "plantf_official32" else None,
        "dump_keys": sorted(list(common) + [f"{k}_seed{s}" for s, r in per_seed.items() for k in r]),
        "caliber": [
            "窗口 60 点 / t0+0.1 … t0+6.0 s；我方口径 = 预测减自己的首点（基线原始输出锚在 t0，我方 GT 缺 t0→0.1 s 段）",
            "周车：PlanTF 辅助头，category==1、t0 几何一对一配对（≤1.0 m），我方 AGENT_FUTURES 与我方 mask；按行微平均",
            "moving/static 是 GT 分层，仅诊断；test 分层只能用 agent_is_t0_observable（只读 t0）",
            "plantf_official32：训练口径 max_agents=32（D3 = (a)），超出最近 32 个的周车不预测 ⇒ 配对率 < 1",
            "三 seed 只报 sd 不做 CI（H-1）；CI 只来自 token 维配对 log-cluster bootstrap",
        ],
    }
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "per_token.npz", **common, **{f"{k}_seed{s}": v for s, r in per_seed.items() for k, v in r.items()})
    (dst / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    logger.info("✅ 产物 %s", dst / "summary.json")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--stage", choices=("features", "eval"), required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--open_test_once", action="store_true", help="冻结 test 专用开关（D4 守卫传入）")
    ap.add_argument("--feature_dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpts", default="", help="seed:绝对路径,…（必填）")
    ap.add_argument("--n_tokens", type=int, default=0, help="只取前 N 个 token（仅 val 冒烟 / 回归）")
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--log_shard_index", type=int, default=0)
    ap.add_argument("--log_shard_total", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.split == "test" and (args.n_tokens or args.random_init):
        raise SystemExit("🔴 冻结 test 不接受 --n_tokens / --random_init（只开一次，必须全量真权重）")
    if (args.random_init or args.n_tokens) and not (args.feature_dir and (args.out or args.stage == "features")):
        raise SystemExit("冒烟 / 回归须显式给 --feature_dir（eval 另需 --out），不得写默认目录")
    args.feature_dir = args.feature_dir or str(FEATURE_ROOT[args.split] / args.model)
    args.out = args.out or str(OUT_ROOT / args.split / args.model)
    adapter = make_adapter(args.model)
    (stage_features if args.stage == "features" else stage_eval)(args, adapter)


if __name__ == "__main__":
    main()
