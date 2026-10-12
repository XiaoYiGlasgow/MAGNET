#!/usr/bin/env python
"""S3 边效用标签生成器的单测（`scripts/analysis/s3_edge_utility_labels.py`）。

判据
----
[A] **D2c 门的语义** —— 与 `g1_dose_ladder_probe._cut_edges_between(ei, j, others_veh)`
    切出**同一批边**；且 **j–lane 边一条不动**（这是 D2c 与 D2 的唯一区别，切错就换了剂量）。
[B] **P2c 门** —— 切随机 lane 与全部车辆的边；计数关系 `cut_p2c == cut_d2c + 2`
    （D2c 切 `2(n_veh−1)` 条有向边、P2c 切 `2·n_veh` 条）⇒ 近似 count-matched，可审计。
[C] **复制批不改图** —— 同一批里两个全 1 副本的轨迹在 CPU 上**逐位相同**（空操作门）；
    单副本 base 与复制批第 0 副本的 err 落在实测噪底内（🔴 不是逐位 —— 批形状变了 BLAS 换 kernel）。
[D] **标签口径** —— `I == relu(U − placebo)` 逐位成立；`err_metric=minade` 时
    err 等于逐模态 ADE 的**最小值**；`U` 不含 placebo（改抽样数只动 `I` 不动 `U`，容差 = 噪底）。
[E] **范围（Q109 §3.3.4）** —— 只落 `scorable ∧ valid` 的边、目标必须 `is_target`；
    🔴 **`j→ego` 的对必须存在**（否则 Q108 §A2 的主目标一条标签都没有）。
[F] **产物自洽** —— `scene_ptr` 单调且末元素 = 总对数；meta 的 `teacher_sha256` 与文件实际
    sha 一致；冻结标记写了，且**再跑一次会被拒**（`--force` 才覆盖）。

🔴 全程 CPU：`MagnetJointAdaptive` 在 GPU 上前向不可复现（§8.2），[C][D] 的逐位判据在 GPU 上必然假失败。

跑法::

    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    /home/team/anaconda3/envs/nuplan/bin/python tests/test_s3_edge_utility_labels.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN), str(_GNN / "scripts" / "training"), str(_GNN / "scripts" / "analysis"),
           str(_GNN.parent), str(_GNN.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PY = "/home/team/anaconda3/envs/nuplan/bin/python"
VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
CKPT = _GNN / "experiments/joint_s2_fixed/seed1/models/checkpoint_best.pt"
#: [C][D] 的容差 = 冒烟实测的数值噪底（复制批规模不同 ⇒ max|ΔU| 7.2e-07，0.8% 的对）。
NOISE_FLOOR = 1e-5
_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def _scene():
    import config  # noqa: F401
    from src.shared.utils.dataset import set_subgraph_agg
    set_subgraph_agg("mean")
    from magnet_agent_aux_dataset import AgentAuxGraphDataset
    return AgentAuxGraphDataset(str(VAL))[0]


def _ctx(data):
    from torch_geometric.data import Batch
    from magnet_joint_encoder import build_entity_index
    from magnet_joint_layers import build_edge_context, build_entity_geometry
    b = Batch.from_data_list([data], follow_batch=["gnn_x"])
    idx = build_entity_index(b)
    return b, idx, build_edge_context(b.gnn_edge_index, idx, build_entity_geometry(b.x, idx))


def _veh_rows(idx):
    rows = torch.nonzero(idx.entity_is_agent & idx.entity_valid, as_tuple=True)[0]
    return torch.unique(torch.cat([rows, idx.ego_index]))


# ── [A][B] 门的语义 ────────────────────────────────────────────────────────
def test_ab_gates() -> None:
    print("[A/B D2c 与 P2c 门的语义]")
    import s3_edge_utility_labels as s3
    from g1_dose_ladder_probe import _cut_edges_between
    from d2_semantic_intervention import split_directed_half
    data = _scene()
    b, idx, ectx = _ctx(data)
    veh = _veh_rows(idx)
    j = int(veh[1])                                   # 取一个非 ego 的车辆
    others = [int(v) for v in veh.tolist() if v != j]

    gate = s3.d2c_gate(ectx, j, veh)
    cut = ~gate.bool()
    ei = ectx.edge_index
    # 参照实现：g1 在「正向半边」上判定，这里换算成有向边集合比对
    fwd, _ = split_directed_half(b.gnn_edge_index)
    o = torch.as_tensor(others, dtype=fwd.dtype)
    hit_fwd = ((fwd[0] == j) & torch.isin(fwd[1], o)) | ((fwd[1] == j) & torch.isin(fwd[0], o))
    _, n_undirected = _cut_edges_between(b.gnn_edge_index, j, others)
    check(int(cut.sum()) == 2 * n_undirected,
          f"[A] D2c 切的有向边数 {int(cut.sum())} == 2 × g1 的无向边数 {n_undirected}")
    check(int(hit_fwd.sum()) == n_undirected, "[A] 与 g1 的正向半边判定一致")
    lane_touch = cut & ~ectx.scorable
    check(int(lane_touch.sum()) == 0,
          f"[A] **j–lane 边一条没切**（切了 {int(lane_touch.sum())} 条就变成 D2 剂量了）")
    ends = (ei[0, cut] == j) | (ei[1, cut] == j)
    check(bool(ends.all()), "[A] 切掉的每条边都以 j 为一端")

    lane_rows = torch.nonzero(~idx.entity_is_agent & idx.entity_valid, as_tuple=True)[0]
    pg = s3.p2c_gate(ectx, int(lane_rows[0]), veh)
    n_p, n_d = int((~pg.bool()).sum()), int(cut.sum())
    check(n_p == n_d + 2, f"[B] P2c 切 {n_p} == D2c 切 {n_d} + 2（近似 count-matched，可审计）")
    check(int((~pg.bool() & ectx.scorable).sum()) == 0, "[B] P2c 只切 lane↔车边，不动车↔车边")


# ── [C][D][E] 端到端：真 teacher + 真样本 ─────────────────────────────────
def test_cde_labels() -> None:
    print("[C/D/E 复制批、标签口径、标签范围]")
    import s3_edge_utility_labels as s3
    from src.shared.utils.bn_calibration import load_model_for_inference
    from magnet_joint_loss import per_mode_errors
    dev = torch.device("cpu")
    model = load_model_for_inference(CKPT, device=dev)
    data = _scene()
    b, idx, ectx = _ctx(data)
    e = int(ectx.edge_index.shape[1])
    ones = torch.ones(e)

    # [C] 空操作门：同一批里的两个全 1 副本逐位相同
    err_k, traj_k, tg_k = s3._forward_replicas(model, data, [ones, ones, ones], dev)
    d = float((traj_k[0] - traj_k[1]).abs().max())
    check(d == 0.0, f"[C] 同批两个全 1 副本轨迹逐位相同（max|Δ|={d:.1e}）")
    # 单副本 vs 复制批第 0 副本：批形状变了 ⇒ 只保证落在噪底内
    err_1, _, _ = s3._forward_replicas(model, data, [ones, ones], dev)
    d1 = float((err_k[0] - err_1[0]).abs().max())
    check(d1 <= NOISE_FLOOR, f"[C] 不同批规模下 base err 落在噪底内（max|Δ|={d1:.1e} ≤ {NOISE_FLOOR:.0e}）")

    # [D] err_metric = minade 的定义
    from torch_geometric.data import Batch
    from magnet_joint_encoder import build_entity_index
    from magnet_joint_loss import build_joint_targets_batch
    b1 = Batch.from_data_list([data], follow_batch=["gnn_x"])
    i1 = build_entity_index(b1)
    with torch.no_grad():
        pred = model(b1)
    ade, _ = per_mode_errors(pred.traj, build_joint_targets_batch(b1, i1))
    check(float((s3._err_minade(ade, pred.logits) - ade.min(dim=1).values).abs().max()) == 0.0,
          "[D] minade == 逐模态 ADE 的最小值")
    check(float((s3._err_top1(ade, pred.logits)
                 - ade[torch.arange(ade.shape[0]), pred.logits.argmax(1)]).abs().max()) == 0.0,
          "[D] top1 == argmax π 那一模态的 ADE")

    rec = s3.label_scene(model, data, dev, np.random.default_rng(0), n_placebo=4, noop_tol=0.0)
    check("skip" not in rec, f"[D] 该场景可打标（{rec.get('skip', '')}）")
    check(float(np.abs(rec["I"] - np.maximum(rec["U"] - rec["placebo"], 0.0)).max()) == 0.0,
          "[D] I == relu(U − placebo) 逐位成立")
    rec8 = s3.label_scene(model, data, dev, np.random.default_rng(0), n_placebo=8, noop_tol=0.0)
    check(float(np.abs(rec["U"] - rec8["U"]).max()) <= NOISE_FLOOR,
          f"[D] U 不含 placebo（改抽样数后 max|ΔU|={float(np.abs(rec['U'] - rec8['U']).max()):.1e}）")

    # [E] 范围
    tgt = torch.nonzero(build_joint_targets_batch(b1, i1).is_target, as_tuple=True)[0].numpy()
    veh = _veh_rows(idx).numpy()
    check(bool(np.isin(rec["dst"], tgt).all()), "[E] 全部目标 i 都是监督实体（is_target）")
    check(bool(np.isin(rec["src"], veh).all()), "[E] 全部源 j 都是车辆节点")
    ego = int(idx.ego_index[0])
    n_ego = int((rec["dst"] == ego).sum())
    check(n_ego > 0, f"[E] 🔴 存在 j→ego 的标签（{n_ego} 条）—— Q108 §A2 的主目标")
    n_scor = int((ectx.scorable & ectx.valid).sum())
    check(rec["src"].size <= n_scor, f"[E] 标签对数 {rec['src'].size} ≤ scorable∧valid 边数 {n_scor}")


# ── [H] 故障注入：门必须能抓到最小结构损坏 ──────────────────────────────
def test_h_mutation() -> None:
    """[H] 把「本该全 1」的副本改成**只切 1 条**车↔车边（最小可能的结构损坏），门必须读出远超容差的值。

    这条判据回答的是「容差 1e-3 会不会让门形同虚设」—— 不靠推理，靠实测分离度。
    """
    print("[H 故障注入：门的判别力]")
    import s3_edge_utility_labels as s3
    from src.shared.utils.bn_calibration import load_model_for_inference
    from magnet_joint_encoder import build_entity_index
    from magnet_joint_layers import build_edge_context, build_entity_geometry
    from torch_geometric.data import Batch
    dev = torch.device("cpu")
    model = load_model_for_inference(CKPT, device=dev)
    data = _scene()
    b1 = Batch.from_data_list([data], follow_batch=["gnn_x"]).to(dev)
    idx = build_entity_index(b1)
    ectx = build_edge_context(b1.gnn_edge_index, idx, build_entity_geometry(b1.x, idx))
    ones = torch.ones(int(ectx.edge_index.shape[1]))
    broken = ones.clone()
    broken[torch.nonzero(ectx.scorable & ectx.valid, as_tuple=True)[0][0]] = 0.0
    _, traj, _ = s3._forward_replicas(model, data, [ones, broken], dev)
    hit = float((traj[0] - traj[1]).abs().max())
    check(hit > 100 * s3.GPU_NOOP_TOL,
          f"[H] 只切 1 条边 ⇒ 门读 {hit:.3e} m，是 GPU 容差 {s3.GPU_NOOP_TOL:.0e} 的 {hit / s3.GPU_NOOP_TOL:.0f}×")
    check(s3.GPU_NOOP_TOL >= 1e-4, f"[H] GPU 容差 {s3.GPU_NOOP_TOL:.0e} 高于实测浮点尾部 1.53e-04 的量级")


# ── [G] 多 teacher 聚合 ──────────────────────────────────────────────────
def test_g_multi_teacher() -> None:
    """[G] 多 teacher：U 取跨 seed **中位数**；单 teacher 时 `U_sd_teacher` 恒 0；重复 ckpt 被拒。"""
    print("[G 多 teacher 跨 seed 中位数聚合]")
    import s3_edge_utility_labels as s3
    from src.shared.utils.bn_calibration import load_model_for_inference
    dev = torch.device("cpu")
    ck = [_GNN / f"experiments/joint_s2_fixed/seed{s}/models/checkpoint_best.pt" for s in (1, 2, 42)]
    if not all(c.exists() for c in ck):
        check(False, f"[G] 缺 S2 三 seed ckpt：{[str(c) for c in ck if not c.exists()]}")
        return
    ms = [load_model_for_inference(c, device=dev) for c in ck]
    data = _scene()
    kw = dict(device=dev, n_placebo=4, noop_tol=0.0)
    one = s3.label_scene([ms[0]], data, rng=np.random.default_rng(0), **kw)
    three = s3.label_scene(ms, data, rng=np.random.default_rng(0), **kw)
    check(np.all(one["U_sd_teacher"] == 0.0), "[G] 单 teacher 时 U_sd_teacher 恒 0")
    check(np.array_equal(one["src"], three["src"]) and np.array_equal(one["dst"], three["dst"]),
          "[G] 单/多 teacher 的边集相同（干预门只依赖图，不依赖权重）")
    per = [s3.label_scene([m], data, rng=np.random.default_rng(0), **kw)["U"] for m in ms]
    med = np.median(np.stack(per), axis=0)
    check(float(np.abs(three["U"] - med).max()) <= NOISE_FLOOR,
          f"[G] 三 teacher 的 U == 逐 teacher U 的中位数（max|Δ|={float(np.abs(three['U'] - med).max()):.1e}）")
    sd = np.std(np.stack(per), axis=0)
    check(float(np.abs(three["U_sd_teacher"] - sd).max()) <= NOISE_FLOOR,
          "[G] U_sd_teacher == 逐 teacher U 的标准差（跨 teacher 分歧可审计）")
    check(float(sd.max()) > 0, f"[G] 三个 teacher 确实不一致（max sd = {float(sd.max()):.4f}）—— 判据非平凡")
    r = subprocess.run([PY, str(_GNN / "scripts/analysis/s3_edge_utility_labels.py"),
                        "--teacher", str(ck[0]), str(ck[0]), "--root", str(VAL),
                        "--out", "/tmp/_s3_dup", "--device", "cpu", "--limit", "1"],
                       cwd=str(_GNN), capture_output=True, text=True)
    check(r.returncode != 0 and "重复" in (r.stderr + r.stdout), "[G] 重复的 teacher ckpt 被当场拒绝")


# ── [F] 产物自洽 + 冻结 ───────────────────────────────────────────────────
def test_f_artifact_and_freeze() -> None:
    print("[F 产物自洽与冻结]")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "labels"
        cmd = [PY, str(_GNN / "scripts/analysis/s3_edge_utility_labels.py"),
               "--teacher", str(CKPT), "--root", str(VAL), "--out", str(out),
               "--device", "cpu", "--limit", "3", "--shard_size", "10", "--n_placebo", "2"]
        r = subprocess.run(cmd, cwd=str(_GNN), capture_output=True, text=True)
        check(r.returncode == 0, f"[F] 首次生成 rc=0（stderr 尾: {r.stderr.strip()[-160:]}）")
        z = np.load(out / "labels_0000.npz")
        ptr = z["scene_ptr"]
        check(bool(np.all(np.diff(ptr) >= 0)) and int(ptr[-1]) == int(z["src"].size),
              f"[F] scene_ptr 单调且末元素 {int(ptr[-1])} == 总对数 {int(z['src'].size)}")
        check(z["token"].size == ptr.size - 1 and all(len(t) > 8 for t in z["token"]),
              "[F] 每个场景一个真 scenario token（不是数据集下标）")
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        check(meta["teacher_sha256"] == [hashlib.sha256(CKPT.read_bytes()).hexdigest()],
              "[F] meta 里的 teacher sha 列表与 ckpt 文件实际 sha 一致（冻结依据）")
        check((out / s3_frozen()).read_text().strip().split("\n") == meta["teacher_sha256"],
              "[F] 冻结标记内容 == meta 的 sha 列表")
        check(meta["dose"] == "D2c" and meta["placebo"] == "P2c", "[F] 剂量/对照写进 meta")
        check((out / s3_frozen()).exists(), "[F] 冻结标记已写")
        r2 = subprocess.run(cmd, cwd=str(_GNN), capture_output=True, text=True)
        check(r2.returncode != 0 and "已冻结" in (r2.stderr + r2.stdout),
              "[F] 再跑一次被拒（要 --force 才覆盖）")


def s3_frozen() -> str:
    import s3_edge_utility_labels as s3
    return s3.FROZEN_MARK


if __name__ == "__main__":
    for f in (CKPT, VAL):
        if not f.exists():
            print(f"🔴 缺 {f}")
            sys.exit(1)
    test_ab_gates()
    test_cde_labels()
    test_h_mutation()
    test_g_multi_teacher()
    test_f_artifact_and_freeze()
    if _FAILED:
        print(f"\n🔴 {len(_FAILED)} 项失败:")
        for f in _FAILED:
            print("   -", f)
        sys.exit(1)
    print("\n✅ test_s3_edge_utility_labels 5/5 通过")
