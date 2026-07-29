"""D0#4 + D0#5：feature schema v2 的声明式定义 + 统一 result JSON 清单（阶段15 Z7）。

这个模块是**唯一真相源**，同时服务三处：

  D0#4  feature schema v2 冻结 —— `FEATURE_SCHEMA_V2` + `schema_hash()`，hash 进每份结果 JSON；
  D0#5  provenance —— `build_result_manifest()` 产出「单文件可复原一次 run」的清单；
  Z11   闭环 provenance —— `scripts/eval_closed_loop/` 的 `<tag>.provenance.json` 复用同一份实现。

为什么要 schema hash：col6/7/8 的语义**按 node type 分岔**（lane 上是 route 三通道，
ego/agent 上仍是 timestamp/v_start/v_end）。这不是注释问题 —— 一份 pkl 是 v1 还是 v2、
一个 checkpoint 见过的是哪种语义，事后无法从数据本身看出来。把 schema 的规范化序列化取 hash
写进结果 JSON，才能在半年后回答「这张表用的是哪套输入契约」。

用法
----
    python3 scripts/analysis/result_manifest.py --print_schema      # 人读的 schema 表
    python3 scripts/analysis/result_manifest.py --self_test         # 自检
    from result_manifest import build_result_manifest, schema_hash  # 代码里调用
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "feature_schema_v2"

# ---------------------------------------------------------------------------
# feature schema v2 —— 与 `data_utils.encoding_vectornet_features` 的实际 hstack 顺序逐列对齐
# （ego: :826-837，agent: :869-878，lane: :904-915）。改代码必须同步改这里，hash 会跟着变。
# ---------------------------------------------------------------------------
FEATURE_SCHEMA_V2: Dict[str, object] = {
    "version": SCHEMA_VERSION,
    "n_cols": 12,
    "dtype": "float32",
    "frame": "ego-centric; 原点=ego 当前 rear_axle 位姿; x=ego heading(前); y=左",
    "row_semantics": "一行 = 一条 polyline 的一个线段（相邻两点），非一个时刻",
    "columns": [
        {"idx": 0, "name": "x_start", "all_types": "线段起点 x", "transform": "rigid"},
        {"idx": 1, "name": "y_start", "all_types": "线段起点 y", "transform": "rigid"},
        {"idx": 2, "name": "x_end", "all_types": "线段终点 x", "transform": "rigid"},
        {"idx": 3, "name": "y_end", "all_types": "线段终点 y", "transform": "rigid"},
        {"idx": 4, "name": "heading_start", "all_types": "起点朝向 (rad)", "transform": "heading"},
        {"idx": 5, "name": "heading_end", "all_types": "终点朝向 (rad)", "transform": "heading"},
        # ---- col6/7/8：**语义按 node type 分岔**，这是 v2 相对 v1 的唯一实质变化 ----
        {"idx": 6, "name": "ts_or_route_proximity", "transform": "invariant",
         "ego": "归一化 timestamp ∈ [0,1)",
         "agent": "归一化 timestamp ∈ [0,1)",
         "lane": "★v2: route 邻近度 = clip(1 - d/map_radius, 0, 1)；on-route lane 恒为 1.0"},
        {"idx": 7, "name": "vstart_or_route_progress", "transform": "invariant",
         "ego": "v_start (m/s)", "agent": "v_start (m/s)",
         "lane": "★v2: 沿 route 归一化进度 ∈ [0,1]；off-route lane = 0"},
        {"idx": 8, "name": "vend_or_is_on_route", "transform": "invariant",
         "ego": "v_end (m/s)", "agent": "v_end (m/s)",
         "lane": "★v2: is_on_route ∈ {0,1}"},
        {"idx": 9, "name": "subgraph_id", "all_types": "子图 ID（cluster），gnn_x 按它均值池化",
         "transform": "invariant"},
        {"idx": 10, "name": "t_start", "transform": "invariant",
         "ego": "相对 T_ref 的起点时间 (s)", "agent": "相对 T_ref 的起点时间 (s)",
         "lane": "恒 0（占位）"},
        {"idx": 11, "name": "t_end", "transform": "invariant",
         "ego": "相对 T_ref 的终点时间 (s)", "agent": "相对 T_ref 的终点时间 (s)",
         "lane": "恒 0（占位）"},
    ],
    "route_mode": {
        "none": "col6/7/8 在 lane 行上**全 0** —— 与 route 引入前的产出逐字节一致",
        "three_channel": "col6/7/8 在 lane 行上承载 route 三通道；ego/agent 行不变",
    },
    "model_input_slice": "模型 `_extract_core_features` 只取前 9 列 -> route 三通道经 Transformer 分支逐样本生效",
    "known_limits": [
        "route 三通道是**复用 lane 节点的占位通道**，不是独立 route 分支；论文不得写成 reference-line 条件化",
        "route 信号为空的样本约 11.8%（95%CI [10.7%, 13.0%]），源自 devkit prune_route_by_connectivity 的半径裁剪",
        "约 19.9% 样本 lane 数 > LANE_NUM=40 被按距离截断；截断对 route 通道无影响但应进 limitation",
        "EGO_HISTORY 是 [20,2] 局部逐帧增量，2s@10Hz —— 论文不得写 4s history",
    ],
    "contract_constants": {
        "NUM_AGENTS": 20, "LANE_NUM": 40, "LANE_POINTS": 50,
        "MAP_FEATURES": ["LANE"], "QUERY_RADIUS": 50,
        "PAST_TIME_HORIZON": 2, "NUM_PAST_POSES": 20, "NUM_FUTURE_POSES": 60,
        "nearby_only_threshold_m": 30,
    },
}


def schema_hash(schema: Optional[Dict] = None) -> str:
    """schema 的规范化 SHA256（前 16 位）。键排序 + 紧凑分隔符 => 与书写顺序无关。"""
    blob = json.dumps(schema or FEATURE_SCHEMA_V2, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- provenance 原语


def sha256_file(path, chunk: int = 1 << 20) -> Optional[str]:
    """文件 SHA256。找不到返回 None —— 清单里宁可写 null，也不能写一个猜的值。"""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git %s 失败: %s", " ".join(args), exc)
        return None


def git_provenance(repo: Optional[Path] = None) -> Dict[str, object]:
    """commit + 工作区脏文件数。

    `dirty_files` 不为 0 意味着「commit 无法完整还原这次 run」—— 这正是 D0#2 要消除的状态。
    正式 run 起跑前必须让它为 0，或至少把脏文件列进清单。
    """
    repo = Path(repo or Path(__file__).resolve().parents[2])
    status = _git(["status", "--porcelain"], repo)
    dirty = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "repo": str(repo),
        "commit": _git(["rev-parse", "HEAD"], repo),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty_files": len(dirty),
        "dirty_sample": [line[3:] for line in dirty[:20]],
        "clean": len(dirty) == 0,
    }


def env_provenance() -> Dict[str, object]:
    """Python / torch / CUDA / GPU 快照。torch 不可用时不抛异常，只记 null。"""
    info: Dict[str, object] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })
    except Exception as exc:  # noqa: BLE001
        info["torch"] = None
        info["torch_error"] = str(exc)
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def dir_manifest_hash(root, pattern: str = "*.pkl", manifest_name: str = "log_manifest.json") -> Dict:
    """数据目录指纹。

    优先对 `log_manifest.json`（token 清单）取 hash —— 它才是"用了哪批样本"的真相源，
    而且不随文件系统顺序变化。没有 manifest 时退回「文件数 + 排序后文件名的 hash」，
    并在返回值里标明用了哪种口径，绝不让两种口径混在同一个字段里。
    """
    root = Path(root)
    if not root.is_dir():
        return {"root": str(root), "exists": False}
    manifest = root / manifest_name
    if manifest.is_file():
        return {"root": str(root), "exists": True, "basis": manifest_name,
                "manifest_sha256": sha256_file(manifest),
                "n_files": sum(1 for _ in root.glob(pattern))}
    names = sorted(p.name for p in root.glob(pattern))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return {"root": str(root), "exists": True, "basis": f"sorted({pattern}) names",
            "names_sha256": digest, "n_files": len(names)}


def devkit_patch_state() -> Dict[str, object]:
    """受管 devkit 文件（在线 planner / 父类模型 / planner yaml）的同步状态。

    `nuplan-devkit` 是零提交仓库，那三个文件不受任何 commit 保护。方案 A 把它们 vendor 进
    GNN_Trans 并强制逐字节一致 —— 本函数把「一致与否 + 各自 SHA256」写进结果清单，
    使一次 run 的**两侧代码**都可追溯，而不只是 GNN_Trans 这一侧。
    """
    patch_dir = Path(__file__).resolve().parents[2] / "devkit_patches"
    script = patch_dir / "sync_devkit_patches.py"
    manifest = patch_dir / "manifest.json"
    if not script.is_file():
        return {"available": False, "reason": f"缺 {script}"}
    try:
        proc = subprocess.run([sys.executable, str(script), "--check"],
                              capture_output=True, text=True, timeout=60)
        in_sync = proc.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}
    files = None
    if manifest.is_file():
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                files = json.load(fh).get("files")
        except (OSError, json.JSONDecodeError):
            files = None
    return {"available": True, "in_sync": in_sync, "files": files,
            "note": None if in_sync else "devkit 与 vendor 副本不一致 —— 本次 run 的在线代码不可追溯"}


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


# --------------------------------------------------------------------------- 统一清单


REQUIRED_KEYS = (
    "schema_version", "schema_hash", "data_manifest", "split_manifest", "route_mode",
    "perturbation", "seed", "n_parameters", "git", "devkit_patches", "env", "command",
    "checkpoint",
)


def build_result_manifest(
    *,
    route_mode: str,
    seed: int,
    data_root,
    split_manifest_path=None,
    checkpoint_path=None,
    perturbation: Optional[Dict] = None,
    n_parameters: Optional[Dict[str, int]] = None,
    metrics: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    command: Optional[List[str]] = None,
) -> Dict:
    """产出一次 run 的完整 provenance 清单（D0#5 的「任一 formal run 可单文件复原」）。

    :param route_mode: ``none`` | ``three_channel``。
    :param seed: 实验 seed（扰动 seed 若不同须放进 ``perturbation``）。
    :param data_root: 训练/评测数据目录，用于算 data manifest hash。
    :param split_manifest_path: `split3_manifest.json`，划分的唯一真相源。
    :param checkpoint_path: 结果对应的 checkpoint；会算 SHA256。
    :param perturbation: 扰动配置字典（含 enabled/prob/max_*/seed），关闭时传 ``{"enabled": False}``。
    :param metrics: 该 run 的指标。**必须是 saved-checkpoint 口径**，不得放训练曲线 minima。
    """
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "data_manifest": dir_manifest_hash(data_root),
        "split_manifest": ({"path": str(split_manifest_path),
                            "sha256": sha256_file(split_manifest_path)}
                           if split_manifest_path else None),
        "route_mode": route_mode,
        "perturbation": perturbation or {"enabled": False},
        "seed": seed,
        "n_parameters": n_parameters,
        "git": git_provenance(),
        "devkit_patches": devkit_patch_state(),
        "env": env_provenance(),
        "command": command if command is not None else list(sys.argv),
        "checkpoint": ({"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
                       if checkpoint_path else None),
        "metrics": metrics,
        "metrics_caliber": "saved-checkpoint (checkpoint_best.pt); 训练曲线独立 minima 禁用",
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def validate_manifest(manifest: Dict) -> List[str]:
    """返回问题列表（空 = 通过）。给 D0 决策会当 checklist 用。"""
    problems = [f"缺字段: {key}" for key in REQUIRED_KEYS if key not in manifest]
    if manifest.get("schema_hash") and manifest["schema_hash"] != schema_hash():
        problems.append(f"schema_hash 与当前代码不符（清单 {manifest['schema_hash']} vs "
                        f"当前 {schema_hash()}）—— 输入契约变过，结果不可直接并表")
    git = manifest.get("git") or {}
    if git.get("dirty_files"):
        problems.append(f"git 工作区有 {git['dirty_files']} 个脏文件 —— 「命令 + commit」无法还原此 run")
    if manifest.get("checkpoint") and not manifest["checkpoint"].get("sha256"):
        problems.append("checkpoint 路径存在但 SHA256 为 null —— 文件缺失或不可读")
    patches = manifest.get("devkit_patches") or {}
    if not patches.get("available"):
        problems.append(f"devkit 受管文件状态不可用（{patches.get('reason')}）—— "
                        f"在线 planner / 父类模型的版本无法追溯")
    elif not patches.get("in_sync"):
        problems.append("devkit 生效文件与 vendor 副本不一致 —— GNN_Trans 的 commit "
                        "还原不出这次 run 的在线代码（方案 A 的核心保证被打破）")
    if manifest.get("route_mode") not in ("none", "three_channel"):
        problems.append(f"非法 route_mode: {manifest.get('route_mode')!r}")
    return problems


def write_manifest(manifest: Dict, out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return out


# --------------------------------------------------------------------------- CLI


def print_schema() -> None:
    s = FEATURE_SCHEMA_V2
    print("=" * 92)
    print(f" {s['version']}   hash={schema_hash()}   n_cols={s['n_cols']}  dtype={s['dtype']}")
    print(f" 坐标系: {s['frame']}")
    print(f" 行语义: {s['row_semantics']}")
    print("=" * 92)
    print(f" {'col':<4}{'name':<26}{'ego/agent':<34}{'lane'}")
    print("-" * 92)
    for col in s["columns"]:
        shared = col.get("all_types")
        ego = shared or col.get("ego", "")
        lane = shared or col.get("lane", "")
        print(f" {col['idx']:<4}{col['name']:<26}{ego[:33]:<34}{lane}")
    print("-" * 92)
    for mode, desc in s["route_mode"].items():
        print(f" route_mode={mode:<15} {desc}")
    print(f" 模型输入: {s['model_input_slice']}")
    print("\n 已知边界（写进论文 limitation）:")
    for line in s["known_limits"]:
        print(f"   - {line}")
    print("=" * 92)


def self_test() -> int:
    print("=" * 70)
    print(" result_manifest.py —— 自检")
    print("=" * 70)
    ok = True

    h1, h2 = schema_hash(), schema_hash()
    print(f"  {'✅' if h1 == h2 else '❌'} schema_hash 稳定: {h1}")
    ok &= h1 == h2

    mutated = json.loads(json.dumps(FEATURE_SCHEMA_V2))
    mutated["columns"][8]["lane"] = "篡改"
    changed = schema_hash(mutated) != h1
    print(f"  {'✅' if changed else '❌'} schema 变动后 hash 改变（col8 语义敏感）")
    ok &= changed

    reordered = {k: FEATURE_SCHEMA_V2[k] for k in reversed(list(FEATURE_SCHEMA_V2))}
    stable = schema_hash(reordered) == h1
    print(f"  {'✅' if stable else '❌'} 顶层键顺序不影响 hash")
    ok &= stable

    manifest = build_result_manifest(
        route_mode="three_channel", seed=42,
        data_root=Path(__file__).resolve().parents[2] / "data" / "processed_features_magnet_e2_route_v2",
        checkpoint_path=None, perturbation={"enabled": True, "prob": 0.5},
        metrics={"val_ade": 2.048, "val_fde": 4.791}, command=["self_test"],
    )
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    print(f"  {'✅' if not missing else '❌'} 清单包含全部 {len(REQUIRED_KEYS)} 个必填字段"
          + (f"，缺 {missing}" if missing else ""))
    ok &= not missing

    problems = validate_manifest(manifest)
    dirty_only = all("脏文件" in p for p in problems)
    print(f"  {'✅' if dirty_only else '❌'} validate 只报 git dirty（本仓当前确实脏）: {problems or '无'}")
    ok &= dirty_only

    bad = dict(manifest, schema_hash="deadbeef")
    caught = any("schema_hash" in p for p in validate_manifest(bad))
    print(f"  {'✅' if caught else '❌'} validate 能抓到 schema_hash 漂移")
    ok &= caught

    print("-" * 70)
    print(" 自检" + ("通过" if ok else "失败"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="feature schema v2 + 统一 result 清单")
    p.add_argument("--print_schema", action="store_true")
    p.add_argument("--print_schema_hash", action="store_true")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--validate", type=Path, default=None, help="校验一份已有的 result JSON")
    a = p.parse_args()

    if a.print_schema_hash:
        print(schema_hash())
        return 0
    if a.print_schema:
        print_schema()
        return 0
    if a.validate:
        with open(a.validate, "r", encoding="utf-8") as fh:
            problems = validate_manifest(json.load(fh))
        print("\n".join(f"  ❌ {x}" for x in problems) if problems else "  ✅ 清单通过校验")
        return 1 if problems else 0
    if a.self_test:
        return self_test()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
