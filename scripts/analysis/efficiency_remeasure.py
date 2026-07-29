"""R1-7 / R2-8 / R4-8：效率指标重测（阶段15 Z8，协议见阶段14 §2.6）。

为什么必须重测
--------------
提交稿的效率数字有 **4 类可复原的硬伤**（阶段14 §2.1/§2.4）：

1. 正文 L1010 与 Table IV 对同一配置给出相差 **1.42×** 的吞吐（69.9 vs 99.010）；
2. Table IV 全部 20 个吞吐值都等于 `1000 / (一位小数的 ms)` —— 是**单点耗时四舍五入后取倒数**，
   不是实测吞吐。后果：High Magnitude Speed 的三行吞吐**完全相同（70.922）**，纯属量化撞车；
3. 单点测量，无重复、无 std；
4. batch size 口径未记录，而在线 planner 跑的是 batch=1、训练跑的是 batch=32，两者不可互推。

本脚本按 §2.6 协议重测，**只报 mean ± std，不报单点，不由 latency 倒推 throughput**。

协议（脚本强制，不靠自觉）
--------------------------
    独占卡    : 启动前查 nvidia-smi，有他人进程直接拒跑（--allow_shared 可强制，但会标进结果）
    预热      : 100 iters（沿用原协议）
    重复      : >= 5 轮，报 mean ± std
    计时      : torch.cuda.synchronize() 前后夹，**只计 forward**
    批量      : batch=1（在线 planner 真实口径）与 batch=32（训练口径）**分列**
    吞吐      : 端到端（含 collate），与 latency **分表**，两者不可互推
    指标      : latency(ms) / throughput(samples/s) / peak GPU mem(MB) / params / FLOPs
    样本      : 独立 test set，与精度表**同一 manifest**

用法
----
    # 接线自检（不占 GPU）
    python3 scripts/analysis/efficiency_remeasure.py --dry_run

    # 真实重测（**必须独占 GPU**；G9，排在 G3 之后）
    PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/efficiency_remeasure.py \
        --test_root /abs/.../processed_features_magnet_e1_full_split3/test \
        --checkpoint /abs/.../models/best_model_default.pth \
        --configs default,ego_only,no_ego_road,nearby_only \
        --repeats 5 --device cuda:0 \
        --out /abs/.../experiments/efficiency/efficiency_20260812.json

⚠️ **无论重测结果如何**，`.tex` L94 的 "improving computational efficiency" 贡献主张都必须删除
—— 稿内 L1012 已自认 "Reducing graph connectivity does not produce proportional latency reduction"，
贡献列表与结论段自相矛盾（阶段14 §2.6）。重测是为了给出诚实数字，不是为了救那条主张。
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT), str(_DEVKIT / "models"),
           str(_THIS.parent), str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("efficiency_remeasure")

WARMUP_ITERS = 100
MIN_REPEATS = 5
BATCH_SIZES = (1, 32)
ADJACENCY_CONFIGS = ("default", "ego_only", "no_ego_road", "nearby_only")


@dataclass
class Measurement:
    """一个 (config, batch_size) 的重复测量结果。全部字段带 std，没有单点。"""

    config: str
    batch_size: int
    n_repeats: int
    latency_ms: List[float] = field(default_factory=list)      # 每轮的平均 forward 耗时
    throughput: List[float] = field(default_factory=list)      # 每轮的端到端吞吐
    peak_mem_mb: Optional[float] = None
    n_batches: int = 0

    @staticmethod
    def _ms(values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(statistics.fmean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "n": len(values)}

    def as_dict(self) -> Dict:
        return {
            "config": self.config, "batch_size": self.batch_size,
            "n_repeats": self.n_repeats, "n_batches_per_repeat": self.n_batches,
            "latency_ms": self._ms(self.latency_ms),
            "throughput_samples_per_s": self._ms(self.throughput),
            "peak_gpu_mem_mb": self.peak_mem_mb,
            "note": "latency=仅 forward（cuda.synchronize 夹）；throughput=端到端含 collate。"
                    "两者不可互推，禁止由 1000/latency 反推 throughput。",
        }


# --------------------------------------------------------------------------- 环境守门


def gpu_exclusive(device: str) -> Dict:
    """检查目标 GPU 是否被他人占用。效率数字在共享卡上测出来没有意义。"""
    index = device.split(":")[-1] if ":" in device else "0"
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        used = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,utilization.gpu,persistence_mode",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"checked": False, "reason": str(exc), "exclusive": False}

    lines = [ln for ln in procs.stdout.strip().splitlines() if ln.strip()]
    mem_used, util, persistence = (
        [f.strip() for f in used.stdout.strip().split(",")] + ["?", "?", "?"])[:3]
    mem_mb = int(mem_used.split()[0]) if mem_used and mem_used.split()[0].isdigit() else -1
    return {
        "checked": True, "device": device, "compute_apps": lines,
        "memory_used": mem_used, "utilization": util, "persistence_mode": persistence,
        # 我们自己还没建模型，此刻显存应接近 0；> 1000 MiB 说明卡上有别人
        "exclusive": mem_mb >= 0 and mem_mb < 1000,
    }


def flops_backend() -> Optional[str]:
    for name in ("fvcore", "thop", "ptflops"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


# --------------------------------------------------------------------------- 测量


def measure(model, dataset, config: str, batch_size: int, device, repeats: int,
            max_batches: Optional[int]) -> Measurement:
    """一个 (config, batch_size) 的完整测量：预热 100 iters + repeats 轮计时。"""
    import torch
    from torch_geometric.loader import DataLoader

    from train_magnet_imitation import move_batch_to_device

    m = Measurement(config=config, batch_size=batch_size, n_repeats=repeats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = []
    for i, data in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batches.append(data)
    m.n_batches = len(batches)
    if not batches:
        raise RuntimeError(f"config={config} batch={batch_size}: 没有可测样本")

    model.eval()
    with torch.no_grad():
        for i in range(WARMUP_ITERS):                          # 预热，不计时
            model(move_batch_to_device(batches[i % len(batches)], device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(repeats):
            # --- latency：只夹 forward，逐 batch 计时后取均值 ---
            per_batch: List[float] = []
            for data in batches:
                moved = move_batch_to_device(data, device)     # 搬运不计入 latency
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                model(moved)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                per_batch.append((time.perf_counter() - t0) * 1000.0)
            m.latency_ms.append(float(statistics.fmean(per_batch)))

            # --- throughput：端到端，含 collate 与搬运 ---
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0, n = time.perf_counter(), 0
            for data in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                if max_batches is not None and n // batch_size >= max_batches:
                    break
                model(move_batch_to_device(data, device))
                n += int(data.num_graphs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            m.throughput.append(n / max(time.perf_counter() - t0, 1e-9))

        if device.type == "cuda":
            m.peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return m


def _num(value: Optional[float], nd: int) -> str:
    """缺测一律显示 `-`，绝不用 0 或估算值占位。"""
    return "-" if value is None else f"{value:.{nd}f}"


def render(rows: List[Dict], meta: Dict) -> str:
    lines: List[str] = []
    add = lines.append
    add("=" * 96)
    add(" 效率重测（阶段14 §2.6 协议）—— 全部为 mean ± std，无单点，无倒数反推")
    add(f" device={meta['device']}  repeats={meta['repeats']}  warmup={WARMUP_ITERS}"
        f"  独占={meta['gpu']['exclusive']}  FLOPs backend={meta['flops_backend'] or '未安装'}")
    add("=" * 96)

    add("")
    add("── 表 A：forward latency（ms/batch，只夹 forward）" + "─" * 40)
    add(f" {'config':<14}{'batch':<8}{'latency mean':>15}{'std':>10}{'peak mem MB':>14}")
    for r in rows:
        lat = r["latency_ms"]
        mean = _num(lat["mean"], 3)
        std = _num(lat["std"], 3)
        mem = _num(r["peak_gpu_mem_mb"], 1)
        add(f" {r['config']:<14}{r['batch_size']:<8}{mean:>15}{std:>10}{mem:>14}")

    add("")
    add("── 表 B：端到端 throughput（samples/s，含 collate）" + "─" * 36)
    add(f" {'config':<14}{'batch':<8}{'throughput mean':>18}{'std':>10}")
    for r in rows:
        thr = r["throughput_samples_per_s"]
        mean = _num(thr["mean"], 2)
        std = _num(thr["std"], 2)
        add(f" {r['config']:<14}{r['batch_size']:<8}{mean:>18}{std:>10}")

    add("")
    add(f" 参数量: {meta['n_parameters']}")
    add(" ⚠️ 表 A 与表 B **不可互推**：throughput ≠ 1000/latency（后者不含 collate 与搬运）。")
    add(" ⚠️ batch=1 是在线 planner 的真实口径，batch=32 是训练口径；两者不得混列在同一列。")
    if not meta["flops_backend"]:
        add(" ⚠️ FLOPs 未测：环境无 fvcore/thop/ptflops。要报 FLOPs 须先 `uv pip install fvcore`，")
        add("    **不得**用估算值或引用他文数字填表。")
    if not meta["gpu"]["exclusive"]:
        add(" ❌ GPU 非独占 —— 本轮数字不得进论文，需在独占卡上重跑。")
    add("=" * 96)
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def dry_run() -> int:
    print("=" * 70)
    print(" efficiency_remeasure.py —— dry run（不占 GPU）")
    print("=" * 70)
    ok = True
    for mod, label in (("train_magnet_imitation", "move_batch_to_device"),
                       ("transformer_predictor", "TransformerPredictor"),
                       ("result_manifest", "build_result_manifest")):
        try:
            m = __import__(mod)
            assert hasattr(m, label)
            print(f"  ✅ {mod}.{label} 可用")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  ❌ {mod}.{label}: {exc}")

    try:
        from src.shared.utils.dataset import get_adjacency_config
        for cfg in ADJACENCY_CONFIGS:
            get_adjacency_config(cfg)
        print(f"  ✅ 四拓扑均可解析: {', '.join(ADJACENCY_CONFIGS)}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ❌ 拓扑配置: {exc}")

    m = Measurement("default", 32, 5)
    m.latency_ms = [10.0, 10.5, 9.5, 10.2, 9.8]
    m.throughput = [100.0, 102.0, 98.0, 101.0, 99.0]
    d = m.as_dict()
    assert abs(d["latency_ms"]["mean"] - 10.0) < 1e-9 and d["latency_ms"]["std"] > 0
    assert d["throughput_samples_per_s"]["n"] == 5
    print(f"  ✅ 统计聚合: latency {d['latency_ms']['mean']:.3f}±{d['latency_ms']['std']:.3f} ms, "
          f"throughput {d['throughput_samples_per_s']['mean']:.2f}±"
          f"{d['throughput_samples_per_s']['std']:.2f} /s")

    single = Measurement("x", 1, 1)
    single.latency_ms = [10.0]
    assert single.as_dict()["latency_ms"]["std"] == 0.0
    print("  ✅ 单轮时 std=0（而非 None）—— 报表不会出现"
          "「有 mean 无 std」的旧 Table IV 形态")

    gpu = gpu_exclusive("cuda:0")
    print(f"  ℹ️ 当前 GPU0: mem={gpu.get('memory_used')} util={gpu.get('utilization')} "
          f"独占={gpu.get('exclusive')}（现在跑着 dev，本来就不独占，符合预期）")
    print(f"  ℹ️ FLOPs backend: {flops_backend() or '未安装（结果里会记 null，不编数字）'}")

    print("-" * 70)
    print(" dry run " + ("通过" if ok else "失败") + "；真实重测须等 G9（G3 之后、独占卡）。")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="效率重测（R1-7/R2-8/R4-8）")
    p.add_argument("--test_root", type=Path, default=None, help="独立 test set 目录（与精度表同一 manifest）")
    p.add_argument("--checkpoint", type=Path, default=None, help="best_model_default.pth（bare state_dict）")
    p.add_argument("--configs", default=",".join(ADJACENCY_CONFIGS))
    p.add_argument("--batch_sizes", default=",".join(str(b) for b in BATCH_SIZES))
    p.add_argument("--repeats", type=int, default=MIN_REPEATS)
    p.add_argument("--max_batches", type=int, default=None, help="每轮最多测多少 batch（调试用）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow_shared", action="store_true", help="GPU 非独占时仍强行测（结果会被标记）")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run()
    if not args.test_root or not args.checkpoint:
        raise SystemExit("--test_root 与 --checkpoint 必填（除非 --dry_run）")
    if args.repeats < MIN_REPEATS:
        raise SystemExit(f"--repeats 至少 {MIN_REPEATS}（阶段14 §2.6 协议），当前 {args.repeats}")

    import torch

    from result_manifest import build_result_manifest, count_parameters, write_manifest
    from src.shared.utils.dataset import GraphDataset, get_adjacency_config
    from transformer_predictor import TransformerPredictor

    gpu = gpu_exclusive(args.device)
    if not gpu["exclusive"] and not args.allow_shared:
        raise SystemExit(f"GPU 非独占（{gpu}）—— 效率数字在共享卡上无意义。"
                         f"等卡空出，或显式 --allow_shared（结果会被标记为不可进论文）。")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    n_params = count_parameters(model)

    rows: List[Dict] = []
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        get_adjacency_config(cfg)                      # 设全局，GraphDataset 据此选 edge builder
        dataset = GraphDataset(str(args.test_root), transform=None)
        for bs in [int(b) for b in args.batch_sizes.split(",") if b.strip()]:
            logger.info("测量 config=%s batch=%d", cfg, bs)
            rows.append(measure(model, dataset, cfg, bs, device, args.repeats,
                                args.max_batches).as_dict())

    meta = {"device": str(device), "repeats": args.repeats, "warmup_iters": WARMUP_ITERS,
            "gpu": gpu, "flops_backend": flops_backend(), "n_parameters": n_params,
            "flops": None, "flops_reason": None if flops_backend() else "无 fvcore/thop/ptflops，未测"}
    text = render(rows, meta)
    print(text)

    if args.out:
        manifest = build_result_manifest(
            route_mode="none", seed=-1, data_root=args.test_root,
            checkpoint_path=args.checkpoint, n_parameters=n_params,
            metrics={"efficiency_rows": rows}, extra={"protocol": "阶段14 §2.6", **meta},
        )
        write_manifest(manifest, args.out)
        args.out.with_suffix(".txt").write_text(text, encoding="utf-8")
        logger.info("written: %s (+ .txt)", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
