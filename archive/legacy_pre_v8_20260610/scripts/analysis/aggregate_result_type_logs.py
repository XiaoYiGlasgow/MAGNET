#!/usr/bin/env python3
"""遍历所有以 results_type 开头的文件夹, 汇总其下 training_* 子目录 logs/training_log.txt 指标.
输出: experiments/analysis/result_type_training_logs_summary.csv 与 .txt
"""
from __future__ import annotations
import os
import re
import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

ROOT_DIR = config.ALL_RESULTS_DIR
RESULT_PREFIX = "results_type"
OUTPUT_DIR = config.ANALYSIS_RESULTS_DIR
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "result_type_training_logs_summary.csv")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "result_type_training_logs_summary.txt")

EPOCH_LINE_RE = re.compile(r"Epoch\s+(\d+)\/(\d+):\s*Train=([0-9.]+),\s*Val=([0-9.]+),\s*ADE=([0-9.]+)m,\s*FDE=([0-9.]+)m", re.IGNORECASE)
LR_RE = re.compile(r"Initial learning rate:\s*([0-9.eE+-]+)")
DEVICE_RE = re.compile(r"Device:\s*(\S+)")
PATIENCE_RE = re.compile(r"Early stopping patience:\s*(\d+)")
TOTAL_EPOCH_RE = re.compile(r"Total epochs:\s*(\d+)")
TRAIN_SAMPLES_RE = re.compile(r"Train samples:\s*(\d+)")
VAL_SAMPLES_RE = re.compile(r"Val samples:\s*(\d+)")
ADJ_INLINE_RE = re.compile(r"Adjacency config:\s*(\w+)")

@dataclass
class RunSummary:
    scenario_type: str
    adjacency: str
    run_id: str
    dataset_size: str
    batch_size: str
    learning_rate: Optional[float]
    device: Optional[str]
    patience: Optional[int]
    total_epochs_declared: Optional[int]
    epochs_completed: int
    best_val_epoch: int
    best_val_loss: float
    best_val_ADE: float
    best_val_FDE: float
    final_train_loss: float
    final_val_loss: float
    final_ADE: float
    final_FDE: float
    train_samples: Optional[int]
    val_samples: Optional[int]
    log_path: str


def parse_training_dir_name(dir_name: str) -> Tuple[str, str, str, str]:
    parts = dir_name.split('_')
    ds = next((p for p in parts if p.startswith('ds')), '')
    bs = next((p for p in parts if p.startswith('bs')), '')
    lr = next((p for p in parts if p.startswith('lr')), '')
    run_id = parts[1] if len(parts) > 1 else dir_name
    return run_id, ds.replace('ds', ''), bs.replace('bs', ''), lr.replace('lr', '')


def extract_scenario_adjacency(results_folder_name: str) -> Tuple[str, str]:
    # results_type_<scenario_tokens>_<adjacency>_adjacency
    core = results_folder_name[len(RESULT_PREFIX + '_'):]
    tokens = core.split('_')
    if len(tokens) < 2:
        return core, ''
    if tokens[-1] == 'adjacency':
        adjacency = tokens[-2]  # strip suffix
        scenario_tokens = tokens[:-2]
    else:
        adjacency = tokens[-1]
        scenario_tokens = tokens[:-1]
    scenario_type = '_'.join(scenario_tokens)
    return scenario_type, adjacency


def parse_log_file(path: str, scenario_type: str, adjacency: str) -> Optional[RunSummary]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return None

    learning_rate: Optional[float] = None
    device: Optional[str] = None
    patience: Optional[int] = None
    total_epochs_declared: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None

    epochs: List[Tuple[int, float, float, float, float]] = []

    for line in lines:
        if learning_rate is None:
            m = LR_RE.search(line)
            if m:
                try:
                    learning_rate = float(m.group(1))
                except ValueError:
                    pass
        if device is None:
            m = DEVICE_RE.search(line)
            if m:
                device = m.group(1)
        if patience is None:
            m = PATIENCE_RE.search(line)
            if m:
                patience = int(m.group(1))
        if total_epochs_declared is None:
            m = TOTAL_EPOCH_RE.search(line)
            if m:
                total_epochs_declared = int(m.group(1))
        if train_samples is None:
            m = TRAIN_SAMPLES_RE.search(line)
            if m:
                train_samples = int(m.group(1))
        if val_samples is None:
            m = VAL_SAMPLES_RE.search(line)
            if m:
                val_samples = int(m.group(1))
        if adjacency == '':
            m = ADJ_INLINE_RE.search(line)
            if m:
                adjacency = m.group(1)
        m = EPOCH_LINE_RE.search(line)
        if m:
            ep = int(m.group(1))
            train_loss = float(m.group(3))
            val_loss = float(m.group(4))
            ade = float(m.group(5))
            fde = float(m.group(6))
            epochs.append((ep, train_loss, val_loss, ade, fde))

    if not epochs:
        return None

    best = min(epochs, key=lambda r: r[2])
    best_val_epoch, best_train_loss_at_best, best_val_loss, best_ade, best_fde = best
    final_ep, final_train_loss, final_val_loss, final_ade, final_fde = epochs[-1]

    run_dir = os.path.dirname(os.path.dirname(path))
    run_folder = os.path.basename(run_dir)
    run_id, dataset_size, batch_size, lr_str = parse_training_dir_name(run_folder)
    if learning_rate is None:
        try:
            learning_rate = float(lr_str)
        except ValueError:
            learning_rate = None

    return RunSummary(
        scenario_type=scenario_type,
        adjacency=adjacency,
        run_id=run_id,
        dataset_size=dataset_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        patience=patience,
        total_epochs_declared=total_epochs_declared,
        epochs_completed=final_ep,
        best_val_epoch=best_val_epoch,
        best_val_loss=best_val_loss,
        best_val_ADE=best_ade,
        best_val_FDE=best_fde,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_ADE=final_ade,
        final_FDE=final_fde,
        train_samples=train_samples,
        val_samples=val_samples,
        log_path=path,
    )


def collect_runs(root: str) -> List[RunSummary]:
    runs: List[RunSummary] = []
    for entry in os.listdir(root):
        if not entry.startswith(RESULT_PREFIX):
            continue
        result_dir = os.path.join(root, entry)
        if not os.path.isdir(result_dir):
            continue
        scenario_type, adjacency = extract_scenario_adjacency(entry)
        for sub in os.listdir(result_dir):
            if not sub.startswith("training_"):
                continue
            log_path = os.path.join(result_dir, sub, "logs", "training_log.txt")
            if os.path.isfile(log_path):
                summary = parse_log_file(log_path, scenario_type, adjacency)
                if summary:
                    runs.append(summary)
    return runs


def write_csv(runs: List[RunSummary], path: str) -> None:
    if not runs:
        return
    fieldnames = list(asdict(runs[0]).keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            w.writerow(asdict(r))


def write_txt(runs: List[RunSummary], path: str) -> None:
    lines: List[str] = []
    grouped: Dict[Tuple[str, str], List[RunSummary]] = {}
    for r in runs:
        grouped.setdefault((r.scenario_type, r.adjacency), []).append(r)
    for (scenario, adj), items in sorted(grouped.items()):
        lines.append(f"=== {scenario} | {adj} ===")
        for r in sorted(items, key=lambda x: x.run_id):
            lines.append(
                f"run={r.run_id} epochs={r.epochs_completed} best_epoch={r.best_val_epoch} "
                f"best_val={r.best_val_loss:.4f} best_ADE/FDE={r.best_val_ADE:.2f}/{r.best_val_FDE:.2f} "
                f"final_val={r.final_val_loss:.4f} final_ADE/FDE={r.final_ADE:.2f}/{r.final_FDE:.2f} lr={r.learning_rate} bs={r.batch_size} ds={r.dataset_size}" )
        lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = collect_runs(ROOT_DIR)
    if not runs:
        print("未找到日志")
        return
    write_csv(runs, OUTPUT_CSV)
    write_txt(runs, OUTPUT_TXT)
    print(f"汇总完成: {len(runs)} 条运行记录")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"TXT: {OUTPUT_TXT}")

if __name__ == '__main__':
    main()
