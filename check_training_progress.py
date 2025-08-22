#!/usr/bin/env python3
"""
Enhanced Training Progress Monitor
实时监控训练进度，显示详细的训练信息
"""

import os
import json
import time
import glob
from datetime import datetime
import argparse

def find_training_folders():
    """查找所有训练文件夹"""
    training_folders = glob.glob("training_*")
    training_folders.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序
    return training_folders

def read_progress_info(folder_path):
    """读取训练进度信息"""
    progress_file = os.path.join(folder_path, 'progress.json')
    log_file = os.path.join(folder_path, 'logs', 'training_log.txt')

    progress_info = {}
    log_info = {}

    # 读取progress.json
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_info = json.load(f)
        except:
            pass

    # 读取日志文件的最后几行
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_info['total_lines'] = len(lines)
                log_info['last_lines'] = lines[-10:] if lines else []  # 最后10行
        except:
            pass

    return progress_info, log_info

def get_training_status(folder_path):
    """获取训练状态"""
    progress_file = os.path.join(folder_path, 'progress.json')
    final_results_file = os.path.join(folder_path, 'final_results.json')

    if os.path.exists(final_results_file):
        return "Completed"
    elif os.path.exists(progress_file):
        # 检查最后更新时间
        last_modified = os.path.getmtime(progress_file)
        time_diff = time.time() - last_modified
        if time_diff > 300:  # 5分钟没更新
            return "Stopped/Error"
        else:
            return "Running"
    else:
        return "Starting"

def display_progress_bar(percentage, width=50):
    """显示进度条"""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:.1f}%"

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def display_training_summary():
    """显示训练总览"""
    folders = find_training_folders()

    if not folders:
        print("❌ No training folders found!")
        return

    print("🔍 TRAINING PROGRESS OVERVIEW")
    print("=" * 80)

    for i, folder in enumerate(folders):
        progress_info, log_info = read_progress_info(folder)
        status = get_training_status(folder)

        print(f"\n📁 {i+1}. {folder}")
        print(f"   Status: {status}")

        if progress_info:
            epoch = progress_info.get('epoch', 0)
            total_epochs = progress_info.get('total_epochs', 0)
            percentage = progress_info.get('progress_percentage', 0)

            print(f"   Progress: {display_progress_bar(percentage)}")
            print(f"   Epoch: {epoch}/{total_epochs}")

            if 'train_loss' in progress_info:
                print(f"   Train Loss: {progress_info['train_loss']:.6f}")
            if 'val_loss' in progress_info:
                print(f"   Val Loss: {progress_info['val_loss']:.6f}")
            if 'ade' in progress_info:
                print(f"   ADE: {progress_info['ade']:.4f}m")
            if 'fde' in progress_info:
                print(f"   FDE: {progress_info['fde']:.4f}m")

            if 'estimated_remaining_minutes' in progress_info:
                remaining = progress_info['estimated_remaining_minutes'] * 60
                print(f"   ETA: {format_time(remaining)}")

            if 'timestamp' in progress_info:
                print(f"   Last Update: {progress_info['timestamp']}")

        # 显示文件夹修改时间
        folder_mtime = os.path.getmtime(folder)
        folder_time = datetime.fromtimestamp(folder_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   Folder Modified: {folder_time}")

def monitor_specific_training(folder_path, refresh_interval=10):
    """监控特定训练的详细进度"""
    if not os.path.exists(folder_path):
        print(f"❌ Training folder '{folder_path}' not found!")
        return

    print(f"🔍 MONITORING TRAINING: {folder_path}")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # 清屏

            print(f"🔍 MONITORING TRAINING: {folder_path}")
            print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            progress_info, log_info = read_progress_info(folder_path)
            status = get_training_status(folder_path)

            print(f"📊 Training Status: {status}")

            if progress_info:
                epoch = progress_info.get('epoch', 0)
                total_epochs = progress_info.get('total_epochs', 0)
                percentage = progress_info.get('progress_percentage', 0)

                print(f"\n📈 Progress Overview:")
                print(f"   {display_progress_bar(percentage, 60)}")
                print(f"   Epoch: {epoch}/{total_epochs}")

                # 显示详细指标
                metrics = []
                if 'train_loss' in progress_info:
                    metrics.append(f"Train Loss: {progress_info['train_loss']:.6f}")
                if 'val_loss' in progress_info:
                    metrics.append(f"Val Loss: {progress_info['val_loss']:.6f}")
                if 'ade' in progress_info:
                    metrics.append(f"ADE: {progress_info['ade']:.4f}m")
                if 'fde' in progress_info:
                    metrics.append(f"FDE: {progress_info['fde']:.4f}m")
                if 'learning_rate' in progress_info:
                    metrics.append(f"LR: {progress_info['learning_rate']:.2e}")

                if metrics:
                    print(f"\n📊 Current Metrics:")
                    for metric in metrics:
                        print(f"   {metric}")

                # 显示时间信息
                if 'elapsed_time_minutes' in progress_info:
                    elapsed = progress_info['elapsed_time_minutes'] * 60
                    print(f"\n⏱️  Time Information:")
                    print(f"   Elapsed: {format_time(elapsed)}")

                    if 'estimated_remaining_minutes' in progress_info:
                        remaining = progress_info['estimated_remaining_minutes'] * 60
                        print(f"   Remaining: {format_time(remaining)}")

                        total_estimated = elapsed + remaining
                        print(f"   Total Estimated: {format_time(total_estimated)}")

                if 'timestamp' in progress_info:
                    print(f"   Last Update: {progress_info['timestamp']}")

            # 显示最新的日志行
            if log_info.get('last_lines'):
                print(f"\n📝 Recent Log Entries:")
                for line in log_info['last_lines'][-5:]:  # 显示最后5行
                    if line.strip():
                        print(f"   {line.strip()}")

            # 检查是否完成
            final_results_file = os.path.join(folder_path, 'final_results.json')
            if os.path.exists(final_results_file):
                try:
                    with open(final_results_file, 'r') as f:
                        final_results = json.load(f)
                    print(f"\n🎉 TRAINING COMPLETED!")
                    print(f"   Total Time: {final_results.get('total_time_minutes', 0):.1f} minutes")
                    print(f"   Best Val Loss: {final_results.get('best_val_loss', 'N/A')}")
                    print(f"   Final ADE: {final_results.get('final_ade', 'N/A')}")
                    print(f"   Final FDE: {final_results.get('final_fde', 'N/A')}")
                    break
                except:
                    pass

            print(f"\n🔄 Refreshing in {refresh_interval} seconds... (Ctrl+C to stop)")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user.")

def main():
    parser = argparse.ArgumentParser(description='Monitor GNN Training Progress')
    parser.add_argument('--folder', '-f', type=str, help='Specific training folder to monitor')
    parser.add_argument('--interval', '-i', type=int, default=10, help='Refresh interval in seconds (default: 10)')
    parser.add_argument('--list', '-l', action='store_true', help='List all training folders')

    args = parser.parse_args()

    if args.list or not args.folder:
        display_training_summary()

        if not args.folder:
            folders = find_training_folders()
            if folders:
                print(f"\n💡 To monitor a specific training, use:")
                print(f"   python check_training_progress.py --folder {folders[0]}")
    else:
        monitor_specific_training(args.folder, args.interval)

if __name__ == "__main__":
    main()
