"""
消融训练脚本 - 危险场景生成

支持三组消融：
  --ablation no_danger_loss   A1: danger_weight=0，无危险监督
  --ablation no_svo           A2: SVO输入清零，无行为意图条件
  --ablation no_gnn           A3: GNN特征清零，仅用Transformer分支

用法：
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    conda activate nuplan
    python danger_gen_ablation/train_ablation.py --ablation no_danger_loss
    python danger_gen_ablation/train_ablation.py --ablation no_svo
    python danger_gen_ablation/train_ablation.py --ablation no_gnn

结果保存到 danger_gen_ablation/results/<ablation_name>/
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json
from tqdm import tqdm

# 路径解析：从 danger_gen_ablation/ 向上找到 GNN_Trans 根目录
_ablation_dir = os.path.dirname(os.path.abspath(__file__))
gnn_trans_root = os.path.dirname(_ablation_dir)

sys.path.insert(0, gnn_trans_root)
sys.path.insert(0, os.path.join(gnn_trans_root, 'models'))
sys.path.insert(0, os.path.join(gnn_trans_root, 'utils'))

_models_path = '/mnt/local/Xiao_Yi/nuplan-devkit/models'
if _models_path not in sys.path:
    sys.path.insert(0, _models_path)

import config
from scene_condition_encoder import SceneConditionEncoder, ConditionEncoderConfig
from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
from danger_score_torch import DangerExceedLoss, DangerScoreConfigTorch
from transformer_predictor import TransformerPredictor

# 复用当前主训练脚本的数据集和 collate
sys.path.insert(0, gnn_trans_root)
from train_diffusion_gnn_trans import IntegratedDiffusionDataset, collate_fn

VALID_ABLATIONS = ('no_danger_loss', 'no_svo', 'no_gnn')

# 与 v3.10 一致的基础配置
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DANGER_WEIGHT_BASE = 5.0   # v3.10 最优配置
DANGER_MARGIN = 0.5
DEVICE = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation training for danger scene generation')
    parser.add_argument('--ablation', required=True, choices=VALID_ABLATIONS,
                        help='Ablation group: no_danger_loss / no_svo / no_gnn')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (None = all). Use small number for debugging.')
    parser.add_argument('--device', default=DEVICE)
    return parser.parse_args()


def get_encoder_features_no_gnn(encoder, data):
    """获取场景特征，然后将 GNN 分量（后128维）清零。"""
    feat = encoder.get_encoder_features(data)  # [B, 384]
    feat = feat.clone()
    feat[:, 256:] = 0.0   # ego[0:128] + context[128:256] + gnn[256:384]，清零 gnn 分量
    return feat


def train_epoch(diffusion_model, gnn_trans_encoder, condition_encoder, loss_fn,
                dataloader, optimizer, device, epoch, ablation):
    diffusion_model.train()
    gnn_trans_encoder.train()
    condition_encoder.train()

    metrics = dict(total=0.0, recon=0.0, danger=0.0, margin_met=0, batches=0)
    bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in bar:
        batched_graph    = batch['batched_graph'].to(device)
        joint_increments = batch['joint_increments'].to(device)
        ego_history      = batch['ego_history'].to(device)
        agent_history    = batch['agent_history'].to(device)
        ego_last_pos     = batch['ego_last_pos'].to(device)
        ego_last_heading = batch['ego_last_heading'].to(device)
        ego_last_vx      = batch['ego_last_vx'].to(device)
        ego_last_vy      = batch['ego_last_vy'].to(device)
        svo_angle        = batch['svo_angle'].to(device)
        inherent_danger  = batch['inherent_danger'].to(device)
        danger_target    = batch['danger_target'].to(device)
        gt_outcome_danger = batch['gt_outcome_danger'].to(device)

        optimizer.zero_grad()

        # A2: SVO 清零
        if ablation == 'no_svo':
            svo_angle = torch.zeros_like(svo_angle)

        # 场景编码
        if ablation == 'no_gnn':
            scene_feat = get_encoder_features_no_gnn(gnn_trans_encoder, batched_graph)
        else:
            scene_feat = gnn_trans_encoder.get_encoder_features(batched_graph)

        condition = condition_encoder(scene_feat, svo_angle, inherent_danger, danger_target)

        # Diffusion 前向
        B = joint_increments.size(0)
        t = torch.randint(0, diffusion_model.config.num_diffusion_steps, (B,), device=device)
        x_t, noise = diffusion_model.add_noise(joint_increments, t)
        noise_pred = diffusion_model.denoiser(x_t, condition, t)

        alpha_bar_t = diffusion_model.alpha_bars[t]
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt().unsqueeze(-1) * noise_pred) / \
                   alpha_bar_t.sqrt().unsqueeze(-1)
        x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)

        total_loss, loss_dict = loss_fn(
            x_0_pred, joint_increments,
            ego_history, agent_history,
            ego_last_pos, ego_last_heading, ego_last_vx, ego_last_vy,
            gt_outcome_danger.squeeze(-1),
            svo_angle
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gnn_trans_encoder.parameters()) +
            list(condition_encoder.parameters()) +
            list(diffusion_model.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        metrics['total']      += total_loss.item()
        metrics['recon']      += loss_dict['recon_loss'].item()
        metrics['danger']     += loss_dict['danger_exceed_loss'].item()
        metrics['margin_met'] += loss_dict['danger_margin_met']
        metrics['batches']    += 1

        bar.set_postfix(loss=f"{total_loss.item():.3f}",
                        met=f"{loss_dict['danger_margin_met']:.0%}")

    n = max(metrics['batches'], 1)
    return {
        'total_loss':       metrics['total'] / n,
        'recon_loss':       metrics['recon'] / n,
        'danger_loss':      metrics['danger'] / n,
        'margin_met_rate':  metrics['margin_met'] / n,
    }


def main():
    args = parse_args()
    ablation = args.ablation

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[WARNING] CUDA not available, using CPU')
        device = 'cpu'

    # 输出目录
    output_path = Path(_ablation_dir) / 'results' / ablation
    output_path.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Ablation training: {ablation}')
    print(f'Output: {output_path}')
    print('=' * 60)

    # 保存配置
    cfg_dict = {
        'ablation': ablation,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_samples': args.max_samples,
        'device': device,
        'danger_weight': 0.0 if ablation == 'no_danger_loss' else DANGER_WEIGHT_BASE,
        'danger_margin': DANGER_MARGIN,
        'learning_rate': LEARNING_RATE,
        'base_version': 'v3.10',
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    processed_dir   = config.DIFFUSION_PROCESSED_DATA_DIR
    danger_pair_dir = config.DIFFUSION_DANGER_PAIR_DIR

    dataset = IntegratedDiffusionDataset(
        processed_dir, danger_pair_dir, max_samples=args.max_samples
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    print(f'Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch')

    # 模型
    gnn_trans_encoder = TransformerPredictor().to(device)
    condition_encoder = SceneConditionEncoder(ConditionEncoderConfig()).to(device)
    diffusion_model   = DangerSceneDiffusion(DiffusionConfig()).to(device)

    # A1: danger_weight = 0
    danger_weight = 0.0 if ablation == 'no_danger_loss' else DANGER_WEIGHT_BASE
    loss_fn = DangerExceedLoss(
        recon_weight=1.0,
        danger_weight=danger_weight,
        danger_margin=DANGER_MARGIN,
        config=DangerScoreConfigTorch(),
    )

    optimizer = torch.optim.Adam(
        list(gnn_trans_encoder.parameters()) +
        list(condition_encoder.parameters()) +
        list(diffusion_model.parameters()),
        lr=LEARNING_RATE
    )

    history = []
    best_margin_met = -1.0

    for epoch in range(1, args.epochs + 1):
        ep = train_epoch(diffusion_model, gnn_trans_encoder, condition_encoder,
                         loss_fn, dataloader, optimizer, device, epoch, ablation)
        history.append({'epoch': epoch, **ep})

        print(f'Epoch {epoch}/{args.epochs}: '
              f'total={ep["total_loss"]:.3f}  '
              f'recon={ep["recon_loss"]:.3f}  '
              f'danger={ep["danger_loss"]:.3f}  '
              f'margin_met={ep["margin_met_rate"]:.1%}')

        if ep['margin_met_rate'] > best_margin_met:
            best_margin_met = ep['margin_met_rate']
            torch.save({
                'gnn_trans_encoder': gnn_trans_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model':   diffusion_model.state_dict(),
                'optimizer':         optimizer.state_dict(),
                'epoch':             epoch,
                'metrics':           ep,
                'ablation':          ablation,
            }, output_path / 'best_model.pt')
            print(f'  [Best] margin_met={best_margin_met:.1%} at epoch {epoch}')

        if epoch % 10 == 0:
            torch.save({
                'gnn_trans_encoder': gnn_trans_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model':   diffusion_model.state_dict(),
                'epoch':             epoch,
            }, output_path / f'checkpoint_epoch_{epoch}.pt')

    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print('\n' + '=' * 60)
    print(f'Training done. Best margin_met = {best_margin_met:.1%}')
    print(f'Model saved to: {output_path / "best_model.pt"}')
    print('=' * 60)


if __name__ == '__main__':
    main()
