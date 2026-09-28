"""Dedicated PointKAN training entry point for MilliFlow.

MilliFlow has a different supervision/prior path from MMBody:
the initial flow is an oracle pseudo-Doppler projection of GT flow plus the
configured legacy noise.  Its test frames also have variable point counts, so
test evaluation is deliberately batch size 1 and keeps the original points.
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MilliFlowSceneFlowDataset
from flow_matching import SceneFlowMatching
from flow_models import ConditionalPointFlowMatcher
from pointkan_encoder import SingleScale_PointKAN_Flow


def evaluate_split(
    condition_network,
    flow_matcher_network,
    data_loader,
    sfm,
    device,
    steps,
):
    """Evaluate a MilliFlow split with oracle pseudo-Doppler initialization."""
    condition_network.eval()
    flow_matcher_network.eval()
    total_error = 0.0
    total_accs = 0.0
    total_accr = 0.0
    total_points = 0
    dt = 1.0 / steps

    with torch.no_grad():
        for pc1, pc2, gt_flow in tqdm(data_loader, desc='Evaluating MilliFlow', leave=False):
            pc1 = pc1.to(device, dtype=torch.float32).permute(0, 2, 1)
            pc2 = pc2.to(device, dtype=torch.float32).permute(0, 2, 1)
            gt_flow = gt_flow.to(device, dtype=torch.float32)

            current_flow = sfm.get_doppler_flow(pc1, gt_flow)
            feat1, feat2 = condition_network.encode_pair(pc1, pc2)
            condition = condition_network.condition_from_features(
                pc1,
                pc2,
                feat1,
                feat2,
                flow=current_flow.permute(0, 2, 1).contiguous(),
            )

            for step in range(steps):
                t = torch.full(
                    (pc1.shape[0],), step * dt, device=device, dtype=torch.float32
                )
                velocity = flow_matcher_network(
                    S_t=current_flow,
                    pc1_xyz=pc1,
                    t=t,
                    condition=condition,
                )
                current_flow = sfm.euler_solver(current_flow, velocity, dt)

            point_error = torch.linalg.vector_norm(current_flow - gt_flow, dim=-1)
            gt_length = torch.linalg.vector_norm(gt_flow, dim=-1)
            relative_error = point_error / (gt_length + 1e-8)
            total_error += point_error.sum().item()
            total_accs += torch.logical_or(
                point_error <= 0.025, relative_error <= 0.025
            ).sum().item()
            total_accr += torch.logical_or(
                point_error <= 0.05, relative_error <= 0.05
            ).sum().item()
            total_points += point_error.numel()

    if total_points == 0:
        raise RuntimeError('MilliFlow evaluation split contains no points.')
    return (
        total_error / total_points,
        total_accs / total_points,
        total_accr / total_points,
    )


def save_loss_curve(losses, save_dir):
    path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average training loss')
    plt.title('PointKAN MilliFlow training loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def train(config):
    if config.get('dataset_name', '').lower() != 'milliflow':
        raise ValueError('train_milliflow.py requires dataset_name: milliflow')
    validate_during_training = bool(config.get('validate_each_epoch', False))
    if validate_during_training and config.get('val_partition') != 'val':
        raise ValueError('train_milliflow.py requires val_partition: val')
    if int(config.get('test_batch_size', 1)) != 1:
        raise ValueError('Final MilliFlow test evaluation requires test_batch_size: 1')

    device = config['device']
    save_dir = os.path.join('./ckpt', config['run_name'])
    os.makedirs(save_dir, exist_ok=True)
    print(f'Checkpoints will be saved to: {save_dir}')
    print('MilliFlow mode: oracle pseudo-Doppler + legacy noise')
    if validate_during_training:
        print('Validation mode: val split sampled to num_points, batched')
    print('Final test mode: original variable point counts, batch size 1')
    print(
        'PointKAN KNN: feature={}, flow_correlation={}'.format(
            config.get('feature_kneighbors', 6),
            config.get('correlation_kneighbors', 4),
        )
    )

    metrics_path = os.path.join(save_dir, 'training_metrics.csv')
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'session_id', 'epoch', 'learning_rate', 'train_loss',
                'val_epe3d',
                'val_accs',
                'val_accr',
            ])
    print(f'Metrics will be appended to: {metrics_path}')

    root = config['dataset_path']
    train_dataset = MilliFlowSceneFlowDataset(
        root, partition='train', num_points=config['num_points']
    )
    validation_dataset = None
    if validate_during_training:
        validation_dataset = MilliFlowSceneFlowDataset(
            root,
            partition=config.get('val_partition', 'val'),
            num_points=config['num_points'],
            sample_for_eval=config.get('val_sample_to_num_points', True),
        )
    raw_test_dataset = MilliFlowSceneFlowDataset(
        root,
        partition=config.get('test_partition', 'test'),
        num_points=config['num_points'],
        sample_for_eval=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.get('val_batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
        )
    raw_test_loader = DataLoader(
        raw_test_dataset,
        batch_size=config.get('test_batch_size', 1),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    print(f'Found {len(train_dataset)} samples in train partition.')
    if validation_dataset is not None:
        print(f'Found {len(validation_dataset)} samples in val partition.')
    print(f'Found {len(raw_test_dataset)} samples in raw test evaluation.')

    condition_network = SingleScale_PointKAN_Flow(
        embed_dim=64,
        cost_hidden_dim=128,
        feature_kneighbors=config.get('feature_kneighbors', 6),
        correlation_kneighbors=config.get('correlation_kneighbors', 4),
    ).to(device)
    flow_matcher_network = ConditionalPointFlowMatcher(
        condition_dim=128 + 64,
        kan_use=config.get('kan_use', True),
    ).to(device)

    if config.get('pretrained_path'):
        raise ValueError('This from-scratch MilliFlow configuration must not load a checkpoint.')

    optimizer = AdamW(
        list(condition_network.parameters()) + list(flow_matcher_network.parameters()),
        lr=config['learning_rate'],
    )
    scheduler_name = str(config.get('lr_scheduler', 'cosine')).lower()
    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_learning_rate', 0.0),
        )
    elif scheduler_name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['lr_adjust_epoch'],
            gamma=config.get('scheduler_gamma', 0.5),
        )
    else:
        raise ValueError(f'Unsupported lr_scheduler: {scheduler_name}')

    sfm = SceneFlowMatching()
    best_train_loss = float('inf')
    best_val_epe = float('inf')
    loss_history = []

    print('--- Starting dedicated MilliFlow training ---')
    for epoch in range(config['epochs']):
        condition_network.train()
        flow_matcher_network.train()
        batch_losses = []
        learning_rate = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]}')

        for pc1, pc2, gt_flow in pbar:
            pc1 = pc1.to(device, dtype=torch.float32).permute(0, 2, 1)
            pc2 = pc2.to(device, dtype=torch.float32).permute(0, 2, 1)
            gt_flow = gt_flow.to(device, dtype=torch.float32)
            t = torch.rand(pc1.shape[0], device=device)

            # MilliFlow-specific path: pseudo-Doppler is projected from GT.
            S_t, S_0 = sfm.get_flow_and_noise_Seudo_Doppler(
                pc=pc1,
                target_flow=gt_flow,
                t=t,
                noise_coefficient=config.get('noise_coefficient', 0.01),
            )
            optimizer.zero_grad()
            condition = condition_network(
                pc1, pc2, flow=S_t.permute(0, 2, 1).contiguous()
            )
            v_pred = flow_matcher_network(
                S_t=S_t, pc1_xyz=pc1, t=t, condition=condition
            )
            loss = sfm.loss_fn(v_pred, gt_flow, S_0)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.7f}')

        scheduler.step()
        train_loss = float(np.mean(batch_losses))
        loss_history.append(train_loss)

        val_epe = val_accs = val_accr = None
        interval = max(1, int(config.get('validation_interval', 5)))
        if validate_during_training and (
            (epoch + 1) % interval == 0 or (epoch + 1) == config['epochs']
        ):
            val_epe, val_accs, val_accr = evaluate_split(
                condition_network,
                flow_matcher_network,
                validation_loader,
                sfm,
                device,
                config.get('eval_steps', 10),
            )
            print(
                f'Epoch {epoch + 1} Validation EPE3D: {val_epe:.7f}, '
                f'ACCS: {val_accs:.4f}, ACCR: {val_accr:.4f}'
            )

        with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                session_id, epoch + 1, learning_rate, train_loss,
                '' if val_epe is None else val_epe,
                '' if val_accs is None else val_accs,
                '' if val_accr is None else val_accr,
            ])

        print(f'Epoch {epoch + 1} Average Loss: {train_loss:.7f}')
        state = {
            'condition_network_state_dict': condition_network.state_dict(),
            'flow_matcher_network_state_dict': flow_matcher_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'epoch_avg_losses': loss_history,
        }
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({**state, 'train_loss': best_train_loss},
                       os.path.join(save_dir, 'best_train_model.pth'))
        if val_epe is not None and val_epe < best_val_epe:
            best_val_epe = val_epe
            torch.save({
                **state,
                'val_epe3d': best_val_epe,
                'val_accs': val_accs,
                'val_accr': val_accr,
            }, os.path.join(save_dir, 'best_val_model.pth'))
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config['epochs']:
            torch.save(state, os.path.join(save_dir, f'ckpt_epoch_{epoch + 1}.pth'))

    # Final report: raw test points, strictly one sample per batch.
    raw_epe, raw_accs, raw_accr = evaluate_split(
        condition_network,
        flow_matcher_network,
        raw_test_loader,
        sfm,
        device,
        config.get('eval_steps', 10),
    )
    with open(os.path.join(save_dir, 'final_raw_test_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EPE3D: {raw_epe:.10f}\n')
        f.write(f'ACCS: {raw_accs:.10f}\n')
        f.write(f'ACCR: {raw_accr:.10f}\n')
        f.write('test_batch_size: 1\n')
        f.write('raw_test_points: true\n')
    print(
        f'Final raw test EPE3D: {raw_epe:.7f}, '
        f'ACCS: {raw_accs:.4f}, ACCR: {raw_accr:.4f}'
    )
    save_loss_curve(loss_history, save_dir)
    print('--- Dedicated MilliFlow training completed ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_milliflow.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        train(yaml.safe_load(f))
