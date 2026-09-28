import torch
import yaml
import os
import argparse
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets import MilliFlowSceneFlowDataset, RadarHumanSceneFlowDataset
from pointkan_encoder import SingleScale_PointKAN_Flow
from flow_models import *
from flow_matching import SceneFlowMatching


def evaluate_validation_epe(
    condition_network,
    flow_matcher_network,
    val_loader,
    dataset_name,
    sfm,
    device,
    steps,
):
    """Evaluate one deterministic trajectory on the validation split."""
    condition_network.eval()
    flow_matcher_network.eval()
    total_error = 0.0
    total_points = 0
    total_accs = 0
    total_accr = 0
    dt = 1.0 / steps

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            if dataset_name == 'mmbody':
                pc1, pc2, doppler_flow, gt_flow = batch
                initial_flow = doppler_flow.to(device, dtype=torch.float32)
            else:
                pc1, pc2, gt_flow = batch
                initial_flow = None

            pc1 = pc1.to(device, dtype=torch.float32).permute(0, 2, 1)
            pc2 = pc2.to(device, dtype=torch.float32).permute(0, 2, 1)
            gt_flow = gt_flow.to(device, dtype=torch.float32)

            # MMBody starts from measured Doppler displacement. MilliFlow has no
            # measured Doppler, so its oracle prior is projected from validation GT.
            if initial_flow is None:
                initial_flow = sfm.get_doppler_flow(pc1, gt_flow)

            current_flow = initial_flow
            feat1, feat2 = condition_network.encode_pair(pc1, pc2)
            condition_y = condition_network.condition_from_features(
                pc1, pc2, feat1, feat2,
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
                    condition=condition_y,
                )
                current_flow = sfm.euler_solver(current_flow, velocity, dt)

            point_error = torch.linalg.vector_norm(current_flow - gt_flow, dim=-1)
            gt_flow_length = torch.linalg.vector_norm(gt_flow, dim=-1)
            relative_error = point_error / (gt_flow_length + 1e-8)
            total_error += point_error.sum().item()
            total_accs += torch.logical_or(
                point_error <= 0.025, relative_error <= 0.025
            ).sum().item()
            total_accr += torch.logical_or(
                point_error <= 0.05, relative_error <= 0.05
            ).sum().item()
            total_points += point_error.numel()

    if total_points == 0:
        raise RuntimeError('The validation dataset contains no points.')
    return (
        total_error / total_points,
        total_accs / total_points,
        total_accr / total_points,
    )
def plot_training_losses(epoch_avg_losses, save_dir):
    """Plot and save training loss curve"""
    plt.figure(figsize=(12, 8))

    plt.plot(range(1, len(epoch_avg_losses) + 1), epoch_avg_losses,
             'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)

    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to: {loss_curve_path}")



def train(config):
    # --- 1. Parameter and Path Settings ---
    device = config['device']
    base_ckpt_path = './ckpt'
    run_name = config.get(
        'run_name', f"{config['backbone_name']}_{config['dataset_name'].lower()}"
    )
    save_dir = os.path.join(base_ckpt_path, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # Append one crash-safe metrics row after every completed epoch. A session
    # identifier keeps separate restarts distinguishable when the same run
    # directory is reused.
    metrics_csv_path = os.path.join(save_dir, 'training_metrics.csv')
    metrics_session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(metrics_csv_path) or os.path.getsize(metrics_csv_path) == 0:
        with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as metrics_file:
            csv.writer(metrics_file).writerow([
                'session_id',
                'epoch',
                'learning_rate',
                'train_loss',
                'val_epe3d',
                'accs',
                'accr',
            ])
    print(f"Training metrics will be appended to: {metrics_csv_path}")

    # --- 2. Data Loading ---
    dataset_name = config['dataset_name'].lower()
    dataset_kwargs = {
        'root_path': config.get('train_dataset_path', config['dataset_path']),
        'partition': config['train_partition'],
        'num_points': config['num_points'],
    }
    if dataset_name == 'mmbody':
        train_dataset = RadarHumanSceneFlowDataset(
            **dataset_kwargs,
            frame_rate=config['frame_rate'],
        )
        prior_name = 'measured Doppler + orthogonal noise'
    elif dataset_name == 'milliflow':
        train_dataset = MilliFlowSceneFlowDataset(**dataset_kwargs)
        prior_name = 'oracle pseudo-Doppler + orthogonal noise'
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            "Expected 'mmbody' or 'milliflow'."
        )

    print(f"Using dataset: {dataset_name}")
    print(f"Using initial-flow prior: {prior_name}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # Do not even scan/load the validation partition when validation is disabled.
    validate_during_training = bool(config.get('validate_each_epoch', False))
    if validate_during_training:
        val_kwargs = {
            'root_path': config.get('val_dataset_path', config['dataset_path']),
            'partition': config.get('val_partition', 'val'),
            'num_points': config['num_points'],
        }
        if dataset_name == 'mmbody':
            val_dataset = RadarHumanSceneFlowDataset(
                **val_kwargs,
                frame_rate=config['frame_rate'],
                sample_for_eval=config.get('val_sample_to_num_points', False),
            )
        else:
            val_dataset = MilliFlowSceneFlowDataset(**val_kwargs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('val_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        val_loader = None
        print('Training-time validation disabled; validation partition will not be loaded.')

    # --- 3. Model Instantiation ---
    print(f"Using backbone: {config['backbone_name']}")
    condition_network = SingleScale_PointKAN_Flow(embed_dim=64, cost_hidden_dim=128).to(device)
    if config['backbone_name'] in ('PointKANFlowMatcher', 'DoubleStreamDGCNN'):
        flow_matcher_network = ConditionalPointFlowMatcher(
            condition_dim=128 + 64,
            kan_use=config.get('kan_use', True),
        ).to(device)
    else:
        raise ValueError(f"Backbone {config['backbone_name']} not supported.")
    # Load model weights only; initialize optimization from the current config.
    if config.get('pretrained_path'):
        pretrained_path = config['pretrained_path']
        print(f"Loading pretrained model weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        condition_network.load_state_dict(
            checkpoint['condition_network_state_dict']
        )
        flow_matcher_network.load_state_dict(
            checkpoint['flow_matcher_network_state_dict']
        )
        print(f"Warm-start checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    # --- 4. Optimizer and Learning Rate Scheduler ---
    optimizer = AdamW(
        list(condition_network.parameters()) + list(flow_matcher_network.parameters()),
        lr=config['learning_rate']
    )
    scheduler_name = str(config.get('lr_scheduler', 'cosine')).lower()
    if scheduler_name == 'cosine':
        # Matches the paper: cosine annealing over all training epochs.
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

    # Track the best training loss, best validation EPE3D, and histories.
    best_train_loss = float('inf')
    best_val_epe = float('inf')
    epoch_avg_losses = []  # List for storing average loss of each epoch
    val_epe_history = []
    val_accs_history = []
    val_accr_history = []

    print("--- Starting scene flow model training ---")
    for epoch in range(config['epochs']):
        epoch_learning_rate = optimizer.param_groups[0]['lr']
        current_epoch_batch_losses = []  # Temporary list for calculating current epoch's average loss
        condition_network.train()
        flow_matcher_network.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for i, batch in enumerate(pbar):
            if dataset_name == 'mmbody':
                pc1, pc2, doppler_flow, gt_flow = batch
                doppler_flow = doppler_flow.to(device, dtype=torch.float32)
            else:
                pc1, pc2, gt_flow = batch

            pc1 = pc1.to(device, dtype=torch.float32).permute(0, 2, 1)
            pc2 = pc2.to(device, dtype=torch.float32).permute(0, 2, 1)
            gt_flow = gt_flow.to(device, dtype=torch.float32)

            B = pc1.shape[0]
            optimizer.zero_grad()

            t = torch.rand(B, device=device)
            if dataset_name == 'mmbody':
                # doppler_flow is measured radial velocity converted to displacement
                # by RadarHumanSceneFlowDataset.
                tangential_noise = sfm.get_orthogonal_noise(
                    pc1,
                    noise_coefficient=config.get('noise_coefficient', 0.01),
                )
                tangential_noise = tangential_noise.permute(0, 2, 1).contiguous()
                initial_flow = doppler_flow + tangential_noise
                S_t, S_0 = sfm.get_flow_and_noise(
                    target_flow=gt_flow,
                    t=t,
                    noise_flow=initial_flow,
                )
            else:  # milliflow: no measured Doppler is available
                S_t, S_0 = sfm.get_flow_and_noise_Seudo_Doppler(
                    pc=pc1,
                    target_flow=gt_flow,
                    t=t,
                    noise_coefficient=config.get('noise_coefficient', 0.01),
                )
            condition_y = condition_network(
                pc1,
                pc2,
                flow=S_t.permute(0, 2, 1).contiguous(),
            )
            v_pred = flow_matcher_network(S_t=S_t, pc1_xyz=pc1, t=t, condition=condition_y)

            # Keep the original unweighted flow-matching MSE loss.
            loss = sfm.loss_fn(v_pred, gt_flow, S_0)
            loss.backward()
            optimizer.step()

            current_epoch_batch_losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.7f}')

        scheduler.step()

        epoch_avg_loss = np.mean(current_epoch_batch_losses)
        epoch_avg_losses.append(epoch_avg_loss)  # Store current epoch's average loss in history list
        print(f"\nEpoch {epoch + 1} Average Loss: {epoch_avg_loss:.7f}")

        val_epe = None
        val_accs = None
        val_accr = None
        validation_interval = max(1, int(config.get('validation_interval', 1)))
        should_validate = (
            config.get('validate_each_epoch', False)
            and ((epoch + 1) % validation_interval == 0
                 or (epoch + 1) == config['epochs'])
        )
        if should_validate:
            val_epe, val_accs, val_accr = evaluate_validation_epe(
                condition_network=condition_network,
                flow_matcher_network=flow_matcher_network,
                val_loader=val_loader,
                dataset_name=dataset_name,
                sfm=sfm,
                device=device,
                steps=config.get('eval_steps', 10),
            )
            val_epe_history.append(val_epe)
            val_accs_history.append(val_accs)
            val_accr_history.append(val_accr)
            print(
                f"Epoch {epoch + 1} Validation EPE3D: {val_epe:.7f}, "
                f"ACCS: {val_accs:.4f}, ACCR: {val_accr:.4f}"
            )

        # Open in append mode for every epoch so the row is flushed to disk
        # immediately and survives an interrupted training process.
        with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as metrics_file:
            csv.writer(metrics_file).writerow([
                metrics_session_id,
                epoch + 1,
                epoch_learning_rate,
                epoch_avg_loss,
                '' if val_epe is None else val_epe,
                '' if val_accs is None else val_accs,
                '' if val_accr is None else val_accr,
            ])

        if epoch_avg_loss < best_train_loss:
            best_train_loss = epoch_avg_loss
            print(
                f"New best training loss: {best_train_loss:.7f}. "
                "Saving best_train_model.pth..."
            )
            torch.save({
                'condition_network_state_dict': condition_network.state_dict(),
                'flow_matcher_network_state_dict': flow_matcher_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': best_train_loss,
                'epoch_avg_losses': epoch_avg_losses,
                'val_epe_history': val_epe_history,
                'val_accs_history': val_accs_history,
                'val_accr_history': val_accr_history,
            }, os.path.join(save_dir, 'best_train_model.pth'))

        # Save the model selected by validation EPE3D.
        if val_epe is not None and val_epe < best_val_epe:
            best_val_epe = val_epe
            print(f"New best validation EPE3D: {best_val_epe:.7f}. Saving best_val_model.pth...")
            torch.save({
                'condition_network_state_dict': condition_network.state_dict(),
                'flow_matcher_network_state_dict': flow_matcher_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_epe3d': best_val_epe,
                'epoch_avg_losses': epoch_avg_losses,
                'val_epe_history': val_epe_history,
                'val_accs_history': val_accs_history,
                'val_accr_history': val_accr_history,
            }, os.path.join(save_dir, 'best_val_model.pth'))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == config['epochs']:
            print(f"Saving periodic checkpoint at epoch {epoch + 1}...")
            torch.save({
                'condition_network_state_dict': condition_network.state_dict(),
                'flow_matcher_network_state_dict': flow_matcher_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'epoch_avg_losses': epoch_avg_losses,
                'val_epe_history': val_epe_history,
                'val_accs_history': val_accs_history,
                'val_accr_history': val_accr_history,
            }, os.path.join(save_dir, f'ckpt_epoch_{epoch + 1}.pth'))

    print("--- Training completed ---")
    plot_training_losses(epoch_avg_losses, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
