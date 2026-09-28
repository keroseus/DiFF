import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MilliFlowSceneFlowDataset, RadarHumanSceneFlowDataset
from flow_matching import SceneFlowMatching
from flow_models import ConditionalPointFlowMatcher, FlowMatcherDGCNN
from pointkan_encoder import SingleScale_PointKAN_Flow


def compute_epe3d(predicted_flow, gt_flow):
    return torch.linalg.vector_norm(predicted_flow - gt_flow, dim=-1).mean()


def build_test_dataset(config):
    dataset_name = config['dataset_name'].lower()
    dataset_kwargs = {
        'root_path': config.get('test_dataset_path', config['dataset_path']),
        'partition': config.get('test_partition', 'test'),
        'num_points': config['num_points'],
    }
    if dataset_name == 'mmbody':
        dataset = RadarHumanSceneFlowDataset(
            **dataset_kwargs,
            frame_rate=config['frame_rate'],
            sample_for_eval=False,
        )
    elif dataset_name == 'milliflow':
        dataset = MilliFlowSceneFlowDataset(
            **dataset_kwargs,
            sample_for_eval=False,
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. Expected 'mmbody' or 'milliflow'."
        )
    return dataset_name, dataset


def evaluate(config, checkpoint_path, max_samples=None):
    device = config['device']
    condition_network = SingleScale_PointKAN_Flow(
        embed_dim=64,
        cost_hidden_dim=128,
        feature_kneighbors=config.get('feature_kneighbors', 24),
        correlation_kneighbors=config.get('correlation_kneighbors', 16),
    ).to(device).eval()

    if config['backbone_name'] == 'DGCNN':
        flow_matcher_network = FlowMatcherDGCNN(condition_dim=64 + 128).to(device).eval()
    elif config['backbone_name'] == 'PointKANFlowMatcher':
        flow_matcher_network = ConditionalPointFlowMatcher(
            condition_dim=64 + 128,
            kan_use=config.get('kan_use', True),
        ).to(device).eval()
    else:
        raise ValueError(f"Backbone {config['backbone_name']} not supported.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    condition_network.load_state_dict(checkpoint['condition_network_state_dict'])
    flow_matcher_network.load_state_dict(checkpoint['flow_matcher_network_state_dict'])

    dataset_name, test_dataset = build_test_dataset(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )
    print(
        'Test mode: partition={}, raw points, batch size 1, KNN={}/{}'.format(
            config.get('test_partition', 'test'),
            config.get('feature_kneighbors', 24),
            config.get('correlation_kneighbors', 16),
        )
    )
    sfm = SceneFlowMatching()
    steps = config.get('eval_steps', 10)
    dt = 1.0 / steps
    total_sample_epe3d = 0.0
    total_sample_accs = 0.0
    total_sample_accr = 0.0
    test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Evaluating {dataset_name} test set'):
            if dataset_name == 'mmbody':
                pc1, pc2, doppler_flow, gt_flow = batch
                current_flow = doppler_flow.to(device, dtype=torch.float32)
            else:
                pc1, pc2, gt_flow = batch
                current_flow = None

            pc1 = pc1.to(device, dtype=torch.float32).permute(0, 2, 1)
            pc2 = pc2.to(device, dtype=torch.float32).permute(0, 2, 1)
            gt_flow = gt_flow.to(device, dtype=torch.float32)

            if current_flow is None:
                current_flow = sfm.get_doppler_flow(pc1, gt_flow)

            feat1, feat2 = condition_network.encode_pair(pc1, pc2)
            condition = condition_network.condition_from_features(
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
                    condition=condition,
                )
                current_flow = sfm.euler_solver(current_flow, velocity, dt)

            predicted_flow = current_flow.squeeze(0)
            target_flow = gt_flow.squeeze(0)
            point_errors = torch.linalg.vector_norm(
                predicted_flow - target_flow, dim=-1
            )
            gt_flow_length = torch.linalg.vector_norm(target_flow, dim=-1)
            relative_error = point_errors / (gt_flow_length + 1e-8)

            total_sample_epe3d += point_errors.mean().item()
            total_sample_accs += torch.logical_or(
                point_errors <= 0.025, relative_error <= 0.025
            ).float().mean().item()
            total_sample_accr += torch.logical_or(
                point_errors <= 0.05, relative_error <= 0.05
            ).float().mean().item()
            test_samples += 1
            if max_samples is not None and test_samples >= max_samples:
                break

    if test_samples == 0:
        raise RuntimeError('The test dataset contains no samples.')
    print(f'Test samples: {test_samples}')
    print(f'Average 3D EPE (sample-weighted): {total_sample_epe3d / test_samples:.6f}')
    print(f'ACCS: {total_sample_accs / test_samples:.4f}')
    print(f'ACCR: {total_sample_accr / test_samples:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a scene-flow checkpoint.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional smoke-test limit; point clouds are still evaluated at raw size.',
    )
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        evaluate(yaml.safe_load(config_file), args.ckpt, args.max_samples)
