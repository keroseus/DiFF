import os
import numpy as np
import ujson
from torch.utils.data import Dataset
import glob

class BasicSceneFlowDataset(Dataset):
    def __init__(self, root_path, partition='train', num_points=1024):
        """
        Scene-flow dataset loader for point-cloud pairs and ground-truth flow.

        Args:
            root_path (str): Root directory of the dataset.
            partition (str): 'train' or 'val'/'test'.
            num_points (int): Number of points to uniformly downsample each point cloud to.
        """
        self.root = os.path.join(root_path, partition)
        self.npoints = num_points
        self.partition = partition
        self.samples = []

        for seq_folder in os.listdir(self.root):
            seq_path = os.path.join(self.root, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            json_files = sorted(
                glob.glob(os.path.join(seq_path, '*.json')),
                key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
            )
            self.samples.extend(json_files)

        print(f'Found {len(self.samples)} samples in {partition} partition.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with open(self.samples[index], 'r') as f:
            data = ujson.load(f)

        pc1 = np.array(data["pc1"])[:, :3].astype('float32')
        pc2 = np.array(data["pc2"])[:, :3].astype('float32')
        gt_flow = np.array(data["gt_flow"]).astype('float32')

        if self.partition == 'train':
            n_points1 = pc1.shape[0]
            if n_points1 < self.npoints:
                sample_idx1 = np.concatenate([
                    np.arange(n_points1),
                    np.random.choice(n_points1, self.npoints - n_points1, replace=True)
                ])
            else:
                sample_idx1 = np.random.choice(n_points1, self.npoints, replace=False)

            pc1 = pc1[sample_idx1, :]
            gt_flow = gt_flow[sample_idx1, :]  # flow should follow pc1 sampling

            n_points2 = pc2.shape[0]
            if n_points2 < self.npoints:
                sample_idx2 = np.concatenate([
                    np.arange(n_points2),
                    np.random.choice(n_points2, self.npoints - n_points2, replace=True)
                ])
            else:
                sample_idx2 = np.random.choice(n_points2, self.npoints, replace=False)

            pc2 = pc2[sample_idx2, :]

        return pc1, pc2, gt_flow


class RadialDopplerSceneFlowDataset(Dataset):
    def __init__(self, root_path, partition='train', num_points=512):
        """
        Scene-flow dataset loader with a radial Doppler motion prior.

        Args:
            root_path (str): Root directory of the dataset.
            partition (str): 'train' or 'val'/'test'.
            num_points (int): Number of points to uniformly downsample each point cloud to.
        """
        self.root = os.path.join(root_path, partition)
        self.npoints = num_points
        self.partition = partition
        self.samples = []

        for seq_folder in os.listdir(self.root):
            seq_path = os.path.join(self.root, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            json_files = sorted(
                glob.glob(os.path.join(seq_path, '*.json')),
                key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
            )
            self.samples.extend(json_files)

        print(f'Found {len(self.samples)} samples in {partition} partition.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with open(self.samples[index], 'r') as f:
            data = ujson.load(f)

        pc1 = np.array(data["pc1"])[:, :3].astype('float32')
        pc2 = np.array(data["pc2"])[:, :3].astype('float32')
        doppler = np.array(data["pc1"])[:, 4:5].astype('float32')
        pc1_range = np.linalg.norm(pc1, axis=1, keepdims=True).clip(min=1e-8)
        doppler_flow = doppler * pc1 / pc1_range / 30.0
        gt_flow = np.array(data["gt_flow"]).astype('float32')

        if self.partition == 'train':
            n_points1 = pc1.shape[0]
            if n_points1 < self.npoints:
                sample_idx1 = np.concatenate([
                    np.arange(n_points1),
                    np.random.choice(n_points1, self.npoints - n_points1, replace=True)
                ])
            else:
                sample_idx1 = np.random.choice(n_points1, self.npoints, replace=False)

            pc1 = pc1[sample_idx1, :]
            gt_flow = gt_flow[sample_idx1, :]  # flow should follow pc1 sampling
            doppler_flow = doppler_flow[sample_idx1, :]
            n_points2 = pc2.shape[0]
            if n_points2 < self.npoints:
                sample_idx2 = np.concatenate([
                    np.arange(n_points2),
                    np.random.choice(n_points2, self.npoints - n_points2, replace=True)
                ])
            else:
                sample_idx2 = np.random.choice(n_points2, self.npoints, replace=False)

            pc2 = pc2[sample_idx2, :]

        return pc1, pc2, gt_flow,doppler_flow


class MilliFlowSceneFlowDataset(Dataset):
    def __init__(self, root_path, partition='train', num_points=1024,
                 sample_for_eval=False):
        """
        Scene-flow dataset loader for the MilliFlow directory and annotation format.

        Args:
            root_path (str): Root directory of the dataset.
            partition (str): 'train' or 'val'/'test'.
            num_points (int): Number of points to uniformly downsample each point cloud to.
        """
        self.root = os.path.join(root_path, partition)
        self.npoints = num_points
        self.partition = partition
        self.sample_for_eval = bool(sample_for_eval)
        self.samples = []

        for p in os.listdir(self.root):
            clips_path = os.path.join(self.root, p)  # /train/0
            clips = os.listdir(clips_path)
            for clip in clips:
                clip_path = os.path.join(clips_path, clip)  # train/0/arm_00
                samples = sorted(
                    os.listdir(clip_path),
                    key=lambda x: int(x.split("_")[-1].split("-")[-1].split(".")[0]),
                )
                for j in range(len(samples)):
                    self.samples.append(os.path.join(clip_path, samples[j]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with open(self.samples[index], 'r') as f:
            data = ujson.load(f)

        pc1 = np.array(data["pc1"])[:, :3].astype('float32')
        pc2 = np.array(data["pc2"])[:, :3].astype('float32')
        gt_flow = np.array(data["gt"]).astype('float32') - pc1

        if self.partition == 'train' or self.sample_for_eval:
            n_points1 = pc1.shape[0]
            if n_points1 < self.npoints:
                sample_idx1 = np.concatenate([
                    np.arange(n_points1),
                    np.random.choice(n_points1, self.npoints - n_points1, replace=True)
                ])
            else:
                sample_idx1 = np.random.choice(n_points1, self.npoints, replace=False)

            pc1 = pc1[sample_idx1, :]
            gt_flow = gt_flow[sample_idx1, :]  # flow should follow pc1 sampling

            n_points2 = pc2.shape[0]
            if n_points2 < self.npoints:
                sample_idx2 = np.concatenate([
                    np.arange(n_points2),
                    np.random.choice(n_points2, self.npoints - n_points2, replace=True)
                ])
            else:
                sample_idx2 = np.random.choice(n_points2, self.npoints, replace=False)

            pc2 = pc2[sample_idx2, :]
        return pc1, pc2, gt_flow
class RadarHumanSceneFlowDataset(Dataset):
    def __init__(self, root_path, partition='train', num_points=512,
                 frame_rate=30.0, sample_for_eval=False):
        """
        Human radar scene-flow loader with a measured Doppler displacement prior.

        Args:
            root_path (str): Root directory of the dataset.
            partition (str): 'train' or 'val'/'test'.
            num_points (int): Number of points to uniformly downsample each point cloud to.
            frame_rate (float): Radar frame rate used to convert radial velocity
                to per-frame displacement.
            sample_for_eval (bool): Also sample/pad validation points to
                ``num_points``. This is needed when validation uses a batch
                size greater than one because raw frame point counts vary.
        """
        self.root = os.path.join(root_path, partition)
        self.npoints = num_points
        self.partition = partition
        self.sample_for_eval = bool(sample_for_eval)
        self.samples = []
        if frame_rate <= 0:
            raise ValueError('frame_rate must be positive.')
        self.frame_rate = float(frame_rate)
        for seq_folder in os.listdir(self.root):
            seq_path = os.path.join(self.root, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            json_files = sorted(
                glob.glob(os.path.join(seq_path, '*.json')),
                key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
            )
            self.samples.extend(json_files)

        print(f'Found {len(self.samples)} samples in {partition} partition.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with open(self.samples[index], 'r') as f:
            data = ujson.load(f)

        pc1 = np.array(data["pc1"])[:, :3].astype('float32')
        pc2 = np.array(data["pc2"])[:, :3].astype('float32')
        doppler_pc1 = np.array(data["pc1"])[:, 4:5].astype('float32')
        pc1_range = np.linalg.norm(pc1, axis=1, keepdims=True).clip(min=1e-8)
        doppler_flow = doppler_pc1 * pc1 / pc1_range / self.frame_rate
        gt_flow = np.array(data["gt_flow"]).astype('float32')

        # During training, perform random sampling to ensure consistent number
        # of points. Evaluation can opt into the same fixed-size sampling so
        # that validation can use the training batch size.
        if self.partition in ('train', 'filtered_sequences') or self.sample_for_eval:
            n_points1 = pc1.shape[0]
            if n_points1 < self.npoints:
                sample_idx1 = np.concatenate([
                    np.arange(n_points1),
                    np.random.choice(n_points1, self.npoints - n_points1, replace=True)
                ])
            else:
                sample_idx1 = np.random.choice(n_points1, self.npoints, replace=False)

            pc1 = pc1[sample_idx1, :]
            gt_flow = gt_flow[sample_idx1, :]  # flow should follow pc1 sampling
            doppler_flow = doppler_flow[sample_idx1, :]
            n_points2 = pc2.shape[0]
            if n_points2 < self.npoints:
                sample_idx2 = np.concatenate([
                    np.arange(n_points2),
                    np.random.choice(n_points2, self.npoints - n_points2, replace=True)
                ])
            else:
                sample_idx2 = np.random.choice(n_points2, self.npoints, replace=False)

            pc2 = pc2[sample_idx2, :]
        return pc1, pc2, doppler_flow, gt_flow
