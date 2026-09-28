import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointkan import *
from models.kan import KAN

def get_activation(activation):
    """Get specified activation function"""
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class ConvBNReLU1D(nn.Module):
    """Standard module for 1D convolution + BN + activation function"""

    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class PointWarping(nn.Module):
    """PointConv-style point-cloud warping."""

    def forward(self, xyz1, xyz2, flow1):
        # xyz1: [B, 3, N1], xyz2: [B, 3, N2], flow1: [B, 3, N1]
        if flow1 is None:
            return xyz2

        xyz1_to_2 = xyz1 + flow1

        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape

        xyz1_to_2_t = xyz1_to_2.permute(0, 2, 1)
        xyz2_t = xyz2.permute(0, 2, 1)
        flow1_t = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2_t, xyz2_t)

        grouped_xyz_norm = index_points(xyz1_to_2_t, knn_idx) - xyz2_t.view(B, N2, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points(flow1_t, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = xyz2_t - flow2

        return warped_xyz2.permute(0, 2, 1)


class KANFeatureBlock(nn.Module):
    """Single-scale PointKAN feature extraction block for deepening local features."""

    def __init__(self, channels, kneighbors=24):
        super().__init__()
        self.kneighbors = kneighbors
        fused_dim = 2 * channels + 3

        self.kan_aggregator = nn.Sequential(
            nn.Linear(fused_dim, channels),
            nn.LayerNorm(channels),
            KAT_Group(mode='gelu'),  # Match the original DoubleStreamDGCNN setting
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, xyz, features):
        B, C, N = features.shape
        features_t = features.permute(0, 2, 1)
        identity = features_t

        idx = knn_point(self.kneighbors, xyz, xyz)
        grouped_features = index_points(features_t, idx)
        grouped_xyz = index_points(xyz, idx)

        expanded_features = features_t.unsqueeze(2).expand(-1, -1, self.kneighbors, -1)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)

        fused = torch.cat([expanded_features, grouped_features, relative_xyz], dim=-1)

        B, N, K, D = fused.shape
        fused = fused.view(B * N, K, D)
        kan_out = self.kan_aggregator(fused)

        pooled = torch.max(kan_out, dim=1)[0].view(B, N, -1)

        output_features = self.norm(identity + pooled).permute(0, 2, 1)
        return F.relu(output_features)


class GlobalAttention(nn.Module):
    """Global self-attention module"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        identity = x
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.norm(identity + out)


class KANFlowCorrelation(nn.Module):
    """Correlation module using efficient KAN ideas to learn cost volume"""

    def __init__(self, in_channels, hidden_channels=128, kneighbors=16):
        super().__init__()
        self.kneighbors = kneighbors
        self.warping = PointWarping()
        fused_dim = 2 * in_channels + 3

        self.kan_aggregator = nn.Sequential(
            nn.Linear(fused_dim, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            KAT_Group(mode='gelu'),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

    def forward(self, xyz1, xyz2, feat1, feat2, flow=None):
        if flow is not None:
            xyz2_warped = self.warping(xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1), flow).permute(0, 2, 1)
        else:
            xyz2_warped = xyz2

        feat1_t, feat2_t = feat1.permute(0, 2, 1), feat2.permute(0, 2, 1)
        idx = knn_point(self.kneighbors, xyz2_warped, xyz1)

        grouped_xyz2 = index_points(xyz2_warped, idx)
        grouped_feat2 = index_points(feat2_t, idx)
        expanded_feat1 = feat1_t.unsqueeze(2).expand(-1, -1, self.kneighbors, -1)
        relative_xyz = grouped_xyz2 - xyz1.unsqueeze(2)

        fused = torch.cat([expanded_feat1, grouped_feat2, relative_xyz], dim=-1)

        B, N, K, D = fused.shape
        fused = fused.view(-1, K, D)
        kan_out = self.kan_aggregator(fused)

        cost_volume = torch.max(kan_out, dim=1)[0].view(B, N, -1)
        return cost_volume.permute(0, 2, 1)


class SingleScale_PointKAN_Flow(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        num_feat_blocks=3,
        num_heads=4,
        cost_hidden_dim=128,
        feature_kneighbors=24,
        correlation_kneighbors=16,
    ):
        super().__init__()
        self.embedding = ConvBNReLU1D(3, embed_dim)
        self.feature_blocks = nn.ModuleList([
            KANFeatureBlock(embed_dim, kneighbors=feature_kneighbors)
            for _ in range(num_feat_blocks)
        ])
        self.attention = GlobalAttention(dim=embed_dim, num_heads=num_heads)
        self.correlation = KANFlowCorrelation(
            in_channels=embed_dim,
            hidden_channels=cost_hidden_dim,
            kneighbors=correlation_kneighbors,
        )

    def _encode(self, xyz_input):
        xyz = xyz_input.permute(0, 2, 1)
        features = self.embedding(xyz_input)
        for block in self.feature_blocks:
            features = block(xyz, features)
        features_t = self.attention(features.permute(0, 2, 1))
        return features_t.permute(0, 2, 1)

    def forward(self, pc1_xyz, pc2_xyz, flow=None):
        feat1, feat2 = self.encode_pair(pc1_xyz, pc2_xyz)
        return self.condition_from_features(
            pc1_xyz, pc2_xyz, feat1, feat2, flow=flow
        )

    def encode_pair(self, pc1_xyz, pc2_xyz):
        """Encode geometry that remains constant along the ODE trajectory."""
        return self._encode(pc1_xyz), self._encode(pc2_xyz)

    def condition_from_features(
        self, pc1_xyz, pc2_xyz, feat1, feat2, flow=None
    ):
        """Update the flow-dependent correlation using cached point features."""
        cost_volume = self.correlation(
            pc1_xyz.permute(0, 2, 1),
            pc2_xyz.permute(0, 2, 1),
            feat1,
            feat2,
            flow=flow
        )
        condition_y = torch.cat([feat1, cost_volume], dim=1)
        return condition_y

if __name__ == '__main__':
    B, N = 4, 1024
    pc1 = torch.randn(B, 3, N).cuda()
    pc2 = torch.randn(B, 3, N).cuda()

    print("--- Testing Single-scale PointKAN-Flow Model ---")

    model = SingleScale_PointKAN_Flow(
        embed_dim=64,
        num_feat_blocks=3,
        num_heads=4,
        cost_hidden_dim=128
    ).cuda()

    with torch.no_grad():
        condition_y = model(pc1, pc2)

    print(f"Input pc1 shape: {pc1.shape}")
    print(f"Input pc2 shape: {pc2.shape}")
    print(f"Model ran successfully!")
    print(f"Output condition information y shape: {condition_y.shape}")

    expected_dim = 64 + 128
    print(f"Expected feature dimension: {expected_dim}")
    print(f"Actual feature dimension: {condition_y.shape[1]}")
    assert condition_y.shape == (B, expected_dim, N)
    print("\nDimension check passed! This condition_y tensor is ready to be fed into the Flow Matching network.")
