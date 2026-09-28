import torch
import torch.nn as nn
import torch.nn.functional as F

from pointkan_encoder import *

def get_timestep_embedding(timesteps, dim):
    """Sinusoidal positional encoding for time"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


def get_graph_feature(x, k=20):
    """Build graph features for DGCNN"""
    # x shape: [B, C, N]
    B, C, N = x.shape
    idx = knn_point(k, x.permute(0, 2, 1), x.permute(0, 2, 1))  # [B, N, k]

    # Adjust idx shape for index_select
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    # Extract neighbor features
    features = x.transpose(2, 1).reshape(-1, C)  # [B*N, C]
    neighbors = features[idx, :].view(B, N, k, C)  # [B, N, k, C]

    # Expand central point features
    central = x.transpose(2, 1).unsqueeze(2).expand(B, N, k, C)

    # Concatenate central point features with neighbor-center difference features
    graph_feature = torch.cat([central, neighbors - central], dim=3).permute(0, 3, 1, 2)  # [B, 2C, N, k]
    return graph_feature


class FlowMatcherDGCNN(nn.Module):
    def __init__(self, condition_dim, point_dim=3, time_emb_dim=64):
        super().__init__()

        # Improved time encoding - using sinusoidal positional encoding
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input feature dimension
        input_channel = point_dim + time_emb_dim + condition_dim

        # Improved convolution layers - adding residual connections and better activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # Add skip connection
        self.skip_conv = nn.Conv1d(input_channel, 256, kernel_size=1)

        # Improved global feature processing
        self.conv3 = nn.Sequential(
            nn.Conv1d(128 + 256 + 256, 512, kernel_size=1, bias=False),  # Adding skip connection
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1)  # Adding dropout
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Output layer - adding small initialization
        self.output_conv = nn.Conv1d(256, 3, kernel_size=1)
        nn.init.normal_(self.output_conv.weight, std=0.001)  # Small weight initialization
        nn.init.zeros_(self.output_conv.bias)

    def get_timestep_embedding(self, timesteps, dim):
        """Sinusoidal positional encoding for time"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, S_t, pc1_xyz, t, condition):
        B, N, _ = S_t.shape
        pc1_xyz = pc1_xyz.permute(0, 2, 1)

        # Improved time encoding
        t_emb = self.get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1).expand(-1, N, -1)

        # Feature concatenation
        initial_features = torch.cat([S_t, t_emb, condition.permute(0, 2, 1)], dim=2)
        x = initial_features.permute(0, 2, 1)

        # Save skip connection
        skip_features = self.skip_conv(x)

        # EdgeConv layer
        x1 = get_graph_feature(x, k=20)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(x1, k=20)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # Fuse features + skip connection
        x_global = torch.cat((x1, x2, skip_features), dim=1)
        x_global = self.conv3(x_global)
        x_global = self.conv4(x_global)

        # Output prediction
        v_pred = self.output_conv(x_global)

        return v_pred.permute(0, 2, 1)

class ConditionalPointFlowMatcher(nn.Module):
    def __init__(self, condition_dim, point_dim=3, time_emb_dim=64,kan_use = False):
        super().__init__()
        self.kan_use = kan_use
        self.time_emb_dim = time_emb_dim
        # --- 1. Geometric Encoding Stream ---
        # This stream specifically processes pc1 coordinates to extract pure geometric features
        self.geo_conv1 = nn.Sequential(nn.Conv2d(point_dim * 2, 64, 1, bias=False), nn.BatchNorm2d(64),
                                       nn.LeakyReLU(0.2))
        self.geo_conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # fusion_dim = 128 + point_dim + condition_dim
        # fusion_dim = point_dim + condition_dim
        fusion_dim = condition_dim
        # --- 2. State Information Processing ---
        # Time encoding module
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim,condition_dim))
        self.pointwise_conv = nn.Conv1d(3, fusion_dim, kernel_size=1)
        # --- 3. Fusion and Decoding ---
        # Fused feature dimension = geometric features(128) + S_t(3) + time encoding(64) + condition y

        if kan_use == None:
            self.decoder = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )
        else :
            self.decoder = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.LayerNorm(512),
                KAT_Group(mode='gelu'),

                nn.Linear(512, 256),
                nn.LayerNorm(256),
                KAT_Group(mode='gelu'),

                nn.Linear(256, 3)
            )

    def time_emb(self, t, dim):
        """Sinusoidal encoding for time, single dimension
       Goal: Let the model perceive the time t of input x_t
       Implementation methods: Various
       Input x: [B, C, H, W] x += temb is space-independent, meaning each spatial position (H, W) needs to add the same time encoding vector [B, C]
       Assuming B=1 t=0.1
       1. Simple brute force method
       temb = [0.1] * C -> [0.1, 0.1, 0.1, ……]
       x += temb.reshape(1, C, 1, 1)
       2. Similar to absolute position encoding
       Implementation method in this code
       3. Through learning (ensure T is discrete 0, 1, 2, 3, ……, T)
       temb_learn = nn.Parameter(T+1, dim)
       x += temb_learn[t, :].reshape(1, C, 1, 1)


        Args:
            t (float): Time, dimension [B]
            dim (int): Encoding dimension

        Returns:
            torch.Tensor: Encoded time, dimension [B, dim]  Input is [B, C, H, W]
        """
        # Generate sinusoidal encoding
        # Map t to [0, 1000]
        t = t * 1000
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)
    def forward(self, S_t, pc1_xyz, t, condition):
        # S_t: Current flow field [B, N, 3]
        # pc1_xyz: pc1 coordinates [B, 3, N]
        # t: Time [B]
        # condition: Condition from PointKAN-Flow [B, C, N]

        B, _, N = pc1_xyz.shape

        t_sinu_emb = self.time_emb(t, self.time_emb_dim)  # -> [B, time_emb_dim]
        temb = self.time_mlp(t_sinu_emb).unsqueeze(1).expand(-1, N, -1).permute(0, 2, 1) # -> [B, fusion_dim, N]
        S_t_permuted = S_t.permute(0, 2, 1)  # -> [B, 3, N]

        time_condition = temb + condition
        S_t_new = self.pointwise_conv(S_t_permuted)
        final_fusion = S_t_new + time_condition
        if not self.kan_use:
            v_pred_permuted = self.decoder(final_fusion.permute(0, 2, 1))  # Output [B, 3, N]
            return v_pred_permuted  # Return [B, N, 3]
        else:
            final_fusion_t = final_fusion.permute(0, 2, 1)
            v_pred = self.decoder(final_fusion_t)

            return v_pred


if __name__ == "__main__":
    # Configure test parameters
    B = 2  # Batch size
    N = 1024  # Number of point clouds
    C = 192  # Dimension of condition
    time_emb_dim = 64

    # Generate test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc1_xyz = torch.rand(B, 3, N).to(device)  # [B, 3, N]
    S_t = torch.rand(B, N, 3).to(device)  # [B, N, 3]
    t = torch.rand(B).to(device)  # [B]
    condition = torch.rand(B, C, N).to(device)  # [B, C, N]

    print("\n" + "=" * 50)
    print("Testing FlowMatcherDGCNN:")
    print("=" * 50)
    model = FlowMatcherDGCNN(condition_dim=C, time_emb_dim=time_emb_dim).to(device)
    v_pred = model(S_t, pc1_xyz, t, condition)
    print(f"Input: pc1_xyz {pc1_xyz.shape}, S_t {S_t.shape}, t {t.shape}, condition {condition.shape}")
    print(f"Output velocity field: {v_pred.shape} (should be [B, N, 3])")

    print("\n" + "=" * 50)
    print("Testing ConditionalPointFlowMatcher:")
    print("=" * 50)
    model_two = ConditionalPointFlowMatcher(condition_dim=C, time_emb_dim=time_emb_dim,kan_use=True).to(device)
    v_pred_two = model_two(S_t, pc1_xyz, t, condition)
    print(f"Input: pc1_xyz {pc1_xyz.shape}, S_t {S_t.shape}, t {t.shape}, condition {condition.shape}")
    print(f"Output velocity field: {v_pred_two.shape} (should be [B, N, 3])")

    # Backward propagation test
    loss = v_pred.sum() + v_pred_two.sum()
    loss.backward()
    print("\nGradient backpropagation test passed (no errors)")
