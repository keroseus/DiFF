import torch
import torch.nn.functional as F


class SceneFlowMatching:
    """
    Rectified Flow class for 3D scene flow.
    "Car" is the Euler integrator, "Route" is the straight path from random vector field to real scene flow, "Driver" is the loss function.
    """

    def get_doppler_flow(self, pc, target_flow):
        """
        Calculate the radial component (doppler flow) of target_flow in the radar direction.
        Args:
            pc (torch.Tensor): Point cloud coordinates [B, 3, N].
            target_flow (torch.Tensor): Target scene flow [B, N, 3].
        Returns:
            doppler_flow (torch.Tensor): Radial component [B, N, 3].
        """
        # Unit radial vector: direction of pc
        points = pc.permute(0, 2, 1).contiguous()  # [B, N, 3]
        radial_dir = F.normalize(points, p=2, dim=-1)

        # Project onto radial direction: (target_flow · radial_dir) * radial_dir
        doppler_flow = torch.sum(target_flow * radial_dir, dim=-1, keepdim=True) * radial_dir
        return doppler_flow

    def get_orthogonal_noise(self, pc, noise_coefficient=0.01):
        """
        Generate random noise using the legacy source-code procedure.

        The training code passes ``pc`` as [B, 3, N]. The original source
        normalizes along the last dimension and performs the projection along
        that same dimension. Keep that behavior for old-run comparability.
        Args:
            pc (torch.Tensor): Point cloud coordinates [B, 3, N].
            noise_coefficient (float): Direct random-noise coefficient.
        Returns:
            orthogonal_noise (torch.Tensor): Tangential noise [B, 3, N].
        """
        # Legacy source behavior for pc shaped [B, 3, N]. This is not the
        # per-point XYZ normalization used in the newer implementation.
        radial_dir = F.normalize(pc, p=2, dim=-1)  # [B, 3, N]

        # Generate random vector
        random_flow = torch.randn_like(pc) * noise_coefficient  # [B, 3, N]

        # Legacy source projection: the dot product is accumulated over the
        # last dimension, not over the XYZ/channel dimension.
        orthogonal_noise = random_flow - torch.sum(
            random_flow * radial_dir, dim=-1, keepdim=True
        ) * radial_dir

        return orthogonal_noise

    def get_flow_and_noise_Seudo_Doppler(
        self, pc, target_flow, t, noise_coefficient=0.01
    ):
        """
        Generate noise flow with doppler characteristics S_0 = doppler_flow + orthogonal noise.
        Args:
            pc (torch.Tensor): Point cloud coordinates [B, 3, N].
            target_flow (torch.Tensor): Target scene flow [B, N, 3].
            t (torch.Tensor): Time [B].
            noise_coefficient (float): Direct random-noise coefficient.
        Returns:
            S_t (torch.Tensor): Flow field at time t [B, N, 3].
            S_0 (torch.Tensor): Noise flow field [B, N, 3].
        """

        # 1. Calculate doppler component
        doppler_flow = self.get_doppler_flow(pc, target_flow)

        # 2. Generate orthogonal noise
        orthogonal_noise = self.get_orthogonal_noise(
            pc, noise_coefficient=noise_coefficient
        )

        # 3. Combine noise flow
        noise_flow = doppler_flow + orthogonal_noise.permute(0,2,1).contiguous()

        # 4. Interpolate to time t
        t = t.view(-1, 1, 1)
        flow_at_t = t * target_flow + (1 - t) * noise_flow

        return flow_at_t, noise_flow


    '''
    The above functions are all for the case of using pseudo-Doppler motion prior
    '''

    def euler_solver(self, current_flow, velocity, dt):
        """ Update current flow field using Euler method """
        return current_flow + velocity * dt

    def get_flow_and_noise(self, target_flow, t, noise_flow=None):
        """
        Build straight line path from random noise flow to target scene flow.

        Args:
            target_flow (torch.Tensor): Target scene flow S_gt, shape [B, N, 3].
            t (torch.Tensor): Time t, shape [B].
            noise_flow (torch.Tensor, optional): Random noise flow S_0. If None, sampled from standard normal distribution.

        Returns:
            S_t (torch.Tensor): Flow field at time t.
            S_0 (torch.Tensor): Noise flow field.
        """
        if noise_flow is None:
            noise_flow = torch.randn_like(target_flow)

        # Adjust t shape for broadcasting
        t = t.view(-1, 1, 1)

        # Core formula: S_t = t * S_gt + (1 - t) * S_0
        flow_at_t = t * target_flow + (1 - t) * noise_flow

        return flow_at_t, noise_flow

    def loss_fn(
        self,
        predicted_velocity,
        target_flow,
        noise_flow,
        t=None,
        t_weighted=False,
        t_weight_strength=1.0,
        t_weight_power=1.0,
    ):
        """
        Compute loss function for Flow Matching.
        Goal is to make model-predicted velocity field v(S_t, t, y) approximate S_gt - S_0.

        Args:
            predicted_velocity (torch.Tensor): Model-predicted velocity field v.
            target_flow (torch.Tensor): Target scene flow S_gt.
            noise_flow (torch.Tensor): Random noise flow S_0.
        """
        target_velocity = target_flow - noise_flow
        per_element_loss = F.mse_loss(
            predicted_velocity, target_velocity, reduction='none'
        )

        if not t_weighted:
            return per_element_loss.mean()
        if t is None:
            raise ValueError('t is required when t_weighted=True.')

        # Earlier flow-matching times receive more influence. Normalize the
        # weights within each batch to keep the overall loss scale comparable.
        per_sample_loss = per_element_loss.mean(dim=(1, 2))
        t = t.to(dtype=per_sample_loss.dtype).clamp(0.0, 1.0)
        weights = 1.0 + t_weight_strength * (1.0 - t).pow(t_weight_power)
        weights = weights / weights.mean().detach().clamp_min(1e-8)
        return (per_sample_loss * weights).mean()

    def l2_loss_fn_max(self, predicted_velocity, target_flow, noise_flow):
        """
        Compute loss function for Flow Matching, focusing on maximum L2 distance in each batch.

        Args:
            predicted_velocity (torch.Tensor): Model-predicted velocity field v (B, N, 3)
            target_flow (torch.Tensor): Target scene flow S_gt (B, N, 3)
            noise_flow (torch.Tensor): Random noise flow S_0 (B, N, 3)
        """
        # Compute target velocity
        target_velocity = target_flow - noise_flow  # (B, N, 3)

        # Compute squared error for each point (square of L2 distance)
        squared_errors = (predicted_velocity - target_velocity) ** 2  # (B, N, 3)

        # Compute sum of squared errors for each point (point-wise squared error)
        per_point_loss = torch.sum(squared_errors, dim=-1)  # (B, N)

        # Find maximum L2 distance (squared error) in each batch
        max_per_batch, _ = torch.max(per_point_loss, dim=-1)  # (B,)

        # Return sum of maximum values in each batch as final loss
        loss = torch.sum(max_per_batch)  # scalar

        return loss
