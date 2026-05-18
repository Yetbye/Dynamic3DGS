"""
Loss Functions for Dynamic3DGS

This module implements various loss functions including:
- Reconstruction losses (L1, SSIM, D-SSIM)
- Temporal consistency losses
- Motion smoothness constraints
- Regularization terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss combining multiple metrics.
    """

    def __init__(self,
                 l1_weight: float = 1.0,
                 ssim_weight: float = 0.8,
                 dssim_weight: float = 0.5,
                 perceptual_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.dssim_weight = dssim_weight
        self.perceptual_weight = perceptual_weight

        # D-SSIM parameters
        self.window_size = 11
        self.channel = 3
        self.size_average = True

        # Perceptual loss (placeholder - would use VGG features in practice)
        self.perceptual_loss = nn.L1Loss()

    def forward(self,
                predicted: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            predicted: [B, C, H, W] or [B, H, W, C] predicted image
            target: [B, C, H, W] or [B, H, W, C] ground truth image
            mask: Optional mask for valid regions

        Returns:
            losses: Dictionary of individual loss components
        """
        # Ensure same format
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")

        # Apply mask if provided
        if mask is not None:
            predicted = predicted * mask
            target = target * mask

        losses = {}

        # L1 loss
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(predicted, target)
            losses['l1'] = l1_weight * l1_loss

        # SSIM loss
        if self.ssim_weight > 0:
            ssim_loss = 1 - self._ssim(predicted, target)
            losses['ssim'] = self.ssim_weight * ssim_loss

        # D-SSIM loss (differentiable SSIM)
        if self.dssim_weight > 0:
            dssim_loss = self._dssim(predicted, target)
            losses['dssim'] = self.dssim_weight * dssim_loss

        # Perceptual loss
        if self.perceptual_weight > 0:
            # Resize to smaller resolution for efficiency
            pred_small = F.interpolate(predicted, size=(64, 64), mode='bilinear', align_corners=False)
            target_small = F.interpolate(target, size=(64, 64), mode='bilinear', align_corners=False)
            perceptual_loss = self.perceptual_loss(pred_small, target_small)
            losses['perceptual'] = self.perceptual_weight * perceptual_loss

        # Total reconstruction loss
        total_loss = sum(losses.values())

        return {
            'total': total_loss,
            'components': losses
        }

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM score."""
        # Simplified SSIM implementation
        # In practice, would use proper windowed computation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size//2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size//2)

        sigma_x = F.avg_pool2d(x*x, self.window_size, stride=1, padding=self.window_size//2) - mu_x**2
        sigma_y = F.avg_pool2d(y*y, self.window_size, stride=1, padding=self.window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, self.window_size, stride=1, padding=self.window_size//2) - mu_x*mu_y

        ssim_map = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2)) / \
                  ((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))

        return ssim_map.mean()

    def _dssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute differentiable SSIM loss."""
        return 1 - self._ssim(x, y)


class TemporalConsistencyLoss(nn.Module):
    """
    Loss that enforces temporal consistency across frames.
    """

    def __init__(self,
                 flow_weight: float = 1.0,
                 motion_smoothness_weight: float = 0.1,
                 rigidity_constraint_weight: float = 0.05,
                 occlusion_aware: bool = True):
        super().__init__()
        self.flow_weight = flow_weight
        self.motion_smoothness_weight = motion_smoothness_weight
        self.rigidity_constraint_weight = rigidity_constraint_weight
        self.occlusion_aware = occlusion_aware

        self.motion_smoothness_loss = MotionSmoothnessLoss()
        self.rigidity_constraint_loss = RigidityConstraintLoss()

    def forward(self,
                current_frame: torch.Tensor,
                previous_frame: torch.Tensor,
                optical_flow: Optional[torch.Tensor] = None,
                gaussian_deformations: Optional[torch.Tensor] = None,
                time_steps: Optional[torch.Tensor] = None,
                occlusion_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute temporal consistency loss.

        Args:
            current_frame: Current frame prediction
            previous_frame: Previous frame prediction
            optical_flow: Ground truth optical flow
            gaussian_deformations: Gaussian position deformations
            time_steps: Time step values
            occlusion_mask: Occlusion-aware mask

        Returns:
            losses: Dictionary of temporal loss components
        """
        losses = {}

        # Optical flow consistency
        if self.flow_weight > 0 and optical_flow is not None:
            flow_consistency_loss = self._flow_consistency_loss(
                current_frame, previous_frame, optical_flow, occlusion_mask
            )
            losses['flow_consistency'] = self.flow_weight * flow_consistency_loss

        # Motion smoothness
        if self.motion_smoothness_weight > 0 and gaussian_deformations is not None:
            smoothness_loss = self.motion_smoothness_loss(gaussian_deformations, time_steps)
            losses['motion_smoothness'] = self.motion_smoothness_weight * smoothness_loss

        # Rigidity constraint
        if self.rigidity_constraint_weight > 0:
            rigidity_loss = self.rigidity_constraint_loss(current_frame, previous_frame)
            losses['rigidity_constraint'] = self.rigidity_constraint_weight * rigidity_loss

        # Total temporal loss
        total_loss = sum(losses.values())

        return {
            'total': total_loss,
            'components': losses
        }

    def _flow_consistency_loss(self,
                              current_frame: torch.Tensor,
                              previous_frame: torch.Tensor,
                              optical_flow: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute optical flow consistency loss.

        Args:
            current_frame: Current frame
            previous_frame: Previous frame
            optical_flow: Ground truth optical flow
            mask: Optional mask

        Returns:
            loss: Flow consistency loss
        """
        # Warp previous frame according to optical flow
        B, C, H, W = current_frame.shape
        grid = self._create_grid(B, H, W, current_frame.device)

        # Apply optical flow to create warped frame
        warped_previous = self._warp_image(previous_frame, optical_flow)

        # Compute difference
        diff = current_frame - warped_previous

        # Apply mask if provided
        if mask is not None:
            diff = diff * mask.unsqueeze(1)

        # L1 loss on warped difference
        loss = F.l1_loss(diff, torch.zeros_like(diff))

        return loss

    def _create_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create normalized grid for warping."""
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def _warp_image(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image using optical flow."""
        B, C, H, W = image.shape
        grid = self._create_grid(B, H, W, image.device)

        # Add flow to grid
        flow_norm = torch.cat([
            flow[:, :2],  # dx, dy
            torch.zeros_like(flow[:, :1])  # dz (identity)
        ], dim=1)

        warped_grid = grid + flow_norm.permute(0, 2, 3, 1)

        # Sample image at warped locations
        warped_image = F.grid_sample(
            image, warped_grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped_image


class MotionSmoothnessLoss(nn.Module):
    """
    Loss that encourages smooth motion over time.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self,
                deformation_field: torch.Tensor,
                time_steps: torch.Tensor) -> torch.Tensor:
        """
        Compute motion smoothness loss.

        Args:
            deformation_field: [B, T, N, 3] deformation field over time
            time_steps: [B, T] time values

        Returns:
            loss: Motion smoothness loss
        """
        # Compute first-order differences (velocity)
        velocities = deformation_field[:, 1:] - deformation_field[:, :-1]

        # Compute second-order differences (acceleration)
        accelerations = velocities[:, 1:] - velocities[:, :-1]

        # L2 norm of accelerations (penalize rapid changes)
        acceleration_loss = torch.mean(torch.norm(accelerations, dim=-1))

        return self.weight * acceleration_loss


class RigidityConstraintLoss(nn.Module):
    """
    Loss that encourages rigid motion in static regions.
    """

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self,
                current_frame: torch.Tensor,
                previous_frame: torch.Tensor) -> torch.Tensor:
        """
        Compute rigidity constraint loss.

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            loss: Rigidity constraint loss
        """
        # Compute optical flow between frames (simplified)
        flow = current_frame - previous_frame

        # Encourage small flow magnitudes (rigid motion assumption)
        flow_magnitude = torch.norm(flow, dim=1, keepdim=True)
        rigidity_loss = torch.mean(F.relu(flow_magnitude - 0.1))  # Threshold for rigid motion

        return self.weight * rigidity_loss


class RegularizationLoss(nn.Module):
    """
    Various regularization terms for the model.
    """

    def __init__(self,
                 opacity_regularization_weight: float = 0.1,
                 sh_coefficients_regularization_weight: float = 0.01,
                 covariance_regularization_weight: float = 0.01,
                 deformation_l2_weight: float = 0.01):
        super().__init__()
        self.opacity_regularization_weight = opacity_regularization_weight
        self.sh_coefficients_regularization_weight = sh_coefficients_regularization_weight
        self.covariance_regularization_weight = covariance_regularization_weight
        self.deformation_l2_weight = deformation_l2_weight

    def forward(self,
                gaussians: Dict[str, torch.Tensor],
                deformation_field: Optional[nn.Module] = None,
                time_steps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.

        Args:
            gaussians: Gaussian parameters dictionary
            deformation_field: Deformation field network
            time_steps: Time steps

        Returns:
            losses: Dictionary of regularization losses
        """
        losses = {}

        # Opacity regularization (encourage reasonable opacity values)
        if self.opacity_regularization_weight > 0:
            opacity = gaussians['opacities']
            # Encourage opacities to be in reasonable range
            opacity_reg = torch.mean(F.relu(opacity - 1.0) + F.relu(0.0 - opacity))
            losses['opacity_regularization'] = self.opacity_regularization_weight * opacity_reg

        # SH coefficients regularization
        if self.sh_coefficients_regularization_weight > 0:
            sh_coeffs = gaussians['sh_coefficients']
            # Encourage small SH coefficients (smooth lighting)
            sh_reg = torch.mean(sh_coeffs ** 2)
            losses['sh_coefficients_regularization'] = \
                self.sh_coefficients_regularization_weight * sh_reg

        # Covariance regularization
        if self.covariance_regularization_weight > 0:
            covariances = gaussians['covariances']
            # Encourage reasonable covariance values
            # Penalize very large or very small covariances
            log_cov = torch.log(covariances + 1e-8)
            cov_reg = torch.mean(log_cov ** 2)
            losses['covariance_regularization'] = \
                self.covariance_regularization_weight * cov_reg

        # Deformation L2 regularization
        if self.deformation_l2_weight > 0 and deformation_field is not None and time_steps is not None:
            # Compute L2 norm of deformation field weights
            deformation_weights = []
            for param in deformation_field.parameters():
                if param.requires_grad:
                    deformation_weights.append(param.view(-1))
            if deformation_weights:
                weight_norm = torch.cat(deformation_weights).norm()
                losses['deformation_l2'] = self.deformation_l2_weight * weight_norm

        # Total regularization loss
        total_loss = sum(losses.values())

        return {
            'total': total_loss,
            'components': losses
        }


class VectorQuantizationLoss(nn.Module):
    """
    Loss for vector quantization component.
    """

    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self,
                quantized: torch.Tensor,
                inputs: torch.Tensor,
                encoding_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute vector quantization loss.

        Args:
            quantized: Quantized vectors
            inputs: Original input vectors
            encoding_indices: Indices of selected embeddings

        Returns:
            losses: Quantization loss components
        """
        # Commitment loss: encourage encoder outputs to be close to quantized vectors
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)

        # Codebook loss: encourage quantized vectors to be close to encoder outputs
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        # Total quantization loss
        quant_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return {
            'total': quant_loss,
            'codebook_loss': q_latent_loss,
            'commitment_loss': self.commitment_cost * e_latent_loss
        }


class TotalLoss(nn.Module):
    """
    Combined total loss function for training Dynamic3DGS.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Individual loss modules
        self.reconstruction_loss = ReconstructionLoss(
            l1_weight=config.get('reconstruction', {}).get('l1_weight', 1.0),
            ssim_weight=config.get('reconstruction', {}).get('ssim_weight', 0.8),
            dssim_weight=config.get('reconstruction', {}).get('dssim_weight', 0.5),
            perceptual_weight=config.get('reconstruction', {}).get('perceptual_weight', 0.1)
        )

        self.temporal_loss = TemporalConsistencyLoss(
            flow_weight=config.get('temporal', {}).get('flow_weight', 1.0),
            motion_smoothness_weight=config.get('temporal', {}).get('motion_smoothness_weight', 0.1),
            rigidity_constraint_weight=config.get('temporal', {}).get('rigidity_constraint_weight', 0.05)
        )

        self.regularization_loss = RegularizationLoss(
            opacity_regularization_weight=config.get('regularization', {}).get('opacity_weight', 0.1),
            sh_coefficients_regularization_weight=config.get('regularization', {}).get('sh_weight', 0.01),
            covariance_regularization_weight=config.get('regularization', {}).get('covariance_weight', 0.01),
            deformation_l2_weight=config.get('regularization', {}).get('deformation_l2_weight', 0.01)
        )

        self.vector_quantization_loss = VectorQuantizationLoss(
            commitment_cost=config.get('model', {}).get('commitment_cost', 0.25)
        )

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                gaussians: Optional[Dict[str, torch.Tensor]] = None,
                deformation_field: Optional[nn.Module] = None,
                time_steps: Optional[torch.Tensor] = None,
                additional_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            gaussians: Gaussian parameters
            deformation_field: Deformation field network
            time_steps: Time steps
            additional_data: Additional data for losses

        Returns:
            total_loss_dict: Complete loss breakdown
        """
        losses = {}

        # Reconstruction loss
        recon_data = additional_data.get('recon_data', {})
        recon_loss_dict = self.reconstruction_loss(
            predictions['rgb'], targets['rgb'],
            mask=recon_data.get('mask')
        )
        losses.update({f'recon_{k}': v for k, v in recon_loss_dict.items()})

        # Temporal consistency loss
        temporal_data = additional_data.get('temporal_data', {})
        temporal_loss_dict = self.temporal_loss(
            predictions['rgb'], temporal_data.get('previous_rgb'),
            optical_flow=temporal_data.get('optical_flow'),
            gaussian_deformations=temporal_data.get('gaussian_deformations'),
            time_steps=time_steps,
            occlusion_mask=temporal_data.get('occlusion_mask')
        )
        losses.update({f'temporal_{k}': v for k, v in temporal_loss_dict.items()})

        # Regularization loss
        if gaussians is not None:
            reg_loss_dict = self.regularization_loss(gaussians, deformation_field, time_steps)
            losses.update({f'reg_{k}': v for k, v in reg_loss_dict.items()})

        # Vector quantization loss
        if 'quantized' in predictions and 'inputs' in predictions:
            quant_loss_dict = self.vector_quantization_loss(
                predictions['quantized'], predictions['inputs'],
                predictions.get('encoding_indices', torch.zeros(1))
            )
            losses.update({f'quant_{k}': v for k, v in quant_loss_dict.items()})

        # Weighted total loss
        total_loss = sum(w * losses[f'total'] for w, f in zip(
            [self.config.get('reconstruction', {}).get('weight', 1.0),
             self.config.get('temporal', {}).get('weight', 0.1),
             self.config.get('regularization', {}).get('weight', 0.01)],
            ['recon_total', 'temporal_total', 'reg_total']
        ))

        return {
            'total_loss': total_loss,
            'detailed': losses
        }


if __name__ == "__main__":
    # Example usage
    print("Testing loss functions...")

    # Test reconstruction loss
    recon_loss = ReconstructionLoss(l1_weight=1.0, ssim_weight=0.8)
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss_dict = recon_loss(pred, target)
    print(f"Reconstruction loss: {loss_dict['total']:.4f}")
    print(f"Components: {list(loss_dict['components'].keys())}")

    # Test temporal loss
    temporal_loss = TemporalConsistencyLoss(flow_weight=1.0, motion_smoothness_weight=0.1)
    current = torch.randn(2, 3, 64, 64)
    previous = torch.randn(2, 3, 64, 64)
    flow = torch.randn(2, 2, 64, 64)
    temporal_dict = temporal_loss(current, previous, optical_flow=flow)
    print(f"Temporal loss: {temporal_dict['total']:.4f}")

    # Test regularization loss
    reg_loss = RegularizationLoss()
    gaussians = {
        'opacities': torch.ones(1000),
        'sh_coefficients': torch.randn(1000, 3, 16),
        'covariances': torch.eye(3).unsqueeze(0).repeat(1000, 1, 1)
    }
    reg_dict = reg_loss(gaussians)
    print(f"Regularization loss: {reg_dict['total']:.4f}")

    # Test total loss
    total_loss = TotalLoss({
        'reconstruction': {'weight': 1.0},
        'temporal': {'weight': 0.1},
        'regularization': {'weight': 0.01}
    })
    print("Total loss module created successfully")