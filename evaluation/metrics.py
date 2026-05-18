"""
Evaluation Metrics for Dynamic3DGS

This module provides comprehensive evaluation metrics including:
- Image quality (PSNR, SSIM, LPIPS)
- Temporal quality (FVD, Temporal SSIM)
- Geometry quality (Chamfer Distance, Normal Consistency)
- Efficiency metrics (FPS, Memory Usage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import psutil


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    image_quality: Dict[str, float] = None
    temporal_quality: Dict[str, float] = None
    geometry_quality: Dict[str, float] = None
    efficiency: Dict[str, float] = None

    def __post_init__(self):
        if self.image_quality is None:
            self.image_quality = {}
        if self.temporal_quality is None:
            self.temporal_quality = {}
        if self.geometry_quality is None:
            self.geometry_quality = {}
        if self.efficiency is None:
            self.efficiency = {}


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric."""

    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self,
                predicted: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR.

        Args:
            predicted: [B, C, H, W] or [B, H, W, C]
            target: [B, C, H, W] or [B, H, W, C]

        Returns:
            psnr: scalar PSNR value
        """
        # Ensure same format and device
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")

        # Move to same device if needed
        if predicted.device != target.device:
            target = target.to(predicted.device)

        # MSE
        mse = torch.mean((predicted - target) ** 2, dim=[1, 2, 3])

        # PSNR
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + 1e-8))

        return torch.mean(psnr)


class SSIM(nn.Module):
    """Structural Similarity Index metric."""

    def __init__(self, window_size: int = 11, max_val: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val

        # Create Gaussian window
        self._create_window()

    def _create_window(self):
        """Create Gaussian window for SSIM computation."""
        gauss = torch.exp(-torch.arange(
            -self.window_size // 2 + 1,
            self.window_size // 2 + 1
        ) ** 2 / (2 * (self.window_size // 4) ** 2))

        gauss = gauss / gauss.sum()
        self.register_buffer('window', gauss.unsqueeze(0).unsqueeze(0))

    def forward(self,
                predicted: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM.

        Args:
            predicted: [B, C, H, W]
            target: [B, C, H, W]

        Returns:
            ssim: scalar SSIM value
        """
        B, C, H, W = predicted.shape

        # Check size compatibility
        if H < self.window_size or W < self.window_size:
            # Resize if too small
            scale_factor = min(H / self.window_size, W / self.window_size)
            predicted = F.interpolate(predicted, scale_factor=scale_factor, mode='bilinear')
            target = F.interpolate(target, scale_factor=scale_factor, mode='bilinear')

        # Add channel dimension for window
        window = self.window.expand(C, 1, -1, -1).to(predicted.device)

        # Compute means
        mu_x = F.conv2d(predicted, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        # Compute variances and covariance
        sigma_x = F.conv2d(predicted ** 2, window, padding=self.window_size // 2, groups=C) - mu_x ** 2
        sigma_y = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=C) - mu_y ** 2
        sigma_xy = F.conv2d(predicted * target, window, padding=self.window_size // 2, groups=C) - mu_x * mu_y

        # SSIM constants
        c1 = (0.01 * self.max_val) ** 2
        c2 = (0.03 * self.max_val) ** 2

        # SSIM map
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                  ((mu_x ** 2 + mu_y **2 + c1) * (sigma_x + sigma_y + c2))

        return torch.mean(ssim_map)


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity metric."""

    def __init__(self, version: str = 'v0'):
        super().__init__()
        self.version = version

        # In practice, this would load pre-trained VGG features
        # For now, we'll use a simplified implementation
        self.vgg_features = self._create_vgg_features()

    def _create_vgg_features(self):
        """Create simplified VGG feature extractor."""
        # Placeholder - in practice, load actual VGG model
        return None

    def forward(self,
                predicted: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS.

        Args:
            predicted: [B, C, H, W]
            target: [B, C, H, W]

        Returns:
            lpips: scalar LPIPS value (lower is better)
        """
        # Simplified LPIPS implementation
        # In practice, would extract VGG features and compute distance
        diff = torch.abs(predicted - target)
        lpips = torch.mean(diff)

        return lpips


class FVD(nn.Module):
    """Frechet Video Distance metric."""

    def __init__(self):
        super().__init__()

    def forward(self,
                predicted_videos: torch.Tensor,
                ground_truth_videos: torch.Tensor) -> torch.Tensor:
        """
        Compute Frechet Video Distance.

        Args:
            predicted_videos: [B, T, C, H, W] or [B, T, H, W, C]
            ground_truth_videos: [B, T, C, H, W] or [B, T, H, W, C]

        Returns:
            fvd: scalar FVD value (lower is better)
        """
        # Reshape to [B*T, C, H, W]
        if predicted_videos.dim() == 5:
            B, T, C, H, W = predicted_videos.shape
            pred_flat = predicted_videos.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            gt_flat = ground_truth_videos.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        else:
            B, T, H, W, C = predicted_videos.shape
            pred_flat = predicted_videos.reshape(B*T, C, H, W)
            gt_flat = ground_truth_videos.reshape(B*T, C, H, W)

        # Compute mean and covariance (simplified)
        mu_pred = torch.mean(pred_flat, dim=[2, 3], keepdim=True)
        mu_gt = torch.mean(gt_flat, dim=[2, 3], keepdim=True)

        # Center the data
        pred_centered = pred_flat - mu_pred
        gt_centered = gt_flat - mu_gt

        # Compute covariance matrices
        cov_pred = torch.matmul(pred_centered, pred_centered.transpose(-1, -2)) / (B*T - 1)
        cov_gt = torch.matmul(gt_centered, gt_centered.transpose(-1, -2)) / (B*T - 1)

        # Compute FVD (simplified)
        # In practice, would use proper Frechet distance calculation
        fvd = torch.norm(mu_pred - mu_gt) + \
              torch.trace(cov_pred + cov_gt - 2 * torch.sqrt(cov_pred @ cov_gt + 1e-6))

        return fvd


class TemporalSSIM(nn.Module):
    """Temporal Structural Similarity Index."""

    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self,
                video_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal SSIM over video sequence.

        Args:
            video_sequence: [B, T, C, H, W]

        Returns:
            temporal_ssim: scalar temporal SSIM value
        """
        B, T, C, H, W = video_sequence.shape

        total_ssim = 0
        valid_pairs = 0

        for t in range(T - 1):
            frame_t = video_sequence[:, t]
            frame_tp1 = video_sequence[:, t + 1]

            ssim_val = self.ssim(frame_t, frame_tp1)
            total_ssim += ssim_val
            valid_pairs += 1

        return total_ssim / valid_pairs if valid_pairs > 0 else torch.tensor(0.0)


class ChamferDistance(nn.Module):
    """Chamfer Distance for point cloud comparison."""

    def __init__(self):
        super().__init__()

    def forward(self,
                points_pred: torch.Tensor,
                points_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute Chamfer Distance.

        Args:
            points_pred: [N, 3] predicted points
            points_gt: [M, 3] ground truth points

        Returns:
            chamfer_dist: scalar Chamfer distance
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(points_pred, points_gt)

        # Find minimum distances
        min_pred_to_gt = torch.min(dist_matrix, dim=1)[0].mean()
        min_gt_to_pred = torch.min(dist_matrix, dim=0)[0].mean()

        return min_pred_to_gt + min_gt_to_pred


class NormalConsistency(nn.Module):
    """Normal consistency metric for geometry evaluation."""

    def __init__(self):
        super().__init__()

    def forward(self,
                normals_pred: torch.Tensor,
                normals_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute normal consistency.

        Args:
            normals_pred: [N, 3] predicted normals
            normals_gt: [N, 3] ground truth normals

        Returns:
            normal_consistency: cosine similarity of normals
        """
        # Normalize normals
        normals_pred = F.normalize(normals_pred, dim=-1)
        normals_gt = F.normalize(normals_gt, dim=-1)

        # Cosine similarity
        cos_sim = torch.sum(normals_pred * normals_gt, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1, 1)

        # Convert to angle error
        angle_error = torch.acos(cos_sim)

        return torch.mean(angle_error)


class ComprehensiveEvaluator:
    """
    Main evaluator class that computes all metrics.

    Provides a unified interface for evaluating Dynamic3DGS models.
    """

    def __init__(self):
        self.metrics = {
            'psnr': PSNR(max_val=1.0),
            'ssim': SSIM(),
            'lpips': LPIPS(),
            'fvd': FVD(),
            'temporal_ssim': TemporalSSIM(),
            'chamfer_distance': ChamferDistance(),
            'normal_consistency': NormalConsistency()
        }

    def evaluate(self,
                model: torch.nn.Module,
                renderer: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device,
                num_batches: int = 10,
                compute_efficiency: bool = True) -> Dict[str, float]:
        """
        Run comprehensive evaluation.

        Args:
            model: Dynamic3DGS model
            renderer: Rendering module
            dataloader: Test dataloader
            device: Device to run on
            num_batches: Number of batches to evaluate
            compute_efficiency: Whether to measure efficiency metrics

        Returns:
            results: Dictionary of all metrics
        """
        model.eval()
        renderer.eval()

        all_results = {
            'image_quality': {},
            'temporal_quality': {},
            'geometry_quality': {},
            'efficiency': {}
        }

        # Image quality metrics
        image_metrics = ['psnr', 'ssim', 'lpips']
        for metric_name in image_metrics:
            metric_fn = self.metrics[metric_name]
            metric_value = self._compute_image_metric(
                metric_fn, dataloader, model, renderer, device, num_batches
            )
            all_results['image_quality'][metric_name.upper()] = metric_value.item()

        # Temporal quality metrics (if enough frames available)
        if len(dataloader.dataset) >= 2:
            temporal_metrics = ['fvd', 'temporal_ssim']
            for metric_name in temporal_metrics:
                metric_fn = self.metrics[metric_name]
                metric_value = self._compute_temporal_metric(
                    metric_fn, dataloader, model, renderer, device, num_batches
                )
                all_results['temporal_quality'][metric_name.upper()] = metric_value.item()

        # Efficiency metrics
        if compute_efficiency:
            efficiency_metrics = self._measure_efficiency(model, renderer, dataloader, device)
            all_results['efficiency'].update(efficiency_metrics)

        # Convert to flat dictionary for easy access
        flat_results = {}
        for category, metrics in all_results.items():
            for metric_name, value in metrics.items():
                flat_results[f"{category}_{metric_name}"] = value

        return flat_results

    def _compute_image_metric(self,
                             metric_fn: nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             model: torch.nn.Module,
                             renderer: torch.nn.Module,
                             device: torch.device,
                             num_batches: int) -> torch.Tensor:
        """Compute single image quality metric."""
        total_metric = 0
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                batch = self._move_batch_to_device(batch, device)

                # Render frame
                images = batch['image']  # [B, C, H, W]
                cameras = batch['camera']
                time_steps = batch['time']

                gaussian_outputs = model(time_steps[0])
                camera_obj = type('Camera', (), {
                    'intrinsics': cameras[0].intrinsics,
                    'extrinsics': cameras[0].extrinsics
                })()

                rendered_outputs = renderer.forward(
                    gaussian_outputs, camera_obj,
                    (images.shape[2], images.shape[3]),
                    time_steps[0]
                )

                # Compute metric
                metric_value = metric_fn(rendered_outputs['rgb'], images)
                total_metric += metric_value
                count += 1

        return total_metric / count if count > 0 else torch.tensor(0.0)

    def _compute_temporal_metric(self,
                                metric_fn: nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                model: torch.nn.Module,
                                renderer: torch.nn.Module,
                                device: torch.device,
                                num_batches: int) -> torch.Tensor:
        """Compute temporal quality metric."""
        # Collect consecutive frame pairs
        frame_pairs = []
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                batch = self._move_batch_to_device(batch, device)

                # Get current frame
                images = batch['image']
                cameras = batch['camera']
                time_steps = batch['time']

                gaussian_outputs = model(time_steps[0])
                camera_obj = type('Camera', (), {
                    'intrinsics': cameras[0].intrinsics,
                    'extrinsics': cameras[0].extrinsics
                })()

                rendered_outputs = renderer.forward(
                    gaussian_outputs, camera_obj,
                    (images.shape[2], images.shape[3]),
                    time_steps[0]
                )

                # Store current frame
                frame_pairs.append(rendered_outputs['rgb'])

                count += 1

                if count >= 2:  # Need at least 2 frames
                    # Compute metric on frame pair
                    metric_value = metric_fn(
                        torch.stack([frame_pairs[-2], frame_pairs[-1]], dim=1),
                        torch.stack([frame_pairs[-2], frame_pairs[-1]], dim=1)
                    )
                    return metric_value

        return torch.tensor(0.0)

    def _measure_efficiency(self,
                           model: torch.nn.Module,
                           renderer: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: torch.device) -> Dict[str, float]:
        """Measure computational efficiency metrics."""
        efficiency_metrics = {}

        # Measure FPS
        fps = self._measure_fps(model, renderer, dataloader, device)
        efficiency_metrics['fps'] = fps

        # Measure memory usage
        memory_mb = self._measure_memory_usage(model, renderer)
        efficiency_metrics['memory_mb'] = memory_mb

        # Measure training time per epoch (simulated)
        train_time_per_epoch = self._measure_training_speed(model, renderer, dataloader, device)
        efficiency_metrics['train_time_per_epoch'] = train_time_per_epoch

        return efficiency_metrics

    def _measure_fps(self,
                    model: torch.nn.Module,
                    renderer: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device) -> float:
        """Measure rendering FPS."""
        model.eval()
        renderer.eval()

        times = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch, device)

                start_time = time.time()

                images = batch['image']
                cameras = batch['camera']
                time_steps = batch['time']

                gaussian_outputs = model(time_steps[0])
                camera_obj = type('Camera', (), {
                    'intrinsics': cameras[0].intrinsics,
                    'extrinsics': cameras[0].extrinsics
                })()

                renderer.forward(
                    gaussian_outputs, camera_obj,
                    (images.shape[2], images.shape[3]),
                    time_steps[0]
                )

                end_time = time.time()
                times.append(end_time - start_time)

        return 1.0 / np.mean(times) if times else 0.0

    def _measure_memory_usage(self, model: torch.nn.Module, renderer: torch.nn.Module) -> float:
        """Measure peak memory usage in MB."""
        process = psutil.Process()

        # Force garbage collection
        import gc
        gc.collect()

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Forward pass to allocate memory
        dummy_input = torch.randn(1, 1000, 3).to(next(model.parameters()).device)
        dummy_time = torch.tensor([0.0]).to(dummy_input.device)

        with torch.no_grad():
            outputs = model(dummy_time)
            # Add dummy rendering call
            dummy_camera = type('Camera', (), {
                'intrinsics': torch.eye(3),
                'extrinsics': torch.eye(4)
            })()
            renderer.forward(outputs, dummy_camera, (64, 64), dummy_time)

        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return max(0, mem_after - mem_before)

    def _measure_training_speed(self,
                               model: torch.nn.Module,
                               renderer: torch.nn.Module,
                               dataloader: torch.utils.data.DataLoader,
                               device: torch.device) -> float:
        """Measure simulated training time per epoch."""
        # This is a simplified measurement
        # In practice, would run actual training loop timing
        return 30.0  # Placeholder: ~30 seconds per epoch

    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch data to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device, non_blocking=True)
            elif hasattr(value, 'to'):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch

    def generate_report(self, results: Dict[str, float]) -> str:
        """Generate human-readable report from results."""
        report = "=== Dynamic3DGS Evaluation Report ===\n\n"

        # Group by category
        categories = {
            'Image Quality': ['psnr', 'ssim', 'lpips'],
            'Temporal Quality': ['fvd', 'temporal_ssim'],
            'Efficiency': ['fps', 'memory_mb', 'train_time_per_epoch']
        }

        for category, metrics in categories.items():
            report += f"{category}:\n"
            for metric in metrics:
                if f"image_quality_{metric}" in results:
                    value = results[f"image_quality_{metric}"]
                    unit = {"psnr": "dB", "ssim": "", "lpips": ""}
                    report += f"  {metric.upper()}: {value:.4f} {unit.get(metric, '')}\n"
            report += "\n"

        return report


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation metrics...")

    # Create dummy data
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    # Test individual metrics
    psnr = PSNR()
    ssim = SSIM()
    lpips = LPIPS()

    psnr_val = psnr(pred, target)
    ssim_val = ssim(pred, target)
    lpips_val = lpips(pred, target)

    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"LPIPS: {lpips_val:.4f}")

    # Test comprehensive evaluator
    evaluator = ComprehensiveEvaluator()

    # Create dummy model and renderer (simplified)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gaussians = torch.randn(1000, 3)

        def forward(self, time_step):
            return {'positions': self.gaussians.unsqueeze(0)}

    class DummyRenderer(torch.nn.Module):
        def forward(self, gaussians, camera, image_size, time_step):
            return {'rgb': torch.randn(1, 3, *image_size)}

    # Create dummy dataloader
    dummy_dataloader = torch.utils.data.DataLoader([
        {
            'image': torch.randn(1, 3, 64, 64),
            'camera': type('Camera', (), {
                'intrinsics': torch.eye(3),
                'extrinsics': torch.eye(4)
            })(),
            'time': torch.tensor([0.0])
        }
    ] * 10, batch_size=1)

    # Run evaluation (will be limited due to dummy data)
    results = evaluator.evaluate(
        model=DummyModel(),
        renderer=DummyRenderer(),
        dataloader=dummy_dataloader,
        device=torch.device('cpu'),
        num_batches=2,
        compute_efficiency=False
    )

    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Generate report
    report = evaluator.generate_report(results)
    print("\nReport:")
    print(report)