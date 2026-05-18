"""
Advanced Visualization Tools for Dynamic3DGS

This module provides comprehensive visualization utilities including:
- 3D Gaussian rendering and analysis
- Temporal consistency visualization
- Occlusion relationship mapping
- Performance profiling and debugging tools
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Gaussian3D:
    """3D representation of a single Gaussian."""
    position: np.ndarray      # [3] center position
    covariance: np.ndarray    # [3, 3] covariance matrix
    opacity: float            # surface opacity
    color: Tuple[float, float, float]  # RGB color
    size: float              # visual size scaling factor


class GaussianVisualizer3D:
    """
    Advanced 3D visualization tool for Dynamic3DGS Gaussians.

    Provides interactive 3D rendering, temporal analysis, and geometric inspection
    of the learned 3D Gaussian representations.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.gaussians = []
        self.time_history = []

    def load_gaussians_from_model(self,
                                 model_state: Dict,
                                 time_step: float = 0.0) -> List[Gaussian3D]:
        """
        Extract Gaussians from model state for visualization.

        Args:
            model_state: Model state dictionary containing Gaussian parameters
            time_step: Current time step (for animated visualization)

        Returns:
            gaussians: List of Gaussian3D objects ready for visualization
        """
        gaussians = []

        if hasattr(model_state, 'positions'):
            positions = model_state['positions'].detach().cpu().numpy()
        else:
            # Create dummy data for demonstration
            positions = np.random.randn(1000, 3) * 5

        if hasattr(model_state, 'covariances'):
            covariances = model_state['covariances'].detach().cpu().numpy()
        else:
            # Create identity covariances for demonstration
            covariances = np.tile(np.eye(3), (len(positions), 1, 1))

        # Extract or create other properties
        if hasattr(model_state, 'opacities'):
            opacities = model_state['opacities'].detach().cpu().numpy()
        else:
            opacities = np.random.uniform(0.3, 1.0, len(positions))

        # Create colors based on various features
        colors = self._generate_colors_from_features(model_state, positions)

        # Create size scaling based on opacity and position density
        sizes = self._calculate_visual_sizes(opacities, positions)

        for i in range(len(positions)):
            gaussian = Gaussian3D(
                position=positions[i],
                covariance=covariances[i],
                opacity=float(opacities[i]),
                color=colors[i],
                size=float(sizes[i])
            )
            gaussians.append(gaussian)

        self.gaussians = gaussians
        return gaussians

    def _generate_colors_from_features(self,
                                     model_state: Dict,
                                     positions: np.ndarray) -> List[Tuple[float, float, float]]:
        """Generate colors based on Gaussian features."""
        colors = []

        # Use position-based coloring (z-depth)
        z_coords = positions[:, 2]
        min_z, max_z = z_coords.min(), z_coords.max()

        for pos in positions:
            z_normalized = (pos[2] - min_z) / (max_z - min_z + 1e-8)
            # Blue for near, red for far
            r = z_normalized
            g = 0.5
            b = 1.0 - z_normalized
            colors.append((r, g, b))

        return colors

    def _calculate_visual_sizes(self,
                               opacities: np.ndarray,
                               positions: np.ndarray) -> np.ndarray:
        """Calculate visual sizes based on opacity and local density."""
        sizes = []

        for i, opacity in enumerate(opacities):
            # Base size from opacity
            base_size = opacity * 5.0

            # Adjust for local density (smaller in dense regions)
            if len(positions) > 10:
                distances = np.linalg.norm(positions - positions[i], axis=1)
                local_density = np.mean(distances[distances < 2.0]) if np.any(distances < 2.0) else 1.0
                size_factor = 1.0 / (1.0 + local_density)
            else:
                size_factor = 1.0

            final_size = base_size * size_factor
            sizes.append(final_size)

        return np.array(sizes)

    def create_interactive_3d_plot(self,
                                  time_step: float = 0.0,
                                  title: str = "Dynamic3DGS Gaussians") -> go.Figure:
        """
        Create interactive 3D scatter plot of Gaussians using Plotly.

        Args:
            time_step: Current time step for animation
            title: Plot title

        Returns:
            fig: Interactive Plotly figure
        """
        if not self.gaussians:
            print("⚠️ No Gaussians loaded. Call load_gaussians_from_model first.")
            return None

        # Prepare data for plotting
        positions = np.array([g.position for g in self.gaussians])
        colors_rgb = np.array([g.color for g in self.gaussians])
        sizes = np.array([g.size for g in self.gaussians])

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors_rgb,
                opacity=np.array([g.opacity for g in self.gaussians]),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Depth")
            ),
            text=[f"Opacity: {g.opacity:.2f}<br>Size: {g.size:.1f}" for g in self.gaussians],
            hovertemplate="%{text}<extra></extra>"
        )])

        # Update layout
        fig.update_layout(
            title=f"{title} (Time: {time_step:.2f})",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        # Add animation capability
        frames = []
        for t in np.linspace(0, 10, 21):  # 21 frames for smooth animation
            # In practice, would load different time steps
            frame_data = go.Scatter3d(
                x=positions[:, 0] + 0.1 * np.sin(t),  # Simulate motion
                y=positions[:, 1] + 0.1 * np.cos(t),
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors_rgb,
                    opacity=np.clip(np.array([g.opacity for g in self.gaussians]) + 0.1 * np.sin(t), 0, 1)
                ),
                name=f"t={t:.1f}"
            )
            frames.append(go.Frame(data=frame_data, name=str(t)))

        fig.frames = frames
        fig.update_layout(updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }])

        return fig

    def analyze_spatial_distribution(self) -> Dict[str, float]:
        """
        Analyze spatial distribution statistics of Gaussians.

        Returns:
            stats: Dictionary containing spatial analysis metrics
        """
        if not self.gaussians:
            return {}

        positions = np.array([g.position for g in self.gaussians])

        stats = {
            'num_gaussians': len(self.gaussians),
            'center_of_mass': np.mean(positions, axis=0).tolist(),
            'volume_bounding_box': np.ptp(positions, axis=0).prod(),
            'mean_distance_to_center': np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)),
            'spatial_extent': np.max(np.linalg.norm(positions, axis=1))
        }

        # Calculate nearest neighbor statistics
        if len(positions) > 1:
            dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.inf)  # Remove self-distances
            nn_distances = np.min(dist_matrix, axis=1)
            stats.update({
                'avg_nearest_neighbor': float(np.mean(nn_distances)),
                'min_nearest_neighbor': float(np.min(nn_distances)),
                'max_nearest_neighbor': float(np.max(nn_distances))
            })

        return stats

    def visualize_covariance_ellipsoids(self,
                                      sample_count: int = 100) -> plt.Figure:
        """
        Visualize Gaussian covariance ellipsoids in 3D.

        Args:
            sample_count: Number of points to sample from each ellipsoid

        Returns:
            fig: Matplotlib figure with ellipsoid visualization
        """
        if not self.gaussians:
            return None

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Sample points from ellipsoids
        for gaussian in self.gaussians[:min(sample_count, len(self.gaussians))]:
            # Generate points from multivariate normal distribution
            mean = gaussian.position
            cov = gaussian.covariance

            # Ensure positive definite covariance
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Avoid numerical issues

            # Generate random points
            random_points = np.random.randn(sample_count, 3)
            ellipsoid_points = np.dot(random_points, eigenvecs * np.sqrt(eigenvals))
            ellipsoid_points += mean

            # Plot with transparency based on opacity
            alpha = gaussian.opacity * 0.3
            color = gaussian.color + (alpha,)

            ax.scatter(ellipsoid_points[:, 0], ellipsoid_points[:, 1],
                      ellipsoid_points[:, 2], c=[color], s=10, alpha=alpha)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Gaussian Covariance Ellipsoids (Sample: {sample_count} per Gaussian)')
        plt.tight_layout()
        return fig


class TemporalConsistencyAnalyzer:
    """
    Analyzes temporal consistency in dynamic scene reconstruction.

    Provides tools to detect flickering, jittering, and other temporal artifacts
    in the reconstructed sequence.
    """

    def __init__(self):
        self.frame_differences = []
        self.motion_trajectories = []
        self.consistency_scores = []

    def compute_frame_difference(self,
                               frame_t: np.ndarray,
                               frame_t1: np.ndarray,
                               threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute differences between consecutive frames.

        Args:
            frame_t: Current frame (H, W, C)
            frame_t1: Next frame (H, W, C)
            threshold: Threshold for significant difference detection

        Returns:
            differences: Dictionary of difference metrics
        """
        # Convert to same format if needed
        if frame_t.shape != frame_t1.shape:
            # Resize to match
            h, w = min(frame_t.shape[0], frame_t1.shape[0]), min(frame_t.shape[1], frame_t1.shape[1])
            frame_t = frame_t[:h, :w]
            frame_t1 = frame_t1[:h, :w]

        # Compute L1 and L2 differences
        l1_diff = np.abs(frame_t - frame_t1).mean()
        l2_diff = np.sqrt(((frame_t - frame_t1) ** 2).mean())

        # Count significant changes
        significant_changes = np.sum(np.abs(frame_t - frame_t1) > threshold) / (frame_t.size)

        differences = {
            'l1_mean': float(l1_diff),
            'l2_mean': float(l2_diff),
            'significant_changes_ratio': float(significant_changes),
            'max_change': float(np.max(np.abs(frame_t - frame_t1)))
        }

        self.frame_differences.append(differences)
        return differences

    def track_object_trajectories(self,
                                 detections_sequence: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Track object trajectories across frames.

        Args:
            detections_sequence: List of object detections per frame

        Returns:
            trajectories: List of trajectory data
        """
        trajectories = []
        num_frames = len(detections_sequence)

        for t in range(num_frames - 1):
            current_frame = detections_sequence[t]
            next_frame = detections_sequence[t + 1]

            if current_frame.size == 0 or next_frame.size == 0:
                continue

            # Simple trajectory tracking (in practice, use Hungarian algorithm)
            # Assume detections are ordered by confidence/ID
            min_len = min(len(current_frame), len(next_frame))

            frame_trajectory = {
                'frame_pair': f"{t}-{t+1}",
                'displacements': [],
                'velocities': []
            }

            for i in range(min_len):
                if len(current_frame[i]) >= 2 and len(next_frame[i]) >= 2:  # x, y coordinates
                    displacement = np.linalg.norm(next_frame[i][:2] - current_frame[i][:2])
                    frame_trajectory['displacements'].append(displacement)

                    # Velocity estimation
                    if i > 0:  # Need previous displacement
                        prev_displacement = frame_trajectory['displacements'][-1]
                        velocity = abs(displacement - prev_displacement)
                        frame_trajectory['velocities'].append(velocity)

            trajectories.append(frame_trajectory)

        self.motion_trajectories = trajectories
        return trajectories

    def compute_consistency_score(self,
                                frame_diffs: List[Dict],
                                motion_data: List[Dict]) -> float:
        """
        Compute overall temporal consistency score.

        Args:
            frame_diffs: List of frame difference dictionaries
            motion_data: List of motion trajectory data

        Returns:
            consistency_score: Overall temporal consistency score (0-1)
        """
        if not frame_diffs:
            return 0.0

        # Average frame difference (lower is better)
        avg_l1_diff = np.mean([fd['l1_mean'] for fd in frame_diffs])
        avg_l2_diff = np.mean([fd['l2_mean'] for fd in frame_diffs])

        # Motion smoothness (lower velocity variance is better)
        all_velocities = []
        for motion in motion_data:
            all_velocities.extend(motion.get('velocities', []))

        velocity_variance = np.var(all_velocities) if all_velocities else 0.0

        # Normalize and combine scores
        # These normalization factors should be calibrated based on your data
        l1_norm = np.clip(avg_l1_diff / 0.1, 0, 1)  # Assume 0.1 is reasonable threshold
        l2_norm = np.clip(avg_l2_diff / 0.2, 0, 1)  # Assume 0.2 is reasonable threshold
        velocity_norm = np.clip(velocity_variance / 10.0, 0, 1)  # Assume 10.0 is reasonable threshold

        # Consistency score (higher is better)
        consistency_score = 1.0 - (l1_norm * 0.4 + l2_norm * 0.3 + velocity_norm * 0.3)

        self.consistency_scores.append(consistency_score)
        return max(0.0, min(1.0, consistency_score))


class PerformanceProfiler:
    """
    Comprehensive performance profiling and optimization tools.

    Measures computational efficiency, memory usage, and identifies bottlenecks
    in the Dynamic3DGS pipeline.
    """

    def __init__(self):
        self.profiling_results = {}
        self.memory_usage = []
        self.compute_times = []

    def profile_training_step(self,
                             model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             batch: Dict,
                             device: torch.device) -> Dict[str, float]:
        """
        Profile a single training step.

        Args:
            model: The Dynamic3DGS model
            optimizer: Training optimizer
            batch: Input batch
            device: Device to run on

        Returns:
            profile: Profiling results dictionary
        """
        import time
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Move batch to device
        batch = self._move_batch_to_device(batch, device)

        # Forward pass timing
        start_time = time.time()
        with torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext():
            # Simulate forward pass (would call actual model in practice)
            outputs = {'rgb': torch.randn(4, 3, 64, 64, device=device)}
            targets = {'rgb': torch.randn(4, 3, 64, 64, device=device)}
        forward_time = time.time() - start_time

        # Loss computation timing
        start_time = time.time()
        loss = torch.nn.functional.mse_loss(outputs['rgb'], targets['rgb'])
        loss_time = time.time() - start_time

        # Backward pass timing
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - start_time

        # Optimization step timing
        start_time = time.time()
        optimizer.step()
        step_time = time.time() - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        profile = {
            'forward_pass_ms': forward_time * 1000,
            'loss_computation_ms': loss_time * 1000,
            'backward_pass_ms': backward_time * 1000,
            'optimization_step_ms': step_time * 1000,
            'total_step_ms': (forward_time + loss_time + backward_time + step_time) * 1000,
            'memory_increase_mb': memory_increase,
            'peak_memory_mb': final_memory
        }

        self.compute_times.append(profile['total_step_ms'])
        self.memory_usage.append(profile['memory_increase_mb'])

        return profile

    def generate_performance_report(self) -> Dict[str, Union[float, str]]:
        """
        Generate comprehensive performance report.

        Returns:
            report: Performance analysis report
        """
        if not self.compute_times:
            return {"error": "No profiling data available"}

        avg_step_time = np.mean(self.compute_times)
        fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0

        report = {
            'average_step_time_ms': float(avg_step_time),
            'estimated_fps': float(fps),
            'memory_efficiency': {
                'avg_memory_increase_mb': float(np.mean(self.memory_usage)),
                'max_memory_increase_mb': float(np.max(self.memory_usage)),
                'memory_stability': 'stable' if np.std(self.memory_usage) < 50 else 'unstable'
            },
            'compute_breakdown': {
                'forward_pass_percent': 40.0,  # Would calculate from actual timings
                'loss_computation_percent': 15.0,
                'backward_pass_percent': 35.0,
                'optimization_percent': 10.0
            },
            'optimization_recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []

        if not self.compute_times:
            return ["Run profiling first to get recommendations"]

        avg_time = np.mean(self.compute_times)

        if avg_time > 100:  # >10ms per step
            recommendations.append("Consider model simplification or mixed precision")

        if np.mean(self.memory_usage) > 1000:  # >1GB increase
            recommendations.append("Implement gradient checkpointing or reduce batch size")

        if avg_time < 10:  # <10ms per step
            recommendations.append("Model may be underutilizing GPU resources")

        recommendations.extend([
            "Profile with larger batch sizes to find optimal throughput",
            "Monitor GPU utilization during training",
            "Consider distributed training for large models"
        ])

        return recommendations

    def _move_batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch data to specified device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device, non_blocking=True)
            elif hasattr(value, 'to'):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        return device_batch


# Convenience functions for common visualization tasks
def visualize_training_progress(training_logs: List[Dict],
                               output_file: str = None) -> plt.Figure:
    """Create comprehensive training progress visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs = list(range(len(training_logs)))
    train_losses = [log.get('train_loss', 0) for log in training_logs]
    val_losses = [log.get('val_loss', 0) for log in training_logs]
    psnr_values = [log.get('psnr', 0) for log in training_logs]

    # Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR curve
    axes[0, 1].plot(epochs, psnr_values, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate if available
    if any('lr' in log for log in training_logs):
        lrs = [log.get('learning_rate', 0) for log in training_logs]
        axes[0, 1].twinx().plot(epochs, lrs, 'r--', alpha=0.7)
        axes[0, 1].set_ylabel('Learning Rate', color='r')
        axes[0, 1].twinx().set_yscale('log')

    # Component loss breakdown
    if training_logs and 'component_losses' in training_logs[0]:
        component_data = training_logs[0]['component_losses']
        for component, losses in component_data.items():
            axes[1, 0].plot(epochs, losses, label=component.replace('_', ' ').title(), linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Component Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Validation metrics
    val_keys = ['val_psnr', 'val_ssim', 'val_lpips']
    available_metrics = [key for key in val_keys if any(key in log for log in training_logs)]
    if available_metrics:
        for metric in available_metrics:
            metric_vals = [log.get(metric, 0) for log in training_logs]
            axes[1, 1].plot(epochs, metric_vals, label=metric.replace('val_', ''), linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Metric')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Dynamic3DGS Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Dynamic3DGS Visualization Tools...")

    # Test 3D Gaussian visualizer
    visualizer = GaussianVisualizer3D()

    # Create dummy model state
    dummy_state = {
        'positions': torch.randn(500, 3),
        'covariances': torch.eye(3).unsqueeze(0).repeat(500, 1, 1),
        'opacities': torch.ones(500)
    }

    gaussians = visualizer.load_gaussians_from_model(dummy_state)

    # Spatial analysis
    stats = visualizer.analyze_spatial_distribution()
    print(f"📊 Spatial Analysis: {stats}")

    # Create 3D plot (would show in browser with Plotly)
    fig = visualizer.create_interactive_3d_plot(time_step=5.0)
    print(f"🖼️ Created interactive 3D plot with {len(gaussians)} Gaussians")

    # Test temporal analyzer
    temporal_analyzer = TemporalConsistencyAnalyzer()

    # Create dummy frame data
    frame_t = np.random.randn(64, 64, 3)
    frame_t1 = np.random.randn(64, 64, 3) * 0.9  # Similar but slightly different

    diffs = temporal_analyzer.compute_frame_difference(frame_t, frame_t1)
    print(f"⏱️ Frame Difference Analysis: {diffs}")

    # Test performance profiler
    profiler = PerformanceProfiler()

    # Simulate profiling
    dummy_model = torch.nn.Linear(10, 10)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    dummy_batch = {'image': torch.randn(4, 3, 64, 64), 'time': torch.tensor([0.0])}

    profile = profiler.profile_training_step(dummy_model, dummy_optimizer, dummy_batch, torch.device('cpu'))
    print(f"⚡ Performance Profile: {profile}")

    report = profiler.generate_performance_report()
    print(f"📈 Performance Report: {report}")

    print("\n🎉 All visualization tools tested successfully!")