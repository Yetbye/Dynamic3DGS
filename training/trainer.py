"""
Training Pipeline for Dynamic3DGS

This module implements the main training loop with:
- Multi-GPU support
- Mixed precision training
- Checkpointing and logging
- Validation and testing
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from models.deformable_gaussian import DeformableGaussianModel
from models.renderer import OcclusionAwareRenderer, RenderConfig
from models.losses import TotalLoss
from data.datasets import create_data_loaders
from evaluation.metrics import ComprehensiveEvaluator
from utils.tensorboard_logger import TensorBoardLogger
from utils.wandb_logger import WandBLogger


@dataclass
class TrainerConfig:
    """Configuration for the trainer"""
    experiment_name: str = "dynamic_3dgs"
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    find_unused_parameters: bool = False

    # Training parameters
    max_epochs: int = 1000
    warmup_epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 100
    save_best_only: bool = True
    monitor_metric: str = "val_psnr"
    monitor_mode: str = "max"

    # Hardware
    num_workers: int = 8
    pin_memory: bool = True


class Dynamic3DGSTrainer(nn.Module):
    """
    Main trainer class for Dynamic3DGS.

    Handles the complete training pipeline including:
    - Model initialization and optimization
    - Forward/backward passes
    - Loss computation
    - Validation and testing
    - Checkpointing and logging
    """

    def __init__(self,
                 config: TrainerConfig,
                 model_config: Dict,
                 dataset_config: Dict,
                 experiment_dir: str = "./experiments"):
        super().__init__()

        self.config = config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.experiment_dir = experiment_dir

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create experiment directory
        self.exp_dir = os.path.join(experiment_dir, config.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Initialize components
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_dataloaders()
        self._setup_evaluator()
        self._setup_loggers()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf') if config.monitor_mode == "max" else float('inf')

        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    def _setup_model(self):
        """Initialize the Dynamic3DGS model"""
        # Create deformable Gaussian model
        self.model = DeformableGaussianModel(
            num_gaussians=self.model_config.get('num_gaussians', 10000),
            max_time_steps=self.model_config.get('max_time_steps', 100),
            embedding_dim=self.model_config.get('embedding_dim', 64),
            num_embeddings=self.model_config.get('num_embeddings', 512)
        )

        # Create renderer
        render_config = RenderConfig(
            depth_sorting=True,
            alpha_blending=True,
            occlusion_threshold=self.model_config.get('renderer', {}).get('occlusion_threshold', 0.5)
        )
        self.renderer = OcclusionAwareRenderer(render_config)

        # Move to device
        self.model.to(self.device)
        self.renderer.to(self.device)

        # Enable mixed precision training
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.model = self.model.half()
            self.renderer = self.renderer.half()

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        # Separate parameters for different parts of the model
        deformation_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'deformation_field' in name or 'occlusion_predictor' in name:
                    deformation_params.append(param)
                else:
                    other_params.append(param)

        # Group parameters by learning rate (deformation field gets lower LR)
        param_groups = [
            {'params': other_params, 'lr': self.config.learning_rate},
            {'params': deformation_params, 'lr': self.config.learning_rate * 0.1}
        ]

        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs
        )

    def _setup_criterion(self):
        """Setup loss functions"""
        self.criterion = TotalLoss({
            'reconstruction': {
                'weight': self.model_config.get('loss_weights', {}).get('reconstruction', 1.0),
                'l1_weight': 1.0,
                'ssim_weight': 0.8,
                'dssim_weight': 0.5,
                'perceptual_weight': 0.1
            },
            'temporal': {
                'weight': self.model_config.get('loss_weights', {}).get('temporal_consistency', 0.1),
                'flow_weight': 1.0,
                'motion_smoothness_weight': 0.1,
                'rigidity_constraint_weight': 0.05
            },
            'regularization': {
                'weight': self.model_config.get('loss_weights', {}).get('regularization', 0.01),
                'opacity_weight': 0.1,
                'sh_weight': 0.01,
                'covariance_weight': 0.01,
                'deformation_l2_weight': 0.01
            }
        })

    def _setup_dataloaders(self):
        """Setup data loaders"""
        # Get dataset configuration
        dataset_name = self.dataset_config.get('name', 'nvidia_dynamic')
        data_path = self.dataset_config.get('path', './data/nvidia_dynamic')

        # Create dataset config
        from data.datasets import get_dataset_config
        base_config = get_dataset_config(dataset_name, data_path, 'train')

        # Update with dataset config
        dataset_config_dict = dict(base_config._asdict())
        dataset_config_dict.update(self.dataset_config)

        # Create actual config object
        from data.datasets import DatasetConfig
        final_config = DatasetConfig(**dataset_config_dict)

        # Create dataloaders
        self.dataloaders = create_data_loaders(
            config=final_config,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            train={'load_depth': True, 'load_flow': True},
            val={'load_depth': True, 'load_flow': True},
            test={'load_depth': True, 'load_flow': True}
        )

        print(f"Train samples: {len(self.dataloaders['train'].dataset)}")
        print(f"Val samples: {len(self.dataloaders['val'].dataset)}")
        print(f"Test samples: {len(self.dataloaders['test'].dataset)}")

    def _setup_evaluator(self):
        """Setup evaluation metrics"""
        self.evaluator = ComprehensiveEvaluator()

    def _setup_loggers(self):
        """Setup logging systems"""
        self.tb_logger = TensorBoardLogger(os.path.join(self.exp_dir, "tensorboard"))
        self.wandb_logger = None

        # Initialize wandb if requested
        if self.config.log_interval > 0:  # Placeholder for wandb config check
            try:
                self.wandb_logger = WandBLogger(
                    project="dynamic_3dgs",
                    name=self.config.experiment_name,
                    config={
                        **self.config.__dict__,
                        **self.model_config,
                        **self.dataset_config
                    }
                )
            except ImportError:
                print("Weights & Biases not available, skipping wandb logging")

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.max_epochs} epochs...")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = self._validate_epoch()

            # Testing phase (every few epochs)
            if epoch % 5 == 0:
                test_metrics = self._test_epoch()
            else:
                test_metrics = {}

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics, test_metrics)

            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_metrics)

            # Early stopping (optional)
            if self._should_stop_training(val_metrics):
                print("Early stopping triggered")
                break

        print("Training completed!")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.train()
        epoch_losses = {}
        batch_count = 0

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            losses, outputs = self._training_step(batch)

            # Backward pass
            self._backward_step(losses['total_loss'])

            # Accumulate losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
                else:
                    epoch_losses[key] = value.item()

            batch_count += 1

            # Log progress
            if batch_idx % self.config.log_interval == 0:
                avg_losses = {k: v / batch_count for k, v in epoch_losses.items()}
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                      f"Total Loss = {avg_losses.get('total_loss', 0):.6f}")

        # Average losses over epoch
        return {k: v / len(self.dataloaders['train']) for k, v in epoch_losses.items()}

    def _training_step(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single training step"""
        self.optimizer.zero_grad(set_to_none=True)

        # Extract batch data
        images = batch['image']  # [B, C, H, W]
        cameras = batch['camera']
        time_steps = batch['time']  # [B]

        B, C, H, W = images.shape

        # Forward pass through model
        gaussian_outputs = self.model(time_steps)

        # Render images
        camera_obj = type('Camera', (), {
            'intrinsics': cameras[0].intrinsics,
            'extrinsics': cameras[0].extrinsics
        })()

        rendered_outputs = self.renderer.forward(
            gaussian_outputs, camera_obj, (H, W), time_steps[0]
        )

        # Prepare targets
        targets = {'rgb': images}

        # Prepare additional data for losses
        recon_data = {'mask': None}  # Could add valid mask here
        temporal_data = {
            'previous_rgb': None,  # Would come from previous frame
            'optical_flow': batch.get('flow'),
            'gaussian_deformations': gaussian_outputs.get('position_offsets'),
            'occlusion_mask': gaussian_outputs.get('occlusion_probs')
        }

        additional_data = {
            'recon_data': recon_data,
            'temporal_data': temporal_data
        }

        # Compute losses
        losses = self.criterion(
            predictions=rendered_outputs,
            targets=targets,
            gaussians=gaussian_outputs,
            deformation_field=self.model.deformation_field,
            time_steps=time_steps,
            additional_data=additional_data
        )

        return losses, rendered_outputs

    def _backward_step(self, loss: torch.Tensor):
        """Perform backward pass with mixed precision"""
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.gradient_accumulation_steps <= self.global_step % self.config.gradient_accumulation_steps:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Step optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1

        # Update learning rate
        if self.current_epoch < self.config.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.eval()
        val_metrics = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                batch = self._move_batch_to_device(batch)

                # Forward pass (simplified validation)
                losses, outputs = self._validation_step(batch)

                # Accumulate metrics
                for key, value in losses.items():
                    if key in val_metrics:
                        val_metrics[key] += value.item()
                    else:
                        val_metrics[key] = value.item()

        # Average metrics
        return {k: v / len(self.dataloaders['val']) for k, v in val_metrics.items()}

    def _validation_step(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single validation step"""
        # Similar to training step but without gradients
        images = batch['image']
        cameras = batch['camera']
        time_steps = batch['time']

        B, C, H, W = images.shape

        # Forward pass
        gaussian_outputs = self.model(time_steps)

        # Render
        camera_obj = type('Camera', (), {
            'intrinsics': cameras[0].intrinsics,
            'extrinsics': cameras[0].extrinsics
        })()

        rendered_outputs = self.renderer.forward(
            gaussian_outputs, camera_obj, (H, W), time_steps[0]
        )

        # Compute losses
        targets = {'rgb': images}
        recon_data = {'mask': None}
        temporal_data = {
            'previous_rgb': None,
            'optical_flow': batch.get('flow'),
            'gaussian_deformations': gaussian_outputs.get('position_offsets'),
            'occlusion_mask': gaussian_outputs.get('occlusion_probs')
        }

        additional_data = {
            'recon_data': recon_data,
            'temporal_data': temporal_data
        }

        losses = self.criterion(
            predictions=rendered_outputs,
            targets=targets,
            gaussians=gaussian_outputs,
            deformation_field=self.model.deformation_field,
            time_steps=time_steps,
            additional_data=additional_data
        )

        return losses, rendered_outputs

    def _test_epoch(self) -> Dict[str, float]:
        """Run comprehensive test evaluation"""
        self.eval()
        test_results = self.evaluator.evaluate(
            model=self.model,
            renderer=self.renderer,
            dataloader=self.dataloaders['test'],
            device=self.device
        )
        return test_results

    def _log_metrics(self,
                    epoch: int,
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float],
                    test_metrics: Dict[str, float]):
        """Log all metrics"""
        # TensorBoard logging
        self.tb_logger.log_scalars({
            'train/loss': train_metrics.get('total_loss', 0),
            'val/loss': val_metrics.get('total_loss', 0),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, step=epoch)

        # WandB logging
        if self.wandb_logger is not None:
            metrics = {
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                **{f'test/{k}': v for k, v in test_metrics.items()},
                'epoch': epoch
            }
            self.wandb_logger.log(metrics)

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics.get('total_loss', 0):.6f}")
        print(f"  Val Loss: {val_metrics.get('total_loss', 0):.6f}")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.exp_dir, 'latest.pth'))

        # Save best checkpoint if applicable
        current_metric = metrics.get(self.config.monitor_metric, -float('inf'))
        if ((self.config.monitor_mode == "max" and current_metric > self.best_metric) or
            (self.config.monitor_mode == "min" and current_metric < self.best_metric)):

            self.best_metric = current_metric
            torch.save(checkpoint, os.path.join(self.exp_dir, 'best.pth'))
            print(f"Saved best checkpoint at epoch {epoch} with {self.config.monitor_metric} = {current_metric:.4f}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def _should_stop_training(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        # Implement early stopping logic based on patience
        # This is a simplified version
        return False

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif hasattr(value, 'to'):
                # Handle custom objects with .to() method
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def save_model(self, path: str):
        """Save model weights only"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model weights only"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    # Example usage
    config = TrainerConfig(
        experiment_name="debug_experiment",
        max_epochs=10,
        batch_size=2,
        learning_rate=0.001
    )

    model_config = {
        'num_gaussians': 1000,
        'max_time_steps': 50,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'loss_weights': {
            'reconstruction': 1.0,
            'temporal_consistency': 0.1,
            'regularization': 0.01
        }
    }

    dataset_config = {
        'name': 'nvidia_dynamic',
        'path': './data/nvidia_dynamic'
    }

    trainer = Dynamic3DGSTrainer(
        config=config,
        model_config=model_config,
        dataset_config=dataset_config
    )

    # Start training
    trainer.train()