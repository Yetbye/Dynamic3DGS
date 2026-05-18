"""
Weights & Biases Logger for Dynamic3DGS

This module provides Weights & Biases logging functionality for
advanced experiment tracking and visualization.
"""

import os
from typing import Dict, Optional, Union


class WandBLogger:
    """
    Weights & Biases logger for Dynamic3DGS.

    Provides integration with Weights & Biases for experiment tracking,
    hyperparameter management, and result visualization.
    """

    def __init__(self,
                 project: str = "dynamic_3dgs",
                 name: str = None,
                 config: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name (defaults to timestamp if not provided)
            config: Configuration dictionary
            **kwargs: Additional W&B arguments
        """
        self.project = project
        self.name = name or f"run_{os.times().elapsed:.0f}"
        self.config = config or {}
        self.kwargs = kwargs

        # Initialize wandb
        try:
            import wandb
            self.wandb = wandb

            # Start run
            self.run = self.wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                **self.kwargs
            )

            print(f"W&B initialized: {self.run.id}")

        except ImportError:
            print("Warning: wandb not available. Install with 'pip install wandb'")
            self.wandb = None
            self.run = None

    def log(self,
            data: Dict[str, Union[float, int, str]],
            step: Optional[int] = None,
            commit: bool = True) -> None:
        """
        Log data to W&B.

        Args:
            data: Dictionary of data to log
            step: Global step
            commit: Whether to immediately send data
        """
        if self.wandb is None or self.run is None:
            return

        # Add step if provided
        if step is not None:
            data['_step'] = step

        # Log data
        self.run.log(data, step=step, commit=commit)

    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """Log metrics."""
        self.log({f"metrics/{k}": v for k, v in metrics.items()}, step=step)

    def log_scalars(self,
                   scalars: Dict[str, float],
                   step: Optional[int] = None) -> None:
        """Log scalar values."""
        self.log(scalars, step=step)

    def log_images(self,
                  images_dict: Dict[str, Union[str, list]],
                  step: Optional[int] = None) -> None:
        """Log images (supports file paths or wandb.Image objects)."""
        if self.wandb is None:
            return

        processed_images = {}
        for key, value in images_dict.items():
            if isinstance(value, str):
                # File path - wrap in wandb.Image
                processed_images[key] = self.wandb.Image(value)
            elif isinstance(value, list):
                # List of images - wrap in wandb.Image list
                processed_images[key] = [self.wandb.Image(img) if isinstance(img, str) else img
                                       for img in value]
            else:
                # Already a wandb object
                processed_images[key] = value

        self.log(processed_images, step=step)

    def log_histogram(self,
                     name: str,
                     values: list,
                     step: Optional[int] = None) -> None:
        """Log histogram data."""
        if self.wandb is None:
            return

        hist_data = self.wandb.Histogram(values)
        self.log({f"histograms/{name}": hist_data}, step=step)

    def log_table(self,
                 columns: list,
                 data: list,
                 title: str = None,
                 step: Optional[int] = None) -> None:
        """Log table data."""
        if self.wandb is None:
            return

        table = self.wandb.Table(columns=columns, data=data)
        if title:
            self.log({f"tables/{title}": table}, step=step)
        else:
            self.log({"table": table}, step=step)

    def log_model(self,
                 model_path: str,
                 aliases: list = None,
                 step: Optional[int] = None) -> None:
        """Log model artifact."""
        if self.wandb is None:
            return

        aliases = aliases or ["latest"]
        artifact = self.wandb.Artifact(f"model_{self.run.id}", type="model")
        artifact.add_file(model_path)

        self.run.log_artifact(artifact, aliases=aliases)
        if step is not None:
            self.log({"model_checkpoint": step}, step=step)

    def log_config(self,
                  config: Dict,
                  prefix: str = "") -> None:
        """Log configuration parameters."""
        if self.wandb is None:
            return

        # Flatten nested config
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_config[f"{prefix}{key}_{subkey}" if prefix else f"{key}_{subkey}"] = subvalue
            else:
                flat_config[f"{prefix}{key}" if prefix else key] = value

        self.run.config.update(flat_config)

    def watch(self,
             model: 'torch.nn.Module',
             criterion=None,
             log='gradients',
             log_freq=100) -> None:
        """
        Watch model for parameter and gradient statistics.

        Args:
            model: PyTorch model
            criterion: Loss function
            log: What to log ('all', 'parameters', 'gradients')
            log_freq: Frequency of logging
        """
        if self.wandb is None:
            return

        try:
            import torch
            self.wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"Failed to watch model: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.wandb is None or self.run is None:
            return

        self.run.finish()

    def join(self) -> None:
        """Wait for W&B process to finish."""
        if self.wandb is None or self.run is None:
            return

        self.wandb.join()


# Convenience functions for common W&B patterns
def setup_wandb_defaults() -> dict:
    """Return default W&B configuration."""
    return {
        'entity': None,  # Set to your W&B username/organization
        'mode': 'online',  # 'online', 'offline', or 'disabled'
        'tags': ['dynamic_3dgs'],
        'notes': 'Dynamic3DGS experiment'
    }


def create_experiment_name(base_name: str,
                          epoch: int = None,
                          metric: float = None,
                          mode: str = 'auto') -> str:
    """
    Create experiment name based on current state.

    Args:
        base_name: Base name for experiment
        epoch: Current epoch
        metric: Current metric value
        mode: Naming mode ('epoch', 'metric', 'auto')

    Returns:
        formatted_name: Formatted experiment name
    """
    if mode == 'epoch' and epoch is not None:
        return f"{base_name}_ep{epoch:04d}"
    elif mode == 'metric' and metric is not None:
        return f"{base_name}_m{metric:.4f}"
    elif mode == 'auto' and epoch is not None:
        return f"{base_name}_ep{epoch:04d}"
    else:
        return base_name


def log_training_progress(logger: WandBLogger,
                         epoch: int,
                         train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float],
                         lr: float,
                         **additional_data) -> None:
    """Log comprehensive training progress."""
    if logger.wandb is None:
        return

    # Combine all metrics
    all_metrics = {
        **{f"train/{k}": v for k, v in train_metrics.items()},
        **{f"val/{k}": v for k, v in val_metrics.items()},
        "learning_rate": lr
    }

    # Log everything
    logger.log(all_metrics, step=epoch)

    # Log additional structured data
    if 'images' in additional_data:
        logger.log_images(additional_data['images'], step=epoch)
    if 'histograms' in additional_data:
        for name, values in additional_data['histograms'].items():
            logger.log_histogram(name, values, step=epoch)


def log_reconstruction_comparison(logger: WandBLogger,
                                 epoch: int,
                                 predicted: 'torch.Tensor',
                                 target: 'torch.Tensor',
                                 sample_idx: int = 0) -> None:
    """Log reconstruction comparison using W&B Image."""
    if logger.wandb is None:
        return

    try:
        import torch
        import numpy as np
        from PIL import Image

        # Convert tensors to PIL images
        pred_np = (predicted[sample_idx].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        target_np = (target[sample_idx].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)

        pred_image = Image.fromarray(pred_np)
        target_image = Image.fromarray(target_np)

        # Create side-by-side comparison
        width, height = pred_image.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(target_image, (0, 0))
        comparison.paste(pred_image, (width, 0))

        # Log to W&B
        logger.log_images({
            f"reconstruction/comparison_ep{epoch:04d}": logger.wandb.Image(comparison),
            f"reconstruction/target_ep{epoch:04d}": logger.wandb.Image(target_image),
            f"reconstruction/predicted_ep{epoch:04d}": logger.wandb.Image(pred_image)
        }, step=epoch)

    except Exception as e:
        print(f"Failed to log reconstruction comparison: {e}")


if __name__ == "__main__":
    # Example usage
    import tempfile
    import torch

    # Setup example
    config = {
        'num_gaussians': 1000,
        'learning_rate': 0.001,
        'batch_size': 4
    }

    # Initialize logger
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = WandBLogger(
            project="dynamic_3dgs_example",
            name="test_run",
            config=config,
            dir=temp_dir
        )

        # Log some example data
        logger.log_scalars({
            'loss': 0.5,
            'accuracy': 0.85,
            'learning_rate': 0.001
        }, step=0)

        # Log model (placeholder - would be actual model path)
        # logger.log_model("/path/to/model.pth")

        # Use convenience functions
        log_training_progress(
            logger,
            epoch=0,
            train_metrics={'loss': 0.5, 'psnr': 25.0},
            val_metrics={'loss': 0.6, 'psnr': 24.0},
            lr=0.001
        )

        # Clean up
        logger.finish()

    print("W&B logging example completed.")