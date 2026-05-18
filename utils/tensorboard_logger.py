"""
TensorBoard Logger for Dynamic3DGS

This module provides TensorBoard logging functionality for tracking
training progress, metrics, and model visualizations.
"""

import os
import torch
from typing import Dict, List, Optional, Union
from datetime import datetime


class TensorBoardLogger:
    """
    TensorBoard logger for Dynamic3DGS training.

    Handles logging of scalars, images, histograms, and other data
    to TensorBoard for visualization.
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        self.step = 0

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Try to import tensorboardX or torch.utils.tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("Warning: tensorboardX not available. Install with 'pip install tensorboard'")

    def log_scalars(self,
                   scalars: Dict[str, Union[float, torch.Tensor]],
                   step: Optional[int] = None,
                   prefix: str = "") -> None:
        """
        Log scalar values.

        Args:
            scalars: Dictionary of scalar names and values
            step: Global step (defaults to current step)
            prefix: Prefix to add to all logged keys
        """
        if self.writer is None:
            return

        current_step = step if step is not None else self.step

        for key, value in scalars.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            log_key = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(log_key, value, current_step)

        self.step = current_step + 1

    def log_images(self,
                  images_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                  step: Optional[int] = None,
                  prefix: str = "",
                  **kwargs) -> None:
        """
        Log image data.

        Args:
            images_dict: Dictionary of image names and tensors
            step: Global step (defaults to current step)
            prefix: Prefix to add to all logged keys
        """
        if self.writer is None:
            return

        current_step = step if step is not None else self.step

        for key, images in images_dict.items():
            if isinstance(images, list):
                # Multiple images - create grid
                self.writer.add_image(
                    f"{prefix}/{key}",
                    self._make_image_grid(images),
                    current_step,
                    **kwargs
                )
            elif isinstance(images, torch.Tensor):
                # Single image
                self.writer.add_image(
                    f"{prefix}/{key}",
                    self._normalize_image(images),
                    current_step,
                    **kwargs
                )

        self.step = current_step + 1

    def log_histograms(self,
                      tensors: Dict[str, torch.Tensor],
                      step: Optional[int] = None,
                      prefix: str = "") -> None:
        """
        Log histogram data.

        Args:
            tensors: Dictionary of tensor names and values
            step: Global step (defaults to current step)
            prefix: Prefix to add to all logged keys
        """
        if self.writer is None:
            return

        current_step = step if step is not None else self.step

        for key, tensor in tensors.items():
            log_key = f"{prefix}/{key}"
            self.writer.add_histogram(log_key, tensor, current_step)

        self.step = current_step + 1

    def log_model_graph(self,
                       model: torch.nn.Module,
                       dummy_input: torch.Tensor,
                       step: Optional[int] = None) -> None:
        """
        Log model computational graph.

        Args:
            model: PyTorch model
            dummy_input: Dummy input tensor
            step: Global step
        """
        if self.writer is None:
            return

        current_step = step if step is not None else self.step

        self.writer.add_graph(model, dummy_input)
        self.step = current_step + 1

    def log_text(self,
                text_dict: Dict[str, str],
                step: Optional[int] = None,
                prefix: str = "") -> None:
        """
        Log text data.

        Args:
            text_dict: Dictionary of text names and strings
            step: Global step
            prefix: Prefix to add to all logged keys
        """
        if self.writer is None:
            return

        current_step = step if step is not None else self.step

        for key, text in text_dict.items():
            log_key = f"{prefix}/{key}"
            self.writer.add_text(log_key, text, current_step)

        self.step = current_step + 1

    def log_hparams(self,
                   hparams: Dict[str, Union[str, float, int, bool]],
                   metrics: Dict[str, float]) -> None:
        """
        Log hyperparameters and corresponding metrics.

        Args:
            hparams: Hyperparameter dictionary
            metrics: Metrics dictionary
        """
        if self.writer is None:
            return

        # Convert all hparams to strings for compatibility
        hparam_dict = {k: str(v) for k, v in hparams.items()}

        # Log hparams and metrics together
        self.writer.add_hparams(hparam_dict, metrics)

    def flush(self) -> None:
        """Flush the writer buffer."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        if self.writer is not None:
            self.writer.close()

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image tensor for TensorBoard.

        Args:
            image: Image tensor [C, H, W] or [H, W, C]

        Returns:
            normalized_image: Normalized image tensor
        """
        # Handle different formats
        if image.dim() == 4:
            # Batch of images - take first
            image = image[0]
        if image.dim() == 3:
            # [C, H, W] format
            if image.shape[0] == 1:
                # Grayscale
                image = image.squeeze(0)
            elif image.shape[0] == 3:
                # RGB
                pass
            else:
                raise ValueError(f"Unsupported channel dimension: {image.shape[0]}")
        elif image.dim() == 2:
            # [H, W] format - add channel dimension
            image = image.unsqueeze(0)

        # Ensure correct range [0, 1]
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Ensure correct shape [C, H, W]
        if image.shape[0] != 1 and image.shape[0] != 3:
            image = image.unsqueeze(0)

        return image.float()

    def _make_image_grid(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Create image grid from list of images.

        Args:
            images: List of image tensors

        Returns:
            grid: Concatenated image grid
        """
        if len(images) == 0:
            return torch.zeros(3, 64, 64)

        # Normalize all images
        normalized_images = []
        for img in images:
            normalized_img = self._normalize_image(img)
            normalized_images.append(normalized_img)

        # Stack into batch
        batch = torch.stack(normalized_images)

        # Create grid using torchvision's make_grid if available
        try:
            from torchvision.utils import make_grid
            return make_grid(batch, nrow=int(len(images)**0.5 + 0.5))
        except ImportError:
            # Fallback simple concatenation
            return torch.cat(normalized_images[:4], dim=-1).unsqueeze(0)


# Convenience functions for common logging patterns
def log_training_metrics(logger: TensorBoardLogger,
                        epoch: int,
                        train_loss: float,
                        val_loss: float,
                        lr: float,
                        **additional_metrics) -> None:
    """Log standard training metrics."""
    metrics = {
        'train/loss': train_loss,
        'val/loss': val_loss,
        'learning_rate': lr
    }

    # Add any additional metrics
    metrics.update({f"train/{k}": v for k, v in additional_metrics.items()})

    logger.log_scalars(metrics, step=epoch, prefix="metrics")


def log_reconstruction_results(logger: TensorBoardLogger,
                              epoch: int,
                              predicted: torch.Tensor,
                              target: torch.Tensor,
                              prefix: str = "reconstruction") -> None:
    """Log reconstruction comparison images."""
    # Create side-by-side comparison
    comparison = torch.cat([target, predicted], dim=-1)  # [B, C, H, 2*W]

    logger.log_images({
        f"{prefix}/comparison": comparison,
        f"{prefix}/target": target,
        f"{prefix}/predicted": predicted
    }, step=epoch)


def log_gaussian_statistics(logger: TensorBoardLogger,
                           gaussians: dict,
                           step: int,
                           prefix: str = "gaussians") -> None:
    """Log Gaussian parameter statistics."""
    stats = {}

    for key, value in gaussians.items():
        if isinstance(value, torch.Tensor):
            stats[f"{prefix}/{key}_mean"] = value.mean().item()
            stats[f"{prefix}/{key}_std"] = value.std().item()
            stats[f"{prefix}/{key}_min"] = value.min().item()
            stats[f"{prefix}/{key}_max"] = value.max().item()

    logger.log_scalars(stats, step=step)


if __name__ == "__main__":
    # Example usage
    import tempfile

    # Create temporary log directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize logger
        logger = TensorBoardLogger(os.path.join(temp_dir, "logs"))

        # Log some example data
        logger.log_scalars({
            'loss': 0.5,
            'accuracy': 0.85,
            'learning_rate': 0.001
        }, step=0)

        # Log images
        dummy_image = torch.randn(1, 3, 64, 64)
        logger.log_images({
            'sample_image': dummy_image,
            'grayscale': torch.randn(1, 64, 64)
        }, step=0)

        # Log histograms
        logger.log_histograms({
            'weights': torch.randn(100),
            'biases': torch.randn(50)
        }, step=0)

        # Use convenience functions
        log_training_metrics(logger, epoch=0, train_loss=0.5, val_loss=0.6, lr=0.001)
        log_reconstruction_results(logger, epoch=0, predicted=dummy_image, target=dummy_image)

        # Clean up
        logger.close()

    print("TensorBoard logging example completed.")