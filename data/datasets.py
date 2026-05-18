"""
Dynamic3DGS Dataset Loaders

This module provides dataset classes for loading dynamic scene data
including monocular videos and multiview sequences.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.camera_utils import CameraPose
from utils.depth_utils import depth_to_normal
from utils.flow_utils import compute_optical_flow


@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    name: str
    path: str
    type: str  # "monocular_video" or "multiview_dynamic"
    image_size: Tuple[int, int]
    num_frames: int
    frame_interval: int
    normalize_rgb: bool = True
    rgb_range: Tuple[float, float] = (-1.0, 1.0)


class DynamicSceneDataset(Dataset):
    """
    Base dataset class for dynamic scenes.

    Handles loading of:
    - RGB images
    - Camera poses (intrinsics + extrinsics)
    - Time stamps
    - Optional depth maps and optical flows
    """

    def __init__(self,
                 config: DatasetConfig,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 load_depth: bool = False,
                 load_flow: bool = False):
        """
        Initialize the dynamic scene dataset.

        Args:
            config: Dataset configuration
            split: "train", "val", or "test"
            transform: Image transformations
            load_depth: Whether to load depth maps
            load_flow: Whether to load optical flow
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.load_depth = load_depth
        self.load_flow = load_flow

        # Data storage
        self.images: List[torch.Tensor] = []
        self.cameras: List[CameraPose] = []
        self.time_stamps: List[float] = []
        self.depth_maps: List[Optional[torch.Tensor]] = []
        self.flow_fields: List[Optional[torch.Tensor]] = []

        self._load_data()

    def _load_data(self):
        """Load dataset from disk"""
        dataset_path = Path(self.config.path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Load metadata
        metadata_file = dataset_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load images and camera parameters
        self._load_images_and_cameras(dataset_path)

        # Load additional data if requested
        if self.load_depth:
            self._load_depth_maps(dataset_path)
        if self.load_flow:
            self._load_flow_fields(dataset_path)

        # Apply frame selection based on split
        self._select_frames()

    def _load_images_and_cameras(self, dataset_path: Path):
        """Load images and camera parameters"""
        # This is a placeholder implementation
        # In practice, you'd implement specific loading logic
        # based on your dataset format (Nvidia Dynamic, HyperNeRF, etc.)

        print(f"Loading {self.split} data from {dataset_path}...")

        # Example structure:
        # dataset_path/
        #   images/frame_0000.png, frame_0001.png, ...
        #   cameras.npz (intrinsics, extrinsics)
        #   times.txt (time stamps)

        image_dir = dataset_path / "images"
        if image_dir.exists():
            image_files = sorted(list(image_dir.glob("*.png")) +
                              list(image_dir.glob("*.jpg")))
            print(f"Found {len(image_files)} images")

            for img_file in image_files[:self.config.num_frames]:
                try:
                    # Load image
                    from PIL import Image
                    img = Image.open(img_file).convert('RGB')
                    img_tensor = transforms.ToTensor()(img)

                    if self.config.normalize_rgb:
                        # Normalize to [-1, 1] range
                        img_tensor = (img_tensor * 2) - 1

                    self.images.append(img_tensor)

                    # Create camera pose (placeholder - should load real data)
                    camera = CameraPose.create_default(
                        width=img_tensor.shape[2],
                        height=img_tensor.shape[1]
                    )
                    self.cameras.append(camera)

                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue

        # Load time stamps
        times_file = dataset_path / "times.txt"
        if times_file.exists():
            with open(times_file, 'r') as f:
                self.time_stamps = [float(line.strip()) for line in f.readlines()]
        else:
            # Default: uniform time steps
            self.time_stamps = list(range(len(self.images)))

    def _load_depth_maps(self, dataset_path: Path):
        """Load depth maps if available"""
        depth_dir = dataset_path / "depth"
        if depth_dir.exists():
            depth_files = sorted(list(depth_dir.glob("*.npy")) +
                              list(depth_dir.glob("*.npz")))
            for depth_file in depth_files:
                try:
                    depth = np.load(depth_file)
                    self.depth_maps.append(torch.from_numpy(depth).float())
                except Exception as e:
                    print(f"Error loading depth {depth_file}: {e}")
                    self.depth_maps.append(None)
        else:
            # Generate dummy depth maps
            for i in range(len(self.images)):
                H, W = self.images[i].shape[1:]
                depth = torch.rand(H, W) * 5.0 + 1.0  # Random depth [1, 6]
                self.depth_maps.append(depth)

    def _load_flow_fields(self, dataset_path: Path):
        """Load optical flow fields if available"""
        flow_dir = dataset_path / "flow"
        if flow_dir.exists():
            flow_files = sorted(list(flow_dir.glob("*.npy")))
            for flow_file in flow_files:
                try:
                    flow = np.load(flow_file)
                    self.flow_fields.append(torch.from_numpy(flow).float())
                except Exception as e:
                    print(f"Error loading flow {flow_file}: {e}")
                    self.flow_fields.append(None)
        else:
            # Compute optical flow between consecutive frames
            for i in range(len(self.images) - 1):
                try:
                    flow = compute_optical_flow(
                        self.images[i], self.images[i + 1]
                    )
                    self.flow_fields.append(flow)
                except Exception as e:
                    print(f"Error computing flow: {e}")
                    self.flow_fields.append(None)

    def _select_frames(self):
        """Select frames based on split and configuration"""
        total_frames = len(self.images)

        if self.split == "train":
            start_idx = 0
            end_idx = min(total_frames, self.config.num_frames)
        elif self.split == "val":
            start_idx = self.config.train_frames
            end_idx = start_idx + self.config.val_frames
        else:  # test
            start_idx = self.config.train_frames + self.config.val_frames
            end_idx = start_idx + self.config.test_frames

        # Apply frame interval
        selected_indices = list(range(start_idx, min(end_idx, total_frames),
                                     self.config.frame_interval))

        # Filter data to selected indices
        self.images = [self.images[i] for i in selected_indices]
        self.cameras = [self.cameras[i] for i in selected_indices]
        self.time_stamps = [self.time_stamps[i] for i in selected_indices]
        self.depth_maps = [self.depth_maps[i] for i in selected_indices]
        self.flow_fields = [self.flow_fields[i] for i in selected_indices]

        print(f"{self.split} set: {len(self.images)} frames")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index"""
        sample = {
            'image': self.images[idx],
            'camera': self.cameras[idx],
            'time': torch.tensor([self.time_stamps[idx]], dtype=torch.float32),
        }

        if self.load_depth and idx < len(self.depth_maps):
            sample['depth'] = self.depth_maps[idx]

        if self.load_flow and idx < len(self.flow_fields):
            sample['flow'] = self.flow_fields[idx]

        # Apply augmentations if training
        if self.split == "train" and self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class NvidiaDynamicDataset(DynamicSceneDataset):
    """Specific loader for Nvidia Dynamic Dataset"""

    def _load_data(self):
        """Load Nvidia Dynamic dataset format"""
        # Nvidia Dynamic has specific file structure
        # Override base implementation for this specific case
        super()._load_data()

        # Add Nvidia-specific processing here
        print("Loaded Nvidia Dynamic dataset")


class HyperNeRFDataset(DynamicSceneDataset):
    """Specific loader for HyperNeRF Dataset"""

    def _load_data(self):
        """Load HyperNeRF dataset format"""
        # HyperNeRF has multiview data
        # Override base implementation
        super()._load_data()

        # Add HyperNeRF-specific processing here
        print("Loaded HyperNeRF dataset")


def create_data_loaders(config: DatasetConfig,
                       batch_size: int = 4,
                       num_workers: int = 8,
                       **kwargs) -> Dict[str, DataLoader]:
    """
    Create data loaders for train/val/test splits.

    Args:
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Dictionary with train/val/test dataloaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                             saturation=0.1, hue=0.1),
        transforms.RandomResizedCrop(config.image_size,
                                  scale=(0.8, 1.2)),
    ])

    val_test_transform = transforms.Compose([])  # No augmentation

    # Create datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        dataset_kwargs = kwargs.get(split, {})

        dataset = DynamicSceneDataset(
            config=config,
            split=split,
            transform=train_transform if split == "train" else val_test_transform,
            load_depth=dataset_kwargs.get("load_depth", False),
            load_flow=dataset_kwargs.get("load_flow", False)
        )

        datasets[split] = dataset

    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if split == "train" else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    return dataloaders


# Utility functions for dataset creation
def get_dataset_config(dataset_name: str, data_path: str,
                      split: str, **overrides) -> DatasetConfig:
    """Get dataset configuration based on name"""
    default_configs = {
        "nvidia_dynamic": DatasetConfig(
            name="nvidia_dynamic",
            path=data_path,
            type="monocular_video",
            image_size=(512, 512),
            num_frames=100,
            frame_interval=1
        ),
        "hypernerf": DatasetConfig(
            name="hypernerf",
            path=data_path,
            type="multiview_dynamic",
            image_size=(800, 800),
            num_frames=120,
            frame_interval=1
        )
    }

    config = default_configs.get(dataset_name, default_configs["nvidia_dynamic"])
    config = config._replace(**overrides)  # Apply overrides

    return config


if __name__ == "__main__":
    # Example usage
    config = get_dataset_config("nvidia_dynamic", "./data/nvidia_dynamic",
                               "train", num_frames=50)

    dataloaders = create_data_loaders(config, batch_size=2)

    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")

    # Show sample batch
    for batch in dataloaders['train']:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch time shape: {batch['time'].shape}")
        break