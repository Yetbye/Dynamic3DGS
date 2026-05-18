"""
Camera Utilities for Dynamic3DGS

This module provides camera-related utilities including pose handling,
projection matrices, and coordinate transformations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class CameraPose:
    """Camera pose representation (intrinsics + extrinsics)"""
    intrinsics: torch.Tensor  # [3, 3]
    extrinsics: torch.Tensor  # [4, 4] or [3, 4]

    @property
    def width(self) -> int:
        return int(self.intrinsics[0, 2] * 2)

    @property
    def height(self) -> int:
        return int(self.intrinsics[1, 2] * 2)

    @classmethod
    def create_default(cls, width: int, height: int,
                      focal_length: float = None) -> 'CameraPose':
        """Create a default camera with given image size"""
        if focal_length is None:
            focal_length = min(width, height) * 0.75

        # Intrinsic matrix
        intrinsics = torch.tensor([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Extrinsic matrix (identity - camera at origin)
        extrinsics = torch.eye(4, dtype=torch.float32)
        extrinsics[:3, :3] = torch.eye(3, dtype=torch.float32)
        extrinsics[:3, 3] = torch.zeros(3, dtype=torch.float32)

        return cls(intrinsics=intrinsics, extrinsics=extrinsics)

    def to_homogeneous(self, points_2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D pixel coordinates to homogeneous coordinates"""
        if points_2d.dim() == 1:
            points_2d = points_2d.unsqueeze(0)

        # Add homogeneous coordinate (1)
        ones = torch.ones(points_2d.shape[0], 1, device=points_2d.device)
        points_homo = torch.cat([points_2d, ones], dim=1)

        return points_homo

    def unproject_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Unproject depth map to 3D points in camera space"""
        H, W = depth_map.shape[-2:]

        # Create pixel grid
        i, j = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=depth_map.device),
            torch.arange(W, dtype=torch.float32, device=depth_map.device),
            indexing='ij'
        )

        # Convert to normalized device coordinates
        x_cam = (j - self.intrinsics[0, 2]) * depth_map / self.intrinsics[0, 0]
        y_cam = (i - self.intrinsics[1, 2]) * depth_map / self.intrinsics[1, 1]
        z_cam = depth_map

        # Stack into homogeneous coordinates
        points_camera = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)

        # Transform to world coordinates
        points_world = torch.matmul(self.extrinsics[:3], points_camera.transpose(-1, -2)).transpose(-1, -2)

        return points_world[..., :3]  # Remove homogeneous coordinate


def compute_relative_pose(from_pose: CameraPose, to_pose: CameraPose) -> torch.Tensor:
    """Compute relative transformation from one pose to another"""
    # Convert extrinsics to transformation matrices
    T_from = torch.eye(4, dtype=torch.float32)
    T_from[:3] = from_pose.extrinsics

    T_to = torch.eye(4, dtype=torch.float32)
    T_to[:3] = to_pose.extrinsics

    # Compute relative transform: T_to^-1 * T_from
    T_relative = torch.matmul(torch.inverse(T_to), T_from)

    return T_relative


def normalize_camera_coordinates(x: torch.Tensor, y: torch.Tensor,
                               intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize pixel coordinates to camera space"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    return x_norm, y_norm


def project_points_to_image(points_3d: torch.Tensor,
                           camera: CameraPose) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to image plane.

    Args:
        points_3d: [N, 3] 3D points in world coordinates
        camera: CameraPose object

    Returns:
        points_2d: [N, 2] projected pixel coordinates
        depths: [N] corresponding depths
    """
    # Homogeneous coordinates
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)

    # Transform to camera space
    points_camera = torch.matmul(camera.extrinsics[:3], points_homo.transpose(-1, -2)).transpose(-1, -2)

    # Project to image plane
    x_proj = points_camera[..., 0] / points_camera[..., 2]
    y_proj = points_camera[..., 1] / points_camera[..., 2]

    # Apply intrinsic transformation
    u = camera.intrinsics[0, 0] * x_proj + camera.intrinsics[0, 2]
    v = camera.intrinsics[1, 1] * y_proj + camera.intrinsics[1, 2]

    points_2d = torch.stack([u, v], dim=-1)
    depths = points_camera[..., 2]

    return points_2d, depths


def get_view_frustum(camera: CameraPose, near: float = 0.1, far: float = 100.0) -> torch.Tensor:
    """Get view frustum corners in camera space"""
    # Define frustum corners at near and far planes
    H, W = camera.height, camera.width

    # Near plane corners
    near_corners = torch.tensor([
        [-W/2, -H/2, near],
        [ W/2, -H/2, near],
        [ W/2,  H/2, near],
        [-W/2,  H/2, near]
    ], dtype=torch.float32)

    # Far plane corners
    far_corners = torch.tensor([
        [-W/2, -H/2, far],
        [ W/2, -H/2, far],
        [ W/2,  H/2, far],
        [-W/2,  H/2, far]
    ], dtype=torch.float32)

    # Transform to world coordinates
    near_corners_world = torch.matmul(camera.extrinsics[:3], near_corners.T).T
    far_corners_world = torch.matmul(camera.extrinsics[:3], far_corners.T).T

    # Return frustum corners
    return torch.cat([near_corners_world, far_corners_world], dim=0)


def estimate_camera_intrinsics(image_size: Tuple[int, int],
                             fov_x: float = None,
                             fov_y: float = None) -> torch.Tensor:
    """
    Estimate camera intrinsics from image size and field of view.

    Args:
        image_size: (width, height)
        fov_x: Horizontal field of view in degrees
        fov_y: Vertical field of view in degrees

    Returns:
        intrinsics: [3, 3] intrinsic matrix
    """
    W, H = image_size

    if fov_x is not None:
        focal_x = W / (2 * np.tan(np.radians(fov_x) / 2))
    else:
        focal_x = W * 0.75

    if fov_y is not None:
        focal_y = H / (2 * np.tan(np.radians(fov_y) / 2))
    else:
        focal_y = H * 0.75

    intrinsics = torch.tensor([
        [focal_x, 0, W / 2],
        [0, focal_y, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32)

    return intrinsics


def compose_extrinsics(R1: torch.Tensor, t1: torch.Tensor,
                       R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Compose two extrinsics (R1, t1) and (R2, t2).

    The result transforms from frame 2 to frame 1.
    """
    R_composed = torch.matmul(R1, R2)
    t_composed = torch.matmul(R1, t2) + t1

    extrinsics = torch.eye(4, dtype=torch.float32)
    extrinsics[:3, :3] = R_composed
    extrinsics[:3, 3] = t_composed

    return extrinsics


if __name__ == "__main__":
    # Example usage
    camera = CameraPose.create_default(width=640, height=480)

    print(f"Camera intrinsics:\n{camera.intrinsics}")
    print(f"Camera extrinsics:\n{camera.extrinsics}")

    # Test projection
    points_3d = torch.tensor([[1, 2, 5], [3, 4, 10]], dtype=torch.float32)
    points_2d, depths = project_points_to_image(points_3d, camera)
    print(f"Projected 2D points:\n{points_2d}")
    print(f"Corresponding depths:\n{depths}")