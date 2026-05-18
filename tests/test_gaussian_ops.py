"""
Unit tests for Gaussian operations in Dynamic3DGS.

This module contains comprehensive tests for the core Gaussian
operations including projection, sorting, and rendering.
"""

import torch
import pytest
import numpy as np
from utils.camera_utils import CameraPose


def test_camera_pose_creation():
    """Test camera pose creation and basic operations."""
    # Create default camera
    camera = CameraPose.create_default(width=640, height=480)

    assert camera.intrinsics.shape == (3, 3)
    assert camera.extrinsics.shape == (4, 4)
    assert camera.width == 640
    assert camera.height == 480

    # Test intrinsic matrix structure
    assert camera.intrinsics[0, 0] > 0  # fx > 0
    assert camera.intrinsics[1, 1] > 0  # fy > 0
    assert camera.intrinsics[0, 2] == 320  # cx = width/2
    assert camera.intrinsics[1, 2] == 240  # cy = height/2


def test_camera_projection():
    """Test 3D point projection to image plane."""
    camera = CameraPose.create_default(width=640, height=480)

    # Test points at different depths
    points_3d = torch.tensor([
        [0.0, 0.0, 5.0],   # Center, depth 5
        [1.0, 1.0, 5.0],   # Offset, same depth
        [-1.0, -1.0, 5.0], # Negative offset, same depth
        [0.0, 0.0, 10.0],  # Same position, deeper
    ]).unsqueeze(0)  # Add batch dimension

    points_2d, depths = camera.project_points_to_image(points_3d)

    assert points_2d.shape == (1, 4, 2)
    assert depths.shape == (1, 4)

    # Check that center point projects to image center
    center_idx = 0
    expected_u = 640 / 2 + (0 - 320) * (1000 / 5)  # fx * x/z + cx
    expected_v = 480 / 2 + (0 - 240) * (1000 / 5)  # fy * y/z + cy

    assert abs(points_2d[0, center_idx, 0].item() - expected_u) < 1e-5
    assert abs(points_2d[0, center_idx, 1].item() - expected_v) < 1e-5
    assert abs(depths[0, center_idx].item() - 5.0) < 1e-5


def test_camera_unprojection():
    """Test depth map unprojection to 3D points."""
    camera = CameraPose.create_default(width=640, height=480)

    # Create synthetic depth map
    H, W = 240, 320
    depth_map = torch.ones(H, W) * 5.0  # Constant depth
    depth_map[120, 160] = 10.0  # Point at center but farther

    points_3d = camera.unproject_depth(depth_map)

    assert points_3d.shape == (H, W, 3)

    # Check center point
    center_point = points_3d[120, 160]
    expected_z = 10.0
    assert abs(center_point[2].item() - expected_z) < 1e-5

    # Check that x,y coordinates are correct for center pixel
    center_x = (160 - 320) * (1000 / 10.0)  # (u - cx) * z / fx
    center_y = (120 - 240) * (1000 / 10.0)  # (v - cy) * z / fy
    assert abs(center_point[0].item() - center_x) < 1e-5
    assert abs(center_point[1].item() - center_y) < 1e-5


def test_relative_pose_computation():
    """Test relative pose computation between two cameras."""
    # Create two cameras with known relative pose
    T_world_cam1 = torch.eye(4)
    T_world_cam1[:3, 3] = torch.tensor([1.0, 0.0, 0.0])  # 1m translation along X

    T_world_cam2 = torch.eye(4)
    T_world_cam2[:3, 3] = torch.tensor([0.0, 1.0, 0.0])  # 1m translation along Y

    cam1_pose = CameraPose(
        intrinsics=torch.eye(3),
        extrinsics=T_world_cam1
    )
    cam2_pose = CameraPose(
        intrinsics=torch.eye(3),
        extrinsics=T_world_cam2
    )

    # Compute relative transformation from cam1 to cam2
    T_cam1_cam2 = compute_relative_pose(cam1_pose, cam2_pose)

    # Expected: rotation by 90 degrees around Z, translation by (-1, 1, 0)
    expected_R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    expected_t = torch.tensor([-1.0, 1.0, 0.0])

    # Check rotation
    assert torch.allclose(T_cam1_cam2[:3, :3], expected_R, atol=1e-6)

    # Check translation
    assert torch.allclose(T_cam1_cam2[:3, 3], expected_t, atol=1e-6)


def test_normalize_camera_coordinates():
    """Test camera coordinate normalization."""
    intrinsics = torch.tensor([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

    u, v = torch.tensor([320.0, 400.0]), torch.tensor([240.0, 300.0])
    x_norm, y_norm = normalize_camera_coordinates(u, v, intrinsics)

    # For center pixel (320, 240), normalized coords should be (0, 0)
    assert abs(x_norm[0].item()) < 1e-10
    assert abs(y_norm[0].item()) < 1e-10

    # For offset pixel (400, 300):
    # x_norm = (400 - 320) / 1000 = 0.08
    # y_norm = (300 - 240) / 1000 = 0.06
    assert abs(x_norm[1].item() - 0.08) < 1e-6
    assert abs(y_norm[1].item() - 0.06) < 1e-6


def test_view_frustum():
    """Test view frustum generation."""
    camera = CameraPose.create_default(width=640, height=480)

    near, far = 0.1, 100.0
    frustum = get_view_frustum(camera, near=near, far=far)

    # Frustum should have 8 corners (4 near + 4 far)
    assert frustum.shape == (8, 3)

    # Check that near and far planes have correct depths
    depths = frustum[:, 2]
    unique_depths = torch.unique(torch.round(depths * 1000) / 1000)  # Round to avoid floating point errors

    assert len(unique_depths) == 2
    assert torch.any(torch.abs(unique_depths - near) < 1e-5)
    assert torch.any(torch.abs(unique_depths - far) < 1e-5)


def test_intrinsics_estimation():
    """Test camera intrinsics estimation from FOV."""
    # Test with known FOV
    image_size = (640, 480)
    fov_x = 60.0  # 60 degree horizontal FOV

    intrinsics = estimate_camera_intrinsics(image_size, fov_x=fov_x)

    # Compute expected focal length
    expected_fx = image_size[0] / (2 * np.tan(np.radians(fov_x) / 2))

    assert torch.allclose(intrinsics[0, 0], torch.tensor(expected_fx))
    assert intrinsics[0, 2] == image_size[0] / 2  # cx = width/2
    assert intrinsics[1, 2] == image_size[1] / 2  # cy = height/2


def test_extrinsics_composition():
    """Test composition of extrinsics matrices."""
    # Define two transformations
    R1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity rotation
    t1 = torch.tensor([1.0, 0.0, 0.0])  # Translation by (1, 0, 0)

    R2 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 deg around Z
    t2 = torch.tensor([0.0, 1.0, 0.0])  # Translation by (0, 1, 0)

    # Compose transformations
    composed = compose_extrinsics(R1, t1, R2, t2)

    # Expected result: R1 @ R2 = R2 (since R1 is identity)
    # t_composed = R1 @ t2 + t1 = t2 + t1 = (1, 1, 0)
    expected_R = R2
    expected_t = torch.tensor([1.0, 1.0, 0.0])

    assert torch.allclose(composed[:3, :3], expected_R)
    assert torch.allclose(composed[:3, 3], expected_t)


if __name__ == "__main__":
    # Run all tests
    print("Running Dynamic3DGS utility tests...")

    test_camera_pose_creation()
    print("✓ Camera pose creation test passed")

    test_camera_projection()
    print("✓ Camera projection test passed")

    test_camera_unprojection()
    print("✓ Camera unprojection test passed")

    test_relative_pose_computation()
    print("✓ Relative pose computation test passed")

    test_normalize_camera_coordinates()
    print("✓ Camera coordinate normalization test passed")

    test_view_frustum()
    print("✓ View frustum test passed")

    test_intrinsics_estimation()
    print("✓ Intrinsics estimation test passed")

    test_extrinsics_composition()
    print("✓ Extrinsics composition test passed")

    print("\n🎉 All tests passed!")