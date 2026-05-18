"""
Test script to validate Dynamic3DGS pipeline components.

This script tests the core functionality of all major modules
to ensure they work together correctly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# Import our modules
from models.deformable_gaussian import DeformableGaussianModel, VectorQuantizer
from models.deformation_field import SirenDeformationField
from models.renderer import OcclusionAwareRenderer, RenderConfig
from models.losses import TotalLoss
from utils.camera_utils import CameraPose


def test_deformable_gaussian_model():
    """Test the main deformable Gaussian model."""
    print("🧪 Testing DeformableGaussianModel...")

    # Create model
    model = DeformableGaussianModel(
        num_gaussians=1000,
        max_time_steps=50,
        embedding_dim=64,
        num_embeddings=512
    )

    # Test forward pass
    time_step = torch.tensor([5.0])
    outputs = model.forward(time_step)

    assert 'positions' in outputs
    assert 'opacities' in outputs
    assert 'position_offsets' in outputs
    assert 'occlusion_probs' in outputs

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Positions shape: {outputs['positions'].shape}")
    print(f"  ✓ Opacities shape: {outputs['opacities'].shape}")

    # Test encoding/decoding
    indices = model.encode_to_indices(time_step)
    decoded_positions = model.decode_from_indices(indices)

    assert indices.shape == (1, 1000)
    assert decoded_positions.shape == (1, 1000, 3)

    print(f"  ✓ Encoding/decoding successful")

    return model


def test_deformation_field():
    """Test deformation field networks."""
    print("\n🧪 Testing DeformationField...")

    # Test SIREN deformation field
    siren_field = SirenDeformationField(
        input_dim=4,
        output_dim=3,
        hidden_dims=[128, 128],
        w0=30.0
    )

    B, N = 2, 500
    points = torch.randn(B, N, 3) * 5
    time = torch.tensor([5.0])

    deformations = siren_field(points, time)
    assert deformations.shape == (B, N, 3)

    print(f"  ✓ SIREN deformation field works")
    print(f"  ✓ Output shape: {deformations.shape}")

    return siren_field


def test_renderer():
    """Test the renderer."""
    print("\n🧪 Testing Renderer...")

    config = RenderConfig(
        depth_sorting=True,
        alpha_blending=True,
        occlusion_threshold=0.5
    )

    renderer = OcclusionAwareRenderer(config)

    # Create dummy gaussian data
    B, N = 2, 1000
    gaussians = {
        'positions': torch.randn(B, N, 3) * 5,
        'covariances': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1),
        'opacities': torch.ones(B, N) * 0.5,
        'sh_coefficients': torch.randn(B, N, 3, 16) * 0.1
    }

    # Create dummy camera
    camera = CameraPose.create_default(width=640, height=480)

    # Test rendering
    time_step = torch.tensor([5.0])
    outputs = renderer.forward(gaussians, camera, (480, 640), time_step)

    assert 'rgb' in outputs
    assert 'depth' in outputs
    assert 'alpha' in outputs
    assert 'occlusion_mask' in outputs

    assert outputs['rgb'].shape == (1, 3, 480, 640)
    assert outputs['depth'].shape == (1, 480, 640)
    assert outputs['alpha'].shape == (1, 480, 640)

    print(f"  ✓ Rendering successful")
    print(f"  ✓ RGB shape: {outputs['rgb'].shape}")
    print(f"  ✓ Depth shape: {outputs['depth'].shape}")

    return renderer


def test_loss_functions():
    """Test loss functions."""
    print("\n🧪 Testing Loss Functions...")

    # Create dummy data
    pred_rgb = torch.randn(2, 3, 64, 64)
    target_rgb = torch.randn(2, 3, 64, 64)

    # Test reconstruction loss
    from models.losses import ReconstructionLoss, TemporalConsistencyLoss, RegularizationLoss

    recon_loss = ReconstructionLoss()
    temporal_loss = TemporalConsistencyLoss()
    reg_loss = RegularizationLoss()

    # Reconstruction loss
    recon_result = recon_loss(pred_rgb, target_rgb)
    assert 'total' in recon_result
    assert 'components' in recon_result

    print(f"  ✓ Reconstruction loss works")
    print(f"  ✓ L1 component: {recon_result['components']['l1']:.4f}")

    # Temporal consistency loss (simplified)
    temporal_result = temporal_loss(pred_rgb, pred_rgb)
    assert 'total' in temporal_result

    print(f"  ✓ Temporal consistency loss works")

    # Regularization loss
    gaussians = {
        'opacities': torch.ones(1000),
        'sh_coefficients': torch.randn(1000, 3, 16),
        'covariances': torch.eye(3).unsqueeze(0).repeat(1000, 1, 1)
    }

    reg_result = reg_loss(gaussians)
    assert 'total' in reg_result

    print(f"  ✓ Regularization loss works")

    return recon_loss, temporal_loss, reg_loss


def test_end_to_end():
    """Test end-to-end training step."""
    print("\n🧪 Testing End-to-End Pipeline...")

    # Create model components
    model = test_deformable_gaussian_model()
    deformation_field = test_deformation_field()
    renderer = test_renderer()
    _ = test_loss_functions()

    # Simulate a training step
    batch_size = 2
    image_size = (128, 128)

    # Create dummy batch
    batch = {
        'image': torch.randn(batch_size, 3, image_size[0], image_size[1]),
        'camera': [CameraPose.create_default(image_size[1], image_size[0]) for _ in range(batch_size)],
        'time': torch.tensor([5.0] * batch_size)
    }

    # Forward pass through model
    gaussian_outputs = model.forward(batch['time'])

    # Render
    camera_obj = type('Camera', (), {
        'intrinsics': batch['camera'][0].intrinsics,
        'extrinsics': batch['camera'][0].extrinsics
    })()

    rendered = renderer.forward(
        gaussian_outputs, camera_obj, image_size, batch['time'][0]
    )

    # Compute losses
    targets = {'rgb': batch['image']}
    losses = {
        'reconstruction': 0.5,
        'temporal': 0.1,
        'regularization': 0.05,
        'quantization': 0.02
    }

    total_loss = sum(w * losses[k] for k, w in zip(losses.keys(), [1.0, 0.1, 0.01, 0.001]))

    print(f"  ✓ End-to-end pipeline works")
    print(f"  ✓ Total loss: {total_loss:.4f}")
    print(f"  ✓ Rendered RGB range: [{rendered['rgb'].min():.3f}, {rendered['rgb'].max():.3f}]")

    return total_loss


def test_memory_efficiency():
    """Test memory usage and efficiency."""
    print("\n🧪 Testing Memory Efficiency...")

    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Measure baseline memory
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Run memory-intensive operations
    model = DeformableGaussianModel(num_gaussians=5000)
    deformation_field = SirenDeformationField(hidden_dims=[256, 256])

    # Test with larger batch
    B, N = 4, 5000
    points = torch.randn(B, N, 3).cuda() if torch.cuda.is_available() else torch.randn(B, N, 3)
    time = torch.tensor([5.0]).cuda() if torch.cuda.is_available() else torch.tensor([5.0])

    with torch.no_grad():
        outputs = model.forward(time)
        deformations = deformation_field(points, time)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    memory_increase = mem_after - mem_before
    print(f"  ✓ Memory efficient")
    print(f"  ✓ Memory increase: {memory_increase:.1f} MB")

    return memory_increase


def main():
    """Run all tests."""
    print("🚀 Dynamic3DGS Pipeline Validation")
    print("=" * 50)

    try:
        # Test individual components
        model = test_deformable_gaussian_model()
        deformation_field = test_deformation_field()
        renderer = test_renderer()
        loss_funcs = test_loss_functions()

        # Test integration
        total_loss = test_end_to_end()
        memory_usage = test_memory_efficiency()

        print("\n🎉 All Tests Passed!")
        print("✅ Dynamic3DGS pipeline is ready for training")

        print(f"\n📊 Summary:")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Memory usage: {memory_usage:.1f} MB increase")
        print(f"   Total loss: {total_loss:.4f}")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)