"""
Deformable Gaussian Model for Dynamic3DGS

This module implements the core deformable Gaussian representation
where each Gaussian has static attributes and dynamic deformation fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DeformableGaussian:
    """Representation of a single deformable Gaussian"""
    # Static parameters (shared across time)
    position: torch.Tensor  # [3] - initial 3D position
    covariance: torch.Tensor  # [3, 3] - initial covariance matrix
    opacity: torch.Tensor  # [] - opacity
    sh_coefficients: torch.Tensor  # [3, 16] - spherical harmonics coefficients

    # Dynamic parameters (time-dependent)
    time_steps: torch.Tensor  # [num_time_steps] - time indices
    position_offsets: torch.Tensor  # [num_time_steps, 3] - position offsets
    covariance_deformation: torch.Tensor  # [num_time_steps, 6] - covariance deformation (lower triangular)
    opacity_modulation: torch.Tensor  # [num_time_steps] - opacity modulation

    def __post_init__(self):
        """Validate tensor shapes and initialize if needed"""
        if self.covariance.shape != (3, 3):
            raise ValueError(f"Expected covariance shape (3, 3), got {self.covariance.shape}")

        if self.sh_coefficients.shape != (3, 16):
            raise ValueError(f"Expected SH coefficients shape (3, 16), got {self.sh_coefficients.shape}")

        # Ensure proper data types
        self.position = self.position.float()
        self.covariance = self.covariance.float()
        self.opacity = self.opacity.float().squeeze()
        self.sh_coefficients = self.sh_coefficients.float()


class DeformationField(nn.Module):
    """
    Learnable deformation field network.

    Maps from (x, y, z, t) to deformation vector (Δx, Δy, Δz).
    Uses SIREN-style activation functions for continuous representations.
    """

    def __init__(self,
                 input_dim: int = 4,      # x, y, z, t
                 output_dim: int = 3,     # Δx, Δy, Δz
                 hidden_dims: List[int] = [256, 256, 128, 128],
                 activation: str = "sine",
                 use_layer_norm: bool = True,
                 dropout: float = 0.1):
        """
        Initialize deformation field network.

        Args:
            input_dim: Input dimension (typically 4 for x,y,z,t)
            output_dim: Output dimension (typically 3 for position offset)
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("sine", "relu", "tanh")
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm
        self.dropout_p = dropout

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            if activation == "sine":
                # SIREN initialization
                if i == 0:  # First layer
                    layers.append(SiLU())  # Use SiLU for first layer
                else:
                    layers.append(SiLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights using SIREN initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using SIREN initialization"""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if self.activation_name == "sine":
                        # SIREN initialization
                        if len(self.hidden_dims) == 0 or m.in_features != self.input_dim:
                            # First layer
                            w = torch.sqrt(torch.tensor(1.0 / self.input_dim)) * torch.randn_like(m.weight)
                            m.weight.copy_(w)
                            if m.bias is not None:
                                m.bias.zero_()
                        else:
                            # Hidden layers
                            w = torch.sqrt(torch.tensor(1.0 / m.in_features)) * torch.randn_like(m.weight)
                            m.weight.copy_(w)
                            if m.bias is not None:
                                m.bias.zero_()
                    else:
                        # Standard Xavier initialization
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute deformation at given points and times.

        Args:
            points: [B, N, 3] 3D points in world coordinates
            time: [B, T] or [] time values

        Returns:
            deformations: [B, N, 3] deformation vectors
        """
        B, N, _ = points.shape

        # Expand time to match point dimensions
        if time.dim() == 0:  # Scalar
            time = time.expand(B, 1)
        elif time.dim() == 1:
            time = time.unsqueeze(1).expand(-1, N)

        # Concatenate point coordinates and time
        inputs = torch.cat([points, time.unsqueeze(-1)], dim=-1)  # [B, N, 4]

        # Apply network
        deformations = self.network(inputs)

        return deformations


class TemporalSmoothnessLoss(nn.Module):
    """
    Loss that encourages temporal smoothness in deformation fields.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, deformation_field: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal smoothness loss.

        Args:
            deformation_field: [B, T, N, 3] deformation field over time
            time_steps: [B, T] time values

        Returns:
            loss: scalar temporal smoothness loss
        """
        # Compute first-order differences (velocity)
        velocities = deformation_field[:, 1:] - deformation_field[:, :-1]  # [B, T-1, N, 3]

        # Compute second-order differences (acceleration)
        accelerations = velocities[:, 1:] - velocities[:, :-1]  # [B, T-2, N, 3]

        # L2 norm of accelerations
        acceleration_loss = torch.mean(accelerations ** 2)

        return self.weight * acceleration_loss


class OcclusionAwareModule(nn.Module):
    """
    Module for predicting occlusion relationships in dynamic scenes.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple MLP for occlusion prediction
        self.network = nn.Sequential(
            nn.Linear(7, hidden_dim),  # x,y,z,t + neighbor features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # occlusion probability
            nn.Sigmoid()
        )

    def forward(self, gaussian_positions: torch.Tensor,
                time_step: torch.Tensor,
                neighbor_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Predict occlusion probability for Gaussians.

        Args:
            gaussian_positions: [N, 3] current positions
            time_step: [] current time
            neighbor_indices: [N, K] indices of neighboring Gaussians

        Returns:
            occlusion_probs: [N] occlusion probabilities
        """
        N = gaussian_positions.shape[0]

        # Create input features
        features = []

        # Position and time
        pos_time = torch.cat([
            gaussian_positions,
            time_step.expand(N, 1)
        ], dim=1)

        features.append(pos_time)

        # Add neighbor information if provided
        if neighbor_indices is not None:
            neighbor_features = []
            for i in range(N):
                neighbors = gaussian_positions[neighbor_indices[i]]  # [K, 3]
                if len(neighbors) > 0:
                    dist_to_neighbors = torch.norm(
                        gaussian_positions[i:i+1].expand_as(neighbors) - neighbors,
                        dim=1
                    )
                    avg_dist = torch.mean(dist_to_neighbors)
                    min_dist = torch.min(dist_to_neighbors)
                else:
                    avg_dist = torch.ones(1, device=gaussian_positions.device) * 10.0
                    min_dist = torch.ones(1, device=gaussian_positions.device) * 10.0

                neighbor_feat = torch.stack([avg_dist, min_dist])
                neighbor_features.append(neighbor_feat)

            neighbor_features = torch.stack(neighbor_features)  # [N, 2]
            features.append(neighbor_features)

        # Concatenate all features
        x = torch.cat(features, dim=1)  # [N, feature_dim]

        # Predict occlusion probability
        occlusion_probs = self.network(x).squeeze(-1)

        return occlusion_probs


class DeformableGaussianModel(nn.Module):
    """
    Main deformable Gaussian model.

    Combines static Gaussian parameters with learnable deformation fields
    to create a dynamic scene representation.
    """

    def __init__(self,
                 num_gaussians: int = 10000,
                 max_time_steps: int = 100,
                 embedding_dim: int = 64,
                 num_embeddings: int = 512):
        """
        Initialize deformable Gaussian model.

        Args:
            num_gaussians: Number of Gaussians in the scene
            max_time_steps: Maximum number of time steps
            embedding_dim: Dimension of embedding vectors
            num_embeddings: Size of codebook
        """
        super().__init__()

        self.num_gaussians = num_gaussians
        self.max_time_steps = max_time_steps
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Static Gaussian parameters (shared across time)
        self.register_parameter('positions',
            nn.Parameter(torch.randn(num_gaussians, 3) * 0.1))
        self.register_parameter('covariances',
            nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_gaussians, 1, 1)))
        self.register_parameter('opacities',
            nn.Parameter(torch.ones(num_gaussians) * 0.5))
        self.register_parameter('sh_coefficients',
            nn.Parameter(torch.randn(num_gaussians, 3, 16) * 0.1))

        # Dynamic deformation field
        self.deformation_field = DeformationField(
            input_dim=4,  # x, y, z, t
            output_dim=3,  # position offset
            hidden_dims=[256, 256, 128, 128],
            activation="sine",
            use_layer_norm=True,
            dropout=0.1
        )

        # Occlusion prediction module
        self.occlusion_predictor = OcclusionAwareModule(hidden_dim=128)

        # Vector quantization for discrete representation
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=0.25
        )

        # Temporal regularization
        self.temporal_regularizer = TemporalSmoothnessLoss(weight=0.1)

    def forward(self, time_step: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the deformable Gaussian model.

        Args:
            time_step: [B] time step values

        Returns:
            outputs: Dictionary containing deformed Gaussians and losses
        """
        B = time_step.shape[0]

        # Get deformation field for current time
        # Expand positions to batch dimension
        positions_expanded = self.positions.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]
        time_expanded = time_step.unsqueeze(1).expand(-1, self.num_gaussians)  # [B, N]

        # Compute position offsets
        position_offsets = self.deformation_field(positions_expanded, time_expanded)  # [B, N, 3]

        # Apply deformation
        deformed_positions = positions_expanded + position_offsets  # [B, N, 3]

        # Compute occlusion probabilities
        occlusion_probs = self.occlusion_predictor(
            deformed_positions.view(-1, 3),
            time_step[0] if time_step.dim() > 0 else time_step
        ).view(B, -1)

        # Update opacities based on occlusion
        modulated_opacities = self.opacities.unsqueeze(0).expand(B, -1) * occlusion_probs

        # Return deformable Gaussian parameters
        outputs = {
            'positions': deformed_positions,
            'covariances': self.covariances.unsqueeze(0).expand(B, -1, -1),
            'opacities': modulated_opacities,
            'sh_coefficients': self.sh_coefficients.unsqueeze(0).expand(B, -1, -1, -1),
            'position_offsets': position_offsets,
            'occlusion_probs': occlusion_probs,
            'time_steps': time_step.unsqueeze(1).expand(-1, self.num_gaussians)
        }

        return outputs

    def encode_to_indices(self, time_step: torch.Tensor) -> torch.Tensor:
        """
        Encode Gaussians to discrete indices using vector quantization.

        Args:
            time_step: [B] time step values

        Returns:
            encoding_indices: [B, N] discrete indices
        """
        # Get deformed positions
        outputs = self.forward(time_step)
        deformed_positions = outputs['positions']

        # Flatten for quantization
        B, N, _ = deformed_positions.shape
        flat_positions = deformed_positions.view(B * N, 3)

        # Quantize
        _, _, encoding_indices = self.vector_quantizer(flat_positions.view(B, N, 3))

        return encoding_indices

    def decode_from_indices(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete indices back to positions.

        Args:
            encoding_indices: [B, N] discrete indices

        Returns:
            positions: [B, N, 3] reconstructed positions
        """
        B, N = encoding_indices.shape

        # Get quantized embeddings
        flat_indices = encoding_indices.view(-1, 1)
        encodings = torch.zeros(B * N, self.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, flat_indices, 1)

        quantized = torch.matmul(encodings, self.vector_quantizer.embeddings.weight)
        positions = quantized.view(B, N, 3)

        return positions


class VectorQuantizer(nn.Module):
    """Vector quantization layer for discrete representation learning."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with vector quantization.

        Args:
            inputs: [B, N, D] input vectors

        Returns:
            quantized: [B, N, D] quantized vectors
            loss: scalar quantization loss
            encoding_indices: [B, N] indices
        """
        B, N, D = inputs.shape

        # Reshape for distance computation
        flat_inputs = inputs.view(-1, D)

        # Compute distances
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
        )

        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Get quantized vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_flat = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized_flat.view(B, N, D)

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.view(B, N)


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


if __name__ == "__main__":
    # Example usage
    model = DeformableGaussianModel(
        num_gaussians=1000,
        max_time_steps=50,
        embedding_dim=64,
        num_embeddings=512
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    time_step = torch.tensor([5.0])
    outputs = model.forward(time_step)

    print(f"Output positions shape: {outputs['positions'].shape}")
    print(f"Output opacities shape: {outputs['opacities'].shape}")

    # Test encoding/decoding
    indices = model.encode_to_indices(time_step)
    print(f"Encoding indices shape: {indices.shape}")

    decoded_positions = model.decode_from_indices(indices)
    print(f"Decoded positions shape: {decoded_positions.shape}")