"""
Deformation Field Network for Dynamic3DGS

This module implements advanced deformation field architectures including
SIREN networks, Transformer-based deformation fields, and hybrid approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math


class Sine(nn.Module):
    """Sine activation function for SIREN networks."""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MLPDeformationField(nn.Module):
    """
    Multi-layer perceptron deformation field network.

    Simple but effective baseline for deformation learning.
    """

    def __init__(self,
                 input_dim: int = 4,      # x, y, z, t
                 output_dim: int = 3,     # Δx, Δy, Δz
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "relu",
                 use_layer_norm: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            ])

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute deformation vectors.

        Args:
            points: [B, N, 3] 3D points
            time: [B, T] or [] time values

        Returns:
            deformations: [B, N, 3] deformation vectors
        """
        B, N, _ = points.shape

        # Expand time to match point dimensions
        if time.dim() == 0:  # Scalar
            time_expanded = time.expand(B, N, 1)
        elif time.dim() == 1:
            time_expanded = time.unsqueeze(-1).expand(-1, N, 1)
        else:
            time_expanded = time

        # Concatenate inputs
        inputs = torch.cat([points, time_expanded], dim=-1)  # [B, N, 4]

        return self.network(inputs)


class SirenDeformationField(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) deformation field.

    Uses sine activations with specific weight initialization for
    continuous, high-frequency function representation.
    """

    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 3,
                 hidden_dims: List[int] = [256, 256, 128],
                 w0: float = 30.0,
                 first_w0: float = 30.0,
                 use_layer_norm: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.w0 = w0
        self.first_w0 = first_w0

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Determine w0 for this layer
            current_w0 = self.first_w0 if i == 0 else self.w0

            # Linear layer with proper scaling
            layers.append(
                ScaledLinear(prev_dim, hidden_dim, w0=current_w0)
            )

            # Activation
            if i == 0:
                layers.append(SiLU())  # Use SiLU for first layer
            else:
                layers.append(Sine(w0=self.w0))

            # Normalization and dropout
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(ScaledLinear(prev_dim, output_dim, w0=self.w0))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute deformation using SIREN architecture.

        Args:
            points: [B, N, 3] 3D points
            time: [B, T] or [] time values

        Returns:
            deformations: [B, N, 3] deformation vectors
        """
        B, N, _ = points.shape

        # Prepare inputs
        if time.dim() == 0:
            time_input = time.expand(B, N, 1)
        elif time.dim() == 1:
            time_input = time.unsqueeze(-1).expand(-1, N, 1)
        else:
            time_input = time

        inputs = torch.cat([points, time_input], dim=-1)

        return self.network(inputs)


class TransformerDeformationField(nn.Module):
    """
    Transformer-based deformation field network.

    Uses attention mechanisms to capture long-range dependencies
    in the deformation space.
    """

    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 3,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding for spatial coordinates
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights appropriately for transformer."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute deformation using transformer architecture.

        Args:
            points: [B, N, 3] 3D points
            time: [B, T] or [] time values

        Returns:
            deformations: [B, N, 3] deformation vectors
        """
        B, N, _ = points.shape

        # Prepare inputs
        if time.dim() == 0:
            time_input = time.expand(B, N, 1)
        elif time.dim() == 1:
            time_input = time.unsqueeze(-1).expand(-1, N, 1)
        else:
            time_input = time

        inputs = torch.cat([points, time_input], dim=-1)  # [B, N, 4]

        # Embed inputs
        x = self.input_embedding(inputs)  # [B, N, hidden_dim]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer_encoder(x)  # [B, N, hidden_dim]

        # Project to output
        deformations = self.output_projection(x)  # [B, N, 3]

        return deformations


class HybridDeformationField(nn.Module):
    """
    Hybrid deformation field combining local and global information.

    Uses a combination of MLP for local details and transformer
    for global context.
    """

    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 3,
                 local_hidden_dims: List[int] = [128, 128],
                 global_hidden_dim: int = 256,
                 num_global_layers: int = 2,
                 num_global_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # Local deformation branch (MLP)
        self.local_branch = MLPDeformationField(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=local_hidden_dims,
            activation="relu",
            dropout=dropout
        )

        # Global deformation branch (Transformer)
        self.global_branch = TransformerDeformationField(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_global_layers,
            num_heads=num_global_heads,
            hidden_dim=global_hidden_dim,
            dropout=dropout
        )

        # Fusion network
        fusion_dim = output_dim * 2
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize fusion network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and 'fusion' in m.__class__.__name__.lower():
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid deformation.

        Args:
            points: [B, N, 3] 3D points
            time: [B, T] or [] time values

        Returns:
            deformations: [B, N, 3] deformation vectors
        """
        # Get local and global deformations
        local_deform = self.local_branch(points, time)
        global_deform = self.global_branch(points, time)

        # Concatenate and fuse
        combined = torch.cat([local_deform, global_deform], dim=-1)
        fused_deform = self.fusion_network(combined)

        return fused_deform


class PositionalEncoding(nn.Module):
    """
    Positional encoding for spatial coordinates in transformers.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class ScaledLinear(nn.Linear):
    """
    Linear layer with scaled initialization for SIREN networks.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, w0: float = 30.0):
        super().__init__(in_features, out_features, bias)
        self.w0 = w0

    def reset_parameters(self):
        """Reset parameters with proper scaling."""
        # Scale weights by 1/sqrt(in_features)
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.normal_(self.weight, mean=0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class DeformationFieldFactory:
    """
    Factory class for creating different types of deformation fields.
    """

    @staticmethod
    def create(deformation_type: str,
               **kwargs) -> nn.Module:
        """
        Create deformation field based on type.

        Args:
            deformation_type: Type of deformation field
            **kwargs: Arguments for the specific implementation

        Returns:
            deformation_field: Configured deformation field network
        """
        if deformation_type.lower() == "mlp":
            return MLPDeformationField(**kwargs)
        elif deformation_type.lower() == "siren":
            return SirenDeformationField(**kwargs)
        elif deformation_type.lower() == "transformer":
            return TransformerDeformationField(**kwargs)
        elif deformation_type.lower() == "hybrid":
            return HybridDeformationField(**kwargs)
        else:
            raise ValueError(f"Unknown deformation type: {deformation_type}")


if __name__ == "__main__":
    # Example usage
    print("Testing deformation field implementations...")

    # Test MLP
    mlp_field = MLPDeformationField(
        input_dim=4,
        output_dim=3,
        hidden_dims=[128, 128],
        activation="relu"
    )
    print(f"MLP field parameters: {sum(p.numel() for p in mlp_field.parameters()):,}")

    # Test SIREN
    siren_field = SirenDeformationField(
        input_dim=4,
        output_dim=3,
        hidden_dims=[256, 256],
        w0=30.0
    )
    print(f"SIREN field parameters: {sum(p.numel() for p in siren_field.parameters()):,}")

    # Test Transformer
    transformer_field = TransformerDeformationField(
        input_dim=4,
        output_dim=3,
        num_layers=2,
        num_heads=4,
        hidden_dim=128
    )
    print(f"Transformer field parameters: {sum(p.numel() for p in transformer_field.parameters()):,}")

    # Test forward pass
    B, N = 2, 1000
    points = torch.randn(B, N, 3)
    time = torch.tensor([5.0])

    mlp_out = mlp_field(points, time)
    siren_out = siren_field(points, time)
    transformer_out = transformer_field(points, time)

    print(f"MLP output shape: {mlp_out.shape}")
    print(f"SIREN output shape: {siren_out.shape}")
    print(f"Transformer output shape: {transformer_out.shape}")

    # Test factory
    field = DeformationFieldFactory.create("siren", input_dim=4, output_dim=3)
    print(f"Factory-created field parameters: {sum(p.numel() for p in field.parameters()):,}")