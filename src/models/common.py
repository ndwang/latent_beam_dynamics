"""
Common components for latent beam dynamics models.

Contains configuration, element encoding, and positional encoding
shared across model variants.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Beam latent state dimension (from pre-trained VAE)
    latent_dim: int = 256
    # Transformer hidden dimension
    d_model: int = 256
    # Number of transformer layers
    n_layers: int = 6
    # Number of attention heads
    n_heads: int = 8
    # Number of Fourier frequency pairs for positional encoding
    n_freq: int = 32
    # Raw element parameter dimension: [L, K1, K2, Angle, V_rf, f_rf, phi_rf]
    element_dim: int = 7
    # Wavelength range for Fourier positional encoding (meters)
    lambda_min: float = 0.01
    lambda_max: float = 1000.0
    # Dropout rate
    dropout: float = 0.1
    # Feed-forward expansion ratio
    mlp_ratio: int = 4
    # Fusion mode for combining latent state and element embeddings
    # Valid values: "add", "concat", "bilinear"
    fusion: str = "concat"


# Physics-informed normalization scales (design doc Section 3, Module A.2)
# Applied after any per-channel transforms (see ElementEncoder.forward).
#   L     -> / 1.0 m
#   K1    -> / 10.0 m^-2
#   K2    -> / 10.0 m^-2
#   Angle -> / 2π
#   V_rf  -> log1p(MV) / log1p(10)   [normalises at 10 MV; 0 stays 0 for non-RF]
#   f_rf  -> log1p(GHz) / log1p(1)   [normalises at 1 GHz;  0 stays 0 for non-RF]
#   phi_rf-> / 2π
_NORM_SCALES = [1.0, 10.0, 10.0, 2.0 * math.pi, math.log1p(10.0), math.log(2.0), 2.0 * math.pi]


class ElementEncoder(nn.Module):
    """Maps heterogeneous raw element parameters to position-aware embeddings.

    Combines physics-informed normalization, an MLP projection,
    and Fourier positional encoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.register_buffer(
            "norm_scales", torch.tensor(_NORM_SCALES, dtype=torch.float32)
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.element_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.pos_encoder = ContinuousPositionalEncoding(config)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, N, element_dim) raw element parameters.
        Returns:
            (B, N, d_model) position-mixed element embeddings.
        """
        # Log-transform V_rf (idx 4, MV) and f_rf (idx 5, GHz) before normalising.
        # log1p maps 0→0 so non-RF elements (V_rf=f_rf=0) are unaffected.
        x = x_raw.clone()
        x[..., 4] = torch.log1p(x_raw[..., 4])
        x[..., 5] = torch.log1p(x_raw[..., 5])
        x_norm = x / self.norm_scales
        element_emb = self.mlp(x_norm)
        lengths = x_raw[..., 0]
        positions = ContinuousPositionalEncoding.compute_positions(lengths)
        return self.pos_encoder(element_emb, positions)


class ContinuousPositionalEncoding(nn.Module):
    """Fourier-feature positional encoding for longitudinal position s.

    Frequencies are geometrically spaced so the model can resolve structure
    at scales from centimetres to kilometres.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # ω = 1/λ,  geometrically spaced
        omega_min = 1.0 / config.lambda_max
        omega_max = 1.0 / config.lambda_min
        log_freqs = torch.linspace(math.log(omega_min), math.log(omega_max), config.n_freq)
        self.register_buffer("freqs", torch.exp(log_freqs))  # (n_freq,)

        pos_dim = 2 * config.n_freq  # sin + cos for each frequency
        self.mix_proj = nn.Linear(config.d_model + pos_dim, config.d_model)

    @staticmethod
    def compute_positions(lengths: torch.Tensor) -> torch.Tensor:
        """Starting position of each element:  s_i = Σ_{j<i} L_j.

        Args:
            lengths: (B, N) element lengths.
        Returns:
            (B, N) longitudinal start positions.
        """
        return torch.cumsum(lengths, dim=-1) - lengths

    def forward(
        self, element_emb: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            element_emb: (B, N, d_model) from element encoder.
            positions:   (B, N)           longitudinal positions s.
        Returns:
            (B, N, d_model) position-mixed element embeddings.
        """
        angles = 2.0 * math.pi * positions.unsqueeze(-1) * self.freqs  # (B, N, n_freq)
        pos_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mix_proj(torch.cat([element_emb, pos_features], dim=-1))
