"""
Latent-Space Causal Transformer for Accelerator Beam Dynamics

Implements the architecture from MODEL_DESIGN.md:
  Element Encoder
  Continuous Positional Encoding (Fourier features)
  GPT-Style Causal Transformer
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
    latent_dim: int = 64
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


# Physics-informed normalization scales (design doc Section 3, Module A.2)
#   L     -> / 1.0 m
#   K1    -> / 10.0 m^-2
#   K2    -> / 10.0 m^-2
#   Angle -> / 2π
#   V_rf  -> / 10.0 MV
#   f_rf  -> / 1.0 GHz
#   phi_rf-> / 2π
_NORM_SCALES = [1.0, 10.0, 10.0, 2.0 * math.pi, 10.0, 1.0, 2.0 * math.pi]


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
        x_norm = x_raw / self.norm_scales
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


class CausalTransformer(nn.Module):
    """Multi-layer causal self-attention transformer (GPT-style).

    Token construction:  Token_t = Project(z_{t-1}) + h_t
    A causal mask ensures the prediction of z_t depends only on
    {z_0 … z_{t-1}} and {e_1 … e_t}.

    The output is the predicted Δz.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN — more stable for deep models
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.out_norm = nn.LayerNorm(config.d_model)
        self.delta_proj = nn.Linear(config.d_model, config.latent_dim)

    def _init_residual_weights(self) -> None:
        """Scale residual-path projections by 1/sqrt(2*n_layers) for stable init."""
        n_layers = len(self.transformer.layers)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for layer in self.transformer.layers:
            nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular additive mask (-inf above the diagonal)."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self, z_prev: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_prev: (B, N, latent_dim)  previous beam states.
            h:      (B, N, d_model)     element + position embeddings.
        Returns:
            delta_z: (B, N, latent_dim) predicted state updates.
        """
        tokens = self.z_proj(z_prev) + h
        mask = self._causal_mask(tokens.size(1), tokens.device)
        out = self.out_norm(self.transformer(tokens, mask=mask))
        return self.delta_proj(out)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class LatentBeamTransformer(nn.Module):
    """Latent-Space Causal Transformer for Accelerator Beam Dynamics.

    Predicts the evolution of a VAE-encoded beam state through a variable
    sequence of accelerator elements by combining element encoding,
    Fourier positional encoding, and a causal transformer with
    delta-dynamics output.

    Three forward modes:
      1. **Teacher forcing** (``sampling_prob=0``, ``z_gt`` provided) —
         fully parallel, used at the start of training.
      2. **Scheduled sampling** (``0 < sampling_prob < 1``, ``z_gt`` provided) —
         sequential, gradually exposes the model to its own predictions.
      3. **Autoregressive** (``z_gt=None``) — sequential inference.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.element_encoder = ElementEncoder(config)
        self.transformer = CausalTransformer(config)

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 style weight initialization.

        Linear/embedding weights ~ N(0, 0.02).  Biases to zero.
        """
        for p in self.parameters():
            if p.dim() < 2:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)
        self.transformer._init_residual_weights()

    # -- forward modes ------------------------------------------------------

    def forward_teacher_forcing(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel forward pass using ground-truth beam states.

        Args:
            z0:    (B, latent_dim)       initial beam state.
            x_raw: (B, N, element_dim)   raw element parameters.
            z_gt:  (B, N, latent_dim)    ground-truth exit states.
        Returns:
            z_pred: (B, N, latent_dim)   predicted beam states.
        """
        h = self.element_encoder(x_raw)

        # z_prev = [z0, z_gt_1, z_gt_2, …, z_gt_{N-1}]
        z_prev = torch.cat([z0.unsqueeze(1), z_gt[:, :-1]], dim=1)

        delta_z = self.transformer(z_prev, h)
        return z_prev + delta_z

    def forward_autoregressive(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential inference using the model's own predictions.

        Args:
            z0:    (B, latent_dim)       initial beam state.
            x_raw: (B, N, element_dim)   raw element parameters.
        Returns:
            z_pred: (B, N, latent_dim)   predicted beam states.
        """
        h = self.element_encoder(x_raw)
        B, N, _ = h.shape
        d_z = self.config.latent_dim

        z_prev_buf = torch.empty(B, N, d_z, device=z0.device, dtype=z0.dtype)
        z_pred_buf = torch.empty(B, N, d_z, device=z0.device, dtype=z0.dtype)
        z_cur = z0

        for t in range(N):
            z_prev_buf[:, t] = z_cur

            delta_z = self.transformer(z_prev_buf[:, :t+1], h[:, :t+1])
            z_cur = z_cur + delta_z[:, -1]
            z_pred_buf[:, t] = z_cur

        return z_pred_buf

    def forward_scheduled_sampling(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor,
        sampling_prob: float,
    ) -> torch.Tensor:
        """Forward pass that randomly replaces GT inputs with predictions.

        At each step the model uses its own predicted z_{t-1} with
        probability ``sampling_prob``, and z_{t-1}^GT otherwise.

        Args:
            z0:            (B, latent_dim)
            x_raw:         (B, N, element_dim)
            z_gt:          (B, N, latent_dim)
            sampling_prob: probability of using the model's own prediction.
        Returns:
            z_pred: (B, N, latent_dim)
        """
        h = self.element_encoder(x_raw)
        B, N, _ = h.shape
        d_z = self.config.latent_dim

        z_prev_buf = torch.empty(B, N, d_z, device=z0.device, dtype=z0.dtype)
        z_pred_buf = torch.empty(B, N, d_z, device=z0.device, dtype=z0.dtype)
        z_cur = z0

        for t in range(N):
            z_prev_buf[:, t] = z_cur

            delta_z = self.transformer(z_prev_buf[:, :t+1], h[:, :t+1])
            z_predicted = z_cur + delta_z[:, -1]
            z_pred_buf[:, t] = z_predicted

            # Choose next input: predicted or ground truth
            if t < N - 1:
                use_pred = (torch.rand(B, 1, device=z0.device) < sampling_prob)
                z_cur = torch.where(use_pred, z_predicted.detach(), z_gt[:, t])

        return z_pred_buf

    # -- unified entry point ------------------------------------------------

    def forward(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor | None = None,
        sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        """Unified forward method.

        Args:
            z0:            (B, latent_dim)       initial beam state.
            x_raw:         (B, N, element_dim)   raw element parameters.
            z_gt:          (B, N, latent_dim)     ground truth (None → inference).
            sampling_prob: 0 = teacher forcing, (0,1) = scheduled sampling,
                           ignored when z_gt is None (pure autoregressive).
        Returns:
            z_pred: (B, N, latent_dim) predicted beam states.
        """
        if z_gt is not None and sampling_prob == 0.0:
            return self.forward_teacher_forcing(z0, x_raw, z_gt)
        if z_gt is not None:
            return self.forward_scheduled_sampling(z0, x_raw, z_gt, sampling_prob)
        return self.forward_autoregressive(z0, x_raw)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def trajectory_mse_loss(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error averaged over batch, sequence, and latent dims.

    L = (1 / (B·N·d)) Σ (z^GT − ẑ)²
    """
    return ((z_pred - z_gt) ** 2).mean()


def scheduled_sampling_prob(epoch: int, warmup: int = 10, k: float = 0.05) -> float:
    """Linearly increasing sampling probability after a warmup period.

    Returns 0 during warmup, then increases toward 1.
    """
    if epoch < warmup:
        return 0.0
    return min(1.0, (epoch - warmup) * k)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig(latent_dim=32, d_model=128, n_layers=4, n_heads=8)
    model = LatentBeamTransformer(cfg)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Config:     {cfg}")
    print(f"Parameters: {param_count:,}")

    B, N = 4, 20  # batch of 4, lattice with 20 elements
    z0 = torch.randn(B, cfg.latent_dim)
    x_raw = torch.randn(B, N, cfg.element_dim).abs()  # lengths should be positive
    z_gt = torch.randn(B, N, cfg.latent_dim)

    # Teacher forcing (parallel)
    z_pred_tf = model(z0, x_raw, z_gt=z_gt, sampling_prob=0.0)
    loss_tf = trajectory_mse_loss(z_pred_tf, z_gt)
    print(f"Teacher forcing  — output shape: {z_pred_tf.shape}, loss: {loss_tf.item():.4f}")

    # Scheduled sampling
    z_pred_ss = model(z0, x_raw, z_gt=z_gt, sampling_prob=0.3)
    loss_ss = trajectory_mse_loss(z_pred_ss, z_gt)
    print(f"Scheduled samp.  — output shape: {z_pred_ss.shape}, loss: {loss_ss.item():.4f}")

    # Autoregressive inference
    with torch.no_grad():
        z_pred_ar = model(z0, x_raw)
    print(f"Autoregressive   — output shape: {z_pred_ar.shape}")

    # Backward pass check
    loss_tf.backward()
    print("Backward pass OK")
