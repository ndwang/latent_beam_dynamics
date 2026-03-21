"""
Tracking Transformer for latent beam dynamics.

GPT-style causal transformer that predicts beam state evolution
through a sequence of accelerator elements.
"""

import math

import torch
import torch.nn as nn

from .common import ElementEncoder, ModelConfig

TrackingConfig = ModelConfig


class CausalTransformer(nn.Module):
    """Multi-layer causal self-attention transformer (GPT-style).

    Token construction:  Token_t = Project(z_{t-1}) + h_t
    A causal mask ensures the prediction of z_t depends only on
    {z_0 … z_{t-1}} and {e_1 … e_t}.

    The output is the predicted Δz.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fusion = config.fusion
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)

        if config.fusion == "concat":
            self.fusion_proj = nn.Linear(2 * config.d_model, config.d_model)
        elif config.fusion == "bilinear":
            self.fusion_proj = nn.Linear(3 * config.d_model, config.d_model)
        elif config.fusion != "add":
            raise ValueError(f"Unknown fusion mode: {config.fusion!r}")

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
        z_p = self.z_proj(z_prev)
        if self.fusion == "add":
            tokens = z_p + h
        elif self.fusion == "concat":
            tokens = self.fusion_proj(torch.cat([z_p, h], dim=-1))
        else:  # bilinear
            tokens = self.fusion_proj(torch.cat([z_p, h, z_p * h], dim=-1))
        mask = self._causal_mask(tokens.size(1), tokens.device)
        out = self.out_norm(self.transformer(tokens, mask=mask))
        return self.delta_proj(out)


class TrackingTransformer(nn.Module):
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
