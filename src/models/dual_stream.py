"""
DualStreamTransformer for latent beam dynamics.

Two separate streams kept in their own spaces throughout:
  - Element stream: ElementEncoder → positionally-encoded tokens (B, N, d_model)
  - Beam stream:    z_proj(z_prev) + positional encoding  (B, T, d_model)

Each layer applies:
  1. Causal self-attention over beam tokens (beam sees its own history)
  2. Causal cross-attention, beam tokens as Q, element tokens as K/V
     (beam attends to elements it has already passed through)
  3. Feed-forward network

The two streams never share the same sequence — cross-attention is the
only bridge, with separate projection matrices for each direction.
Causality is enforced by the same upper-triangular mask for both attention
operations: at position t the beam may only attend to positions 0..t.

Forward modes (same interface as TrackingTransformer):
  - Teacher forcing  (z_gt provided, sampling_prob=0)  — fully parallel
  - Scheduled sampling (z_gt provided, sampling_prob>0) — sequential
  - Autoregressive   (z_gt=None)                        — sequential
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import ElementEncoder, ModelConfig

DualStreamConfig = ModelConfig


# ---------------------------------------------------------------------------
# Single transformer layer
# ---------------------------------------------------------------------------

class DualStreamLayer(nn.Module):
    """Causal self-attention + causal cross-attention + FFN, pre-norm."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        # 1. Causal self-attention over beam tokens
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross-attention: beam (Q) → element (K, V), causal
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        beam: torch.Tensor,
        elements: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            beam:       (B, T, d_model)  beam token sequence (length T)
            elements:   (B, T, d_model)  element token sequence (same length T)
            causal_mask: (T, T)          upper-triangular -inf mask
        Returns:
            (B, T, d_model) updated beam tokens
        """
        # Self-attention
        x = beam
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                attn_mask=causal_mask)[0]

        # Cross-attention: beam attends to elements it has passed through
        x = x + self.cross_attn(self.norm2(x), elements, elements,
                                 attn_mask=causal_mask)[0]

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DualStreamTransformer(nn.Module):
    """Dual-stream causal transformer for latent beam dynamics.

    Maintains separate beam and element streams throughout, coupled only
    by causal cross-attention. Beam states are predicted autoregressively;
    z_t = z_{t-1} + delta_z_t.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # Element stream (unchanged from LatticeTransformer / TrackingTransformer)
        self.element_encoder = ElementEncoder(config)

        # Beam stream projection — no explicit positional encoding; z already
        # encodes s-position as a latent direction learned by the VAE
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            DualStreamLayer(config.d_model, config.n_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.out_norm = nn.LayerNorm(config.d_model)
        self.delta_proj = nn.Linear(config.d_model, config.latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 style init; residual paths scaled by 1/sqrt(2*n_layers)."""
        for p in self.parameters():
            if p.dim() < 2:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.out_proj.weight,  std=residual_std)
            nn.init.normal_(layer.cross_attn.out_proj.weight, std=residual_std)
            nn.init.normal_(layer.ffn[2].weight,              std=residual_std)
        nn.init.normal_(self.delta_proj.weight, std=residual_std)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def _encode_beam_tokens(self, z_prev: torch.Tensor) -> torch.Tensor:
        """Project beam states from VAE latent space into d_model space.

        No explicit positional encoding: the VAE latent already encodes
        s-position as a learned direction, so the projection carries implicit
        positional information without redundant Fourier features.

        Args:
            z_prev: (B, T, latent_dim)
        Returns:
            (B, T, d_model)
        """
        return self.z_proj(z_prev)

    # ------------------------------------------------------------------
    # Forward modes
    # ------------------------------------------------------------------

    def forward_teacher_forcing(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Fully parallel forward pass using ground-truth beam states.

        Args:
            z0:    (B, latent_dim)
            x_raw: (B, N, element_dim)
            z_gt:  (B, N, latent_dim)
        Returns:
            z_pred: (B, N, latent_dim)
        """
        B, N, _ = x_raw.shape

        # Element stream — computed once in parallel
        h = self.element_encoder(x_raw)                          # (B, N, d_model)

        # Beam stream — shifted ground truth: [z0, z_gt_0, ..., z_gt_{N-2}]
        z_prev = torch.cat([z0.unsqueeze(1), z_gt[:, :-1]], dim=1)  # (B, N, latent_dim)
        beam = self._encode_beam_tokens(z_prev)                  # (B, N, d_model)

        causal_mask = self._causal_mask(N, x_raw.device)

        for layer in self.layers:
            beam = layer(beam, h, causal_mask)

        delta_z = self.delta_proj(self.out_norm(beam))           # (B, N, latent_dim)
        return z_prev + delta_z

    def _forward_sequential(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor | None = None,
        sampling_prob: float = 1.0,
    ) -> torch.Tensor:
        """Sequential forward for autoregressive inference or scheduled sampling."""
        # Element stream pre-computed in full — the lattice is known in advance
        h = self.element_encoder(x_raw)                          # (B, N, d_model)
        B, N, _ = h.shape

        z_cur = z0
        z_prev_list: list[torch.Tensor] = []
        z_pred_list: list[torch.Tensor] = []

        for t in range(N):
            z_prev_list.append(z_cur)
            z_prev_seq = torch.stack(z_prev_list, dim=1)         # (B, t+1, latent_dim)
            beam = self._encode_beam_tokens(z_prev_seq)          # (B, t+1, d_model)

            causal_mask = self._causal_mask(t + 1, x_raw.device)

            x = beam
            for layer in self.layers:
                x = layer(x, h[:, :t + 1], causal_mask)

            delta_z   = self.delta_proj(self.out_norm(x))[:, -1]  # (B, latent_dim)
            z_predicted = z_cur + delta_z
            z_pred_list.append(z_predicted)

            if t < N - 1:
                if z_gt is not None:
                    use_pred = torch.rand(B, 1, device=z0.device) < sampling_prob
                    z_cur = torch.where(use_pred, z_predicted.detach(), z_gt[:, t])
                else:
                    z_cur = z_predicted

        return torch.stack(z_pred_list, dim=1)                    # (B, N, latent_dim)

    def forward(
        self,
        z0: torch.Tensor,
        x_raw: torch.Tensor,
        z_gt: torch.Tensor | None = None,
        sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        """Unified forward — same interface as TrackingTransformer.

        Args:
            z0:            (B, latent_dim)
            x_raw:         (B, N, element_dim)
            z_gt:          (B, N, latent_dim) or None
            sampling_prob: 0 = teacher forcing; (0,1) = scheduled sampling;
                           ignored when z_gt is None (pure autoregressive).
        Returns:
            z_pred: (B, N, latent_dim)
        """
        if z_gt is not None and sampling_prob == 0.0:
            return self.forward_teacher_forcing(z0, x_raw, z_gt)
        return self._forward_sequential(z0, x_raw, z_gt, sampling_prob)
