"""
Lattice Transformer for latent beam dynamics.

Parallel (non-autoregressive) architecture that conditions on the initial
beam state z₀ via Adaptive Layer Norm, predicts per-element Δz, and
recovers the full trajectory with a cumulative sum.
"""

import torch
import torch.nn as nn

from .common import ElementEncoder, ModelConfig

LatticeConfig = ModelConfig


# ---------------------------------------------------------------------------
# Adaptive Layer Norm Transformer Layer
# ---------------------------------------------------------------------------

class AdaLNTransformerLayer(nn.Module):
    """Single transformer layer with Adaptive Layer Norm conditioning."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, causal_mask, gamma1, beta1, gamma2, beta2):
        # gamma/beta shapes: (B, d_model), broadcast over seq_len via unsqueeze(1)

        # Pre-norm attention with AdaLN
        x_norm = gamma1.unsqueeze(1) * self.norm1(x) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + attn_out

        # Pre-norm FFN with AdaLN
        x_norm = gamma2.unsqueeze(1) * self.norm2(x) + beta2.unsqueeze(1)
        x = x + self.ffn(x_norm)

        return x


# ---------------------------------------------------------------------------
# Beam Conditioner
# ---------------------------------------------------------------------------

class BeamConditioner(nn.Module):
    """Maps z₀ to all AdaLN parameters in a single forward pass."""

    def __init__(self, latent_dim, d_model, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.proj = nn.Linear(d_model, n_layers * 4 * d_model)

    def forward(self, z0):
        # z0: (B, latent_dim)
        c = self.mlp(z0)                              # (B, d_model)
        params = self.proj(c)                          # (B, n_layers * 4 * d_model)
        return params.view(-1, self.n_layers, 4, self.d_model)  # (B, n_layers, 4, d_model)

    def _init_to_identity(self):
        """Zero-init the projection so AdaLN starts as standard LayerNorm (gamma=1, beta=0)."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            for i in range(self.n_layers):
                # gamma1 = index 0, gamma2 = index 2 within each layer's 4 params
                base = i * 4 * self.d_model
                self.proj.bias[base : base + self.d_model] = 1.0                          # gamma1
                self.proj.bias[base + 2 * self.d_model : base + 3 * self.d_model] = 1.0   # gamma2


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class LatticeTransformer(nn.Module):
    """Lattice Transformer for Accelerator Beam Dynamics.

    Parallel architecture that conditions on the initial beam state z₀
    via Adaptive Layer Norm. Predicts per-element delta-z and recovers
    the full trajectory with a cumulative sum.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.element_encoder = ElementEncoder(config)

        self.beam_cond = BeamConditioner(config.latent_dim, config.d_model, config.n_layers)
        self.layers = nn.ModuleList([
            AdaLNTransformerLayer(config.d_model, config.n_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.out_norm = nn.LayerNorm(config.d_model)
        self.delta_proj = nn.Linear(config.d_model, config.latent_dim)

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style init, then set AdaLN to identity."""
        for p in self.parameters():
            if p.dim() < 2:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)
        self.beam_cond._init_to_identity()

    def forward(self, z0: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z0:    (B, latent_dim)       initial beam state.
            x_raw: (B, N, element_dim)   raw element parameters.
        Returns:
            z_traj: (B, N, latent_dim)   beam state at exit of each element.
        """
        B, N, _ = x_raw.shape

        # 1. Encode elements into tokens
        h = self.element_encoder(x_raw)                # (B, N, d_model)

        # 2. Beam conditioning parameters (all layers, one shot)
        cond = self.beam_cond(z0)                      # (B, n_layers, 4, d_model)

        # 3. Causal mask
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=h.device), diagonal=1
        )

        # 4. Transformer layers with AdaLN
        x = h
        for i, layer in enumerate(self.layers):
            x = layer(
                x, causal_mask,
                gamma1=cond[:, i, 0],
                beta1=cond[:, i, 1],
                gamma2=cond[:, i, 2],
                beta2=cond[:, i, 3],
            )

        # 5. Output head
        delta_z = self.delta_proj(self.out_norm(x))    # (B, N, latent_dim)

        # 6. Cumulative sum to recover trajectory
        return z0.unsqueeze(1) + torch.cumsum(delta_z, dim=1)
