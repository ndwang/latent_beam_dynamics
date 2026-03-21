#!/usr/bin/env python
"""Sanity check for TrackingTransformer and LatticeTransformer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models import (
    TrackingTransformer, TrackingConfig,
    LatticeTransformer, LatticeConfig,
    trajectory_mse_loss,
)

B, N = 4, 20  # batch of 4, lattice with 20 elements

for fusion in ("add", "concat", "bilinear"):
    print(f"\n{'='*60}")
    print(f"Fusion mode: {fusion}")
    print(f"{'='*60}")

    cfg = TrackingConfig(latent_dim=32, d_model=128, n_layers=4, n_heads=8, fusion=fusion)
    model = TrackingTransformer(cfg)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Config:     {cfg}")
    print(f"Parameters: {param_count:,}")

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

print(f"\n{'='*60}")
print("All fusion modes passed.")

# ---------------------------------------------------------------------------
# LatticeTransformer
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("LatticeTransformer (Option B)")
print(f"{'='*60}")

cfg_b = LatticeConfig(latent_dim=32, d_model=128, n_layers=4, n_heads=8)
model_b = LatticeTransformer(cfg_b)
param_count_b = sum(p.numel() for p in model_b.parameters())
print(f"Config:     {cfg_b}")
print(f"Parameters: {param_count_b:,}")

z0 = torch.randn(B, cfg_b.latent_dim)
x_raw = torch.randn(B, N, cfg_b.element_dim).abs()
z_gt = torch.randn(B, N, cfg_b.latent_dim)

z_pred_b = model_b(z0, x_raw)
loss_b = trajectory_mse_loss(z_pred_b, z_gt)
print(f"Forward pass     — output shape: {z_pred_b.shape}, loss: {loss_b.item():.4f}")

loss_b.backward()
print("Backward pass OK")

# Verify no autoregressive dependency: same output regardless of call count
model_b.eval()
with torch.no_grad():
    out1 = model_b(z0, x_raw)
    out2 = model_b(z0, x_raw)
    assert torch.allclose(out1, out2), "Outputs should be deterministic in eval mode"
print("Determinism check OK")
