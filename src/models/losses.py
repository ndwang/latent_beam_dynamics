"""Loss functions for latent beam dynamics."""

import torch


def trajectory_mse_loss(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error averaged over batch, sequence, and latent dims.

    L = (1 / (B·N·d)) Σ (z^GT − ẑ)²
    """
    return ((z_pred - z_gt) ** 2).mean()
