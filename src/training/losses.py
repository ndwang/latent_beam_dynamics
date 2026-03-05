"""Loss functions for latent beam trajectory training."""

import torch


def trajectory_mse_loss(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error averaged over batch, sequence, and latent dims."""
    return ((z_pred - z_gt) ** 2).mean()
