"""
Loss functions and training schedule helpers for latent beam dynamics.
"""

import torch


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
