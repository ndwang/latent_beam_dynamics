"""Dataset for latent beam trajectory sequences."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentTrajectoryDataset(Dataset):
    """Dataset of VAE-encoded beam trajectories through accelerator lattices.

    Expected directory layout (memory-mapped .npy files):
        data_dir/
            z_traj.npy    : (N, seq_len+1, latent_dim)  — latent states
                            z_traj[:, 0]  = initial state z0
                            z_traj[:, 1:] = ground-truth exit states
            elements.npy  : (N, seq_len, element_dim)   — raw element parameters

    Args:
        path: Path to the data directory containing z_traj.npy and elements.npy.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {path}")

        # Memory-map for zero-copy, lazy loading
        self.z_traj = np.load(path / "z_traj.npy", mmap_mode="r")    # (N, L+1, d_z)
        self.elements = np.load(path / "elements.npy", mmap_mode="r") # (N, L, d_e)

        assert self.z_traj.shape[0] == self.elements.shape[0], (
            f"Mismatch: z_traj has {self.z_traj.shape[0]} samples, "
            f"elements has {self.elements.shape[0]}"
        )
        assert self.z_traj.shape[1] == self.elements.shape[1] + 1, (
            "z_traj seq_len must be elements seq_len + 1"
        )

    def __len__(self) -> int:
        return self.z_traj.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z0, elements, z_gt).

        z0       : (latent_dim,)
        elements : (seq_len, element_dim)
        z_gt     : (seq_len, latent_dim)
        """
        z0 = self.z_traj[idx, 0]
        z_gt = self.z_traj[idx, 1:]
        elements = self.elements[idx]
        return z0, elements, z_gt
