"""Encode tracked particle data into VAE latent vectors.

Reads Bmad output HDF5 files containing particle coordinates at each
element boundary, converts to frequency maps, and encodes with the VAE.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from pmd_beamphysics import ParticleGroup

# Add VAE project to path
VAE_ROOT = Path(__file__).resolve().parents[3] / 'vae'
sys.path.insert(0, str(VAE_ROOT))

from src.data.preprocessing import particles_to_frequency_maps
from src.models.vae2d import VAE2D


def load_vae(checkpoint_path: str | Path, device: str = 'cpu') -> VAE2D:
    """Load a trained VAE model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to load model on.

    Returns:
        VAE2D model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config matching the trained model
        config = {
            'model': {
                'input_channels': 15,
                'hidden_channels': [32, 64, 128, 256, 512],
                'latent_dim': 64,
                'input_size': 64,
                'kernel_size': 3,
                'activation': 'relu',
                'batch_norm': True,
                'dropout_rate': 0.0,
                'weight_init': 'kaiming_normal',
                'output_activation': 'sigmoid',
                'use_reparameterization': True,
            }
        }

    model = VAE2D(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def particles_to_latent(
    particles: np.ndarray,
    vae: VAE2D,
    device: str = 'cpu',
    bins: int = 64,
    n_sigma: int = 4,
) -> np.ndarray:
    """Encode a single particle distribution into a latent vector.

    Args:
        particles: (N, 6) array of [x, y, z, px, py, pz].
        vae: Loaded VAE model.
        device: Device for inference.
        bins: Histogram resolution.
        n_sigma: Grid extent in sigma units.

    Returns:
        (latent_dim,) latent vector (mu, deterministic).
    """
    maps, scales = particles_to_frequency_maps(particles, bins=bins, n_sigma=n_sigma)

    maps_t = torch.from_numpy(maps.astype(np.float32)).unsqueeze(0).to(device)
    scales_t = torch.from_numpy(scales.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = vae.encode(maps_t, scales_t)

    return mu.squeeze(0).cpu().numpy()


def encode_tracked_sample(
    tracked_h5_path: str | Path,
    vae: VAE2D,
    device: str = 'cpu',
) -> np.ndarray:
    """Encode all beam snapshots from a tracked HDF5 file.

    Expects the HDF5 file to contain particle groups at each element
    boundary, readable by openPMD-beamphysics.

    Args:
        tracked_h5_path: Path to Bmad output HDF5.
        vae: Loaded VAE model.
        device: Device for inference.

    Returns:
        (n_snapshots, latent_dim) array of latent vectors.
    """
    import h5py

    tracked_h5_path = Path(tracked_h5_path)
    latents = []

    with h5py.File(tracked_h5_path, 'r') as f:
        # Bmad typically writes groups numbered by element index
        group_names = sorted(f.keys(), key=_sort_key)

        for name in group_names:
            P = ParticleGroup(h5=f[name])
            coords = np.column_stack([P.x, P.y, P.z, P.px, P.py, P.pz])
            z = particles_to_latent(coords, vae, device=device)
            latents.append(z)

    return np.array(latents)


def _sort_key(name: str):
    """Sort HDF5 group names numerically if possible."""
    try:
        return int(name)
    except ValueError:
        return name
