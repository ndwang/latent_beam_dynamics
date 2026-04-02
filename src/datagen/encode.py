"""Encode tracked particle data into VAE latent vectors.

Reads Bmad output HDF5 files containing particle coordinates at each
element boundary, converts to frequency maps, and encodes with the VAE.
"""

import numpy as np
import torch
from pathlib import Path

from beam_vae.data.preprocessing import particles_to_frequency_maps
from beam_vae.models import VAE2D, ResidualVAE2D
from beam_vae.utils.config import load_yaml


def load_vae(checkpoint_path: str | Path, device: str = 'cpu'):
    """Load a trained VAE model from checkpoint.

    Loads config.yaml from the run directory to reconstruct the model.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to load model on.

    Returns:
        VAE model in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent
    config_path = run_dir / 'config.yaml'

    if config_path.exists():
        config = load_yaml(config_path)
    else:
        raise FileNotFoundError(
            f"No config.yaml found in {run_dir}. "
            "Cannot reconstruct model architecture."
        )

    model_name = config.get('model', {}).get('name', 'vae2d')
    model_cls = ResidualVAE2D if model_name == 'residual_vae2d' else VAE2D

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_cls(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model


def particles_to_latent(
    particles: np.ndarray,
    vae,
    device: str = 'cpu',
    bins: int = 64,
    n_sigma: int = 4,
) -> np.ndarray:
    """Encode a single particle distribution into a latent vector.

    Args:
        particles: (N, 6) array of [x, x', y, y', z, delta].
        vae: Loaded VAE model.
        device: Device for inference.
        bins: Histogram resolution.
        n_sigma: Grid extent in sigma units.

    Returns:
        (latent_dim,) latent vector (mu, deterministic).
    """
    maps, scales, centroids = particles_to_frequency_maps(
        particles, bins=bins, n_sigma=n_sigma,
    )

    maps_t = torch.from_numpy(maps.astype(np.float32)).unsqueeze(0).to(device)
    scales_t = torch.from_numpy(scales.astype(np.float32)).unsqueeze(0).to(device)
    centroids_t = torch.from_numpy(centroids.astype(np.float32)).unsqueeze(0).to(device)

    # Normalize scales/centroids if the model was trained with normalization
    scales_t = vae.normalize_scales(scales_t)
    centroids_t = vae.normalize_centroids(centroids_t)

    with torch.no_grad():
        mu, _ = vae.encode(maps_t, scales_t, centroids_t)

    return mu.squeeze(0).cpu().numpy()


def encode_tracked_sample(
    tracked_h5_path: str | Path,
    vae,
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
    from pmd_beamphysics import ParticleGroup
    from pmd_beamphysics.readers import particle_paths

    SPEED_OF_LIGHT = 299792458.0

    tracked_h5_path = Path(tracked_h5_path)
    latents = []

    with h5py.File(tracked_h5_path, 'r') as f:
        pp = particle_paths(f)
        # Skip last snapshot (end marker, redundant)
        for path in pp[:-1]:
            pg = ParticleGroup(h5=f[path + "electron"])
            alive = pg.status == 1
            if alive.sum() < 100:
                continue

            pz_alive = pg.pz[alive]
            h5_group = f[path + "electron"]
            p0c = h5_group["totalMomentumOffset"].attrs["value"][0]

            xp = pg.px[alive] / p0c
            yp = pg.py[alive] / p0c
            delta = (pz_alive - p0c) / p0c
            z_bunch = -pg.beta[alive] * SPEED_OF_LIGHT * pg.t[alive]

            particles = np.column_stack([
                pg.x[alive], xp, pg.y[alive], yp, z_bunch, delta,
            ])
            z = particles_to_latent(particles, vae, device=device)
            latents.append(z)

    return np.array(latents)
