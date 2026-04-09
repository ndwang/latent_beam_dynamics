"""Encode tracked beam data into VAE latent trajectories for transformer training.

Loads pre-computed frequency maps, scales, and centroids, encodes them through
a trained VAE (using the deterministic mu, not sampled z), and assembles the
results into trajectory arrays compatible with LatentTrajectoryDataset.

Outputs:
    z_traj.npy    : (N, seq_len+1, latent_dim)  — latent beam states per sample
    elements.npy  : (N, seq_len, element_dim)    — element parameters per sample
    vae_meta.json : provenance metadata (VAE checkpoint, config, encoding settings)

Usage:
    python scripts/encode_latent.py \
        --vae-run /pscratch/sd/n/ndwang/vae/runs/beta_1e-5_260401_1523 \
        --maps data/vae_training/sectioned_10k_maps.npy \
        --scales data/vae_training/sectioned_10k_scales.npy \
        --centroids data/vae_training/sectioned_10k_centroids.npy \
        --data-dir data/sectioned_10k \
        --output data/encoded_sectioned_10k \
        --batch-size 1024
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "/pscratch/sd/n/ndwang/vae")

from beam_vae.models import VAE2D


def load_vae(run_dir: Path, device: torch.device) -> tuple[VAE2D, dict]:
    """Load a trained VAE from a run directory.

    Returns the model in eval mode and the full config dict.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = VAE2D(config)

    # Find best checkpoint
    ckpt_path = run_dir / f"{config['run_name']}_best.pth"
    if not ckpt_path.exists():
        # Fall back to final checkpoint
        ckpt_path = run_dir / f"{config['run_name']}.pth"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded VAE from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  latent_dim={model.latent_dim}, val_loss={checkpoint.get('val_loss', '?'):.6f}")
    return model, config, str(ckpt_path)


@torch.no_grad()
def encode_maps(
    model: VAE2D,
    maps: np.ndarray,
    scales: np.ndarray,
    centroids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode all frequency maps through the VAE encoder, returning mu vectors.

    Args:
        model: Trained VAE in eval mode.
        maps: (N, C, H, W) frequency maps.
        scales: (N, 6) raw (unnormalized) scales.
        centroids: (N, 6) raw (unnormalized) centroids.
        batch_size: Encoding batch size.
        device: Torch device.

    Returns:
        (N, latent_dim) array of mu vectors.
    """
    n_total = maps.shape[0]
    latent_dim = model.latent_dim
    all_mu = np.empty((n_total, latent_dim), dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)

        maps_batch = torch.from_numpy(maps[start:end].copy()).float().to(device)
        scales_batch = torch.from_numpy(scales[start:end].copy()).float().to(device)
        centroids_batch = torch.from_numpy(centroids[start:end].copy()).float().to(device)

        # Normalize using the model's stored stats (same as training)
        scales_norm = model.normalize_scales(scales_batch)
        centroids_norm = model.normalize_centroids(centroids_batch)

        mu, _ = model.encode(maps_batch, scales_norm, centroids_norm)
        all_mu[start:end] = mu.cpu().numpy()

        if (start // batch_size) % 20 == 0:
            print(f"  Encoded {end}/{n_total} maps")

    return all_mu


def collect_elements(data_dir: Path, n_samples: int) -> np.ndarray:
    """Load per-sample elements.npy files and stack into a single array."""
    sample_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "elements.npy").exists()
    )[:n_samples]

    if len(sample_dirs) != n_samples:
        raise ValueError(
            f"Expected {n_samples} sample dirs with elements.npy, found {len(sample_dirs)}"
        )

    elements_list = []
    for d in sample_dirs:
        elements_list.append(np.load(d / "elements.npy"))

    elements = np.stack(elements_list)  # (N, seq_len, element_dim)
    print(f"Collected elements: {elements.shape}")
    return elements


def main():
    parser = argparse.ArgumentParser(description="Encode beam data into VAE latent trajectories.")
    parser.add_argument("--vae-run", type=Path, required=True, help="VAE run directory (contains config.yaml + checkpoints)")
    parser.add_argument("--maps", type=Path, required=True, help="Path to frequency maps .npy")
    parser.add_argument("--scales", type=Path, required=True, help="Path to scales .npy")
    parser.add_argument("--centroids", type=Path, required=True, help="Path to centroids .npy")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with tracked sample subdirs (for elements.npy)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for encoded dataset")
    parser.add_argument("--batch-size", type=int, default=1024, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load VAE
    model, vae_config, ckpt_path = load_vae(args.vae_run, device)

    # Load pre-computed data
    print(f"Loading maps from {args.maps}")
    maps = np.load(args.maps, mmap_mode="r")
    scales = np.load(args.scales, mmap_mode="r")
    centroids = np.load(args.centroids, mmap_mode="r")
    print(f"  maps: {maps.shape}, scales: {scales.shape}, centroids: {centroids.shape}")

    n_total = maps.shape[0]

    # Count samples from the tracked data directory
    sample_dirs = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "elements.npy").exists()
    )
    n_samples = len(sample_dirs)

    if n_total % n_samples != 0:
        raise ValueError(
            f"Total maps ({n_total}) is not evenly divisible by number of samples ({n_samples}). "
            f"Some samples may have had snapshots skipped. Re-run prepare_vae_data.py "
            f"or filter to samples with complete trajectories."
        )

    snapshots_per_sample = n_total // n_samples
    seq_len = snapshots_per_sample - 1  # first snapshot is z0, rest are after each element
    print(f"  {n_samples} samples, {snapshots_per_sample} snapshots each (seq_len={seq_len})")

    # Encode
    print("Encoding maps through VAE...")
    mu_flat = encode_maps(model, maps, scales, centroids, args.batch_size, device)
    print(f"  Encoded: {mu_flat.shape}")

    # Reshape to per-sample trajectories
    latent_dim = mu_flat.shape[1]
    z_traj = mu_flat.reshape(n_samples, snapshots_per_sample, latent_dim)
    print(f"  z_traj: {z_traj.shape}")

    # Collect elements
    elements = collect_elements(args.data_dir, n_samples)
    assert elements.shape[1] == seq_len, (
        f"elements seq_len ({elements.shape[1]}) != expected ({seq_len})"
    )

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    z_path = args.output / "z_traj.npy"
    e_path = args.output / "elements.npy"
    np.save(z_path, z_traj)
    np.save(e_path, elements)
    print(f"Saved z_traj:   {z_path}  shape={z_traj.shape}")
    print(f"Saved elements: {e_path}  shape={elements.shape}")

    # Save provenance metadata
    meta = {
        "vae_checkpoint": ckpt_path,
        "vae_run_dir": str(args.vae_run),
        "vae_config": vae_config,
        "source_maps": str(args.maps),
        "source_scales": str(args.scales),
        "source_centroids": str(args.centroids),
        "source_data_dir": str(args.data_dir),
        "encoding": {
            "method": "mu",
            "note": "Deterministic encoding using encoder mean (no sampling)",
        },
        "dataset": {
            "n_samples": n_samples,
            "snapshots_per_sample": snapshots_per_sample,
            "seq_len": seq_len,
            "latent_dim": latent_dim,
            "element_dim": int(elements.shape[2]),
            "z_traj_shape": list(z_traj.shape),
            "elements_shape": list(elements.shape),
        },
        "created": datetime.now().isoformat(),
    }
    meta_path = args.output / "vae_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # Quick stats
    print(f"\nLatent stats:")
    print(f"  mu range: [{z_traj.min():.3f}, {z_traj.max():.3f}]")
    print(f"  mu mean:  {z_traj.mean():.4f}")
    print(f"  mu std:   {z_traj.std():.4f}")
    print(f"  NaN count: {np.isnan(z_traj).sum()}")


if __name__ == "__main__":
    main()
