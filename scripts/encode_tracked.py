"""Stage 3: Encode tracked particle data into VAE latent vectors.

Reads Bmad output HDF5 files, encodes each beam snapshot with the VAE,
and produces the final z_traj.npy and elements.npy for training.

Parallelism strategy:
  - CPU workers (--workers, default: all CPUs): read HDF5 + compute
    frequency maps in parallel across samples.
  - GPU (single device): batched VAE forward passes over the maps
    accumulated from all workers.

Usage:
    python scripts/encode_tracked.py \
        --input-dir data/sectioned_1sec_10k \
        --vae-checkpoint ../vae/runs/beta_1e-5_260401_1523/beta_1e-5_260401_1523_best.pth \
        --output-dir data/encoded_sectioned_1sec_10k
"""

import argparse
import json
import os
import yaml
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.encode import load_vae


# ---------------------------------------------------------------------------
# CPU worker: read HDF5 + compute frequency maps for one sample
# ---------------------------------------------------------------------------

def _process_sample(args):
    """Read HDF5 and compute frequency maps for all snapshots in one sample.

    Runs in a worker process (CPU-only). Returns None on failure.

    Returns:
        (maps, scales, centroids, elements) where maps/scales/centroids are
        (n_snapshots, ...) arrays ready for batched VAE encoding, or None.
    """
    sample_dir, tracked_filename, bins, n_sigma = args
    sample_dir = Path(sample_dir)
    tracked_path = sample_dir / tracked_filename
    elements_path = sample_dir / 'elements.npy'

    if not tracked_path.exists() or not elements_path.exists():
        return None

    try:
        import h5py
        from pmd_beamphysics import ParticleGroup
        from pmd_beamphysics.readers import particle_paths
        from beam_vae.data.preprocessing import particles_to_frequency_maps

        SPEED_OF_LIGHT = 299792458.0

        maps_list, scales_list, centroids_list = [], [], []

        with h5py.File(tracked_path, 'r') as f:
            pp = particle_paths(f)
            for path in pp[:-1]:  # skip last (redundant end marker)
                pg = ParticleGroup(h5=f[path + "electron"])
                alive = pg.status == 1
                if alive.sum() < 100:
                    return None  # too many lost particles

                h5_group = f[path + "electron"]
                p0c = h5_group["totalMomentumOffset"].attrs["value"][0]

                xp = pg.px[alive] / p0c
                yp = pg.py[alive] / p0c
                pz_alive = pg.pz[alive]
                delta = (pz_alive - p0c) / p0c
                # time field in HDF5 is t - t_ref; pg.t is absolute and must not be used
                z_bunch = -pg.beta[alive] * SPEED_OF_LIGHT * h5_group["time"][:][alive]

                particles = np.column_stack([
                    pg.x[alive], xp, pg.y[alive], yp, z_bunch, delta,
                ])

                maps, scales, centroids = particles_to_frequency_maps(
                    particles, bins=bins, n_sigma=n_sigma,
                )
                maps_list.append(maps)
                scales_list.append(scales)
                centroids_list.append(centroids)

        if not maps_list:
            return None

        elements = np.load(elements_path)
        return (
            np.array(maps_list, dtype=np.float32),      # (T, C, H, W)
            np.array(scales_list, dtype=np.float32),     # (T, n_scales)
            np.array(centroids_list, dtype=np.float32),  # (T, n_centroids)
            elements.astype(np.float32),                 # (seq_len, 7)
        )

    except Exception as e:
        print(f"  Warning: skipping {sample_dir.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batched GPU encoding
# ---------------------------------------------------------------------------

def _encode_maps_batched(maps_list, scales_list, centroids_list, vae, device, batch_size):
    """Encode pre-computed frequency maps with the VAE in batches.

    Args:
        maps_list: list of (T, C, H, W) arrays, one per sample.
        scales_list: list of (T, n_scales) arrays.
        centroids_list: list of (T, n_centroids) arrays.
        vae: loaded VAE model on `device`.
        batch_size: number of snapshots per GPU forward pass.

    Returns:
        List of (T, latent_dim) latent arrays, one per sample.
    """
    # Flatten to (total_snapshots, ...) for batched encoding
    sample_lengths = [m.shape[0] for m in maps_list]
    all_maps = np.concatenate(maps_list, axis=0)
    all_scales = np.concatenate(scales_list, axis=0)
    all_centroids = np.concatenate(centroids_list, axis=0)

    n_total = all_maps.shape[0]
    all_latents = np.zeros((n_total, vae.latent_dim), dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        maps_t = torch.from_numpy(all_maps[start:end]).to(device)
        scales_t = torch.from_numpy(all_scales[start:end]).to(device)
        centroids_t = torch.from_numpy(all_centroids[start:end]).to(device)

        scales_t = vae.normalize_scales(scales_t)
        centroids_t = vae.normalize_centroids(centroids_t)

        with torch.no_grad():
            mu, _ = vae.encode(maps_t, scales_t, centroids_t)

        all_latents[start:end] = mu.cpu().numpy()

    # Split back into per-sample trajectories
    latents_per_sample = []
    offset = 0
    for length in sample_lengths:
        latents_per_sample.append(all_latents[offset:offset + length])
        offset += length

    return latents_per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode tracked data with VAE")
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--vae-checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--tracked-filename', type=str, default='beam_dump.h5')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=None,
                        help="CPU workers for histogram computation (default: all CPUs)")
    parser.add_argument('--batch-size', type=int, default=512,
                        help="Snapshots per GPU forward pass (default: 512)")
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--n-sigma', type=int, default=4)
    parser.add_argument('--flush-samples', type=int, default=5000,
                        help="Flush accumulated maps to GPU after this many samples "
                             "to bound RAM usage (default: 5000)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or os.cpu_count()

    print(f"Device: {args.device}  |  CPU workers: {n_workers}  |  "
          f"GPU batch size: {args.batch_size}  |  flush every: {args.flush_samples} samples")

    # Load VAE on GPU
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device=args.device)

    # Find all sample directories
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / args.tracked_filename).exists()
    ])
    print(f"Found {len(sample_dirs)} tracked samples")

    if not sample_dirs:
        print("No tracked samples found. Run Bmad tracking first.")
        return

    tasks = [
        (str(d), args.tracked_filename, args.bins, args.n_sigma)
        for d in sample_dirs
    ]

    all_latents_list  = []
    all_elements_list = []
    total_skipped = 0
    seq_len = None

    # Accumulation buffers — flushed to GPU every `flush_samples` valid samples.
    # This pipelines CPU histogram work with GPU encoding and caps RAM to
    # O(flush_samples) frequency maps rather than O(n_total).
    maps_buf, scales_buf, centroids_buf, elements_buf = [], [], [], []

    def flush():
        if not maps_buf:
            return
        latents = _encode_maps_batched(
            maps_buf, scales_buf, centroids_buf, vae, args.device, args.batch_size,
        )
        all_latents_list.extend(latents)
        all_elements_list.extend(elements_buf)
        maps_buf.clear(); scales_buf.clear()
        centroids_buf.clear(); elements_buf.clear()

    print("Processing (CPU histograms → GPU encode, pipelined)...")
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap(_process_sample, tasks),
            total=len(tasks),
            desc="Samples",
        ):
            if result is None:
                total_skipped += 1
                continue

            maps, scales, centroids, elements = result

            if seq_len is None:
                seq_len = elements.shape[0]
            if maps.shape[0] != seq_len + 1:
                total_skipped += 1
                continue

            maps_buf.append(maps)
            scales_buf.append(scales)
            centroids_buf.append(centroids)
            elements_buf.append(elements)

            if len(maps_buf) >= args.flush_samples:
                flush()

    flush()  # encode any remaining samples

    # --- Save ---
    z_traj_all   = np.array(all_latents_list, dtype=np.float32)   # (N, T, latent_dim)
    elements_all = np.array(all_elements_list, dtype=np.float32)  # (N, seq_len, 7)

    np.save(output_dir / 'z_traj.npy', z_traj_all)
    np.save(output_dir / 'elements.npy', elements_all)

    n_saved = len(all_latents_list)
    print(f"\nSaved {n_saved} samples (skipped {total_skipped})")
    print(f"  z_traj.npy:  {z_traj_all.shape}")
    print(f"  elements.npy: {elements_all.shape}")

    # Save provenance metadata so evaluate.py can load the VAE for physical-space plots
    vae_ckpt_path = Path(args.vae_checkpoint)
    vae_run_dir = vae_ckpt_path.parent
    vae_config_path = vae_run_dir / "config.yaml"
    vae_config = yaml.safe_load(vae_config_path.read_text()) if vae_config_path.exists() else None
    meta = {
        "vae_checkpoint": str(vae_ckpt_path),
        "vae_run_dir": str(vae_run_dir),
        "vae_config": vae_config,
        "source_data_dir": str(input_dir),
        "encoding": {
            "method": "mu",
            "note": "Deterministic encoding using encoder mean (no sampling)",
        },
        "dataset": {
            "n_samples": int(z_traj_all.shape[0]),
            "snapshots_per_sample": int(z_traj_all.shape[1]),
            "seq_len": int(z_traj_all.shape[1]) - 1,
            "latent_dim": int(z_traj_all.shape[2]),
            "element_dim": int(elements_all.shape[2]),
            "z_traj_shape": list(z_traj_all.shape),
            "elements_shape": list(elements_all.shape),
        },
        "created": datetime.now().isoformat(),
    }
    meta_path = output_dir / "vae_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  vae_meta.json: {meta_path}")


if __name__ == '__main__':
    main()
