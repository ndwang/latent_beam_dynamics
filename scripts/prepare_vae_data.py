"""Convert tracked beam data (beam_dump.h5) to VAE training data (frequency maps).

Reads all snapshots from beam_dump.h5 files, converts each to a 15-channel
64x64 frequency map using the VAE preprocessing pipeline, and saves the
results as .npy files ready for VAE training.

Coordinates
-----------
Particles are converted to Courant-Snyder trace-space coordinates before
building frequency maps:

    (x, x', y, y', z, delta)

where:
    x, y     — transverse position [m]
    x' = px / p_ref  — horizontal angle [rad]
    y' = py / p_ref  — vertical angle [rad]
    z = -beta * c * time  — bunch-frame longitudinal position [m] (time = t - t_ref from HDF5)
    delta = (pz - p0c) / p0c  — relative momentum deviation [1] (p0c from HDF5)

Usage:
    python scripts/prepare_vae_data.py --data-dir data/sectioned_10k \
        --output data/vae_training/sectioned_10k --workers 128
"""

import argparse
import os
import sys
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/pscratch/sd/n/ndwang/vae")

import h5py
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.readers import particle_paths

from beam_vae.data.preprocessing import particles_to_frequency_maps

SPEED_OF_LIGHT = 299792458.0  # m/s


def _process_sample(args_tuple):
    """Process one sample directory: read all snapshots, convert to frequency maps.

    Skips the last snapshot (end marker, identical to previous).
    Skips snapshots with fewer than 100 alive particles.

    Returns:
        List of (maps, scales) tuples, or None on failure.
    """
    sample_dir, bins, n_sigma = args_tuple
    sample_dir = Path(sample_dir)
    dump_path = sample_dir / "beam_dump.h5"

    if not dump_path.exists():
        return None

    try:
        results = []
        with h5py.File(dump_path, "r") as f:
            pp = particle_paths(f)
            # Skip last snapshot (end marker, redundant)
            for path in pp[:-1]:
                pg = ParticleGroup(h5=f[path + "electron"])
                alive = pg.status == 1
                if alive.sum() < 100:
                    continue

                # Reference momentum from HDF5 design value
                pz_alive = pg.pz[alive]
                h5_group = f[path + "electron"]
                p0c = h5_group["totalMomentumOffset"].attrs["value"][0]

                xp = pg.px[alive] / p0c
                yp = pg.py[alive] / p0c

                delta = (pz_alive - p0c) / p0c

                # time field in HDF5 is t - t_ref; pg.t is absolute and must not be used
                z_bunch = -pg.beta[alive] * SPEED_OF_LIGHT * h5_group["time"][:][alive]

                particles = np.column_stack([
                    pg.x[alive], xp, pg.y[alive], yp, z_bunch, delta,
                ])
                maps, scales, centroids = particles_to_frequency_maps(
                    particles, bins=bins, n_sigma=n_sigma,
                )
                results.append((maps, scales, centroids))
        return results
    except Exception as e:
        print(f"  Skip {sample_dir.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert tracked beams to VAE frequency maps.",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory with tracked sample subdirs (containing beam_dump.h5)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output base path (produces <path>_maps.npy and <path>_scales.npy)",
    )
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins per axis")
    parser.add_argument("--n-sigma", type=float, default=4, help="Grid extent in sigma")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: all CPUs)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of sample dirs to process",
    )
    args = parser.parse_args()

    # Find sample directories
    sample_dirs = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "beam_dump.h5").exists()
    )
    if args.max_samples:
        sample_dirs = sample_dirs[: args.max_samples]

    print(f"Found {len(sample_dirs)} tracked samples in {args.data_dir}")

    n_workers = args.workers or os.cpu_count()
    n_workers = min(n_workers, len(sample_dirs))
    print(f"Processing with {n_workers} workers")

    tasks = [(d, args.bins, args.n_sigma) for d in sample_dirs]

    all_maps = []
    all_scales = []
    all_centroids = []
    n_skipped = 0

    with Pool(n_workers) as pool:
        from tqdm import tqdm
        for result in tqdm(
            pool.imap(_process_sample, tasks),
            total=len(tasks),
            desc="Converting",
        ):
            if result is None:
                n_skipped += 1
                continue
            for maps, scales, centroids in result:
                all_maps.append(maps)
                all_scales.append(scales)
                all_centroids.append(centroids)

    print(f"\nProcessed {len(sample_dirs) - n_skipped} samples, skipped {n_skipped}")
    print(f"Total frequency maps: {len(all_maps)}")

    # Stack and save
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    maps_arr = np.array(all_maps, dtype=np.float32)
    scales_arr = np.array(all_scales, dtype=np.float32)
    centroids_arr = np.array(all_centroids, dtype=np.float32)

    maps_path = f"{args.output}_maps.npy"
    scales_path = f"{args.output}_scales.npy"
    centroids_path = f"{args.output}_centroids.npy"

    np.save(maps_path, maps_arr)
    np.save(scales_path, scales_arr)
    np.save(centroids_path, centroids_arr)

    print(f"Saved maps:      {maps_path}  shape={maps_arr.shape}")
    print(f"Saved scales:    {scales_path}  shape={scales_arr.shape}")
    print(f"Saved centroids: {centroids_path}  shape={centroids_arr.shape}")

    # Save coordinate metadata for downstream consumers
    import json
    meta = {
        "coordinates": ["x", "xp", "y", "yp", "z", "delta"],
        "coordinate_units": ["m", "rad", "m", "rad", "m", "1"],
        "coordinate_descriptions": [
            "Horizontal position",
            "Horizontal angle (px / p_ref)",
            "Vertical position",
            "Vertical angle (py / p_ref)",
            "Bunch-frame longitudinal position (-beta*c*(t-t_ref))",
            "Relative momentum deviation ((pz - p0c) / p0c)",
        ],
        "n_samples": len(all_maps),
        "maps_shape": list(maps_arr.shape),
        "scales_shape": list(scales_arr.shape),
        "bins": args.bins,
        "n_sigma": args.n_sigma,
        "source": str(args.data_dir),
    }
    meta_path = f"{args.output}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta:   {meta_path}")


if __name__ == "__main__":
    main()
