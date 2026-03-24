#!/usr/bin/env python3
"""Scan beam_dump.h5 files and plot histogram of living particles at the last element."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics.readers import particle_paths


def count_alive(beam_dump_path: Path) -> int | None:
    """Return the number of alive particles (status==1) at the last element."""
    try:
        with h5py.File(beam_dump_path, "r") as f:
            pp = particle_paths(f)
            if not pp:
                return None
            last_path = pp[-1] + "electron/particleStatus"
            node = f[last_path]
            if isinstance(node, h5py.Dataset):
                # Per-particle status array
                status = np.array(node)
                return int(np.sum(status == 1))
            else:
                # openPMD constant record: all particles share the same status
                n_particles = int(node.attrs["shape"][0])
                value = int(node.attrs["value"][0])
                return n_particles if value == 1 else 0
    except Exception as e:
        print(f"  Error reading {beam_dump_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/structured"),
        help="Path to structured data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("alive_histogram.png"),
        help="Output plot path",
    )
    args = parser.parse_args()

    sample_dirs = sorted(
        d for d in args.data_dir.iterdir() if d.is_dir() and (d / "beam_dump.h5").exists()
    )
    print(f"Found {len(sample_dirs)} samples with beam_dump.h5")

    alive_counts = []
    missing = []
    for d in sample_dirs:
        n = count_alive(d / "beam_dump.h5")
        if n is not None:
            alive_counts.append(n)
        else:
            missing.append(d.name)

    alive_counts = np.array(alive_counts)
    print(f"\nScanned {len(alive_counts)} samples ({len(missing)} failed)")
    if missing:
        print(f"  Failed: {missing}")

    total = 100_000
    pct = alive_counts / total * 100
    print(f"\nAlive particle stats:")
    print(f"  Mean:   {alive_counts.mean():.0f} ({pct.mean():.2f}%)")
    print(f"  Median: {np.median(alive_counts):.0f} ({np.median(pct):.2f}%)")
    print(f"  Min:    {alive_counts.min()} ({pct.min():.2f}%)")
    print(f"  Max:    {alive_counts.max()} ({pct.max():.2f}%)")
    print(f"  Samples with >50% alive: {np.sum(pct > 50)}")
    print(f"  Samples with >90% alive: {np.sum(pct > 90)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(alive_counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of living particles (out of 100k)")
    ax.set_ylabel("Number of lattices")
    ax.set_title("Living particles at last element across lattices")
    ax.axvline(alive_counts.mean(), color="red", linestyle="--", label=f"Mean: {alive_counts.mean():.0f}")
    ax.axvline(np.median(alive_counts), color="orange", linestyle="--", label=f"Median: {np.median(alive_counts):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
