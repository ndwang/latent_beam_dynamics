#!/usr/bin/env python3
"""Post-generation data quality analysis.

Produces a comprehensive diagnostic report after Stage 2 (Bmad tracking):
survival statistics, beam size distribution, growth factors, and breakdowns
by section configuration (for sectioned lattices).

Usage:
    python scripts/analyze_data.py --data-dir data/sectioned
    python scripts/analyze_data.py --data-dir data/structured --output-dir data/structured/diagnostics
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.readers import particle_paths


# ==========================================================================
# Data reading
# ==========================================================================

def read_sample(sample_dir: Path) -> dict | None:
    """Read all relevant data from a tracked sample directory.

    Returns None if the sample cannot be read (missing files, no alive
    particles, etc.).
    """
    try:
        bp = json.load(open(sample_dir / "beam_params.json"))
        elements = np.load(sample_dir / "elements.npy")

        with h5py.File(sample_dir / "beam_dump.h5", "r") as f:
            pp = particle_paths(f)
            pg = ParticleGroup(h5=f[pp[-1] + "electron"])
            alive = pg.status == 1
            n_alive = int(alive.sum())
            n_total = len(pg.status)
            if n_alive == 0:
                rms_x = rms_y = np.nan
            else:
                rms_x = float(np.std(pg.x[alive]))
                rms_y = float(np.std(pg.y[alive]))

        init_rms_x = np.sqrt(bp["emit_x"] * bp["beta_x"])
        init_rms_y = np.sqrt(bp["emit_y"] * bp["beta_y"])

        # Growth factor (max of x, y)
        growth_x = rms_x / init_rms_x if init_rms_x > 0 else np.nan
        growth_y = rms_y / init_rms_y if init_rms_y > 0 else np.nan
        growth = max(growth_x, growth_y)

        # Element counts
        n_quads = int(np.sum(np.abs(elements[:, 1]) > 1e-12))
        n_bends = int(np.sum(np.abs(elements[:, 3]) > 1e-12))
        n_rf = int(np.sum(np.abs(elements[:, 4]) > 1e-12))
        n_sext = int(np.sum(np.abs(elements[:, 2]) > 1e-12))
        n_drifts = len(elements) - n_quads - n_bends - n_rf - n_sext

        record = {
            "name": sample_dir.name,
            "n_alive": n_alive,
            "n_total": n_total,
            "rms_x": rms_x,
            "rms_y": rms_y,
            "init_rms_x": init_rms_x,
            "init_rms_y": init_rms_y,
            "growth_x": growth_x,
            "growth_y": growth_y,
            "growth": growth,
            "n_quads": n_quads,
            "n_bends": n_bends,
            "n_rf": n_rf,
            "n_sext": n_sext,
            "n_drifts": n_drifts,
        }

        # Beam params
        for k in ["energy_GeV", "beta_x", "beta_y", "emit_x", "emit_y",
                   "sigma_delta"]:
            record[k] = bp[k]
        record["bmag"] = bp.get("bmag", np.nan)

        # Lattice info (sectioned mode only)
        li_path = sample_dir / "lattice_info.json"
        if li_path.exists():
            li = json.load(open(li_path))
            record["section_types"] = tuple(s["type"] for s in li["sections"])
            record["n_sections"] = len(li["sections"])
            record["mu_first"] = li["mu_first_section"]
        else:
            record["section_types"] = None
            record["n_sections"] = None
            record["mu_first"] = np.nan

        return record
    except Exception as e:
        print(f"  Skip {sample_dir.name}: {e}")
        return None


# ==========================================================================
# Report printing
# ==========================================================================

def print_report(records: list[dict]) -> None:
    """Print a text summary of data quality metrics."""
    n = len(records)
    alive_frac = np.array([r["n_alive"] / r["n_total"] for r in records])
    growth = np.array([r["growth"] for r in records])
    max_rms = np.maximum(
        np.array([r["rms_x"] for r in records]),
        np.array([r["rms_y"] for r in records]),
    )
    finite = np.isfinite(growth)

    print(f"\n{'='*60}")
    print(f" Data Quality Report — {n} samples")
    print(f"{'='*60}")

    # Survival
    print(f"\n--- Particle Survival ---")
    print(f"  Mean:    {np.mean(alive_frac)*100:.2f}%")
    print(f"  Median:  {np.median(alive_frac)*100:.2f}%")
    print(f"  Min:     {np.min(alive_frac)*100:.2f}%")
    print(f"  >90%:    {np.sum(alive_frac > 0.9)} / {n}")
    print(f"  100%:    {np.sum(alive_frac == 1.0)} / {n}")

    # Growth
    g = growth[finite]
    print(f"\n--- Beam Size Growth (max of x, y) ---")
    print(f"  Median:  {np.median(g):.1f}x")
    print(f"  Mean:    {np.mean(g):.1f}x")
    print(f"  Min:     {np.min(g):.2f}x")
    print(f"  Max:     {np.max(g):.1f}x")
    print(f"  <2x:     {np.sum(g < 2):>4} / {len(g)} ({np.mean(g < 2)*100:.1f}%)")
    print(f"  <10x:    {np.sum(g < 10):>4} / {len(g)} ({np.mean(g < 10)*100:.1f}%)")
    print(f"  <100x:   {np.sum(g < 100):>4} / {len(g)} ({np.mean(g < 100)*100:.1f}%)")
    print(f"  <1000x:  {np.sum(g < 1000):>4} / {len(g)} ({np.mean(g < 1000)*100:.1f}%)")

    # Beam size
    mr = max_rms[np.isfinite(max_rms)]
    print(f"\n--- Final RMS Beam Size (max of x, y) ---")
    print(f"  Median:  {np.median(mr)*1e3:.3f} mm")
    print(f"  <0.1mm:  {np.sum(mr < 1e-4):>4} / {len(mr)} ({np.mean(mr < 1e-4)*100:.1f}%)")
    print(f"  <1mm:    {np.sum(mr < 1e-3):>4} / {len(mr)} ({np.mean(mr < 1e-3)*100:.1f}%)")
    print(f"  <1cm:    {np.sum(mr < 1e-2):>4} / {len(mr)} ({np.mean(mr < 1e-2)*100:.1f}%)")
    print(f"  <10cm:   {np.sum(mr < 1e-1):>4} / {len(mr)} ({np.mean(mr < 1e-1)*100:.1f}%)")

    # Element composition
    print(f"\n--- Element Composition (averages) ---")
    for key, label in [("n_quads", "Quads"), ("n_drifts", "Drifts"),
                        ("n_bends", "Bends"), ("n_rf", "RF"), ("n_sext", "Sext")]:
        vals = [r[key] for r in records]
        print(f"  {label:<8} mean={np.mean(vals):.1f}  min={np.min(vals)}  max={np.max(vals)}")

    # Section breakdown (sectioned mode)
    section_types = [r["section_types"] for r in records if r["section_types"] is not None]
    if section_types:
        print(f"\n--- Growth by Section Configuration ---")
        type_counts = Counter(section_types)
        for config, count in type_counts.most_common():
            mask = np.array([r["section_types"] == config for r in records])
            g_sub = growth[mask & finite]
            if len(g_sub) > 0:
                label = "+".join(config)
                print(f"  {label:<30} {count:>4} samples  "
                      f"median={np.median(g_sub):.1f}x  p90={np.percentile(g_sub, 90):.0f}x")


# ==========================================================================
# Plotting
# ==========================================================================

def make_plots(records: list[dict], output_dir: Path) -> None:
    """Generate diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    growth = np.array([r["growth"] for r in records])
    rms_x = np.array([r["rms_x"] for r in records])
    rms_y = np.array([r["rms_y"] for r in records])
    max_rms = np.maximum(rms_x, rms_y)
    init_rms_x = np.array([r["init_rms_x"] for r in records])
    init_rms_y = np.array([r["init_rms_y"] for r in records])
    alive_frac = np.array([r["n_alive"] / r["n_total"] for r in records])
    bmag = np.array([r["bmag"] for r in records])
    mu = np.array([r["mu_first"] for r in records])
    finite = np.isfinite(growth)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Growth factor histogram
    ax = axes[0, 0]
    log_g = np.log10(growth[finite])
    ax.hist(log_g, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(log_g), color="red", linestyle="--",
               label=f"median={np.median(growth[finite]):.1f}x")
    ax.set_xlabel("log10(growth factor)")
    ax.set_ylabel("Count")
    ax.set_title("Beam size growth distribution")
    ax.legend()

    # 2. Final beam size histogram
    ax = axes[0, 1]
    log_rms = np.log10(max_rms[np.isfinite(max_rms)])
    ax.hist(log_rms, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(log_rms), color="red", linestyle="--",
               label=f"median={np.median(max_rms[np.isfinite(max_rms)])*1e3:.2f} mm")
    ax.set_xlabel("log10(max RMS beam size) [m]")
    ax.set_ylabel("Count")
    ax.set_title("Final beam size distribution")
    ax.legend()

    # 3. Survival histogram
    ax = axes[0, 2]
    ax.hist(alive_frac * 100, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Alive particles (%)")
    ax.set_ylabel("Count")
    ax.set_title("Particle survival")

    # 4. Initial vs final beam size
    ax = axes[1, 0]
    init_max = np.maximum(init_rms_x, init_rms_y)
    ax.scatter(init_max, max_rms, s=5, alpha=0.3, rasterized=True)
    lims = [min(init_max.min(), max_rms[np.isfinite(max_rms)].min()) * 0.5,
            max(init_max.max(), max_rms[np.isfinite(max_rms)].max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.3, label="no growth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Initial max RMS (m)")
    ax.set_ylabel("Final max RMS (m)")
    ax.set_title("Initial vs final beam size")
    ax.legend()

    # 5. B_mag vs growth (if available)
    ax = axes[1, 1]
    has_bmag = np.isfinite(bmag) & finite
    if has_bmag.sum() > 10:
        ax.scatter(bmag[has_bmag], growth[has_bmag], s=5, alpha=0.3, rasterized=True)
        ax.set_xlabel("B_mag (mismatch)")
        ax.set_ylabel("Growth factor")
        ax.set_yscale("log")
        r = np.corrcoef(bmag[has_bmag], np.log10(growth[has_bmag]))[0, 1]
        ax.set_title(f"Mismatch vs growth (r={r:.3f})")
    else:
        ax.scatter(np.degrees(mu[finite]), growth[finite], s=5, alpha=0.3, rasterized=True)
        ax.set_xlabel("Phase advance first section (deg)")
        ax.set_ylabel("Growth factor")
        ax.set_yscale("log")
        ax.set_title("Phase advance vs growth")

    # 6. Growth CDF
    ax = axes[1, 2]
    g_sorted = np.sort(growth[finite])
    cdf = np.arange(1, len(g_sorted) + 1) / len(g_sorted) * 100
    ax.plot(g_sorted, cdf)
    ax.set_xscale("log")
    ax.set_xlabel("Growth factor")
    ax.set_ylabel("Cumulative % of samples")
    ax.set_title("Growth factor CDF")
    ax.grid(True, alpha=0.3)
    # Mark key thresholds
    for thresh in [10, 100, 1000]:
        pct = np.mean(growth[finite] < thresh) * 100
        ax.axvline(thresh, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"{pct:.0f}%", xy=(thresh, pct), fontsize=8,
                    xytext=(5, -10), textcoords="offset points")

    fig.suptitle(f"Data Quality Diagnostics ({len(records)} samples)", fontsize=14)
    fig.tight_layout()
    path = output_dir / "diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-generation data quality analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/analyze_data.py --data-dir data/sectioned
  python scripts/analyze_data.py --data-dir data/structured --output-dir data/structured/diag
""",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing sample subdirs with beam_dump.h5",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save plots (default: <data-dir>/diagnostics)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples to analyze (for speed)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir / "diagnostics"

    # Find samples
    sample_dirs = sorted(
        d for d in args.data_dir.iterdir()
        if d.is_dir() and (d / "beam_dump.h5").exists()
    )
    if args.max_samples:
        sample_dirs = sample_dirs[: args.max_samples]

    print(f"Found {len(sample_dirs)} tracked samples in {args.data_dir}")

    # Read all samples
    records = []
    for d in sample_dirs:
        r = read_sample(d)
        if r is not None:
            records.append(r)

    if not records:
        print("No valid samples found.")
        return

    print(f"Successfully read {len(records)} / {len(sample_dirs)} samples")

    # Report and plots
    print_report(records)
    make_plots(records, args.output_dir)


if __name__ == "__main__":
    main()
