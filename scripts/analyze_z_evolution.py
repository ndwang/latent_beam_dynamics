"""Visualize how the z population evolves across element index.

Three panels:
  1. PCA scatter at selected element positions (spreading over time)
  2. Per-PC variance vs element index (which directions grow)
  3. Per-sample step-size heatmap (|Δz|² for each sample × step)

Usage:
    python scripts/visualize_z_evolution.py --data data/encoded_sectioned_1sec_10k
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("data/encoded_sectioned_1sec_10k"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=3000,
                   help="Samples to use for PCA scatter (speed/clarity tradeoff)")
    p.add_argument("--n-pcs", type=int, default=8,
                   help="Number of PCs to track variance for")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output or args.data / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading z_traj from {args.data} ...")
    z = np.load(args.data / "z_traj.npy", mmap_mode="r")  # (N, T+1, d)
    N, T1, d = z.shape
    T = T1 - 1  # number of steps
    print(f"  shape: {z.shape}  ({N} samples, {T} steps, {d} dims)")

    # ── subsample for scatter ───────────────────────────────────────────────
    rng = np.random.default_rng(0)
    idx = rng.choice(N, min(args.n_samples, N), replace=False)
    z_sub = np.array(z[idx])  # (n, T+1, d)  — materialize the subset

    # ── PCA fit on all positions combined ──────────────────────────────────
    print("Fitting PCA ...")
    z_flat = z_sub.reshape(-1, d)  # (n*(T+1), d)
    pca = PCA(n_components=args.n_pcs)
    pca.fit(z_flat)
    print(f"  variance explained by first {args.n_pcs} PCs: "
          f"{pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

    # Project all positions (full dataset) onto PCs for variance computation
    z_full = np.array(z).reshape(-1, d)  # (N*(T+1), d) — load once
    z_pc_full = pca.transform(z_full).reshape(N, T + 1, args.n_pcs)  # (N, T+1, n_pcs)

    # ── step sizes (full dataset) ───────────────────────────────────────────
    dz = np.array(z[:, 1:, :]) - np.array(z[:, :-1, :])   # (N, T, d)
    step_sq = (dz ** 2).mean(axis=2)                        # (N, T)

    # ── Figure 1: PCA scatter at selected positions ─────────────────────────
    scatter_positions = np.linspace(0, T, 9, dtype=int)     # 9 evenly spaced
    z_pca_sub = pca.transform(z_sub.reshape(-1, d)).reshape(
        len(idx), T + 1, args.n_pcs
    )

    # Common axis limits from the final position (widest spread)
    lim = np.percentile(np.abs(z_pca_sub[:, -1, :2]), 98) * 1.1

    fig1, axes = plt.subplots(3, 3, figsize=(11, 10))
    fig1.suptitle("Population in PC1–PC2 space vs element index", fontsize=13)
    cmap = plt.cm.viridis
    for ax, t in zip(axes.flat, scatter_positions):
        xy = z_pca_sub[:, t, :2]
        ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.25,
                   color=cmap(t / T), rasterized=True)
        # 1-sigma ellipse from covariance
        cov = np.cov(xy.T)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 0)
        angle = np.degrees(np.arctan2(*evecs[:, -1][::-1]))
        from matplotlib.patches import Ellipse
        for nsig in (1, 2):
            ell = Ellipse(
                xy=(xy[:, 0].mean(), xy[:, 1].mean()),
                width=2 * nsig * np.sqrt(evals[-1]),
                height=2 * nsig * np.sqrt(evals[0]),
                angle=angle,
                edgecolor=cmap(t / T), facecolor="none",
                linewidth=1.2, linestyle="--" if nsig == 2 else "-",
            )
            ax.add_patch(ell)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"element {t}", fontsize=9)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=7)

    fig1.tight_layout()
    p1 = out_dir / "z_pca_scatter.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved {p1}")

    # ── Figure 2: per-PC variance vs element index ──────────────────────────
    pc_var = z_pc_full.var(axis=0)   # (T+1, n_pcs) — variance across samples
    total_var = z_full.reshape(N, T + 1, d).var(axis=0).mean(axis=1)  # (T+1,)

    steps = np.arange(T + 1)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Variance growth across element index", fontsize=12)

    ax = axes2[0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, args.n_pcs))
    for k in range(args.n_pcs):
        ax.plot(steps, pc_var[:, k], color=colors[k],
                label=f"PC{k+1} ({pca.explained_variance_ratio_[k]:.1%})")
    ax.set_xlabel("Element index")
    ax.set_ylabel("Variance across samples")
    ax.set_title("Per-PC variance")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = axes2[1]
    ax2.plot(steps, total_var, color="k", linewidth=2, label="Total (all dims)")
    ax2.fill_between(steps, 0, total_var, alpha=0.15, color="k")
    ax2.set_xlabel("Element index")
    ax2.set_ylabel("Mean variance across samples")
    ax2.set_title("Total latent variance")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    p2 = out_dir / "z_variance_growth.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved {p2}")

    # ── Figure 3: per-sample step-size heatmap ──────────────────────────────
    # Sort samples by total trajectory MSE so structure is visible
    total_mse = step_sq.mean(axis=1)
    sort_idx = np.argsort(total_mse)
    step_sq_sorted = step_sq[sort_idx]

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Per-sample step size |Δz|² (sample × element index)", fontsize=12)

    # Show a manageable subset for the heatmap
    n_show = min(500, N)
    stride = max(1, N // n_show)
    ss = step_sq_sorted[::stride]  # (≤500, T)

    im = axes3[0].imshow(
        ss, aspect="auto", cmap="hot",
        vmin=0, vmax=np.percentile(step_sq, 95),
        extent=[0.5, T + 0.5, n_show, 0],
    )
    axes3[0].set_xlabel("Element index")
    axes3[0].set_ylabel("Sample (sorted by mean step size)")
    axes3[0].set_title("Heatmap (clipped at 95th percentile)")
    fig3.colorbar(im, ax=axes3[0], label="|Δz|²")

    # Mean ± std curve on the right
    mean_ss = step_sq.mean(axis=0)
    std_ss = step_sq.std(axis=0)
    element_idx = np.arange(1, T + 1)
    axes3[1].plot(element_idx, mean_ss, color="C0", linewidth=2, label="mean")
    axes3[1].fill_between(element_idx,
                          np.maximum(0, mean_ss - std_ss),
                          mean_ss + std_ss,
                          alpha=0.3, color="C0", label="±1 std")
    axes3[1].fill_between(element_idx,
                          np.maximum(0, mean_ss - 2 * std_ss),
                          mean_ss + 2 * std_ss,
                          alpha=0.1, color="C0", label="±2 std")
    axes3[1].set_xlabel("Element index")
    axes3[1].set_ylabel("|Δz|²")
    axes3[1].set_title("Population mean ± std of step size")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    fig3.tight_layout()
    p3 = out_dir / "z_step_heatmap.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved {p3}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
