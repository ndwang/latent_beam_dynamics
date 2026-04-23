#!/usr/bin/env python
"""Analyze AR outlier samples to identify element types driving error growth.

Produces three figures in the output directory:
  mse_distribution.png       — histogram + CDF of per-sample mean AR MSE
  delta_mse_by_element.png   — violin plot of per-step error increment by element type
  outlier_trajectories.png   — AR error curves for top-k worst samples, element-type annotated

Usage:
    python scripts/analyze_ar_outliers.py runs/<run>/lbd_best.pth
    python scripts/analyze_ar_outliers.py runs/<run>/lbd_best.pth --data /path/to/data --top-k 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.eval import build_val_loader, load_checkpoint

# Element feature indices: [L, K1, K2, Angle, V_rf, f_rf, phi_rf]
_K1, _K2, _ANGLE, _VRF = 1, 2, 3, 4

# Priority: drift < quad < bend < sextupole < RF (last write wins)
ETYPE_ORDER  = ["drift", "quad", "bend", "sextupole", "RF"]
ETYPE_COLORS = {
    "drift":     "#aaaaaa",
    "quad":      "#4477AA",
    "bend":      "#EE6677",
    "sextupole": "#228833",
    "RF":        "#CCBB44",
}
_AR_MODELS = {"tracking", "dual_stream"}


def classify_elements(elements: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Classify each (N, T) element by dominant type. Returns object array of str."""
    types = np.full(elements.shape[:2], "drift", dtype=object)
    types[np.abs(elements[..., _K1])    > threshold] = "quad"
    types[np.abs(elements[..., _ANGLE]) > threshold] = "bend"
    types[np.abs(elements[..., _K2])    > threshold] = "sextupole"
    types[np.abs(elements[..., _VRF])   > threshold] = "RF"
    return types


@torch.no_grad()
def run_inference(model, model_name: str, loader, device):
    """AR + TF inference, also collecting raw element arrays."""
    z_gt_list, z_ar_list, z_tf_list, elem_list = [], [], [], []
    is_ar = model_name in _AR_MODELS

    for z0, elements, z_gt in tqdm(loader, desc="  inference", leave=False):
        z0_d    = z0.to(device)
        elem_d  = elements.to(device)
        z_gt_d  = z_gt.to(device)

        z_ar_list.append(model(z0_d, elem_d).cpu().numpy())
        if is_ar:
            z_tf_list.append(
                model(z0_d, elem_d, z_gt=z_gt_d, sampling_prob=0.0).cpu().numpy()
            )
        z_gt_list.append(z_gt.numpy())
        elem_list.append(elements.numpy())

    z_gt   = np.concatenate(z_gt_list)
    z_ar   = np.concatenate(z_ar_list)
    z_tf   = np.concatenate(z_tf_list) if z_tf_list else None
    elems  = np.concatenate(elem_list)
    return z_gt, z_ar, z_tf, elems


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 1: MSE distribution ──────────────────────────────────────────────────

def plot_mse_distribution(sample_mse: np.ndarray, output_dir: Path):
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))

    ax_hist.hist(sample_mse, bins=60, color="C0", edgecolor="none", alpha=0.8)
    for pct, ls in [(50, "-"), (90, "--"), (95, ":"), (99, "-.")]:
        v = np.percentile(sample_mse, pct)
        ax_hist.axvline(v, color="k", linestyle=ls, linewidth=1.0,
                        label=f"p{pct} = {v:.4f}")
    ax_hist.set_xlabel("Mean AR MSE (over trajectory)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Per-sample AR MSE distribution")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    sorted_mse = np.sort(sample_mse)
    cdf = np.arange(1, len(sorted_mse) + 1) / len(sorted_mse)
    ax_cdf.plot(sorted_mse, cdf, color="C0")
    for pct, ls in [(50, "-"), (90, "--"), (95, ":"), (99, "-.")]:
        v = np.percentile(sample_mse, pct)
        ax_cdf.axvline(v, color="k", linestyle=ls, linewidth=1.0, label=f"p{pct}")
    ax_cdf.set_xlabel("Mean AR MSE")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Cumulative distribution")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir / "mse_distribution.png")


# ── Plot 2: Delta MSE by element type ────────────────────────────────────────

def plot_delta_by_element_type(mse_per_step: np.ndarray, etypes: np.ndarray,
                                output_dir: Path):
    """Violin plot: per-step Δ MSE = mse[t] - mse[t-1], grouped by element type."""
    baseline = np.zeros((mse_per_step.shape[0], 1))
    delta = np.diff(np.concatenate([baseline, mse_per_step], axis=1), axis=1)  # (N, T)

    data_by_type = {et: delta[etypes == et].tolist() for et in ETYPE_ORDER}
    present = [et for et in ETYPE_ORDER if len(data_by_type[et]) > 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(1, len(present) + 1))

    parts = ax.violinplot(
        [data_by_type[et] for et in present],
        positions=positions, showmedians=True, showextrema=False, widths=0.7,
    )
    for body, et in zip(parts["bodies"], present):
        body.set_facecolor(ETYPE_COLORS[et])
        body.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{et}\n(n={len(data_by_type[et]):,})" for et in present], fontsize=9
    )
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylabel("Per-step AR MSE increment (Δ)")
    ax.set_title("Error growth by element type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "delta_mse_by_element.png")


# ── Plot 3: Outlier trajectories ──────────────────────────────────────────────

def plot_outlier_trajectories(mse_ar: np.ndarray, mse_tf: np.ndarray | None,
                               etypes: np.ndarray, top_k: int, output_dir: Path):
    """Per-step error curve + element-type strip for the top-k worst samples."""
    sample_mse = mse_ar.mean(axis=1)
    top_k = min(top_k, len(sample_mse))
    worst_idx = np.argsort(sample_mse)[-top_k:][::-1]

    steps = np.arange(mse_ar.shape[1])
    ncols = 2
    nrows = (top_k + ncols - 1) // ncols

    fig = plt.figure(figsize=(7 * ncols, 4 * nrows))
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for plot_i, sample_i in enumerate(worst_idx):
        row, col = divmod(plot_i, ncols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[10, 1], hspace=0.05,
        )
        ax_main  = fig.add_subplot(inner[0])
        ax_strip = fig.add_subplot(inner[1], sharex=ax_main)

        ax_main.semilogy(steps, mse_ar[sample_i], color="C0", label="AR")
        if mse_tf is not None:
            ax_main.semilogy(steps, mse_tf[sample_i], color="C1",
                             linestyle="--", label="TF")
        ax_main.set_ylabel("MSE", fontsize=8)
        ax_main.set_title(
            f"Sample {sample_i}   mean AR MSE = {sample_mse[sample_i]:.4f}",
            fontsize=9,
        )
        ax_main.legend(fontsize=7, loc="upper left")
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        for t, et in enumerate(etypes[sample_i]):
            ax_strip.axvspan(t - 0.5, t + 0.5, facecolor=ETYPE_COLORS[et], alpha=0.85)
        ax_strip.set_yticks([])
        ax_strip.set_xlabel("Element index", fontsize=8)
        ax_strip.set_xlim(-0.5, len(steps) - 0.5)

    # Hide unused subplots
    for j in range(top_k, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.add_subplot(outer[r, c]).set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=ETYPE_COLORS[et], label=et) for et in ETYPE_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(ETYPE_ORDER),
               fontsize=9, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    fig.suptitle(f"Top-{top_k} worst AR samples by mean MSE", fontsize=12)
    _save(fig, output_dir / "outlier_trajectories.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyze AR outlier samples")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data",       type=Path, default=None,
                   help="Override data path from config")
    p.add_argument("--output",     type=Path, default=None,
                   help="Output dir (default: <run_dir>/eval_ar_outliers/)")
    p.add_argument("--top-k",      type=int,  default=10,
                   help="Number of worst samples to plot (default: 10)")
    p.add_argument("--batch-size", type=int,  default=256)
    p.add_argument("--device",     type=str,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, model_name, label, config, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    if model_name not in _AR_MODELS:
        print(f"Warning: {model_name} is not an AR model — TF curve will be absent")

    output_dir = args.output or (args.checkpoint.parent / "eval_ar_outliers")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    loader = build_val_loader(config, args.data, args.batch_size)
    z_gt, z_ar, z_tf, elements = run_inference(model, model_name, loader, device)

    mse_ar = ((z_ar - z_gt) ** 2).mean(axis=2)                          # (N, T)
    mse_tf = ((z_tf - z_gt) ** 2).mean(axis=2) if z_tf is not None else None
    etypes = classify_elements(elements)                                  # (N, T)

    unique, counts = np.unique(etypes, return_counts=True)
    print("Element type counts:", dict(zip(unique, counts)))

    sample_mse = mse_ar.mean(axis=1)
    print(f"Per-sample AR MSE:  mean={sample_mse.mean():.5f}  "
          f"median={np.median(sample_mse):.5f}  "
          f"p90={np.percentile(sample_mse, 90):.5f}  "
          f"p99={np.percentile(sample_mse, 99):.5f}")

    plot_mse_distribution(sample_mse, output_dir)
    plot_delta_by_element_type(mse_ar, etypes, output_dir)
    plot_outlier_trajectories(mse_ar, mse_tf, etypes, args.top_k, output_dir)


if __name__ == "__main__":
    main()
