#!/usr/bin/env python
"""Visualize full trajectories for worst / median / best AR-inference samples.

For each group (worst, median, best — 5 samples each by default):
  {group}/s{idx}_mse.png        — per-step MSE (AR+TF) + element-type color strip
  {group}/s{idx}_scales.png     — scales gt vs pred + relative error + element strip
  {group}/s{idx}_centroids.png  — centroids gt vs pred + absolute error + element strip
  {group}/s{idx}_phase.png      — phase-space maps at 8 elements (3 planes × gt/pred)

Also produces:
  group_mse_comparison.png      — per-step MSE for all selected samples, colored by group

Requires vae_meta.json in the data directory for physical-space plots.

Usage:
    python scripts/analyze_trajectory_cases.py runs/<run>/lbd_best.pth
    python scripts/analyze_trajectory_cases.py runs/<run>/lbd_best.pth \\
        --data data/encoded_sectioned_1sec_10k --n-per-group 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import ModelConfig, LatticeConfig, TrackingTransformer, LatticeTransformer, DualStreamTransformer
from src.eval import load_checkpoint, build_val_loader

_AR_MODELS = {"tracking", "dual_stream"}

# Element feature indices: [L, K1, K2, Angle, V_rf, f_rf, phi_rf]
_K1, _K2, _ANGLE, _VRF = 1, 2, 3, 4

ETYPE_ORDER  = ["drift", "quad", "bend", "sextupole", "RF"]
ETYPE_COLORS = {
    "drift":     "#aaaaaa",
    "quad":      "#4477AA",
    "bend":      "#EE6677",
    "sextupole": "#228833",
    "RF":        "#CCBB44",
}

SCALE_LABELS    = [r"$\sigma_x$", r"$\sigma_{x'}$", r"$\sigma_y$",
                   r"$\sigma_{y'}$", r"$\sigma_z$", r"$\sigma_\delta$"]
CENTROID_LABELS = [r"$\langle x\rangle$", r"$\langle x'\rangle$",
                   r"$\langle y\rangle$", r"$\langle y'\rangle$",
                   r"$\langle z\rangle$", r"$\langle\delta\rangle$"]


# ── Element classification ────────────────────────────────────────────────────

def classify_elements(elements: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Classify (N, T, 7) or (T, 7) element array → str labels (same leading dims)."""
    types = np.full(elements.shape[:-1], "drift", dtype=object)
    types[np.abs(elements[..., _K1])    > threshold] = "quad"
    types[np.abs(elements[..., _ANGLE]) > threshold] = "bend"
    types[np.abs(elements[..., _K2])    > threshold] = "sextupole"
    types[np.abs(elements[..., _VRF])   > threshold] = "RF"
    return types


# ── Element-type strip helpers ────────────────────────────────────────────────

def _fill_strip(ax, etypes_1d: np.ndarray):
    """Fill ax with per-element color spans (no ticks, no labels)."""
    T = len(etypes_1d)
    for t, et in enumerate(etypes_1d):
        ax.axvspan(t - 0.5, t + 0.5, facecolor=ETYPE_COLORS[et], alpha=0.85)
    ax.set_yticks([])
    ax.set_xlim(-0.5, T - 0.5)


def _strip_legend(fig, etypes_present, loc="lower center", ncol=None, anchor=(0.5, -0.02)):
    handles = [Patch(facecolor=ETYPE_COLORS[et], label=et)
               for et in ETYPE_ORDER if et in etypes_present]
    fig.legend(handles=handles, loc=loc,
               ncol=ncol or len(handles), fontsize=8,
               bbox_to_anchor=anchor, framealpha=0.9)


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, model_name: str, loader: DataLoader, device):
    """Returns z_gt (N,T,d), z_ar (N,T,d), z_tf or None, elements (N,T,7)."""
    is_ar = model_name in _AR_MODELS
    z_gt_list, z_ar_list, z_tf_list, elem_list = [], [], [], []

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

    return (
        np.concatenate(z_gt_list),
        np.concatenate(z_ar_list),
        np.concatenate(z_tf_list) if z_tf_list else None,
        np.concatenate(elem_list),
    )


# ── Sample selection ──────────────────────────────────────────────────────────

def select_groups(z_ar, z_gt, n):
    mse = ((z_ar - z_gt) ** 2).mean(axis=(1, 2))
    sorted_idx = np.argsort(mse)[::-1]
    worst  = sorted_idx[:n]
    best   = sorted_idx[-n:][::-1]
    mid    = len(sorted_idx) // 2
    half_n = n // 2
    median = sorted_idx[mid - half_n: mid - half_n + n]
    for name, idx in [("worst", worst), ("median", median), ("best", best)]:
        print(f"  {name:6s}: samples {list(idx)}  MSE={mse[idx].round(5)}")
    return worst, median, best, mse


# ── VAE loading & decoding ────────────────────────────────────────────────────

def load_vae(data_dir: Path, device):
    meta_path = data_dir / "vae_meta.json"
    if not meta_path.exists():
        print(f"Warning: {meta_path} not found — skipping physical-space plots")
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    vae_run_dir = Path(meta["vae_run_dir"])
    if str(vae_run_dir.parent) not in sys.path:
        sys.path.insert(0, str(vae_run_dir.parent))
    try:
        from beam_vae.models import VAE2D
        with open(vae_run_dir / "config.yaml") as f:
            vae_cfg = yaml.safe_load(f)
        vae = VAE2D(vae_cfg)
        run_name = vae_cfg.get("run_name", vae_run_dir.name)
        ckpt_path = vae_run_dir / f"{run_name}_best.pth"
        if not ckpt_path.exists():
            ckpt_path = vae_run_dir / f"{run_name}.pth"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        vae.to(device).eval()
        print(f"Loaded VAE from {ckpt_path}")
        return vae
    except Exception as exc:
        print(f"Warning: could not load VAE ({exc}) — skipping physical-space plots")
        return None


@torch.no_grad()
def decode_batch(vae, z_traj: np.ndarray, device, batch_size=512):
    """(N,T,d) → scales (N,T,6), centroids (N,T,6), maps (N,T,15,H,W)."""
    N, T, d = z_traj.shape
    z_flat = z_traj.reshape(N * T, d)
    sc_list, ct_list, mp_list = [], [], []
    for s in range(0, N * T, batch_size):
        z_b = torch.from_numpy(z_flat[s:s + batch_size]).float().to(device)
        recon, sc_norm, ct_norm = vae.decode(z_b)
        sc_list.append(vae.denormalize_scales(sc_norm).cpu().numpy())
        ct_list.append(vae.denormalize_centroids(ct_norm).cpu().numpy())
        mp_list.append(recon.cpu().numpy())
    scales    = np.concatenate(sc_list).reshape(N, T, 6)
    centroids = np.concatenate(ct_list).reshape(N, T, 6)
    H, W      = mp_list[0].shape[-2:]
    maps      = np.concatenate(mp_list).reshape(N, T, 15, H, W)
    return scales, centroids, maps


# ── Per-sample plots ──────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_mse(mse_ar_1d, mse_tf_1d, etypes_1d, sample_idx, group, out):
    """Per-step MSE curve + element-type strip."""
    T = len(mse_ar_1d)
    steps = np.arange(T)

    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.05)
    ax_main  = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_main)

    ax_main.semilogy(steps, mse_ar_1d, color="C0", linewidth=1.6, label="AR")
    if mse_tf_1d is not None:
        ax_main.semilogy(steps, mse_tf_1d, color="C1", linestyle="--",
                         linewidth=1.4, label="TF")
    ax_main.set_ylabel("MSE", fontsize=9)
    ax_main.set_title(f"{group} — sample {sample_idx}", fontsize=10)
    ax_main.legend(fontsize=8, loc="upper left")
    ax_main.grid(True, alpha=0.3)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    _fill_strip(ax_strip, etypes_1d)
    ax_strip.set_xlabel("Element index", fontsize=8)

    _strip_legend(fig, set(etypes_1d), anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, out / f"s{sample_idx}_mse.png")


def _physical_figure(n_cols, n_data_rows, title, figsize):
    """Figure with n_data_rows data rows + 1 thin strip row, each n_cols wide.

    Returns (fig, axes_rows, ax_strip) where axes_rows is a list of n_data_rows
    lists of n_cols axes, and ax_strip spans all columns.
    """
    hr = [3] * n_data_rows + [0.4]
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=12)
    gs = gridspec.GridSpec(n_data_rows + 1, n_cols, figure=fig,
                           height_ratios=hr, hspace=0.10, wspace=0.35)

    axes_rows = []
    for row in range(n_data_rows):
        row_axes = []
        for col in range(n_cols):
            sharex = axes_rows[0][col] if row > 0 else None
            ax = fig.add_subplot(gs[row, col], sharex=sharex)
            row_axes.append(ax)
        axes_rows.append(row_axes)

    ax_strip = fig.add_subplot(gs[n_data_rows, :], sharex=axes_rows[0][0])
    return fig, axes_rows, ax_strip


def plot_scales(scales_gt_1: np.ndarray, scales_ar_1: np.ndarray,
                etypes_1d: np.ndarray, sample_idx: int, group: str, out: Path):
    """2 data rows × 6 cols (value top, relative error bottom) + element strip."""
    T = scales_gt_1.shape[0]
    steps = np.arange(T)

    fig, axes_rows, ax_strip = _physical_figure(
        6, 2, f"Scales — {group} sample {sample_idx}", (22, 7)
    )

    for dim, label in enumerate(SCALE_LABELS):
        gt   = scales_gt_1[:, dim]
        pred = scales_ar_1[:, dim]
        rel  = (pred - gt) / (np.abs(gt) + 1e-30) * 100

        ax_val = axes_rows[0][dim]
        ax_val.plot(steps, gt,   color="steelblue", label="gt")
        ax_val.plot(steps, pred, color="tomato", linestyle="--", label="pred")
        ax_val.set_title(label, fontsize=9)
        ax_val.tick_params(labelsize=7)
        ax_val.grid(True, alpha=0.3)
        plt.setp(ax_val.get_xticklabels(), visible=False)
        if dim == 0:
            ax_val.legend(fontsize=7)
            ax_val.set_ylabel("Value", fontsize=8)

        ax_err = axes_rows[1][dim]
        ax_err.plot(steps, rel, color="k")
        ax_err.axhline(0, color="k", linewidth=0.7, linestyle="--")
        ax_err.tick_params(labelsize=7)
        ax_err.grid(True, alpha=0.3)
        plt.setp(ax_err.get_xticklabels(), visible=False)
        if dim == 0:
            ax_err.set_ylabel("Rel. error (%)", fontsize=8)

    _fill_strip(ax_strip, etypes_1d)
    ax_strip.set_xlabel("Element index", fontsize=8)

    _strip_legend(fig, set(etypes_1d), anchor=(0.5, -0.03))
    _save(fig, out / f"s{sample_idx}_scales.png")


def plot_centroids(centroids_gt_1: np.ndarray, centroids_ar_1: np.ndarray,
                   etypes_1d: np.ndarray, sample_idx: int, group: str, out: Path):
    """2 data rows × 6 cols (value top, absolute error bottom) + element strip."""
    T = centroids_gt_1.shape[0]
    steps = np.arange(T)

    fig, axes_rows, ax_strip = _physical_figure(
        6, 2, f"Centroids — {group} sample {sample_idx}", (22, 7)
    )

    for dim, label in enumerate(CENTROID_LABELS):
        gt   = centroids_gt_1[:, dim]
        pred = centroids_ar_1[:, dim]
        err  = np.abs(pred - gt)

        ax_val = axes_rows[0][dim]
        ax_val.plot(steps, gt,   color="steelblue", label="gt")
        ax_val.plot(steps, pred, color="tomato", linestyle="--", label="pred")
        ax_val.set_title(label, fontsize=9)
        ax_val.tick_params(labelsize=7)
        ax_val.grid(True, alpha=0.3)
        plt.setp(ax_val.get_xticklabels(), visible=False)
        if dim == 0:
            ax_val.legend(fontsize=7)
            ax_val.set_ylabel("Value", fontsize=8)

        ax_err = axes_rows[1][dim]
        ax_err.plot(steps, err, color="k")
        ax_err.axhline(0, color="k", linewidth=0.7, linestyle="--")
        ax_err.tick_params(labelsize=7)
        ax_err.grid(True, alpha=0.3)
        plt.setp(ax_err.get_xticklabels(), visible=False)
        if dim == 0:
            ax_err.set_ylabel("Abs. error", fontsize=8)

    _fill_strip(ax_strip, etypes_1d)
    ax_strip.set_xlabel("Element index", fontsize=8)

    _strip_legend(fig, set(etypes_1d), anchor=(0.5, -0.03))
    _save(fig, out / f"s{sample_idx}_centroids.png")


def plot_phase(maps_gt: np.ndarray, maps_pred: np.ndarray,
               sample_idx: int, group: str, out: Path):
    """6 rows × 8 cols: 3 planes × (gt, pred) × 8 evenly-spaced elements."""
    try:
        from beam_vae.data.preprocessing import PLANE_NAMES
    except ImportError:
        PLANE_NAMES = sorted([
            "x-x'", "y-y'", "z-d", "x-y", "x-z", "y-z",
            "x-y'", "x-d", "x'-y", "x'-y'", "x'-z", "x'-d",
            "y-d", "y'-z", "y'-d",
        ])

    diag_planes  = ["x-x'", "y-y'", "z-d"]
    plane_labels = ["x–x'", "y–y'", "z–δ"]
    channels     = [PLANE_NAMES.index(p) for p in diag_planes]

    T = maps_gt.shape[0]
    el_indices = np.linspace(0, T - 1, 8, dtype=int)

    fig, axes = plt.subplots(6, 8, figsize=(20, 14))
    fig.suptitle(f"Phase space — {group} sample {sample_idx}", fontsize=12)

    for plane_row, (ch, pl_label) in enumerate(zip(channels, plane_labels)):
        gt_row   = plane_row * 2
        pred_row = plane_row * 2 + 1
        for col, t in enumerate(el_indices):
            ax_gt   = axes[gt_row, col]
            ax_pred = axes[pred_row, col]
            all_vals = np.concatenate([maps_gt[t, ch].ravel(), maps_pred[t, ch].ravel()])
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
            ax_gt.imshow(maps_gt[t, ch],   origin="lower",
                         vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
            ax_pred.imshow(maps_pred[t, ch], origin="lower",
                           vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
            for ax in (ax_gt, ax_pred):
                ax.set_xticks([])
                ax.set_yticks([])
            if col == 0:
                ax_gt.set_ylabel(f"{pl_label}\ngt",   fontsize=8)
                ax_pred.set_ylabel(f"{pl_label}\npred", fontsize=8)
            if plane_row == 0:
                ax_gt.set_title(f"el {t}", fontsize=8)

    fig.tight_layout()
    _save(fig, out / f"s{sample_idx}_phase.png")


# ── Group comparison (MSE + strip, all samples in one figure) ─────────────────

def plot_group_comparison(groups: dict, mse_ar_step, mse_tf_step,
                          etypes: np.ndarray, out: Path):
    """Per-group figure: n_per_group panels (MSE + element strip) in 2 columns."""
    group_colors = {"worst": "C3", "median": "C0", "best": "C2"}

    for group_name, indices in groups.items():
        n = len(indices)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        T = mse_ar_step.shape[1]
        steps = np.arange(T)

        fig = plt.figure(figsize=(7 * ncols, 4 * nrows))
        outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)
        fig.suptitle(f"{group_name} samples — per-step MSE", fontsize=12)

        color = group_colors[group_name]
        all_etypes = set()

        for plot_i, idx in enumerate(indices):
            row, col = divmod(plot_i, ncols)
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row, col],
                height_ratios=[10, 1], hspace=0.05,
            )
            ax_main  = fig.add_subplot(inner[0])
            ax_strip = fig.add_subplot(inner[1], sharex=ax_main)

            ax_main.semilogy(steps, mse_ar_step[idx], color=color,
                             linewidth=1.6, label="AR")
            if mse_tf_step is not None:
                ax_main.semilogy(steps, mse_tf_step[idx], color=color,
                                 linestyle="--", linewidth=1.2, alpha=0.7, label="TF")
            ax_main.set_ylabel("MSE", fontsize=8)
            mean_mse = mse_ar_step[idx].mean()
            ax_main.set_title(f"sample {idx}   mean={mean_mse:.4f}", fontsize=9)
            ax_main.legend(fontsize=7, loc="upper left")
            ax_main.grid(True, alpha=0.3)
            plt.setp(ax_main.get_xticklabels(), visible=False)

            et = etypes[idx]
            _fill_strip(ax_strip, et)
            ax_strip.set_xlabel("Element index", fontsize=8)
            all_etypes.update(et)

        # Hide unused panels
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            fig.add_subplot(outer[r, c]).set_visible(False)

        _strip_legend(fig, all_etypes, anchor=(0.5, -0.02))
        _save(fig, out / f"group_{group_name}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data",         type=Path, default=None)
    p.add_argument("--output",       type=Path, default=None)
    p.add_argument("--n-per-group",  type=int,  default=5)
    p.add_argument("--batch-size",   type=int,  default=256)
    p.add_argument("--device",       type=str,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, model_name, label, config, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    output_dir = args.output or (args.checkpoint.parent / "eval_cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    data_dir = Path(args.data or config["data"]["path"])
    loader = build_val_loader(config, args.data, args.batch_size)

    print("Running inference...")
    z_gt, z_ar, z_tf, elements = run_inference(model, model_name, loader, device)
    print(f"  z_gt shape: {z_gt.shape}")

    mse_ar_step = ((z_ar - z_gt) ** 2).mean(axis=2)        # (N, T)
    mse_tf_step = ((z_tf - z_gt) ** 2).mean(axis=2) if z_tf is not None else None
    etypes = classify_elements(elements)                    # (N, T)

    print("Selecting groups...")
    worst, median, best, mse_all = select_groups(z_ar, z_gt, args.n_per_group)
    groups = {"worst": worst, "median": median, "best": best}

    plot_group_comparison(groups, mse_ar_step, mse_tf_step, etypes, output_dir)

    vae = load_vae(data_dir, device)

    for group_name, indices in groups.items():
        grp_dir = output_dir / group_name
        grp_dir.mkdir(exist_ok=True)
        print(f"\n--- {group_name} ---")

        z_gt_sel = z_gt[indices]
        z_ar_sel = z_ar[indices]

        if vae is not None:
            print(f"  Decoding {len(indices)} × 2 trajectories through VAE...")
            scales_gt,   centroids_gt,   maps_gt = decode_batch(vae, z_gt_sel,  device)
            scales_ar,   centroids_ar,   maps_ar = decode_batch(vae, z_ar_sel,  device)

        for i, idx in enumerate(indices):
            print(f"  Sample {idx}  (mean MSE={mse_all[idx]:.5f})")
            tf_1d = mse_tf_step[idx] if mse_tf_step is not None else None
            plot_mse(mse_ar_step[idx], tf_1d, etypes[idx], idx, group_name, grp_dir)

            if vae is not None:
                plot_scales(scales_gt[i], scales_ar[i], etypes[idx],
                            idx, group_name, grp_dir)
                plot_centroids(centroids_gt[i], centroids_ar[i], etypes[idx],
                               idx, group_name, grp_dir)
                plot_phase(maps_gt[i], maps_ar[i], idx, group_name, grp_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
