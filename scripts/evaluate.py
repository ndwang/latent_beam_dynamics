#!/usr/bin/env python
"""Evaluate a trained latent beam dynamics checkpoint and produce diagnostic plots.

Produces five figures:
  per_step_mse.png    — latent-space MSE vs element index (all val samples)
  scales_error_s{N}.png    — 2×6: raw gt+pred values (top) and relative error (bottom), median sample N
  centroids_error_s{N}.png — 2×6: raw gt+pred values (top) and absolute error (bottom), median sample N
  phase_space_s{N}.png     — frequency map comparison at 8 element positions, median sample N
  latent_pca.png      — predicted vs ground-truth trajectories in PCA space

Plots 2–4 require a vae_meta.json file in the data directory (written by encode_tracked.py).

Usage:
    python scripts/evaluate.py runs/<run>/lbd_best.pth
    python scripts/evaluate.py runs/<run>/lbd_best.pth --data /path/to/data --n-samples 4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models import ModelConfig, LatticeConfig, TrackingTransformer, LatticeTransformer, DualStreamTransformer
from src.data import LatentTrajectoryDataset
from src.utils.config import load_config
from src.eval import run_ar_inference, plot_mse_curve

_MODELS = {"tracking": TrackingTransformer, "lattice": LatticeTransformer, "dual_stream": DualStreamTransformer}

SCALE_LABELS = [
    r"$\sigma_x$ [m]", r"$\sigma_{x'}$ [rad]",
    r"$\sigma_y$ [m]", r"$\sigma_{y'}$ [rad]",
    r"$\sigma_z$ [m]", r"$\sigma_\delta$",
]
CENTROID_LABELS = [
    r"$\langle x \rangle$ [m]", r"$\langle x' \rangle$ [rad]",
    r"$\langle y \rangle$ [m]", r"$\langle y' \rangle$ [rad]",
    r"$\langle z \rangle$ [m]", r"$\langle \delta \rangle$",
]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a latent beam dynamics checkpoint")
    p.add_argument("checkpoint", type=Path, help="Path to .pth checkpoint file")
    p.add_argument("--data", type=Path, default=None, help="Override data path from config")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: <checkpoint_dir>/eval/)")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of samples for per-sample plots (default: 4)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ── Model & data loading ─────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device):
    run_dir = checkpoint_path.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_cfg = dict(config["model"])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "name" not in model_cfg:
        # Infer model type from checkpoint keys for pre-name checkpoints
        ckpt_keys = set(ckpt["model_state_dict"].keys())
        model_name = "lattice" if any("beam_cond" in k for k in ckpt_keys) else "tracking"
    else:
        model_name = model_cfg.pop("name")
    model_cls = _MODELS.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model.name={model_name!r}")

    if model_name == "lattice":
        cfg_cls = LatticeConfig
    else:
        cfg_cls = ModelConfig
        model_cfg.pop("output_mode", None)  # lattice-only field
    model = model_cls(cfg_cls(**model_cfg))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"train_loss={ckpt.get('train_loss', float('nan')):.6f}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
    return model, model_name, config, ckpt


def build_val_loader(config: dict, data_override, batch_size: int):
    data_path = Path(data_override or config["data"]["path"])
    training = config["training"]
    seed = training.get("seed", 42)
    val_split = training.get("val_split", 0.1)

    dataset = LatentTrajectoryDataset(data_path)
    n = len(dataset)
    val_size = int(val_split * n)
    train_size = n - val_size
    _, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Val set: {val_size} / {n} samples  (seed={seed})")
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader, data_path


# ── Sample selection ──────────────────────────────────────────────────────────

def select_samples(z_pred: np.ndarray, z_gt: np.ndarray, n: int) -> np.ndarray:
    """Pick n val indices spread across the error-percentile distribution."""
    mse = ((z_pred - z_gt) ** 2).mean(axis=(1, 2))   # (N,)
    sorted_idx = np.argsort(mse)
    pct = np.linspace(10, 90, n)
    pos = np.clip((pct / 100 * len(sorted_idx)).astype(int), 0, len(sorted_idx) - 1)
    sel = sorted_idx[pos]
    print(f"Selected samples {sel}  (percentiles {pct.astype(int)})  "
          f"MSEs={mse[sel].round(5)}")
    return sel


# ── VAE loading & decoding ────────────────────────────────────────────────────

def load_vae(data_dir: Path, device: torch.device):
    """Load VAE from vae_meta.json.  Returns (vae_model, meta) or (None, None)."""
    meta_path = data_dir / "vae_meta.json"
    if not meta_path.exists():
        print(f"Warning: {meta_path} not found — skipping physical-space plots (2–4)")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    vae_run_dir = Path(meta["vae_run_dir"])
    vae_project = vae_run_dir.parent
    if str(vae_project) not in sys.path:
        sys.path.insert(0, str(vae_project))

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
        return vae, meta

    except Exception as exc:
        print(f"Warning: could not load VAE ({exc}) — skipping physical-space plots")
        return None, None


@torch.no_grad()
def decode_trajectories(vae, z_traj: np.ndarray, device: torch.device,
                        batch_size: int = 512):
    """Decode (N, T, d_z) → scales (N,T,6), centroids (N,T,6), maps (N,T,15,H,W)."""
    N, T, d_z = z_traj.shape
    z_flat = z_traj.reshape(N * T, d_z)

    scales_list, cent_list, maps_list = [], [], []
    for start in range(0, N * T, batch_size):
        end = min(start + batch_size, N * T)
        z_b = torch.from_numpy(z_flat[start:end]).float().to(device)
        recon, sc_norm, ct_norm = vae.decode(z_b)
        scales_list.append(vae.denormalize_scales(sc_norm).cpu().numpy())
        cent_list.append(vae.denormalize_centroids(ct_norm).cpu().numpy())
        maps_list.append(recon.cpu().numpy())

    scales    = np.concatenate(scales_list).reshape(N, T, 6)
    centroids = np.concatenate(cent_list).reshape(N, T, 6)
    H, W      = maps_list[0].shape[-2:]
    maps      = np.concatenate(maps_list).reshape(N, T, 15, H, W)
    return scales, centroids, maps


# ── Plotting ──────────────────────────────────────────────────────────────────

def _sample_colors(n: int):
    return plt.cm.tab10(np.linspace(0, 0.9, n))


def plot_per_step_mse(z_gt: np.ndarray, z_ar: np.ndarray, model_name: str,
                      output_dir: Path, z_tf: np.ndarray | None = None):
    mse_ar_samples = ((z_ar - z_gt) ** 2).mean(axis=2)
    mse_tf_samples = ((z_tf - z_gt) ** 2).mean(axis=2) if z_tf is not None else None

    steps = np.arange(z_gt.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))

    plot_mse_curve(ax, steps, mse_ar_samples, color="C0", label_prefix="AR  ")
    if mse_tf_samples is not None:
        plot_mse_curve(ax, steps, mse_tf_samples, color="C1", linestyle="--",
                       label_prefix="TF  ")

    ax.set_xlabel("Element index")
    ax.set_ylabel("MSE (latent space)")
    ax.set_title(f"Per-step MSE — {model_name}, validation set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "per_step_mse.png")


def plot_scales_error(scales_pred: np.ndarray, scales_gt: np.ndarray,
                      sample_idx: int, output_dir: Path):
    """2 rows × 6 cols: raw gt+pred values (top) and relative error (bottom)."""
    T = scales_pred.shape[1]
    steps = np.arange(T)

    fig, axes = plt.subplots(2, 6, figsize=(22, 6), sharey=False)
    for dim, label in enumerate(SCALE_LABELS):
        gt   = scales_gt[0, :, dim]
        pred = scales_pred[0, :, dim]
        rel_err = (pred - gt) / gt * 100

        ax_top = axes[0, dim]
        ax_top.plot(steps, gt,   color="steelblue",  label="gt")
        ax_top.plot(steps, pred, color="tomato", linestyle="--", label="pred")
        ax_top.set_title(label, fontsize=10)
        ax_top.grid(True, alpha=0.3)
        if dim == 0:
            ax_top.legend(fontsize=8)
            ax_top.set_ylabel("Value")

        ax_bot = axes[1, dim]
        ax_bot.plot(steps, rel_err, color="k")
        ax_bot.axhline(0, color="k", linewidth=0.7, linestyle="--")
        ax_bot.set_xlabel("Element index")
        ax_bot.grid(True, alpha=0.3)
        if dim == 0:
            ax_bot.set_ylabel("Rel. error (%)")

    fig.suptitle(rf"Scales — sample {sample_idx}", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"scales_error_s{sample_idx}.png")


def plot_centroids_error(centroids_pred: np.ndarray, centroids_gt: np.ndarray,
                         sample_idx: int, output_dir: Path):
    """2 rows × 6 cols: raw gt+pred values (top) and absolute error (bottom)."""
    T = centroids_pred.shape[1]
    steps = np.arange(T)

    fig, axes = plt.subplots(2, 6, figsize=(22, 6), sharey=False)
    for dim, label in enumerate(CENTROID_LABELS):
        gt   = centroids_gt[0, :, dim]
        pred = centroids_pred[0, :, dim]
        abs_err = np.abs(pred - gt)

        ax_top = axes[0, dim]
        ax_top.plot(steps, gt,   color="steelblue",  label="gt")
        ax_top.plot(steps, pred, color="tomato", linestyle="--", label="pred")
        ax_top.set_title(label, fontsize=10)
        ax_top.grid(True, alpha=0.3)
        if dim == 0:
            ax_top.legend(fontsize=8)
            ax_top.set_ylabel("Value")

        ax_bot = axes[1, dim]
        ax_bot.plot(steps, abs_err, color="k")
        ax_bot.axhline(0, color="k", linewidth=0.7, linestyle="--")
        ax_bot.set_xlabel("Element index")
        ax_bot.grid(True, alpha=0.3)
        if dim == 0:
            ax_bot.set_ylabel("Abs. error")

    fig.suptitle(rf"Centroids — sample {sample_idx}", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / f"centroids_error_s{sample_idx}.png")


def plot_phase_space(maps_pred: np.ndarray, maps_gt: np.ndarray,
                     sample_idx: int, output_dir: Path):
    """6 rows × 8 cols: 3 diagonal planes × (gt row, pred row) × 8 elements."""
    try:
        from beam_vae.data.preprocessing import PLANE_NAMES
    except ImportError:
        # Fallback: sorted keys of PLANE_MAP
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

    for plane_row, (ch, pl_label) in enumerate(zip(channels, plane_labels)):
        gt_row   = plane_row * 2
        pred_row = plane_row * 2 + 1

        for col, t in enumerate(el_indices):
            ax_gt   = axes[gt_row, col]
            ax_pred = axes[pred_row, col]

            # Color range shared between gt and pred for this element only
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

    fig.suptitle(f"Phase space portraits — sample {sample_idx}", fontsize=12)
    fig.tight_layout()
    _save(fig, output_dir / f"phase_space_s{sample_idx}.png")


def _pca2(z_all: np.ndarray):
    """Simple 2-component PCA via SVD.  Returns (components (2,d), mean (d,), var_ratio (2,))."""
    mu = z_all.mean(axis=0)
    z_c = z_all - mu
    # Use economy SVD; limit to 2 components for speed on large arrays
    _, S, Vt = np.linalg.svd(z_c, full_matrices=False)
    components = Vt[:2]           # (2, d_z)
    var = S[:2] ** 2 / max(len(z_all) - 1, 1)
    total_var = (S ** 2).sum() / max(len(z_all) - 1, 1)
    var_ratio = var / total_var * 100
    return components, mu, var_ratio


def plot_latent_pca(z_gt: np.ndarray, z_pred: np.ndarray,
                    sample_indices: np.ndarray, output_dir: Path):
    """Predicted vs ground-truth trajectories in top-2 PCA directions.

    Fit PCA on all val z_gt, then show each selected sample in its own subplot.
    Lines colored by element index (viridis); solid = gt, dashed = pred.
    """
    N, T, d_z = z_gt.shape
    n_sel = len(sample_indices)

    # Fit PCA on all val ground-truth states
    components, mu, var_ratio = _pca2(z_gt.reshape(N * T, d_z))

    def project(z_traj):  # (T, d_z) → (T, 2)
        return (z_traj - mu) @ components.T

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=T - 1)
    t_arr = np.arange(T)

    ncols = min(n_sel, 4)
    nrows = (n_sel + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for i, idx in enumerate(sample_indices):
        ax = axes[i // ncols, i % ncols]

        gt_proj   = project(z_gt[idx])    # (T, 2)
        pred_proj = project(z_pred[idx])  # (T, 2)

        # Colored line segments
        for t in range(T - 1):
            c = cmap(norm(t))
            ax.plot(gt_proj[t:t+2, 0],   gt_proj[t:t+2, 1],
                    color=c, linewidth=1.8)
            ax.plot(pred_proj[t:t+2, 0], pred_proj[t:t+2, 1],
                    color=c, linewidth=1.8, linestyle="--", alpha=0.75)

        # Start marker
        ax.scatter(*gt_proj[0],   s=50, color="white", edgecolors="k", zorder=5)
        ax.scatter(*pred_proj[0], s=50, marker="x",    color="k",       zorder=5)

        ax.set_title(f"sample {idx}", fontsize=9)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}%)", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_sel, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Element index", shrink=0.6, pad=0.02)

    # Legend proxy
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=1.8,                  label="gt"),
        Line2D([0], [0], color="gray", linewidth=1.8, linestyle="--",  label="pred"),
    ]
    fig.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.8)

    fig.suptitle("Latent trajectories in PCA space", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / "latent_pca.png")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, model_name, config, ckpt = load_model(args.checkpoint, device)

    output_dir: Path = args.output or (args.checkpoint.parent / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    loader, data_dir = build_val_loader(config, args.data, args.batch_size)

    z_gt, z_ar, z_tf = run_ar_inference(model, model_name, loader, device)

    sel = select_samples(z_ar, z_gt, args.n_samples)

    # ── Metrics ──
    metrics: dict = {
        "checkpoint_epoch":    ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "ar_mean_mse":         float(((z_ar - z_gt) ** 2).mean()),
    }
    if z_tf is not None:
        metrics["tf_mean_mse"] = float(((z_tf - z_gt) ** 2).mean())

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")
    print(f"Saved {metrics_path}")

    # ── Plot 1: per-step MSE ──
    plot_per_step_mse(z_gt, z_ar, model_name, output_dir, z_tf)

    # ── Plot 5: latent PCA ──
    plot_latent_pca(z_gt, z_ar, sel, output_dir)

    # ── Plots 2–4: require VAE ──
    vae, _ = load_vae(data_dir, device)
    if vae is None:
        return

    print("Decoding selected samples through VAE...")
    z_gt_sel   = z_gt[sel]   # (n_sel, T, d_z)
    z_pred_sel = z_ar[sel]   # (n_sel, T, d_z)

    scales_gt,   centroids_gt,   maps_gt   = decode_trajectories(vae, z_gt_sel,   device)
    scales_pred, centroids_pred, maps_pred = decode_trajectories(vae, z_pred_sel, device)

    # Plots 2–4 use the median-error sample
    mid = len(sel) // 2
    mid_idx = sel[mid]

    # ── Plot 2: scales relative error ──
    plot_scales_error(scales_pred[mid:mid+1], scales_gt[mid:mid+1], mid_idx, output_dir)

    # ── Plot 3: centroids absolute error ──
    plot_centroids_error(centroids_pred[mid:mid+1], centroids_gt[mid:mid+1], mid_idx, output_dir)

    # ── Plot 4: phase space (median-error sample) ──
    plot_phase_space(maps_pred[mid], maps_gt[mid], mid_idx, output_dir)


if __name__ == "__main__":
    main()
