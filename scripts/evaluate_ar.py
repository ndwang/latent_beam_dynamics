#!/usr/bin/env python
"""Evaluate a single checkpoint in autoregressive (open-loop) inference mode.

Produces:
  ar_per_step_mse.png   — per-step latent MSE vs element index (log scale)
  ar_mse_curve.npy      — shape (T,), AR MSE per step  [loadable by compare_ar.py]
  ar_tf_mse_curve.npy   — shape (T,), TF MSE per step  [Tracking/DualStream only]
  ar_metrics.json       — scalar summary: mean/final-step AR MSE (and TF if applicable)

For Tracking and DualStream models, both AR and teacher-forcing curves are shown
on the same plot (AR solid, TF dashed) so one-step vs. open-loop accuracy can be
compared directly.  For Lattice, the single forward pass is already open-loop.

Usage:
    python scripts/evaluate_ar.py runs/<run>/<run>_best.pth
    python scripts/evaluate_ar.py runs/<run>/<run>_best.pth --data data/encoded_sectioned_10k
    python scripts/evaluate_ar.py runs/<run>/<run>_best.pth --output runs/<run>/eval_ar/
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

_MODELS = {
    "tracking":    TrackingTransformer,
    "lattice":     LatticeTransformer,
    "dual_stream": DualStreamTransformer,
}
_AR_MODELS = {"tracking", "dual_stream"}


# ── Core functions (imported by compare_ar.py) ───────────────────────────────

def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model and config from a checkpoint file.

    Returns:
        model        — loaded, eval-mode model on device
        model_name   — "tracking" | "lattice" | "dual_stream"
        label        — short display label, e.g. "tracking d256"
        config       — raw config dict from config.yaml
        ckpt         — raw checkpoint dict (for epoch/loss metadata)
    """
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    if model_name == "lattice":
        cfg = LatticeConfig(**model_cfg)
    else:
        model_cfg.pop("output_mode", None)
        cfg = ModelConfig(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _MODELS[model_name](cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    d_model = config["model"].get("d_model", "?")
    label = f"{model_name} d{d_model}"
    return model, model_name, label, config, ckpt


def build_val_loader(config: dict, data_override, batch_size: int):
    """Build the validation DataLoader using the config's seed and val_split."""
    data_path = Path(data_override or config["data"]["path"])
    seed = config["training"].get("seed", 42)
    val_split = config["training"].get("val_split", 0.1)

    dataset = LatentTrajectoryDataset(data_path)
    n = len(dataset)
    val_size = int(val_split * n)
    _, val_ds = random_split(
        dataset, [n - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"  Val set: {val_size}/{n} samples  (seed={seed}, data={data_path})")
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def run_ar_inference(model, model_name: str, loader: DataLoader, device: torch.device):
    """Run open-loop AR inference (and TF for AR-capable models).

    Returns:
        z_gt   — (N, T, d_z)  ground-truth trajectories
        z_ar   — (N, T, d_z)  AR predictions
        z_tf   — (N, T, d_z)  teacher-forcing predictions, or None for Lattice
    """
    z_gt_list, z_ar_list, z_tf_list = [], [], []
    is_ar_model = model_name in _AR_MODELS

    for z0, elements, z_gt in tqdm(loader, desc="  AR inference", leave=False):
        z0 = z0.to(device)
        elements = elements.to(device)
        z_gt_d = z_gt.to(device)

        z_ar_list.append(model(z0, elements).cpu().numpy())
        if is_ar_model:
            z_tf_list.append(
                model(z0, elements, z_gt=z_gt_d, sampling_prob=0.0).cpu().numpy()
            )
        z_gt_list.append(z_gt.numpy())

    z_gt_arr = np.concatenate(z_gt_list)
    z_ar_arr  = np.concatenate(z_ar_list)
    z_tf_arr  = np.concatenate(z_tf_list) if z_tf_list else None
    return z_gt_arr, z_ar_arr, z_tf_arr


def per_sample_step_mse(z_pred: np.ndarray, z_gt: np.ndarray) -> np.ndarray:
    """MSE averaged over latent dims only. Returns shape (N, T)."""
    return ((z_pred - z_gt) ** 2).mean(axis=2)


def per_step_mse(z_pred: np.ndarray, z_gt: np.ndarray) -> np.ndarray:
    """MSE averaged over samples and latent dims. Returns shape (T,)."""
    return ((z_pred - z_gt) ** 2).mean(axis=(0, 2))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_mse_curve(ax, steps, mse_per_sample: np.ndarray, color,
                   linestyle: str = "-", label_prefix: str = ""):
    """Plot mean line with 25–75 and 10–90 percentile bands on a log-scale axis."""
    mean = mse_per_sample.mean(axis=0)
    p10, p25, p75, p90 = np.percentile(mse_per_sample, [10, 25, 75, 90], axis=0)
    ax.semilogy(steps, mean, color=color, linestyle=linestyle,
                label=f"{label_prefix}mean {mean.mean():.5f}")
    ax.fill_between(steps, p25, p75, color=color, alpha=0.25)
    ax.fill_between(steps, p10, p90, color=color, alpha=0.10)


def plot_ar_mse(mse_ar_samples: np.ndarray, label: str, output_dir: Path,
                mse_tf_samples: np.ndarray | None = None):
    """Plot per-step MSE with percentile bands.

    Args:
        mse_ar_samples: (N, T) per-sample AR MSE (mean over latent dims).
        mse_tf_samples: (N, T) per-sample TF MSE, or None.
    """
    steps = np.arange(mse_ar_samples.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))

    plot_mse_curve(ax, steps, mse_ar_samples, color="C0", label_prefix="AR  ")
    if mse_tf_samples is not None:
        plot_mse_curve(ax, steps, mse_tf_samples, color="C1", linestyle="--",
                       label_prefix="TF  ")

    ax.set_xlabel("Element index")
    ax.set_ylabel("MSE (latent space)")
    ax.set_title(f"AR per-step MSE — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "ar_per_step_mse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint in AR inference mode")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, default=None,
                   help="Override data path from config")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: <run_dir>/eval_ar/)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, model_name, label, config, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")

    output_dir = args.output or (args.checkpoint.parent / "eval_ar")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_val_loader(config, args.data, args.batch_size)
    z_gt, z_ar, z_tf = run_ar_inference(model, model_name, loader, device)

    mse_ar_samples = per_sample_step_mse(z_ar, z_gt)   # (N, T)
    mse_tf_samples = per_sample_step_mse(z_tf, z_gt) if z_tf is not None else None

    mse_ar = mse_ar_samples.mean(axis=0)
    mse_tf = mse_tf_samples.mean(axis=0) if mse_tf_samples is not None else None

    # Save mean curves for compare_ar.py
    np.save(output_dir / "ar_mse_curve.npy", mse_ar)
    if mse_tf is not None:
        np.save(output_dir / "ar_tf_mse_curve.npy", mse_tf)

    # Metrics
    metrics = {
        "label":          label,
        "checkpoint":     str(args.checkpoint),
        "epoch":          ckpt.get("epoch"),
        "val_loss":       ckpt.get("val_loss"),
        "ar_mean_mse":    float(mse_ar.mean()),
        "ar_final_mse":   float(mse_ar[-1]),
    }
    if mse_tf is not None:
        metrics["tf_mean_mse"]  = float(mse_tf.mean())
        metrics["tf_final_mse"] = float(mse_tf[-1])

    metrics_path = output_dir / "ar_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"AR mean MSE={mse_ar.mean():.5f}  final={mse_ar[-1]:.5f}")
    print(f"Saved {metrics_path}")

    plot_ar_mse(mse_ar_samples, label, output_dir, mse_tf_samples)


if __name__ == "__main__":
    main()
