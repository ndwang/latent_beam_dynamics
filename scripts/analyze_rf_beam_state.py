#!/usr/bin/env python
"""Correlate TF MSE at RF slots with the incoming longitudinal beam state.

Uses pre-computed scales/centroids from the VAE training data directly
(no VAE decode needed). Checks sigma_z, sigma_delta, |centroid_z|, |centroid_delta|.

Usage:
    python scripts/analyze_rf_beam_state.py runs/<run>/lbd_best.pth [runs/<run2>/...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.eval import load_checkpoint, build_val_loader
from src.data import LatentTrajectoryDataset

_AR_MODELS = {"tracking", "dual_stream"}

SCALES_PATH    = Path("data/vae_training/sectioned_10k_scales.npy")
CENTROIDS_PATH = Path("data/vae_training/sectioned_10k_centroids.npy")
N_TOTAL, T_PLUS1 = 10000, 33


@torch.no_grad()
def run_tf_inference(model, model_name, loader, device):
    """Run only TF inference, also collecting elements."""
    z_gt_list, z_tf_list, elem_list = [], [], []
    is_ar = model_name in _AR_MODELS
    for z0, elements, z_gt in tqdm(loader, desc="  TF inference", leave=False):
        if is_ar:
            z_tf_list.append(
                model(z0.to(device), elements.to(device),
                      z_gt=z_gt.to(device), sampling_prob=0.0).cpu().numpy()
            )
        else:
            z_tf_list.append(model(z0.to(device), elements.to(device)).cpu().numpy())
        z_gt_list.append(z_gt.numpy())
        elem_list.append(elements.numpy())
    return (
        np.concatenate(z_gt_list),
        np.concatenate(z_tf_list),
        np.concatenate(elem_list),
    )


def analyze_one(ckpt_path: Path, device: torch.device,
                scales_all: np.ndarray, cents_all: np.ndarray,
                val_indices: np.ndarray, val_ds):
    model, model_name, label, config, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch','?')}")

    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
    z_gt, z_tf, elems = run_tf_inference(model, model_name, loader, device)

    mse_tf = ((z_tf - z_gt) ** 2).mean(axis=2)   # (N_val, T)
    rf_mask = np.abs(elems[..., 4]) > 1e-6        # (N_val, T)
    n_idx, t_idx = np.where(rf_mask)

    # Incoming state at element t is snapshot index t (0 = initial beam)
    scales_val = scales_all[val_indices]           # (N_val, T+1, 6)
    cents_val  = cents_all[val_indices]

    sigma_z     = scales_val[n_idx, t_idx, 4]
    sigma_delta = scales_val[n_idx, t_idx, 5]
    cent_z      = cents_val[n_idx, t_idx, 4]
    cent_delta  = cents_val[n_idx, t_idx, 5]
    mse_rf      = mse_tf[rf_mask]
    log_mse     = np.log10(mse_rf + 1e-9)

    print(f"  RF slots: {rf_mask.sum()}")
    for name, vals in [
        ("sigma_z",     sigma_z),
        ("sigma_delta", sigma_delta),
        ("|cent_z|",    np.abs(cent_z)),
        ("|cent_delta|",np.abs(cent_delta)),
    ]:
        r = np.corrcoef(vals, log_mse)[0, 1]
        print(f"  corr(log10 TF MSE, {name}) = {r:.4f}")

    out = ckpt_path.parent / "eval_ar_outliers"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pairs = [
        (sigma_z,             r"$\sigma_z$ (incoming)",       axes[0, 0]),
        (sigma_delta,         r"$\sigma_\delta$ (incoming)",  axes[0, 1]),
        (np.abs(cent_z),      r"$|c_z|$ (incoming)",          axes[1, 0]),
        (np.abs(cent_delta),  r"$|c_\delta|$ (incoming)",     axes[1, 1]),
    ]
    for x, xlabel, ax in pairs:
        ax.scatter(x, log_mse, s=5, alpha=0.3, color="C0")
        r = np.corrcoef(x, log_mse)[0, 1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log10 TF MSE at RF")
        ax.set_title(f"r = {r:.3f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Incoming longitudinal state vs TF MSE at RF — {label}", fontsize=12)
    fig.tight_layout()
    path = out / "rf_beam_state.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    scales_all = np.load(SCALES_PATH).reshape(N_TOTAL, T_PLUS1, 6)
    cents_all  = np.load(CENTROIDS_PATH).reshape(N_TOTAL, T_PLUS1, 6)

    dataset  = LatentTrajectoryDataset("data/encoded_sectioned_10k")
    val_size = int(0.1 * N_TOTAL)
    _, val_ds = random_split(dataset, [N_TOTAL - val_size, val_size],
                             generator=torch.Generator().manual_seed(42))
    val_indices = np.array(val_ds.indices)
    print(f"Val set: {len(val_indices)} samples")

    for ckpt in args.checkpoints:
        analyze_one(ckpt, device, scales_all, cents_all, val_indices, val_ds)


if __name__ == "__main__":
    main()
