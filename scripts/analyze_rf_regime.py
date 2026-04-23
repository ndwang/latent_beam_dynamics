#!/usr/bin/env python
"""Analyze which RF parameter regimes cause high TF/AR error.

Usage:
    python scripts/analyze_rf_regime.py runs/<run>/lbd_best.pth [runs/<run2>/lbd_best.pth ...]
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
from tqdm import tqdm

from src.eval import load_checkpoint, build_val_loader

_AR_MODELS = {"tracking", "dual_stream"}


@torch.no_grad()
def run_inference(model, model_name, loader, device):
    z_gt_list, z_ar_list, z_tf_list, elem_list = [], [], [], []
    is_ar = model_name in _AR_MODELS
    for z0, elements, z_gt in tqdm(loader, desc="  inference", leave=False):
        z_ar_list.append(model(z0.to(device), elements.to(device)).cpu().numpy())
        if is_ar:
            z_tf_list.append(
                model(z0.to(device), elements.to(device),
                      z_gt=z_gt.to(device), sampling_prob=0.0).cpu().numpy()
            )
        z_gt_list.append(z_gt.numpy())
        elem_list.append(elements.numpy())
    return (
        np.concatenate(z_gt_list),
        np.concatenate(z_ar_list),
        np.concatenate(z_tf_list) if z_tf_list else None,
        np.concatenate(elem_list),
    )


def analyze_one(ckpt_path: Path, device: torch.device, data_override):
    model, model_name, label, config, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch','?')}")

    loader = build_val_loader(config, data_override, 256)
    z_gt, z_ar, z_tf, elems = run_inference(model, model_name, loader, device)

    mse_ar = ((z_ar - z_gt) ** 2).mean(axis=2)                           # (N, T)
    mse_tf = ((z_tf - z_gt) ** 2).mean(axis=2) if z_tf is not None else None

    rf_mask = np.abs(elems[..., 4]) > 1e-6                               # (N, T)
    V_rf    = elems[..., 4][rf_mask]
    phi_rf  = elems[..., 6][rf_mask]
    f_rf    = elems[..., 5][rf_mask]
    mse_ar_rf = mse_ar[rf_mask]
    mse_tf_rf = mse_tf[rf_mask] if mse_tf is not None else None

    print(f"  RF slots: {rf_mask.sum()}")
    print(f"  V_rf   : [{V_rf.min():.3f}, {V_rf.max():.3f}]  median={np.median(V_rf):.3f}")
    print(f"  phi_rf : [{phi_rf.min():.3f}, {phi_rf.max():.3f}]  median={np.median(phi_rf):.3f}")
    print(f"  f_rf unique: {sorted(np.unique(f_rf.round(3)))}")
    for name, mse_rf in [("AR", mse_ar_rf), ("TF", mse_tf_rf)]:
        if mse_rf is None:
            continue
        print(f"  {name} MSE at RF: mean={mse_rf.mean():.5f}  "
              f"median={np.median(mse_rf):.5f}  "
              f"p90={np.percentile(mse_rf,90):.5f}  "
              f"p99={np.percentile(mse_rf,99):.5f}")

    # Plot
    out = ckpt_path.parent / "eval_ar_outliers"
    out.mkdir(exist_ok=True)

    mse_plot = mse_tf_rf if mse_tf_rf is not None else mse_ar_rf
    mode_label = "TF" if mse_tf_rf is not None else "AR"
    log_mse = np.log10(mse_plot + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sc = axes[0].scatter(V_rf, phi_rf, c=log_mse, cmap="hot", s=8, alpha=0.6)
    axes[0].set_xlabel("V_rf")
    axes[0].set_ylabel("phi_rf")
    axes[0].set_title(f"V_rf vs phi_rf  (color = log10 {mode_label} MSE)")
    fig.colorbar(sc, ax=axes[0], label=f"log10 {mode_label} MSE")

    axes[1].scatter(V_rf, log_mse, s=6, alpha=0.4, color="C0")
    axes[1].set_xlabel("V_rf")
    axes[1].set_ylabel(f"log10 {mode_label} MSE")
    axes[1].set_title("V_rf vs MSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(phi_rf, log_mse, s=6, alpha=0.4, color="C1")
    axes[2].set_xlabel("phi_rf")
    axes[2].set_ylabel(f"log10 {mode_label} MSE")
    axes[2].set_title("phi_rf vs MSE")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"RF parameter regime — {label}", fontsize=12)
    fig.tight_layout()
    path = out / "rf_param_regime.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--data",   type=Path, default=None)
    p.add_argument("--device", type=str,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    for ckpt in args.checkpoints:
        analyze_one(ckpt, device, args.data)


if __name__ == "__main__":
    main()
