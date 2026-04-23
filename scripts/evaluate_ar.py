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
import numpy as np
import torch

from src.eval import (load_checkpoint, build_val_loader, run_ar_inference,
                      per_sample_step_mse, per_step_mse, plot_mse_curve, plot_ar_mse)


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

    np.save(output_dir / "ar_mse_curve.npy", mse_ar)
    if mse_tf is not None:
        np.save(output_dir / "ar_tf_mse_curve.npy", mse_tf)

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
