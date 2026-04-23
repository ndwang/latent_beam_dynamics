#!/usr/bin/env python
"""Overlay AR per-step MSE curves from multiple checkpoints on one plot.

Calls src.eval.load_checkpoint, build_val_loader, and run_ar_inference for
each checkpoint, then overlays the resulting curves.  All checkpoints are run on
the same validation set (data path and seed taken from the first checkpoint's
config; override with --data).

Produces:
  ar_per_step_mse.png  — overlaid AR MSE curves (log scale)
  ar_summary.json      — per-checkpoint AR metrics table

Usage:
    python scripts/compare_ar.py runs/run_a/best.pth runs/run_b/best.pth ...
    python scripts/compare_ar.py runs/*/best.pth --data data/encoded_sectioned_10k --output eval_ar/
    python scripts/compare_ar.py runs/*/best.pth --include-tf
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

from src.eval import (load_checkpoint, build_val_loader, run_ar_inference,
                      per_sample_step_mse, per_step_mse, plot_mse_curve)


def parse_args():
    p = argparse.ArgumentParser(description="Compare checkpoints in AR inference mode")
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--data", type=Path, default=None,
                   help="Override data path (default: from first checkpoint's config)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: compare_ar/ in CWD)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--include-tf", action="store_true",
                   help="Overlay TF curves for AR-capable models (dashed, same color)")
    return p.parse_args()


def plot_comparison(results: list[dict], output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for res, color in zip(results, colors):
        steps = np.arange(res["mse_ar_samples"].shape[1])
        plot_mse_curve(ax, steps, res["mse_ar_samples"], color=color,
                       label_prefix=f"{res['label']}  ")
        if res.get("mse_tf_samples") is not None:
            plot_mse_curve(ax, steps, res["mse_tf_samples"], color=color,
                           linestyle="--", label_prefix=f"{res['label']} TF  ")

    ax.set_xlabel("Element index")
    ax.set_ylabel("MSE (latent space)")
    ax.set_title("AR per-step MSE — validation set")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "ar_per_step_mse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    output_dir = args.output or Path("compare_ar")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading checkpoints:")
    entries = []
    for ckpt_path in args.checkpoints:
        model, model_name, label, config, ckpt = load_checkpoint(ckpt_path, device)
        print(f"  {label}  epoch={ckpt.get('epoch','?')}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
        entries.append((model, model_name, label, config))

    print("\nBuilding val loader:")
    loader = build_val_loader(entries[0][3], args.data, args.batch_size)

    print("\nRunning inference:")
    results = []
    summary = {}

    for model, model_name, label, _ in entries:
        z_gt, z_ar, z_tf = run_ar_inference(model, model_name, loader, device)
        mse_ar_samples = per_sample_step_mse(z_ar, z_gt)
        mse_tf_samples = per_sample_step_mse(z_tf, z_gt) if (z_tf is not None and args.include_tf) else None

        results.append({"label": label, "mse_ar_samples": mse_ar_samples,
                         "mse_tf_samples": mse_tf_samples})

        mse_ar = mse_ar_samples.mean(axis=0)
        entry = {
            "label":        label,
            "ar_mean_mse":  float(mse_ar.mean()),
            "ar_final_mse": float(mse_ar[-1]),
        }
        if z_tf is not None:
            mse_tf = per_step_mse(z_tf, z_gt)
            entry["tf_mean_mse"]  = float(mse_tf.mean())
            entry["tf_final_mse"] = float(mse_tf[-1])
        summary[label] = entry

        print(f"  {label}: AR mean={mse_ar.mean():.5f}  final={mse_ar[-1]:.5f}")

    plot_comparison(results, output_dir)

    summary_path = output_dir / "ar_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
