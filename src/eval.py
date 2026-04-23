"""Shared utilities for checkpoint evaluation and AR inference.

Imported by evaluate_ar.py, evaluate.py, compare_ar.py, and the analyze_* scripts.
"""

from pathlib import Path

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


def plot_mse_curve(ax, steps, mse_per_sample: np.ndarray, color,
                   linestyle: str = "-", label_prefix: str = ""):
    """Plot mean and median lines with 10–90 percentile band (AR only) on a log-scale axis."""
    mean = mse_per_sample.mean(axis=0)
    median = np.median(mse_per_sample, axis=0)
    p10, p90 = np.percentile(mse_per_sample, [10, 90], axis=0)
    ax.semilogy(steps, mean, color=color, linestyle=linestyle,
                label=f"{label_prefix}mean {mean.mean():.5f}")
    if linestyle == "-":
        ax.semilogy(steps, median, color=color, linestyle=":", linewidth=1,
                    label=f"{label_prefix}median {median.mean():.5f}")
        ax.fill_between(steps, p10, p90, color=color, alpha=0.12)


def plot_ar_mse(mse_ar_samples: np.ndarray, label: str, output_dir: Path,
                mse_tf_samples: np.ndarray | None = None):
    """Plot per-step MSE with percentile bands and save to output_dir/ar_per_step_mse.png."""
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
