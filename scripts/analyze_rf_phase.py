#!/usr/bin/env python
"""Correlate TF MSE at RF slots with the incoming beam phase spread.

Phase convention: phase = 2π × f_rf × time, where `time` = t − t_ref
(stored directly in HDF5). The reference particle (time=0) sees phase 0;
zero_phase = phi_rf (element setting) marks where 0 sits in the RF cycle.

Quantities per RF slot:
  zero_phase   — phi_rf from element settings
  min_phase    — 2π f_rf × min(time) over particles
  max_phase    — 2π f_rf × max(time) over particles
  phase_spread — max_phase − min_phase
  sigma_phase  — 2π f_rf × std(time)

Usage:
    python scripts/analyze_rf_phase.py runs/<run>/lbd_best.pth [--raw-data DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pmd_beamphysics.readers import particle_paths
from src.eval import load_checkpoint
from src.data import LatentTrajectoryDataset

_AR_MODELS = {"tracking", "dual_stream"}

ENCODED_DATA = Path("data/encoded_sectioned_10k")
RAW_DATA_DEFAULT = Path("data/sectioned_10k")
N_TOTAL = 10000


@torch.no_grad()
def run_tf_inference(model, model_name, loader, device):
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


def load_phase_stats(raw_data: Path, global_idx: int, rf_slots: list[int],
                     f_rf_ghz: np.ndarray) -> dict[int, tuple]:
    """Read HDF5 for one sample; return phase stats for each RF slot.

    Returns dict: slot_index → (min_phase, max_phase, sigma_phase) in radians.
    """
    hdf5_path = raw_data / f"{global_idx:06d}" / "beam_dump.h5"
    result = {}
    try:
        with h5py.File(hdf5_path, "r") as f:
            pp = particle_paths(f)
            for t in rf_slots:
                if t >= len(pp):
                    continue
                grp = f[pp[t] + "electron"]
                rel_time = grp["time"][:]          # t − t_ref, seconds
                omega = 2.0 * np.pi * f_rf_ghz[t] * 1e9  # rad/s
                phases = omega * rel_time          # rad
                result[t] = (phases.min(), phases.max(), phases.std())
    except Exception as e:
        print(f"  Warning: could not read {hdf5_path}: {e}")
    return result


def analyze_one(ckpt_path: Path, device: torch.device, raw_data: Path):
    model, model_name, label, config, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded: {label}  epoch={ckpt.get('epoch', '?')}")

    dataset = LatentTrajectoryDataset(str(ENCODED_DATA))
    val_size = int(0.1 * N_TOTAL)
    _, val_ds = random_split(dataset, [N_TOTAL - val_size, val_size],
                             generator=torch.Generator().manual_seed(42))
    val_indices = np.array(val_ds.indices)
    print(f"Val set: {len(val_indices)} samples")

    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
    z_gt, z_tf, elems = run_tf_inference(model, model_name, loader, device)

    mse_tf = ((z_tf - z_gt) ** 2).mean(axis=2)   # (N_val, T)
    rf_mask = np.abs(elems[..., 4]) > 1e-6        # (N_val, T)

    # Collect phase stats for all RF slots
    # Group RF slots by val sample index to open each HDF5 once
    from collections import defaultdict
    slots_by_sample = defaultdict(list)
    n_idx, t_idx = np.where(rf_mask)
    for n, t in zip(n_idx, t_idx):
        slots_by_sample[n].append(t)

    min_phases, max_phases, sigma_phases, zero_phases, log_mses = [], [], [], [], []

    print("  Loading HDF5 phase data...")
    for n in tqdm(sorted(slots_by_sample), desc="  Samples", leave=False):
        global_idx = int(val_indices[n])
        rf_slots = slots_by_sample[n]
        f_rf_ghz = elems[n, :, 5]   # GHz per element slot
        stats = load_phase_stats(raw_data, global_idx, rf_slots, f_rf_ghz)
        for t in rf_slots:
            if t not in stats:
                continue
            mn, mx, sig = stats[t]
            min_phases.append(mn)
            max_phases.append(mx)
            sigma_phases.append(sig)
            zero_phases.append(float(elems[n, t, 6]))   # phi_rf
            log_mses.append(np.log10(mse_tf[n, t] + 1e-9))

    min_phases  = np.array(min_phases)
    max_phases  = np.array(max_phases)
    sigma_phases = np.array(sigma_phases)
    zero_phases  = np.array(zero_phases)
    phase_spread = max_phases - min_phases
    log_mse      = np.array(log_mses)

    print(f"  RF slots with phase data: {len(log_mse)}")
    for name, vals in [
        ("zero_phase",   zero_phases),
        ("min_phase",    min_phases),
        ("max_phase",    max_phases),
        ("phase_spread", phase_spread),
        ("sigma_phase",  sigma_phases),
    ]:
        r = np.corrcoef(vals, log_mse)[0, 1]
        print(f"  corr(log10 TF MSE, {name}) = {r:.4f}")

    out = ckpt_path.parent / "eval_ar_outliers"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    pairs = [
        (zero_phases,   r"$\phi_{rf}$ (element setting, rad)",  axes[0, 0]),
        (min_phases,    r"$\phi_{min}$ (rad)",                  axes[0, 1]),
        (max_phases,    r"$\phi_{max}$ (rad)",                  axes[0, 2]),
        (phase_spread,  r"Phase spread $\phi_{max}-\phi_{min}$ (rad)", axes[1, 0]),
        (sigma_phases,  r"$\sigma_\phi$ (rad)",                 axes[1, 1]),
    ]
    for x, xlabel, ax in pairs:
        r = np.corrcoef(x, log_mse)[0, 1]
        ax.scatter(x, log_mse, s=5, alpha=0.3, color="C0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log10 TF MSE at RF")
        ax.set_title(f"r = {r:.3f}")
        ax.grid(True, alpha=0.3)
    axes[1, 2].axis("off")

    fig.suptitle(f"Incoming beam RF phase vs TF MSE — {label}", fontsize=12)
    fig.tight_layout()
    path = out / "rf_phase.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--raw-data", type=Path, default=RAW_DATA_DEFAULT)
    p.add_argument("--device",   type=str,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    for ckpt in args.checkpoints:
        analyze_one(ckpt, device, args.raw_data)


if __name__ == "__main__":
    main()
