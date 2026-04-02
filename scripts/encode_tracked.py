"""Stage 3: Encode tracked particle data into VAE latent vectors.

Reads Bmad output HDF5 files, encodes each beam snapshot with the VAE,
and produces the final z_traj.npy and elements.npy for training.

Usage:
    python scripts/encode_tracked.py \
        --input-dir data/sectioned_10k \
        --vae-checkpoint ../vae/runs/beta_1e-5_260401_1523/beta_1e-5_260401_1523_best.pth \
        --output-dir data/sectioned_10k_encoded
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.encode import load_vae, encode_tracked_sample


def main():
    parser = argparse.ArgumentParser(description="Encode tracked data with VAE")
    parser.add_argument('--input-dir', type=str, required=True,
                        help="Directory with sample subdirs from generate_inputs.py")
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                        help="Path to trained VAE .pth checkpoint")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to write z_traj.npy and elements.npy")
    parser.add_argument('--tracked-filename', type=str, default='beam_dump.h5',
                        help="Name of tracked HDF5 file in each sample dir")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device=args.device)
    latent_dim = vae.latent_dim

    # Find all sample directories
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / args.tracked_filename).exists()
    ])
    print(f"Found {len(sample_dirs)} tracked samples")

    if not sample_dirs:
        print("No tracked samples found. Run Bmad tracking first.")
        return

    # Determine seq_len from first sample
    first_elements = np.load(sample_dirs[0] / 'elements.npy')
    seq_len = first_elements.shape[0]

    # Encode all samples
    z_traj_list = []
    elements_list = []
    skipped = 0

    for sample_dir in tqdm(sample_dirs, desc="Encoding"):
        tracked_path = sample_dir / args.tracked_filename
        elements_path = sample_dir / 'elements.npy'

        try:
            z_traj = encode_tracked_sample(tracked_path, vae, device=args.device)
        except Exception as e:
            print(f"Skipping {sample_dir.name}: {e}")
            skipped += 1
            continue

        # z_traj should be (seq_len+1, latent_dim): initial + after each element
        if z_traj.shape != (seq_len + 1, latent_dim):
            print(
                f"Skipping {sample_dir.name}: expected z_traj shape "
                f"({seq_len + 1}, {latent_dim}), got {z_traj.shape}"
            )
            skipped += 1
            continue

        elements = np.load(elements_path)
        z_traj_list.append(z_traj)
        elements_list.append(elements)

    if not z_traj_list:
        print("No samples successfully encoded.")
        return

    # Stack and save
    z_traj_all = np.array(z_traj_list, dtype=np.float32)   # (N, seq_len+1, latent_dim)
    elements_all = np.array(elements_list, dtype=np.float32)  # (N, seq_len, 7)

    np.save(output_dir / 'z_traj.npy', z_traj_all)
    np.save(output_dir / 'elements.npy', elements_all)

    print(f"Saved {len(z_traj_list)} samples (skipped {skipped})")
    print(f"  z_traj.npy: {z_traj_all.shape}")
    print(f"  elements.npy: {elements_all.shape}")


if __name__ == '__main__':
    main()
