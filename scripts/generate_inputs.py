"""Stage 1: Generate Bmad input files (lattice + beam) for tracking.

Usage:
    python scripts/generate_inputs.py --mode sectioned --n-samples 5000 \
        --seq-len 32 --output-dir data/sectioned

    python scripts/generate_inputs.py --mode structured --n-samples 5000 \
        --seq-len 50 --output-dir data/structured

    # Control parallelism (default: all CPUs)
    python scripts/generate_inputs.py --mode sectioned --n-samples 10000 \
        --output-dir data/sectioned --workers 64
"""

import argparse
import json
import os
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.lattice import (
    sample_sectioned_lattice,
    sample_structured_lattice,
    sample_random_lattice,
    sample_fodo_lattice,
    write_bmad_lattice,
)
from src.datagen.beam import sample_beam_params, sample_matched_beam_params, generate_particles


LEGACY_SAMPLERS = {
    'structured': sample_structured_lattice,
    'random': sample_random_lattice,
    'fodo': sample_fodo_lattice,
}


def _generate_one(args_tuple):
    """Generate a single sample. Runs in a worker process.

    Takes a single tuple for compatibility with Pool.imap_unordered.
    """
    idx, output_dir, mode, seq_len, n_particles, linear, n_sections, seed_seq = args_tuple
    rng = np.random.default_rng(seed_seq)
    sample_dir = output_dir / f"{idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'sectioned':
        elements, lattice_info = sample_sectioned_lattice(seq_len, rng, linear=linear,
                                                          n_sections=n_sections)
        beam_params = sample_matched_beam_params(rng, lattice_info)
        with open(sample_dir / 'lattice_info.json', 'w') as f:
            json.dump(lattice_info, f, indent=2)
    else:
        elements = LEGACY_SAMPLERS[mode](seq_len, rng)
        beam_params = sample_beam_params(rng)

    write_bmad_lattice(
        elements,
        energy_GeV=beam_params['energy_GeV'],
        output_path=sample_dir / 'lattice.bmad',
    )
    generate_particles(
        beam_params,
        n_particles=n_particles,
        output_path=sample_dir / 'beam.h5',
    )
    np.save(sample_dir / 'elements.npy', elements)

    beam_params_serializable = {k: float(v) for k, v in beam_params.items()}
    with open(sample_dir / 'beam_params.json', 'w') as f:
        json.dump(beam_params_serializable, f, indent=2)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate Bmad input files")
    parser.add_argument('--mode', choices=['sectioned', 'structured', 'random', 'fodo'],
                        required=True)
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--n-particles', type=int, default=100_000)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear', action='store_true',
                        help="Linear optics only: no sextupoles or RF cavities")
    parser.add_argument('--n-sections', type=int, default=None,
                        help="Number of sections per lattice (sectioned mode only). "
                             "Default: random {2, 3}. Use 1 to avoid cross-section mismatch.")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel workers (default: all CPUs)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spawn independent, deterministic seeds for each sample
    parent_seq = np.random.SeedSequence(args.seed)
    child_seeds = parent_seq.spawn(args.n_samples)

    n_workers = args.workers or os.cpu_count()
    n_workers = min(n_workers, args.n_samples)

    print(f"Generating {args.n_samples} {args.mode} samples with {n_workers} workers")

    tasks = [
        (i, output_dir, args.mode, args.seq_len, args.n_particles, args.linear,
         args.n_sections, child_seeds[i])
        for i in range(args.n_samples)
    ]

    with Pool(n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_generate_one, tasks),
            total=args.n_samples,
            desc=f"Generating {args.mode}",
        ):
            pass

    print(f"Generated {args.n_samples} samples in {output_dir}")

    # Save generation metadata
    metadata = {
        'mode': args.mode,
        'linear': args.linear,
        'n_sections': args.n_sections,
        'n_samples': args.n_samples,
        'seq_len': args.seq_len,
        'n_particles': args.n_particles,
        'seed': args.seed,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
