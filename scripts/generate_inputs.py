"""Stage 1: Generate Bmad input files (lattice + beam) for tracking.

Usage:
    python scripts/generate_inputs.py --mode sectioned --n-samples 5000 \
        --seq-len 32 --output-dir data/sectioned

    python scripts/generate_inputs.py --mode structured --n-samples 5000 \
        --seq-len 50 --output-dir data/structured

    python scripts/generate_inputs.py --mode random --n-samples 5000 \
        --seq-len 50 --output-dir data/random
"""

import argparse
import json
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description="Generate Bmad input files")
    parser.add_argument('--mode', choices=['sectioned', 'structured', 'random', 'fodo'],
                        required=True)
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--n-particles', type=int, default=100_000)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Legacy modes use unmatched beam sampling
    legacy_samplers = {
        'structured': sample_structured_lattice,
        'random': sample_random_lattice,
        'fodo': sample_fodo_lattice,
    }

    for i in tqdm(range(args.n_samples), desc=f"Generating {args.mode}"):
        sample_dir = output_dir / f"{i:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == 'sectioned':
            # Sectioned mode: matched beam sampling
            elements, lattice_info = sample_sectioned_lattice(args.seq_len, rng)
            beam_params = sample_matched_beam_params(rng, lattice_info)

            # Save lattice info (section structure, periodic Twiss)
            with open(sample_dir / 'lattice_info.json', 'w') as f:
                json.dump(lattice_info, f, indent=2)
        else:
            # Legacy modes
            elements = legacy_samplers[args.mode](args.seq_len, rng)
            beam_params = sample_beam_params(rng)

        # Write Bmad lattice file
        write_bmad_lattice(
            elements,
            energy_GeV=beam_params['energy_GeV'],
            output_path=sample_dir / 'lattice.bmad',
        )

        # Generate and save particles
        generate_particles(
            beam_params,
            n_particles=args.n_particles,
            output_path=sample_dir / 'beam.h5',
        )

        # Save element parameters (used by Stage 3)
        np.save(sample_dir / 'elements.npy', elements)

        # Save beam params as JSON for reproducibility
        beam_params_serializable = {
            k: float(v) for k, v in beam_params.items()
        }
        with open(sample_dir / 'beam_params.json', 'w') as f:
            json.dump(beam_params_serializable, f, indent=2)

    # Save generation metadata
    metadata = {
        'mode': args.mode,
        'n_samples': args.n_samples,
        'seq_len': args.seq_len,
        'n_particles': args.n_particles,
        'seed': args.seed,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {args.n_samples} samples in {output_dir}")


if __name__ == '__main__':
    main()
