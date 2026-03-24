# Latent Beam Dynamics

Latent-space causal transformer for accelerator beam dynamics, trained on NERSC Perlmutter.

## Project Structure

```
├── configs/           # YAML configuration files
│   ├── model/        # Model hyperparameters
│   ├── training/     # Training hyperparameters
│   └── data/         # Dataset paths
├── scripts/          # Entry point scripts
│   ├── generate_inputs.py  # Stage 1: generate lattice + beam inputs
│   ├── track_one.sh  # Track one sample with Tao (used by GNU parallel)
│   ├── analyze_data.py  # Post-tracking data quality diagnostics
│   ├── scan_alive.py # Quick particle survival check
│   ├── train.py      # Main training script
│   └── check_models.py  # Sanity checks for all model variants
├── slurm/            # NERSC job submission scripts
└── src/              # Source code
    ├── models/       # Model definitions (subpackage)
    │   ├── common.py     # ModelConfig, ElementEncoder, ContinuousPositionalEncoding
    │   ├── tracking.py   # TrackingTransformer (autoregressive), TrackingConfig
    │   ├── lattice.py    # LatticeTransformer (parallel/AdaLN), LatticeConfig (alias)
    │   └── losses.py     # trajectory_mse_loss
    ├── data/         # LatentTrajectoryDataset
    ├── training/     # BaseTrainer, TrackingTrainer, LatticeTrainer
    └── utils/        # Config, validation, logging, W&B
```

## Models

Two transformer architectures in `src/models/`, sharing `ElementEncoder` and `ContinuousPositionalEncoding` from `common.py`:

### TrackingTransformer (`tracking.py`)

Autoregressive GPT-style model. Each token fuses the previous beam state z_{t-1} with the element embedding h_t, then a causal transformer predicts Δz. Three forward modes: teacher forcing, scheduled sampling, and autoregressive inference.

- **Config:** `TrackingConfig` (alias for `ModelConfig`)
- **Fusion modes:** `"add"` (z_proj + h), `"concat"` (linear projection of concatenation), `"bilinear"` (linear projection of [z, h, z*h])
- Set via `config.fusion` (default: `"concat"`)

### LatticeTransformer (`lattice.py`)

Parallel (non-autoregressive) model. The initial beam state z₀ conditions all transformer layers via Adaptive Layer Norm (AdaLN). Per-element Δz predictions are accumulated with cumsum to produce the trajectory.

- **Config:** `LatticeConfig` (alias for `ModelConfig`)
- **Key components:**
  - `BeamConditioner`: maps z₀ → per-layer AdaLN parameters (gamma/beta), initialized to identity
  - `AdaLNTransformerLayer`: pre-norm attention and FFN with external gamma/beta conditioning
- Single `forward(z0, x_raw)` path for both training and inference

## Quick Commands

```bash
# Generate sectioned lattices (recommended)
python scripts/generate_inputs.py --mode sectioned --n-samples 5000 --seq-len 32 --output-dir data/sectioned

# Track particles through lattices
find data/sectioned -mindepth 1 -maxdepth 1 -type d | sort | \
    parallel -j$(nproc) bash scripts/track_one.sh {}

# Train TrackingTransformer
python scripts/train.py model.name=tracking

# Train LatticeTransformer
python scripts/train.py model.name=lattice

# Override hyperparameters
python scripts/train.py model.d_model=512 training.epochs=300

# Specify data path
python scripts/train.py data.path=/path/to/data_dir

# Resume from checkpoint
python scripts/train.py --resume runs/my_run/lbd_best.pth

# Run model sanity checks
python scripts/check_models.py

# Submit to SLURM
sbatch slurm/submit_single.sh

# Sync W&B logs from login node
./slurm/sync_wandb.sh
```

## Data

- Memory-mapped `.npy` files in a directory:
  - `z_traj.npy`: `(N, seq_len+1, latent_dim)` — VAE-encoded beam states
  - `elements.npy`: `(N, seq_len, element_dim)` — raw element parameters `[L, K1, K2, Angle, V_rf, f_rf, phi_rf]`

## Conventions

- Config overrides use dot notation: `model.d_model=512`
- Configs validated with Pydantic (`extra="forbid"` catches typos)
- Run outputs saved to `runs/<run_name>/` with config.yaml snapshot
- Run names: `lbd_d{d_model}_L{n_layers}_{YYMMDD}_{HHMM}` (auto-generated)
- SLURM logs go to `logs/` directory (must exist before submission)
- W&B runs in offline mode by default, sync from login node

## Environment

```bash
ml load conda
conda activate lbd
```
