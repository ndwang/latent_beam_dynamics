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
│   ├── evaluate.py   # Post-training checkpoint evaluation + plots
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

# Compare runs from a scan — run this first to identify which checkpoints to evaluate
python scripts/compare_runs.py runs/d_model_scan_*             # summary table (best val_loss, epoch)
python scripts/compare_runs.py runs/d_model_scan_* --all       # all analyses: convergence, overfitting, trajectory, config-diff
python scripts/compare_runs.py runs/d_model_scan_* --config-diff  # which params vary across runs

# Evaluate a checkpoint (runs in lbd env; auto-detects config from run dir)
python scripts/evaluate.py runs/<run>/lbd_best.pth
# Override data path or output dir:
python scripts/evaluate.py runs/<run>/lbd_best.pth --data /path/to/data --output /path/to/out
# Outputs go to runs/<run>/eval/ by default:
#   per_step_mse.png    — latent MSE vs element index (AR + teacher-forcing for Tracking)
#   scales_error.png    — relative scale error (pred−gt)/gt per dimension, selected samples
#   centroids_error.png — absolute centroid error |pred−gt| per dimension, selected samples
#   phase_space.png     — x-x'/y-y'/z-δ frequency maps at 8 elements (gt row vs pred row)
#   latent_pca.png      — predicted vs gt trajectory in top-2 PCA directions
#   metrics.json        — scalar MSE summary
# Plots 2–4 require vae_meta.json in the data dir (written by encode_latent.py).
# --n-samples controls how many val samples appear in per-sample plots (default 4).

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

## Experiment Tracking

`EXPERIMENTS.md` is the running log of all training runs and their conclusions. **Always update it:**
- **Before** launching a new scan: write a prose section explaining the motivation, what question is being asked, what you expect to see and why, and what the result would imply either way
- **After** runs complete: record the results and write a detailed analysis — what the numbers mean, whether the outcome matched expectations, what it implies about the model or training dynamics, and what to try next

## Conventions

- Config overrides use dot notation: `model.d_model=512`
- Configs validated with Pydantic (`extra="forbid"` catches typos)
- Run outputs saved to `runs/<run_name>/` with config.yaml snapshot
- Run names: `lbd_d{d_model}_L{n_layers}_{YYMMDD}_{HHMM}` (auto-generated)
- SLURM logs go to `logs/` directory (must exist before submission)
- W&B runs in offline mode by default, sync from login node

## Environment

Three conda environments for different pipeline stages (all on Perlmutter):

```bash
ml load conda

conda activate lbd_datagen   # Stage 1-2: lattice generation + Tao tracking
conda activate vae            # Stage 3: prepare_vae_data.py (beam_vae preprocessing)
conda activate lbd            # Stage 4: transformer training (PyTorch + CUDA)
```

- **lbd_datagen**: NumPy, distgen, pmd_beamphysics, Bmad/Tao — no PyTorch
- **vae**: beam_vae package (frequency maps, VAE encode) + PyTorch
- **lbd**: PyTorch, model training, evaluation
