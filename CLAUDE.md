# Latent Beam Dynamics

Latent-space causal transformer for accelerator beam dynamics, trained on NERSC Perlmutter.

## Project Structure

```
├── configs/           # YAML configuration files
│   ├── model/        # Model hyperparameters
│   ├── training/     # Training hyperparameters
│   └── data/         # Dataset paths
├── scripts/          # Entry point scripts (see Scripts Reference section below)
├── slurm/            # NERSC job submission scripts
└── src/              # Source code
    ├── models/       # Model definitions (subpackage)
    │   ├── common.py       # ModelConfig, ElementEncoder, ContinuousPositionalEncoding
    │   ├── tracking.py     # TrackingTransformer (autoregressive), TrackingConfig
    │   ├── lattice.py      # LatticeTransformer (parallel/AdaLN), LatticeConfig (alias)
    │   ├── dual_stream.py  # DualStreamTransformer (two-stream causal), DualStreamConfig
    │   └── losses.py       # trajectory_mse_loss
    ├── eval.py       # Shared eval utilities: load_checkpoint, build_val_loader,
    │                 #   run_ar_inference, per_step_mse, per_sample_step_mse,
    │                 #   plot_mse_curve, plot_ar_mse
    ├── data/         # LatentTrajectoryDataset
    ├── training/     # BaseTrainer, TrackingTrainer, LatticeTrainer
    └── utils/        # Config, validation, logging, W&B
```

## Models

Three transformer architectures in `src/models/`, sharing `ElementEncoder` and `ContinuousPositionalEncoding` from `common.py`:

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

### DualStreamTransformer (`dual_stream.py`)

Two-stream causal model. Element tokens and beam tokens are kept in separate sequences throughout; cross-attention is the only bridge. Beam tokens come from projecting `z_{t-1}` into `d_model`; element tokens come from `ElementEncoder` (computed once in parallel). Each layer applies: (1) causal self-attention over beam tokens, (2) causal cross-attention (beam as Q, elements as K/V), (3) FFN. Δz is predicted and added to `z_{t-1}`.

- **Config:** `DualStreamConfig` (alias for `ModelConfig`)
- **Trainer:** `DualStreamTrainer` (alias for `TrackingTrainer`)
- **No beam positional encoding:** VAE latent already encodes `s`-position; element stream retains Fourier PE
- **Same three forward modes as TrackingTransformer:** teacher forcing, scheduled sampling, autoregressive
- Full design rationale in `docs/MODEL_DESIGN.md` §3c

## Quick Commands

```bash
# Data versioning: data/v1/ = legacy (old encoding + generation); data/v2/ = current.
# DATA_VERSION in generate_inputs.py and encode_tracked.py is written to metadata.json.
# Bump both constants when generation or encoding logic changes, and generate to data/vN/.

# Generate + track on a single 128-CPU SLURM node (recommended)
# Args: <output_dir> [mode] [n_samples] [seq_len] [n_sections] [seed]
# n_sections=1 (default) uses single-section lattices — avoids cross-section mismatch bias
sbatch slurm/generate_and_track.sh data/v2/sectioned_1sec_10k
sbatch slurm/generate_and_track.sh data/v2/sectioned_1sec_10k sectioned 10000 32 1 200

# Chain with encoding via job dependency (submit all at once, run sequentially):
jid=$(sbatch --parsable slurm/generate_and_track.sh data/v2/sectioned_1sec_10k sectioned 10000 32 1 200)
sbatch --dependency=afterok:$jid slurm/encode_tracked.sh data/v2/sectioned_1sec_10k data/v2/encoded_sectioned_1sec_10k

# Interactive generation + tracking (lbd_datagen env, for testing):
python scripts/generate_inputs.py --mode sectioned --n-samples 10000 --seq-len 32 --n-sections 1 --output-dir data/v2/sectioned_1sec_10k --seed 200
find data/v2/sectioned_1sec_10k -mindepth 1 -maxdepth 1 -type d | sort | \
    parallel -j$(nproc) bash scripts/track_one.sh {}

# Train TrackingTransformer
python scripts/train.py model.name=tracking

# Train LatticeTransformer
python scripts/train.py model.name=lattice

# Train DualStreamTransformer
python scripts/train.py model.name=dual_stream

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
# Plots 2–4 require vae_meta.json in the data dir (written by encode_tracked.py).
# --n-samples controls how many val samples appear in per-sample plots (default 4).

# Run model sanity checks
python scripts/check_models.py

# Submit a single training run to SLURM
# Args: <run_prefix> <sweep_group> <overrides>
sbatch slurm/submit_single.sh "tracking_d512" "scan_dmodel" "model.d_model=512 training.epochs=500 data.path=data/encoded_sectioned_10k"

# Submit a 1D hyperparameter scan to SLURM (4 parallel GPU jobs)
# Args: <param_name> <space-separated values> <fixed overrides> <sweep group>
sbatch slurm/submit_1d_scan.sh "model.n_layers" "1 2 3 4" "model.d_model=512 data.path=data/encoded_sectioned_10k training.epochs=500" "scan2_nlayers"

# Sync W&B logs from login node
bash scripts/sync_wandb.sh
```

## Data

- Memory-mapped `.npy` files in a directory:
  - `z_traj.npy`: `(N, seq_len+1, latent_dim)` — VAE-encoded beam states
  - `elements.npy`: `(N, seq_len, element_dim)` — raw element parameters `[L, K1, K2, Angle, V_rf, f_rf, phi_rf]`

### Data versioning

Datasets live under `data/vN/` where `N` matches `DATA_VERSION` in `generate_inputs.py` and `encode_tracked.py`. Each `metadata.json` / `vae_meta.json` records `data_version` for provenance. Bump both constants together when generation or encoding logic changes.

| Version | Directory | What changed |
|---------|-----------|--------------|
| v1 | `data/v1/` | Legacy: absolute `pg.t` for z, no RF constraints, min energy 0.1 GeV |
| v2 | `data/v2/` | Current: t−t_ref from HDF5, RF quarter-wavelength constraint, min energy 1 GeV |

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

## Scripts Reference

Scripts are grouped by pipeline stage. All Python scripts use `sys.path.insert` to find `src/` relative to the repo root, so they must be run from the repo root.

### Data pipeline (lbd_datagen env)

| Script | What it does | Key args |
|---|---|---|
| `generate_inputs.py` | Stage 1: generate Bmad lattice + beam inputs | `--mode sectioned --n-samples N --seq-len 32 --output-dir DIR` |
| `track_one.sh` | Stage 2: track one sample dir with Tao (called by GNU parallel) | positional: sample dir |
| `track_node.sh` | Stage 2 parallel driver for multi-node SLURM jobs | `<joblog_dir>` on stdin |
| `scan_alive.py` | Quick check: histogram of surviving particles at last element | `--data-dir DIR` |
| `analyze_data.py` | Post-tracking diagnostics: survival, beam size, growth factors | `--data-dir DIR [--output-dir DIR]` |

### VAE encoding (vae env)

| Script | What it does | Key args |
|---|---|---|
| `prepare_vae_data.py` | Convert beam_dump.h5 snapshots to 15-channel frequency maps for VAE training | `--data-dir DIR --output DIR --workers N` |
| `encode_tracked.py` | Stage 3: VAE-encode tracked data → `z_traj.npy` + `elements.npy` | `--input-dir DIR --vae-checkpoint PTH --output-dir DIR` |

### Training (lbd env)

| Script | What it does | Key args |
|---|---|---|
| `train.py` | Train any model variant | `model.name=tracking|lattice|dual_stream`, overrides |
| `check_models.py` | Sanity-check all three model architectures (forward pass, shapes, loss) | none |

### Checkpoint evaluation (lbd env)

All evaluation scripts auto-detect the model config from `<run_dir>/config.yaml`.

| Script | What it does | Outputs |
|---|---|---|
| `evaluate.py` | Full eval: MSE, scale/centroid errors, phase-space maps, PCA | `eval/per_step_mse.png`, `scales_error_s*.png`, `centroids_error_s*.png`, `phase_space_s*.png`, `latent_pca.png`, `metrics.json` |
| `evaluate_ar.py` | AR-only eval for a single checkpoint | `eval_ar/ar_per_step_mse.png`, `ar_mse_curve.npy`, `ar_tf_mse_curve.npy`, `ar_metrics.json` |
| `compare_ar.py` | Overlay AR MSE curves from multiple checkpoints on one plot | `ar_per_step_mse.png`, `ar_summary.json` |
| `compare_runs.py` | Summarise training runs from a scan (loss table, convergence, overfitting, config diff) | printed table / plots |

```bash
python scripts/evaluate_ar.py runs/<run>/lbd_best.pth [--data DIR] [--output DIR]
python scripts/compare_ar.py runs/*/lbd_best.pth [--data DIR] [--output DIR] [--include-tf]
python scripts/compare_runs.py runs/scan_* [--all | --convergence | --overfitting | --config-diff]
```

### Deep error analysis (lbd env)

These scripts dig into *why* AR error is high. All take one or more checkpoint paths as positional arguments and write plots to `<run_dir>/analysis/` by default.

| Script | Question answered | Key outputs |
|---|---|---|
| `analyze_ar_outliers.py` | Which element types drive error growth in the worst samples? | `mse_distribution.png`, `delta_mse_by_element.png` (violin), `outlier_trajectories.png` |
| `analyze_trajectory_cases.py` | What do worst / median / best trajectories look like end-to-end? | `worst/`, `median/`, `best/` subdirs with per-sample MSE, scales, centroids, phase-space plots; `group_mse_comparison.png` |
| `analyze_rf_regime.py` | Which RF parameter values (V_rf, f_rf, phi_rf) cause high error? | scatter/violin plots of AR & TF MSE vs each RF parameter |
| `analyze_rf_beam_state.py` | Does the incoming beam state (σ_z, σ_δ, centroid) predict RF error? | scatter plots of TF MSE at RF slots vs beam state variables |
| `analyze_rf_phase.py` | Does incoming beam phase spread (cycles of RF spanned) predict RF error? Reads raw HDF5 `time` field directly. | scatter plots of TF MSE vs zero/min/max phase and phase_spread; `--raw-data DIR` |

```bash
python scripts/analyze_ar_outliers.py runs/<run>/lbd_best.pth [--data DIR] [--top-k 10]
python scripts/analyze_trajectory_cases.py runs/<run>/lbd_best.pth [--data DIR] [--n-per-group 5]
python scripts/analyze_rf_regime.py runs/<run>/lbd_best.pth [runs/<run2>/lbd_best.pth ...]
python scripts/analyze_rf_beam_state.py runs/<run>/lbd_best.pth [runs/<run2>/lbd_best.pth ...]
python scripts/analyze_rf_phase.py runs/<run>/lbd_best.pth [--raw-data data/sectioned_10k]
```

Note: `analyze_rf_beam_state.py` also reads `data/vae_training/sectioned_10k_scales.npy` and `sectioned_10k_centroids.npy` (hardcoded paths). `analyze_rf_phase.py` reads raw HDF5 files directly; phase convention: `phase = 2π f_rf × time` where `time` = t − t_ref from HDF5 (reference particle sees phase 0).

### Data diagnostics (lbd env)

| Script | What it does | Key args |
|---|---|---|
| `analyze_z_evolution.py` | PCA scatter at selected elements, per-PC variance vs index, per-sample step-size heatmap | `--data DIR [--output DIR] [--n-samples 3000]` |

### Infra

| Script | What it does |
|---|---|
| `sync_wandb.sh` | Sync all offline W&B runs under `runs/` to the cloud |
