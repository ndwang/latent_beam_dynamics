# Latent Beam Dynamics

Latent-space causal transformer for accelerator beam dynamics, trained on NERSC Perlmutter.

## Project Structure

```
├── configs/           # YAML configuration files
│   ├── model/        # Model hyperparameters
│   ├── training/     # Training hyperparameters
│   └── data/         # Dataset paths
├── scripts/          # Entry point scripts
│   └── train.py      # Main training script
├── slurm/            # NERSC job submission scripts
└── src/              # Source code
    ├── model.py      # LatentBeamTransformer, ModelConfig
    ├── data/         # LatentTrajectoryDataset
    ├── training/     # Trainer, losses
    └── utils/        # Config, validation, logging, W&B
```

## Quick Commands

```bash
# Train with defaults
python scripts/train.py

# Override hyperparameters
python scripts/train.py model.d_model=512 training.epochs=300

# Specify data path
python scripts/train.py data.path=/path/to/data_dir

# Resume from checkpoint
python scripts/train.py --resume runs/my_run/lbd_best.pth

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
