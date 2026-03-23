# User Manual

This manual covers every command needed to generate training data and train
the latent beam dynamics model on NERSC Perlmutter.

---

## Prerequisites

Two conda environments are used:

```bash
ml load conda

# For data generation: Stages 1 (generate inputs) and 2 (Bmad tracking)
# Includes Bmad/Tao (installed from conda-forge), distgen, and openpmd-beamphysics
conda activate lbd_datagen

# For Stage 3 (VAE encoding) — uses the VAE project's environment with PyTorch
conda activate vae

# For model training
conda activate lbd
```

Stage 2 (Bmad tracking) also requires GNU Parallel:

```bash
ml load parallel
```

Ensure the `logs/` directory exists before submitting any SLURM jobs:

```bash
mkdir -p logs
```

---

## Data Generation Pipeline

The pipeline has three stages:

```
Stage 1               Stage 2                Stage 3
Generate inputs  -->  Track (Bmad/Tao)  -->  Encode (VAE)  -->  Training data
(lattice + beam)      (particles)            (latent vectors)    (.npy files)
```

### Stage 1: Generate Bmad Input Files

Generates random lattices and initial particle distributions.

**Direct:**

```bash
conda activate lbd_datagen

# Structured lattices (alternating-quad backbone, physically realistic)
python scripts/generate_inputs.py \
    --mode structured \
    --n-samples 5000 \
    --seq-len 50 \
    --n-particles 100000 \
    --output-dir data/structured \
    --seed 42

# Random lattices (independent element sampling, many will blow up)
python scripts/generate_inputs.py \
    --mode random \
    --n-samples 5000 \
    --seq-len 50 \
    --output-dir data/random
```

**Via SLURM:**

```bash
sbatch slurm/generate_inputs.sh structured 5000 50
sbatch slurm/generate_inputs.sh random 5000 50
```

SLURM positional arguments: `<mode> <n_samples> [seq_len] [n_particles] [seed]`
(seq_len defaults to 50, n_particles to 100000, seed to 42).

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | Yes | — | `structured` or `random` |
| `--n-samples` | Yes | — | Number of lattice-beam pairs to generate |
| `--seq-len` | No | 50 | Number of elements per lattice |
| `--n-particles` | No | 100000 | Particles per beam distribution |
| `--output-dir` | Yes | — | Where to write sample directories |
| `--seed` | No | 42 | Random seed for reproducibility |

**Output structure:**

```
data/structured/
├── metadata.json          # Generation parameters
├── 000000/
│   ├── lattice.bmad       # Bmad lattice definition
│   ├── beam.h5            # Initial particle distribution (HDF5)
│   ├── elements.npy       # Element parameters (seq_len, 7)
│   └── beam_params.json   # Sampled beam parameters
├── 000001/
│   └── ...
└── ...
```

**Choosing between modes:**

- **`structured`**: Alternating focusing/defocusing quad backbone with drifts
  between elements. Randomly inserts dipoles, RF cavities, and sextupoles.
  Produces physically realistic dynamics. Most samples are usable.
- **`random`**: Independent element type sampling. Produces unusual dynamics
  but 50-80% of samples will have beam blowup. These are filtered in Stage 3.

See `docs/DATA_GENERATION.md` for the physics behind all parameter ranges.


### Stage 2: Track Particles Through Bmad/Tao

Propagates particles through each lattice using the Tao beam tracking tool.
Uses GNU Parallel to run one Tao instance per sample directory, filling all
CPUs on the node.

**Direct (single sample):**

```bash
conda activate lbd_datagen

cd data/structured/000000
tao -init_file /pscratch/sd/n/ndwang/latent_beam_dynamics/tao.init \
    -noplot -lat lattice.bmad -beam_init_position_file beam.h5 <<'EOF'
set global track_type = beam
quit
EOF
```

**Direct (all samples with GNU Parallel):**

```bash
conda activate lbd_datagen
ml load parallel

export OMP_NUM_THREADS=1
find data/structured -mindepth 1 -maxdepth 1 -type d | sort | \
    parallel -j$(nproc) 'cd {} && tao -init_file /pscratch/sd/n/ndwang/latent_beam_dynamics/tao.init -noplot -lat lattice.bmad -beam_init_position_file beam.h5 <<< "set global track_type = beam
quit"'
```

**Via SLURM (recommended for large runs):**

```bash
sbatch slurm/track_beam.sh data/structured
sbatch slurm/track_beam.sh data/random
```

SLURM positional arguments: `<data_dir>`

This job requests 128 CPUs and runs all samples in parallel. Bmad/Tao is
installed via conda-forge in the `lbd_datagen` environment. The Tao
configuration is read from `tao.init` in the project root, which tells Tao to:

- Read `lattice.bmad` and `beam.h5` from each sample directory
- Track 100,000 particles through all elements
- Dump beam coordinates at every element boundary to `beam_dump.h5`

**Output:** Each sample directory gets a `beam_dump.h5` file containing
particle coordinates at every element boundary.

**Checking progress:**

```bash
# Count completed samples
find data/structured -name "beam_dump.h5" | wc -l

# Check the GNU Parallel job log
cat data/structured/tracking.log
```


### Stage 3: Encode Tracked Data with VAE

Reads tracked particle distributions, encodes each beam snapshot into a VAE
latent vector, and produces the final `.npy` training files.

**Direct:**

```bash
conda activate vae

python scripts/encode_tracked.py \
    --input-dir data/structured \
    --vae-checkpoint /pscratch/sd/n/ndwang/vae/runs/baseline_20260127/baseline_20260127_best.pth \
    --output-dir data/structured_encoded
```

**Via SLURM:**

```bash
sbatch slurm/encode_tracked.sh data/structured data/structured_encoded
sbatch slurm/encode_tracked.sh data/random data/random_encoded
```

SLURM positional arguments: `<input_dir> <output_dir> [vae_checkpoint]`
(vae_checkpoint defaults to the baseline VAE).

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | — | Directory with sample subdirs from Stage 1 |
| `--vae-checkpoint` | Yes | — | Path to trained VAE `.pth` file |
| `--output-dir` | Yes | — | Where to write final `.npy` files |
| `--tracked-filename` | No | `tracked.h5` | Name of tracked HDF5 in each sample dir |
| `--device` | No | auto | `cuda` or `cpu` (auto-detects GPU) |

**Output:**

```
data/structured_encoded/
├── z_traj.npy      # (N, seq_len+1, latent_dim) — VAE latent trajectories
└── elements.npy    # (N, seq_len, 7) — element parameters
```

Samples that fail to encode (e.g., beam blowup) are automatically skipped.
The script reports how many were skipped at the end.


### Complete Example: End to End

```bash
# 1. Generate 5000 structured lattice-beam pairs
sbatch slurm/generate_inputs.sh structured 5000 50

# 2. Track particles (submit after Stage 1 completes)
sbatch slurm/track_beam.sh data/structured

# 3. Encode with VAE (submit after Stage 2 completes)
sbatch slurm/encode_tracked.sh data/structured data/structured_encoded

# 4. Train model on the resulting data
sbatch slurm/submit_single.sh   # edit OVERRIDES in the script first
# or directly:
python scripts/train.py data.path=data/structured_encoded
```

---

## Model Training

### Default Training Run

```bash
conda activate lbd

python scripts/train.py data.path=/path/to/encoded_data
```

### Override Hyperparameters

Use dot notation to override any config value:

```bash
python scripts/train.py \
    data.path=data/structured_encoded \
    model.d_model=512 \
    model.n_layers=8 \
    model.fusion=bilinear \
    training.epochs=300 \
    training.lr=1e-4 \
    training.batch_size=64
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    data.path=data/structured_encoded \
    --resume runs/my_run/lbd_best.pth
```

### Submit to SLURM

Edit `slurm/submit_single.sh` to set `OVERRIDES`, then:

```bash
sbatch slurm/submit_single.sh
```

The script auto-generates a timestamped run name and syncs W&B logs on
completion.

### Key Training Arguments

| Override | Default | Description |
|----------|---------|-------------|
| `data.path` | — | Path to directory with `z_traj.npy` and `elements.npy` |
| `model.d_model` | 256 | Transformer hidden dimension |
| `model.n_layers` | 6 | Number of transformer layers |
| `model.fusion` | `concat` | Fusion mode: `add`, `concat`, or `bilinear` |
| `training.epochs` | 200 | Number of training epochs |
| `training.lr` | 3e-4 | Learning rate |
| `training.batch_size` | 32 | Batch size |
| `training.weight_decay` | 1e-2 | AdamW weight decay |
| `training.grad_clip` | 1.0 | Gradient clipping norm |
| `training.val_split` | 0.1 | Fraction of data for validation |
| `training.seed` | 42 | Random seed |
| `--config` / `-c` | — | Path to custom YAML config file |
| `--resume` | — | Path to checkpoint `.pth` to resume from |


### Training Output

```
runs/<run_name>/
├── config.yaml          # Snapshot of all config values
├── lbd_best.pth         # Best model checkpoint (lowest val loss)
├── lbd_epoch_050.pth    # Periodic checkpoints
├── lbd_epoch_100.pth
└── wandb/               # W&B offline logs
```

Run names are auto-generated as `lbd_d{d_model}_L{n_layers}_{YYMMDD}_{HHMM}`.

---

## Syncing W&B Logs

W&B runs in offline mode on compute nodes. Sync from a login node after
training:

```bash
./slurm/sync_wandb.sh
```

This finds and syncs all offline runs under `runs/`.

---

## Model Sanity Checks

Verify that all model variants build and run correctly:

```bash
python scripts/check_models.py
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Generate structured inputs | `sbatch slurm/generate_inputs.sh structured 5000 50` |
| Generate random inputs | `sbatch slurm/generate_inputs.sh random 5000 50` |
| Track particles | `sbatch slurm/track_beam.sh data/structured` |
| Encode with VAE | `sbatch slurm/encode_tracked.sh data/structured data/structured_encoded` |
| Train model | `python scripts/train.py data.path=data/structured_encoded` |
| Train via SLURM | `sbatch slurm/submit_single.sh` |
| Resume training | `python scripts/train.py data.path=... --resume runs/.../lbd_best.pth` |
| Sync W&B | `./slurm/sync_wandb.sh` |
| Model sanity check | `python scripts/check_models.py` |
