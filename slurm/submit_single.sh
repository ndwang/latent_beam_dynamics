#!/bin/bash
#SBATCH --job-name=lbd_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# SINGLE TRAINING RUN — LatentBeamTransformer
# ============================================
# Usage: sbatch slurm/submit_single.sh <overrides...>
# Example:
#   sbatch slurm/submit_single.sh model.d_model=512 training.epochs=500 data.path=data/encoded_sectioned_10k
# All arguments are passed directly to train.py as Hydra overrides.
# ============================================

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch slurm/submit_single.sh <overrides...>" >&2
    exit 1
fi

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd

python scripts/train.py "$@" training.wandb.enabled=true

echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
