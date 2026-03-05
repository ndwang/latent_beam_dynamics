#!/bin/bash
#SBATCH --job-name=lbd_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
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
# Usage: sbatch slurm/submit_single.sh
# Modify RUN_PREFIX and OVERRIDES below.
# ============================================

RUN_PREFIX="baseline"
OVERRIDES="data.path=/path/to/data_dir"

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd

RUN_NAME="${RUN_PREFIX}_$(date +%y%m%d_%H%M)"

python scripts/train.py $OVERRIDES run_name=${RUN_NAME} training.wandb.enabled=true

echo "Syncing W&B logs..."
wandb sync runs/${RUN_NAME}/wandb/offline-run-* --sync-all
