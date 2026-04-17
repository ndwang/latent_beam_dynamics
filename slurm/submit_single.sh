#!/bin/bash
#SBATCH --job-name=lbd_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# SINGLE TRAINING RUN — LatentBeamTransformer
# ============================================
# Usage: sbatch slurm/submit_single.sh <run_prefix> <sweep_group> <overrides>
# Example:
#   sbatch slurm/submit_single.sh "dual_stream_fusion_add" "scan_fusion" "model.name=dual_stream model.fusion=add data.path=data/encoded_sectioned_10k"
# ============================================

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <run_prefix> <sweep_group> <overrides>" >&2
    exit 1
fi
RUN_PREFIX="$1"
SWEEP_GROUP="$2"
OVERRIDES="$3"

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd

RUN_NAME="${RUN_PREFIX}_$(date +%y%m%d_%H%M)"

python scripts/train.py $OVERRIDES run_name=${RUN_NAME} training.wandb.enabled=true training.wandb.group=${SWEEP_GROUP}

echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
