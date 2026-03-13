#!/bin/bash
#SBATCH --job-name=lbd_encode
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# STAGE 3: Encode tracked beams with VAE
# ============================================
# Usage:
#   sbatch slurm/encode_tracked.sh data/structured data/structured_encoded
#   sbatch slurm/encode_tracked.sh data/random data/random_encoded
# ============================================

INPUT_DIR=${1:?Usage: sbatch encode_tracked.sh <input_dir> <output_dir>}
OUTPUT_DIR=${2:?Missing output_dir}
VAE_CKPT=${3:-"/pscratch/sd/n/ndwang/vae/runs/baseline_20260127/baseline_20260127_best.pth"}

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd_datagen

python scripts/encode_tracked.py \
    --input-dir $INPUT_DIR \
    --vae-checkpoint $VAE_CKPT \
    --output-dir $OUTPUT_DIR

echo "Done. Output in $OUTPUT_DIR"
