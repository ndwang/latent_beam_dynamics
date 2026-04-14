#!/bin/bash
#SBATCH --job-name=lbd_encode
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
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
# STAGE 3: Encode tracked beams with VAE
# ============================================
# Usage:
#   sbatch slurm/encode_tracked.sh data/structured data/structured_encoded
#   sbatch slurm/encode_tracked.sh data/random data/random_encoded
# ============================================

INPUT_DIR=${1:?Usage: sbatch encode_tracked.sh <input_dir> <output_dir>}
OUTPUT_DIR=${2:?Missing output_dir}
VAE_CKPT=${3:-"/pscratch/sd/n/ndwang/vae/runs/beta_1e-5_260401_1523/beta_1e-5_260401_1523_best.pth"}

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate vae

python scripts/encode_tracked.py \
    --input-dir $INPUT_DIR \
    --vae-checkpoint $VAE_CKPT \
    --output-dir $OUTPUT_DIR \
    --workers $SLURM_CPUS_ON_NODE \
    --batch-size 512

echo "Done. Output in $OUTPUT_DIR"
