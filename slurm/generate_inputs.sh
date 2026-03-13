#!/bin/bash
#SBATCH --job-name=lbd_geninput
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# STAGE 1: Generate Bmad input files
# ============================================
# Usage:
#   sbatch slurm/generate_inputs.sh structured 5000 50
#   sbatch slurm/generate_inputs.sh random 5000 50
# ============================================

MODE=${1:?Usage: sbatch generate_inputs.sh <structured|random> <n_samples> <seq_len>}
N_SAMPLES=${2:?Missing n_samples}
SEQ_LEN=${3:-50}
N_PARTICLES=${4:-100000}
SEED=${5:-42}

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd_datagen

OUTPUT_DIR="data/${MODE}"

python scripts/generate_inputs.py \
    --mode $MODE \
    --n-samples $N_SAMPLES \
    --seq-len $SEQ_LEN \
    --n-particles $N_PARTICLES \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

echo "Done. Output in $OUTPUT_DIR"
