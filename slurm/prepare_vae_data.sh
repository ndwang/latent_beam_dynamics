#!/bin/bash
#SBATCH --job-name=lbd_vae_prep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================================
# STAGE 2.5: Convert tracked HDF5 data to VAE frequency maps
# ============================================================
# Usage:
#   sbatch slurm/prepare_vae_data.sh <input_dir> <output_base>
#
# Arguments:
#   input_dir    Raw tracked data dir (contains per-sample subdirs with beam_dump.h5)
#   output_base  Base path for output files (no extension);
#                produces <output_base>_{maps,scales,centroids,meta}.npy/.json
#
# Example:
#   sbatch slurm/prepare_vae_data.sh data/v2/sectioned_1sec_10k data/v2/vae_training/sectioned_1sec_10k
#
# Chain via job dependency:
#   jid=$(sbatch --parsable slurm/generate_and_track.sh data/v2/sectioned_1sec_10k sectioned 10000 32 1 200)
#   sbatch --dependency=afterok:$jid slurm/prepare_vae_data.sh data/v2/sectioned_1sec_10k data/v2/vae_training/sectioned_1sec_10k
# ============================================================

INPUT_DIR=${1:?Usage: sbatch prepare_vae_data.sh <input_dir> <output_base>}
OUTPUT_BASE=${2:?Missing output_base}

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate vae

echo "=== Preparing VAE training data ==="
echo "  input_dir=$INPUT_DIR"
echo "  output_base=$OUTPUT_BASE"
echo "  workers=$SLURM_CPUS_ON_NODE"

python scripts/prepare_vae_data.py \
    --data-dir "$INPUT_DIR" \
    --output "$OUTPUT_BASE" \
    --workers "$SLURM_CPUS_ON_NODE"

echo "=== Done. Outputs: ${OUTPUT_BASE}_{maps,scales,centroids,meta}.npy/.json ==="
echo "Next: train VAE in /pscratch/sd/n/ndwang/vae/ with data=data/v2_sectioned_1sec_10k.yaml"
