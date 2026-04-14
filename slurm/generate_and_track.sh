#!/bin/bash
#SBATCH --job-name=lbd_datagen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================================
# STAGES 1–2: Generate lattice/beam inputs + track with Tao
# ============================================================
# Stage 1 (generation) runs on the head node using all 128 local CPUs.
# Stage 2 (tracking) distributes across all 4 nodes via srun + GNU parallel,
# giving 4×128=512 parallel Tao instances.
#
# Usage:
#   sbatch slurm/generate_and_track.sh <output_dir> [mode] [n_samples] [seq_len] [n_sections] [seed]
#
# Arguments:
#   output_dir   Directory to write samples into (required)
#   mode         Lattice mode: sectioned|structured|random|fodo (default: sectioned)
#   n_samples    Number of lattice-beam pairs (default: 10000)
#   seq_len      Elements per lattice (default: 32)
#   n_sections   Sections per lattice; 1 = single-section, no cross-section mismatch (default: 1)
#   seed         RNG seed for reproducibility (default: 42)
#
# Examples:
#   sbatch slurm/generate_and_track.sh data/sectioned_1sec_10k
#   sbatch slurm/generate_and_track.sh data/sectioned_1sec_10k sectioned 10000 32 1 200
#
# Chain with encoding stage via job dependencies:
#   jid=$(sbatch --parsable slurm/generate_and_track.sh data/sectioned_1sec_10k)
#   sbatch --dependency=afterok:$jid slurm/encode_tracked.sh data/sectioned_1sec_10k data/encoded_sectioned_1sec_10k
# ============================================================

DATA_DIR=${1:?Usage: sbatch generate_and_track.sh <output_dir> [mode] [n_samples] [seq_len] [n_sections] [seed]}
MODE=${2:-sectioned}
N_SAMPLES=${3:-10000}
SEQ_LEN=${4:-32}
N_SECTIONS=${5:-1}
SEED=${6:-42}
N_PARTICLES=100000

TAO_INIT="/pscratch/sd/n/ndwang/latent_beam_dynamics/tao.init"

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd_datagen
ml load parallel

export OMP_NUM_THREADS=1

# --- Stage 1: Generate inputs (distributed across all nodes) ---

echo "=== Stage 1: Generating inputs ==="
echo "  output_dir=$DATA_DIR  mode=$MODE  n_samples=$N_SAMPLES  seq_len=$SEQ_LEN  n_sections=$N_SECTIONS  seed=$SEED"
echo "  nodes=$SLURM_NNODES  cpus_per_node=$SLURM_CPUS_ON_NODE"

# Each node generates its slice [start, end). Seeds are derived from the full
# n_samples sequence so output is identical regardless of node count.
chunk=$(( (N_SAMPLES + SLURM_NNODES - 1) / SLURM_NNODES ))

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
    NODE_ID=$SLURM_NODEID
    START=$(( NODE_ID * '"$chunk"' ))
    END=$(( START + '"$chunk"' ))
    END=$(( END > '"$N_SAMPLES"' ? '"$N_SAMPLES"' : END ))
    python scripts/generate_inputs.py \
        --mode "'"$MODE"'" \
        --n-samples '"$N_SAMPLES"' \
        --seq-len '"$SEQ_LEN"' \
        --n-sections '"$N_SECTIONS"' \
        --n-particles '"$N_PARTICLES"' \
        --output-dir "'"$DATA_DIR"'" \
        --seed '"$SEED"' \
        --start-idx "$START" \
        --end-idx "$END" \
        --workers '"$SLURM_CPUS_ON_NODE"'
'

echo "Stage 1 complete."

# --- Stage 2: Track particles with Tao (multi-node) ---

echo "=== Stage 2: Tracking with Tao ($SLURM_NNODES nodes × $SLURM_CPUS_ON_NODE CPUs) ==="

find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort | \
    srun --no-kill --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --wait=0 \
        bash /pscratch/sd/n/ndwang/latent_beam_dynamics/scripts/track_node.sh "$DATA_DIR"

echo "=== Done. Data written to $DATA_DIR ==="
echo "Next: sbatch slurm/encode_tracked.sh $DATA_DIR <encoded_output_dir>"
