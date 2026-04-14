#!/bin/bash
#SBATCH --job-name=lbd_datagen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================================
# STAGES 1–2: Generate lattice/beam inputs + track with Tao
# ============================================================
# Both stages run on the same 128-CPU node. Stage 1 uses all
# CPUs for parallel generation; Stage 2 uses GNU Parallel to
# run one Tao instance per sample.
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

# --- Stage 1: Generate inputs ---

echo "=== Stage 1: Generating inputs ==="
echo "  output_dir=$DATA_DIR  mode=$MODE  n_samples=$N_SAMPLES  seq_len=$SEQ_LEN  n_sections=$N_SECTIONS  seed=$SEED"

python scripts/generate_inputs.py \
    --mode "$MODE" \
    --n-samples "$N_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --n-sections "$N_SECTIONS" \
    --n-particles "$N_PARTICLES" \
    --output-dir "$DATA_DIR" \
    --seed "$SEED" \
    --workers "$SLURM_CPUS_ON_NODE"

echo "Stage 1 complete."

# --- Stage 2: Track particles with Tao ---

echo "=== Stage 2: Tracking with Tao ==="

track_one() {
    local sample_dir=$1
    local tao_init=$2

    if [ ! -f "$sample_dir/lattice.bmad" ]; then
        echo "SKIP $sample_dir: no lattice.bmad"
        return 1
    fi

    cd "$sample_dir"
    tao -init_file "$tao_init" -noplot \
        -lat lattice.bmad \
        -beam_init_position_file beam.h5 <<'EOF'
set global track_type = beam
quit
EOF
    echo "DONE $sample_dir"
}
export -f track_one

find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort | \
    parallel -j "$SLURM_CPUS_ON_NODE" --joblog "$DATA_DIR/tracking.log" \
    track_one {} "$TAO_INIT"

echo "=== Done. Data written to $DATA_DIR ==="
echo "Next: sbatch slurm/encode_tracked.sh $DATA_DIR <encoded_output_dir>"
