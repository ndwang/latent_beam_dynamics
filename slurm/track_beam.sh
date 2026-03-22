#!/bin/bash
#SBATCH --job-name=lbd_track
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# STAGE 2: Track particles through Bmad/Tao
# ============================================
# Uses GNU Parallel to run one Tao instance per sample,
# filling all available CPUs on the node.
#
# Usage:
#   sbatch slurm/track_beam.sh data/structured
#   sbatch slurm/track_beam.sh data/random
# ============================================

DATA_DIR=${1:?Usage: sbatch track_beam.sh <data_dir>}
TAO_INIT="/pscratch/sd/n/ndwang/latent_beam_dynamics/tao.init"

ml load conda
conda activate lbd_datagen
ml load parallel

export OMP_NUM_THREADS=1

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
    parallel -j $SLURM_CPUS_ON_NODE --joblog "$DATA_DIR/tracking.log" \
    track_one {} "$TAO_INIT"

echo "All tracking complete. See $DATA_DIR/tracking.log for summary."
