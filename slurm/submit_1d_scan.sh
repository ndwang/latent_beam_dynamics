#!/bin/bash
#SBATCH --job-name=lbd_1d_scan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# 1D HYPERPARAMETER SCAN — LatentBeamTransformer
# ============================================
# Usage: sbatch slurm/submit_1d_scan.sh
# Runs multiple configs in parallel using GNU parallel + srun.
# ============================================

# --- CONFIGURATION ---
# Usage: sbatch submit_1d_scan.sh <param_name> <space-separated values> <fixed overrides> <sweep group>
# Example: sbatch submit_1d_scan.sh "model.n_layers" "1 2 3 4" "model.d_model=512 data.path=data/encoded_sectioned_10k training.epochs=500" "scan2_nlayers"
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <param_name> <values> <fixed_overrides> <sweep_group>" >&2
    exit 1
fi
PARAM_NAME="$1"
IFS=' ' read -ra PARAM_VALUES <<< "$2"
FIXED_OVERRIDES="$3"
SWEEP_GROUP="$4"

cd /pscratch/sd/n/ndwang/latent_beam_dynamics
ml load conda
conda activate lbd

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 16"

run_single() {
    local val=$1
    local param_short=$(echo $PARAM_NAME | sed 's/.*\.//')
    local ts=$(date +%y%m%d_%H%M)
    local run_name="${SWEEP_GROUP}_${param_short}${val}_${ts}"
    srun $SRUN_ARGS python scripts/train.py \
        ${PARAM_NAME}=${val} \
        run_name=${run_name} \
        ${FIXED_OVERRIDES} \
        training.wandb.enabled=true \
        training.wandb.group=${SWEEP_GROUP} \
        > logs/${param_short}_${val}.log 2>&1
}
export -f run_single
export PARAM_NAME FIXED_OVERRIDES SRUN_ARGS SWEEP_GROUP

parallel -j 4 --delay 0.2 run_single ::: "${PARAM_VALUES[@]}"

echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
