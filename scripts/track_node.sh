#!/bin/bash
# Per-node tracking driver for multi-node GNU parallel jobs.
# Called by srun — each node receives the full task list on stdin
# and processes its 1/N slice (determined by $SLURM_NODEID).
#
# Usage: srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
#            scripts/track_node.sh <joblog_dir> < dirs.txt

JOBLOG_DIR=${1:?Usage: track_node.sh <joblog_dir>}

awk -v N="$SLURM_NNODES" -v ID="$SLURM_NODEID" 'NR % N == ID' | \
    parallel -j "$SLURM_CPUS_ON_NODE" \
        --joblog "${JOBLOG_DIR}/tracking_node${SLURM_NODEID}.log" \
        bash /pscratch/sd/n/ndwang/latent_beam_dynamics/scripts/track_one.sh {}
