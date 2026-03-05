#!/bin/bash
# ============================================
# SYNC ALL W&B OFFLINE RUNS
# ============================================
# Usage: ./slurm/sync_wandb.sh
# Run from login node after training completes
# ============================================

cd /pscratch/sd/n/ndwang/latent_beam_dynamics

# Find and sync all offline runs
OFFLINE_DIRS=$(find runs -type d -name "offline-run-*" 2>/dev/null)

if [ -z "$OFFLINE_DIRS" ]; then
    echo "No offline W&B runs found in runs/"
    exit 0
fi

echo "Found $(echo "$OFFLINE_DIRS" | wc -l) offline run(s) to sync:"
echo "$OFFLINE_DIRS" | head -10
[ $(echo "$OFFLINE_DIRS" | wc -l) -gt 10 ] && echo "  ..."

echo ""
echo "Syncing..."

for dir in $OFFLINE_DIRS; do
    echo "  Syncing: $dir"
    wandb sync "$dir" 2>&1 | grep -E "(Syncing|success|error)" || true
done

echo ""
echo "Sync complete."
