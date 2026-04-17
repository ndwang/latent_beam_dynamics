#!/bin/bash
# Sync all offline W&B runs to the cloud.
# Usage: ./scripts/sync_wandb.sh [runs_dir]

RUNS_DIR="${1:-runs}"

OFFLINE_DIRS=$(find "$RUNS_DIR" -type d -name "offline-run-*" 2>/dev/null | sort)

if [ -z "$OFFLINE_DIRS" ]; then
    echo "No offline W&B runs found in $RUNS_DIR/"
    exit 0
fi

COUNT=$(echo "$OFFLINE_DIRS" | wc -l)
echo "Found $COUNT offline run(s) to sync"
echo ""

SYNCED=0
SKIPPED=0
FAILED=0
for dir in $OFFLINE_DIRS; do
    if compgen -G "$dir/*.wandb.synced" > /dev/null 2>&1; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "  Syncing: $dir"
    if wandb sync "$dir" 2>&1 | grep -qiE "(success|synced|done)"; then
        SYNCED=$((SYNCED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "    FAILED"
    fi
done

echo ""
echo "Done: $SYNCED synced, $SKIPPED skipped, $FAILED failed"
