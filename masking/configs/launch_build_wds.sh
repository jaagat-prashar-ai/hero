#!/usr/bin/env bash
# Launch 64 parallel Lilypad workers to shard the PhysicalAI-AV dataset.
# Each worker processes 1/64 of clips and writes rank-namespaced shards.
# Run from the workspace root: bash masking/configs/launch_build_wds.sh

set -euo pipefail

WORLD_SIZE=64
CONFIG="masking/configs/build_wds_cluster.yaml"

echo "Launching ${WORLD_SIZE} workers..."

for rank in $(seq 1 $((WORLD_SIZE - 1))); do
    printf "  rank %02d/%d ... " "$rank" "$((WORLD_SIZE - 1))"
    lilypad workload launch "$CONFIG" \
        -o workload_variant_config.entrypoint_fn_config.rank "$rank" \
        -o workload_variant_config.entrypoint_fn_config.world_size "$WORLD_SIZE" \
        -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN:?HF_TOKEN env var is required}" \
        -n "build-physicalai-wds-r${rank}" \
        2>&1 | grep -iE "workload.id|launched|error" | head -2 || true
done

echo "All ${WORLD_SIZE} workers submitted."
