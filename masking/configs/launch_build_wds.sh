#!/usr/bin/env bash
# Launch 32 parallel Lilypad workers to shard the PhysicalAI-AV dataset.
# Each worker processes 1/32 of clips and writes rank-namespaced shards.
# Run from the workspace root: bash masking/configs/launch_build_wds.sh

set -euo pipefail

WORLD_SIZE=32
CONFIG="masking/configs/build_wds_cluster.yaml"

echo "Launching ${WORLD_SIZE} workers..."

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    printf "  rank %02d/%d ... " "$rank" "$((WORLD_SIZE - 1))"
    lilypad workload launch "$CONFIG" \
        -o workload_variant_config.entrypoint_fn_config.rank "$rank" \
        -o workload_variant_config.entrypoint_fn_config.world_size "$WORLD_SIZE" \
        -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN:?HF_TOKEN env var is required}" \
        -n "build-physicalai-wds-r${rank}" \
        2>&1 | grep -E "workload_id|Launched|error|Error" | head -2
done

echo "All ${WORLD_SIZE} workers submitted."
