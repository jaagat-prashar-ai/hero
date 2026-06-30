#!/usr/bin/env bash
# Launch WORLD_SIZE parallel Lilypad generic jobs to shard PhysicalAI-AV.
# Each job processes clips[rank::world_size], giving WORLD_SIZE× throughput.
# Metadata parquets are assumed already uploaded — skip_metadata_upload=true.
#
# Usage: HF_TOKEN=hf_... bash masking/configs/launch_build_wds_parallel.sh

set -euo pipefail

CONFIG="masking/configs/build_wds_cluster.yaml"
WORLD_SIZE=50
WORKERS=2

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile oci.chi)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile oci.chi)

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var is required}"

echo "Launching ${WORLD_SIZE} parallel WDS build jobs (workers=${WORKERS} each)..."
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    lilypad workload launch "$CONFIG" \
        -o workload_variant_config.entrypoint_fn_config.hf_token "$HF_TOKEN" \
        -o workload_variant_config.entrypoint_fn_config.rank      "$rank" \
        -o workload_variant_config.entrypoint_fn_config.world_size "$WORLD_SIZE" \
        -o workload_variant_config.entrypoint_fn_config.workers    "$WORKERS" \
        -o workload_variant_config.entrypoint_fn_config.skip_metadata_upload true \
        -n "build-physicalai-wds-p${rank}" \
        2>&1 | grep -iE "workload.id|launched|error" || true
    echo "  Submitted rank ${rank}/${WORLD_SIZE}"
done

echo "All ${WORLD_SIZE} jobs submitted."
