#!/usr/bin/env bash
# Relaunch ranks 31-49 with staggered starts to avoid HF rate limiting.
# Run after launch_build_wds_parallel.sh when higher ranks fail due to 429s.

set -euo pipefail

CONFIG="masking/configs/build_wds_cluster.yaml"
WORLD_SIZE=50
WORKERS=2
START_RANK=31
END_RANK=49
SLEEP_BETWEEN=15  # seconds — keeps HF API requests well under 1000/5min

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile oci.chi)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile oci.chi)

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var is required}"

echo "Relaunching ranks ${START_RANK}-${END_RANK} with ${SLEEP_BETWEEN}s stagger..."
for rank in $(seq ${START_RANK} ${END_RANK}); do
    lilypad workload launch "$CONFIG" \
        -o workload_variant_config.entrypoint_fn_config.hf_token "$HF_TOKEN" \
        -o workload_variant_config.entrypoint_fn_config.rank      "$rank" \
        -o workload_variant_config.entrypoint_fn_config.world_size "$WORLD_SIZE" \
        -o workload_variant_config.entrypoint_fn_config.workers    "$WORKERS" \
        -o workload_variant_config.entrypoint_fn_config.skip_metadata_upload true \
        -n "build-physicalai-wds-p${rank}" \
        2>&1 | grep -iE "workload.id|launched|error" || true
    echo "  Submitted rank ${rank}/${WORLD_SIZE}"
    sleep ${SLEEP_BETWEEN}
done

echo "All staggered ranks submitted."
