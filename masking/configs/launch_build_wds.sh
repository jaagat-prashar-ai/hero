#!/usr/bin/env bash
# Launch a single 64-node Lilypad job to shard the PhysicalAI-AV dataset.
# Lilypad injects RANK / WORLD_SIZE into each node automatically.
# Run from the workspace root: bash masking/configs/launch_build_wds.sh

set -euo pipefail

CONFIG="masking/configs/build_wds_cluster.yaml"

# Pull OCI Chicago credentials from local AWS profile into env vars so
# Lilypad can pass them to the worker nodes via required_environment_variables.
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile oci.chi)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile oci.chi)

echo "Launching 64-node WDS build job..."
lilypad workload launch "$CONFIG" \
    -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN:?HF_TOKEN env var is required}" \
    -n "build-physicalai-wds" \
    2>&1 | grep -iE "workload.id|launched|error" || true

echo "Done."
