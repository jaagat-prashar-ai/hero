#!/usr/bin/env bash
set -euo pipefail

: "${AUTH_TOKEN:?export AUTH_TOKEN first}"
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"
: "${HF_TOKEN:?export HF_TOKEN first}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hosts_file="$(mktemp)"
trap 'rm -f "${hosts_file}"' EXIT
cp /etc/hosts "${hosts_file}"
printf '\n137.131.42.171 ml-infra.applied.dev\n' >> "${hosts_file}"
printf '134.70.16.1 idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com\n' >> "${hosts_file}"

launch_one() {
  local suffix="$1" score_mode="$2"
  unshare -Ur -m bash -c '
    mount --bind "$1" /etc/hosts
    shift
    export GRPC_DNS_RESOLVER=native
    exec "$@"
  ' bash "${hosts_file}" python3 lilypad/launch.py \
    rl_posttrain/configs/faithfulness_compare_fixed602.yaml \
    --skip-dependency-validation \
    -n "alpamayo-rl-clipgen602-${suffix}" \
    -o workload_variant_config.entrypoint_fn_config.workspace_dir "/mnt/work/tmp/alpamayo_rl_clipgen602_${suffix}" \
    -o workload_variant_config.entrypoint_fn_config.ckpt_s3_prefix "alpamayo_rl/checkpoints/clipgen602_${suffix}" \
    -o workload_variant_config.entrypoint_fn_config.wandb_experiment "clipgen602_${suffix}" \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_GATE_MODE two_tier \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_VERIFY_TOP_K top3 \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_VERIFY_MIN_PASSES all \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_SCORE_MODE "${score_mode}" \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID "clipgen602-${suffix}"
}

cd "${repo_root}"
failures=0
launch_one strict-top3-rank-v1 rank || failures=$((failures + 1))
launch_one strict-top3-hybrid75-v1 hybrid75 || failures=$((failures + 1))
launch_one strict-top3-hybrid50-v1 hybrid50 || failures=$((failures + 1))
exit "${failures}"
