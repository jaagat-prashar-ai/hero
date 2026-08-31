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
  local suffix="$1" gate_mode="$2" top_k="$3"
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
    -o runtime_environment.constant_environment_variables.CODE_REWARD_GATE_MODE "${gate_mode}" \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_VERIFY_TOP_K "top${top_k}" \
    -o runtime_environment.constant_environment_variables.CODE_REWARD_DEBUG_DUMP_RUN_ID "clipgen602-${suffix}"
}

cd "${repo_root}"
arms=("${@:-hard-top1 twotier-top1 twotier-top3}")
failures=0
for arm_list in "${arms[@]}"; do
  for arm in ${arm_list}; do
    case "${arm}" in
      hard-top1) launch_one hard-top1 hard 1 || failures=$((failures + 1)) ;;
      twotier-top1) launch_one twotier-top1 two_tier 1 || failures=$((failures + 1)) ;;
      twotier-top3) launch_one twotier-top3 two_tier 3 || failures=$((failures + 1)) ;;
      *) echo "unknown arm: ${arm}" >&2; failures=$((failures + 1)) ;;
    esac
  done
done
exit "${failures}"
