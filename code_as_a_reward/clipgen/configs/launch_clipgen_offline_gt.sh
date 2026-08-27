#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

# Launch only the GT/images/observations -> cached-reward phase. No command
# in this file accepts a rollout prefix or invokes clipgen_real_rollout_loop.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEMPLATE="${REPO_ROOT}/code_as_a_reward/clipgen/configs/clipgen_offline_gt_shard0.yaml"
RUN_TAG="${CLIPGEN_OFFLINE_RUN_TAG:-full1050-v1}"
SELECTION_RUN_TAG="${CLIPGEN_SELECTION_RUN_TAG:-full1050-v1}"
SELECTION_MODE="${CLIPGEN_SELECTION_MODE:-repair}"
START_SHARD="${CLIPGEN_START_SHARD:-0}"
END_SHARD="${CLIPGEN_END_SHARD:-24}"

case "${1:-}" in
  shard0)
    shift
    python3 "${REPO_ROOT}/lilypad/launch.py" "${TEMPLATE}" "$@"
    ;;
  full)
    shift
    for shard in $(seq "${START_SHARD}" "${END_SHARD}"); do
      python3 "${REPO_ROOT}/lilypad/launch.py" "${TEMPLATE}" \
        -n "clipgen-offline-${RUN_TAG}-shard${shard}" \
        -o workload_variant_config.training_fn_config.manifest_data_s3_prefix \
          "code_as_a_reward/clipgen/manifest_full1050/shard_${shard}" \
        -o workload_variant_config.training_fn_config.manifest_local_dir \
          "/mnt/work/tmp/clipgen_offline_manifest_${RUN_TAG}_shard${shard}" \
        -o workload_variant_config.training_fn_config.out_dir \
          "/mnt/work/tmp/clipgen_offline_out_${RUN_TAG}_shard${shard}" \
        -o workload_variant_config.training_fn_config.s3_prefix \
          "code_as_a_reward/clipgen/offline_rewards/${RUN_TAG}-shard${shard}" \
        "$@"
    done
    ;;
  repair)
    shift
    for shard in $(seq "${START_SHARD}" "${END_SHARD}"); do
      python3 "${REPO_ROOT}/lilypad/launch.py" "${TEMPLATE}" \
        -n "clipgen-offline-${RUN_TAG}-shard${shard}" \
        -o workload_variant_config.training_fn_config.manifest_data_s3_prefix \
          "code_as_a_reward/clipgen/manifest_full1050/shard_${shard}" \
        -o workload_variant_config.training_fn_config.selection_report_s3_key \
          "code_as_a_reward/clipgen/offline_rewards/${SELECTION_RUN_TAG}-shard${shard}/report.json" \
        -o workload_variant_config.training_fn_config.selection_mode \
          "${SELECTION_MODE}" \
        -o workload_variant_config.training_fn_config.manifest_local_dir \
          "/mnt/work/tmp/clipgen_offline_manifest_${RUN_TAG}_shard${shard}" \
        -o workload_variant_config.training_fn_config.out_dir \
          "/mnt/work/tmp/clipgen_offline_out_${RUN_TAG}_shard${shard}" \
        -o workload_variant_config.training_fn_config.s3_prefix \
          "code_as_a_reward/clipgen/offline_rewards/${RUN_TAG}-shard${shard}" \
        "$@"
    done
    ;;
  *)
    echo "usage: $0 {shard0|full|repair} [launch.py flags]" >&2
    exit 2
    ;;
esac
