#!/usr/bin/env bash
# Unified Lilypad launcher for hero/masking workloads.
#
# All submissions go through launch.py (Python SDK). Run from repo root:
#
#   bash masking/configs/launch.sh masking [--dry-run] [--watch] [-n NAME] [-o KEY VALUE ...]
#   bash masking/configs/launch.sh local
#   bash masking/configs/launch.sh build-wds-parallel [WORLD_SIZE] [WORKERS]
#
# HF_TOKEN is loaded from ~/.creds/lilypad.env automatically.
#
# Examples:
#   bash masking/configs/launch.sh masking -o workload_variant_config.training_fn_config.experiment b
#   bash masking/configs/launch.sh build-wds-parallel 50 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_PY="${SCRIPT_DIR}/launch.py"
LILYPAD_PYTHON="${LILYPAD_PYTHON:-${HOME}/.local/share/lilypad-tools/venv/bin/python}"

cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash masking/configs/launch.sh <command> [args...]

Commands:
  masking              Launch masking experiment (cluster.yaml)
  local                Launch local masking run (local.yaml)
  build-wds            Single-node WDS build (build_wds_cluster.yaml)
  build-wds-parallel   Launch WORLD_SIZE parallel WDS shard jobs
  build-wds-staggered  Relaunch a rank range with stagger (HF rate limits)

Common flags (passed through to launch.py):
  --dry-run            Validate config + uv dependency resolve before submit
  --skip-dependency-validation
                       Skip uv pip compile preflight (not recommended)
  --watch              lilypad watch after submit
  -n NAME              Override workload name
  -o KEY VALUE         Dot-path override (repeatable)

Environment:
  HF_TOKEN             Loaded from ~/.creds/lilypad.env
  AWS creds            Pulled from oci.chi profile for build-wds* commands
EOF
}

export_aws_creds() {
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id --profile oci.chi)"
    AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key --profile oci.chi)"
}

source_lilypad_creds() {
    if [[ ! -f "${HOME}/.creds/lilypad.env" ]]; then
        echo "warning: ~/.creds/lilypad.env not found" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "${HOME}/.creds/lilypad.env"
}

require_hf_token() {
    source_lilypad_creds || true
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is not set — add it to ~/.creds/lilypad.env" >&2
        exit 1
    fi
}

launch_py() {
    source_lilypad_creds || true
    if [[ -x "${LILYPAD_PYTHON}" ]]; then
        "${LILYPAD_PYTHON}" "${LAUNCH_PY}" "$@"
    else
        python3 "${LAUNCH_PY}" "$@"
    fi
}

cmd="${1:-}"
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
    usage
    exit 0
fi
shift

case "${cmd}" in
    masking)
        launch_py "${SCRIPT_DIR}/cluster.yaml" "$@"
        ;;

    local)
        launch_py "${SCRIPT_DIR}/local.yaml" "$@"
        ;;

    build-wds)
        export_aws_creds
        require_hf_token
        launch_py "${SCRIPT_DIR}/build_wds_cluster.yaml" \
            -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
            -n build-physicalai-wds \
            "$@"
        ;;

    build-wds-parallel)
        export_aws_creds
        require_hf_token
        WORLD_SIZE="${1:-50}"
        WORKERS="${2:-2}"
        shift $(( $# >= 2 ? 2 : $# ))

        echo "Launching ${WORLD_SIZE} parallel WDS build jobs (workers=${WORKERS} each)..."
        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            launch_py "${SCRIPT_DIR}/build_wds_cluster.yaml" \
                -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
                -o workload_variant_config.entrypoint_fn_config.rank "${rank}" \
                -o workload_variant_config.entrypoint_fn_config.world_size "${WORLD_SIZE}" \
                -o workload_variant_config.entrypoint_fn_config.workers "${WORKERS}" \
                -o workload_variant_config.entrypoint_fn_config.skip_metadata_upload true \
                -n "build-physicalai-wds-p${rank}" \
                "$@"
            echo "  Submitted rank ${rank}/${WORLD_SIZE}"
        done
        echo "All ${WORLD_SIZE} jobs submitted."
        ;;

    build-wds-staggered)
        export_aws_creds
        require_hf_token
        START_RANK="${1:-31}"
        END_RANK="${2:-49}"
        SLEEP_BETWEEN="${3:-15}"
        shift $(( $# >= 3 ? 3 : $# ))

        echo "Relaunching ranks ${START_RANK}-${END_RANK} with ${SLEEP_BETWEEN}s stagger..."
        for rank in $(seq "${START_RANK}" "${END_RANK}"); do
            launch_py "${SCRIPT_DIR}/build_wds_cluster.yaml" \
                -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
                -o workload_variant_config.entrypoint_fn_config.rank "${rank}" \
                -o workload_variant_config.entrypoint_fn_config.world_size 50 \
                -o workload_variant_config.entrypoint_fn_config.workers 2 \
                -o workload_variant_config.entrypoint_fn_config.skip_metadata_upload true \
                -n "build-physicalai-wds-p${rank}" \
                "$@"
            echo "  Submitted rank ${rank}"
            sleep "${SLEEP_BETWEEN}"
        done
        echo "All staggered ranks submitted."
        ;;

    *)
        echo "Unknown command: ${cmd}" >&2
        usage
        exit 1
        ;;
esac
