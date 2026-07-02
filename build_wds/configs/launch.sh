#!/usr/bin/env bash
# Lilypad launcher for hero/build_wds workloads.
#
# Run from repo root:
#   bash build_wds/configs/launch.sh build-wds [--dry-run] [--watch]
#   bash build_wds/configs/launch.sh build-wds-parallel [WORLD_SIZE] [WORKERS]
#   bash build_wds/configs/launch.sh build-wds-smoke-parallel [WORLD_SIZE] [MAX_CLIPS]
#   bash build_wds/configs/launch.sh build-wds-staggered [START_RANK] [END_RANK] [SLEEP_BETWEEN] [WORLD_SIZE] [WORKERS]
#
# HF_TOKEN and AWS creds are loaded from ~/.creds/lilypad.env automatically.
#
# Examples:
#   bash build_wds/configs/launch.sh build-wds-parallel 50 1
#   bash build_wds/configs/launch.sh build-wds-smoke-parallel 8 2
#   bash build_wds/configs/launch.sh build-wds-staggered 0 33 15 100 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_PY="${REPO_ROOT}/lilypad/launch.py"
LILYPAD_PYTHON="${LILYPAD_PYTHON:-${HOME}/.local/share/lilypad-tools/venv/bin/python}"

cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash build_wds/configs/launch.sh <command> [args...]

Commands:
  build-wds               Single-node WDS build (cluster.yaml)
  build-wds-smoke         Smoke test: 1 clip → 1 shard upload (smoke.yaml)
  build-wds-parallel      Launch WORLD_SIZE parallel WDS shard jobs
  build-wds-smoke-parallel  Launch WORLD_SIZE parallel smoke jobs (verify
                            chunk-aware partitioning before a full launch)
  build-wds-staggered     Relaunch a rank range with stagger (HF rate limits)

Common flags (passed through to launch.py):
  --dry-run            Validate config before submit
  --skip-dependency-validation
  --watch              lilypad watch after submit
  -n NAME              Override workload name
  -o KEY VALUE         Dot-path override (repeatable)

Environment:
  HF_TOKEN             Loaded from ~/.creds/lilypad.env
  AWS creds            Loaded from ~/.creds/lilypad.env (forwarded by Lilypad)
EOF
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
    build-wds)
        require_hf_token
        launch_py "${SCRIPT_DIR}/cluster.yaml" \
            -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
            -n build-physicalai-wds \
            "$@"
        ;;

    build-wds-smoke)
        require_hf_token
        launch_py "${SCRIPT_DIR}/smoke.yaml" \
            -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
            -n build-wds-smoke-1shard \
            "$@"
        ;;

    build-wds-parallel)
        require_hf_token
        WORLD_SIZE="${1:-50}"
        # BEFORE: WORKERS="${2:-2}"
        # cluster.yaml's own `workers: 1` comment warns to keep this at 1 —
        # physical_ai_av downloads each camera as a large chunk ZIP, and
        # concurrent downloads on one node exhaust its ~30GB RAM and OOM-kill
        # the job. This default of 2 contradicted that warning ever since
        # both files were added together in 839cc2b.
        # AFTER: default matches the documented-safe value; override
        # explicitly (accepting the OOM risk) only if you know what you're doing.
        WORKERS="${2:-1}"
        shift $(( $# >= 2 ? 2 : $# ))

        echo "Launching ${WORLD_SIZE} parallel WDS build jobs (workers=${WORKERS} each)..."
        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            launch_py "${SCRIPT_DIR}/cluster.yaml" \
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

    build-wds-smoke-parallel)
        require_hf_token
        WORLD_SIZE="${1:-8}"
        MAX_CLIPS="${2:-2}"
        shift $(( $# >= 2 ? 2 : $# ))

        # workers=1 always: AV1 transcoding is CPU-bound, so multiple threads
        # on one node just contend for the same limited cores. Parallelism
        # comes from launching more separate jobs (more nodes), not more
        # threads per node.
        echo "Launching ${WORLD_SIZE} parallel smoke jobs (workers=1 each, max_clips=${MAX_CLIPS})..."
        for rank in $(seq 0 $((WORLD_SIZE - 1))); do
            launch_py "${SCRIPT_DIR}/smoke.yaml" \
                -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
                -o workload_variant_config.entrypoint_fn_config.rank "${rank}" \
                -o workload_variant_config.entrypoint_fn_config.world_size "${WORLD_SIZE}" \
                -o workload_variant_config.entrypoint_fn_config.workers 1 \
                -o workload_variant_config.entrypoint_fn_config.max_clips "${MAX_CLIPS}" \
                -n "build-wds-smoke-p${rank}" \
                "$@"
            echo "  Submitted rank ${rank}/${WORLD_SIZE}"
        done
        echo "All ${WORLD_SIZE} smoke jobs submitted."
        ;;

    build-wds-staggered)
        require_hf_token
        START_RANK="${1:-31}"
        END_RANK="${2:-49}"
        SLEEP_BETWEEN="${3:-15}"
        # BEFORE: world_size was hardcoded to 50 below (baked in for a
        # one-off incident at that scale) and workers was hardcoded to 2.
        # Reusing this command at a different WORLD_SIZE (e.g. relaunching
        # ranks from a world_size=100 run) silently broke
        # chunk_id % world_size == rank partitioning — any rank >= 50 also
        # tripped build_webdataset.py's own `--rank must be < --world_size`
        # guard. workers=2 also contradicted cluster.yaml's `workers: 1`
        # OOM-safety comment, same issue as build-wds-parallel above.
        # AFTER: WORLD_SIZE is a 4th explicit argument (must match the
        # original launch's world_size — this only relaunches specific
        # ranks, it doesn't repartition), and workers defaults to 1.
        WORLD_SIZE="${4:-100}"
        WORKERS="${5:-1}"
        shift $(( $# >= 5 ? 5 : $# ))

        echo "Relaunching ranks ${START_RANK}-${END_RANK} of world_size=${WORLD_SIZE} with ${SLEEP_BETWEEN}s stagger (workers=${WORKERS})..."
        for rank in $(seq "${START_RANK}" "${END_RANK}"); do
            launch_py "${SCRIPT_DIR}/cluster.yaml" \
                -o workload_variant_config.entrypoint_fn_config.hf_token "${HF_TOKEN}" \
                -o workload_variant_config.entrypoint_fn_config.rank "${rank}" \
                -o workload_variant_config.entrypoint_fn_config.world_size "${WORLD_SIZE}" \
                -o workload_variant_config.entrypoint_fn_config.workers "${WORKERS}" \
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
