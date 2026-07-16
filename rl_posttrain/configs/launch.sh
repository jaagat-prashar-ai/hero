#!/usr/bin/env bash
# Lilypad launcher for the alpamayo1_x_rl RL post-training local test.
#
# Run from repo root:
#   bash rl_posttrain/configs/launch.sh local-test [--dry-run] [--watch] [-n NAME] [-o KEY VALUE ...]
#
# HF_TOKEN and WANDB_API_KEY are loaded from ~/.creds/lilypad.env automatically.
#
# Examples:
#   bash rl_posttrain/configs/launch.sh local-test --dry-run
#   bash rl_posttrain/configs/launch.sh local-test --watch
#   bash rl_posttrain/configs/launch.sh local-test -o workload_variant_config.entrypoint_fn_config.reasoning true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_PY="${REPO_ROOT}/lilypad/launch.py"
LILYPAD_PYTHON="${LILYPAD_PYTHON:-${HOME}/.local/share/lilypad-tools/venv/bin/python}"

cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash rl_posttrain/configs/launch.sh <command> [args...]

Commands:
  local-test           Launch the alpamayo1_x_rl single-node local test (cluster.yaml)

Common flags (passed through to launch.py):
  --dry-run            Validate config + uv dependency resolve before submit
  --skip-dependency-validation
                       Skip uv pip compile preflight (not recommended)
  --watch              lilypad watch after submit
  -n NAME              Override workload name
  -o KEY VALUE         Dot-path override (repeatable)

Environment:
  HF_TOKEN, WANDB_API_KEY   Loaded from ~/.creds/lilypad.env
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
    local-test)
        launch_py "${SCRIPT_DIR}/cluster.yaml" "$@"
        ;;

    *)
        echo "Unknown command: ${cmd}" >&2
        usage
        exit 1
        ;;
esac
