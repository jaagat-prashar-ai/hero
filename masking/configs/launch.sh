#!/usr/bin/env bash
# Lilypad launcher for hero/masking inference workloads.
#
# Run from repo root:
#   bash masking/configs/launch.sh masking [--dry-run] [--watch] [-n NAME] [-o KEY VALUE ...]
#   bash masking/configs/launch.sh local
#
# HF_TOKEN is loaded from ~/.creds/lilypad.env automatically.
#
# Examples:
#   bash masking/configs/launch.sh masking -o workload_variant_config.training_fn_config.experiment b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_PY="${REPO_ROOT}/lilypad/launch.py"
LILYPAD_PYTHON="${LILYPAD_PYTHON:-${HOME}/.local/share/lilypad-tools/venv/bin/python}"

cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash masking/configs/launch.sh <command> [args...]

Commands:
  masking              Launch masking experiment (cluster.yaml)
  local                Launch local masking run (local.yaml)

Common flags (passed through to launch.py):
  --dry-run            Validate config + uv dependency resolve before submit
  --skip-dependency-validation
                       Skip uv pip compile preflight (not recommended)
  --watch              lilypad watch after submit
  -n NAME              Override workload name
  -o KEY VALUE         Dot-path override (repeatable)

Environment:
  HF_TOKEN             Loaded from ~/.creds/lilypad.env
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
    masking)
        launch_py "${SCRIPT_DIR}/cluster.yaml" "$@"
        ;;

    local)
        launch_py "${SCRIPT_DIR}/local.yaml" "$@"
        ;;

    *)
        echo "Unknown command: ${cmd}" >&2
        usage
        exit 1
        ;;
esac
