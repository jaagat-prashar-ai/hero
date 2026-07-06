#!/usr/bin/env bash
# Lilypad launcher for pref_pairs rollout-harvest + maneuver-classification
# workloads. Reuses masking/configs/launch.py as-is (it's generic over
# config path -- see that file's docstring) rather than duplicating it.
#
# Run from repo root:
#   bash pref_pairs/configs/launch.sh cluster [--dry-run] [--watch] [-n NAME] [-o KEY VALUE ...]
#   bash pref_pairs/configs/launch.sh local
#
# HF_TOKEN is loaded from ~/.creds/lilypad.env automatically.
#
# Examples:
#   bash pref_pairs/configs/launch.sh local --dry-run
#   bash pref_pairs/configs/launch.sh cluster -o workload_variant_config.training_fn_config.max_scenes 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_PY="${REPO_ROOT}/masking/configs/launch.py"
LILYPAD_PYTHON="${LILYPAD_PYTHON:-${HOME}/.local/share/lilypad-tools/venv/bin/python}"

cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash pref_pairs/configs/launch.sh <command> [args...]

Commands:
  cluster               Launch pref_pairs cluster run (cluster.yaml)
  local                 Launch local pref_pairs run (local.yaml)

Common flags (passed through to masking/configs/launch.py):
  --dry-run            Validate config + uv dependency resolve before submit
  --skip-dependency-validation
                       Skip uv pip compile preflight (not recommended)
  --watch              lilypad watch after submit
  -n NAME              Override workload name
  -o KEY VALUE         Dot-path override (repeatable)

Environment:
  HF_TOKEN             Loaded from ~/.creds/lilypad.env

Before the FIRST run, build the scene manifest (separate, lighter step --
no GPU needed, see masking/data/sample_clips.py's --all usage example):
  python -m masking.data.sample_clips --all \
      --out pref_pairs/configs/sample_clips_all.json
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
    cluster)
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
