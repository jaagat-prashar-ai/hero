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
  llm-judge            Launch GRPO with the LLM-as-judge reasoning reward
                       (llm_judge_cluster.yaml; bridges ~/.creds/anthropic.key
                       into ANTHROPIC_API_KEY when unset)
  llm-judge-full       Extensive OOD run: all OOD clips in the 100 densest
                       chunks (~394 clips, ~570 GB), 3 epochs, S3 warm cache
                       (llm_judge_full_cluster.yaml; same key bridging)
  code-reward          Launch GRPO with the deterministic code-as-a-reward
                       claim verifier (code_reward_cluster.yaml; no
                       Anthropic key needed -- reward computed on-node)
  inspect-logs         GPU-free: read a prior run's per-process cosmos-rl logs
                       from /mnt/work (reward/wandb lines) without re-running
                       the expensive GPU job (inspect_logs.yaml)

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

# Pick an interpreter that can actually import the lilypad SDK. The naive
# `python3` fallback breaks when the repo's .venv is active: that venv has no
# lilypad_py installed, and the repo-root `lilypad/` dir (which holds only our
# launch.py) then shadows the SDK namespace package entirely, so launch.py
# dies with `ModuleNotFoundError: No module named 'lilypad.public'`. The
# system python merges both namespace-package halves (repo lilypad/ + user
# site-packages lilypad/) and works, so probe candidates instead of assuming.
find_lilypad_python() {
    local candidate
    for candidate in "${LILYPAD_PYTHON}" python3 /usr/bin/python3; do
        if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
            if "${candidate}" -c "import lilypad.public.schemas.workload_config" >/dev/null 2>&1; then
                echo "${candidate}"
                return 0
            fi
        fi
    done
    return 1
}

launch_py() {
    source_lilypad_creds || true
    local py
    if ! py="$(find_lilypad_python)"; then
        echo "error: no python with the lilypad SDK found (tried LILYPAD_PYTHON, python3, /usr/bin/python3); pip install lilypad_py" >&2
        return 1
    fi
    "${py}" "${LAUNCH_PY}" "$@"
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

    llm-judge|llm-judge-full)
        # Bridge the project's ~/.creds/anthropic.key convention into
        # ANTHROPIC_API_KEY (required_environment_variables entry) when the
        # caller hasn't exported one -- mirrors
        # pref_pairs.perturbation_generator.load_api_key.
        if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "${HOME}/.creds/anthropic.key" ]]; then
            ANTHROPIC_API_KEY="$(<"${HOME}/.creds/anthropic.key")"
            export ANTHROPIC_API_KEY
        fi
        if [[ "${cmd}" == "llm-judge-full" ]]; then
            launch_py "${SCRIPT_DIR}/llm_judge_full_cluster.yaml" "$@"
        else
            launch_py "${SCRIPT_DIR}/llm_judge_cluster.yaml" "$@"
        fi
        ;;

    code-reward)
        launch_py "${SCRIPT_DIR}/code_reward_cluster.yaml" "$@"
        ;;

    inspect-logs)
        launch_py "${SCRIPT_DIR}/inspect_logs.yaml" "$@"
        ;;

    *)
        echo "Unknown command: ${cmd}" >&2
        usage
        exit 1
        ;;
esac
