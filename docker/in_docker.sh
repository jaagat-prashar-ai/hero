#!/usr/bin/env bash
# in_docker.sh — Run a command inside the mnist-template dev container.
#
# Usage:
#   ./docker/in_docker.sh -c "<command>"
#   ./docker/in_docker.sh -c "bazel test //mnist/..."
#
# The container name is read from .dev_docker_name in the workspace root and is
# expected to have been created by docker/run.sh (which mounts the workspace 1:1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEV_DOCKER_NAME_FILE="${WORKSPACE_ROOT}/.dev_docker_name"
if [[ ! -f "${DEV_DOCKER_NAME_FILE}" ]]; then
    echo "ERROR: .dev_docker_name not found at ${DEV_DOCKER_NAME_FILE}" >&2
    exit 1
fi
CONTAINER_NAME="$(tr -d '[:space:]' < "${DEV_DOCKER_NAME_FILE}")"

# Parse arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--command)
            COMMAND="$2"
            shift 2
            ;;
        *)
            COMMAND="$*"
            break
            ;;
    esac
done

if [[ -z "${COMMAND}" ]]; then
    echo "Usage: $0 -c '<command>'" >&2
    exit 1
fi

if ! docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q "true"; then
    echo "ERROR: Container '${CONTAINER_NAME}' is not running." >&2
    echo "Start it with: ./docker/build.sh && ./docker/run.sh" >&2
    exit 1
fi

# Forward important environment variables for AWS / WandB.
ENV_ARGS=()
for var in AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN WANDB_API_KEY; do
    if [[ -n "${!var:-}" ]]; then
        ENV_ARGS+=(-e "${var}=${!var}")
    fi
done

# We exec as the host uid:gid (so files written to the mounted workspace are
# host-owned), but the research-core base image bakes HOME=/home/lilypad owned by
# its fixed lilypad user — unreadable to a different host uid. Point HOME at the
# host home instead: run.sh mounts the host ${HOME}/.cache 1:1, so this makes
# bazel's ~/.cache (output base, bazelisk, disk cache) land in that writable,
# host-owned mount rather than the unwritable /home/lilypad.
exec docker exec \
    -u "$(id -u):$(id -g)" \
    -e "HOME=${HOME}" \
    -e "USER=$(id -un)" \
    -w "${WORKSPACE_ROOT}" \
    "${ENV_ARGS[@]}" \
    "${CONTAINER_NAME}" \
    bash -c "${COMMAND}"
