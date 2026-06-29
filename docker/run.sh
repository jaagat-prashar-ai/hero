#!/usr/bin/env bash
# run.sh — Start (or restart) the mnist-template dev container.
#
# Creates a long-lived container named per .dev_docker_name from the
# mnist-template-dev image, mounting the workspace 1:1 (so the research-core
# submodule's paths resolve) along with AWS / Lilypad / WandB credentials and a
# Bazel disk cache. Run commands inside it with docker/in_docker.sh.
#
# Usage:
#   ./docker/build.sh && ./docker/run.sh
#   ./docker/in_docker.sh -c "bazel test //mnist/..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-mnist-template-dev:latest}"
CONTAINER_NAME="$(tr -d '[:space:]' < "${WORKSPACE_ROOT}/.dev_docker_name")"

# Remove any existing container with this name.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Optional host mounts (only added if present on the host).
MOUNTS=(-v "${WORKSPACE_ROOT}:${WORKSPACE_ROOT}")
for path in "${HOME}/.aws" "${HOME}/.lilypad" "${HOME}/.netrc" "${HOME}/.docker"; do
    [[ -e "${path}" ]] && MOUNTS+=(-v "${path}:${path}")
done
[[ -e /var/run/docker.sock ]] && MOUNTS+=(-v /var/run/docker.sock:/var/run/docker.sock)
# Mount the whole host ~/.cache so it is host-owned (writable by the container
# user). This covers bazel's output base (~/.cache/bazel), the bazelisk download
# cache (~/.cache/bazelisk), and the disk cache (~/.cache/bazel_cache_mnist_template
# from .bazelrc). Bind-mounting only deep subdirs would leave ~/.cache itself
# root-owned (the base image does not pre-create it), breaking bazel startup.
mkdir -p "${HOME}/.cache/bazel_cache_mnist_template"
MOUNTS+=(-v "${HOME}/.cache:${HOME}/.cache")

GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia; then
    GPU_ARGS+=(--gpus all)
fi

# Grant the container user access to the mounted docker socket so a local
# `lilypad workload launch` can `docker load` / `docker run` the built OCI image
# (docker-in-docker). The socket is owned by root:<docker-gid>; add that gid as a
# supplementary group (inherited by `docker exec`, even with -u uid:gid).
GROUP_ARGS=()
if [[ -S /var/run/docker.sock ]]; then
    GROUP_ARGS+=(--group-add "$(stat -c %g /var/run/docker.sock)")
fi

echo "Starting ${CONTAINER_NAME} from ${IMAGE_TAG} ..."
exec docker run -d \
    --name "${CONTAINER_NAME}" \
    "${GPU_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    "${MOUNTS[@]}" \
    -w "${WORKSPACE_ROOT}" \
    "${IMAGE_TAG}" \
    sleep infinity
