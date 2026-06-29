#!/usr/bin/env bash
# build.sh — Build the mnist-template dev/build image.
#
# The image is a NOOP layer on top of the research-core dev base image (see
# docker/Dockerfile). Override the base with RESEARCH_CORE_DEV_IMAGE if needed.
#
# Usage:
#   ./docker/build.sh
#   RESEARCH_CORE_DEV_IMAGE=<image:tag> ./docker/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-mnist-template-dev:latest}"

BUILD_ARGS=()
if [[ -n "${RESEARCH_CORE_DEV_IMAGE:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RESEARCH_CORE_DEV_IMAGE=${RESEARCH_CORE_DEV_IMAGE}")
fi

echo "Building ${IMAGE_TAG} from ${WORKSPACE_ROOT}/docker/Dockerfile ..."
exec docker build \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_TAG}" \
    -f "${WORKSPACE_ROOT}/docker/Dockerfile" \
    "${WORKSPACE_ROOT}"
