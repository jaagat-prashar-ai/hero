#!/usr/bin/env bash
# lilypad_aliases.sh — Bazel-managed lilypad CLI for the mnist-template.
#
# Source this file inside the dev container (after mounting the workspace):
#   source docker/lilypad_aliases.sh
#
# The lilypad CLI is built via Bazel and invoked from bazel-bin, ensuring the
# version tracked by the workspace lockfile is always used.
#
# Usage:
#   lilypad workload launch mnist/training/configs/local.yaml -n template-mnist-local

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"

unalias lilypad 2>/dev/null || true

lilypad() {
    "${WORKSPACE_ROOT}/docker/in_docker.sh" -c \
        "bazel build @python_deps_lilypad_py//:rules_python_wheel_entry_point_lilypad && \
         bazel-bin/external/python_deps_lilypad_py/rules_python_wheel_entry_point_lilypad -- $*"
}

export -f lilypad

echo "lilypad alias loaded (Bazel-managed, workspace: ${WORKSPACE_ROOT})"
echo "Usage: lilypad workload launch mnist/training/configs/local.yaml -n template-mnist-local"
