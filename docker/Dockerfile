# syntax=docker/dockerfile:1
#
# MNIST template dev / build image.
#
# This image inherits the research-core dev base image *unchanged* — it is a NOOP
# today. It exists as the single, obvious place for maintainers to add
# project-specific container requirements (extra system packages, native
# libraries, fonts, env vars, ...) WITHOUT forking research-core.
#
# The base is the research-core development image (built and published by the
# research-core repo's docker/build_and_push.sh): an ubuntu:22.04 image with the
# Python ML stack, ray/lilypad-py, the AWS CLI, and Bazel (bazelisk) already
# present, which is what bazel build/test runs inside. It is supplied via the
# RESEARCH_CORE_DEV_IMAGE build arg so it can be repinned without editing this
# file:
#
#   docker build \
#     --build-arg RESEARCH_CORE_DEV_IMAGE=<image:tag> \
#     -t mnist-template-dev:latest -f docker/Dockerfile .
#
# See docker/build.sh for the canonical build command and docker/run.sh to start
# the dev container.

ARG RESEARCH_CORE_DEV_IMAGE=us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:research-core-dev-v0.1.1
FROM ${RESEARCH_CORE_DEV_IMAGE}

# ---------------------------------------------------------------------------
# Project-specific container customizations go here.
#
# NOOP for the MNIST template — it needs nothing beyond the research-core base.
# Python dependencies are NOT installed here; they are resolved hermetically by
# Bazel via pip_parse (lockfiles/requirements_lock.txt). This Dockerfile is only
# for *system-level* requirements that Bazel cannot provide.
#
# Example — add a system package:
#
#   USER root
#   RUN apt-get update \
#       && apt-get install -y --no-install-recommends libfoo-dev \
#       && rm -rf /var/lib/apt/lists/*
#   USER lilypad
# ---------------------------------------------------------------------------
