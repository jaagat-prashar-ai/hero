# SPDX-License-Identifier: Apache-2.0
"""Self-bootstrap the alpamayo1_x_rl recipe's own Python 3.12 venv.

Unlike perplexity/training/bootstrap_venv.py (which hand-installs pinned
packages because alpamayo_r1's own pyproject/lock didn't resolve cleanly in
that context), the alpamayo1_x_rl recipe under
third_party/alpamayo-recipes/recipes/alpamayo1_x_rl ships its own working
uv.lock (392KB, pins alpamayo_r1 and cosmos-rl to git revisions). We
replicate the recipe README's own two-step `uv sync`:

    uv venv --python 3.12 <venv_dir>
    uv sync --active --no-install-package flash-attn        # torch etc. first
    uv sync --active --no-build-isolation-package flash-attn # then flash-attn

UV_NO_CONFIG=1 avoids this repo's root-level [tool.uv] index config (ursa /
pytorch-cu128 / pypi.nvidia.com, no pypi.org) leaking into the recipe's own
dependency resolution -- the same failure mode documented in
perplexity/training/bootstrap_venv.py (broke hatchling's packaging>=24.2
requirement there). The cost: UV_NO_CONFIG also blocks uv from reading this
recipe's OWN pyproject.toml [tool.uv] table, including
`no-build-isolation-package = ["flash-attn"]` -- flash-attn imports torch at
build time and a fully-isolated build env can't see the torch just
installed, so that setting has to be passed explicitly as a CLI flag on the
second sync instead of relying on the (suppressed) pyproject.toml config.
(First hit exactly this failure on run alpamayo-rl-local-test-svmxnm,
2026-07-16: `ModuleNotFoundError: No module named 'torch'` building
flash-attn==2.8.3 in an isolated build venv despite torch==2.8.0 already
being installed in the target venv.)

Idempotent: writes a marker file after a successful sync so a re-invoked
job (e.g. after preemption/requeue) doesn't redo the flash-attn build,
which alone takes 20-40+ minutes from source.
"""

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

MARKER_NAME = "BOOTSTRAP_OK"
_VALIDATE_IMPORT = "import torch, cosmos_rl, alpamayo1_x_rl"


def _run(cmd: list[str], **kwargs) -> None:
    logger.info("bootstrap: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _ensure_extra_packages(python_bin: str, packages: tuple[str, ...]) -> None:
    """Idempotently install packages OUR code (reward/entry scripts) needs
    into the recipe venv, outside the BOOTSTRAP_OK marker flow: the venv
    persists on /mnt/work across jobs, so a venv built by an earlier run --
    before a given extra was needed -- validates fine and skips the sync
    path entirely, meaning the marker can never be trusted to imply extras
    are present. Presence is checked by import (assumes import name ==
    distribution name, true for everything we install this way), so a
    re-run with everything present costs one subprocess call per package.

    Installed via `uv pip` because `uv venv` environments ship without pip
    (`python -m pip` would fail inside them)."""
    missing = [
        p
        for p in packages
        if subprocess.run([python_bin, "-c", f"import {p}"], capture_output=True).returncode != 0
    ]
    if not missing:
        return
    uv_bin = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv_bin):
        # ~ is pod-local while the venv lives on persistent /mnt/work: a
        # requeued job on a fresh pod can reach here with the venv present
        # but uv itself gone.
        _run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    env = dict(os.environ)
    env["UV_NO_CONFIG"] = "1"
    logger.info("bootstrap: installing extra packages into recipe venv: %s", missing)
    _run([uv_bin, "pip", "install", "--python", python_bin, *missing], env=env)


def _validate(python_bin: str) -> bool:
    check = subprocess.run(
        [python_bin, "-c", _VALIDATE_IMPORT], capture_output=True, text=True
    )
    if check.returncode != 0:
        logger.warning(
            "bootstrap: venv failed validation (%s)",
            (check.stderr or "").strip().splitlines()[-1] if check.stderr else "no stderr",
        )
    return check.returncode == 0


def wait_for_recipe_venv(venv_dir: str, timeout_s: float = 90 * 60) -> str:
    """Wait for local rank 0 / another process to finish building the venv."""
    python_bin = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, MARKER_NAME)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(marker) and os.path.exists(python_bin) and _validate(python_bin):
            return python_bin
        time.sleep(10)
    raise TimeoutError(f"venv at {venv_dir} was not built within {timeout_s:.0f}s")


def ensure_recipe_venv(
    venv_dir: str, recipe_dir: str, extra_packages: tuple[str, ...] = ()
) -> str:
    """Build (once) the alpamayo1_x_rl venv, return the path to its python binary.

    Args:
        venv_dir: where to build the private Python 3.12 venv. Should live on
            persistent shared storage (/mnt/work/...) so a preempted/requeued
            job doesn't redo the multi-GB torch/vllm install + flash-attn build.
        recipe_dir: absolute path to
            third_party/alpamayo-recipes/recipes/alpamayo1_x_rl (code_assets
            copies the whole repo to the pod, so this exists on disk at job
            runtime).
        extra_packages: additional pip distributions OUR code needs inside the
            recipe venv (e.g. ("anthropic",) for the LLM-judge reward).
            Installed idempotently on EVERY call -- including the
            venv-already-built early return -- because the persistent venv
            may predate the need for them (see _ensure_extra_packages).
    """
    python_bin = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, MARKER_NAME)
    if os.path.exists(marker) and os.path.exists(python_bin):
        if _validate(python_bin):
            logger.info("bootstrap: venv already built at %s, validated, skipping", venv_dir)
            _ensure_extra_packages(python_bin, extra_packages)
            return python_bin
        logger.warning("bootstrap: venv at %s has a stale/corrupt marker -- rebuilding", venv_dir)
        shutil.rmtree(venv_dir, ignore_errors=True)

    uv_bin = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv_bin):
        _run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

    env = dict(os.environ)
    env["PATH"] = f"{os.path.dirname(uv_bin)}:{env.get('PATH', '')}"
    env["UV_NO_CONFIG"] = "1"
    cache_dir = os.environ.get("UV_CACHE_DIR")
    if not cache_dir:
        # Persist across preemption/requeue -- avoids re-downloading the
        # multi-GB torch/vllm wheels and rebuilding flash-attn from source.
        cache_dir = os.path.join(os.path.dirname(venv_dir.rstrip("/")), ".uv_cache")
        env["UV_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    _run([uv_bin, "python", "install", "3.12"], env=env)
    if not os.path.exists(python_bin):
        # Only create if missing -- a retry after this function raised further
        # down (e.g. the flash-attn build failure below) would otherwise wipe
        # an already-populated multi-GB venv and redo the whole torch/vllm
        # install from scratch.
        _run([uv_bin, "venv", "--python", "3.12", venv_dir], env=env)

    sync_env = {**env, "VIRTUAL_ENV": venv_dir}
    sync_cmd = [uv_bin, "sync", "--active"]
    _run(sync_cmd + ["--no-install-package", "flash-attn"], cwd=recipe_dir, env=sync_env)
    # UV_NO_CONFIG=1 above (needed to dodge the repo-root pyproject.toml's
    # [tool.uv].index leaking in) also blocks uv from reading THIS recipe's
    # own pyproject.toml [tool.uv] table -- including
    # `no-build-isolation-package = ["flash-attn"]`, which is exactly what
    # flash-attn needs (it imports torch at build time, and a fully-isolated
    # build env can't see the torch just installed above). Pass the
    # equivalent as an explicit CLI flag instead of relying on that config.
    _run(sync_cmd + ["--no-build-isolation-package", "flash-attn"], cwd=recipe_dir, env=sync_env)

    if not _validate(python_bin):
        raise RuntimeError(f"venv at {venv_dir} failed post-sync validation")

    _ensure_extra_packages(python_bin, extra_packages)

    with open(marker, "w") as f:
        f.write("ok\n")
    return python_bin
