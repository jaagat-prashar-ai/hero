# Self-bootstrap an isolated Python 3.12 venv with AlpamayoR1's real
# dependencies, matching this project's local perplexity/alpamayo/ar1_venv
# setup exactly (same recipe: `uv python install 3.12` + `uv venv` + `uv pip
# install`) -- necessary because Lilypad's base worker environment is Python
# 3.10 and cannot run AlpamayoR1 at all (torch==2.8.0/transformers==4.57.1
# need Python 3.12; see requirements.txt's module docstring for why this
# can't just be a pip_requirements_path entry in the base env).
#
# Deliberately skips flash-attn on this first pass -- it compiles from
# source and takes 20-40+ minutes, too slow for a "get it working quickly"
# first cluster attempt. cluster_worker.py loads the model with
# attn_implementation="eager" instead (sdpa is rejected for this
# architecture by transformers 4.57.1's dispatch check -- see
# cluster_worker.load_model). Revisit adding flash-attn only if eager
# proves too slow for the real 135-clip sweep.
#
# Idempotent: writes a marker file after a successful install so a
# re-invoked job (e.g. after preemption/requeue) doesn't redo the
# multi-minute install.

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

MARKER_NAME = "BOOTSTRAP_OK"


def wait_for_alpamayo_venv(venv_dir: str, timeout_s: float = 45 * 60) -> str:
    """Wait for another local rank to finish building the venv (marker file),
    then return its python binary. See run.py: with 8 ranks per a100.8 node
    sharing one venv_dir, only local rank 0 builds -- everyone else calling
    ensure_alpamayo_venv concurrently would race eight `uv venv` recreations
    of the same directory."""
    python_bin = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, MARKER_NAME)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(marker) and os.path.exists(python_bin):
            return python_bin
        time.sleep(10)
    raise TimeoutError(f"venv at {venv_dir} was not built by local rank 0 within {timeout_s:.0f}s")


def _run(cmd: list[str], **kwargs) -> None:
    logger.info("bootstrap: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_alpamayo_venv(venv_dir: str, perplexity_dir: str) -> str:
    """Build (once) the venv, return the path to its python binary.

    Args:
        venv_dir: where to build the private Python 3.12 venv.
        perplexity_dir: absolute path to this repo's perplexity/ directory
            (code_assets copies the whole repo to the pod, so
            perplexity/alpamayo -- AlpamayoR1's own pyproject.toml package --
            exists on disk at job runtime and can be pip-installed from
            there directly, same as this workstation's local ar1_venv).
    """
    python_bin = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, MARKER_NAME)
    if os.path.exists(marker) and os.path.exists(python_bin):
        logger.info("bootstrap: venv already built at %s, skipping", venv_dir)
        return python_bin

    uv_bin = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv_bin):
        _run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

    env = dict(os.environ)
    env["PATH"] = f"{os.path.dirname(uv_bin)}:{env.get('PATH', '')}"
    # Ignore uv config discovered from the CWD: the repo-root pyproject.toml
    # defines [tool.uv].index = [ursa, pytorch-cu128, pypi.nvidia.com] (for
    # the Bazel/lockfile toolchain) with NO pypi.org, and uv's
    # first-index-wins strategy then caps e.g. packaging at the cu128
    # index's 24.1 -- which broke hatchling resolution inside the editable
    # install's build env (killed canary6, reproduced locally). This venv's
    # packages come from explicit --index-url flags or the pypi.org default,
    # deterministically, regardless of where the bootstrap runs from.
    env["UV_NO_CONFIG"] = "1"

    _run([uv_bin, "python", "install", "3.12"], env=env)
    _run([uv_bin, "venv", "--python", "3.12", venv_dir], env=env)

    pip_install = [uv_bin, "pip", "install", "--python", python_bin]

    # torch/torchvision -- pinned to AlpamayoR1's own pyproject.toml version,
    # cu128 index to match the a100.8 machine type's CUDA driver.
    _run(
        pip_install
        + [
            "torch==2.8.0",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        env=env,
    )

    # Everything else AlpamayoR1/its data loaders need, same pins as
    # perplexity/alpamayo/pyproject.toml plus s3_clip_loader.py's own real
    # imports (av, physical_ai_av, scipy, pandas) and our plotting/upload
    # needs (matplotlib, boto3).
    _run(
        pip_install
        + [
            "transformers==4.57.1",
            "einops>=0.8.1",
            "hydra-core>=1.3.2",
            "huggingface_hub>=0.23",
            "scipy",
            "colorlog>=6.0.0",
            "pillow>=12.0.0",
            "physical_ai_av>=0.2.0",
            "av>=16.0.1",
            "pandas>=2.0",
            "boto3>=1.34",
            "matplotlib",
        ],
        env=env,
    )

    # Build backend for the editable install below, into the venv itself.
    # Pinned to an explicit pypi.org index: the pod's ambient index config
    # resolves first-index-wins against the pytorch cu128 index, which caps
    # packaging at 24.1 and can't satisfy hatchling's packaging>=24.2 --
    # that resolution failure inside uv's isolated build env is what killed
    # canary6. (editables is hatchling's build-time requirement for editable
    # wheels specifically.)
    _run(
        pip_install
        + ["--index-url", "https://pypi.org/simple", "hatchling>=1.27.0", "editables"],
        env=env,
    )

    # AlpamayoR1 itself, editable, from this same repo checkout.
    # --no-deps actually implements the "skip flash-attn" promise above:
    # alpamayo-r1's pyproject.toml declares flash-attn>=2.8.3, and without
    # the flag uv resolves it and tries to BUILD it -- which fails outright
    # (flash-attn's sdist needs torch at build time; killed canary5), and
    # even if it built it would compile from source for 20-40+ min. Every
    # runtime dep alpamayo-r1 actually needs is already installed explicitly
    # above (the local ar1_venv only survived the with-deps -e install
    # because flash-attn was already present there).
    # --no-build-isolation makes the build use the venv's own hatchling
    # (installed just above) instead of resolving a fresh isolated build env
    # against the pod's broken index order -- combined with --no-deps, this
    # step now touches no package index at all.
    _run(
        pip_install
        + ["--no-deps", "--no-build-isolation", "-e", os.path.join(perplexity_dir, "alpamayo")],
        env=env,
    )

    with open(marker, "w") as f:
        f.write("ok\n")
    return python_bin
