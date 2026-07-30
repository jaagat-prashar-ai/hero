# SPDX-License-Identifier: Apache-2.0
"""
bootstrap_venv.py — self-bootstraps an isolated Python 3.12 venv with
Alpamayo 1.5's real dependencies (torch==2.8.0, transformers==4.57.1,
physical-ai-av==0.2.0, ...), because Lilypad's base worker environment is
Python 3.10 and cannot import physical_ai_av (needs >= 3.11) or alpamayo1_5
(needs == 3.12) at all.

Copied structure from perplexity/training/bootstrap_venv.py (the existing,
cluster-validated recipe for exactly this problem with AlpamayoR1) with one
substitution: the editable install target is third_party/alpamayo1.5 (this
package's own pyproject.toml) instead of perplexity/alpamayo. Deliberately
skips flash-attn for the same reason perplexity's bootstrap does -- it
compiles from source (20-40+ min) and worker.py's load_model() passes
attn_implementation="eager" instead.

Idempotent: writes a marker file after a validated install so a re-invoked
job (e.g. after preemption/requeue) doesn't redo the multi-minute install.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

MARKER_NAME = "BOOTSTRAP_OK"
_VALIDATE_IMPORTS = "import torch, physical_ai_av, alpamayo1_5"


def _run(cmd: list[str], **kwargs) -> None:
    logger.info("bootstrap: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _validate(python_bin: str) -> bool:
    check = subprocess.run([python_bin, "-c", _VALIDATE_IMPORTS], capture_output=True, text=True)
    return check.returncode == 0


def ensure_alpamayo15_venv(venv_dir: str, repo_root: str) -> str:
    """Build (once) the venv, return the path to its python binary.

    Args:
        venv_dir: where to build the private Python 3.12 venv.
        repo_root: absolute path to this repo's root (code_assets copies the
            whole repo to the pod, so third_party/alpamayo1.5 -- Alpamayo
            1.5's own pyproject.toml package -- exists on disk at job
            runtime and can be pip-installed from there directly).
    """
    python_bin = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, MARKER_NAME)
    if os.path.exists(marker) and os.path.exists(python_bin):
        # The marker alone isn't proof of a working venv -- see
        # perplexity/training/bootstrap_venv.py's identical comment for the
        # concurrent-rebuild corpse scenario this validates against.
        if _validate(python_bin):
            logger.info("bootstrap: venv already built at %s, validated, skipping", venv_dir)
            return python_bin
        logger.warning("bootstrap: venv at %s has a marker but failed validation, rebuilding", venv_dir)
        shutil.rmtree(venv_dir, ignore_errors=True)

    uv_bin = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv_bin):
        _run(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])

    env = dict(os.environ)
    env["PATH"] = f"{os.path.dirname(uv_bin)}:{env.get('PATH', '')}"
    # Same rationale as perplexity/training/bootstrap_venv.py: the repo-root
    # pyproject.toml's [tool.uv].index has no pypi.org entry, and uv's
    # first-index-wins strategy would otherwise cap unrelated packages at
    # whatever version the cu128/nvidia indexes happen to carry.
    env["UV_NO_CONFIG"] = "1"

    _run([uv_bin, "python", "install", "3.12"], env=env)
    _run([uv_bin, "venv", "--python", "3.12", venv_dir], env=env)

    pip_install = [uv_bin, "pip", "install", "--python", python_bin]

    # torch/torchvision pinned to alpamayo1.5's pyproject.toml versions,
    # cu128 index to match the cluster's CUDA driver.
    _run(
        pip_install + ["torch==2.8.0", "torchvision>=0.23.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
        env=env,
    )

    # Everything else alpamayo1_5 needs EXCEPT flash-attn (module docstring).
    _run(
        pip_install
        + [
            "transformers==4.57.1",
            "accelerate>=1.12.0",
            "einops>=0.8.1",
            "hydra-core>=1.3.2",
            "hydra-colorlog>=1.2.0",
            "huggingface_hub>=0.23",
            "physical-ai-av==0.2.0",
            "av>=16.0.1",
            "pandas>=2.3.3",
            "pillow>=12.0.0",
            "matplotlib>=3.10.7",
            "seaborn>=0.13.2",
            "boto3>=1.34",
        ],
        env=env,
    )

    # Build backend for the editable install below -- pinned to an explicit
    # pypi.org index for the same reason as perplexity's bootstrap (the
    # ambient pod index order can't satisfy hatchling's packaging pin).
    _run(pip_install + ["--index-url", "https://pypi.org/simple", "hatchling>=1.27.0", "editables"], env=env)

    # alpamayo1_5 itself, editable, from this same repo checkout. --no-deps
    # is what actually implements "skip flash-attn": its pyproject.toml
    # declares flash-attn>=2.8.3 and would otherwise try to BUILD it.
    # --no-build-isolation uses the venv's own hatchling instead of
    # resolving a fresh isolated build env against the pod's broken index
    # order.
    _run(
        pip_install
        + ["--no-deps", "--no-build-isolation", "-e", os.path.join(repo_root, "third_party", "alpamayo1.5")],
        env=env,
    )

    with open(marker, "w") as f:
        f.write("ok\n")
    return python_bin
