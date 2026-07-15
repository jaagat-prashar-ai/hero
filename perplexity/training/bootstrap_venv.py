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
# attn_implementation="sdpa" instead (a standard transformers override), not
# flash_attention_2. Revisit adding flash-attn only if sdpa proves too slow
# for the real 135-clip sweep.
#
# Idempotent: writes a marker file after a successful install so a
# re-invoked job (e.g. after preemption/requeue) doesn't redo the
# multi-minute install.

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

MARKER_NAME = "BOOTSTRAP_OK"


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

    # AlpamayoR1 itself, editable, from this same repo checkout. --no-deps
    # is what actually implements the "skip flash-attn" promise above:
    # alpamayo-r1's pyproject.toml declares flash-attn>=2.8.3, and without
    # the flag uv resolves it and tries to BUILD it -- which fails outright
    # (flash-attn's sdist needs torch at build time and uv's isolated build
    # env doesn't have it; killed canary5), and even with build isolation
    # worked around it would compile from source for 20-40+ min. Every
    # runtime dep alpamayo-r1 actually needs is already installed explicitly
    # by the two steps above (the local ar1_venv only survived the -e
    # install with deps because flash-attn was already present there).
    _run(pip_install + ["--no-deps", "-e", os.path.join(perplexity_dir, "alpamayo")], env=env)

    with open(marker, "w") as f:
        f.write("ok\n")
    return python_bin
