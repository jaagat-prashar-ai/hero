# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad generic entrypoint for the alpamayo1_x_rl local-test RL
post-training run (Cosmos-RL + GRPO), vendored from NVlabs/alpamayo-recipes
at third_party/alpamayo-recipes.

This runs the recipe's own "Getting started" single-node local-test steps
end-to-end on one Lilypad a100.8 node (8 GPUs). workload_type: "generic"
runs entrypoint_fn directly on the Ray HEAD node/pod (Lilypad's
GenericWorkloadRunner just calls it synchronously, no GPU-aware scheduling)
-- and that head pod has NO GPU/driver passthrough in this cluster (confirmed
2026-07-16 on run alpamayo-rl-local-test-6kfkvp: `nvidia-smi` missing,
`libcuda.so.1` missing, vLLM/Triton both report zero GPUs). The real 8 GPUs
requested via cluster_resources go to a separate Ray WORKER pod instead. So
the actual work (venv bootstrap onward) is wrapped in a
`@ray.remote(num_gpus=...)` task and dispatched via `ray.get(...)` from the
plain entrypoint, so Ray schedules it onto that GPU worker pod. This does
NOT use Lilypad's per-rank training_fn replica convention (which would start
N independent copies, one per GPU) -- `cosmos-rl` itself fans out to all of
a single node's GPUs via its own controller/policy/rollout subprocess model,
so this needs to run as one process on one GPU-attached node, not N replicas.

  1. Bootstrap the recipe's isolated Python 3.12 uv venv (see
     bootstrap_venv.py) -- idempotent, persisted under workspace_dir so a
     preempted/requeued job doesn't redo the flash-attn build.
  2. Convert the released HF checkpoint into a training-ready checkpoint
     (scripts/convert_release_config_to_training.py) -- idempotent, skipped
     if the output dir already has a config.json.
  3. Download + curate a mini Physical AI (PAI) dataset subset
     (scripts/download_pai.py + scripts/curate_pai_samples.py) -- idempotent,
     skipped if the curated parquet already exists.
  4. Patch the recipe's local-test TOML (output_dir, model path, W&B
     naming) into a scratch copy -- never edits the vendored submodule.
  5. Launch `cosmos-rl --policy 1 --rollout 1 ...` as a streamed subprocess.

Full config reference (all keys optional, defaults shown):
    workspace_dir:      "/mnt/work/tmp/alpamayo_rl_job"  # persistent scratch root
    alpamayo_model:     "nvidia/Alpamayo-1.5-10B"        # or "nvidia/Alpamayo-R1-10B"
    pai_chunk_ids:      "3116"                           # download_pai.py --chunk-ids
    pai_num_samples:    16                                # clips curated for the mini subset
    pai_seed:           0
    dp_shard_size:      4       # [policy.parallelism].dp_shard_size (4 for 1 a100.8 node)
    train_epoch:         15      # [train].epoch
    wandb_project:      "alpamayo-rl"
    wandb_experiment:   "reasoning_vla_local_test"
    wandb_entity:       "research"
    reasoning:          false   # use the joint reasoning-motion reward variant
    num_gpus:           8       # GPUs to request for the ray.remote GPU-node task
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import ray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "third_party" / "alpamayo-recipes"
RECIPE_DIR = RECIPE_ROOT / "recipes" / "alpamayo1_x_rl"

_DEFAULTS: dict[str, Any] = {
    "workspace_dir": "/mnt/work/tmp/alpamayo_rl_job",
    "alpamayo_model": "nvidia/Alpamayo-1.5-10B",
    "pai_chunk_ids": "3116",
    "pai_num_samples": 16,
    "pai_seed": 0,
    "dp_shard_size": 4,
    "train_epoch": 15,
    "wandb_project": "alpamayo-rl",
    "wandb_experiment": "reasoning_vla_local_test",
    "wandb_entity": "research",
    "reasoning": False,
    "num_gpus": 8,
}

_CAMERA_SUBPARTS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
_CALIBRATION_SUBPARTS = ["camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"]


def _run_streamed(cmd: list[str], **kwargs) -> None:
    """Run a subprocess with stdout/stderr flowing through to this pod's own
    log stream (not captured) so `lilypad workload logs` sees it live."""
    logger.info("run: %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _convert_model(python_bin: str, model_dir: Path, alpamayo_model: str) -> None:
    if (model_dir / "config.json").exists():
        logger.info("model conversion: %s already has config.json, skipping", model_dir)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            python_bin,
            "scripts/convert_release_config_to_training.py",
            "--output-dir",
            str(model_dir),
            "--alpamayo-model",
            alpamayo_model,
        ],
        cwd=RECIPE_ROOT,
    )


def _download_pai(
    python_bin: str,
    pai_dir: Path,
    chunk_ids: str,
    num_samples: int,
) -> Path:
    mini_path = pai_dir / "clip_index_mini.parquet"
    if mini_path.exists():
        logger.info("PAI dataset: %s already exists, skipping download+curate", mini_path)
        return mini_path

    pai_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            python_bin,
            "scripts/download_pai.py",
            "--chunk-ids",
            chunk_ids,
            "--camera",
            *_CAMERA_SUBPARTS,
            "--calibration",
            *_CALIBRATION_SUBPARTS,
            "--labels",
            "egomotion",
            "--output-dir",
            str(pai_dir),
        ],
        cwd=RECIPE_ROOT,
    )
    _run_streamed(
        [
            python_bin,
            "scripts/curate_pai_samples.py",
            "--clip-index-path",
            str(pai_dir / "clip_index.parquet"),
            "--chunk",
            chunk_ids,
            "--num-samples",
            str(num_samples),
            "--output-path",
            str(mini_path),
        ],
        cwd=RECIPE_ROOT,
    )
    return mini_path


def _patch_toml(
    template_path: Path,
    output_path: Path,
    *,
    output_dir: Path,
    model_dir: Path,
    dp_shard_size: int,
    epoch: int,
    wandb_project: str,
    wandb_experiment: str,
) -> None:
    import tomlkit

    doc = tomlkit.parse(template_path.read_text())
    doc["train"]["output_dir"] = str(output_dir)
    doc["train"]["epoch"] = epoch
    doc["policy"]["model_name_or_path"] = str(model_dir)
    doc["policy"]["parallelism"]["dp_shard_size"] = dp_shard_size
    doc["logging"]["project_name"] = wandb_project
    doc["logging"]["experiment_name"] = wandb_experiment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tomlkit.dumps(doc))


def _launch_cosmos_rl(
    python_bin: str,
    toml_path: Path,
    entry_script: Path,
    log_dir: Path,
    env: dict[str, str],
) -> None:
    cosmos_rl_bin = str(Path(python_bin).parent / "cosmos-rl")
    log_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            cosmos_rl_bin,
            "--config",
            str(toml_path),
            "--policy",
            "1",
            "--rollout",
            "1",
            "--log-dir",
            str(log_dir),
            str(entry_script),
        ],
        cwd=RECIPE_ROOT,
        env=env,
    )


def _run_on_gpu_node(cfg: dict[str, Any]) -> None:
    """Runs on a Ray worker with real GPUs attached (see module docstring for why
    this can't just run inline in the plain `generic` entrypoint on the head)."""
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in the environment (gated PAI dataset + model)")

    workspace_dir = Path(cfg["workspace_dir"])
    venv_dir = workspace_dir / "venv"
    model_dir = workspace_dir / "alpamayo_model_converted_from_hf"
    pai_dir = workspace_dir / "PAI_mini"
    log_dir = workspace_dir / "logs"
    hf_home = workspace_dir / ".cache" / "huggingface"
    train_output_dir = workspace_dir / "outputs"
    toml_out = workspace_dir / "toml" / "alpamayo_rvla_rl_local_test.local.toml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    logger.info("rl_posttrain: workspace_dir=%s", workspace_dir)

    from rl_posttrain.training.bootstrap_venv import ensure_recipe_venv

    python_bin = ensure_recipe_venv(str(venv_dir), str(RECIPE_DIR))

    _convert_model(python_bin, model_dir, cfg["alpamayo_model"])

    _download_pai(python_bin, pai_dir, str(cfg["pai_chunk_ids"]), int(cfg["pai_num_samples"]))

    reasoning = bool(cfg["reasoning"])
    template_name = (
        "alpamayo_rvla_rl_local_test_with_reasoning.toml"
        if reasoning
        else "alpamayo_rvla_rl_local_test.toml"
    )
    entry_name = (
        "alpamayo_cosmos_rl_post_training_reasoning_entry.py"
        if reasoning
        else "alpamayo_cosmos_rl_post_training_entry.py"
    )
    template_path = RECIPE_DIR / "toml" / template_name
    entry_script = RECIPE_DIR / "models" / "reasoning_vla" / entry_name

    _patch_toml(
        template_path,
        toml_out,
        output_dir=train_output_dir,
        model_dir=model_dir,
        dp_shard_size=int(cfg["dp_shard_size"]),
        epoch=int(cfg["train_epoch"]),
        wandb_project=cfg["wandb_project"],
        wandb_experiment=cfg["wandb_experiment"],
    )

    subprocess_env = dict(os.environ)
    subprocess_env.update(
        {
            "ALPAMAYO_WORKSPACE": str(RECIPE_ROOT),
            "ALPAMAYO_MODEL_DIR": str(model_dir),
            "ALPAMAYO_PAI_LOCAL_DIR": str(pai_dir),
            "ALPAMAYO_LOG_DIR": str(log_dir),
            "HF_HOME": str(hf_home),
            "WANDB_ENTITY": cfg["wandb_entity"],
        }
    )
    if hf_token:
        subprocess_env["HF_TOKEN"] = hf_token
    if wandb_key:
        subprocess_env["WANDB_API_KEY"] = wandb_key

    _launch_cosmos_rl(python_bin, toml_out, entry_script, log_dir, subprocess_env)

    logger.info("rl_posttrain: cosmos-rl run finished, checkpoints under %s", train_output_dir)


def rl_local_test_loop(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    """Lilypad-compatible generic entrypoint: rl_posttrain.training.run.rl_local_test_loop.

    Runs on the Ray head node (no GPUs here -- see module docstring). Does
    only cheap validation itself, then dispatches the actual work to a
    ray.remote task with num_gpus set so Ray schedules it onto the real
    GPU worker node.
    """
    cfg = {**_DEFAULTS, **training_fn_config}

    if not RECIPE_DIR.is_dir():
        raise RuntimeError(
            "Missing alpamayo-recipes vendor source at third_party/alpamayo-recipes -- run: "
            "git submodule update --init third_party/alpamayo-recipes"
        )
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required in the environment (gated PAI dataset + model)")

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    remote_fn = ray.remote(_run_on_gpu_node).options(num_gpus=int(cfg["num_gpus"]))
    ray.get(remote_fn.remote(cfg))
