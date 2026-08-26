# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint for AutoVLA's counterfactual-CoT pilot generation.

Qwen2.5-VL-72B-Instruct (bf16, ~144GB) doesn't fit on one A100-80GB, so this
can't use the "1 GPU per rank" pattern simlingo_lilypad/run.py or
counterfactual/run.py use (each pins one rank's subprocess to exactly one
GPU via CUDA_VISIBLE_DEVICES). Instead this follows rl_posttrain/training/
run.py's validated `workload_type: generic` pattern: the entrypoint itself
runs on the Ray head node (no GPUs there), and dispatches PARALLEL
ray.remote tasks, each requesting gpus_per_worker GPUs, so Ray schedules
each onto a real GPU node with that many GPUs visible to it -- letting
transformers' device_map="auto" shard its own 72B model instance across just
those GPUs. num_workers x gpus_per_worker should equal the job's total GPU
count (default 4 x 2 = 8): running one model instance across all 8 GPUs (an
earlier version of this file) has NO cross-scene parallelism -- all 8 GPUs
just work together on one generation at a time, which a real cluster run
(autovla-counterfactual-pilot-7kq0x9) showed extends wall-clock a lot for no
throughput benefit, since 144GB comfortably fits in 2 GPUs' 160GB already.
4 independent 2-GPU workers process disjoint scene shards concurrently
instead, via generate_counterfactual_cot.py's existing hashlib.md5-based
--rank/--world_size sharding (see its _scene_owner).

outdir is node-local (/mnt/work/...) and not reliably reachable from the
submitting workstation after the job runs (same caveat documented in
counterfactual/run.py and masking/pref_pairs' cluster runs), so this uploads
every produced JSON to S3 itself rather than relying on log-line scraping --
unlike the token-sweep experiment, we actually need the output files back to
build a training dataset from them.

DATA SOURCES (none of this touches /media/training_data -- confirmed absent
on Lilypad cluster nodes via a real run's AssertionError):
  - nuScenes trainval: research-datasets-chicago/nuscenes/ on S3 mirrors the
    exact NuScenes devkit layout. generate_counterfactual_cot.py downloads
    metadata tables + only the specific camera images it touches, lazily.
  - Qwen2.5-VL-3B/72B-Instruct: public HF models, not found on S3 (checked
    research-datasets-chicago's hf_downloads/, models/huggingface/,
    model_checkpoints/*/huggingface/) -- passed as HF repo ids, downloaded via
    from_pretrained + HF_TOKEN.
  - AutoVLA SFT checkpoint: also not on S3 -- it's Zewei-Zhou/AutoVLA's
    AutoVLA_PDMS_89.ckpt on HF hub (per the AutoVLA README's checkpoint-release
    note). Downloaded here via hf_hub_download and passed to
    generate_counterfactual_cot.py via --sft_checkpoint_override.

Full config reference (all keys optional except where noted):
    nuscenes_version:   "v1.0-trainval"
    sft_config:         "config/training/qwen2.5-vl-3B-nuplan-cluster-pilot-eval.yaml"
    sft_checkpoint_repo: "Zewei-Zhou/AutoVLA"
    sft_checkpoint_file: "AutoVLA_PDMS_89.ckpt"
    num_scenes:         300   # job-wide total, split evenly across num_workers
    k_counterfactuals:  4
    max_new_tokens:     128
    seed:               0
    num_workers:        4     # parallel ray.remote tasks
    gpus_per_worker:    2     # GPUs visible to each worker's device_map="auto"
    workdir:            "/mnt/work/autovla_counterfactual_pilot"
    s3_bucket:          "research-datasets-chicago"   # required
    s3_prefix:          required, e.g. "autovla/contrastive_pilot/nuscenes_train"
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "nuscenes_version": "v1.0-trainval",
    "sft_config": "config/training/qwen2.5-vl-3B-nuplan-cluster-pilot-eval.yaml",
    "sft_checkpoint_repo": "Zewei-Zhou/AutoVLA",
    "sft_checkpoint_file": "AutoVLA_PDMS_89.ckpt",
    "num_scenes": 300,
    "k_counterfactuals": 4,
    "max_new_tokens": 128,
    "seed": 0,
    "num_workers": 4,
    "gpus_per_worker": 2,
    "workdir": "/mnt/work/autovla_counterfactual_pilot",
}


def _s3_client():
    import boto3
    from botocore.config import Config

    # Same OCI-compat workaround as rl_posttrain/training/run.py's
    # _pai_cache_client: OCI's S3 endpoint rejects AWS chunked encoding, so
    # payload_signing_enabled=True + put_object (never multipart upload_file).
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"payload_signing_enabled": True},
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def _upload_results(local_dir: Path, bucket: str, prefix: str) -> int:
    s3 = _s3_client()
    n = 0
    for path in sorted(local_dir.glob("*.json")):
        key = f"{prefix.rstrip('/')}/{path.name}"
        with open(path, "rb") as fh:
            s3.put_object(Bucket=bucket, Key=key, Body=fh)
        n += 1
    return n


def _download_checkpoint(cfg: dict[str, Any], local_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=cfg["sft_checkpoint_repo"],
        filename=cfg["sft_checkpoint_file"],
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    return local_path


def _run_on_gpu_node(cfg: dict[str, Any], rank: int, world_size: int, per_worker_num_scenes: int) -> int:
    """Runs on a Ray worker with gpus_per_worker real GPUs attached -- see
    module docstring for why this can't just run inline in the generic
    entrypoint. Owns scene shard `rank` of `world_size` (see
    generate_counterfactual_cot.py's _scene_owner)."""
    rank_dir = Path(cfg["workdir"]) / f"rank{rank}"
    local_outdir = rank_dir / "output"
    local_outdir.mkdir(parents=True, exist_ok=True)
    local_nuscenes_dir = rank_dir / "nuscenes"

    print(f"[autovla-counterfactual] rank {rank} downloading SFT checkpoint from HF hub "
          f"({cfg['sft_checkpoint_repo']}/{cfg['sft_checkpoint_file']})...", flush=True)
    checkpoint_path = _download_checkpoint(cfg, rank_dir)

    repo_root = Path(__file__).resolve().parents[1]
    autovla_root = repo_root / "autovla" / "AutoVLA"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{autovla_root}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "tools/preprocessing/generate_counterfactual_cot.py",
        "--nuscenes_path", str(local_nuscenes_dir),
        "--nuscenes_version", cfg["nuscenes_version"],
        "--sft_config", cfg["sft_config"],
        "--sft_checkpoint_override", checkpoint_path,
        "--output_dir", str(local_outdir),
        "--num_scenes", str(per_worker_num_scenes),
        "--k_counterfactuals", str(cfg["k_counterfactuals"]),
        "--max_new_tokens", str(cfg["max_new_tokens"]),
        "--seed", str(cfg["seed"]),
        "--device", "cuda:0",
        "--rank", str(rank),
        "--world_size", str(world_size),
        "--verbose",
    ]
    print(f"[autovla-counterfactual] rank {rank}/{world_size} launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=autovla_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"generate_counterfactual_cot.py exited with code {result.returncode} on rank {rank}")

    n_uploaded = _upload_results(local_outdir, cfg["s3_bucket"], cfg["s3_prefix"])
    print(f"[autovla-counterfactual] rank {rank} uploaded {n_uploaded} scene JSONs to "
          f"s3://{cfg['s3_bucket']}/{cfg['s3_prefix']}", flush=True)
    return n_uploaded


def generate_counterfactual_cot_loop(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    """Lilypad-compatible generic entrypoint: autovla_lilypad.run.generate_counterfactual_cot_loop.
    Runs on the Ray head node (no GPUs here) and dispatches num_workers parallel
    ray.remote tasks, each with gpus_per_worker GPUs, so Ray schedules each onto
    the GPU node with only that many GPUs visible to it."""
    import ray

    cfg = {**_DEFAULTS, **training_fn_config}
    if "s3_bucket" not in cfg or "s3_prefix" not in cfg:
        raise ValueError("training_fn_config must set s3_bucket and s3_prefix")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required in the environment")

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    world_size = int(cfg["num_workers"])
    per_worker_num_scenes = max(1, -(-int(cfg["num_scenes"]) // world_size))

    remote_fn = ray.remote(_run_on_gpu_node).options(num_gpus=int(cfg["gpus_per_worker"]))
    futures = [
        remote_fn.remote(cfg, rank, world_size, per_worker_num_scenes)
        for rank in range(world_size)
    ]
    counts = ray.get(futures)
    print(f"[autovla-counterfactual] all {world_size} workers done, "
          f"{sum(counts)} scene JSONs uploaded total", flush=True)
