# SPDX-License-Identifier: Apache-2.0
"""run.py — Lilypad entrypoint for the full J-lens fit.

Fits J_l over the driving corpus on one GPU. jlens checkpoints its running
sum after every prompt; this wrapper calls jlens.fit in chunks so it can log
progress to W&B and mirror the checkpoint to S3 between chunks — /mnt/work is
node-local and dies with the pod (same failure mode as the llm-judge step-148
crash loop), so preemption/requeue resumes from S3 instead of restarting.

Config reference (defaults):
    n_prompts:         100     # paper: ~100 prompts usable, ~1000 ideal
    use_coc:           true    # top up built-ins with real coc strings from HF
    layers:            [2, 6, 10, 14, 18, 22, 26, 30, 34]
    target_layer:      null    # default: final layer
    dim_batch:         32
    max_seq_len:       128
    skip_first:        16
    chunk_size:        5       # prompts per W&B log / S3 checkpoint sync
    outdir:            "/mnt/work/tmp/jspace"
    s3_bucket:         "research-datasets-chicago"
    results_s3_prefix: "jspace/lens_full_v1"
    wandb_project:     "jspace"
    wandb_entity:      "research"
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# jspace/src is not a package; expose load_model/prompts the way fit_lens.py does.
sys.path.insert(0, str(REPO_ROOT / "jspace" / "src"))

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "n_prompts":         100,
    "use_coc":           True,
    "layers":            [2, 6, 10, 14, 18, 22, 26, 30, 34],
    "target_layer":      None,
    "dim_batch":         32,
    "max_seq_len":       128,
    "skip_first":        16,
    "chunk_size":        5,
    "outdir":            "/mnt/work/tmp/jspace",
    "s3_bucket":         "research-datasets-chicago",
    "results_s3_prefix": "jspace/lens_full_v1",
    "wandb_project":     "jspace",
    "wandb_entity":      "research",
}

_HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
_PARQUET = "reasoning/ood_reasoning.parquet"


def _s3_client():
    import boto3
    from botocore.config import Config

    # OCI S3-compat rejects AWS chunked encoding; payload_signing_enabled +
    # put_object (never upload_file) is the working pattern from rl_posttrain.
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


def _s3_upload(s3, bucket: str, key: str, path: Path) -> None:
    with open(path, "rb") as fh:
        s3.put_object(Bucket=bucket, Key=key, Body=fh)


def _s3_try_restore(s3, bucket: str, key: str, dest: Path) -> bool:
    from botocore.exceptions import ClientError

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        for block in obj["Body"].iter_chunks(64 * 1024 * 1024):
            fh.write(block)
    tmp.rename(dest)
    return True


def fit_lens_loop(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = {**_DEFAULTS, **(training_fn_config or {})}

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    out_path, ckpt_path = outdir / "lens.pt", outdir / "lens.pt.ckpt"
    bucket = cfg["s3_bucket"]
    prefix = str(cfg["results_s3_prefix"]).rstrip("/")

    from prompts import corpus

    parquet = None
    if cfg["use_coc"]:
        from huggingface_hub import hf_hub_download

        parquet = hf_hub_download(
            repo_id=_HF_REPO, repo_type="dataset", filename=_PARQUET,
            token=os.environ.get("HF_TOKEN"),
        )
    prompt_list = corpus(parquet)
    if cfg["n_prompts"]:
        prompt_list = prompt_list[: int(cfg["n_prompts"])]

    s3 = _s3_client()
    if not ckpt_path.exists() and _s3_try_restore(s3, bucket, f"{prefix}/lens.pt.ckpt", ckpt_path):
        logger.info("restored checkpoint from s3://%s/%s/lens.pt.ckpt", bucket, prefix)

    import wandb

    run = wandb.init(
        entity=cfg["wandb_entity"], project=cfg["wandb_project"],
        config={k: v for k, v in cfg.items()},
    )
    print(f"W&B run URL: {run.url}", flush=True)

    import jlens
    from load_model import load_lens_model

    _, model = load_lens_model()

    n, chunk = len(prompt_list), int(cfg["chunk_size"])
    ends = list(range(chunk, n, chunk)) + [n]
    t0 = time.time()
    lens = None
    for end in ends:
        lens = jlens.fit(
            model,
            prompt_list[:end],
            source_layers=list(cfg["layers"]),
            target_layer=cfg["target_layer"],
            dim_batch=int(cfg["dim_batch"]),
            max_seq_len=int(cfg["max_seq_len"]),
            skip_first=int(cfg["skip_first"]),
            checkpoint_path=str(ckpt_path),
            checkpoint_every=1,
            resume=True,
        )
        _s3_upload(s3, bucket, f"{prefix}/lens.pt.ckpt", ckpt_path)
        elapsed = time.time() - t0
        # sec_per_prompt only counts this pod's wall clock; skewed after a requeue.
        wandb.log(
            {
                "prompts_done": end,
                "elapsed_s": elapsed,
                "sec_per_prompt": elapsed / end,
                "ckpt_mb": ckpt_path.stat().st_size / 1e6,
            },
            step=end,
        )
        logger.info("progress: %d/%d prompts, %.0fs elapsed", end, n, elapsed)

    lens.save(str(out_path))
    _s3_upload(s3, bucket, f"{prefix}/lens.pt", out_path)
    run.summary["total_prompts"] = n
    run.summary["lens_s3"] = f"s3://{bucket}/{prefix}/lens.pt"
    logger.info("saved lens -> %s and s3://%s/%s/lens.pt", out_path, bucket, prefix)
    wandb.finish()
