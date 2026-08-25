"""Lilypad entrypoint: consolidate DeepSpeed-sharded SimLingo checkpoints into
single-file pytorch_model.pt, matching the manual step previously done for the
k2 arms (simlingo-checkpoints-consolidated/20260817090017_full_k2_w0.5_s*).

Execution model: CPU-only, no CARLA/GPU needed, single rank. Sessions in
cfg["sessions"] are processed sequentially -- each get_fp32_state_dict_from_zero_checkpoint
call materializes a full fp32 state dict in RAM, so running several in
parallel on one node risks an unbounded memory spike for no real benefit
(this job is I/O-bound, not compute-bound).
"""
import os
import shutil
from pathlib import Path
from typing import Any

import boto3
import torch
from botocore.config import Config
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def _s3_client():
    # OCI's S3-compat endpoint rejects s3transfer's chunked encoding
    # ("NotImplemented") -- same bug/fix as eval_b2d.py / s3_checkpoint.py
    # (BUGS.md 2026-07-01). Callers must use put_object, never upload_file.
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


def _download_ckpt_dir(s3, bucket: str, prefix: str, dest: Path) -> None:
    prefix = prefix.rstrip("/") + "/"
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    if not keys:
        raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
    for i, key in enumerate(keys):
        rel = key[len(prefix):]
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[consolidate] fetching ({i + 1}/{len(keys)}) {key}", flush=True)
        s3.download_file(bucket, key, str(local))


def consolidate_checkpoint(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    cfg = training_fn_config
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        print(f"[consolidate] rank {rank} idle (single-rank job)", flush=True)
        return

    bucket = cfg["s3_bucket"]
    epoch = cfg.get("checkpoint_epoch", "epoch=002")
    src_prefix_root = cfg.get("src_prefix", "simlingo-checkpoints").rstrip("/")
    dst_prefix_root = cfg.get("dst_prefix", "simlingo-checkpoints-consolidated").rstrip("/")
    root_workdir = Path(cfg.get("workdir", "/mnt/work/simlingo_consolidate"))
    s3 = _s3_client()

    for session in cfg["sessions"]:
        workdir = root_workdir / session
        workdir.mkdir(parents=True, exist_ok=True)

        # raw (sharded) layout has no "checkpoints/" subdir before the epoch
        # dir, unlike the consolidated destination layout below
        src_prefix = f"{src_prefix_root}/{session}/{epoch}.ckpt"
        local_ckpt_dir = workdir / "ckpt"
        print(f"[consolidate:{session}] mirroring s3://{bucket}/{src_prefix} -> {local_ckpt_dir}", flush=True)
        _download_ckpt_dir(s3, bucket, src_prefix, local_ckpt_dir)

        print(f"[consolidate:{session}] running get_fp32_state_dict_from_zero_checkpoint", flush=True)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(local_ckpt_dir))

        out_local = workdir / "pytorch_model.pt"
        torch.save(state_dict, out_local)
        print(f"[consolidate:{session}] saved {out_local} ({out_local.stat().st_size / 1e9:.2f} GB)", flush=True)
        del state_dict

        dest_key = f"{dst_prefix_root}/{session}/checkpoints/{epoch}.ckpt/pytorch_model.pt"
        print(f"[consolidate:{session}] uploading -> s3://{bucket}/{dest_key}", flush=True)
        with open(out_local, "rb") as fh:
            s3.put_object(Bucket=bucket, Key=dest_key, Body=fh.read())
        print(f"[consolidate:{session}] done", flush=True)

        # free disk before the next session's download
        shutil.rmtree(local_ckpt_dir, ignore_errors=True)
        out_local.unlink(missing_ok=True)
