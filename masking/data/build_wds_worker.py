# SPDX-License-Identifier: Apache-2.0
"""
build_wds_worker.py — Lilypad training_fn entrypoint for distributed WDS sharding.

Each Lilypad replica reads its RANK / WORLD_SIZE from the environment, then
delegates to build_webdataset.main() to process its assigned slice of clips.

Lilypad cluster config key (masking/configs/build_wds_cluster.yaml):
    training_fn: masking.data.build_wds_worker.build_wds_loop

Required training_fn_config keys:
    bucket          S3 bucket name
    prefix          S3 key prefix (default: "physicalai-av/wds")
    hf_token        HuggingFace token (or set HF_TOKEN env var)

Optional training_fn_config keys:
    workers         Local threads per node (default 8)
    clips_per_shard Clips packed per tar shard (default 50)
    splits          List of splits to build (default [train, val, test])
    max_clips       Cap clips per split — for smoke tests
    resume_file     Path to file of already-completed clip IDs
    skip_metadata_upload  If true, skip uploading metadata parquets (default false)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def build_wds_loop(
    training_fn_config: dict[str, Any],
    experiment_tracker: Any,
) -> None:
    """Lilypad-compatible entrypoint for distributed WDS sharding."""
    cfg = training_fn_config

    # Rank is injected by Lilypad when num_replicas > 1; fall back to config.
    rank       = int(os.environ.get("RANK",       cfg.get("rank",       0)))
    world_size = int(os.environ.get("WORLD_SIZE", cfg.get("world_size", 1)))

    bucket = cfg.get("bucket")
    if not bucket:
        raise ValueError("training_fn_config.bucket is required")

    prefix          = cfg.get("prefix", "physicalai-av/wds")
    hf_token        = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    workers         = int(cfg.get("workers", 8))
    clips_per_shard = int(cfg.get("clips_per_shard", 50))
    splits          = cfg.get("splits", ["train", "val", "test"])
    max_clips       = cfg.get("max_clips")
    resume_file     = cfg.get("resume_file")
    skip_meta       = bool(cfg.get("skip_metadata_upload", False))

    argv = [
        "build_webdataset",
        "--bucket",          bucket,
        "--prefix",          prefix,
        "--workers",         str(workers),
        "--clips_per_shard", str(clips_per_shard),
        "--splits",          *splits,
        "--rank",            str(rank),
        "--world_size",      str(world_size),
    ]

    if hf_token:
        argv += ["--hf_token", hf_token]
    if max_clips is not None:
        argv += ["--max_clips", str(max_clips)]
    if resume_file:
        argv += ["--resume_file", resume_file]
    if skip_meta:
        argv.append("--skip_metadata_upload")

    logger.info("build_wds_worker: rank=%d world_size=%d  argv=%s",
                rank, world_size, " ".join(argv[1:]))

    sys.argv = argv
    from masking.data.build_webdataset import main
    main()
