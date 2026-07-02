# SPDX-License-Identifier: Apache-2.0
"""
sample_clips.py — pick N clips per OOD scenario type (event_cluster) for the
masking analysis, restricted to clips whose shard has already been uploaded
by the build_wds job (which is still running and has not covered the full
dataset yet).

Finding "which OOD clips are already in S3" without downloading every shard
(each ~1-12GB) is done via HTTP Range reads: we walk each shard's tar headers
directly (512-byte blocks), reading only the small `{clip_id}.json` payloads
and skipping over the multi-MB camera/parquet payloads by jumping straight to
the next header using the size field. This costs a handful of small requests
per clip instead of downloading ~3.5GB/shard.

Usage:
    python -m masking.data.sample_clips \
        --bucket research-datasets-chicago \
        --prefix nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train \
        --hf_token $HF_TOKEN \
        --n_per_type 10 \
        --out masking/configs/sample_clips.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
TAR_BLOCK = 512


def load_ood_clips(hf_token: str | None) -> pd.DataFrame:
    """Download reasoning/ood_reasoning.parquet — one row per OOD clip."""
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename="reasoning/ood_reasoning.parquet",
        token=hf_token,
    )
    return pd.read_parquet(path)


def list_shards(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].split("/")[-1].startswith("shard_"):
                keys.append(obj["Key"])
    return sorted(keys)


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def scan_shard_for_clips(
    bucket: str, key: str, ood_ids: set[str]
) -> list[tuple[str, str, dict]]:
    """Walk one shard's tar headers, returning (clip_id, shard_key, json_meta)
    for every clip in ood_ids found in this shard."""
    s3 = boto3.client("s3")
    pos = 0
    zero_blocks = 0
    found: list[tuple[str, str, dict]] = []
    while True:
        hdr = _range_get(s3, bucket, key, pos, TAR_BLOCK)
        if len(hdr) < TAR_BLOCK or hdr == b"\x00" * TAR_BLOCK:
            zero_blocks += 1
            pos += TAR_BLOCK
            if zero_blocks >= 2:
                break
            continue
        zero_blocks = 0
        name = hdr[0:100].split(b"\x00")[0].decode(errors="replace")
        size_field = hdr[124:136].split(b"\x00")[0].strip()
        size = int(size_field, 8) if size_field else 0
        data_start = pos + TAR_BLOCK

        if name.endswith(".json") and "@PaxHeader" not in name and not name.endswith(
            "calibration.json"
        ):
            clip_id = name[: -len(".json")]
            if clip_id in ood_ids:
                data = _range_get(s3, bucket, key, data_start, size)
                try:
                    found.append((clip_id, key, json.loads(data.decode())))
                except Exception:
                    logger.warning("shard %s: failed to parse json for %s", key, clip_id)

        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK
    return found


def find_available_ood_clips(
    bucket: str, prefix: str, ood_ids: set[str], max_workers: int = 40
) -> list[tuple[str, str, dict]]:
    """Scan every shard under prefix, returning all OOD clips already uploaded."""
    keys = list_shards(bucket, prefix)
    logger.info("Scanning %d shards under s3://%s/%s for %d target OOD clips",
                len(keys), bucket, prefix, len(ood_ids))
    all_found: list[tuple[str, str, dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(scan_shard_for_clips, bucket, k, ood_ids): k for k in keys}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                all_found.extend(fut.result())
            except Exception as exc:
                logger.error("Failed to scan %s: %s", key, exc)
            done += 1
            if done % 50 == 0:
                logger.info("  %d/%d shards scanned, %d OOD clips found so far",
                            done, len(keys), len(all_found))
    return all_found


def sample_per_type(
    found: list[tuple[str, str, dict]],
    ood_df: pd.DataFrame,
    n_per_type: int,
    seed: int,
) -> list[dict]:
    """Group found clips by event_cluster and sample up to n_per_type each."""
    by_cluster: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    cluster_of = ood_df["event_cluster"].to_dict()
    for clip_id, shard_key, meta in found:
        by_cluster[cluster_of.get(clip_id, "UNKNOWN")].append((clip_id, shard_key, meta))

    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    for cluster in sorted(by_cluster):
        candidates = by_cluster[cluster]
        n_take = min(n_per_type, len(candidates))
        if n_take < n_per_type:
            logger.warning(
                "event_cluster=%s: only %d/%d clips available in already-uploaded shards",
                cluster, n_take, n_per_type,
            )
        idxs = rng.choice(len(candidates), size=n_take, replace=False)
        for i in idxs:
            clip_id, shard_key, meta = candidates[i]
            manifest.append({"clip_id": clip_id, "event_cluster": cluster, "shard_key": shard_key})
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="research-datasets-chicago")
    ap.add_argument(
        "--prefix",
        default="nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train",
    )
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--n_per_type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="masking/configs/sample_clips.json")
    args = ap.parse_args()

    ood_df = load_ood_clips(args.hf_token)
    ood_ids = set(ood_df.index.astype(str))

    found = find_available_ood_clips(args.bucket, args.prefix, ood_ids)
    manifest = sample_per_type(found, ood_df, args.n_per_type, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d sampled clips to %s", len(manifest), out_path)


if __name__ == "__main__":
    main()
