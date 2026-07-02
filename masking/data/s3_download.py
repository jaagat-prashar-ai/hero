# SPDX-License-Identifier: Apache-2.0
"""
s3_download.py — Download WebDataset shards from S3 to a local directory.

S3 path convention:
    s3://<bucket>/<prefix>/shard_NNNNN.tar
    s3://<bucket>/<prefix>/index.json      (optional shard manifest)

The downloader skips shards that already exist locally (resumable), lists the
bucket prefix to discover all available shards, and writes a local index.json
if one was present on S3.

Usage (standalone):
    python -m masking.data.s3_download \
        --s3_bucket my-bucket \
        --s3_prefix wds/masking \
        --local_dir /data/wds
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Placeholder S3 path — override via CLI args or the Lilypad config.
DEFAULT_S3_BUCKET = "PLACEHOLDER_BUCKET"
DEFAULT_S3_PREFIX = "PLACEHOLDER_PREFIX/wds"


def list_shards(s3_client, bucket: str, prefix: str) -> list[str]:
    """Return the S3 keys of all shard_*.tar files under bucket/prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.split("/")[-1]
            if name.startswith("shard_") and name.endswith(".tar"):  # matches both shard_00000.tar and shard_000_00000.tar
                keys.append(key)
    keys.sort()
    logger.info("Found %d shards under s3://%s/%s", len(keys), bucket, prefix)
    return keys


def download_shards(
    bucket: str,
    prefix: str,
    local_dir: str | Path,
    *,
    max_shards: int | None = None,
    num_threads: int = 4,
    only_keys: set[str] | None = None,
) -> list[Path]:
    """Download WDS shards from S3 to local_dir, skipping already-present files.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix (no trailing slash).
        local_dir: Local directory to write shards into.
        max_shards: Cap the number of shards downloaded (useful for smoke tests).
        num_threads: Concurrent download threads (boto3 TransferConfig).
        only_keys: If given, restrict to these exact S3 keys instead of listing
            and downloading everything under prefix -- e.g. the shard_key values
            from a sample_clips.json manifest, so a curated-sample run doesn't
            pull the whole (possibly still-growing) dataset.

    Returns:
        List of local shard paths in sorted order.
    """
    import concurrent.futures

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    if only_keys is not None:
        shard_keys = sorted(only_keys)
        logger.info("Restricting to %d explicit shard keys", len(shard_keys))
    else:
        shard_keys = list_shards(s3, bucket, prefix)

    if max_shards is not None:
        shard_keys = shard_keys[:max_shards]
        logger.info("Limiting to first %d shards", max_shards)

    # Try to download index.json as well
    index_key = prefix.rstrip("/") + "/index.json"
    index_local = local_dir / "index.json"
    if not index_local.exists():
        try:
            s3.download_file(bucket, index_key, str(index_local))
            logger.info("Downloaded index.json")
        except ClientError:
            logger.debug("No index.json found at s3://%s/%s", bucket, index_key)

    def _download_one(key: str) -> Path:
        # Preserve split subdirs (train/, val/, test/) so shard_00000.tar keys
        # from different splits do not overwrite each other locally.
        rel = key[len(prefix.rstrip("/") + "/") :]
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            logger.debug("Skipping %s (already present)", rel)
            return dest
        logger.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
        s3.download_file(bucket, key, str(dest))
        return dest

    local_paths: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_download_one, k): k for k in shard_keys}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                local_paths.append(fut.result())
            except Exception as exc:
                logger.error("Failed to download %s: %s", key, exc)

    local_paths.sort()
    logger.info("Download complete: %d shards in %s", len(local_paths), local_dir)
    return local_paths


def shard_paths(local_dir: str | Path) -> list[Path]:
    """Return sorted shard_*.tar files under local_dir (including split subdirs)."""
    return sorted(Path(local_dir).rglob("shard_*.tar"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download WDS shards from S3")
    ap.add_argument("--s3_bucket", default=DEFAULT_S3_BUCKET)
    ap.add_argument("--s3_prefix", default=DEFAULT_S3_PREFIX)
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--num_threads", type=int, default=4)
    args = ap.parse_args()

    paths = download_shards(
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
        local_dir=args.local_dir,
        max_shards=args.max_shards,
        num_threads=args.num_threads,
    )
    print(f"Downloaded {len(paths)} shards to {args.local_dir}")


if __name__ == "__main__":
    main()
