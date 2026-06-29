#!/usr/bin/env python3
"""
Stream-transfer nvidia/PhysicalAI-Autonomous-Vehicles from HuggingFace to OCI S3.
Downloads each file directly into S3 without full local storage.

Usage:
    python3 transfer_hf_to_oci.py [--shards N] [--workers W] [--shard-id S] [--dry-run]

    --shards N    : total number of shards (default: 8)
    --workers W   : parallel workers per process (default: 4)
    --shard-id S  : which shard this process handles (0-indexed, default: all)
    --dry-run     : list files that would be transferred without doing it
    --resume      : skip files already present in the destination bucket
"""

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import requests
from huggingface_hub import HfApi, hf_hub_url, get_token

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"
DEST_BUCKET = "research-datasets-chicago"
DEST_PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles"
AWS_PROFILE = "oci.chi"
OCI_ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB streaming chunks

# Files to skip
SKIP_FILES = {".gitattributes"}


def get_s3_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("s3", endpoint_url=OCI_ENDPOINT)


def get_all_files():
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    return [f for f in files if f not in SKIP_FILES]


def already_exists(s3, key):
    try:
        s3.head_object(Bucket=DEST_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception:
        return False


def stream_upload(s3, hf_token, filepath, resume=False):
    """Download from HF and upload to OCI S3 using multipart upload."""
    key = f"{DEST_PREFIX}/{filepath}"
    if resume and already_exists(s3, key):
        return "skipped", filepath, 0

    url = hf_hub_url(REPO_ID, filepath, repo_type=REPO_TYPE)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        # Use multipart upload for large files, put_object for small ones
        if total > 100 * 1024 * 1024:  # > 100 MB
            mpu = s3.create_multipart_upload(Bucket=DEST_BUCKET, Key=key)
            upload_id = mpu["UploadId"]
            parts = []
            part_num = 1
            buf = io.BytesIO()
            uploaded = 0

            try:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    buf.write(chunk)
                    if buf.tell() >= CHUNK_SIZE:
                        buf.seek(0)
                        part = s3.upload_part(
                            Bucket=DEST_BUCKET,
                            Key=key,
                            UploadId=upload_id,
                            PartNumber=part_num,
                            Body=buf,
                        )
                        parts.append({"PartNumber": part_num, "ETag": part["ETag"]})
                        uploaded += buf.tell()
                        part_num += 1
                        buf = io.BytesIO()

                # Upload remaining buffer
                if buf.tell() > 0:
                    buf.seek(0)
                    part = s3.upload_part(
                        Bucket=DEST_BUCKET,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_num,
                        Body=buf,
                    )
                    parts.append({"PartNumber": part_num, "ETag": part["ETag"]})

                s3.complete_multipart_upload(
                    Bucket=DEST_BUCKET,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            except Exception as e:
                s3.abort_multipart_upload(Bucket=DEST_BUCKET, Key=key, UploadId=upload_id)
                raise e
        else:
            data = resp.content
            s3.put_object(Bucket=DEST_BUCKET, Key=key, Body=data)
            total = len(data)

    return "ok", filepath, total


def transfer_shard(files, shard_id, resume, dry_run):
    if dry_run:
        print(f"[Shard {shard_id}] Would transfer {len(files)} files")
        for f in files[:5]:
            print(f"  {f}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        return

    hf_token = get_token()
    s3 = get_s3_client()

    done = 0
    failed = []
    skipped = 0
    bytes_total = 0
    t0 = time.time()

    for filepath in files:
        try:
            status, name, size = stream_upload(s3, hf_token, filepath, resume=resume)
            if status == "skipped":
                skipped += 1
            else:
                done += 1
                bytes_total += size
            elapsed = time.time() - t0
            rate = bytes_total / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(
                f"[Shard {shard_id}] {status} {done+skipped}/{len(files)} | "
                f"{bytes_total/1024/1024/1024:.2f} GB | {rate:.1f} MB/s | {name}",
                flush=True,
            )
        except Exception as e:
            failed.append((filepath, str(e)))
            print(f"[Shard {shard_id}] FAIL {filepath}: {e}", flush=True)

    print(f"\n[Shard {shard_id}] Done. ok={done} skipped={skipped} failed={len(failed)}")
    if failed:
        fail_log = Path(f"failed_shard_{shard_id}.txt")
        fail_log.write_text("\n".join(f"{p}\t{e}" for p, e in failed))
        print(f"[Shard {shard_id}] Failed files written to {fail_log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip already-uploaded files")
    args = parser.parse_args()

    print("Fetching file list from HuggingFace...", flush=True)
    all_files = get_all_files()
    print(f"Total files: {len(all_files)}", flush=True)

    # Split into shards
    shards = [[] for _ in range(args.shards)]
    for i, f in enumerate(all_files):
        shards[i % args.shards].append(f)

    if args.shard_id is not None:
        if args.shard_id >= args.shards:
            print(f"Error: shard-id {args.shard_id} >= shards {args.shards}")
            sys.exit(1)
        print(f"Processing shard {args.shard_id}/{args.shards} ({len(shards[args.shard_id])} files)")
        transfer_shard(shards[args.shard_id], args.shard_id, args.resume, args.dry_run)
    else:
        # Run all shards sequentially (use GNU parallel or tmux for true parallelism)
        print(f"Processing all {args.shards} shards sequentially.")
        print("Tip: for parallel execution, run each shard in a separate terminal:")
        for i in range(args.shards):
            print(f"  python3 transfer_hf_to_oci.py --shards {args.shards} --shard-id {i} --resume")
        print()
        for i, shard in enumerate(shards):
            print(f"\n--- Shard {i} ---")
            transfer_shard(shard, i, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
