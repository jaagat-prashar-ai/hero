# SPDX-License-Identifier: Apache-2.0
"""
s3_clip_extract.py — pull only specific clips' files out of S3 WDS shards via
HTTP range reads, instead of downloading the whole multi-GB shard.

Each shard is a WebDataset tar holding ~50 clips; a curated sample (see
masking/data/sample_clips.py) typically needs only 1-2 clips per shard.
Walking the tar headers and range-fetching only the needed clips' payloads
(skipping everything else via a Range jump, the same technique
sample_clips.py already uses to peek at json headers) turns "download every
shard in full" into "download a few hundred MB total" for a 52-clip sample
instead of ~175GB across 50 shards -- and, critically, finishes fast enough
that the job doesn't sit GPU-idle long enough to get reclaimed by whatever
cluster mechanism is doing that (see BUGS.md).

No local file or tar is produced -- extract_clip_members() returns the raw
bytes straight into memory, in exactly the shape
masking.data.wds_dataset._expand_clip_to_events() already expects, so a
caller can go directly from an S3 range read to model input with no
intermediate packaging step.
"""

from __future__ import annotations

import boto3

TAR_BLOCK = 512


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def extract_clip_members(
    bucket: str, shard_key: str, clip_id: str, start_offset: int = 0
) -> dict[str, bytes]:
    """Range-read just one clip's tar members out of a shard, skipping past
    every other clip's payload instead of downloading it.

    start_offset, if known (see masking.data.sample_clips.py, which records
    each clip's own header offset while scanning), lets this jump straight to
    the clip instead of walking every preceding clip's headers to reach it --
    the difference between ~10 range reads and up to ~500 for a clip near the
    end of a 50-clip shard.

    Returns {extension: bytes}, e.g. {"json": b"...", "egomotion.parquet": b"...",
    "camera_front_wide_120fov.mp4": b"..."} -- the same shape build_webdataset.py
    writes and wds_dataset.py expects to read.
    """
    s3 = boto3.client("s3")
    pos = start_offset
    zero_blocks = 0
    members: dict[str, bytes] = {}
    prefix = f"{clip_id}."
    found_any = False
    while True:
        hdr = _range_get(s3, bucket, shard_key, pos, TAR_BLOCK)
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
        is_pax = "@PaxHeader" in name

        if name.startswith(prefix) and not is_pax:
            members[name[len(prefix):]] = _range_get(s3, bucket, shard_key, data_start, size)
            found_any = True
        elif found_any and not is_pax:
            # A clip's files are written contiguously (see build_webdataset.py's
            # S3ShardWriter.write()) -- once we've collected some and hit a
            # different clip's real header, this clip's group is done. Stop
            # instead of walking the rest of the shard to EOF.
            break

        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK
    return members
