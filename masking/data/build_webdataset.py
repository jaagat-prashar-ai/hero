#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
build_webdataset.py — Download nvidia/PhysicalAI-Autonomous-Vehicles from
HuggingFace and package as WebDataset shards, uploaded directly to S3.

Requires Python >= 3.11 (physical_ai_av constraint).
Run inside the dev container or a cluster node with sufficient disk for temp files.

Usage (smoke test — 10 clips):
    python masking/data/build_webdataset.py \
        --bucket my-research-bucket \
        --prefix physicalai-av/wds \
        --hf_token hf_xxxx \
        --max_clips 10

Usage (full dataset):
    python masking/data/build_webdataset.py \
        --bucket my-research-bucket \
        --prefix physicalai-av/wds \
        --hf_token hf_xxxx \
        --workers 16 \
        --resume_file /tmp/build_done_clips.txt

S3 layout produced:
    s3://{bucket}/{prefix}/train/shard_00000.tar
    s3://{bucket}/{prefix}/val/shard_00000.tar
    s3://{bucket}/{prefix}/test/shard_00000.tar
    s3://{bucket}/{prefix}/metadata/feature_presence.parquet
    s3://{bucket}/{prefix}/metadata/data_collection.parquet
    s3://{bucket}/{prefix}/metadata/ood_reasoning.parquet

WDS sample keys per clip (one sample == one 20-second clip):
    {clip_id}.json                       clip metadata + collection info + OOD events
    {clip_id}.egomotion.parquet          ego motion: timestamps, xyz, velocity, curvature
    {clip_id}.calibration.json           camera intrinsics + sensor extrinsics + vehicle dims
    {clip_id}.camera_cross_left_120fov.mp4
    {clip_id}.camera_cross_right_120fov.mp4
    {clip_id}.camera_front_wide_120fov.mp4
    {clip_id}.camera_front_tele_30fov.mp4
    {clip_id}.camera_rear_left_70fov.mp4
    {clip_id}.camera_rear_right_70fov.mp4
    {clip_id}.camera_rear_tele_30fov.mp4
    {clip_id}.lidar_top_360fov.parquet   (if LiDAR present for this clip)
    {clip_id}.radar_{name}.parquet       (one per radar unit, if present)

Splits:
    OOD-labeled clips  → NVIDIA's official train / val / test from ood_reasoning.parquet
    All other clips    → deterministic MD5-hash split: 90 % train, 10 % val
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


def _hf_retry(fn: Callable[[], _T], max_attempts: int = 5) -> _T:
    """Retry a HuggingFace Hub call on transient 5xx / 429 errors."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            transient = any(code in msg for code in ("502", "503", "504", "429", "Gateway Time-out"))
            if transient and attempt < max_attempts - 1:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                logger.warning("HF transient error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, max_attempts, wait, exc)
                time.sleep(wait)
            else:
                raise


def _s3_retry(fn: Callable[[], _T], max_attempts: int = 5) -> _T:
    """Retry a boto3 S3 call on transient OCI / network errors.

    OCI Object Storage can return 503 SlowDown, 503 ServiceUnavailable, and
    occasional 502s under load.  Mirror the backoff pattern used for HF calls.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            transient = any(code in msg for code in (
                "503", "502", "429", "SlowDown", "ServiceUnavailable",
                "RequestTimeout", "InternalError", "ConnectTimeout", "ReadTimeoutError",
            ))
            if transient and attempt < max_attempts - 1:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                logger.warning("S3 transient error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, max_attempts, wait, exc)
                time.sleep(wait)
            else:
                raise


import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotocoreConfig
import numpy as np
import pandas as pd
import webdataset as wds
from huggingface_hub import hf_hub_download, login

# On Lilypad workers, credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# env vars and the endpoint from AWS_ENDPOINT_URL_S3. For local dev, fall back to the
# oci.chi profile which has all of this configured in ~/.aws.
if not os.environ.get("AWS_ACCESS_KEY_ID"):
    os.environ.setdefault("AWS_PROFILE", "oci.chi")

# OCI S3 does not support AWS chunked encoding. payload_signing_enabled=True
# disables chunked encoding and uses standard payload signing instead.
_OCI_BOTO_CONFIG = BotocoreConfig(
    signature_version="s3v4",
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
    s3={
        "payload_signing_enabled": True,
        "multipart_threshold": 16 * 1024 * 1024,
        "multipart_chunksize": 16 * 1024 * 1024,
    },
)
_OCI_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
)

import physical_ai_av
from physical_ai_av import PhysicalAIAVDatasetInterface

logger = logging.getLogger(__name__)

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Maps short key used in WDS filenames → attribute name on avdi.features.CAMERA
CAMERA_FEATURES: dict[str, str] = {
    "camera_cross_left_120fov":  "CAMERA_CROSS_LEFT_120FOV",
    "camera_cross_right_120fov": "CAMERA_CROSS_RIGHT_120FOV",
    "camera_front_wide_120fov":  "CAMERA_FRONT_WIDE_120FOV",
    "camera_front_tele_30fov":   "CAMERA_FRONT_TELE_30FOV",
    "camera_rear_left_70fov":    "CAMERA_REAR_LEFT_70FOV",
    "camera_rear_right_70fov":   "CAMERA_REAR_RIGHT_70FOV",
    "camera_rear_tele_30fov":    "CAMERA_REAR_TELE_30FOV",
}

CLIPS_PER_SHARD = 50  # tune to keep shards ~1–5 GB

# Lilypad injects these env vars when num_replicas > 1
_ENV_RANK       = "RANK"
_ENV_WORLD_SIZE = "WORLD_SIZE"


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _hash_split(clip_id: str, val_frac: float = 0.10) -> str:
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < int(val_frac * 1000) else "train"


def load_splits_and_metadata(
    hf_token: str | None,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, dict[str, list[dict]]]:
    """Download the three metadata parquets and return:
        splits_map       {clip_id -> "train"|"val"|"test"}
        feature_df       feature_presence rows (one per clip)
        collection_df    data_collection rows (country, time-of-day, …)
        ood_by_clip      {clip_id -> [event_dict, …]}
    """
    kwargs = dict(repo_id=HF_REPO, repo_type="dataset", token=hf_token)

    logger.info("Downloading feature_presence.parquet …")
    fp_path = _hf_retry(lambda: hf_hub_download(filename="metadata/feature_presence.parquet", **kwargs))
    feature_df = pd.read_parquet(fp_path)

    logger.info("Downloading data_collection.parquet …")
    dc_path = _hf_retry(lambda: hf_hub_download(filename="metadata/data_collection.parquet", **kwargs))
    collection_df = pd.read_parquet(dc_path)

    logger.info("Downloading ood_reasoning.parquet …")
    ood_path = _hf_retry(lambda: hf_hub_download(filename="reasoning/ood_reasoning.parquet", **kwargs))
    ood_df = pd.read_parquet(ood_path)

    # Build OOD split map and per-clip event list
    ood_by_clip: dict[str, list[dict]] = {}
    ood_split_map: dict[str, str] = {}
    for cid, row in ood_df.iterrows():
        cid = str(cid)
        ood_by_clip.setdefault(cid, []).append(row.to_dict())
        # Use the "split" column if present; fall back to "train"
        ood_split_map[cid] = str(row.get("split", "train"))

    # Assign splits to all clips
    all_clips: list[str] = feature_df.index.astype(str).tolist()
    splits_map: dict[str, str] = {}
    for cid in all_clips:
        splits_map[cid] = ood_split_map.get(cid, _hash_split(cid))

    counts: dict[str, int] = {}
    for s in splits_map.values():
        counts[s] = counts.get(s, 0) + 1
    logger.info("Split counts: %s (total %d clips)", counts, len(all_clips))

    return splits_map, feature_df, collection_df, ood_by_clip


# ---------------------------------------------------------------------------
# Per-clip download
# ---------------------------------------------------------------------------

def _to_bytes(obj: Any) -> bytes:
    """Serialize a DataFrame or bytes-like object to raw bytes."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if hasattr(obj, "to_parquet"):
        buf = io.BytesIO()
        obj.to_parquet(buf, index=False)
        return buf.getvalue()
    # Last resort: json-encode
    return json.dumps(obj, default=str).encode()


def _camera_bytes(video_obj: Any) -> bytes | None:
    """Extract raw MP4 bytes from a video feature object.

    Tries multiple access patterns since the physical_ai_av API may
    expose raw bytes in different ways across versions.
    """
    # Pattern 1: explicit raw bytes method
    if hasattr(video_obj, "get_raw_bytes"):
        return video_obj.get_raw_bytes()
    # Pattern 2: local cached file path
    for attr in ("path", "local_path", "_path", "filepath"):
        p = getattr(video_obj, attr, None)
        if p and Path(p).exists():
            return Path(p).read_bytes()
    # Pattern 3: file-like object
    if hasattr(video_obj, "read"):
        return video_obj.read()
    return None


def build_clip_sample(
    avdi: PhysicalAIAVDatasetInterface,
    clip_id: str,
    collection_row: dict | None,
    ood_events: list[dict],
    feature_row: dict,
) -> dict[str, bytes]:
    """Download every available sensor modality for one clip.

    Missing sensors are silently skipped (not all clips have LiDAR / radar).
    Returns a flat {wds_extension -> bytes} dict ready for TarWriter.
    """
    sample: dict[str, bytes] = {}

    # ── Metadata JSON ────────────────────────────────────────────────────────
    meta = {
        "clip_id":          clip_id,
        "collection":       {k: str(v) for k, v in (collection_row or {}).items()},
        "feature_presence": {k: str(v) for k, v in feature_row.items()},
        "ood_events":       [
            {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in ev.items()}
            for ev in ood_events
        ],
    }
    sample["json"] = json.dumps(meta, ensure_ascii=False).encode()

    # ── Egomotion ────────────────────────────────────────────────────────────
    try:
        egomotion = avdi.get_clip_feature(clip_id, feature=avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        sample["egomotion.parquet"] = _to_bytes(egomotion)
    except Exception as exc:
        logger.warning("clip %s: egomotion error: %s", clip_id, exc)

    # ── Calibration ──────────────────────────────────────────────────────────
    try:
        ext = avdi.get_clip_feature(
            clip_id, feature=avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True
        )
        intr = avdi.get_clip_feature(
            clip_id, feature=avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True
        )
        dims = avdi.get_clip_feature(
            clip_id, feature=avdi.features.CALIBRATION.VEHICLE_DIMENSIONS, maybe_stream=True
        )

        def _serialize(obj: Any) -> Any:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        cal = {
            "sensor_extrinsics":  _serialize(ext),
            "camera_intrinsics":  _serialize(intr),
            "vehicle_dimensions": _serialize(dims),
        }
        sample["calibration.json"] = json.dumps(cal, default=str, ensure_ascii=False).encode()
    except Exception as exc:
        logger.warning("clip %s: calibration error: %s", clip_id, exc)

    # ── Cameras (all 7) ──────────────────────────────────────────────────────
    for cam_key, feat_name in CAMERA_FEATURES.items():
        feat_attr = getattr(avdi.features.CAMERA, feat_name, None)
        if feat_attr is None:
            continue
        try:
            video = avdi.get_clip_feature(clip_id, feature=feat_attr, maybe_stream=True)
            raw = _camera_bytes(video)
            if raw is not None:
                sample[f"{cam_key}.mp4"] = raw
            else:
                # Fallback: store frames as numpy array (10 fps, full 20 s clip)
                timestamps_us = np.arange(0, 20_000_000, 100_000, dtype=np.int64)
                frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)
                buf = io.BytesIO()
                np.savez_compressed(
                    buf,
                    frames=np.array(frames, dtype=np.uint8),
                    timestamps_us=actual_ts,
                )
                sample[f"{cam_key}.npz"] = buf.getvalue()
        except Exception as exc:
            logger.debug("clip %s: camera %s unavailable: %s", clip_id, cam_key, exc)

    # ── LiDAR ────────────────────────────────────────────────────────────────
    try:
        lidar = avdi.get_clip_feature(
            clip_id, feature=avdi.features.LIDAR.LIDAR_TOP_360FOV, maybe_stream=True
        )
        sample["lidar_top_360fov.parquet"] = _to_bytes(lidar)
    except Exception as exc:
        logger.debug("clip %s: lidar unavailable: %s", clip_id, exc)

    # ── Radar (all units present on this clip) ────────────────────────────────
    radar_ns = getattr(avdi.features, "RADAR", None)
    if radar_ns is not None:
        for radar_attr in dir(radar_ns):
            if radar_attr.startswith("_"):
                continue
            try:
                feat = getattr(radar_ns, radar_attr)
                radar_data = avdi.get_clip_feature(clip_id, feature=feat)
                key = f"radar_{radar_attr.lower()}.parquet"
                sample[key] = _to_bytes(radar_data)
            except Exception:
                pass  # not all clips have all radar units

    return sample


# ---------------------------------------------------------------------------
# S3 shard writer
# ---------------------------------------------------------------------------

class S3ShardWriter:
    """Pack WDS samples into tar shards and stream each one to S3."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        split: str,
        clips_per_shard: int,
        worker_rank: int = 0,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.split = split
        self.clips_per_shard = clips_per_shard
        self.worker_rank = worker_rank
        self._s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"), config=_OCI_BOTO_CONFIG)
        self._shard_idx = 0
        self._count = 0
        self._tmpfile: Any = None
        self._writer: Any = None
        self._lock = threading.Lock()
        self._open_shard()

    def _key(self, idx: int) -> str:
        # Encode worker rank in filename so multiple workers never collide on S3.
        return f"{self.prefix}/{self.split}/shard_{self.worker_rank:03d}_{idx:05d}.tar"

    def _open_shard(self) -> None:
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
        self._tmpfile.close()
        self._writer = wds.TarWriter(self._tmpfile.name)

    def _flush(self) -> None:
        self._writer.close()
        key = self._key(self._shard_idx)
        size_mb = Path(self._tmpfile.name).stat().st_size / 1e6
        logger.info(
            "Uploading s3://%s/%s  (%.0f MB, %d clips)",
            self.bucket, key, size_mb, self._count,
        )
        try:
            _s3_retry(lambda: self._s3.upload_file(
                self._tmpfile.name, self.bucket, key, Config=_OCI_TRANSFER_CONFIG
            ))
        finally:
            # Always clean up the tempfile and reset counters so the writer
            # remains usable even when the upload ultimately fails.
            try:
                os.unlink(self._tmpfile.name)
            except OSError:
                pass
            self._shard_idx += 1
            self._count = 0

    def write(self, clip_id: str, sample: dict[str, bytes]) -> None:
        wds_sample: dict[str, Any] = {"__key__": clip_id}
        wds_sample.update(sample)
        with self._lock:
            self._writer.write(wds_sample)
            self._count += 1
            if self._count >= self.clips_per_shard:
                try:
                    self._flush()
                finally:
                    # Open the next shard regardless of whether the upload
                    # succeeded so the writer stays usable after a failure.
                    self._open_shard()

    def close(self) -> None:
        with self._lock:
            if self._count > 0:
                self._flush()
            else:
                self._writer.close()
                if Path(self._tmpfile.name).exists():
                    os.unlink(self._tmpfile.name)


# ---------------------------------------------------------------------------
# Metadata upload helper
# ---------------------------------------------------------------------------

def upload_metadata_parquets(bucket: str, prefix: str, hf_token: str | None) -> None:
    """Copy the three metadata parquets from HuggingFace straight to S3."""
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"), config=_OCI_BOTO_CONFIG)
    kwargs = dict(repo_id=HF_REPO, repo_type="dataset", token=hf_token)
    files = {
        "metadata/feature_presence.parquet": "metadata/feature_presence.parquet",
        "metadata/data_collection.parquet":  "metadata/data_collection.parquet",
        "reasoning/ood_reasoning.parquet":   "metadata/ood_reasoning.parquet",
    }
    for hf_filename, s3_suffix in files.items():
        local = _hf_retry(lambda fn=hf_filename: hf_hub_download(filename=fn, **kwargs))
        key = f"{prefix.rstrip('/')}/{s3_suffix}"
        logger.info("Uploading metadata → s3://%s/%s", bucket, key)
        # Use put_object with bytes in memory to avoid s3transfer's chunked encoding,
        # which OCI S3 rejects. Metadata parquets are small enough to buffer.
        s3.put_object(Bucket=bucket, Key=key, Body=Path(local).read_bytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # physical_ai_av streams via httpx, which logs every chunk request at INFO.
    # Suppress to WARNING so application logs remain readable in OCI Log Analytics.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(
        description="Build WebDataset shards from nvidia/PhysicalAI-Autonomous-Vehicles → S3"
    )
    ap.add_argument("--bucket",          required=True,  help="S3 bucket name")
    ap.add_argument("--prefix",          default="physicalai-av/wds", help="S3 key prefix")
    ap.add_argument("--hf_token",        default=os.environ.get("HF_TOKEN"),
                    help="HuggingFace access token (or set HF_TOKEN env var)")
    ap.add_argument("--workers",         type=int, default=8,
                    help="Parallel download workers per split (local threads per Lilypad node)")
    ap.add_argument("--clips_per_shard", type=int, default=CLIPS_PER_SHARD)
    ap.add_argument("--splits",          nargs="+", default=["train", "val", "test"],
                    help="Which splits to build")
    ap.add_argument("--max_clips",       type=int, default=None,
                    help="Cap total clips per split (smoke test)")
    ap.add_argument("--resume_file",     default=None,
                    help="Path to a text file of already-completed clip IDs (one per line)")
    ap.add_argument("--skip_metadata_upload", action="store_true",
                    help="Skip uploading the three metadata parquets to S3")
    # Distributed sharding: Lilypad injects RANK / WORLD_SIZE; can also be set explicitly.
    ap.add_argument("--rank",       type=int,
                    default=int(os.environ.get(_ENV_RANK, "0")),
                    help="This worker's rank (0-indexed). Auto-read from RANK env var.")
    ap.add_argument("--world_size", type=int,
                    default=int(os.environ.get(_ENV_WORLD_SIZE, "1")),
                    help="Total number of workers. Auto-read from WORLD_SIZE env var.")
    args = ap.parse_args()

    if not args.hf_token:
        ap.error("--hf_token or HF_TOKEN env var is required")

    if args.rank >= args.world_size:
        ap.error(f"--rank ({args.rank}) must be < --world_size ({args.world_size})")

    logger.info("Distributed sharding: rank=%d / world_size=%d", args.rank, args.world_size)

    login(token=args.hf_token, add_to_git_credential=False)

    # ── Upload raw metadata parquets once (rank 0 only) ──────────────────────
    if not args.skip_metadata_upload and args.rank == 0:
        upload_metadata_parquets(args.bucket, args.prefix, args.hf_token)
    elif not args.skip_metadata_upload:
        logger.info("Skipping metadata upload (handled by rank 0)")

    # ── Load split assignments ────────────────────────────────────────────────
    splits_map, feature_df, collection_df, ood_by_clip = load_splits_and_metadata(
        args.hf_token
    )

    collection_by_clip = {
        str(cid): row.to_dict() for cid, row in collection_df.iterrows()
    }
    feature_by_clip = {
        str(cid): row.to_dict() for cid, row in feature_df.iterrows()
    }

    # ── Resume: skip already-done clips ──────────────────────────────────────
    done: set[str] = set()
    if args.resume_file and Path(args.resume_file).exists():
        with open(args.resume_file) as fh:
            done = {line.strip() for line in fh if line.strip()}
        logger.info("Resuming — %d clips already completed", len(done))

    # ── Build per-split work lists, partitioned by rank ──────────────────────
    # Sort clip IDs for a deterministic, stable partition across restarts.
    work: dict[str, list[str]] = {s: [] for s in args.splits}
    for clip_id, split in sorted(splits_map.items()):
        if split in args.splits and clip_id not in done:
            work[split].append(clip_id)

    # Slice this worker's share: every world_size-th clip starting at rank.
    for split in args.splits:
        work[split] = work[split][args.rank :: args.world_size]

    if args.max_clips:
        for split in args.splits:
            work[split] = work[split][: args.max_clips]

    for split in args.splits:
        logger.info("  %s: %d clips queued (rank %d/%d)",
                    split, len(work[split]), args.rank, args.world_size)

    # ── Initialize model interface and S3 writers ─────────────────────────────
    avdi = _hf_retry(PhysicalAIAVDatasetInterface)

    writers = {
        split: S3ShardWriter(args.bucket, args.prefix, split, args.clips_per_shard,
                             worker_rank=args.rank)
        for split in args.splits
    }

    resume_lock = threading.Lock()
    resume_fh = open(args.resume_file, "a") if args.resume_file else None
    n_ok = 0
    n_err = 0
    counter_lock = threading.Lock()

    def process(clip_id: str, split: str) -> None:
        nonlocal n_ok, n_err
        try:
            sample = build_clip_sample(
                avdi,
                clip_id,
                collection_by_clip.get(clip_id),
                ood_by_clip.get(clip_id, []),
                feature_by_clip.get(clip_id, {}),
            )
            writers[split].write(clip_id, sample)
            with counter_lock:
                n_ok += 1
                if n_ok % 100 == 0:
                    logger.info("Progress: %d ok / %d err", n_ok, n_err)
            if resume_fh:
                with resume_lock:
                    resume_fh.write(clip_id + "\n")
                    resume_fh.flush()
        except Exception as exc:
            import traceback
            logger.error("FAIL %s (%s): %s", clip_id, split, exc)
            traceback.print_exc()
            with counter_lock:
                n_err += 1

    # ── Fan out across all splits in parallel ─────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for split in args.splits:
            for clip_id in work[split]:
                futures.append(pool.submit(process, clip_id, split))
        concurrent.futures.wait(futures)

    for w in writers.values():
        w.close()

    if resume_fh:
        resume_fh.close()

    logger.info("Finished: %d succeeded, %d failed", n_ok, n_err)
    logger.info("Shards written to s3://%s/%s/", args.bucket, args.prefix)


if __name__ == "__main__":
    main()
