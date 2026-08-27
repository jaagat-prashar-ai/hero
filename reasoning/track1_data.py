# SPDX-License-Identifier: Apache-2.0
"""Track 1 submission data iterator over the 214-clip test-split WDS shards.

Sibling of reasoning/wds_test_loader.py with three submission-specific
differences (that loader is kept untouched for the perplexity work):

1. Never drops a submission key. 69/284 test events have an
   event_start_timestamp outside the valid conditioning window (49 below the
   1.6 s history floor, 20 too close to the clip end for the 6.4 s future
   window). Instead of skipping, t0 is clamped into the valid range computed
   from the clip's own egomotion track; the emitted dict records the clamp.
2. Camera frames follow load_physical_aiavdataset's convention -- 4 frames at
   0.1 s spacing ending at t0 ([t0-0.3s, t0-0.2s, t0-0.1s, t0]) -- not the
   "4 consecutive stored frames" wds_test_loader uses. The WDS mp4s keep the
   original ~30 fps, so consecutive frames span only ~0.1 s, which is not
   what the model saw in training.
3. Emits camera_indices (required by alpamayo1_5.helper.create_message);
   CAMERA_KEYS order below is already sorted by that index convention
   (load_physical_aiavdataset.py's camera_name_to_index).

Reads the .tar shards with plain tarfile (no webdataset dependency); WDS
grouping convention is "member name up to the first dot is the sample key".
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from physical_ai_av.egomotion import EgomotionState
from physical_ai_av.video import SeekVideoReader

logger = logging.getLogger(__name__)

# (name, camera_name_to_index value) in index-sorted order.
CAMERA_KEYS = (
    ("camera_cross_left_120fov", 0),
    ("camera_front_wide_120fov", 1),
    ("camera_cross_right_120fov", 2),
    ("camera_front_tele_30fov", 6),
)

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP_S = 0.1
NUM_FRAMES = 4
# Margin past the exact history/future window edges so interpolation never
# lands exactly on a track endpoint (same spirit as clipgen's 1.6->1.7 s
# clamp for the sampler's history floor).
CLAMP_MARGIN_US = 100_000


class ClipWindowError(ValueError):
    """The event window cannot be satisfied even after clamping."""


def _egomotion_from_bytes(egomotion_bytes: bytes):
    """Return (interpolator, track_min_us, track_max_us)."""
    df = pd.read_parquet(io.BytesIO(egomotion_bytes))
    ts = df["timestamp_us"].to_numpy(copy=True)
    state = EgomotionState.from_egomotion_df(df)
    return state.create_interpolator(ts), int(ts.min()), int(ts.max())


def clamp_t0_us(t0_us: int, track_min_us: int, track_max_us: int) -> int:
    """Clamp t0 into the range where both the 1.6 s history and 6.4 s future
    windows (and the 0.3 s frame lookback) fit inside the egomotion track.

    _ego_windows' own guard is absolute (t0 must exceed the 1.6 s history
    range on the clip clock), so the lower bound honors both that and the
    track start.
    """
    history_us = int(NUM_HISTORY_STEPS * TIME_STEP_S * 1_000_000)
    future_us = int(NUM_FUTURE_STEPS * TIME_STEP_S * 1_000_000)
    lo = max(history_us, track_min_us + history_us) + CLAMP_MARGIN_US
    hi = track_max_us - future_us - CLAMP_MARGIN_US
    if hi < lo:
        raise ClipWindowError(
            f"egomotion track [{track_min_us}, {track_max_us}] us too short for "
            f"history {history_us} + future {future_us} us"
        )
    return int(min(max(t0_us, lo), hi))


def _ego_windows(egomotion, t0_us: int) -> dict[str, torch.Tensor]:
    """Identical windowing + t0-relative transform as wds_test_loader.py /
    load_physical_aiavdataset.py."""
    history_offsets_us = np.arange(
        -(NUM_HISTORY_STEPS - 1) * TIME_STEP_S * 1_000_000,
        TIME_STEP_S * 1_000_000 / 2,
        TIME_STEP_S * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        TIME_STEP_S * 1_000_000,
        (NUM_FUTURE_STEPS + 0.5) * TIME_STEP_S * 1_000_000,
        TIME_STEP_S * 1_000_000,
    ).astype(np.int64)

    try:
        ego_history = egomotion(t0_us + history_offsets_us)
        ego_future = egomotion(t0_us + future_offsets_us)
    except ValueError as e:
        raise ClipWindowError(f"t0_us={t0_us} window doesn't fit the egomotion track: {e}") from e

    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1]
    t0_rot_inv = spt.Rotation.from_quat(ego_history_quat[-1]).inv()

    return {
        "ego_history_xyz": torch.from_numpy(t0_rot_inv.apply(ego_history_xyz - t0_xyz))
        .float()
        .unsqueeze(0)
        .unsqueeze(0),
        "ego_history_rot": torch.from_numpy(
            (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(0),
    }


def _decode_frames_spaced(mp4_bytes: bytes, timestamps_bytes: bytes, t0_us: int) -> np.ndarray:
    """Decode NUM_FRAMES frames at TIME_STEP_S spacing ending at t0
    (load_physical_aiavdataset's image_timestamps convention), picking for
    each target timestamp the nearest stored frame at/before it via the
    real per-camera frame_timestamps sidecar."""
    frame_ts = pd.read_parquet(io.BytesIO(timestamps_bytes))["timestamp"].to_numpy()
    target_ts = np.array(
        [t0_us - (NUM_FRAMES - 1 - i) * int(TIME_STEP_S * 1_000_000) for i in range(NUM_FRAMES)],
        dtype=np.int64,
    )
    frame_idxs = np.searchsorted(frame_ts, target_ts, side="right") - 1
    if frame_idxs[0] < 0:
        raise ClipWindowError(f"t0_us={t0_us} needs a frame before the first stored frame")
    frame_idxs = np.clip(frame_idxs, 0, len(frame_ts) - 1).astype(np.int64)

    reader = SeekVideoReader(video_data=io.BytesIO(mp4_bytes), timestamps=None)
    try:
        images = reader.decode_images_from_frame_indices(frame_idxs)
    finally:
        reader.close()
    return images


def _iter_tar_samples(shard_path: str):
    """Yield {member-suffix: bytes} per WDS sample key from one .tar shard.

    build_test_split.py wrote shards from a ThreadPoolExecutor, so one clip's
    members are NOT guaranteed contiguous (smoke e2wbdb hit real interleaving:
    'egomotion.parquet' KeyErrors). Index the whole tar by key first, then
    extract per key via random access."""
    with tarfile.open(shard_path, "r") as tar:
        by_key: dict[str, list] = {}
        for member in tar:
            if not member.isfile():
                continue
            key, _, _suffix = member.name.split("/")[-1].partition(".")
            by_key.setdefault(key, []).append(member)
        for key, members in by_key.items():
            sample: dict[str, bytes] = {}
            for member in members:
                f = tar.extractfile(member)
                assert f is not None, member.name
                sample[member.name.split("/")[-1].partition(".")[2]] = f.read()
            yield key, sample


def iter_track1_samples(shard_paths: list[str], fill_dir: str | None = None):
    """Yield one dict per (clip_id, event_idx) across the given shards.

    Each dict has the load_physical_aiavdataset-shaped keys sample_rollout_group
    needs (image_frames (4,4,3,H,W) uint8, camera_indices, ego_history_xyz,
    ego_history_rot) plus clip_id, event_idx, submission_key, t0_us (original),
    t0_us_used, clamped. Events that fail even after clamping are yielded as
    {"submission_key", "clip_id", "event_idx", "error"} so the caller can
    still emit the key.

    fill_dir: directory of <clip_id>.egomotion.parquet sidecars for the 4
    shard-0 clips whose original build dropped the egomotion member
    (backfilled 2026-08-26 from HF via build_webdataset's own serialization,
    at s3 .../wds/test_egomotion_fill/).
    """
    camera_indices = torch.tensor([idx for _, idx in CAMERA_KEYS], dtype=torch.int64)

    for shard_path in shard_paths:
        for clip_id, sample in _iter_tar_samples(shard_path):
            meta = json.loads(sample["json"])
            ood_events = meta.get("ood_events") or []
            if not ood_events:
                continue
            events = json.loads(ood_events[0]["events"])
            if not events:
                continue

            try:
                ego_bytes = sample.get("egomotion.parquet")
                if ego_bytes is None and fill_dir is not None:
                    fill_path = os.path.join(fill_dir, f"{clip_id}.egomotion.parquet")
                    if os.path.exists(fill_path):
                        with open(fill_path, "rb") as f:
                            ego_bytes = f.read()
                        logger.info("using egomotion sidecar for %s", clip_id)
                if ego_bytes is None:
                    raise KeyError("egomotion.parquet (no shard member, no sidecar)")
                egomotion, track_min_us, track_max_us = _egomotion_from_bytes(ego_bytes)
            except Exception as exc:
                for event_idx in range(len(events)):
                    yield {
                        "submission_key": f"{clip_id}_{event_idx}",
                        "clip_id": clip_id,
                        "event_idx": event_idx,
                        "error": f"egomotion load failed: {exc}",
                    }
                continue

            for event_idx, event in enumerate(events):
                t0_us = int(event["event_start_timestamp"])
                try:
                    t0_used = clamp_t0_us(t0_us, track_min_us, track_max_us)
                    windows = _ego_windows(egomotion, t0_used)
                    image_frames = torch.stack(
                        [
                            torch.from_numpy(
                                _decode_frames_spaced(
                                    sample[f"{cam}.mp4"],
                                    sample[f"{cam}.timestamps.parquet"],
                                    t0_used,
                                )
                            )
                            for cam, _ in CAMERA_KEYS
                        ],
                        dim=0,
                    ).permute(0, 1, 4, 2, 3)
                except Exception as exc:
                    logger.warning("event %s_%d unusable: %s", clip_id, event_idx, exc)
                    yield {
                        "submission_key": f"{clip_id}_{event_idx}",
                        "clip_id": clip_id,
                        "event_idx": event_idx,
                        "error": str(exc),
                    }
                    continue

                yield {
                    "image_frames": image_frames,
                    "camera_indices": camera_indices,
                    **windows,
                    "clip_id": clip_id,
                    "event_idx": event_idx,
                    "submission_key": f"{clip_id}_{event_idx}",
                    "t0_us": t0_us,
                    "t0_us_used": t0_used,
                    "clamped": t0_used != t0_us,
                }
