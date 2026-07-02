# SPDX-License-Identifier: Apache-2.0
"""
wds_dataset.py — WebDataset pipeline that converts the RAW per-clip WDS shards
(written by build_wds/data/build_webdataset.py) into the sample format expected
by the masking experiments.

Each WDS sample contains the raw clip data as actually produced by the build
job — there is no pre-baked snapshot format:
  {clip_id}.json                — clip metadata: collection info, feature_presence,
                                   ood_events (event_cluster + timestamped events)
  {clip_id}.egomotion.parquet   — full-clip trajectory: timestamp_us, x,y,z,
                                   qx,qy,qz,qw, vx,vy,vz, ax,ay,az, curvature
  {clip_id}.calibration.json    — camera intrinsics + sensor extrinsics (unused here)
  {clip_id}.camera_*.mp4        — one raw video per camera (7 cameras total)

For each OOD event embedded in a clip's json (`ood_events`), this module:
  - decodes the ego history/future trajectory directly from egomotion.parquet,
    transformed into the ego-local frame at t0 (same transform as
    third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py, fed
    from the parquet instead of a live physical_ai_av call — importing that
    module directly isn't practical here since it requires Python >= 3.11 and
    masking's worker runs on 3.10)
  - decodes the 4 required camera views' frames on-the-fly from the raw mp4
    bytes via PyAV (`av`, already a build_wds dependency)
and yields one item per event with:
  clip_id:        str
  t0_us:          int
  event_cluster:  str
  group:          str    (source camera feature the event was tagged from)
  was_clamped:    bool   (t0 had to move to fit a full history/future window)
  event_idx:      int
  event_coc:      str
  model_inputs:   dict compatible with build_inputs() / run_masked_openloop.py
    image_frames:     (4, 4, 3, H, W) uint8 torch.Tensor
    camera_indices:   (4,) torch.Tensor — fixed [0, 1, 2, 6] for the 4-cam rig
    ego_history_xyz:  (1, 1, 16, 3) float32 torch.Tensor
    ego_history_rot:  (1, 1, 16, 3, 3) float32 torch.Tensor
    ego_future_xyz:   (1, 1, 64, 3) float32 torch.Tensor

Non-OOD clips (empty `ood_events`) yield nothing — this pipeline drives the
OOD/scenario-type masking analysis, not blanket dataset coverage.

NOTE on video/egomotion clock alignment
----------------------------------------
t0_us and all history/future offsets are applied directly as both egomotion
parquet timestamps AND video decode target times (seconds = t0_us / 1e6).
This assumes the per-camera mp4s' PTS=0 reference matches the same clip-clock
origin as the egomotion timestamps (both ultimately come from the same
physical_ai_av clip data before build_webdataset.py transcodes video to AV1).
ffmpeg's default transcode does not add an explicit `-ss`/timestamp-reset flag,
so this should hold, but it has not been cross-verified against ground truth
motion — worth a visual/motion sanity check before trusting analysis results.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Generator, Iterable

import av
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
import webdataset as wds

logger = logging.getLogger(__name__)

# Camera indices for the standard 4-cam rig (CROSS_LEFT, FRONT_WIDE, CROSS_RIGHT, FRONT_TELE)
CAMERA_INDICES = torch.tensor([0, 1, 2, 6], dtype=torch.long)
CAMERA_FEATURES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP_S = 0.1
NUM_FRAMES = 4


def _decode_json(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def _no_node_split(src, group=None):
    """No-op nodesplitter: every rank sees every shard.

    wds.WebDataset defaults to wds.single_node_only, which *raises* under
    multi-node (world_size > 1) training unless an explicit nodesplitter is
    given. We must pass one, but it must NOT slice the shard list by rank —
    run.py's masking_loop already partitions at the sample level via
    _shard_owner() across the FULL shard list given to every rank.
    """
    yield from src


def iter_clip_events(
    shard_paths: Iterable[str | Path],
    *,
    shuffle_shards: bool = False,
    resampled: bool = False,
) -> Generator[dict, None, None]:
    """
    Yield one OOD-event item per sample from the provided WDS shard paths.

    Args:
        shard_paths: Local .tar paths or s3:// URLs understood by webdataset.
        shuffle_shards: Shuffle shard order (useful for training; off for inference).
        resampled: Use wds.ResampledShards for infinite looping (training only).

    Yields:
        Dicts with keys: clip_id, t0_us, event_cluster, group, was_clamped,
        event_idx, event_coc, model_inputs.
    """
    str_paths = [str(p) for p in shard_paths]
    if not str_paths:
        logger.warning("iter_clip_events: no shard paths provided")
        return

    # nodesplitter=_no_node_split (not wds.split_by_node): callers (run.py's
    # masking_loop) already give every rank the FULL shard list and partition
    # at the sample level via _shard_owner() hashing. wds.split_by_node would
    # additionally slice the shard list itself by torch.distributed
    # rank/world_size, which silently drops every shard for any rank >=
    # len(shards) (e.g. 2 shards, 8 ranks -> "No samples found in dataset;
    # perhaps you have fewer shards than workers.").
    if resampled:
        dataset = wds.WebDataset(
            wds.ResampledShards(str_paths),
            shardshuffle=shuffle_shards,
            nodesplitter=_no_node_split,
        )
    else:
        dataset = wds.WebDataset(
            str_paths,
            shardshuffle=shuffle_shards,
            nodesplitter=_no_node_split,
        )

    for raw_sample in dataset:
        try:
            items = _expand_clip_to_events(raw_sample)
        except Exception as exc:
            logger.warning("Error expanding clip %s: %s", raw_sample.get("__key__"), exc)
            continue
        yield from items


def iter_clip_events_from_manifest(
    manifest_path: str | Path, bucket: str
) -> Generator[dict, None, None]:
    """Like iter_clip_events(), but for a masking.data.sample_clips.py manifest:
    pulls each clip's files directly from S3 via range reads (see
    masking.data.s3_clip_extract) and feeds the bytes straight into
    _expand_clip_to_events() -- no shard download, no local tar, no
    intermediate file of any kind. Yields identically-shaped items.
    """
    from masking.data.s3_clip_extract import extract_clip_members

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    for row in manifest:
        clip_id, shard_key = row["clip_id"], row["shard_key"]
        members = extract_clip_members(bucket, shard_key, clip_id)
        if not members:
            logger.warning("clip %s: no members found in %s", clip_id, shard_key)
            continue
        try:
            items = _expand_clip_to_events({"__key__": clip_id, **members})
        except Exception as exc:
            logger.warning("Error expanding clip %s: %s", clip_id, exc)
            continue
        yield from items


def _clamp_t0(t_min: float, t_max: float, t0_us_raw: int) -> tuple[int | None, bool]:
    """Clamp t0_us so a full history+future window fits inside [t_min, t_max].

    Returns (None, True) if the clip is too short to ever fit the window.
    """
    lo = t_min + NUM_HISTORY_STEPS * TIME_STEP_S * 1e6
    hi = t_max - NUM_FUTURE_STEPS * TIME_STEP_S * 1e6
    if lo > hi:
        return None, True
    t0 = min(max(t0_us_raw, lo), hi)
    return int(t0), t0 != t0_us_raw


def _interp_state(
    ego_df: "pd.DataFrame", query_us: np.ndarray
) -> tuple[np.ndarray, "spt.Rotation"]:
    """Linearly interpolate xyz and slerp-interpolate rotation at query_us."""
    t = ego_df["timestamp_us"].to_numpy()
    xyz = np.stack(
        [np.interp(query_us, t, ego_df[c].to_numpy()) for c in ("x", "y", "z")], axis=-1
    )
    quats = ego_df[["qx", "qy", "qz", "qw"]].to_numpy()
    slerp = spt.Slerp(t, spt.Rotation.from_quat(quats))
    rot = slerp(np.clip(query_us, t.min(), t.max()))
    return xyz, rot


def _decode_camera_frames(mp4_bytes: bytes, timestamps_us: np.ndarray) -> np.ndarray:
    """Decode the frame at-or-after each requested timestamp (seconds = us/1e6)
    from raw mp4 bytes. Returns (len(timestamps_us), H, W, 3) uint8."""
    container = av.open(io.BytesIO(mp4_bytes))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        order = np.argsort(timestamps_us)
        frames_by_idx: dict[int, np.ndarray] = {}
        for idx in order:
            target_sec = float(timestamps_us[idx]) / 1e6
            container.seek(
                int(target_sec / time_base), stream=stream, any_frame=False, backward=True
            )
            last = None
            for frame in container.decode(stream):
                last = frame
                if frame.time is not None and frame.time >= target_sec:
                    break
            if last is None:
                raise RuntimeError(f"no frame decoded near t={target_sec:.3f}s")
            frames_by_idx[int(idx)] = last.to_ndarray(format="rgb24")
        return np.stack([frames_by_idx[i] for i in range(len(timestamps_us))])
    finally:
        container.close()


def _expand_clip_to_events(sample: dict) -> list[dict]:
    """Expand one raw WDS clip sample into one item per embedded OOD event."""
    clip_id = sample["__key__"]

    raw_json = sample.get("json", {})
    if isinstance(raw_json, (bytes, bytearray)):
        meta = json.loads(raw_json.decode("utf-8"))
    elif isinstance(raw_json, str):
        meta = json.loads(raw_json)
    else:
        meta = raw_json

    ood_rows = meta.get("ood_events", [])
    if not ood_rows:
        return []

    ego_bytes = sample.get("egomotion.parquet")
    if ego_bytes is None:
        logger.warning("clip %s: missing egomotion.parquet", clip_id)
        return []
    ego_df = (
        pd.read_parquet(io.BytesIO(ego_bytes))
        if isinstance(ego_bytes, (bytes, bytearray))
        else ego_bytes
    )
    t_min = float(ego_df["timestamp_us"].min())
    t_max = float(ego_df["timestamp_us"].max())

    cam_bytes: dict[str, bytes] = {}
    for feat in CAMERA_FEATURES:
        raw = sample.get(f"{feat}.mp4")
        if raw is None:
            logger.warning("clip %s: missing camera %s", clip_id, feat)
            return []
        cam_bytes[feat] = raw

    items: list[dict] = []
    for event_idx, row in enumerate(ood_rows):
        event_cluster = row.get("event_cluster", "UNKNOWN")
        group = row.get("feature", "unknown")
        sub_events = row.get("events", [])
        if isinstance(sub_events, str):
            try:
                sub_events = json.loads(sub_events)
            except Exception:
                logger.warning("clip %s: could not parse stringified events field", clip_id)
                sub_events = []

        for sub_idx, ev in enumerate(sub_events):
            t0_us_raw = int(ev.get("event_start_timestamp", 0))
            t0_us, was_clamped = _clamp_t0(t_min, t_max, t0_us_raw)
            if t0_us is None:
                logger.warning("clip %s: too short for a full event window, skipping", clip_id)
                continue

            history_us = t0_us + np.arange(-(NUM_HISTORY_STEPS - 1), 1) * TIME_STEP_S * 1e6
            future_us = t0_us + np.arange(1, NUM_FUTURE_STEPS + 1) * TIME_STEP_S * 1e6

            hist_xyz, hist_rot = _interp_state(ego_df, history_us)
            fut_xyz, _ = _interp_state(ego_df, future_us)
            t0_xyz = hist_xyz[-1]
            t0_rot_inv = hist_rot[-1].inv()

            hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
            fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
            hist_rot_local = (t0_rot_inv * hist_rot).as_matrix()

            image_us = t0_us + np.arange(-(NUM_FRAMES - 1), 1) * TIME_STEP_S * 1e6
            try:
                frames = np.stack(
                    [_decode_camera_frames(cam_bytes[feat], image_us) for feat in CAMERA_FEATURES]
                )  # (4 cams, 4 frames, H, W, 3)
            except Exception as exc:
                logger.warning("clip %s: camera decode failed: %s", clip_id, exc)
                continue
            frame_t = torch.from_numpy(
                np.ascontiguousarray(frames.transpose(0, 1, 4, 2, 3))
            )  # (4, 4, 3, H, W)

            items.append(
                {
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "event_cluster": event_cluster,
                    "group": group,
                    "was_clamped": was_clamped,
                    "event_idx": event_idx * 1000 + sub_idx,
                    "event_coc": str(ev.get("coc", "")),
                    "model_inputs": {
                        "image_frames": frame_t,
                        "camera_indices": CAMERA_INDICES,
                        "ego_history_xyz": torch.from_numpy(hist_xyz_local.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0),
                        "ego_history_rot": torch.from_numpy(hist_rot_local.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0),
                        "ego_future_xyz": torch.from_numpy(fut_xyz_local.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0),
                    },
                }
            )
    return items
