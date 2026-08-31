# SPDX-License-Identifier: Apache-2.0
"""Small loader for the 214-clip Track 1 test-split WDS shards built by
build_wds/data/build_test_split.py at
s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/test/.

Yields one load_physical_aiavdataset()-shaped dict per (clip_id, event_idx),
ready to hand straight to perplexity/dump_input_template.py's build_prompt().
Frames are picked using the real per-camera frame_timestamps sidecar (see
BUGS.md 2026-08-07 "3rd" entry) instead of the "last N frames by index"
placeholder perplexity/s3_clip_loader.py had to use before that fix existed.
"""

import io
import json

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
import webdataset as wds
from physical_ai_av.egomotion import EgomotionState
from physical_ai_av.video import SeekVideoReader

from perplexity.s3_clip_loader import ClipWindowOutOfRangeError

CAMERA_KEYS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
CAMERA_INDICES = torch.tensor([0, 1, 2, 6], dtype=torch.int64)


def _egomotion_interpolator_from_bytes(egomotion_bytes: bytes):
    df = pd.read_parquet(io.BytesIO(egomotion_bytes))
    state = EgomotionState.from_egomotion_df(df)
    return state.create_interpolator(df["timestamp_us"].to_numpy(copy=True))


def _decode_frames_at_t0(mp4_bytes: bytes, timestamps_bytes: bytes, t0_us: int, num_frames: int = 4) -> np.ndarray:
    """Decode the num_frames frames ending at/before t0_us, using the real
    per-frame timestamp sidecar (column "timestamp", same clock as egomotion's
    timestamp_us -- both come from the same clip capture)."""
    frame_timestamps = pd.read_parquet(io.BytesIO(timestamps_bytes))["timestamp"].to_numpy()
    end_idx = int(np.searchsorted(frame_timestamps, t0_us, side="right")) - 1
    if end_idx < num_frames - 1:
        raise ValueError(f"t0_us={t0_us} is before frame {num_frames - 1} of this camera")

    frame_idxs = np.arange(end_idx - num_frames + 1, end_idx + 1, dtype=np.int64)
    reader = SeekVideoReader(video_data=io.BytesIO(mp4_bytes), timestamps=None)
    images = reader.decode_images_from_frame_indices(frame_idxs)
    reader.close()
    return images


def _ego_windows(
    egomotion,
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Ego-local (relative-to-t0) history/future windows -- same windowing +
    t0-relative transform as load_physical_aiavdataset.py /
    perplexity/s3_clip_loader.py's load_clip_from_s3_extract."""
    history_time_range_us = num_history_steps * time_step * 1_000_000
    if t0_us <= history_time_range_us:
        raise ValueError(f"{t0_us=} must be greater than the history time range ({history_time_range_us=} us)")

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)

    try:
        ego_history = egomotion(t0_us + history_offsets_us)
        ego_future = egomotion(t0_us + future_offsets_us)
    except ValueError as e:
        raise ClipWindowOutOfRangeError(
            f"t0_us={t0_us} window doesn't fit the egomotion track: {e}"
        ) from e

    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1]
    t0_rot_inv = spt.Rotation.from_quat(ego_history_quat[-1]).inv()

    return {
        "ego_history_xyz": torch.from_numpy(t0_rot_inv.apply(ego_history_xyz - t0_xyz)).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy((t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()).float().unsqueeze(0).unsqueeze(0),
        "ego_future_xyz": torch.from_numpy(t0_rot_inv.apply(ego_future_xyz - t0_xyz)).float().unsqueeze(0).unsqueeze(0),
        "ego_future_rot": torch.from_numpy((t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()).float().unsqueeze(0).unsqueeze(0),
    }


def iter_test_samples(shard_paths: list[str]):
    """Yield one dict per (clip_id, event_idx) across the given local shard_*.tar paths.

    Each dict has: image_frames (4, 4, 3, H, W), ego_history_xyz, ego_history_rot,
    ego_future_xyz, ego_future_rot, t0_us, clip_id, event_idx, submission_key
    (f"{clip_id}_{event_idx}", matching the benchmark's submission key format).
    """
    dataset = wds.WebDataset([str(p) for p in shard_paths], shardshuffle=False)
    avdi = None

    for sample in dataset:
        clip_id = sample["__key__"]
        meta = json.loads(sample["json"])
        ood_events = meta.get("ood_events") or []
        if not ood_events:
            continue
        events = json.loads(ood_events[0]["events"])
        if not events:
            continue

        if "egomotion.parquet" in sample:
            egomotion = _egomotion_interpolator_from_bytes(sample["egomotion.parquet"])
        else:
            # A small number of test-shard samples were written after the
            # builder's best-effort egomotion fetch failed. Keep S3 as the
            # primary source, but recover only the missing feature from the
            # canonical PhysicalAI dataset instead of dropping a submission
            # key (Track 1 requires all 284 events).
            if avdi is None:
                import physical_ai_av

                avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
            print(f"recovering missing egomotion for {clip_id}", flush=True)
            egomotion = avdi.get_clip_feature(
                clip_id,
                feature=avdi.features.LABELS.EGOMOTION,
                maybe_stream=True,
            )

        for event_idx, event in enumerate(events):
            t0_us = event["event_start_timestamp"]
            try:
                windows = _ego_windows(egomotion, t0_us)
                image_frames = torch.stack(
                    [
                        torch.from_numpy(
                            _decode_frames_at_t0(
                                sample[f"{cam}.mp4"], sample[f"{cam}.timestamps.parquet"], t0_us
                            )
                        )
                        for cam in CAMERA_KEYS
                    ],
                    dim=0,
                ).permute(0, 1, 4, 2, 3)
            except (ValueError, ClipWindowOutOfRangeError) as exc:
                print(f"skipping {clip_id}_{event_idx}: {exc}")
                continue

            yield {
                "image_frames": image_frames,
                "camera_indices": CAMERA_INDICES.clone(),
                **windows,
                "t0_us": t0_us,
                "clip_id": clip_id,
                "event_idx": event_idx,
                "submission_key": f"{clip_id}_{event_idx}",
            }
