# T1.3 (S3 fallback): reconstruct load_physical_aiavdataset()'s output schema
# from a clip already extracted out of one of our own S3 WDS shards, instead
# of streaming it from HF (whose Xet CDN was hanging).
#
# This is NOT byte-for-byte identical to the true HF source, and callers must
# know that:
#   - Camera video was lossy-transcoded to AV1 (crf=32) by build_webdataset.py
#     before upload. Pixel content will differ from the original HF MP4.
#   - The per-frame capture timestamps (a separate "frame_timestamps" parquet
#     in HF's own zip layout) were never extracted into the WDS shard, so we
#     cannot reproduce physical_ai_av's timestamp-based frame selection
#     (SeekVideoReader.decode_images_from_timestamps). decode_last_n_frames()
#     below picks frames by INDEX instead, and this is a much bigger gap than
#     "slightly different pixels": on the reference clip
#     (030c760c-ae38-49aa-9ad8-f5650a545d26) the video is 604 frames @ 30fps
#     (~20.1s), so "last 4 frames" sits at t~20.0-20.1s -- but t0_us=5.1e6
#     means the real demo wants frames at t~4.8-5.1s. That's ~15s away: a
#     DIFFERENT MOMENT of the drive entirely, not just a recompressed version
#     of the right one. We cannot fix this with index arithmetic either --
#     that would require knowing what absolute-clock time video frame 0
#     corresponds to, which is exactly what the missing frame_timestamps file
#     would tell us. Checking the egomotion timestamps for this same clip
#     (span -0.2s to +140.2s) shows egomotion is logged over a much longer
#     window than the ~20s video, so "frame 0 == absolute time 0" is NOT a
#     safe assumption -- it would just be a different unverified guess.
#   - Egomotion IS exact: it's stored as raw per-sample floats (not derived/
#     lossy), so EgomotionState.from_egomotion_df() + create_interpolator()
#     on this data reconstructs the identical Interpolator physical_ai_av's
#     own HF-streaming path would build from the same clip. VERIFIED, not
#     just argued from reading the code: for 030c760c-...,
#     load_egomotion_interpolator() on the S3 parquet vs. avdi.get_clip_feature
#     (true HF stream) evaluated at the real history_timestamps gives 0.0 max
#     abs difference on xyz, quaternion, and curvature. Video resolution is
#     also confirmed unaffected by the transcode -- build_wds/data/
#     video_transcode.py's ffmpeg invocation has no resize/scale filter, only
#     codec/crf/preset -- so the vision-token SPAN LENGTH matches the true HF
#     path too, on top of the input_ids-structure argument below.
#
# Net effect: ego-history tokens (what fuse_traj_tokens actually encodes into
# input_ids) are exact, and the vision-token SPAN LENGTH in input_ids is still
# exact too, because Qwen3-VL's image token count is a deterministic function
# of image resolution/count, not pixel content or capture time -- so T1.3's
# byte-for-byte input_ids check is unaffected. But the decoded image_frames
# tensor this module returns does NOT depict the same moment described by the
# ego-history tokens in the same prompt. DO NOT reuse this loader for
# anything where visual content matters (qualitative checks, real inference,
# any vision-level teacher forcing) without first fixing frame alignment --
# only for input_ids-structure work like T1.3. See dump_input_template.py's
# fixture "data_source"/"caveats" fields for how this gets recorded.

import io

import av
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from physical_ai_av.egomotion import EgomotionState
from physical_ai_av.video import SeekVideoReader


def load_egomotion_interpolator(parquet_path: str):
    """Rebuild the exact Interpolator[EgomotionState] load_physical_aiavdataset gets from HF.

    Column names here match build_webdataset.py's _egomotion_to_bytes() output,
    which was deliberately written to match what EgomotionState.from_egomotion_df()
    expects (see that function's docstring) -- so this is not a reimplementation
    of the interpolation math, just re-supplying its input from a different source.
    """
    df = pd.read_parquet(parquet_path)
    state = EgomotionState.from_egomotion_df(df)
    return state.create_interpolator(df["timestamp_us"].to_numpy(copy=True))


def decode_last_n_frames(mp4_path: str, num_frames: int = 4) -> np.ndarray:
    """Decode the last `num_frames` frames of a video by index, not timestamp.

    We use physical_ai_av's own SeekVideoReader for the actual decode (real
    vendored decoder, not a reimplementation) but skip
    decode_images_from_timestamps entirely, since we have no per-frame
    timestamps for this WDS-sourced video (see module docstring -- this is a
    real content/timing gap, not just a cosmetic one). Frame COUNT and
    resolution are what input_ids structure depends on, and those are correct
    regardless of which frames we pick; the actual frame CONTENT picked here
    is very likely from the wrong moment of the clip.
    """
    with open(mp4_path, "rb") as f:
        video_bytes = f.read()
    reader = SeekVideoReader(video_data=io.BytesIO(video_bytes), timestamps=None)
    total_frames = reader.container.streams.video[0].frames
    if not total_frames:
        # Some containers don't report a frame count in metadata; fall back to
        # demuxing to count packets (cheap: no decoding, matches
        # SeekVideoReader._build_keyframe_index's own demux-only approach).
        total_frames = sum(1 for _ in reader.container.demux(reader.stream) if True)
        reader.container.seek(reader.start_time_pts, any_frame=True, backward=True, stream=reader.stream)
    frame_idxs = np.arange(total_frames - num_frames, total_frames, dtype=np.int64)
    images = reader.decode_images_from_frame_indices(frame_idxs)
    reader.close()
    return images


def load_clip_from_s3_extract(
    clip_dir: str,
    clip_id: str,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
    camera_keys: tuple[str, ...] = (
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    ),
) -> dict:
    """Reproduce load_physical_aiavdataset()'s output dict from an S3-extracted clip.

    Mirrors alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py's egomotion
    windowing + t0-relative coordinate transform exactly (same math, copied
    from the vendored function) -- the only thing that differs is where the
    Interpolator and video frames come from. See module docstring for the
    fidelity caveats on the video side.
    """
    egomotion = load_egomotion_interpolator(f"{clip_dir}/{clip_id}.egomotion.parquet")

    history_time_range_us = num_history_steps * time_step * 1_000_000
    if t0_us <= history_time_range_us:
        raise ValueError(f"{t0_us=} must be greater than the history time range ({history_time_range_us=} us)")

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_frames_list = [
        torch.from_numpy(decode_last_n_frames(f"{clip_dir}/{clip_id}.{cam_key}.mp4", num_frames))
        for cam_key in camera_keys
    ]
    # (N_cameras, num_frames, H, W, 3) -> (N_cameras, num_frames, 3, H, W)
    image_frames = torch.stack(image_frames_list, dim=0).permute(0, 1, 4, 2, 3)

    return {
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "t0_us": t0_us,
        "clip_id": clip_id,
        "data_source": "s3_wds_shard",
    }
