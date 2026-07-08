# SPDX-License-Identifier: Apache-2.0
"""
render_trajectory_overlay.py — project a sampled rollout's future trajectory
onto the actual dashcam video it was conditioned on, so a reviewer can watch
the model's chosen path rather than only read waypoint stats and reasoning
text (see noise_floor_report.py for the latter). Validated as a one-off
proof of concept against scene 00bbc8b2-7d40-40f7-a1b3-a5853fe5bddc_12206610
before being turned into this reusable tool.

Camera model: NVIDIA's own physical_ai_av.utils.camera_models.FThetaCameraModel
covers exactly the wide-FOV fisheye lenses this dataset uses, but that
package needs Python>=3.11 (typing.Self) and this project's masking worker
runs 3.10 -- ray2pixel is reimplemented here from its published formula
rather than imported. The intrinsics polynomial coefficients it needs are
regex-parsed back out of build_webdataset.py's stringified th2r/r2th (see
parse_ftheta_polynomial's docstring), since that's the only form already
sitting in every clip's calibration.json.

Egomotion handling: a rollout's waypoints are in the EGO-LOCAL frame AT t0
(same convention as masking/data/wds_dataset.py). Since the vehicle keeps
moving after t0, each output frame at time t re-expresses the (fixed) future
trajectory in THAT frame's own ego-local coordinates via the relative pose
between t0 and t, read from egomotion.parquet -- otherwise the overlay would
visibly drift out of alignment with the road as the video plays.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import av
import boto3
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from pref_pairs.fetch_from_logs import parse_marked_lines
from pref_pairs.training.run import ROLLOUT_FULL_LOG_MARKER

logger = logging.getLogger(__name__)

TAR_BLOCK = 512
BUCKET = "research-datasets-chicago"
OOD_REASONING_KEY = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/metadata/ood_reasoning.parquet"

# Same local-dev fallback build_wds/data/build_webdataset.py uses -- on
# Lilypad, real creds + AWS_ENDPOINT_URL_S3 are already in the environment.
if not os.environ.get("AWS_ACCESS_KEY_ID"):
    os.environ.setdefault("AWS_PROFILE", "oci.chi")


def _s3_client():
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def fetch_clip_files(
    s3, bucket: str, shard_key: str, group_start_offset: int, clip_id: str, camera: str
) -> dict[str, bytes]:
    """Walk one shard's tar headers starting at the clip's known group-start
    offset (from a sampling manifest, e.g. sample_clips.py's scan_shard_for_clips
    output) and pull just the 3 members this tool needs, stopping as soon as
    they're all found or the clip's own member group ends."""
    wanted = {
        f"{clip_id}.calibration.json": "calibration",
        f"{clip_id}.egomotion.parquet": "egomotion",
        f"{clip_id}.{camera}.mp4": "video",
    }
    found: dict[str, bytes] = {}
    pos = group_start_offset
    while wanted:
        hdr = _range_get(s3, bucket, shard_key, pos, TAR_BLOCK)
        if len(hdr) < TAR_BLOCK or hdr == b"\x00" * TAR_BLOCK:
            break
        name = hdr[0:100].split(b"\x00")[0].decode(errors="replace")
        size_field = hdr[124:136].split(b"\x00")[0].strip()
        size = int(size_field, 8) if size_field else 0
        data_start = pos + TAR_BLOCK
        if name in wanted:
            found[wanted.pop(name)] = _range_get(s3, bucket, shard_key, data_start, size)
        elif "@PaxHeader" not in name and not name.startswith(clip_id):
            break  # walked past this clip's own member group
        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK
    if wanted:
        raise RuntimeError(
            f"clip {clip_id}: could not find {list(wanted.values())} in "
            f"{shard_key} starting at offset {group_start_offset}"
        )
    return found


def get_camera_for_clip(s3, clip_id: str) -> str:
    """Looks up which camera feature the clip's OOD tag in
    reasoning/ood_reasoning.parquet was keyed to -- the camera the reasoning
    text actually describes, matching the AskUserQuestion decision to overlay
    on whichever camera the scene's own tag used rather than a fixed default."""
    data = s3.get_object(Bucket=BUCKET, Key=OOD_REASONING_KEY)["Body"].read()
    df = pd.read_parquet(io.BytesIO(data))
    if clip_id not in df.index:
        raise ValueError(f"clip {clip_id} not found in ood_reasoning.parquet")
    return df.loc[clip_id, "feature"]


# ---- FThetaCameraModel, reimplemented (see module docstring) --------------

_SUPERSCRIPT_POWERS = {None: 1, "": 1, "²": 2, "³": 3, "⁴": 4}
_TERM_RE = re.compile(r"([+-]?)\s*(?:\(([\d.eE+-]+)\)|([\d.]+))(?:\s*·\s*[a-z]+(²|³|⁴)?)?")


def parse_ftheta_polynomial(expr: str, symbol: str) -> np.polynomial.Polynomial:
    """Parses build_webdataset.py's str(numpy.polynomial.Polynomial) output
    (degree-4, e.g. "0.0 + 931.87·th + 28.17·th² - 66.33·th³ +\\n20.03·th⁴",
    small coefficients wrapped in parens for scientific notation, e.g.
    "- (3.90e-08)·r²") back into actual float coefficients. This is the same
    class of lossy stringification as build_wds/data/build_webdataset.py's
    events-field bug (see _serialize_ood_event_field) -- here it's tolerated
    with a parser instead of a source fix, since re-deriving calibration from
    physical_ai_av directly would require Python>=3.11."""
    text = expr.replace("\n", " ")
    coeffs = [0.0] * 5
    for m in _TERM_RE.finditer(text):
        sign, paren_num, plain_num, power_sym = m.groups()
        num_str = paren_num if paren_num is not None else plain_num
        if num_str is None:
            continue
        value = float(num_str)
        if sign == "-":
            value = -value
        power = _SUPERSCRIPT_POWERS[power_sym] if symbol in m.group(0) else 0
        coeffs[power] = value
    return np.polynomial.Polynomial(coeffs, symbol=symbol)


def ftheta_ray2pixel(ray: np.ndarray, principal_point: np.ndarray, th2r: np.polynomial.Polynomial) -> np.ndarray:
    """Projects rays in camera frame (Z out of camera, X right, Y down) to
    pixel coordinates. Ported directly from
    physical_ai_av.utils.camera_models.FThetaCameraModel.ray2pixel."""
    th = np.arccos(ray[..., 2:3] / np.linalg.norm(ray, axis=-1, keepdims=True))
    xy_norm = ray[..., :2] / np.linalg.norm(ray[..., :2], axis=-1, keepdims=True)
    return principal_point + th2r(th) * xy_norm


def load_camera_model(calibration: dict, camera: str) -> tuple[np.ndarray, np.polynomial.Polynomial, int, int]:
    cam = calibration["camera_intrinsics"]["camera_models"][camera]
    principal_point = np.array(cam["principal_point"])
    th2r = parse_ftheta_polynomial(cam["th2r"], "th")
    return principal_point, th2r, cam["width"], cam["height"]


def load_extrinsics(calibration: dict, camera: str) -> tuple[Rotation, np.ndarray]:
    pose = calibration["sensor_extrinsics"]["sensor_poses"][camera]
    return Rotation.from_quat(pose["rotation_quat_xyzw"]), np.array(pose["translation"])


# ---- rollout fetch from logs ------------------------------------------------

_INFO_RE = re.compile(r"^(Created At|Finished At)\s+(.+)$", re.M)
_TZ_OFFSETS = {"PDT": -7, "PST": -8, "UTC": 0}


def _parse_workload_timestamp(text: str) -> datetime:
    dt_str, tz_abbr = text.rsplit(" ", 1)
    naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    offset = _TZ_OFFSETS.get(tz_abbr)
    if offset is None:
        raise ValueError(f"unrecognized timezone abbreviation {tz_abbr!r} in workload info output")
    return naive.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(timezone.utc)


def get_workload_time_window(workload_id: str, pad_minutes: int = 10) -> tuple[datetime, datetime]:
    """`lilypad workload logs` defaults to "last 4 hours from now", which
    silently returns nothing for a job that ran even a day earlier (hit this
    for real on pref-pairs-action-variance-cluster-ji84ic). Deriving the
    actual window from `lilypad workload info`'s Created At / Finished At
    avoids repeating that failure mode."""
    result = subprocess.run(["lilypad", "workload", "info", workload_id], capture_output=True, text=True, check=True)
    times = dict(_INFO_RE.findall(result.stdout))
    if "Created At" not in times or "Finished At" not in times:
        raise RuntimeError(f"could not find Created At / Finished At in `lilypad workload info {workload_id}` output")
    start = _parse_workload_timestamp(times["Created At"]) - timedelta(minutes=pad_minutes)
    end = _parse_workload_timestamp(times["Finished At"]) + timedelta(minutes=pad_minutes)
    return start, end


def fetch_scene_rollouts(workload_id: str, scene_id: str) -> list[dict[str, Any]]:
    start, end = get_workload_time_window(workload_id)
    result = subprocess.run(
        [
            "lilypad", "workload", "logs", workload_id,
            "--content-filter", ROLLOUT_FULL_LOG_MARKER,
            "--start-time", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--end-time", end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
        capture_output=True, text=True, check=True,
    )
    rows = parse_marked_lines(result.stdout, ROLLOUT_FULL_LOG_MARKER)
    return [r for r in rows if r["scene_id"] == scene_id]


def pick_representative_rollout(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Picks a rollout from the majority maneuver_class -- "representative"
    only means "not an outlier maneuver," not a claim about being the
    single most typical rollout within that class."""
    if not rollouts:
        raise ValueError("no rollouts to pick from")
    counts: dict[str, int] = {}
    for r in rollouts:
        counts[r["maneuver_class"]] = counts.get(r["maneuver_class"], 0) + 1
    majority_class = max(counts, key=counts.get)
    return next(r for r in rollouts if r["maneuver_class"] == majority_class)


# ---- ego motion + rendering -------------------------------------------------

def interp_ego_pose(ego_df: pd.DataFrame, query_us: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """Same convention as masking/data/wds_dataset.py's _interp_state."""
    t = ego_df["timestamp_us"].to_numpy()
    xyz = np.stack([np.interp(query_us, t, ego_df[c].to_numpy()) for c in ("x", "y", "z")], axis=-1)
    quats = ego_df[["qx", "qy", "qz", "qw"]].to_numpy()
    slerp = Slerp(t, Rotation.from_quat(quats))
    rot = slerp(np.clip(query_us, t.min(), t.max()))
    return xyz, rot


def render_overlay_frames(
    video_bytes: bytes,
    ego_df: pd.DataFrame,
    waypoints_t0: np.ndarray,
    t0_us: int,
    dt_s: float,
    principal_point: np.ndarray,
    th2r: np.polynomial.Polynomial,
    width: int,
    height: int,
    R_cam2ego: Rotation,
    t_cam2ego: np.ndarray,
    frame_stride: int = 5,
) -> list[np.ndarray]:
    dt_us = int(dt_s * 1e6)
    future_t_us = t0_us + np.arange(1, len(waypoints_t0) + 1) * dt_us
    t0_xyz, t0_rot = interp_ego_pose(ego_df, np.array([t0_us]))
    t0_xyz, t0_rot = t0_xyz[0], t0_rot[0]
    world_xyz = t0_xyz + t0_rot.apply(waypoints_t0)

    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    out_frames = []
    for fi in range(0, len(waypoints_t0), frame_stride):
        t_us = future_t_us[fi]
        t_xyz, t_rot = interp_ego_pose(ego_df, np.array([t_us]))
        t_xyz, t_rot = t_xyz[0], t_rot[0]
        # re-express the (fixed) future path in ego-local coords AT this frame's time
        local_at_t = t_rot.inv().apply(world_xyz - t_xyz)
        # ego -> camera frame. R_cam2ego already encodes the full axis change
        # from ego (X-fwd, Y-left, Z-up) to camera-native (X-right, Y-down,
        # Z-forward) -- confirmed against real calibration data, no extra
        # manual axis permutation needed.
        cam_frame = R_cam2ego.inv().apply(local_at_t - t_cam2ego)
        in_front = cam_frame[:, 2] > 0.1
        pixels = ftheta_ray2pixel(cam_frame, principal_point, th2r)

        target_sec = t_us / 1e6
        container.seek(int(target_sec / time_base), stream=stream, any_frame=False, backward=True)
        last = None
        for frame in container.decode(stream):
            last = frame
            if frame.time is not None and frame.time >= target_sec:
                break
        img = Image.fromarray(last.to_ndarray(format="rgb24"))
        draw = ImageDraw.Draw(img)
        pts = [(float(x), float(y)) for (x, y), ok in zip(pixels, in_front) if ok and 0 <= x < width and 0 <= y < height]
        if len(pts) > 1:
            draw.line(pts, fill=(255, 160, 40), width=6, joint="curve")
        for x, y in pts[::4]:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 160, 40), outline=(20, 20, 20))
        out_frames.append(np.asarray(img))
    container.close()
    return out_frames


def write_mp4(frames: list[np.ndarray], out_path: str | Path, fps: int = 2) -> None:
    out_container = av.open(str(out_path), mode="w")
    out_stream = out_container.add_stream("libx264", rate=fps)
    out_stream.width = frames[0].shape[1]
    out_stream.height = frames[0].shape[0]
    out_stream.pix_fmt = "yuv420p"
    for fr in frames:
        for packet in out_stream.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out_container.mux(packet)
    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()


def render(
    workload_id: str, clip_id: str, scene_id: str, shard_key: str, group_start_offset: int,
    rollout_id: int | None, camera: str | None, out_path: str | Path,
) -> None:
    s3 = _s3_client()
    camera = camera or get_camera_for_clip(s3, clip_id)
    logger.info("clip %s: using camera %s", clip_id, camera)

    files = fetch_clip_files(s3, BUCKET, shard_key, group_start_offset, clip_id, camera)
    calibration = json.loads(files["calibration"].decode())
    ego_df = pd.read_parquet(io.BytesIO(files["egomotion"]))
    principal_point, th2r, width, height = load_camera_model(calibration, camera)
    R_cam2ego, t_cam2ego = load_extrinsics(calibration, camera)

    rollouts = fetch_scene_rollouts(workload_id, scene_id)
    if not rollouts:
        raise ValueError(f"no rollouts found in workload {workload_id} logs for scene {scene_id}")
    rollout = next((r for r in rollouts if r["rollout_id"] == rollout_id), None) if rollout_id is not None else None
    rollout = rollout or pick_representative_rollout(rollouts)
    logger.info("rendering rollout %d (%s): %r", rollout["rollout_id"], rollout["maneuver_class"], rollout["coc_text"])

    waypoints_t0 = np.array(rollout["waypoints"])
    t0_us = int(scene_id.rsplit("_", 1)[-1])
    frames = render_overlay_frames(
        files["video"], ego_df, waypoints_t0, t0_us, rollout["dt_s"],
        principal_point, th2r, width, height, R_cam2ego, t_cam2ego,
    )
    write_mp4(frames, out_path)
    logger.info("wrote %s (%d frames)", out_path, len(frames))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload_id", required=True, help="Lilypad workload id the rollout was logged under.")
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--scene_id", required=True, help="e.g. {clip_id}_{t0_us}")
    ap.add_argument("--shard_key", required=True, help="S3 key of the shard containing this clip.")
    ap.add_argument("--offset", type=int, required=True, help="Clip's group-start byte offset within the shard.")
    ap.add_argument("--rollout_id", type=int, default=None, help="Defaults to a rollout from the majority maneuver_class.")
    ap.add_argument("--camera", default=None, help="Defaults to the camera the scene's OOD tag used.")
    ap.add_argument("--out", default="overlay_output.mp4")
    args = ap.parse_args()

    render(
        args.workload_id, args.clip_id, args.scene_id, args.shard_key, args.offset,
        args.rollout_id, args.camera, args.out,
    )


if __name__ == "__main__":
    main()
