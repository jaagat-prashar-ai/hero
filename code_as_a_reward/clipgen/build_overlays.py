# SPDX-License-Identifier: Apache-2.0
"""Build per-clip scene-overlay images for the clipgen generator.

For each manifest clip: fetch the front-wide camera frame at t0 from the
pai_warm_cache chunk zips (S3 ranged reads -- the zips are ~2 GB each, so
only the needed members are pulled), project the GT waypoints onto it with
the f-theta model, draw the trajectory polyline, and write
<clip_id>.overlay.jpg next to the clip's other data files. The manifest
gains an "overlay_jpeg" key per entry.

The generator attaches this image to step 1 of the chain so scene
understanding (what the hazard IS, where it sits in view, how the expert's
path responds) is a factor in the reward function it designs -- the text
dossier alone repeatedly left the model guessing at scene semantics
(b7f37a71 cone scene reasoned as a lane change; curvature/lateral
confusion).

Projection math: ported from rl_posttrain/rewards/scene_overlay.py (itself
a port of the vendored viz.py project_waypoints_ftheta) rather than
imported -- scene_overlay pulls scipy at module import, which this repo's
default local env does not carry; the quaternion here follows the same
scipy xyzw convention, unit-tested against known rotations.

Run locally (one-off; the workload consumes the committed JPEGs):
    AWS_PROFILE=oci.chi python3 -m code_as_a_reward.clipgen.build_overlays \
        code_as_a_reward/clipgen/data15/manifest.json
Requires: boto3, pandas, numpy, PIL, ffmpeg on PATH.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from code_as_a_reward.clipgen import dossier as dossier_mod

WARM_CACHE = "pai_warm_cache/ood_dense_100chunks"
BUCKET = "research-datasets-chicago"
CAMERA = "camera_front_wide_120fov"
_INTR_COLUMNS = ("width", "height", "cx", "cy", "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4")
_EXTR_COLUMNS = ("qx", "qy", "qz", "qw", "x", "y", "z")
# gpt-4o tiles images at 512 px; 1024 on the long side keeps two tiles of
# detail without inflating the request (Claude's cap is 1568 -- also fine).
_MAX_IMAGE_DIM = 1024
_OVERLAY_COLOR = (255, 160, 40)
_OVERLAY_OUTLINE = (20, 20, 20)


class S3RangedFile:
    """Seekable read-only file over one S3 object via ranged GETs, so
    zipfile can walk a 2 GB chunk zip's central directory and extract
    single members without downloading the archive."""

    def __init__(self, client, bucket: str, key: str):
        self.client, self.bucket, self.key = client, bucket, key
        self.size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        resp = self.client.get_object(
            Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}"
        )
        data = resp["Body"].read()
        self.pos += len(data)
        return data


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (scipy Rotation.from_quat
    convention, matching scene_overlay.py's use of the calibration rows)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def project_waypoints_ftheta(
    waypoints_xyz: np.ndarray,
    cam_intr: dict[str, float],
    cam_extr: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """(T, 3) ego-frame-at-t0 waypoints -> (T, 2) pixels + (T,) validity.
    Same math as scene_overlay.project_waypoints_ftheta."""
    wp = np.asarray(waypoints_xyz, dtype=np.float64)
    rot = quat_to_matrix(*(cam_extr[k] for k in ("qx", "qy", "qz", "qw")))
    cam_t = np.array([cam_extr[k] for k in ("x", "y", "z")], dtype=np.float64)

    cam_points = (wp - cam_t) @ rot
    x, y, z = cam_points.T
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r_xy, z)
    radius = sum(cam_intr[f"fw_poly_{i}"] * theta**i for i in range(5))
    scale = radius / (r_xy + 1e-8)
    u = cam_intr["cx"] + x * scale
    v = cam_intr["cy"] + y * scale

    pixels = np.stack([u, v], axis=1)
    with np.errstate(invalid="ignore"):
        valid = (
            (z > 0)
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0)
            & (u < cam_intr["width"])
            & (v >= 0)
            & (v < cam_intr["height"])
        )
    return pixels, valid


def render_overlay(
    frame: Image.Image,
    waypoints_xyz: np.ndarray,
    cam_intr: dict[str, float],
    cam_extr: dict[str, float],
    max_dim: int = _MAX_IMAGE_DIM,
) -> Image.Image:
    """Draw the trajectory polyline on the frame (scene_overlay's visual
    language: orange line + every-4th-waypoint dots), then downscale."""
    pixels, valid = project_waypoints_ftheta(waypoints_xyz, cam_intr, cam_extr)
    sx = frame.width / float(cam_intr["width"])
    sy = frame.height / float(cam_intr["height"])
    pts = [(float(u) * sx, float(v) * sy) for (u, v), ok in zip(pixels, valid) if ok]

    draw = ImageDraw.Draw(frame)
    if len(pts) > 1:
        draw.line(pts, fill=_OVERLAY_COLOR, width=6, joint="curve")
    for u, v in pts[::4]:
        draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=_OVERLAY_COLOR, outline=_OVERLAY_OUTLINE)

    if max(frame.size) > max_dim:
        ratio = max_dim / max(frame.size)
        frame = frame.resize((round(frame.width * ratio), round(frame.height * ratio)), Image.LANCZOS)
    return frame


def waypoints_xyz_from_egomotion(df: pd.DataFrame, hz: float) -> np.ndarray:
    """(N, 3) waypoints on the same 1/hz grid as the dossier's 2D ones,
    keeping z (small but real over a 20 s clip) for the projection."""
    xy = dossier_mod.waypoints_from_egomotion(df, hz=hz)
    ts = df["timestamp" if "timestamp" in df.columns else "timestamp_us"].to_numpy(np.float64)
    ts = (ts - ts[0]) / (1e6 if ts.max() > 1e6 else 1.0)
    grid = np.arange(0.0, float(ts[-1]), 1.0 / hz)[: len(xy)]
    z = np.interp(grid, ts, df["z"].to_numpy(np.float64)) if "z" in df.columns else np.zeros(len(xy))
    return np.column_stack([xy, z[: len(xy)]])


def extract_t0_frame(mp4_path: Path, frame_index: int) -> Image.Image:
    """Decode exactly one frame by index via ffmpeg (no cv2 dependency)."""
    out = mp4_path.with_suffix(".t0.png")
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select=eq(n\\,{frame_index})",
            "-frames:v", "1", str(out),
        ],
        check=True,
    )
    return Image.open(out).convert("RGB")


def _read_parquet_s3(client, key: str) -> pd.DataFrame:
    """Whole-object fetch for the small parquets (index/calibration are
    tens of KB); pyarrow wants a real file, not S3RangedFile."""
    import io

    body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def build_all(manifest_path: str) -> None:
    import boto3

    client = boto3.client("s3")
    manifest_file = Path(manifest_path)
    data_dir = manifest_file.parent
    entries = json.loads(manifest_file.read_text())

    chunk_index = _read_parquet_s3(client, f"{WARM_CACHE}/clip_index.parquet")
    calib_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for entry in entries:
        clip_id = entry["clip_id"]
        chunk = int(chunk_index.loc[clip_id, "chunk"])
        if chunk not in calib_cache:
            intr = _read_parquet_s3(
                client,
                f"{WARM_CACHE}/calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk:04d}.parquet",
            )
            extr = _read_parquet_s3(
                client,
                f"{WARM_CACHE}/calibration/sensor_extrinsics/sensor_extrinsics.chunk_{chunk:04d}.parquet",
            )
            calib_cache[chunk] = (intr, extr)
        intr, extr = calib_cache[chunk]
        cam_intr = {k: float(intr.loc[(clip_id, CAMERA)][k]) for k in _INTR_COLUMNS}
        cam_extr = {k: float(extr.loc[(clip_id, CAMERA)][k]) for k in _EXTR_COLUMNS}

        zf = zipfile.ZipFile(S3RangedFile(
            client, BUCKET, f"{WARM_CACHE}/camera/{CAMERA}/{CAMERA}.chunk_{chunk:04d}.zip"
        ))
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(zf.extract(f"{clip_id}.{CAMERA}.mp4", tmp))
            import io as _io

            frame_ts = pd.read_parquet(
                _io.BytesIO(zf.read(f"{clip_id}.{CAMERA}.timestamps.parquet"))
            ).to_numpy(np.float64).reshape(-1)
            ego = pd.read_parquet(data_dir / f"{clip_id}.egomotion.offline.parquet")
            # The waypoints start at the egomotion's FIRST sample (see
            # waypoints_from_egomotion: timestamps re-zeroed to ts[0]), so
            # t0 = clip start; align on offsets from each stream's own start
            # rather than trusting the two files to share an epoch.
            frame_index = int(np.argmin(np.abs(frame_ts - frame_ts[0])))
            frame = extract_t0_frame(mp4, frame_index)

        hz = float(entry.get("hz", 10.0))
        wp3 = waypoints_xyz_from_egomotion(ego, hz)
        overlay = render_overlay(frame, wp3, cam_intr, cam_extr)
        out_path = data_dir / f"{clip_id}.overlay.jpg"
        overlay.save(out_path, format="JPEG", quality=88)
        entry["overlay_jpeg"] = str(out_path)
        n_vis = int(np.sum(project_waypoints_ftheta(wp3, cam_intr, cam_extr)[1]))
        print(f"{clip_id}: chunk {chunk:4d} frame {frame_index:3d} "
              f"{n_vis}/{len(wp3)} waypoints visible -> {out_path.name}", flush=True)

    manifest_file.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"manifest updated: {manifest_file}")


if __name__ == "__main__":
    build_all(sys.argv[1])
