# SPDX-License-Identifier: Apache-2.0
"""
render_trajectory_overlay_test.py — covers the pure-logic pieces (polynomial
parsing, timestamp parsing, rollout selection, tar-walk) with synthetic
fixtures. The actual S3/lilypad-dependent render() pipeline was instead
validated once against a real workload/clip (see the module docstring) --
not repeated here since it needs live cluster/bucket access this sandbox
doesn't have and isn't a fit for a fast unit test anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from pref_pairs.render_trajectory_overlay import (
    _parse_workload_timestamp,
    fetch_clip_files,
    ftheta_ray2pixel,
    parse_ftheta_polynomial,
    pick_representative_rollout,
)


def test_parse_ftheta_polynomial_plain_decimals():
    # Real front_wide_120fov th2r string from this project's calibration.json.
    poly = parse_ftheta_polynomial(
        "0.0 + 931.87799034·th + 28.17536405·th² - 66.33972849·th³ +\n20.03706455·th⁴", "th",
    )
    assert list(poly.coef) == pytest.approx([0.0, 931.87799034, 28.17536405, -66.33972849, 20.03706455])


def test_parse_ftheta_polynomial_scientific_notation_in_parens():
    # Real front_wide_120fov r2th string -- small coefficients wrapped in
    # parens; the embedded "-" in "e-08" must not be mistaken for a term separator.
    poly = parse_ftheta_polynomial(
        "0.0 + 0.00107389·r - (3.90328473e-08)·r² + (9.27189561e-11)·r³ -\n(2.78311761e-14)·r⁴", "r",
    )
    assert list(poly.coef) == pytest.approx([0.0, 0.00107389, -3.90328473e-08, 9.27189561e-11, -2.78311761e-14])


def test_ftheta_ray2pixel_near_axis_lands_close_to_principal_point():
    # A ray exactly on the optical axis (x=y=0) is a genuine singularity in
    # this formula (0/0 normalizing ray[...,:2]) -- shared with NVIDIA's own
    # FThetaCameraModel, not something to paper over here. Real trajectory
    # points never land exactly on-axis, so this tests a near-axis ray instead.
    principal_point = np.array([960.0, 750.0])
    th2r = np.polynomial.Polynomial([0.0, 900.0, 0.0, 0.0, 0.0])  # th2r(0) == 0
    ray = np.array([[1e-6, 0.0, 1.0]])
    pixel = ftheta_ray2pixel(ray, principal_point, th2r)
    assert pixel[0] == pytest.approx(principal_point, abs=1e-3)


def test_ftheta_ray2pixel_off_axis_moves_away_from_principal_point():
    principal_point = np.array([960.0, 750.0])
    th2r = np.polynomial.Polynomial([0.0, 900.0, 0.0, 0.0, 0.0])
    ray = np.array([[0.3, 0.0, 1.0]])  # off to the right of center
    pixel = ftheta_ray2pixel(ray, principal_point, th2r)
    assert pixel[0, 0] > principal_point[0]  # moved right
    assert pixel[0, 1] == pytest.approx(principal_point[1])  # no vertical shift


def test_parse_workload_timestamp_handles_pdt_and_converts_to_utc():
    dt = _parse_workload_timestamp("2026-07-07 17:02:28 PDT")
    assert dt.hour == 0 and dt.day == 8  # PDT = UTC-7, so 17:02 PDT == 00:02 UTC next day
    assert dt.tzinfo is not None


def test_pick_representative_rollout_returns_majority_class():
    rollouts = [
        {"rollout_id": 0, "maneuver_class": "lane_keep"},
        {"rollout_id": 1, "maneuver_class": "lane_change_left"},
        {"rollout_id": 2, "maneuver_class": "lane_change_left"},
        {"rollout_id": 3, "maneuver_class": "lane_change_left"},
    ]
    picked = pick_representative_rollout(rollouts)
    assert picked["maneuver_class"] == "lane_change_left"


def _tar_header(name: str, size: int) -> bytes:
    hdr = bytearray(512)
    hdr[0:len(name)] = name.encode()
    size_field = f"{size:011o}\0".encode()
    hdr[124:124 + len(size_field)] = size_field
    return bytes(hdr)


class _FakeS3:
    """Serves range-GETs against an in-memory fake tar body."""
    def __init__(self, body: bytes):
        self.body = body

    def get_object(self, Bucket, Key, Range):
        start, end = Range.replace("bytes=", "").split("-")
        start, end = int(start), int(end)
        data = self.body[start:end + 1]
        return {"Body": _BytesBody(data)}


class _BytesBody:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data


def _pad_to_block(data: bytes) -> bytes:
    pad = (-len(data)) % 512
    return data + b"\x00" * pad


def test_fetch_clip_files_stops_at_clip_group_boundary():
    clip_id = "clip-a"
    calib_payload = b'{"fake": "calibration"}'
    ego_payload = b"fake-parquet-bytes"
    video_payload = b"fake-mp4-bytes"
    other_clip_payload = b"belongs to the next clip, should never be read"

    body = b"".join([
        _tar_header(f"{clip_id}.calibration.json", len(calib_payload)),
        _pad_to_block(calib_payload),
        _tar_header(f"{clip_id}.egomotion.parquet", len(ego_payload)),
        _pad_to_block(ego_payload),
        _tar_header(f"{clip_id}.camera_front_wide_120fov.mp4", len(video_payload)),
        _pad_to_block(video_payload),
        _tar_header("clip-b.calibration.json", len(other_clip_payload)),  # next clip's group
        _pad_to_block(other_clip_payload),
    ])
    s3 = _FakeS3(body)
    found = fetch_clip_files(s3, "bucket", "key", 0, clip_id, "camera_front_wide_120fov")
    assert found == {"calibration": calib_payload, "egomotion": ego_payload, "video": video_payload}


def test_fetch_clip_files_raises_if_a_member_is_missing():
    clip_id = "clip-a"
    calib_payload = b'{"fake": "calibration"}'
    body = b"".join([
        _tar_header(f"{clip_id}.calibration.json", len(calib_payload)),
        _pad_to_block(calib_payload),
        _tar_header("clip-b.calibration.json", 10),  # next clip's group starts immediately
        _pad_to_block(b"0123456789"),
    ])
    s3 = _FakeS3(body)
    with pytest.raises(RuntimeError, match="could not find"):
        fetch_clip_files(s3, "bucket", "key", 0, clip_id, "camera_front_wide_120fov")
