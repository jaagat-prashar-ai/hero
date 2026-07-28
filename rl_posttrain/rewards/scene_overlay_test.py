# SPDX-License-Identifier: Apache-2.0
"""Tests for scene_overlay's pure helpers.

All-synthetic calibration with analytically known geometry (same style as
pref_pairs/synthetic_trajectory_fixtures.py): a front-wide-like camera 1.5 m
above the ego origin looking straight ahead, with a linear f-theta polynomial
r = 500*theta, so expected pixel positions can be computed by hand. No
network, no dataset -- per the project's no-fake-model-tests convention the
real end-to-end check is the canary cluster run (and eyeballing the JPEG).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_posttrain.rewards.scene_overlay import (  # noqa: E402
    FRONT_WIDE_CAMERA,
    build_scene_reference,
    encode_frame_jpeg,
    project_waypoints_ftheta,
    render_scene_overlay,
    select_t0_front_wide_frame,
)

# Camera->ego rotation for a forward-looking camera: camera X (right) = -ego Y,
# camera Y (down) = -ego Z, camera Z (out) = +ego X. Columns of the matrix are
# the camera axes expressed in ego coordinates.
_R_CAM2EGO = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def _synthetic_calibration(cam_height_m: float = 1.5) -> tuple[dict, dict]:
    quat = Rotation.from_matrix(_R_CAM2EGO).as_quat()  # xyzw
    intr = {
        "width": 1920.0,
        "height": 1080.0,
        "cx": 960.0,
        "cy": 540.0,
        "fw_poly_0": 0.0,
        "fw_poly_1": 500.0,
        "fw_poly_2": 0.0,
        "fw_poly_3": 0.0,
        "fw_poly_4": 0.0,
    }
    extr = {
        "qx": quat[0],
        "qy": quat[1],
        "qz": quat[2],
        "qw": quat[3],
        "x": 0.0,
        "y": 0.0,
        "z": cam_height_m,
    }
    return intr, extr


def _straight_ground_trajectory(n: int = 64, step_m: float = 1.0) -> np.ndarray:
    """x advances step_m per waypoint on the ground plane (z=0, ego origin
    height), directly ahead of the camera."""
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.arange(1, n + 1) * step_m
    return xyz


class TestProjectWaypointsFtheta:
    def test_straight_ahead_lands_on_vertical_centerline(self):
        intr, extr = _synthetic_calibration()
        pixels, valid = project_waypoints_ftheta(_straight_ground_trajectory(), intr, extr)
        assert valid.all()
        # Directly-ahead points have no lateral offset: u == cx exactly.
        np.testing.assert_allclose(pixels[:, 0], intr["cx"], atol=1e-6)
        # Camera sits above the path, so the path appears below image center...
        assert (pixels[:, 1] > intr["cy"]).all()
        # ...rising toward the horizon (v decreasing) as distance grows.
        assert (np.diff(pixels[:, 1]) < 0).all()

    def test_analytic_pixel_position(self):
        # Point 1.5 m ahead of a camera 1.5 m up: ray depression angle is
        # exactly 45 degrees, so v = cy + 500 * (pi/4).
        intr, extr = _synthetic_calibration(cam_height_m=1.5)
        pixels, valid = project_waypoints_ftheta(np.array([[1.5, 0.0, 0.0]]), intr, extr)
        assert valid[0]
        np.testing.assert_allclose(pixels[0, 1], intr["cy"] + 500.0 * np.pi / 4.0, atol=1e-6)

    def test_leftward_points_move_left_in_image(self):
        # Ego +Y is left; image u must decrease (left) as ego y increases.
        intr, extr = _synthetic_calibration()
        wp = np.array([[10.0, 0.0, 0.0], [10.0, 2.0, 0.0], [10.0, 4.0, 0.0]])
        pixels, valid = project_waypoints_ftheta(wp, intr, extr)
        assert valid.all()
        assert pixels[0, 0] > pixels[1, 0] > pixels[2, 0]

    def test_behind_camera_is_invalid_but_correspondence_kept(self):
        intr, extr = _synthetic_calibration()
        wp = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        pixels, valid = project_waypoints_ftheta(wp, intr, extr)
        # Full-length outputs (the reason this port exists -- the vendored
        # viz.py drops invalid rows and loses the index correspondence).
        assert pixels.shape == (2, 2) and valid.shape == (2,)
        assert not valid[0] and valid[1]

    def test_nan_waypoints_flagged_invalid(self):
        # Degenerate rollout decodes can carry NaNs; they must be masked,
        # not crash or leak into the drawing path.
        intr, extr = _synthetic_calibration()
        wp = np.array([[np.nan, 0.0, 0.0], [5.0, 0.0, 0.0]])
        _, valid = project_waypoints_ftheta(wp, intr, extr)
        assert not valid[0] and valid[1]

    def test_on_axis_point_is_finite(self):
        # r_xy == 0 exactly (the arccos-form port in render_trajectory_overlay
        # yields NaN here); the arctan2 form must land on the principal point.
        intr, extr = _synthetic_calibration(cam_height_m=0.0)
        pixels, valid = project_waypoints_ftheta(np.array([[10.0, 0.0, 0.0]]), intr, extr)
        assert valid[0]
        np.testing.assert_allclose(pixels[0], [intr["cx"], intr["cy"]], atol=1e-6)


def _frames_raw_layout(n_cam=4, n_frames=4, h=24, w=32):
    """(N_cam, n_frames, C, H, W) uint8 with a recognizable t0 front-wide
    frame: every pixel of frames[cam=front_wide][t0] is 200."""
    frames = np.zeros((n_cam, n_frames, 3, h, w), dtype=np.uint8)
    camera_indices = np.array([0, 1, 2, 6])  # sorted PAI order
    frames[1, -1] = 200
    return frames, camera_indices


class TestSelectT0FrontWideFrame:
    def test_raw_layout(self):
        frames, idx = _frames_raw_layout()
        out = select_t0_front_wide_frame(frames, idx)
        assert out.shape == (24, 32, 3)
        assert (out == 200).all()

    def test_rl_reshaped_layout(self):
        # PAIDataset with reshape_tensors_for_rl=True (the RL recipe config):
        # (N*G, 1, C, H, W) with camera_indices repeat_interleave'd.
        frames, idx = _frames_raw_layout()
        n_cam, n_frames = frames.shape[:2]
        reshaped = frames.reshape(n_cam * n_frames, *frames.shape[2:])[:, None]
        idx_rep = np.repeat(idx, n_frames)
        out = select_t0_front_wide_frame(reshaped, idx_rep)
        assert (out == 200).all()

    def test_torch_tensors_accepted(self):
        torch = pytest.importorskip("torch")
        frames, idx = _frames_raw_layout()
        out = select_t0_front_wide_frame(torch.from_numpy(frames), torch.from_numpy(idx))
        assert (out == 200).all()

    def test_missing_front_wide_raises(self):
        frames, _ = _frames_raw_layout()
        with pytest.raises(ValueError, match="camera_front_wide"):
            select_t0_front_wide_frame(frames, np.array([0, 2, 3, 6]))

    def test_non_uint8_raises(self):
        frames, idx = _frames_raw_layout()
        with pytest.raises(ValueError, match="uint8"):
            select_t0_front_wide_frame(frames.astype(np.float32), idx)


class TestRenderSceneOverlay:
    def _gray_frame(self, h=1080, w=1920):
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def test_returns_decodable_jpeg_with_overlay(self):
        intr, extr = _synthetic_calibration()
        jpeg = render_scene_overlay(self._gray_frame(), _straight_ground_trajectory(), intr, extr)
        img = np.asarray(Image.open(io.BytesIO(jpeg)))
        assert img.ndim == 3
        # The orange polyline must actually be there: strongly red-over-blue
        # pixels do not occur in a uniform gray image.
        assert (img[..., 0].astype(int) - img[..., 2].astype(int) > 100).any()

    def test_accepts_jpeg_bytes_input(self):
        # The reward path hands over the reference dict's already-encoded
        # scene_frame_jpeg, not a raw array.
        intr, extr = _synthetic_calibration()
        src = encode_frame_jpeg(self._gray_frame())
        jpeg = render_scene_overlay(src, _straight_ground_trajectory(), intr, extr)
        assert Image.open(io.BytesIO(jpeg)).size[0] <= 1568

    def test_downscaled_to_max_dim(self):
        intr, extr = _synthetic_calibration()
        jpeg = render_scene_overlay(
            self._gray_frame(), _straight_ground_trajectory(), intr, extr, max_dim=800
        )
        assert max(Image.open(io.BytesIO(jpeg)).size) == 800

    def test_all_invalid_trajectory_still_renders_plain_scene(self):
        # Everything behind the camera -> no overlay, but the judge still
        # gets the scene (the prompt says the table is authoritative).
        intr, extr = _synthetic_calibration()
        wp = _straight_ground_trajectory() * np.array([-1.0, 1.0, 1.0])
        jpeg = render_scene_overlay(self._gray_frame(), wp, intr, extr)
        img = np.asarray(Image.open(io.BytesIO(jpeg)))
        assert not (img[..., 0].astype(int) - img[..., 2].astype(int) > 100).any()


class TestBuildSceneReference:
    def _sample(self):
        frames, idx = _frames_raw_layout()
        intr, extr = _synthetic_calibration()
        return {
            "image_frames": frames,
            "camera_indices": idx,
            "intr": pd.DataFrame([intr], index=[FRONT_WIDE_CAMERA]),
            "extr": pd.DataFrame([extr], index=[FRONT_WIDE_CAMERA]),
        }

    def test_builds_transportable_payload(self):
        ref = build_scene_reference(self._sample())
        assert isinstance(ref["scene_frame_jpeg"], bytes)
        # Plain float dicts, not DataFrame rows -- reward-worker transport safety.
        assert ref["scene_cam_intr"]["width"] == 1920.0
        assert set(ref["scene_cam_extr"]) == {"qx", "qy", "qz", "qw", "x", "y", "z"}
        assert all(isinstance(v, float) for v in ref["scene_cam_intr"].values())
        # The encoded frame is the planted t0 front-wide frame (uniform 200s).
        img = np.asarray(Image.open(io.BytesIO(ref["scene_frame_jpeg"])))
        assert abs(int(img.mean()) - 200) <= 2

    def test_missing_calibration_raises(self):
        sample = self._sample()
        del sample["intr"]
        with pytest.raises(KeyError, match="include_extr_intr"):
            build_scene_reference(sample)
