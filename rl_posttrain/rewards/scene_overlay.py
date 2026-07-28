# SPDX-License-Identifier: Apache-2.0
"""
scene_overlay.py -- renders the judge's scene image: the ego vehicle's
front-wide camera frame at the decision time t0 with a trajectory projected
onto it. Used by aggregated_reward_llm_judge to give the LLM judge visual
grounding (does the claimed hazard/agent actually exist, and does the drawn
path respond to it?) on top of the text-only waypoint table.

Provenance of the projection math: ported from the vendored
third_party/alpamayo-recipes/src/alpamayo/visualization/viz.py
(project_waypoints_ftheta) rather than imported, for two reasons:
  - the vendored function returns only the *visible* pixels (`projected[valid]`),
    dropping the waypoint<->pixel index correspondence we need to draw a
    time-ordered polyline;
  - importing alpamayo.visualization.viz pulls in cv2 and matplotlib at module
    import time, neither of which the recipe venv guarantees inside cosmos-rl
    reward workers. PIL *is* guaranteed (a hard dependency of vllm and
    transformers), so all drawing here is PIL.
The f-theta model itself matches the repo's other port
(pref_pairs/render_trajectory_overlay.py:ftheta_ray2pixel), but this module
keeps viz.py's arctan2 form (numerically safe on the optical axis, where the
arccos form yields NaN) and viz.py's flat calibration keying (width/height/
cx/cy/fw_poly_0..4 columns), because that is exactly how the PAI dataset the
RL recipe trains on stores calibration (alpamayo/data/pai.py include_extr_intr).

Coordinate conventions (same as load_physical_aiavdataset / viz.py):
  - waypoints are in the ego frame at t0: X forward, Y left, Z up, meters;
    since we draw on the t0 frame, no ego-motion re-expression is needed
    (contrast with pref_pairs/render_trajectory_overlay.py, which animates
    the path over later frames and must interpolate ego poses).
  - camera extrinsics are the camera->ego rigid transform (quaternion xyzw +
    translation); `(wp - t) @ R` is the ego->camera inverse rotation.
  - camera frame: X right, Y down, Z out of the camera.

Camera choice: camera_front_wide_120fov, hardcoded exactly like the vendored
viz_waypoints_pai -- it is the camera the driving decision is about, and the
one the vendored tooling treats as canonical for waypoint overlays.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

FRONT_WIDE_CAMERA = "camera_front_wide_120fov"

# Fixed camera_name_to_index mapping from
# third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py --
# frames are sorted by this index, so front-wide is the block tagged 1.
FRONT_WIDE_CAMERA_INDEX = 1

_INTR_COLUMNS = ("width", "height", "cx", "cy", "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4")
_EXTR_COLUMNS = ("qx", "qy", "qz", "qw", "x", "y", "z")

# Claude vision downsamples anything above ~1568 px on the long side anyway,
# so sending more resolution than that only inflates the request payload.
_MAX_IMAGE_DIM = 1568

# Overlay style matches pref_pairs/render_trajectory_overlay.py (orange
# polyline + every-4th-waypoint dots) so humans eyeballing judge inputs and
# the existing overlay videos read the same visual language.
_OVERLAY_COLOR = (255, 160, 40)
_OVERLAY_OUTLINE = (20, 20, 20)


def select_t0_front_wide_frame(image_frames: Any, camera_indices: Any) -> np.ndarray:
    """Picks the front-wide camera frame at t0 out of a PAI sample's
    image_frames, returning (H, W, 3) uint8.

    Handles both layouts the PAIDataset can emit:
      - raw: image_frames (N_cam, n_frames, C, H, W), camera_indices (N_cam,)
      - reshape_tensors_for_rl (the RL recipe's config): image_frames
        (N_cam * n_frames, 1, C, H, W), camera_indices repeated per frame.
    In both, frames within a camera block are time-ascending ending at t0
    (load_physical_aiavdataset's image_timestamps), so the LAST entry tagged
    with the front-wide index is the t0 front-wide frame.
    """
    idx = np.asarray(camera_indices).reshape(-1)
    positions = np.flatnonzero(idx == FRONT_WIDE_CAMERA_INDEX)
    if positions.size == 0:
        raise ValueError(
            f"no camera_indices entry equals {FRONT_WIDE_CAMERA_INDEX} "
            f"({FRONT_WIDE_CAMERA}); got indices {sorted(set(idx.tolist()))}"
        )
    block = image_frames[int(positions[-1])]
    frame = block[-1]  # (C, H, W): t0 in the raw layout, the only frame in RL layout
    arr = np.asarray(frame.cpu() if hasattr(frame, "cpu") else frame)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        # The PAI decoder emits uint8; anything else means an unexpected
        # preprocessing step upstream -- fail loud rather than guess a scale.
        raise ValueError(f"expected uint8 frame data, got dtype {arr.dtype}")
    return np.transpose(arr, (1, 2, 0))


def project_waypoints_ftheta(
    waypoints_xyz: np.ndarray,
    cam_intr: dict[str, float],
    cam_extr: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Projects (T, 3) ego-frame-at-t0 waypoints to (T, 2) pixels via the
    f-theta model, plus a (T,) validity mask (in front of the camera, inside
    the image, finite). Unlike the vendored viz.py original this keeps the
    full T rows so callers can draw a time-ordered polyline."""
    wp = np.asarray(waypoints_xyz, dtype=np.float64)
    rot = Rotation.from_quat([cam_extr[k] for k in ("qx", "qy", "qz", "qw")]).as_matrix()
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


def encode_frame_jpeg(frame_hwc: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame_hwc).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
