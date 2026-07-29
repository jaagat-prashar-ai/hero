# SPDX-License-Identifier: Apache-2.0
"""Build the per-clip "dossier": the compact scene summary the generator reads.

Inputs are the same ground truth the code reward already uses -- a clip's
obstacle tracks (SceneObstacles) and the expert ego trajectory
(TrajectoryFeatures) -- reduced to ~20-30 lines of structured text. The
dossier deliberately contains the ACTUAL NUMBERS of the scene (times,
distances, speed drops) so the generator can derive scene-specific
thresholds instead of hand-tuned globals.

First-cut heuristics (documented for later calibration, same convention as
RewardConfig): tracks ranked by closest approach, capped at MAX_TRACKS;
bearing buckets at +/-20 deg (ahead) and +/-120 deg (behind).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from code_as_a_reward.obstacle_tracks import SceneObstacles
from pref_pairs.trajectory_features import TrajectoryFeatures, extract_features

MAX_TRACKS = 12

_EGOMOTION_XY_CANDIDATES = (("x", "y"), ("pos_x", "pos_y"), ("position_x", "position_y"), ("tx", "ty"))


@dataclasses.dataclass
class TrackSummary:
    """One obstacle track reduced to the facts a reward function needs."""

    track_id: str
    label_class: str
    t_enter_s: float
    t_exit_s: float
    closest_approach_m: float
    t_closest_s: float
    bearing_at_closest: str  # "ahead" | "left" | "right" | "behind"


def _bearing(x: float, y: float) -> str:
    """Bearing bucket in the ego frame (x fwd, y left)."""
    angle = math.degrees(math.atan2(y, x))
    if abs(angle) <= 20.0:
        return "ahead"
    if abs(angle) >= 120.0:
        return "behind"
    return "left" if angle > 0 else "right"


def summarize_tracks(scene: SceneObstacles) -> list[TrackSummary]:
    """Rank tracks by closest approach and keep the MAX_TRACKS nearest."""
    out: list[TrackSummary] = []
    for tr in scene.tracks:
        dists = np.linalg.norm(tr.centers_m[:, :2], axis=1)
        i = int(np.argmin(dists))
        ts = tr.timestamps_us.astype(np.float64) / 1e6
        out.append(
            TrackSummary(
                track_id=str(tr.track_id),
                label_class=tr.label_class,
                t_enter_s=float(ts[0]),
                t_exit_s=float(ts[-1]),
                closest_approach_m=float(dists[i]),
                t_closest_s=float(ts[i]),
                bearing_at_closest=_bearing(float(tr.centers_m[i, 0]), float(tr.centers_m[i, 1])),
            )
        )
    out.sort(key=lambda t: t.closest_approach_m)
    return out[:MAX_TRACKS]


def waypoints_from_egomotion(df, hz: float = 10.0) -> np.ndarray:
    """(N, 2) planar waypoints resampled at `hz` from a raw egomotion frame.

    Tolerant of the column-name variants seen across the PAI egomotion
    exports; extract_features handles the t=0-heading frame itself, so raw
    world XY is fine here.
    """
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_us"
    for cx, cy in _EGOMOTION_XY_CANDIDATES:
        if cx in df.columns and cy in df.columns:
            break
    else:
        raise ValueError(f"egomotion frame has no known XY columns (got {list(df.columns)})")
    ts = df[ts_col].to_numpy(dtype=np.float64)
    ts = (ts - ts[0]) / (1e6 if ts.max() > 1e6 else 1.0)  # us -> s when needed
    grid = np.arange(0.0, float(ts[-1]), 1.0 / hz)
    x = np.interp(grid, ts, df[cx].to_numpy(dtype=np.float64))
    y = np.interp(grid, ts, df[cy].to_numpy(dtype=np.float64))
    return np.stack([x, y], axis=1)


def ego_lines(traj: TrajectoryFeatures) -> list[str]:
    """Human/LLM-readable summary of what the expert driver actually did."""
    speed = np.asarray(traj.speed_mps, dtype=np.float64)
    t_min = float(np.argmin(speed)) * traj.dt_s
    drop = traj.initial_speed_mps - traj.min_speed_mps
    lat = np.asarray(traj.lateral_offset_m, dtype=np.float64)
    dist = float(np.sum(speed) * traj.dt_s)
    lines = [
        f"duration: {traj.n_waypoints * traj.dt_s:.1f} s at dt={traj.dt_s:.2f} s, distance {dist:.1f} m",
        f"speed: {traj.initial_speed_mps:.1f} m/s initial -> min {traj.min_speed_mps:.1f} m/s"
        f" at t={t_min:.1f} s -> {traj.final_speed_mps:.1f} m/s final (drop {drop:.1f} m/s)",
        f"lateral offset: final {traj.final_lateral_offset_m:+.2f} m,"
        f" max |offset| {float(np.max(np.abs(lat))):.2f} m",
        f"events: stop={traj.stop_event} yield={traj.yield_event}",
    ]
    # Heading is undefined at near-zero speed; on a stationary clip the
    # accumulated number is pure noise and would mislead the generator.
    if dist >= 5.0:
        lines.insert(3, f"total heading change: {traj.total_heading_change_deg:+.1f} deg")
    else:
        lines.append("note: ego is nearly stationary this clip; correct behavior is staying stopped/creeping")
    return lines


def build_dossier(
    scene: SceneObstacles,
    gt_traj: TrajectoryFeatures,
    gt_coc: str | None = None,
) -> str:
    """Render the dossier text handed to the generator."""
    parts = [f"CLIP {scene.clip_id}", "", "OBSTACLE TRACKS (ego-relative, nearest first):"]
    for t in summarize_tracks(scene):
        parts.append(
            f"- track {t.track_id} [{t.label_class}] visible {t.t_enter_s:.1f}-{t.t_exit_s:.1f} s;"
            f" closest {t.closest_approach_m:.1f} m ({t.bearing_at_closest}) at t={t.t_closest_s:.1f} s"
        )
    if len(scene.tracks) > MAX_TRACKS:
        parts.append(f"({len(scene.tracks) - MAX_TRACKS} farther tracks omitted)")
    parts += ["", "EXPERT EGO TRAJECTORY (ground truth):"]
    parts += [f"- {line}" for line in ego_lines(gt_traj)]
    if gt_coc:
        parts += ["", "GROUND-TRUTH REASONING ANNOTATION:", gt_coc.strip()]
    return "\n".join(parts)


def features_from_waypoints(waypoints: np.ndarray, hz: float, scene_id: str) -> TrajectoryFeatures:
    """Thin wrapper so callers outside pref_pairs don't repeat the boilerplate."""
    return extract_features(waypoints, hz=hz, scene_id=scene_id, rollout_id=0)
