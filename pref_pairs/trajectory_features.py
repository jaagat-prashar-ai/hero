# SPDX-License-Identifier: Apache-2.0
"""
trajectory_features.py — pure kinematic feature extraction over rollout
trajectories (maneuver-class labeling task, prerequisite for epsilon
calibration and trajectory-matched pair mining).

Deliberately kept separate from classify_maneuvers.py's rule cascade: this
module only computes numbers (speed, heading, lateral offset, stop/yield
flags), it never assigns a maneuver_class. That split means the downstream
claim verifier (checking CoC claims against kinematics) can reuse this
feature table directly without caring which classifier or thresholds
produced any label.

Frame convention -- why we do NOT rotate:
  Every trajectory in this pipeline (both Alpamayo's diffusion-decoded
  rollouts and the dataset's ground-truth futures) is already expressed in
  an ego-frame anchored at the vehicle's pose at t=0, with the origin
  (0, 0) AS the t=0 position and heading EXACTLY 0 there, by construction:
    - Alpamayo's action_to_traj (unicycle_accel_curvature.py) integrates
      each rollout starting from x=0, y=0, heading=0 -- hardcoded, not
      estimated.
    - load_physical_aiavdataset.py's ground-truth loader subtracts the t0
      position and rotates by the INVERSE of the t0 orientation, which by
      definition also puts t0's heading exactly along local +x.
  An earlier version of this module tried to be "extra safe" by estimating
  the initial heading from the first few waypoints and rotating the WHOLE
  trajectory by that estimate, in case a future trajectory source didn't
  share this convention. That was actively wrong: for any maneuver that
  starts curving immediately -- a lane change or a turn, i.e. exactly the
  cases this module exists to classify -- a few early samples of genuine,
  intentional curvature get misread as "heading estimation noise" and
  rotated away, and that small angular error is a LEVER ARM: rotating a
  ~1 degree misestimate shifts a waypoint 40m out by ~0.7m, which silently
  corrupted `final_lateral_offset_m` on lane-change trajectories (caught by
  trajectory_features_test.py's synthetic lane-change fixture, which
  expected 3.0m and got ~2.2m). Since the true t=0 heading is EXACTLY 0 for
  every trajectory source this pipeline actually uses, there is nothing to
  estimate: we take the waypoints as given. `_initial_heading_rad` is kept
  only to populate `initial_heading_correction_deg` as a DIAGNOSTIC -- if
  that ever comes out large for real data, it's a sign the input isn't in
  the expected ego-frame-at-t0 convention and should be investigated, but
  it is intentionally never applied as a rotation.

Native acceleration -- why we prefer it, and why only for acceleration:
  Alpamayo's diffusion decoder predicts a normalized (accel, curvature)
  action per waypoint natively, THEN a deterministic integration step
  (action_to_traj) turns that into the xyz waypoints we receive.
  rollout_harvester.py now captures that native action and denormalizes it
  to physical units (see its "NATIVE ACTION CAPTURE" docstring note) before
  the value is thrown away. When it's available, extract_features uses it
  directly for accel/decel features instead of re-deriving an
  approximation via np.gradient(speed, dt) -- the model's own value is
  exact, ours is a finite-difference estimate with a smoothing-window
  judgment call baked in. `accel_source` on TrajectoryFeatures records
  which path was actually used, so this is auditable rather than a silent
  swap.
  We do NOT similarly substitute native curvature for the heading/lateral-
  offset features. Curvature alone doesn't give heading directly -- getting
  from kappa to heading requires theta = integral(kappa * v dt), the same
  integration action_to_traj already performs to produce xyz in the first
  place. Re-deriving heading from xyz via finite differences (as we already
  do) is simpler and no less correct than re-implementing that integration
  a second time just to avoid one derivative step; the accel case is
  different because there ISN'T a simpler way to get "the model's exact
  acceleration" other than capturing the value the model already computed.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


@dataclasses.dataclass
class FeatureConfig:
    """Tunable knobs for feature extraction (shared with classify_maneuvers.py
    via one YAML file -- see pref_pairs/configs/maneuver_thresholds.yaml --
    so the threshold-sensitivity report has a single source to perturb)."""

    smoothing_window: int = 3  # samples; moving avg on speed/heading before thresholding
    initial_heading_window: int = 3  # waypoints from the origin used to estimate t=0 heading

    stop_speed_mps: float = 0.5
    stop_duration_s: float = 1.0
    stop_recovery_speed_mps: float = 2.0

    yield_drop_fraction: float = 0.30
    yield_min_speed_mps: float = 2.0
    yield_recovery_fraction: float = 0.50

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureConfig":
        """Build from the flat sections of maneuver_thresholds.yaml that this
        module actually needs (smoothing/initial_heading/stop/yield); the
        turn/lane_change/proceed_accelerate sections belong to
        classify_maneuvers.py's rule cascade, not to feature extraction."""
        return cls(
            smoothing_window=d.get("smoothing", {}).get("window", 3),
            initial_heading_window=d.get("initial_heading", {}).get("window", 3),
            stop_speed_mps=d.get("stop", {}).get("speed_mps", 0.5),
            stop_duration_s=d.get("stop", {}).get("duration_s", 1.0),
            stop_recovery_speed_mps=d.get("stop", {}).get("recovery_speed_mps", 2.0),
            yield_drop_fraction=d.get("yield", {}).get("drop_fraction", 0.30),
            yield_min_speed_mps=d.get("yield", {}).get("min_speed_mps", 2.0),
            yield_recovery_fraction=d.get("yield", {}).get("recovery_fraction", 0.50),
        )


@dataclasses.dataclass
class TrajectoryFeatures:
    """One rollout's kinematic feature row. Everything the maneuver classifier
    AND the downstream claim verifier need, with no maneuver_class attached."""

    scene_id: str
    rollout_id: int
    n_waypoints: int
    dt_s: float

    # Per-waypoint time series (smoothed), useful for plotting/spot-checks.
    speed_mps: list[float]
    heading_deg: list[float]
    lateral_offset_m: list[float]  # == smoothed y in the t=0-heading frame

    # Scalar summary features consumed by the classifier / claim verifier.
    initial_speed_mps: float
    final_speed_mps: float
    min_speed_mps: float
    final_lateral_offset_m: float
    total_heading_change_deg: float
    mean_acceleration_mps2: float  # signed; > 0 means net speeding up
    mean_deceleration_mps2: float  # magnitude, averaged over decelerating samples only
    max_deceleration_mps2: float  # magnitude of the single worst deceleration sample

    stop_event: bool
    yield_event: bool

    # Diagnostic: how much we rotated the raw waypoints to align t=0 heading
    # to +x. Expected to be small (a few degrees) for real Alpamayo rollouts
    # per the frame-convention note above; a large value here is a sign
    # something upstream isn't in the expected ego-frame-at-t0 convention.
    initial_heading_correction_deg: float

    # "native" if mean/mean_deceleration/max_deceleration came from the
    # model's own captured (accel, curvature) action tensor, "finite_difference"
    # if they were derived from xyz via np.gradient instead (native wasn't
    # available -- see module docstring's "Native acceleration" note).
    accel_source: str = "finite_difference"

    def to_row_dict(self) -> dict[str, Any]:
        """Flat dict for the maneuver_labels table -- time series omitted
        (they don't belong in a one-row-per-rollout parquet table), scalar
        features only."""
        return {
            "scene_id": self.scene_id,
            "rollout_id": self.rollout_id,
            "n_waypoints": self.n_waypoints,
            "dt_s": self.dt_s,
            "initial_speed_mps": self.initial_speed_mps,
            "final_speed_mps": self.final_speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "final_lateral_offset_m": self.final_lateral_offset_m,
            "total_heading_change_deg": self.total_heading_change_deg,
            "mean_acceleration_mps2": self.mean_acceleration_mps2,
            "mean_deceleration_mps2": self.mean_deceleration_mps2,
            "max_deceleration_mps2": self.max_deceleration_mps2,
            "stop_event": self.stop_event,
            "yield_event": self.yield_event,
            "initial_heading_correction_deg": self.initial_heading_correction_deg,
            "accel_source": self.accel_source,
        }


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, edge-padded so the output keeps x's length --
    a short boundary window at each end averages over fewer samples rather
    than needing look-ahead/behind that doesn't exist."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    # 'same'-mode convolution already keeps length; edge-pad first so the
    # first/last (window // 2) samples aren't biased toward zero by implicit
    # zero-padding.
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(x)]


def _initial_heading_rad(xy: np.ndarray, window: int) -> float:
    """DIAGNOSTIC ONLY -- see module docstring's "Frame convention" note for
    why this is deliberately never applied as a rotation. Estimates the
    trajectory's apparent heading at t=0 from the vector between the origin
    and the `window`-th waypoint, purely so callers can sanity-check that a
    trajectory source actually honors the ego-frame-at-t0 convention this
    module assumes (a large value here means it doesn't, and the rest of
    this module's numbers should not be trusted for that input)."""
    n = min(window, len(xy) - 1)
    if n < 1:
        return 0.0
    dx, dy = xy[n, 0] - 0.0, xy[n, 1] - 0.0
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


def extract_features(
    waypoints: list[list[float]] | np.ndarray,
    hz: float,
    scene_id: str,
    rollout_id: int,
    config: FeatureConfig | None = None,
    native_accel_mps2: list[float] | np.ndarray | None = None,
) -> TrajectoryFeatures:
    """Compute kinematic features for one rollout's (T, 3) xyz waypoints.

    Only x, y are used (z is held constant by Alpamayo's action space -- see
    unicycle_accel_curvature.py -- so it carries no maneuver information).
    Heading/velocity are NOT present on disk (RolloutRecord only stores
    waypoints), so both are always derived here by finite differences, per
    the task's "use them if present, otherwise derive by finite differences."

    `native_accel_mps2`, if given (rollout_harvester.py's captured
    RolloutRecord.native_accel_mps2), is the model's OWN exact per-waypoint
    acceleration and is used directly for the accel/decel features instead
    of re-deriving an approximation -- see module docstring's "Native
    acceleration" note for why this is accel-only, not also curvature.
    """
    config = config or FeatureConfig()
    xy = np.asarray(waypoints, dtype=np.float64)[:, :2]
    n = xy.shape[0]
    dt = 1.0 / hz

    # Diagnostic only -- NOT applied as a rotation. See module docstring's
    # "Frame convention" note: the input is trusted to already be in the
    # ego-frame-at-t0 convention (heading exactly 0 at the origin), and a
    # large value here is a sign that trust is misplaced for this input.
    heading_correction_rad = _initial_heading_rad(xy, config.initial_heading_window)

    # --- Velocity via central differences (np.gradient handles edges) ---
    vx = np.gradient(xy[:, 0], dt)
    vy = np.gradient(xy[:, 1], dt)
    speed_raw = np.sqrt(vx**2 + vy**2)

    # --- Heading via atan2, unwrapped so total heading change doesn't wrap
    # at +-180 deg, then smoothed the same way speed is ---
    heading_raw = np.unwrap(np.arctan2(vy, vx))

    speed = _moving_average(speed_raw, config.smoothing_window)
    heading = _moving_average(heading_raw, config.smoothing_window)

    # --- Longitudinal acceleration: prefer the model's own exact value when
    # rollout_harvester.py managed to capture it; otherwise fall back to
    # d(speed)/dt of the SMOOTHED speed, so a single-sample speed spike
    # doesn't register as a huge accel/decel spike. See module docstring's
    # "Native acceleration" note. ---
    if native_accel_mps2 is not None:
        accel = np.asarray(native_accel_mps2, dtype=np.float64)
        assert accel.shape == (n,), (
            f"native_accel_mps2 length {accel.shape} must match waypoints length {n}"
        )
        accel_source = "native"
    else:
        accel = np.gradient(speed, dt)
        accel_source = "finite_difference"

    lateral_offset = xy[:, 1]  # already lateral by definition of this frame

    decel_samples = -accel[accel < 0]  # magnitudes of decelerating samples only
    mean_decel = float(decel_samples.mean()) if decel_samples.size else 0.0
    max_decel = float(decel_samples.max()) if decel_samples.size else 0.0

    stop_event = _detect_stop_event(speed, dt, config)
    yield_event = _detect_yield_event(speed, config)

    return TrajectoryFeatures(
        scene_id=scene_id,
        rollout_id=rollout_id,
        n_waypoints=n,
        dt_s=dt,
        speed_mps=speed.tolist(),
        heading_deg=np.degrees(heading).tolist(),
        lateral_offset_m=lateral_offset.tolist(),
        initial_speed_mps=float(speed[0]),
        final_speed_mps=float(speed[-1]),
        min_speed_mps=float(speed.min()),
        final_lateral_offset_m=float(lateral_offset[-1]),
        total_heading_change_deg=float(np.degrees(heading[-1] - heading[0])),
        mean_acceleration_mps2=float(accel.mean()),
        mean_deceleration_mps2=mean_decel,
        max_deceleration_mps2=max_decel,
        stop_event=stop_event,
        yield_event=yield_event,
        initial_heading_correction_deg=math.degrees(heading_correction_rad),
        accel_source=accel_source,
    )


def _detect_stop_event(speed: np.ndarray, dt: float, config: FeatureConfig) -> bool:
    """speed < stop_speed_mps sustained for >= stop_duration_s at any point,
    AND the trajectory does not recover above stop_recovery_speed_mps by the
    end -- a rollout that dips low and then drives off normally is NOT a
    stop, it's (at most) a yield."""
    min_run = max(1, math.ceil(config.stop_duration_s / dt))
    below = speed < config.stop_speed_mps

    # Longest contiguous run of `below` -- a small manual scan rather than
    # pulling in scipy/itertools.groupby, since this is the only place we
    # need contiguous-run detection.
    longest_run = 0
    current_run = 0
    for is_below in below:
        current_run = current_run + 1 if is_below else 0
        longest_run = max(longest_run, current_run)

    sustained_stop = longest_run >= min_run
    recovered = speed[-1] > config.stop_recovery_speed_mps
    return bool(sustained_stop and not recovered)


def _detect_yield_event(speed: np.ndarray, config: FeatureConfig) -> bool:
    """speed drops by > drop_fraction from its initial value (or below
    min_speed_mps outright), THEN recovers by >= recovery_fraction of that
    dip before the trajectory ends. The dip must occur strictly before the
    last sample, otherwise there is no room left in the trajectory to
    "recover" and this is better described by the stop rule instead."""
    initial_speed = speed[0]
    dip_idx = int(np.argmin(speed))
    dip_speed = float(speed[dip_idx])

    dip_amount = initial_speed - dip_speed
    dip_fraction = (dip_amount / initial_speed) if initial_speed > 1e-6 else 0.0
    dropped = dip_fraction > config.yield_drop_fraction or dip_speed < config.yield_min_speed_mps

    if not dropped or dip_idx >= len(speed) - 1 or dip_amount <= 1e-6:
        return False

    recovered_amount = speed[-1] - dip_speed
    recovery_fraction = recovered_amount / dip_amount
    return bool(recovery_fraction >= config.yield_recovery_fraction)
