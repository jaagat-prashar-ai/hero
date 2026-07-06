# SPDX-License-Identifier: Apache-2.0
"""
synthetic_trajectory_fixtures.py — hand-built (T, 3) xyz waypoint arrays,
one per maneuver class, with known analytical kinematics (constant
dt=0.1s / hz=10.0). Used by trajectory_features_test.py,
classify_maneuvers_test.py, and maneuver_report_test.py.

Deliberately NOT a *_test.py file itself, even though its only purpose is
testing: py_test (Bazel) and plain unittest discovery both expect one
importable module per test target, and a test file importing symbols from
ANOTHER test file is awkward for that. Splitting the fixtures out into a
plain module lets all three test files depend on :synthetic_trajectory_fixtures
instead of on each other.
"""

from __future__ import annotations

import numpy as np

HZ = 10.0
DT = 1.0 / HZ


def straight_line(speed_mps: float = 10.0, n: int = 40) -> np.ndarray:
    """Constant-speed, constant-heading trajectory -- textbook lane_keep."""
    t = np.arange(1, n + 1) * DT
    x = speed_mps * t
    y = np.zeros(n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)


def ramp_to_stop(initial_speed: float = 8.0, decel_duration_s: float = 2.0, n: int = 40) -> np.ndarray:
    """Speed ramps linearly to 0 over decel_duration_s, then stays at 0."""
    t = np.arange(1, n + 1) * DT
    decel_steps = int(decel_duration_s / DT)
    speed = np.concatenate(
        [
            np.linspace(initial_speed, 0.0, decel_steps),
            np.zeros(n - decel_steps),
        ]
    )
    x = np.cumsum(speed * DT)
    y = np.zeros(n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)


def lane_change(amplitude_m: float = 3.0, forward_speed: float = 10.0, transition_s: float = 3.0, n: int = 40) -> np.ndarray:
    """Raised-cosine ("sinusoidal") lateral profile: 0 -> amplitude_m over
    transition_s, then holds constant -- textbook lane change. Heading
    starts and ends near 0 (raised cosine has zero derivative at both
    endpoints of the transition), so this should clear the lane_change
    lateral threshold WITHOUT tripping the turn rule."""
    t = np.arange(1, n + 1) * DT
    transition_steps = int(transition_s / DT)
    u = np.linspace(0, np.pi, transition_steps)
    y_transition = (amplitude_m / 2.0) * (1 - np.cos(u))
    y = np.concatenate([y_transition, np.full(n - transition_steps, amplitude_m)])
    x = forward_speed * t
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)


def turn(omega_rad_s: float = 0.4, speed_mps: float = 8.0, n: int = 40) -> np.ndarray:
    """Constant angular velocity arc -- positive omega turns LEFT under this
    module's assumed y-left convention. Over n*DT=4s at omega=0.4 rad/s this
    sweeps ~91 degrees, comfortably past the 45 degree turn threshold."""
    t = np.arange(1, n + 1) * DT
    theta = omega_rad_s * t
    x = np.cumsum(speed_mps * np.cos(theta) * DT)
    y = np.cumsum(speed_mps * np.sin(theta) * DT)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)


def yield_dip(initial_speed: float = 10.0, dip_speed: float = 3.0, final_speed: float = 8.0, n: int = 40) -> np.ndarray:
    """Speed ramps down to a dip then back up, never sustaining < 0.5 m/s
    (so this must NOT trip the stop rule) but dropping > 30% and recovering
    > 50% of that dip -- textbook yield."""
    half = n // 2
    speed = np.concatenate(
        [
            np.linspace(initial_speed, dip_speed, half),
            np.linspace(dip_speed, final_speed, n - half),
        ]
    )
    x = np.cumsum(speed * DT)
    y = np.zeros(n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)


def accelerate(initial_speed: float = 5.0, final_speed: float = 15.0, n: int = 40) -> np.ndarray:
    """Monotonic speed-up, straight line -- textbook proceed/accelerate."""
    speed = np.linspace(initial_speed, final_speed, n)
    x = np.cumsum(speed * DT)
    y = np.zeros(n)
    z = np.zeros(n)
    return np.stack([x, y, z], axis=-1)
