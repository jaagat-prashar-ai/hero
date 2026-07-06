# SPDX-License-Identifier: Apache-2.0
"""
classify_maneuvers_test.py — end-to-end (synthetic waypoints -> features ->
maneuver_class) unit tests, one fixture per class in MANEUVER_CLASSES, plus a
determinism check. Fixtures live in synthetic_trajectory_fixtures.py, since a
maneuver class is defined entirely in terms of the same kinematic features
trajectory_features_test.py already exercises directly.
"""

from __future__ import annotations

from pref_pairs.classify_maneuvers import ManeuverConfig, classify
from pref_pairs.synthetic_trajectory_fixtures import (
    HZ,
    accelerate,
    lane_change,
    ramp_to_stop,
    straight_line,
    turn,
    yield_dip,
)
from pref_pairs.trajectory_features import FeatureConfig, extract_features

FEATURE_CONFIG = FeatureConfig()
MANEUVER_CONFIG = ManeuverConfig()


def _classify_waypoints(waypoints, scene_id, rollout_id=0):
    feats = extract_features(waypoints, HZ, scene_id, rollout_id, config=FEATURE_CONFIG)
    return classify(feats, MANEUVER_CONFIG)


def test_straight_line_is_lane_keep():
    result = _classify_waypoints(straight_line(), "scene_a")
    assert result.maneuver_class == "lane_keep"


def test_ramp_to_stop_is_stop():
    result = _classify_waypoints(ramp_to_stop(), "scene_b")
    assert result.maneuver_class == "stop"


def test_lane_change_left_for_positive_lateral_offset():
    result = _classify_waypoints(lane_change(amplitude_m=3.0), "scene_c")
    assert result.maneuver_class == "lane_change_left"


def test_lane_change_right_for_negative_lateral_offset():
    # Mirror the lane-change fixture across the x-axis (flip y) to get a
    # rightward lane change, exercising the sign branch left uncovered above.
    waypoints = lane_change(amplitude_m=3.0)
    waypoints[:, 1] *= -1
    result = _classify_waypoints(waypoints, "scene_c_right")
    assert result.maneuver_class == "lane_change_right"


def test_turn_left_for_positive_omega():
    result = _classify_waypoints(turn(omega_rad_s=0.4), "scene_d")
    assert result.maneuver_class == "turn_left"


def test_turn_right_for_negative_omega():
    result = _classify_waypoints(turn(omega_rad_s=-0.4), "scene_d_right")
    assert result.maneuver_class == "turn_right"


def test_yield_dip_is_yield():
    result = _classify_waypoints(yield_dip(), "scene_e")
    assert result.maneuver_class == "yield"


def test_accelerate_is_proceed_accelerate():
    result = _classify_waypoints(accelerate(), "scene_f")
    assert result.maneuver_class == "proceed/accelerate"


def test_every_fixture_gets_exactly_one_label_deterministically():
    fixtures = {
        "lane_keep": straight_line(),
        "stop": ramp_to_stop(),
        "lane_change_left": lane_change(),
        "turn_left": turn(),
        "yield": yield_dip(),
        "proceed/accelerate": accelerate(),
    }
    for expected_class, waypoints in fixtures.items():
        first = _classify_waypoints(waypoints, "det_scene")
        second = _classify_waypoints(waypoints, "det_scene")
        assert first.maneuver_class == second.maneuver_class == expected_class


def test_stop_rule_takes_priority_over_turn_rule():
    # A trajectory that BOTH turns sharply and stops should be labeled
    # "stop" -- rule 1 fires before rule 2 ever gets checked, per the
    # priority-order cascade ("first match wins").
    turn_waypoints = turn(omega_rad_s=0.4)
    # Freeze the second half of the arc in place, forcing a sustained
    # stop right after a >45 degree heading change has already accrued.
    half = len(turn_waypoints) // 2
    turn_waypoints[half:] = turn_waypoints[half - 1]
    result = _classify_waypoints(turn_waypoints, "scene_stop_and_turn")
    assert result.maneuver_class == "stop"


def test_boundary_margins_present_for_every_classification():
    result = _classify_waypoints(straight_line(), "scene_margins")
    assert set(result.boundary_margins) == {
        "stop_min_speed",
        "turn_heading",
        "lane_change_lateral",
        "lane_change_heading",
        "proceed_accel",
    }
