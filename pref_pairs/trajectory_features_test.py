# SPDX-License-Identifier: Apache-2.0
"""
trajectory_features_test.py — synthetic-trajectory unit tests for
trajectory_features.py. Fixtures live in synthetic_trajectory_fixtures.py
(shared with classify_maneuvers_test.py and maneuver_report_test.py) so the
expected kinematic numbers are known analytically, independent of any real
Alpamayo rollout.
"""

from __future__ import annotations

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


def test_straight_line_has_no_heading_change_or_lateral_offset():
    feats = extract_features(straight_line(), HZ, "scene_a", 0)
    assert abs(feats.total_heading_change_deg) < 1.0
    assert abs(feats.final_lateral_offset_m) < 0.1
    assert not feats.stop_event
    assert not feats.yield_event


def test_ramp_to_stop_trips_stop_event():
    feats = extract_features(ramp_to_stop(), HZ, "scene_b", 0)
    assert feats.stop_event
    assert feats.final_speed_mps < 2.0


def test_lane_change_reaches_amplitude_with_small_heading_change():
    feats = extract_features(lane_change(), HZ, "scene_c", 0)
    assert feats.final_lateral_offset_m > 2.5
    assert abs(feats.total_heading_change_deg) < 45.0


def test_turn_exceeds_heading_threshold():
    feats = extract_features(turn(), HZ, "scene_d", 0)
    assert abs(feats.total_heading_change_deg) > 45.0
    assert feats.total_heading_change_deg > 0  # positive omega => left turn, by convention


def test_yield_dip_trips_yield_event_not_stop_event():
    feats = extract_features(yield_dip(), HZ, "scene_e", 0)
    assert feats.yield_event
    assert not feats.stop_event


def test_accelerate_has_positive_mean_acceleration_and_no_events():
    feats = extract_features(accelerate(), HZ, "scene_f", 0)
    assert feats.mean_acceleration_mps2 > 0.5
    assert not feats.stop_event
    assert not feats.yield_event


def test_extract_features_is_deterministic():
    waypoints = lane_change()
    a = extract_features(waypoints, HZ, "scene_g", 0)
    b = extract_features(waypoints, HZ, "scene_g", 0)
    assert a == b


def test_custom_feature_config_changes_stop_detection():
    # A dip that would count as a stop under default thresholds should NOT
    # trip stop_event once the sustained-duration requirement is raised
    # well past how long the dip actually lasts.
    waypoints = ramp_to_stop(decel_duration_s=2.0, n=25)  # stopped for only ~0.5s of the 2.5s
    lenient = FeatureConfig(stop_duration_s=5.0)
    feats = extract_features(waypoints, HZ, "scene_h", 0, config=lenient)
    assert not feats.stop_event
