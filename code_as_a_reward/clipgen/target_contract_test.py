# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen import gate
from code_as_a_reward.clipgen.clipgen_test import GOOD_FN, GT_COC, _reactive_waypoints
from code_as_a_reward.clipgen.target_contract import (
    calibrate_spec_against_target,
    classify_rollout,
    derive_target_contract,
    validate_gt_target,
    validate_spec_against_target,
)
from code_as_a_reward.clipgen.reward_spec import compile_reward_spec_to_source
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from pref_pairs.trajectory_features import extract_features


HZ = 10.0


def _features(waypoints: np.ndarray, tag: str = "x"):
    return extract_features(waypoints, hz=HZ, scene_id=tag, rollout_id=0)


def test_no_valid_rollout_is_not_relabelled_as_candidate_positive():
    gt_claims = parse_coc_trace(GT_COC)
    gt_wp = _reactive_waypoints()
    target = derive_target_contract(gt_claims, _features(gt_wp, "gt"))
    flat = gate.flattened_waypoints(gt_wp)
    wrong_claims = parse_coc_trace("Keep lane since the lane is clear ahead")
    assert not classify_rollout(target, wrong_claims, _features(flat, "wrong")).eligible

    selection = agr.select_and_verify(
        "clip",
        "scene",
        HZ,
        [
            {
                "rollout_id": 0,
                "coc_text": "Keep lane since the lane is clear ahead",
                "waypoints": flat.tolist(),
            }
        ],
        GOOD_FN,
        target_contract=target,
    )
    assert selection.selection_status == "no_target_eligible_rollout"
    assert selection.argmax_rollout_id is None
    assert selection.argmax_gate is None


def test_stop_target_requires_sustained_stop_not_brief_touch():
    claims = parse_coc_trace("Stop for the construction worker holding a stop sign ahead")
    stationary = np.zeros((64, 2), dtype=np.float64)
    gt = _features(stationary, "gt_stop")
    target = derive_target_contract(claims, gt)
    assert target.requires_stop

    good = classify_rollout(target, claims, _features(stationary, "good_stop"))
    depart = stationary.copy()
    depart[:, 0] = np.concatenate([np.zeros(20), np.linspace(0.0, 8.0, 44)])
    bad = classify_rollout(target, claims, _features(depart, "brief_stop"))
    assert good.eligible
    assert not bad.eligible
    assert any("sustain" in reason for reason in bad.failures)


def test_stop_target_rejects_speed_drop_only_spec():
    claims = parse_coc_trace("Stop for the construction worker holding a stop sign ahead")
    target = derive_target_contract(claims, _features(np.zeros((64, 2)), "gt"))
    speed_drop = {
        "components": [
            {
                "claim": {"kind": "commitment", "field": "speed_profile", "any_of": ["decelerate"]},
                "trajectory": {"feature": "speed_drop"},
            }
        ]
    }
    failures = validate_spec_against_target(speed_drop, target)
    assert any("sustained stop" in failure for failure in failures)

    speed_drop["components"] = [
        {
            "weight": 0.4,
            "claim": {
                "kind": "perceptual",
                "field": "entity",
                "any_of": ["signal", "workers"],
            },
            "trajectory": None,
        },
        {
            "weight": 0.6,
            "claim": {
                "kind": "commitment",
                "field": "speed_profile",
                "any_of": ["decelerate"],
            },
            "trajectory": {
                "feature": "stop_dwell_fraction",
                "reference_speed_mps": 0.5,
                "floor": 0.1,
                "full": 1.05,
            },
        },
    ]
    assert not validate_spec_against_target(speed_drop, target)


def test_stop_target_rejects_motion_only_and_saturating_stop_specs():
    claims = parse_coc_trace("Stop for the construction worker holding a stop sign ahead")
    target = derive_target_contract(claims, _features(np.zeros((64, 2)), "gt"))
    spec = {
        "components": [
            {
                "weight": 1.0,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                },
                "trajectory": {
                    "feature": "late_stationary_quality",
                    "reference_speed_mps": 0.1,
                    "floor": 0.3,
                    "full": 0.6,
                },
            }
        ]
    }
    failures = validate_spec_against_target(spec, target)
    assert any("entity family" in failure for failure in failures)
    assert any("reference_speed_mps" in failure for failure in failures)
    assert any("theoretical 1.0" in failure for failure in failures)


def test_spec_aware_perturbations_ignore_unrewarded_lateral_axis():
    claims = parse_coc_trace("Accelerate to merge into traffic flow")
    x = np.cumsum(np.linspace(0.05, 0.25, 64))
    waypoints = np.column_stack([x, np.linspace(0.0, 2.0, 64)])
    accel_only_spec = {
        "components": [
            {
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["accelerate"],
                },
                "trajectory": {"feature": "speed_gain"},
            }
        ]
    }
    names = {
        case.name
        for case in gate.build_perturbations(
            "clip", claims, waypoints, HZ, reward_spec=accel_only_spec
        )
    }
    assert "perturb:forced_stop" in names
    assert "perturb:opposite_lateral_traj" not in names


def test_gt_acceleration_claim_with_decelerating_action_is_quarantined():
    claims = parse_coc_trace("Accelerate gently while watching the pedestrian")
    increments = np.linspace(0.7, 0.3, 64)
    waypoints = np.column_stack([np.cumsum(increments), np.zeros(64)])
    traj = _features(waypoints, "contradictory_gt")
    target = derive_target_contract(claims, traj)
    failures = validate_gt_target(target, traj)
    assert any("no sustained late speed gain" in failure for failure in failures)


def test_spec_rejects_unrequested_lateral_scoring_axis():
    claims = parse_coc_trace("Decelerate for the stopped vehicle ahead")
    target = derive_target_contract(claims, _features(_reactive_waypoints(), "straight_gt"))
    spec = {
        "components": [
            {
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                },
                "trajectory": {"feature": "speed_drop"},
            },
            {
                "claim": {
                    "kind": "commitment",
                    "field": "maneuver",
                    "any_of": ["turn"],
                },
                "trajectory": {"feature": "heading_right"},
            },
        ]
    }
    assert any(
        "lateral scoring axis" in failure
        for failure in validate_spec_against_target(spec, target)
    )


def test_spec_requires_the_profile_matching_the_execution_axis():
    claims = parse_coc_trace("Decelerate to maintain distance from the lead vehicle")
    target = derive_target_contract(
        claims, _features(_reactive_waypoints(), "multi_wording_gt")
    )
    assert target.speed_profiles == frozenset({"decelerate", "maintain"})
    spec = {
        "components": [
            {
                "weight": 0.4,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": ["lead_vehicle"],
                },
                "trajectory": None,
            },
            {
                "weight": 0.6,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                },
                "trajectory": {"feature": "speed_drop"},
            },
        ]
    }
    failures = validate_spec_against_target(spec, target)
    assert not any("speed-profile wording" in failure for failure in failures)

    spec["components"][1]["claim"]["any_of"] = ["maintain"]
    assert any(
        "missing ['decelerate']" in failure
        for failure in validate_spec_against_target(spec, target)
    )


def test_motion_full_is_above_gt_to_preserve_rollout_resolution():
    claims = parse_coc_trace("Decelerate for the stopped vehicle ahead")
    target = derive_target_contract(
        claims, _features(_reactive_waypoints(), "calibration_gt")
    )
    spec = {
        "components": [
            {
                "weight": 0.1,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": ["stopped_vehicle"],
                },
                "trajectory": None,
            },
            {
                "weight": 0.9,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                },
                "trajectory": {
                    "feature": "speed_drop",
                    "floor": 0.05 * target.gt_speed_drop_mps,
                    "full": 0.9 * target.gt_speed_drop_mps,
                },
            },
        ]
    }
    failures = validate_spec_against_target(spec, target)
    assert any("at least 1.15x measured GT" in failure for failure in failures)

    spec["components"][1]["trajectory"]["full"] = 1.25 * target.gt_speed_drop_mps
    assert not any(
        "measured GT magnitude" in failure
        for failure in validate_spec_against_target(spec, target)
    )


def test_compiler_calibrates_weights_aliases_and_thresholds_from_gt():
    claims = parse_coc_trace("Decelerate to maintain distance from the lead vehicle")
    target = derive_target_contract(
        claims, _features(_reactive_waypoints(), "compiler_calibration")
    )
    raw = {
        "schema_version": "clipgen.reward.v1",
        "scene_summary": "Slow for traffic ahead.",
        "components": [
            {
                "name": "traffic_context",
                "weight": 0.2,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": ["lead_vehicle"],
                },
                "trajectory": None,
            },
            {
                "name": "slow_execution",
                "weight": 0.8,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                    "direction": "any",
                },
                "trajectory": {
                    "feature": "speed_drop",
                    "window_s": [0.0, 6.4],
                    "floor": 1.0,
                    "full": 20.0,
                },
            },
        ],
    }
    calibrated = calibrate_spec_against_target(raw, target)
    assert calibrated["components"][0]["weight"] == 0.4
    execution = calibrated["components"][1]
    assert execution["weight"] == 0.6
    assert execution["claim"]["any_of"] == ["decelerate"]
    assert execution["trajectory"]["floor"] == 0.05 * target.gt_speed_drop_mps
    assert execution["trajectory"]["full"] == 1.20 * target.gt_speed_drop_mps
    assert execution["trajectory"]["power"] == 0.30
    assert not validate_spec_against_target(calibrated, target)


def test_acceleration_calibration_does_not_credit_maintain_only_reasoning():
    claims = parse_coc_trace(
        "Gentle acceleration and maintain a safe distance from the utility truck ahead."
    )
    progress = 18.0 * np.linspace(0.0, 1.0, 64) ** 1.2
    waypoints = np.column_stack([progress, np.zeros(64)])
    target = derive_target_contract(claims, _features(waypoints, "utility_truck_gt"))
    raw = _semantic_stub("vehicle_generic", "speed_profile", "accelerate")
    calibrated = calibrate_spec_against_target(raw, target)
    execution = calibrated["components"][-1]
    assert execution["trajectory"]["feature"] == "speed_gain"
    assert execution["claim"]["any_of"] == ["accelerate"]


def test_calibration_ignores_malformed_llm_numeric_fields_owned_by_compiler():
    increments = np.linspace(0.35, 0.65, 64)
    waypoints = np.column_stack([np.cumsum(increments), np.zeros(64)])
    claims = parse_coc_trace(
        "Proceed cautiously and adapt speed through the work zone"
    )
    target = derive_target_contract(claims, _features(waypoints, "numeric_repair"))
    raw = _semantic_stub("work_zone", "speed_profile", "adapt")
    raw["components"][0]["weight"] = None
    raw["components"][1]["weight"] = "not-a-number"
    raw["components"][1]["trajectory"] = {
        "feature": "cautious_progress_quality",
        "window_s": [0.0, 6.4],
        "floor": 0.0,
        "full": 1.0,
        "reference_speed_profile_mps": [1.0],
        "speed_tolerance_mps": None,
        "reference_progress_m": None,
        "progress_tolerance_m": None,
    }

    calibrated = calibrate_spec_against_target(raw, target)

    assert calibrated["components"][0]["weight"] == 0.4
    execution = calibrated["components"][-1]
    assert execution["weight"] == 0.6
    assert execution["trajectory"]["feature"] == "cautious_progress_quality"
    assert len(execution["trajectory"]["reference_speed_profile_mps"]) == 9
    assert execution["trajectory"]["speed_tolerance_mps"] > 0.0
    assert not validate_spec_against_target(calibrated, target)


def _semantic_stub(entity: str, field: str, value: str) -> dict:
    return {
        "schema_version": "clipgen.reward.v1",
        "scene_summary": "A broad scene-grounded behavior contract.",
        "components": [
            {
                "name": "scene_context",
                "weight": 0.4,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": [entity],
                },
                "trajectory": None,
            },
            {
                "name": "behavior_execution",
                "weight": 0.6,
                "claim": {
                    "kind": "commitment",
                    "field": field,
                    "any_of": [value],
                    "direction": "any",
                },
                "trajectory": {
                    "feature": "speed_gain",
                    "window_s": [0.0, 6.4],
                    "floor": 0.1,
                    "full": 1.0,
                },
            },
        ],
    }


def _calibrate_and_gate(coc: str, waypoints: np.ndarray, stub: dict):
    claims = parse_coc_trace(coc)
    traj = _features(waypoints, "behavior_gt")
    target = derive_target_contract(claims, traj)
    assert not validate_gt_target(target, traj)
    calibrated = calibrate_spec_against_target(stub, target)
    assert not validate_spec_against_target(calibrated, target)
    cases = gate.build_perturbations(
        "behavior", claims, waypoints, HZ, reward_spec=calibrated
    )
    result = gate.run_gate(compile_reward_spec_to_source(calibrated), cases)
    return target, calibrated, result


def test_keep_lane_is_a_measurable_curved_path_contract():
    x = np.linspace(0.0, 35.0, 64)
    y = 4.0 * (x / x[-1]) ** 2
    target, calibrated, result = _calibrate_and_gate(
        "Go straight following the temporary construction cones",
        np.column_stack([x, y]),
        _semantic_stub("construction_cones", "maneuver", "keep_lane"),
    )
    assert "keep_lane" in target.behavior_maneuvers
    assert calibrated["components"][-1]["trajectory"]["feature"] == "path_corridor_quality"
    assert result.passed, result.feedback()
    assert result.scores["perturb:lane_departure_traj"] <= result.pos_score - 0.4


def test_maintain_speed_uses_stability_and_oscillation_negative():
    waypoints = np.column_stack([np.arange(64) * 0.8, np.zeros(64)])
    _, calibrated, result = _calibrate_and_gate(
        "Maintain speed past the construction cones",
        waypoints,
        _semantic_stub("construction_cones", "speed_profile", "maintain"),
    )
    assert calibrated["components"][-1]["trajectory"]["feature"] == "speed_stability_quality"
    assert result.passed, result.feedback()
    assert result.scores["perturb:unstable_speed_traj"] <= result.pos_score - 0.4


def test_cautious_proceed_uses_under_and_over_progress_negatives():
    increments = np.linspace(0.35, 0.65, 64)
    waypoints = np.column_stack([np.cumsum(increments), np.zeros(64)])
    _, calibrated, result = _calibrate_and_gate(
        "Proceed cautiously and adapt speed through the work zone",
        waypoints,
        _semantic_stub("work_zone", "speed_profile", "adapt"),
    )
    assert calibrated["components"][-1]["trajectory"]["feature"] == "cautious_progress_quality"
    assert result.passed, result.feedback()
    assert result.scores["perturb:no_progress_traj"] <= result.pos_score - 0.4
    assert result.scores["perturb:unsafe_surge_traj"] <= result.pos_score - 0.4


def test_reverse_uses_heading_corridor_and_forward_negative():
    waypoints = np.column_stack([-np.linspace(0.0, 10.0, 64), np.zeros(64)])
    target, calibrated, result = _calibrate_and_gate(
        "Reverse due to the road closure in the work zone ahead",
        waypoints,
        _semantic_stub("work_zone", "maneuver", "reverse"),
    )
    assert "reverse" in target.behavior_maneuvers
    assert calibrated["components"][-1]["trajectory"]["feature"] == "heading_corridor_quality"
    assert result.passed, result.feedback()
    assert (
        result.scores["perturb:forward_instead_of_reverse_traj"]
        <= result.pos_score - 0.4
    )


def test_left_motion_is_positive_in_alpamayo_keyframe_coordinates():
    claims = parse_coc_trace("Steer left to pass the construction vehicle")
    waypoints = np.column_stack(
        [np.linspace(0.0, 20.0, 64), np.linspace(0.0, 3.0, 64)]
    )
    traj = _features(waypoints, "left_sign")
    target = derive_target_contract(claims, traj)
    assert not validate_gt_target(target, traj)
    assert classify_rollout(target, claims, traj).eligible
