# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest

from code_as_a_reward.clipgen import sandbox
from code_as_a_reward.clipgen.build_reward_corpus import build_corpus
from code_as_a_reward.clipgen.reward_spec import (
    RewardSpecError,
    compile_reward_spec_to_source,
    evaluate_reward_spec_components,
    validate_reward_spec,
)
from code_as_a_reward.coc_claim_parser import parse_coc_trace


def _spec() -> dict:
    return {
        "schema_version": "clipgen.reward.v1",
        "scene_summary": "slow for a stopped vehicle",
        "components": [
            {
                "name": "noticed_vehicle",
                "weight": 0.1,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": ["stopped_vehicle", "lead_vehicle"],
                },
                "trajectory": None,
            },
            {
                "name": "executed_slowdown",
                "weight": 0.9,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                    "direction": "any",
                },
                "trajectory": {
                    "feature": "speed_drop",
                    "window_s": [0.0, 6.0],
                    "floor": 1.0,
                    "full": 5.0,
                },
            },
        ],
    }


def _claims():
    return SimpleNamespace(
        perceptual=[SimpleNamespace(entity="stopped_vehicle")],
        commitments=[SimpleNamespace(speed_profile="decelerate", maneuver="stop", direction=None)],
    )


def _traj():
    speed = np.concatenate([np.full(10, 8.0), np.linspace(8.0, 3.0, 51)])
    return SimpleNamespace(
        dt_s=0.1,
        speed_mps=speed,
        heading_deg=np.zeros_like(speed),
        lateral_offset_m=np.zeros_like(speed),
    )


def test_valid_spec_compiles_and_scores_without_generated_python():
    spec = validate_reward_spec(_spec())
    components = evaluate_reward_spec_components(spec, _claims(), _traj())
    assert components["noticed_vehicle"] == pytest.approx(0.1)
    assert components["executed_slowdown"] == pytest.approx(0.9)

    source = compile_reward_spec_to_source(spec)
    reward_fn, components_fn = sandbox.compile_reward_module(source, require_components=True)
    assert sandbox.run_reward_fn(reward_fn, _claims(), _traj()) == pytest.approx(1.0)
    assert sandbox.run_components_fn(components_fn, _claims(), _traj()) == pytest.approx(components)


def test_speed_gain_is_directional_and_reversal_sensitive():
    spec = _spec()
    spec["components"][0]["weight"] = 0.1
    spec["components"][1]["weight"] = 0.9
    spec["components"][1]["claim"]["any_of"] = ["accelerate"]
    spec["components"][1]["trajectory"] = {
        "feature": "speed_gain",
        "window_s": [0.0, 6.0],
        "floor": 1.0,
        "full": 4.0,
    }
    claims = _claims()
    claims.commitments[0].speed_profile = "accelerate"
    increasing = np.linspace(2.0, 7.0, 61)

    def traj(speed):
        return SimpleNamespace(
            dt_s=0.1,
            speed_mps=speed,
            heading_deg=np.zeros_like(speed),
            lateral_offset_m=np.zeros_like(speed),
        )

    forward = evaluate_reward_spec_components(spec, claims, traj(increasing))
    reversed_score = evaluate_reward_spec_components(spec, claims, traj(increasing[::-1]))
    assert forward["executed_slowdown"] == pytest.approx(0.9)
    assert reversed_score["executed_slowdown"] == 0.0


def test_reward_penalizes_extra_action_claim_that_contradicts_trajectory():
    from pref_pairs.trajectory_features import extract_features

    faithful = parse_coc_trace("Slow down for the stopped vehicle")
    contradictory = parse_coc_trace(
        "Slow down for the stopped vehicle, then accelerate"
    )
    speed = np.concatenate([np.full(10, 8.0), np.linspace(8.0, 3.0, 51)])
    waypoints = np.zeros((len(speed), 3), dtype=np.float64)
    waypoints[:, 0] = np.cumsum(speed) * 0.1
    traj = extract_features(waypoints, hz=10.0, scene_id="test", rollout_id=0)
    faithful_score = sum(
        evaluate_reward_spec_components(_spec(), faithful, traj).values()
    )
    contradictory_score = sum(
        evaluate_reward_spec_components(_spec(), contradictory, traj).values()
    )

    assert faithful_score > 0.99
    assert contradictory_score < faithful_score
    assert contradictory_score == pytest.approx(0.5 * faithful_score)


def test_spec_rejects_overweight_perception_and_wrong_budget():
    spec = _spec()
    spec["components"][0]["weight"] = 0.5
    spec["components"][1]["weight"] = 0.5
    with pytest.raises(RewardSpecError, match="perception-only weight"):
        validate_reward_spec(spec)

    spec = _spec()
    spec["components"][1]["weight"] = 0.8
    with pytest.raises(RewardSpecError, match="sum exactly"):
        validate_reward_spec(spec)


def test_spec_allows_weight_split_across_required_action_axes():
    spec = _spec()
    spec["components"][0]["weight"] = 0.1
    spec["components"][1]["weight"] = 0.55
    spec["components"].append(
        {
            "name": "executed_right_nudge",
            "weight": 0.35,
            "claim": {
                "kind": "commitment",
                "field": "maneuver",
                "any_of": ["nudge"],
                "direction": "right",
            },
            "trajectory": {
                "feature": "lateral_right",
                "window_s": [0.0, 6.0],
                "floor": 0.2,
                "full": 1.0,
            },
        }
    )
    # Neither action component plus perception reaches 0.7. That is valid:
    # a target requiring both axes is evaluated by the empirical intact gate.
    assert validate_reward_spec(spec)["components"][2]["weight"] == 0.35


def test_lateral_feature_uses_alpamayo_positive_left_coordinate_convention():
    spec = _spec()
    spec["components"][0]["weight"] = 0.4
    spec["components"][1] = {
        "name": "left_execution",
        "weight": 0.6,
        "claim": {
            "kind": "commitment",
            "field": "maneuver",
            "any_of": ["nudge"],
            "direction": "left",
        },
        "trajectory": {
            "feature": "lateral_left",
            "window_s": [0.0, 6.0],
            "floor": 0.1,
            "full": 1.0,
        },
    }
    claims = parse_coc_trace("Nudge left for the construction cones")
    traj = SimpleNamespace(
        dt_s=0.1,
        speed_mps=np.ones(64),
        heading_deg=np.zeros(64),
        lateral_offset_m=np.linspace(0.0, 2.0, 64),
    )
    assert evaluate_reward_spec_components(spec, claims, traj)["left_execution"] == 0.6


def test_concave_curve_can_reward_tolerant_motion_without_gt_saturation():
    spec = _spec()
    spec["components"][1]["trajectory"].update(
        {"floor": 0.5, "full": 12.0, "power": 0.30}
    )
    normalized = validate_reward_spec(spec)
    assert normalized["components"][1]["trajectory"]["power"] == 0.30


def test_spec_rejects_commitment_without_execution_and_noncanonical_vocab():
    spec = _spec()
    spec["components"][1]["trajectory"] = None
    with pytest.raises(RewardSpecError, match="must be conjoined"):
        validate_reward_spec(spec)

    spec = _spec()
    spec["components"][0]["claim"]["any_of"] = ["automobile"]
    with pytest.raises(RewardSpecError, match="unknown perceptual"):
        validate_reward_spec(spec)


def test_compiled_spec_source_cannot_be_extended_with_arbitrary_code():
    source = compile_reward_spec_to_source(_spec())
    sandbox.compile_reward_module(source)
    with pytest.raises(sandbox.RewardFnError, match="exact output"):
        sandbox.compile_reward_module(source + "\ndef extra():\n    return 1\n")


def test_build_corpus_publishes_only_sealed_spec_rewards(tmp_path):
    run_dir = tmp_path / "run"
    specs_dir = run_dir / "reward_specs"
    specs_dir.mkdir(parents=True)
    artifact = {
        "clip_id": "clip-a",
        "spec": _spec(),
        "provenance": {"parser_sha256": "abc"},
        "validation": {
            "generation": {"passed": True, "gt_gate_passed": True},
            "holdout": {"passed": True},
            "cross_scene": {"passed": True},
        },
    }
    (specs_dir / "clip-a.json").write_text(json.dumps(artifact))
    out = tmp_path / "corpus"
    manifest = build_corpus([str(run_dir)], str(out))
    assert manifest["n_clips"] == 1
    assert manifest["clips"][0]["action_family"] == "decelerate_or_hold"
    assert (out / "clip-a.py").exists()
    sandbox.compile_reward_module((out / "clip-a.py").read_text())


def test_build_corpus_classifies_sustained_stop_reward(tmp_path):
    run_dir = tmp_path / "run"
    specs_dir = run_dir / "reward_specs"
    specs_dir.mkdir(parents=True)
    spec = _spec()
    spec["components"][1]["trajectory"] = {
        "feature": "late_stationary_quality",
        "window_s": [5.0, 6.4],
        "floor": 0.1,
        "full": 1.05,
        "reference_speed_mps": 0.5,
    }
    artifact = {
        "clip_id": "stop-clip",
        "spec": spec,
        "validation": {
            "generation": {"passed": True, "gt_gate_passed": True},
            "holdout": {"passed": True},
            "cross_scene": {"passed": True},
        },
    }
    (specs_dir / "stop-clip.json").write_text(json.dumps(artifact))
    manifest = build_corpus([str(run_dir)], str(tmp_path / "corpus"))
    assert manifest["clips"][0]["action_family"] == "decelerate_or_hold"
