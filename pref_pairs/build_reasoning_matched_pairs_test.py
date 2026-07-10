# SPDX-License-Identifier: Apache-2.0
"""
build_reasoning_matched_pairs_test.py — synthetic-scene tests for the
flatten/join logic (no GPU/model/network needed).
"""

from __future__ import annotations

from pref_pairs.build_reasoning_matched_pairs import build_pairs


def _scene(scene_id: str, n_perturbations: int) -> dict:
    return {
        "scene_id": scene_id,
        "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
        "ground_truth_trace": "keep distance to the lead vehicle",
        "ground_truth_action": {"rollout_id": 0, "maneuver_class": "lane_keep", "waypoints": [[0, 0, 0]]},
        "perturbations": [
            {
                "trace_id": f"{scene_id}__pert{i}",
                "perturbation_type": f"type{i}",
                "original_span": "foo",
                "perturbed_span": "bar",
                "perturbed_trace": f"perturbed trace {i}",
                "semantic_delta": "delta",
                "decision_impact": "impact",
                "plausibility_rationale": "rationale",
            }
            for i in range(n_perturbations)
        ],
    }


def test_build_pairs_one_row_per_perturbation():
    scenes = [_scene("s1", 3), _scene("s2", 2)]
    pairs = build_pairs(scenes)
    assert len(pairs) == 5
    assert {p["scene_id"] for p in pairs} == {"s1", "s2"}


def test_build_pairs_shares_action_across_all_pairs_of_a_scene():
    scenes = [_scene("s1", 2)]
    pairs = build_pairs(scenes)
    assert pairs[0]["action"] == pairs[1]["action"]
    assert pairs[0]["action"]["maneuver_class"] == "lane_keep"


def test_build_pairs_chosen_and_rejected_traces_differ():
    scenes = [_scene("s1", 1)]
    pair = build_pairs(scenes)[0]
    assert pair["chosen_trace"] == "keep distance to the lead vehicle"
    assert pair["rejected_trace"] == "perturbed trace 0"
    assert pair["chosen_trace"] != pair["rejected_trace"]
    assert pair["pair_id"] == "s1__pert0"
    assert pair["perturbation_type"] == "type0"
