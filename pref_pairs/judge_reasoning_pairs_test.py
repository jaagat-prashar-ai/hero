# SPDX-License-Identifier: Apache-2.0
"""
judge_reasoning_pairs_test.py -- covers only the pure parsing/formatting/
scoring helpers in judge_reasoning_pairs.py (swap_seed, format_waypoint_table,
_parse_judgment_response, _build_result_row). call_judge / judge_all_pairs
call the real Fable 5 API and are deliberately NOT covered by a mocked-client
test here -- see feedback_no_fake_model_tests (same convention as
perturbation_generator_test.py): those functions are verified via an actual
--max_pairs smoke-test run against the live API before a full batch, not via
pytest.
"""

from __future__ import annotations

import json

import pytest

from pref_pairs.judge_reasoning_pairs import (
    JudgeError,
    _build_result_row,
    _parse_judgment_response,
    format_waypoint_table,
    swap_seed,
)
from pref_pairs.synthetic_trajectory_fixtures import HZ, straight_line

_VALID_JUDGMENT = {
    "trace_a": {"action_consistency_score": 8, "corruption_type": "none", "one_line_rationale": "matches"},
    "trace_b": {"action_consistency_score": 1, "corruption_type": "causal_flip", "one_line_rationale": "inverted"},
    "preferred": "A",
    "margin_confidence": "high",
}

_SAMPLE_PAIR = {
    "pair_id": "scene_1__causal_flip",
    "scene_id": "scene_1",
    "perturbation_type": "causal_flip",
    "chosen_trace": "Keep distance to the lead vehicle",
    "rejected_trace": "No need to keep distance to the lead vehicle",
}

_SAMPLE_PAIR_IDS = [
    "00bbc8b2-7d40-40f7-a1b3-a5853fe5bddc_12206610__negation_flip",
    "00bbc8b2-7d40-40f7-a1b3-a5853fe5bddc_12206610__spatial_error",
    "01f30837-55ca-496c-9a10-e837ea201144_14014702__causal_flip",
    "01f30837-55ca-496c-9a10-e837ea201144_14014702__attribute_swap",
]


def test_swap_seed_is_deterministic():
    for pair_id in _SAMPLE_PAIR_IDS:
        assert swap_seed(pair_id) == swap_seed(pair_id)


def test_swap_seed_is_not_constant_across_pairs():
    # Not every pair_id should land on the same side of the coin flip --
    # a constant result would silently defeat the whole point of blind A/B.
    assert len({swap_seed(pair_id) for pair_id in _SAMPLE_PAIR_IDS}) == 2


def test_format_waypoint_table_has_one_line_per_waypoint():
    waypoints = straight_line(speed_mps=10.0, n=5)
    action = {"waypoints": waypoints, "hz": HZ, "rollout_id": 0}
    table = format_waypoint_table(action)
    lines = table.splitlines()
    assert len(lines) == 5
    assert lines[0].startswith("0: x=")
    assert lines[-1].startswith("4: x=")


def test_format_waypoint_table_straight_line_has_near_zero_lateral_and_heading():
    # straight_line is constant-speed, constant-heading -- y and heading
    # should both round to ~0 at every step, not just start/end.
    waypoints = straight_line(speed_mps=10.0, n=10)
    action = {"waypoints": waypoints, "hz": HZ, "rollout_id": 0}
    table = format_waypoint_table(action)
    for line in table.splitlines():
        assert "y=0.0" in line
        assert "h=0" in line


def test_parse_judgment_response_passes_through_plain_json():
    assert _parse_judgment_response(json.dumps(_VALID_JUDGMENT)) == _VALID_JUDGMENT


def test_parse_judgment_response_strips_json_fence():
    text = "```json\n" + json.dumps(_VALID_JUDGMENT) + "\n```"
    assert _parse_judgment_response(text) == _VALID_JUDGMENT


def test_parse_judgment_response_raises_on_missing_top_level_key():
    bad = {k: v for k, v in _VALID_JUDGMENT.items() if k != "preferred"}
    with pytest.raises(JudgeError, match="preferred"):
        _parse_judgment_response(json.dumps(bad))


def test_parse_judgment_response_raises_on_missing_trace_key():
    bad = json.loads(json.dumps(_VALID_JUDGMENT))  # deep copy
    del bad["trace_a"]["one_line_rationale"]
    with pytest.raises(JudgeError, match="trace_a"):
        _parse_judgment_response(json.dumps(bad))


def test_parse_judgment_response_raises_on_out_of_range_score():
    bad = json.loads(json.dumps(_VALID_JUDGMENT))
    bad["trace_a"]["action_consistency_score"] = 11
    with pytest.raises(JudgeError, match="action_consistency_score"):
        _parse_judgment_response(json.dumps(bad))


def test_parse_judgment_response_raises_on_invalid_corruption_type():
    bad = json.loads(json.dumps(_VALID_JUDGMENT))
    bad["trace_b"]["corruption_type"] = "typo_error"  # not in the six-type taxonomy + "none"
    with pytest.raises(JudgeError, match="corruption_type"):
        _parse_judgment_response(json.dumps(bad))


def test_parse_judgment_response_raises_on_invalid_preferred():
    bad = json.loads(json.dumps(_VALID_JUDGMENT))
    bad["preferred"] = "C"
    with pytest.raises(JudgeError, match="preferred"):
        _parse_judgment_response(json.dumps(bad))


def test_build_result_row_maps_a_back_to_chosen_when_a_is_chosen():
    # _VALID_JUDGMENT prefers "A" -- with a_is_chosen=True, A *is* chosen,
    # so this should read as the judge agreeing with construction.
    row = _build_result_row(_SAMPLE_PAIR, a_is_chosen=True, verdict=_VALID_JUDGMENT)
    assert row["judge_preferred"] == "chosen"
    assert row["judge_agrees_with_construction"] is True
    assert row["chosen_score"] == 8
    assert row["rejected_score"] == 1
    assert row["margin"] == 7
    assert row["corruption_type_detected"] == "causal_flip"
    assert row["corruption_type_match"] is True  # pair's perturbation_type is causal_flip


def test_build_result_row_undoes_the_swap_when_a_is_rejected():
    # Same _VALID_JUDGMENT (still prefers "A"), but now A is the REJECTED
    # trace -- so preferring A means disagreeing with construction, and the
    # score/corruption fields must swap sides accordingly.
    row = _build_result_row(_SAMPLE_PAIR, a_is_chosen=False, verdict=_VALID_JUDGMENT)
    assert row["judge_preferred"] == "rejected"
    assert row["judge_agrees_with_construction"] is False
    assert row["chosen_score"] == 1
    assert row["rejected_score"] == 8
    assert row["margin"] == -7
    assert row["corruption_type_detected"] == "none"
    assert row["corruption_type_match"] is False  # "none" != pair's causal_flip


def test_build_result_row_tie_has_no_construction_agreement_verdict():
    tie_judgment = json.loads(json.dumps(_VALID_JUDGMENT))
    tie_judgment["preferred"] = "tie"
    row = _build_result_row(_SAMPLE_PAIR, a_is_chosen=True, verdict=tie_judgment)
    assert row["judge_preferred"] == "tie"
    assert row["judge_agrees_with_construction"] is None
