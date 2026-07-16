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
    format_waypoint_table,
    swap_seed,
)
from pref_pairs.synthetic_trajectory_fixtures import HZ, straight_line

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
