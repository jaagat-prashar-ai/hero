# SPDX-License-Identifier: Apache-2.0
"""Tests for llm_judge's pure helpers.

Per the project's no-fake-model-tests preference, the API-calling path
(judge_trace) and the full compute_reward (which needs the recipe venv's
alpamayo1_x_rl + a live model) are NOT tested here -- real verification is
the canary cluster run. These tests cover only pure formatting/parsing/
normalization, using real corpus strings from the judged-pairs dataset
where a trace is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_posttrain.rewards.aggregated_reward_llm_judge import (  # noqa: E402
    _graded_failure_reward,
    _run_judges_parallel,
)
from rl_posttrain.rewards.llm_judge import (  # noqa: E402
    JudgeRewardError,
    _build_user_message,
    _parse_single_judgment,
    normalize_score,
    waypoint_table_from_xyz,
)

# Real chosen_trace from judged_pairs.jsonl (scene 00bbc8b2..._12206610) --
# corpus-derived per the project's testing convention, not invented text.
_REAL_TRACE = "Keep distance to the lead vehicle since it is directly ahead in our lane"


def _straight_constant_speed_xyz(n: int = 64, step_m: float = 1.0) -> np.ndarray:
    """Straight-line constant-speed trajectory: x advances step_m per
    waypoint, y/z stay 0 -- analytically known kinematics, same style as
    pref_pairs/synthetic_trajectory_fixtures.py."""
    xyz = np.zeros((n, 3), dtype=np.float64)
    xyz[:, 0] = np.arange(1, n + 1) * step_m
    return xyz


class TestNormalizeScore:
    def test_bounds_map_to_recipe_scale(self):
        # The vendored reward expects grader output in [-1, 0] after its
        # `raw - 1.0` shift; our mapping must land in the same interval.
        assert normalize_score(0) == -1.0
        assert normalize_score(10) == 0.0

    def test_threshold_alignment(self):
        # The recipe's reasoning_threshold is -0.4 with a strict `>` gate:
        # judge score 6 must sit exactly AT the threshold (rejected), 7
        # clearly above -- matching the judged-pairs calibration where
        # corrupted traces scored median 1 and real traces median 7.
        assert normalize_score(6) == pytest.approx(-0.4)
        assert normalize_score(7) > -0.4
        assert normalize_score(5) < -0.4


class TestWaypointTable:
    def test_matches_pairwise_judge_format(self):
        # One `i: x=.., y=.., h=..` line per waypoint -- the exact format the
        # pairwise judge used for the 717 judged pairs (delegated to the same
        # format_waypoint_table), so the RL judge reads identical tables.
        table = waypoint_table_from_xyz(_straight_constant_speed_xyz())
        lines = table.splitlines()
        assert len(lines) == 64
        assert lines[0] == "0: x=1.0, y=0.0, h=0"
        assert lines[-1].startswith("63: x=64.0")

    def test_accepts_plain_lists(self):
        # decode_rollout_trajectory hands us tensors -> .cpu().numpy(), but
        # the helper is documented to take any (T, 3) array-like.
        as_list = _straight_constant_speed_xyz(8).tolist()
        assert waypoint_table_from_xyz(as_list).count("\n") == 7


class TestBuildUserMessage:
    def test_contains_trace_and_table(self):
        table = waypoint_table_from_xyz(_straight_constant_speed_xyz())
        msg = _build_user_message(_REAL_TRACE, table)
        assert f'Reasoning trace: "{_REAL_TRACE}"' in msg
        assert table in msg
        # Single-trace prompt: the pairwise A/B framing must not leak in.
        assert "Trace A" not in msg and "Trace B" not in msg


class TestParseSingleJudgment:
    def test_valid_judgment(self):
        parsed = _parse_single_judgment(
            '{"action_consistency_score": 7, "one_line_rationale": "gap-keeping fits steady following"}'
        )
        assert parsed["action_consistency_score"] == 7

    def test_markdown_fence_stripped(self):
        # The system prompt forbids fences but models occasionally add them;
        # the shared _extract_json_object normalization must handle it.
        parsed = _parse_single_judgment(
            '```json\n{"action_consistency_score": 3, "one_line_rationale": "r"}\n```'
        )
        assert parsed["action_consistency_score"] == 3

    @pytest.mark.parametrize(
        "bad",
        [
            '{"one_line_rationale": "missing score"}',
            '{"action_consistency_score": 11, "one_line_rationale": "out of range"}',
            '{"action_consistency_score": -1, "one_line_rationale": "out of range"}',
            '{"action_consistency_score": 6.5, "one_line_rationale": "not an int"}',
            '{"action_consistency_score": "7", "one_line_rationale": "string score"}',
            '{"action_consistency_score": true, "one_line_rationale": "bool is not a score"}',
        ],
    )
    def test_invalid_judgments_raise(self, bad):
        with pytest.raises(JudgeRewardError):
            _parse_single_judgment(bad)

    def test_non_json_raises_jsondecode(self):
        # judge_trace catches JSONDecodeError separately from JudgeRewardError;
        # both trigger the same fresh-call retry.
        import json

        with pytest.raises(json.JSONDecodeError):
            _parse_single_judgment("I think this trace is quite good.")


class TestRunJudgesParallel:
    # _run_judges_parallel is the pure fan-out layer of the batched reward
    # path (group_reward_calculation); judge_fn is injected, so these tests
    # need no network -- the real judge is only exercised by the canary run.

    def test_preserves_input_order(self):
        xyz = _straight_constant_speed_xyz()
        jobs = [(f"trace scoring {n}", xyz) for n in (3, 9, 6)]
        scores = _run_judges_parallel(jobs, judge_fn=lambda cot, _: int(cot.split()[-1]))
        assert scores == [3, 9, 6]

    @pytest.mark.parametrize("empty_cot", [None, "", "   "])
    def test_missing_cot_skipped_without_judge_call(self, empty_cot):
        calls = []

        def judge(cot, _):
            calls.append(cot)
            return 7

        scores = _run_judges_parallel(
            [(_REAL_TRACE, None), (empty_cot, None)], judge_fn=judge
        )
        assert scores == [7, None]
        assert calls == [_REAL_TRACE]

    def test_all_empty_returns_all_none(self):
        assert _run_judges_parallel([("", None), (None, None)], judge_fn=None) == [None, None]

    def test_judge_error_propagates(self):
        # Fail-loud policy: a persistent judge failure must crash the reward
        # task, not silently feed a placeholder score into GRPO.
        def judge(cot, _):
            raise JudgeRewardError("persistent API failure")

        with pytest.raises(JudgeRewardError):
            _run_judges_parallel([(_REAL_TRACE, None)], judge_fn=judge)

    def test_actually_runs_concurrently(self):
        # Two judge calls that each block until the other has started can
        # only both finish if they run on separate threads -- the entire
        # point of the batched path.
        import threading

        barrier = threading.Barrier(2, timeout=5)

        def judge(cot, _):
            barrier.wait()
            return 5

        scores = _run_judges_parallel(
            [(_REAL_TRACE, None), (_REAL_TRACE, None)], judge_fn=judge, max_workers=2
        )
        assert scores == [5, 5]


class TestGradedFailureReward:
    # Recipe gate constants (see compute_reward_batch).
    _GATES = {"ade_threshold": 3.0, "reasoning_threshold": -0.4}

    def _r(self, l2, reasoning, cot_decoded=True):
        return _graded_failure_reward(
            l2, reasoning, cot_decoded=cot_decoded, **self._GATES
        )

    def test_band_bounds(self):
        # Worst case (huge l2, reasoning at the floor) approaches -1.0; best
        # case (both quantities exactly at their gates) is the band ceiling.
        assert self._r(1e9, -1.0) == pytest.approx(-1.0, abs=1e-6)
        assert self._r(3.0, -0.4) == pytest.approx(-0.5)

    def test_always_below_passing_region(self):
        # The passing branch's floor with the current TOML weights
        # (traj_l2=0.2, comfort=0.0, reasoning=0.3) is -0.2 (l2 -> 3.0,
        # judge score 10). Every graded failure must rank strictly below it.
        passing_floor = -0.2
        for l2, reasoning in [(3.0, -0.4), (3.0001, 0.0), (17.0, -0.2), (0.5, -0.9)]:
            assert self._r(l2, reasoning) <= -0.5 < passing_floor

    def test_monotone_in_l2(self):
        # Groups whose rollouts all blew the ADE gate (e.g. the all-l2>8
        # groups seen on canary u0j67p) now carry within-group ordering:
        # closer trajectories rank higher instead of all sitting at -1.0.
        rewards = [self._r(l2, -0.2) for l2 in (4.0, 8.0, 16.0)]
        assert rewards == sorted(rewards, reverse=True)

    def test_monotone_in_reasoning(self):
        rewards = [self._r(1.0, r) for r in (-0.5, -0.7, -0.9)]
        assert rewards == sorted(rewards, reverse=True)

    def test_failing_gate_capped_by_other_gate(self):
        # A rollout failing ONLY the reasoning gate has its l2 closeness
        # capped at 1.0 -- an extra-short l2 must not outrank a shorter one
        # through the failure band.
        assert self._r(0.5, -0.9) == pytest.approx(self._r(2.9, -0.9))

    def test_missing_cot_stays_flat(self):
        # No decoded CoC keeps the vendored flat -1.0 penalty: nothing to
        # grade, and l2-closeness alone must not make dropping the CoC cheap.
        assert self._r(0.1, -1.0, cot_decoded=False) == -1.0
