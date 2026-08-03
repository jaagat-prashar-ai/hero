# SPDX-License-Identifier: Apache-2.0
"""Tests for code_reward_entry's pure reward-shaping helpers.

Per the project's no-fake-model-tests preference, compute_reward_batch
(which needs the recipe venv's alpamayo1_x_rl + tokenizers) is NOT tested
here -- real verification is the canary cluster run. These tests pin down
the soft-gate blend's contract: far from the gates it reproduces the two
branch formulas exactly (same endpoints as the hard gate it replaced), and
across the gates it is continuous and monotone, which is the whole point of
the 2026-08-02 change (run n3sxdq's population median sat 0.02 from the
hard gate, so GRPO advantages encoded threshold luck).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_posttrain.rewards.code_reward_entry import (  # noqa: E402
    _graded_failure_reward,
    _soft_gate_blend,
)

ADE_T = 3.0
R_T = -0.4


def _blended(s: float, l2: float, pass_reward: float = -0.1) -> float:
    fail = _graded_failure_reward(
        l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
    )
    return _soft_gate_blend(
        pass_reward, fail, l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T
    )


class TestSoftGateBlend:
    def test_deep_pass_region_matches_pass_formula(self):
        # 5+ tau above the reasoning gate, l2 well under threshold.
        assert _blended(-0.05, 1.0, pass_reward=0.12) == pytest.approx(0.12, abs=1e-3)

    def test_deep_fail_region_matches_graded_failure(self):
        fail = _graded_failure_reward(
            2.0, -0.9, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
        )
        assert _blended(-0.9, 2.0, pass_reward=0.12) == pytest.approx(fail, abs=1e-3)

    def test_continuous_at_reasoning_gate(self):
        # The hard gate jumped ~0.37 across s = -0.4 at l2 = 2.0; the blend
        # must move gradually: no step bigger than the local slope allows.
        eps = 1e-4
        assert abs(_blended(R_T + eps, 2.0) - _blended(R_T - eps, 2.0)) < 0.01

    def test_continuous_at_ade_gate(self):
        eps = 1e-4
        assert abs(_blended(-0.2, ADE_T - eps) - _blended(-0.2, ADE_T + eps)) < 0.01

    def test_monotone_in_reasoning_score_near_gate(self):
        # Better reasoning must never score worse anywhere near the seam --
        # the property the hard gate violated for GRPO's within-group ranks.
        scores = [R_T + i * 0.01 for i in range(-20, 21)]
        rewards = [_blended(s, 2.0) for s in scores]
        assert all(b >= a - 1e-9 for a, b in zip(rewards, rewards[1:]))

    def test_gate_midpoint_is_halfway(self):
        # At exactly (s = R_T, l2 << ADE_T) the reasoning sigmoid is 0.5.
        fail = _graded_failure_reward(
            1.0, R_T, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
        )
        expected = 0.5 * (-0.1) + 0.5 * fail
        # l2 = 1.0 is >13 tau inside the ADE gate; its sigmoid is ~1.
        assert _blended(R_T, 1.0) == pytest.approx(expected, abs=1e-3)


class TestGradedFailureUnchanged:
    def test_no_cot_stays_flat_minus_one(self):
        assert (
            _graded_failure_reward(
                2.0, -0.2, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=False
            )
            == -1.0
        )

    def test_band_bounds(self):
        for s in (-1.0, -0.7, -0.41):
            for l2 in (1.0, 3.5, 10.0):
                r = _graded_failure_reward(
                    l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
                )
                assert -1.0 <= r <= -0.5
