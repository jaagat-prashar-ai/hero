# SPDX-License-Identifier: Apache-2.0
"""Pure unit tests for consistency_entry.effective_consistency -- the
per-arm score shaping for the 2026-08-26 reward ablations. No torch/cosmos
needed; run with `python3 -m pytest rl_posttrain/rewards/consistency_entry_test.py`.
"""

import pytest

from rl_posttrain.rewards.consistency_entry import _CONSISTENCY_MODES, effective_consistency

_CONSISTENT = {"unparseable": False, "malformed": False, "lon_consistent": True, "lat_consistent": True}
_INCONSISTENT = {"unparseable": False, "malformed": False, "lon_consistent": False, "lat_consistent": False}
_LON_ONLY = {"unparseable": False, "malformed": False, "lon_consistent": True, "lat_consistent": False}
_UNPARSEABLE = {"unparseable": True, "malformed": False}
_MALFORMED = {"unparseable": False, "malformed": True}


def test_binary_passes_raw_score_through():
    assert effective_consistency(1.0, _CONSISTENT, "binary") == 1.0
    assert effective_consistency(0.0, _INCONSISTENT, "binary") == 0.0
    assert effective_consistency(0.0, _UNPARSEABLE, "binary") == 0.0
    assert effective_consistency(0.0, _MALFORMED, "binary") == 0.0


def test_two_tier_halves_unparseable_only():
    assert effective_consistency(1.0, _CONSISTENT, "two_tier") == 1.0
    assert effective_consistency(0.0, _INCONSISTENT, "two_tier") == 0.0
    assert effective_consistency(0.0, _UNPARSEABLE, "two_tier") == 0.5
    # Malformed completions keep the full penalty in every mode.
    assert effective_consistency(0.0, _MALFORMED, "two_tier") == 0.0


def test_axis_partial_gives_half_credit_per_axis():
    assert effective_consistency(1.0, _CONSISTENT, "axis_partial") == 1.0
    assert effective_consistency(0.0, _INCONSISTENT, "axis_partial") == 0.0
    assert effective_consistency(0.0, _LON_ONLY, "axis_partial") == 0.5
    assert effective_consistency(0.0, _UNPARSEABLE, "axis_partial") == 0.0
    assert effective_consistency(0.0, _MALFORMED, "axis_partial") == 0.0


def test_unparseable_neutral_exempts_unparseable_only():
    assert effective_consistency(1.0, _CONSISTENT, "unparseable_neutral") == 1.0
    assert effective_consistency(0.0, _INCONSISTENT, "unparseable_neutral") == 0.0
    assert effective_consistency(0.0, _UNPARSEABLE, "unparseable_neutral") == 1.0
    assert effective_consistency(0.0, _MALFORMED, "unparseable_neutral") == 0.0


def test_missing_axis_keys_score_zero_not_raise():
    # score_consistency omits lon/lat_consistent for unparseable traces;
    # a bare diag must never crash the reward path.
    assert effective_consistency(0.0, {}, "axis_partial") == 0.0
    assert effective_consistency(0.0, {}, "two_tier") == 0.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        effective_consistency(1.0, _CONSISTENT, "nonsense")


def test_mode_roster_matches_launch_configs():
    assert _CONSISTENCY_MODES == ("binary", "two_tier", "axis_partial", "unparseable_neutral")
