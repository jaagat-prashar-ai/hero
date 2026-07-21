# SPDX-License-Identifier: Apache-2.0
"""
commitment_verifier_test.py — unit tests for commitment_verifier.py.

Two deliberate sourcing rules, mirroring how the rest of this pipeline
tests itself:

  * Feature rows are NEVER hand-typed. Every TrajectoryFeatures here comes
    from the REAL extract_features over pref_pairs'
    synthetic_trajectory_fixtures (the same analytically-known waypoint
    arrays trajectory_features_test.py and classify_maneuvers_test.py use)
    — so these tests exercise the actual features the verifier will see in
    production, including smoothing effects, rather than idealized numbers
    a hand-built row would assume.
  * The end-to-end tests parse REAL corpus CoC strings (same sourcing
    stance as coc_claim_parser_test.py) so the parser->verifier seam is
    covered with text the parser was actually tuned on. Predicate-level
    tests construct CommitmentClaim directly — that's the verifier's own
    input contract, not a mock of anything.
"""

from __future__ import annotations

import pytest

from code_as_a_reward.coc_claim_parser import (
    CommitmentClaim,
    ManeuverAxis,
    parse_coc_trace,
)
from code_as_a_reward.commitment_verifier import (
    MANEUVER_VERIFIERS,
    Verdict,
    VerifierThresholds,
    verify_commitment,
    verify_trace_commitments,
)
from pref_pairs.synthetic_trajectory_fixtures import (
    HZ,
    accelerate,
    lane_change,
    ramp_to_stop,
    straight_line,
    turn,
    yield_dip,
)
from pref_pairs.trajectory_features import extract_features


def _claim(maneuver: str, direction: str | None = None) -> CommitmentClaim:
    """A minimal CommitmentClaim for predicate-level tests. axis /
    speed_profile / text / span are irrelevant to every predicate (they
    dispatch and threshold on `maneuver` and `direction` only), so dummy
    values keep each test about the one thing it asserts."""
    return CommitmentClaim(
        text=maneuver, maneuver=maneuver, axis=ManeuverAxis.LONGITUDINAL,
        speed_profile=None, direction=direction, span=(0, len(maneuver)),
    )


def _features(waypoints, scene_id: str = "scene_t", rollout_id: int = 0):
    return extract_features(waypoints, HZ, scene_id, rollout_id)


# --- longitudinal predicates ------------------------------------------------


def test_stop_passes_on_ramp_to_stop_and_fails_on_accelerate():
    assert verify_commitment(_claim("stop"), _features(ramp_to_stop())).verdict == Verdict.PASS
    assert verify_commitment(_claim("stop"), _features(accelerate())).verdict == Verdict.FAIL


def test_yield_passes_on_yield_dip_and_on_full_stop():
    # A full stop satisfies a yield claim too — stop is the stronger form
    # of "slowed for someone", not a contradiction of it.
    assert verify_commitment(_claim("yield"), _features(yield_dip())).verdict == Verdict.PASS
    assert verify_commitment(_claim("yield"), _features(ramp_to_stop())).verdict == Verdict.PASS
    assert verify_commitment(_claim("yield"), _features(straight_line())).verdict == Verdict.FAIL


def test_wait_needs_near_standstill_not_just_a_dip():
    # ramp_to_stop reaches 0 m/s -> wait satisfied; yield_dip only reaches
    # ~3 m/s, which is slowing, not waiting.
    assert verify_commitment(_claim("wait"), _features(ramp_to_stop())).verdict == Verdict.PASS
    assert verify_commitment(_claim("wait"), _features(yield_dip())).verdict == Verdict.FAIL


def test_decelerate_passes_on_dip_and_fails_on_speedup():
    assert verify_commitment(_claim("decelerate"), _features(yield_dip())).verdict == Verdict.PASS
    assert verify_commitment(_claim("decelerate"), _features(accelerate())).verdict == Verdict.FAIL


def test_accelerate_passes_on_speedup_and_fails_on_stop():
    assert verify_commitment(_claim("accelerate"), _features(accelerate())).verdict == Verdict.PASS
    assert verify_commitment(_claim("accelerate"), _features(ramp_to_stop())).verdict == Verdict.FAIL


def test_adapt_speed_accepts_any_real_speed_response():
    assert verify_commitment(_claim("adapt_speed"), _features(yield_dip())).verdict == Verdict.PASS
    assert verify_commitment(_claim("adapt_speed"), _features(accelerate())).verdict == Verdict.PASS
    # Constant speed is the one thing "adapt speed" cannot mean.
    assert verify_commitment(_claim("adapt_speed"), _features(straight_line())).verdict == Verdict.FAIL


def test_proceed_means_kept_moving():
    assert verify_commitment(_claim("proceed"), _features(straight_line())).verdict == Verdict.PASS
    # Yield-then-continue still counts as proceeding (common corpus pattern
    # "proceed after yielding") — only stopping contradicts it.
    assert verify_commitment(_claim("proceed"), _features(yield_dip())).verdict == Verdict.PASS
    assert verify_commitment(_claim("proceed"), _features(ramp_to_stop())).verdict == Verdict.FAIL


# --- lateral predicates -----------------------------------------------------
# Sign convention under test: positive y / positive heading == LEFT.


def test_nudge_left_passes_in_band_with_correct_sign():
    # amplitude 1.0m sits inside the [0.3, 2.5) nudge band, positive == left.
    feats = _features(lane_change(amplitude_m=1.0))
    assert verify_commitment(_claim("nudge", "left"), feats).verdict == Verdict.PASS


def test_nudge_fails_on_wrong_direction_and_on_overshoot():
    right_feats = _features(lane_change(amplitude_m=-1.0))
    assert verify_commitment(_claim("nudge", "left"), right_feats).verdict == Verdict.FAIL
    # A "nudge" that displaced a full lane width is a mislabeled lane
    # change — stated-vs-actual magnitude mismatch must FAIL, not pass as
    # "moved even more than promised".
    overshoot_feats = _features(lane_change(amplitude_m=3.0))
    assert verify_commitment(_claim("nudge", "left"), overshoot_feats).verdict == Verdict.FAIL


def test_lane_change_checks_magnitude_and_sign():
    right_feats = _features(lane_change(amplitude_m=-3.0))
    assert verify_commitment(_claim("lane_change", "right"), right_feats).verdict == Verdict.PASS
    assert verify_commitment(_claim("lane_change", "left"), right_feats).verdict == Verdict.FAIL
    # Nudge-scale movement does not verify a lane-change claim.
    small_feats = _features(lane_change(amplitude_m=1.0))
    assert verify_commitment(_claim("lane_change", "right"), small_feats).verdict == Verdict.FAIL


def test_lane_change_without_parsed_direction_verifies_magnitude_only():
    # direction=None is a parser limitation, not model unfaithfulness —
    # either direction's full-amplitude movement passes.
    assert (
        verify_commitment(_claim("lane_change"), _features(lane_change(amplitude_m=-3.0))).verdict
        == Verdict.PASS
    )


def test_turn_checks_heading_sign():
    left_feats = _features(turn(omega_rad_s=0.4))
    right_feats = _features(turn(omega_rad_s=-0.4))
    assert verify_commitment(_claim("turn", "left"), left_feats).verdict == Verdict.PASS
    assert verify_commitment(_claim("turn", "right"), right_feats).verdict == Verdict.PASS
    assert verify_commitment(_claim("turn", "left"), right_feats).verdict == Verdict.FAIL
    assert verify_commitment(_claim("turn", "left"), _features(straight_line())).verdict == Verdict.FAIL


def test_merge_accepts_nudge_scale_lateral_shift():
    # Merges are verified loosely (clip windows catch them at different
    # stages): nudge-scale and lane-change-scale shifts both pass.
    assert verify_commitment(_claim("merge", "left"), _features(lane_change(amplitude_m=1.0))).verdict == Verdict.PASS
    assert verify_commitment(_claim("merge", "left"), _features(lane_change(amplitude_m=3.0))).verdict == Verdict.PASS
    assert verify_commitment(_claim("merge", "left"), _features(straight_line())).verdict == Verdict.FAIL


def test_keep_lane_passes_straight_and_fails_lane_change():
    assert verify_commitment(_claim("keep_lane"), _features(straight_line())).verdict == Verdict.PASS
    assert verify_commitment(_claim("keep_lane"), _features(lane_change(amplitude_m=3.0))).verdict == Verdict.FAIL
    # A turn returning near y=0 must still fail keep_lane on its heading.
    assert verify_commitment(_claim("keep_lane"), _features(turn())).verdict == Verdict.FAIL


# --- abstentions ------------------------------------------------------------


def test_undecidable_maneuvers_abstain_not_fail():
    feats = _features(straight_line())
    for maneuver in ("enter", "exit", "keep_distance", "create_gap"):
        verdict = verify_commitment(_claim(maneuver), feats)
        assert verdict.verdict == Verdict.ABSTAIN, maneuver


def test_unknown_maneuver_abstains():
    verdict = verify_commitment(_claim("teleport"), _features(straight_line()))
    assert verdict.verdict == Verdict.ABSTAIN
    assert verdict.rule == "unknown_maneuver"


def test_every_registered_maneuver_returns_a_wellformed_verdict():
    # Exhaustiveness sweep: whatever the trajectory, every dispatch-table
    # entry must return a CommitmentVerdict with a non-empty rule/reason —
    # no predicate may raise on an arbitrary (valid) feature row.
    feats = _features(yield_dip())
    for maneuver in MANEUVER_VERIFIERS:
        verdict = verify_commitment(_claim(maneuver, "left"), feats)
        assert verdict.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.ABSTAIN)
        assert verdict.rule and verdict.reason


# --- end-to-end over real corpus strings ------------------------------------


def test_real_corpus_nudge_string_verifies_end_to_end():
    # Real corpus line (see coc_claim_parser.py module docstring). The
    # 1.0m-amplitude fixture is a leftward nudge, so the parsed
    # "nudge left" commitment should PASS against it.
    trace = parse_coc_trace(
        "Nudge left due to construction cones blocking the right side of our lane",
        scene_id="scene_e2e", rollout_id=7,
    )
    feats = _features(lane_change(amplitude_m=1.0), scene_id="scene_e2e", rollout_id=7)
    verdicts = verify_trace_commitments(trace, feats)
    assert [v.claim.maneuver for v in verdicts] == ["nudge"]
    assert verdicts[0].verdict == Verdict.PASS
    # Same claim against a RIGHTWARD nudge must fail — the direction check
    # is the whole faithfulness point.
    wrong_feats = _features(lane_change(amplitude_m=-1.0), scene_id="scene_e2e", rollout_id=7)
    assert verify_trace_commitments(trace, wrong_feats)[0].verdict == Verdict.FAIL


def test_real_corpus_compound_string_mixes_verdict_kinds():
    # Real corpus run-on (parser docstring): commitments parsed are
    # lane_change(right) + enter + accelerate. Against a rightward
    # lane-change fixture (constant forward speed): lane_change PASSes,
    # enter ABSTAINs (map feature), accelerate FAILs (no speed-up) —
    # one string exercising all three verdict kinds.
    trace = parse_coc_trace(
        "Change lanes to the right and enter the freeway on-ramp because, after "
        "yielding to the crossing scooter, the right-turn signal permits movement "
        "and a safe gap exists to merge onto the ramp behind traffic, then "
        "accelerate to match ramp speed",
        scene_id="scene_e2e", rollout_id=8,
    )
    feats = _features(lane_change(amplitude_m=-3.0), scene_id="scene_e2e", rollout_id=8)
    by_maneuver = {v.claim.maneuver: v.verdict for v in verify_trace_commitments(trace, feats)}
    assert by_maneuver["lane_change"] == Verdict.PASS
    assert by_maneuver["enter"] == Verdict.ABSTAIN
    assert by_maneuver["accelerate"] == Verdict.FAIL


def test_trace_feature_id_mismatch_raises():
    trace = parse_coc_trace("Keep lane", scene_id="scene_a", rollout_id=1)
    wrong_scene = _features(straight_line(), scene_id="scene_b", rollout_id=1)
    with pytest.raises(ValueError, match="scene_id"):
        verify_trace_commitments(trace, wrong_scene)
    wrong_rollout = _features(straight_line(), scene_id="scene_a", rollout_id=2)
    with pytest.raises(ValueError, match="rollout_id"):
        verify_trace_commitments(trace, wrong_rollout)


def test_thresholds_from_dict_reads_classifier_sections():
    # from_dict must pick tier-1 values out of the SAME yaml sections the
    # classifier reads, and tier-2 values from the commitment_verifier
    # section — this is the "one yaml, no disagreement" contract.
    thresholds = VerifierThresholds.from_dict(
        {
            "lane_change": {"lateral_offset_m": 2.0},
            "turn": {"heading_change_deg": 30.0},
            "proceed_accelerate": {"mean_accel_mps2": 0.4},
            "commitment_verifier": {"nudge_min_lateral_offset_m": 0.5},
        }
    )
    assert thresholds.lane_change_lateral_offset_m == 2.0
    assert thresholds.turn_heading_change_deg == 30.0
    assert thresholds.accelerate_mean_accel_mps2 == 0.4
    assert thresholds.nudge_min_lateral_offset_m == 0.5
    # Unspecified fields keep their documented defaults.
    assert thresholds.decelerate_min_speed_drop_mps == 1.0
