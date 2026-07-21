# SPDX-License-Identifier: Apache-2.0
"""
trace_reward_test.py — unit tests for trace_reward.py. Same sourcing
stance as the component verifier suites: obstacle ground truth is the
committed real fixture, features come from the real extract_features over
the shared synthetic trajectories, and claims come from parse_coc_trace.
Sentences here are corpus-STYLE (each exercises one causal-semantics path
against the fixture scene's known geometry: people ahead-right and
approaching, no rider present, cones unverifiable) — the corpus-verbatim
end-to-end coverage lives in the component suites.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.commitment_verifier import Verdict
from code_as_a_reward.obstacle_tracks import SceneObstacles
from code_as_a_reward.trace_reward import RewardConfig, score_trace, verify_causal
from pref_pairs.synthetic_trajectory_fixtures import HZ, lane_change, straight_line
from pref_pairs.trajectory_features import extract_features

_CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
_T0_US = 12_988_806
_HORIZON_US = 6_400_000
_SCENE_ID = f"{_CLIP_ID}_{_T0_US}"


@pytest.fixture(scope="module")
def scene() -> SceneObstacles:
    df = pd.read_parquet(
        Path(__file__).parent / "testdata" / f"obstacle_offline_{_CLIP_ID}.parquet"
    )
    return SceneObstacles.from_dataframe(df, _CLIP_ID)


def _score(text: str, waypoints, scene, **kwargs):
    trace = parse_coc_trace(text, scene_id=_SCENE_ID, rollout_id=0)
    features = extract_features(waypoints, HZ, _SCENE_ID, 0)
    return score_trace(trace, features, scene, **kwargs)


def _causal(text: str, waypoints, scene):
    trace = parse_coc_trace(text, scene_id=_SCENE_ID, rollout_id=0)
    features = extract_features(waypoints, HZ, _SCENE_ID, 0)
    assert len(trace.causal) == 1, "test sentence must parse to exactly one causal claim"
    return verify_causal(trace.causal[0], features, scene, _T0_US, _HORIZON_US)


def test_causal_passes_when_effects_and_a_cause_verify(scene):
    # nudge-left PASSes against the 1m leftward fixture; "pedestrians
    # ahead" PASSes against the real sidewalk group.
    verdict = _causal("Nudge left due to pedestrians ahead", lane_change(amplitude_m=1.0), scene)
    assert verdict.verdict == Verdict.PASS
    assert [v.verdict for v in verdict.effect_verdicts] == [Verdict.PASS]
    assert [v.verdict for v in verdict.cause_verdicts] == [Verdict.PASS]


def test_causal_fails_when_an_effect_contradicts_trajectory(scene):
    # The claimed nudge never happened (straight-line trajectory) even
    # though the stated cause is real.
    verdict = _causal("Nudge left due to pedestrians ahead", straight_line(), scene)
    assert verdict.verdict == Verdict.FAIL
    assert "effect" in verdict.reason


def test_causal_fails_when_every_cause_is_absent(scene):
    # The maneuver really happened, but its stated justification (a
    # scooter) is a hallucination in this scene — the exact 'parts true,
    # story false' case perceptual verification exists for.
    verdict = _causal(
        "Nudge left due to the crossing scooter", lane_change(amplitude_m=1.0), scene
    )
    assert verdict.verdict == Verdict.FAIL
    assert "cause" in verdict.reason


def test_causal_abstains_when_cause_is_unverifiable(scene):
    # Real-corpus-verbatim string: cones have no ground truth, so the
    # claim is neither confirmed nor contradicted.
    verdict = _causal(
        "Nudge left due to construction cones blocking the right side of our lane",
        lane_change(amplitude_m=1.0),
        scene,
    )
    assert verdict.verdict == Verdict.ABSTAIN


def test_causal_abstains_on_empty_cause_parse_gap(scene):
    # "heavy fog" matches no entity lexicon entry -> the parser records a
    # connective with an empty cause list, which is ITS gap, not the
    # model's unfaithfulness — must not FAIL.
    verdict = _causal("Nudge left due to heavy fog", lane_change(amplitude_m=1.0), scene)
    assert verdict.verdict == Verdict.ABSTAIN
    assert verdict.cause_verdicts == []
    assert "parse gap" in verdict.reason


def test_score_trace_aggregates_counts_precisions_and_reward(scene):
    # Two beats: beat 1 fully verifies (nudge PASS + pedestrians-ahead
    # PASS -> causal PASS); beat 2's accelerate FAILs against the
    # constant-speed fixture. Expected: atomic 2 pass / 1 fail, causal
    # 1 pass / 0 fail.
    result = _score(
        "Nudge left due to pedestrians ahead, then accelerate",
        lane_change(amplitude_m=1.0),
        scene,
    )
    reward = result.reward
    assert reward.n_pass == {"commitment": 1, "perceptual": 1, "causal": 1}
    assert reward.n_fail == {"commitment": 1}
    assert reward.atomic_precision == pytest.approx(2 / 3)
    assert reward.causal_precision == 1.0
    assert reward.decided_fraction == 1.0  # nothing abstained in this trace
    config = RewardConfig()
    expected = (
        config.atomic_weight * (2 / 3) + config.causal_weight * 1.0
    ) / (config.atomic_weight + config.causal_weight)
    expected -= config.unparsed_penalty * reward.unparsed_char_fraction
    assert reward.reward == pytest.approx(expected)


def test_score_trace_without_causal_claims_renormalizes_weights(scene):
    # No connective anywhere -> no causal claims -> reward must equal
    # atomic precision alone (renormalized), not be dragged down by a
    # missing causal component.
    result = _score("Nudge left, then accelerate", lane_change(amplitude_m=1.0), scene)
    assert result.causal_verdicts == []
    assert result.reward.causal_precision is None
    assert result.reward.reward == pytest.approx(
        result.reward.atomic_precision
        - RewardConfig().unparsed_penalty * result.reward.unparsed_char_fraction
    )


def test_score_trace_with_nothing_decided_yields_none_not_a_number(scene):
    # Only an unverifiable entity, no commitments: every verdict abstains,
    # so there is NO signal — None, not 0 or 0.5, per module docstring.
    result = _score("Construction cones ahead", straight_line(), scene)
    assert result.reward.reward is None
    assert result.reward.atomic_precision is None
    assert result.reward.decided_fraction == 0.0


def test_score_trace_requires_scene_id(scene):
    trace = parse_coc_trace("Keep lane")
    features = extract_features(straight_line(), HZ, _SCENE_ID, 0)
    with pytest.raises(ValueError, match="scene_id"):
        score_trace(trace, features, scene)
