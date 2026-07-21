# SPDX-License-Identifier: Apache-2.0
"""
perceptual_verifier_test.py — unit tests for perceptual_verifier.py over
the committed REAL obstacle.offline fixture (clip f0d61901-..., rollout
window t0=12988806us + 6.4s — the same scene the parser tests' reasoning
fixture describes). Expected outcomes were read off the real geometry at
commit time (people ahead-right and approaching on the right side, no
lateral sign flips, no rider within presence distance, nobody in ego's
forward corridor): if a fixture refresh changes them, that's a real data
change to investigate, not a test to loosen. Claims are built directly
(the verifier's input contract) except the end-to-end test, which parses
a real corpus string, same stance as commitment_verifier_test.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from code_as_a_reward.coc_claim_parser import PerceptualClaim, parse_coc_trace
from code_as_a_reward.commitment_verifier import Verdict
from code_as_a_reward.obstacle_tracks import SceneObstacles
from code_as_a_reward.perceptual_verifier import (
    ENTITY_TO_CLASSES,
    PerceptualThresholds,
    split_scene_id,
    verify_perceptual,
    verify_trace_perceptual,
)

_CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
_T0_US = 12_988_806
_HORIZON_US = 6_400_000


@pytest.fixture(scope="module")
def scene() -> SceneObstacles:
    df = pd.read_parquet(
        Path(__file__).parent / "testdata" / f"obstacle_offline_{_CLIP_ID}.parquet"
    )
    return SceneObstacles.from_dataframe(df, _CLIP_ID)


def _claim(entity: str, state: str | None = None) -> PerceptualClaim:
    return PerceptualClaim(text=entity, entity=entity, state=state, span=(0, len(entity)))


def _verify(scene, entity, state=None, thresholds=None):
    return verify_perceptual(_claim(entity, state), scene, _T0_US, _HORIZON_US, thresholds)


def test_split_scene_id_roundtrip_and_rejection():
    assert split_scene_id(f"{_CLIP_ID}_{_T0_US}") == (_CLIP_ID, _T0_US)
    with pytest.raises(ValueError, match="scene_id"):
        split_scene_id("no-t0-suffix")


def test_unverifiable_entity_abstains_with_no_ground_truth_rule(scene):
    verdict = _verify(scene, "construction_cones", "blocking")
    assert verdict.verdict == Verdict.ABSTAIN
    assert verdict.existence == Verdict.ABSTAIN
    assert verdict.rule == "no_ground_truth"


def test_existing_entity_with_no_state_passes(scene):
    # People genuinely are near ego in this window (right-side sidewalk
    # group), and vehicles are too.
    for entity in ("pedestrian", "vehicle_generic"):
        verdict = _verify(scene, entity)
        assert verdict.verdict == Verdict.PASS, entity
        assert verdict.existence == Verdict.PASS
        assert verdict.state is None


def test_absent_entity_fails_existence(scene):
    # The clip's one rider track never comes within presence distance of
    # ego during this rollout window — a claimed scooter/cyclist is a
    # hallucination here, the core case this verifier exists to catch.
    verdict = _verify(scene, "cyclist")
    assert verdict.verdict == Verdict.FAIL
    assert verdict.existence == Verdict.FAIL
    assert verdict.evidence["n_matching_tracks"] == 0


def test_presence_distance_threshold_controls_existence(scene):
    tight = PerceptualThresholds(presence_max_distance_m=1.0)
    verdict = _verify(scene, "pedestrian", thresholds=tight)
    assert verdict.existence == Verdict.FAIL


def test_state_checks_against_real_geometry(scene):
    # Real window geometry: the sidewalk group is AHEAD of ego (mean
    # forward > 0) and APPROACHING (ego closes ~20m on them), but nobody's
    # lateral offset changes sign (no CROSSING), nobody moves away by more
    # than the delta (no PULLING_AWAY), and everyone is >6m off ego's
    # heading line (no BLOCKING).
    assert _verify(scene, "pedestrian", "ahead").verdict == Verdict.PASS
    assert _verify(scene, "pedestrian", "approaching").verdict == Verdict.PASS
    for failing_state in ("crossing", "pulling_away", "blocking"):
        verdict = _verify(scene, "pedestrian", failing_state)
        assert verdict.verdict == Verdict.FAIL, failing_state
        assert verdict.existence == Verdict.PASS  # the people ARE there...
        assert verdict.state == Verdict.FAIL  # ...just not doing the claimed thing


def test_undecidable_state_abstains_but_keeps_existence(scene):
    verdict = _verify(scene, "pedestrian", "stopped")
    assert verdict.existence == Verdict.PASS
    assert verdict.state == Verdict.ABSTAIN
    assert verdict.verdict == Verdict.ABSTAIN


def test_attribute_entities_cap_pass_at_abstain_but_can_still_fail(scene):
    # Vehicles are present, so stopped_vehicle's existence PASSes — but
    # 'stopped' has no ground truth, so the combined verdict is capped.
    capped = _verify(scene, "stopped_vehicle")
    assert capped.existence == Verdict.PASS
    assert capped.verdict == Verdict.ABSTAIN
    assert capped.evidence["attribute_unverified"] == "stopped"
    # The cap is asymmetric: with no vehicle anywhere near ego the claim
    # still FAILS outright.
    tight = PerceptualThresholds(presence_max_distance_m=1.0)
    assert _verify(scene, "stopped_vehicle", thresholds=tight).verdict == Verdict.FAIL


def test_every_entity_key_produces_a_wellformed_verdict(scene):
    for entity in ENTITY_TO_CLASSES:
        verdict = _verify(scene, entity, "ahead")
        assert verdict.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.ABSTAIN)
        assert verdict.reason and verdict.rule


def test_real_corpus_string_end_to_end(scene):
    # Real corpus run-on (parser docstring). Its perceptual claims against
    # THIS scene: the crossing scooter is a FAIL (no rider near ego — see
    # test_absent_entity_fails_existence), the right-turn signal and ramp
    # are ABSTAIN (no ground truth for either).
    trace = parse_coc_trace(
        "Change lanes to the right and enter the freeway on-ramp because, after "
        "yielding to the crossing scooter, the right-turn signal permits movement "
        "and a safe gap exists to merge onto the ramp behind traffic, then "
        "accelerate to match ramp speed",
        scene_id=f"{_CLIP_ID}_{_T0_US}",
        rollout_id=0,
    )
    verdicts = {v.claim.entity: v for v in verify_trace_perceptual(trace, scene)}
    assert verdicts["cyclist"].verdict == Verdict.FAIL
    assert verdicts["signal"].verdict == Verdict.ABSTAIN
    assert verdicts["ramp_or_freeway"].verdict == Verdict.ABSTAIN


def test_trace_scene_mismatch_and_missing_scene_id_raise(scene):
    mismatched = parse_coc_trace("Keep lane", scene_id=f"other-clip_{_T0_US}", rollout_id=0)
    with pytest.raises(ValueError, match="clip"):
        verify_trace_perceptual(mismatched, scene)
    anonymous = parse_coc_trace("Keep lane")
    with pytest.raises(ValueError, match="scene_id"):
        verify_trace_perceptual(anonymous, scene)
