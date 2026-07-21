# SPDX-License-Identifier: Apache-2.0
"""
trace_reward.py — the top of the code-as-a-reward pipeline: composes the
commitment verifier (claims vs. TrajectoryFeatures) and the perceptual
verifier (claims vs. obstacle.offline) into per-CausalClaim verdicts and a
per-trace scalar reward.

Causal semantics — deliberately the CONJUNCTION check, not causation:
a CausalClaim ("nudge left BECAUSE cones blocking our lane") passes when
every commitment on its effect side verifies AND at least one perceptual
claim on its cause side verifies. That already catches the two dominant
unfaithfulness modes (claimed a maneuver that didn't happen; justified a
maneuver with something that wasn't there). What it does NOT check is the
causal LINK itself — the pedestrian may exist and the car may have slowed
without the one causing the other. Testing the link needs counterfactual
machinery (did trajectories without that actor slow too?) that the
pref-pairs K-rollout data could support later; scoring it here as if
conjunction were causation would overclaim, so the limitation is stated
rather than hidden.

Reward shape — precision over DECIDED claims, with ABSTAIN excluded from
the denominator: an undecidable claim (Phase 0's ~27%) must move the
score toward neither 0 nor 1, otherwise the model gets punished (or
credited) for our missing ground truth. Coverage is reported alongside —
a trace with 2 decided claims out of 14 is a very different measurement
than 12 of 14 at the same precision — and unparsed text subtracts a small
penalty so reasoning the parser couldn't account for can't quietly
inflate the score. All weights live in RewardConfig with documented
first-cut defaults, to be calibrated against hand-labeled traces
alongside the verifier thresholds.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from code_as_a_reward.coc_claim_parser import CausalClaim, ParsedCoCTrace
from code_as_a_reward.commitment_verifier import (
    CommitmentVerdict,
    Verdict,
    VerifierThresholds,
    verify_commitment,
    verify_trace_commitments,
)
from code_as_a_reward.obstacle_tracks import SceneObstacles
from code_as_a_reward.perceptual_verifier import (
    PerceptualThresholds,
    PerceptualVerdict,
    split_scene_id,
    verify_perceptual,
    verify_trace_perceptual,
)
from pref_pairs.trajectory_features import TrajectoryFeatures


@dataclasses.dataclass
class CausalVerdict:
    """One CausalClaim's conjunction-check result, with the component
    verdicts attached (same auditability contract as the component
    verifiers: aggregation must never require re-running anything)."""

    claim: CausalClaim
    verdict: Verdict
    effect_verdicts: list[CommitmentVerdict]
    cause_verdicts: list[PerceptualVerdict]
    reason: str


def verify_causal(
    claim: CausalClaim,
    features: TrajectoryFeatures,
    scene: SceneObstacles,
    t0_us: int,
    horizon_us: int,
    commitment_thresholds: VerifierThresholds | None = None,
    perceptual_thresholds: PerceptualThresholds | None = None,
) -> CausalVerdict:
    """Conjunction check for one CausalClaim (module docstring):

      * FAIL if any effect commitment fails or every cause perceptual
        claim fails — some part of the stated story is false.
      * ABSTAIN if the claim is structurally unjudgeable: no effects
        parsed, or empty cause list (the parser documents an empty cause
        as a PARSE gap, not as "no reason was stated" — judging it would
        judge the parser, not the model), or the decidable parts all
        abstained.
      * PASS only when all effects pass and at least one cause passes
        ("at least one": a beat often names several observations; the
        stated justification holds if a real one is among them, and
        demanding all would fail claims for the dataset's missing cone/
        signal ground truth even when the load-bearing cause is real).
    """
    effect_verdicts = [verify_commitment(c, features, commitment_thresholds) for c in claim.effects]
    cause_verdicts = [
        verify_perceptual(p, scene, t0_us, horizon_us, perceptual_thresholds)
        for p in claim.cause
    ]

    if not effect_verdicts or not cause_verdicts:
        missing = "effects" if not effect_verdicts else "cause"
        return CausalVerdict(
            claim, Verdict.ABSTAIN, effect_verdicts, cause_verdicts,
            f"no {missing} parsed from this beat (parse gap, not model unfaithfulness)",
        )

    effect_states = {v.verdict for v in effect_verdicts}
    cause_states = {v.verdict for v in cause_verdicts}

    if Verdict.FAIL in effect_states:
        verdict, reason = Verdict.FAIL, "an effect commitment contradicts the trajectory"
    elif cause_states == {Verdict.FAIL}:
        verdict, reason = Verdict.FAIL, "every stated cause is absent from the scene"
    elif Verdict.PASS in cause_states and effect_states == {Verdict.PASS}:
        verdict, reason = Verdict.PASS, "all effects verified and a stated cause is real"
    else:
        verdict, reason = (
            Verdict.ABSTAIN,
            "decidable parts don't contradict the claim, but not all of it is verifiable",
        )
    return CausalVerdict(claim, verdict, effect_verdicts, cause_verdicts, reason)


@dataclasses.dataclass
class RewardConfig:
    """Aggregation weights. First-cut defaults, documented so recalibration
    is a deliberate act (same stance as VerifierThresholds tier 2)."""

    # Relative weight of atomic-claim precision (commitments + perceptual)
    # vs causal-claim precision. Atomic gets the larger share: causal
    # verdicts are conjunctions OF the atomic ones, so weighting them
    # equally would double-count the components relative to the one thing
    # causal adds (the pairing).
    atomic_weight: float = 0.7
    causal_weight: float = 0.3
    # Subtracted per unit of unparsed-character fraction: reasoning the
    # parser couldn't account for shouldn't inflate precision computed on
    # the part it could. Small because unparsed text is usually benign
    # connective prose, not hidden claims (see corpus notes).
    unparsed_penalty: float = 0.1


@dataclasses.dataclass
class TraceReward:
    """One trace's aggregate score plus everything needed to audit it.
    `reward` is None when NOTHING was decided — a trace whose claims all
    abstained carries no signal and must be distinguishable from a
    mediocre one (0.5-ish), especially if these scores feed DPO pair
    mining later."""

    scene_id: str | None
    rollout_id: int | None
    n_pass: dict[str, int]  # per family: "commitment" / "perceptual" / "causal"
    n_fail: dict[str, int]
    n_abstain: dict[str, int]
    atomic_precision: float | None  # pass/(pass+fail) over commitments+perceptual
    causal_precision: float | None
    decided_fraction: float  # decided / all claims, the coverage caveat
    unparsed_char_fraction: float
    reward: float | None

    def to_row_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "scene_id": self.scene_id,
            "rollout_id": self.rollout_id,
            "atomic_precision": self.atomic_precision,
            "causal_precision": self.causal_precision,
            "decided_fraction": self.decided_fraction,
            "unparsed_char_fraction": self.unparsed_char_fraction,
            "reward": self.reward,
        }
        for family in ("commitment", "perceptual", "causal"):
            row[f"n_pass_{family}"] = self.n_pass.get(family, 0)
            row[f"n_fail_{family}"] = self.n_fail.get(family, 0)
            row[f"n_abstain_{family}"] = self.n_abstain.get(family, 0)
        return row


@dataclasses.dataclass
class TraceVerification:
    """Full verification output for one trace: every verdict, plus the
    aggregate. This is the pipeline's unit of output — one per rollout."""

    trace: ParsedCoCTrace
    commitment_verdicts: list[CommitmentVerdict]
    perceptual_verdicts: list[PerceptualVerdict]
    causal_verdicts: list[CausalVerdict]
    reward: TraceReward


def _precision(n_pass: int, n_fail: int) -> float | None:
    decided = n_pass + n_fail
    return (n_pass / decided) if decided else None


def score_trace(
    trace: ParsedCoCTrace,
    features: TrajectoryFeatures,
    scene: SceneObstacles,
    horizon_us: int = 6_400_000,
    commitment_thresholds: VerifierThresholds | None = None,
    perceptual_thresholds: PerceptualThresholds | None = None,
    config: RewardConfig | None = None,
) -> TraceVerification:
    """Verify every claim in one trace and aggregate to a scalar reward.

    The trace/features/scene pairing is validated by the component
    verifiers (scene_id + rollout_id against features, clip_id against
    scene) — mispairing raises rather than scoring garbage."""
    config = config or RewardConfig()
    if trace.scene_id is None:
        raise ValueError("trace has no scene_id; cannot locate its rollout window")
    _clip_id, t0_us = split_scene_id(trace.scene_id)

    commitment_verdicts = verify_trace_commitments(trace, features, commitment_thresholds)
    perceptual_verdicts = verify_trace_perceptual(trace, scene, horizon_us, perceptual_thresholds)
    causal_verdicts = [
        verify_causal(
            c, features, scene, t0_us, horizon_us, commitment_thresholds, perceptual_thresholds
        )
        for c in trace.causal
    ]

    n_pass: dict[str, int] = {}
    n_fail: dict[str, int] = {}
    n_abstain: dict[str, int] = {}
    for family, verdicts in (
        ("commitment", commitment_verdicts),
        ("perceptual", perceptual_verdicts),
        ("causal", causal_verdicts),
    ):
        for v in verdicts:
            bucket = {Verdict.PASS: n_pass, Verdict.FAIL: n_fail, Verdict.ABSTAIN: n_abstain}[
                v.verdict
            ]
            bucket[family] = bucket.get(family, 0) + 1

    atomic_pass = n_pass.get("commitment", 0) + n_pass.get("perceptual", 0)
    atomic_fail = n_fail.get("commitment", 0) + n_fail.get("perceptual", 0)
    atomic_precision = _precision(atomic_pass, atomic_fail)
    causal_precision = _precision(n_pass.get("causal", 0), n_fail.get("causal", 0))

    n_claims = len(commitment_verdicts) + len(perceptual_verdicts) + len(causal_verdicts)
    n_decided = sum(n_pass.values()) + sum(n_fail.values())
    decided_fraction = (n_decided / n_claims) if n_claims else 0.0

    unparsed_chars = sum(e - s for s, e in trace.unparsed_spans)
    unparsed_fraction = (unparsed_chars / len(trace.raw_text)) if trace.raw_text else 0.0

    # Weighted mean over the precision components that exist, renormalized
    # so a trace with no causal claims isn't penalized for their absence;
    # None (no signal at all) when neither component was decided.
    components = [
        (config.atomic_weight, atomic_precision),
        (config.causal_weight, causal_precision),
    ]
    available = [(w, p) for w, p in components if p is not None]
    if available:
        total_w = sum(w for w, _p in available)
        reward: float | None = sum(w * p for w, p in available) / total_w
        reward = max(0.0, reward - config.unparsed_penalty * unparsed_fraction)
    else:
        reward = None

    return TraceVerification(
        trace=trace,
        commitment_verdicts=commitment_verdicts,
        perceptual_verdicts=perceptual_verdicts,
        causal_verdicts=causal_verdicts,
        reward=TraceReward(
            scene_id=trace.scene_id,
            rollout_id=trace.rollout_id,
            n_pass=n_pass,
            n_fail=n_fail,
            n_abstain=n_abstain,
            atomic_precision=atomic_precision,
            causal_precision=causal_precision,
            decided_fraction=decided_fraction,
            unparsed_char_fraction=unparsed_fraction,
            reward=reward,
        ),
    )
