# SPDX-License-Identifier: Apache-2.0
"""
commitment_verifier.py — checks CommitmentClaims (what the CoC text says
ego WILL do, parsed by coc_claim_parser.py) against what the rollout's
trajectory ACTUALLY did (pref_pairs/trajectory_features.py's
TrajectoryFeatures row). This is the first half of the "code as a reward"
verification stage; the perceptual half (PerceptualClaim vs.
obstacle.offline actor tracks) is a separate module because it depends on
scene-state data this one deliberately never touches.

Verdict semantics — three-valued on purpose:

  * PASS    — the kinematics are consistent with the stated maneuver
              ("nudge left" and final_lateral_offset_m is a leftward shift
              in the nudge band).
  * FAIL    — the kinematics contradict the stated maneuver ("stop" but no
              stop_event; "nudge left" but the trajectory shifted right).
  * ABSTAIN — this claim is not decidable from ego kinematics alone
              ("keep distance" needs the lead vehicle's track; "enter the
              ramp" needs lane/map geometry that Phase 0 confirmed does not
              exist in the dataset). ABSTAIN is NOT a soft FAIL: folding
              undecidable claims into FAIL would penalize the model for
              our missing ground truth, exactly the failure mode the Phase
              0 taxonomy (~27% of claims unverifiable) warned about.

Sign convention (inherited from classify_maneuvers.py, ISO 8855-style):
POSITIVE final_lateral_offset_m / total_heading_change_deg mean LEFT.

Threshold provenance: wherever a threshold already exists in
pref_pairs/configs/maneuver_thresholds.yaml (lane-change lateral offset,
turn heading change, proceed/accelerate mean accel), this module defaults
to the SAME value, so the verifier never disagrees with the maneuver
classifier about what e.g. "a lane change" means. Verifier-only thresholds
(the nudge band, the decelerate speed-drop floor) are new here and
documented at their definition.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from code_as_a_reward.coc_claim_parser import (
    MANEUVER_PATTERNS,
    CommitmentClaim,
    ParsedCoCTrace,
)
from pref_pairs.trajectory_features import TrajectoryFeatures


class Verdict(str, Enum):
    """Three-valued verification outcome — see module docstring for why
    ABSTAIN is a first-class outcome and not a soft FAIL. (str, Enum) for
    the same reason as coc_claim_parser.ManeuverAxis: exhaustive branching
    downstream, plain-string JSON round-tripping."""

    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


@dataclasses.dataclass
class VerifierThresholds:
    """Tunable knobs for commitment verification.

    Two provenance tiers, kept visually separate below:

    1. SHARED with pref_pairs/configs/maneuver_thresholds.yaml — defaults
       here are copied from that file and `from_dict` reads the SAME yaml
       sections, so loading both configs from one file keeps the verifier
       and classify_maneuvers.py's rule cascade agreeing on what counts as
       a lane change / turn / acceleration. Do not retune these here
       without retuning the classifier: a verifier that calls "lane change"
       at 2.0m while the classifier requires 2.5m would produce
       maneuver-label rows and claim verdicts that contradict each other
       for the same rollout.

    2. VERIFIER-ONLY — no classifier equivalent exists. Values are initial
       judgment calls (documented per-field), expected to be recalibrated
       once verdicts can be spot-checked against hand-labeled claims.
    """

    # --- Tier 1: shared with maneuver_thresholds.yaml ---
    lane_change_lateral_offset_m: float = 2.5
    turn_heading_change_deg: float = 45.0
    accelerate_mean_accel_mps2: float = 0.5

    # --- Tier 2: verifier-only ---
    # A "nudge" is a deliberate lateral shift SMALLER than a lane change:
    # lower bound 0.3m is above smoothing/integration noise seen in real
    # rollout lateral offsets, upper bound is the lane-change threshold
    # (at which point the maneuver stops being a nudge). Half-open band
    # [min, lane_change_lateral_offset_m).
    nudge_min_lateral_offset_m: float = 0.3
    # "decelerate"/"slow down" must show a real speed drop end-to-end, not
    # just any momentarily-negative accel sample (braking jitter): initial
    # minus min speed must exceed this. 1.0 m/s (~3.6 km/h) is small enough
    # to catch a gentle comfort brake, large enough to not fire on noise.
    decelerate_min_speed_drop_mps: float = 1.0
    # "accelerate" analog of the above: final minus initial speed. Used
    # alongside (OR) the shared mean-accel threshold so a rollout that
    # accelerates late (mean diluted by an early cruise phase) still passes.
    accelerate_min_speed_gain_mps: float = 1.0
    # "proceed" means "keep moving / go": trajectory must end above this
    # speed with no stop event. 2.0 m/s matches maneuver_thresholds.yaml's
    # stop.recovery_speed_mps — i.e. "proceeding" is exactly "not stopped"
    # by the stop rule's own definition of having recovered.
    proceed_min_final_speed_mps: float = 2.0
    # "keep lane" requires staying within this lateral band AND below the
    # turn heading threshold. 0.5m is looser than nudge_min (0.3m) on
    # purpose: normal in-lane wander on a curving road shouldn't fail a
    # keep-lane claim, and verifying a stated claim is a different question
    # from classifying the single best maneuver label.
    keep_lane_max_lateral_offset_m: float = 0.5
    # "adapt speed" is the vaguest corpus commitment ("adapt/adjust
    # speed"): verified as ANY meaningful longitudinal response — a
    # stop/yield event, or an end-to-end speed change (either sign)
    # exceeding this.
    adapt_speed_min_change_mps: float = 1.0
    # "wait" is satisfied by a stop_event, but ALSO by a stop-then-go
    # ("wait for the pedestrian, then proceed") that stop_event's
    # no-recovery clause deliberately excludes — so we additionally accept
    # min_speed dipping below this. Same value as maneuver_thresholds.yaml
    # stop.speed_mps: "waiting" means the same near-standstill speed as
    # "stopped", just without the must-not-recover requirement.
    wait_max_min_speed_mps: float = 0.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerifierThresholds":
        """Build from a parsed maneuver_thresholds.yaml dict (same contract
        as trajectory_features.FeatureConfig.from_dict): tier-1 fields read
        the classifier's own sections; tier-2 fields read an OPTIONAL
        `commitment_verifier` section so one yaml can carry both configs,
        falling back to the dataclass defaults above."""
        verifier = d.get("commitment_verifier", {})
        return cls(
            lane_change_lateral_offset_m=d.get("lane_change", {}).get("lateral_offset_m", 2.5),
            turn_heading_change_deg=d.get("turn", {}).get("heading_change_deg", 45.0),
            accelerate_mean_accel_mps2=d.get("proceed_accelerate", {}).get("mean_accel_mps2", 0.5),
            nudge_min_lateral_offset_m=verifier.get("nudge_min_lateral_offset_m", 0.3),
            decelerate_min_speed_drop_mps=verifier.get("decelerate_min_speed_drop_mps", 1.0),
            accelerate_min_speed_gain_mps=verifier.get("accelerate_min_speed_gain_mps", 1.0),
            proceed_min_final_speed_mps=verifier.get("proceed_min_final_speed_mps", 2.0),
            keep_lane_max_lateral_offset_m=verifier.get("keep_lane_max_lateral_offset_m", 0.5),
            adapt_speed_min_change_mps=verifier.get("adapt_speed_min_change_mps", 1.0),
            wait_max_min_speed_mps=verifier.get("wait_max_min_speed_mps", 0.5),
        )


@dataclasses.dataclass
class CommitmentVerdict:
    """One claim's verification result, with enough evidence attached that
    a human (or the reward-aggregation stage) can audit WHY without
    re-running the verifier: `rule` names the predicate that decided it,
    `evidence` carries the exact feature values that predicate consulted,
    and `reason` is the one-line human-readable account. Keeping evidence
    on the verdict (rather than logging it) follows the pipeline's
    "no silent gaps" stance — a reward signal nobody can audit is exactly
    how a subtly-wrong verifier would poison downstream DPO unnoticed."""

    claim: CommitmentClaim
    verdict: Verdict
    rule: str  # e.g. "stop_event", "lateral_band", "abstain_needs_scene_state"
    evidence: dict[str, Any]  # feature name -> value actually consulted
    reason: str  # one-line human-readable justification

    def to_row_dict(self) -> dict[str, Any]:
        """Flat dict for a one-row-per-verdict table (same pattern as
        TrajectoryFeatures.to_row_dict): claim fields inlined with a
        claim_ prefix, evidence JSON-friendly as-is."""
        return {
            "claim_text": self.claim.text,
            "claim_maneuver": self.claim.maneuver,
            "claim_axis": self.claim.axis.value,
            "claim_direction": self.claim.direction,
            "claim_speed_profile": self.claim.speed_profile,
            "verdict": self.verdict.value,
            "rule": self.rule,
            "evidence": self.evidence,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Longitudinal predicates. Each takes (claim, features, thresholds) and
# returns a CommitmentVerdict; the shared signature is what lets the
# dispatch table (added with verify_commitment) treat them uniformly. All
# speed evidence comes from TrajectoryFeatures' SMOOTHED scalar summaries —
# these predicates never re-derive kinematics, so the verifier stays in
# agreement with whatever smoothing/thresholding produced the feature row.
# ---------------------------------------------------------------------------


def _verify_stop(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Stop" maps directly onto the feature extractor's stop_event (speed
    below stop threshold, sustained, without recovering by trajectory end)
    — no verifier-side threshold at all, so a "stop" claim verdict can
    never contradict the maneuver classifier's own stop rule."""
    evidence = {
        "stop_event": features.stop_event,
        "min_speed_mps": features.min_speed_mps,
        "final_speed_mps": features.final_speed_mps,
    }
    if features.stop_event:
        return CommitmentVerdict(
            claim, Verdict.PASS, "stop_event", evidence,
            f"stop_event detected (min speed {features.min_speed_mps:.2f} m/s)",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "stop_event", evidence,
        f"no stop_event (min speed {features.min_speed_mps:.2f} m/s, "
        f"final {features.final_speed_mps:.2f} m/s)",
    )


def _verify_yield(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Yield" passes on the extractor's yield_event (drop then partial
    recovery), OR on stop_event: coming to a full stop is a stronger form
    of yielding, not a different maneuver, and the two events are mutually
    exclusive by construction (stop requires NOT recovering, yield requires
    recovering) — so accepting either is what "slowed for someone" actually
    means at the claim level."""
    evidence = {"yield_event": features.yield_event, "stop_event": features.stop_event,
                "min_speed_mps": features.min_speed_mps}
    if features.yield_event or features.stop_event:
        which = "yield_event" if features.yield_event else "stop_event"
        return CommitmentVerdict(
            claim, Verdict.PASS, "yield_or_stop_event", evidence,
            f"{which} detected (min speed {features.min_speed_mps:.2f} m/s)",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "yield_or_stop_event", evidence,
        f"neither yield_event nor stop_event (min speed {features.min_speed_mps:.2f} m/s)",
    )


def _verify_wait(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Wait" = reached (near-)standstill at some point, recovery
    irrelevant: a stop_event qualifies, and so does a stop-then-go that
    stop_event's no-recovery clause excludes — hence the extra min_speed
    check (see wait_max_min_speed_mps' definition comment)."""
    evidence = {"stop_event": features.stop_event, "min_speed_mps": features.min_speed_mps,
                "wait_max_min_speed_mps": thresholds.wait_max_min_speed_mps}
    if features.stop_event or features.min_speed_mps < thresholds.wait_max_min_speed_mps:
        return CommitmentVerdict(
            claim, Verdict.PASS, "standstill_reached", evidence,
            f"reached near-standstill (min speed {features.min_speed_mps:.2f} m/s)",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "standstill_reached", evidence,
        f"never dropped below {thresholds.wait_max_min_speed_mps} m/s "
        f"(min speed {features.min_speed_mps:.2f} m/s)",
    )


def _verify_decelerate(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Decelerate"/"slow down"/"brake" needs a real end-to-end speed drop
    (initial minus min), not just any negative accel sample — braking
    jitter produces those on virtually every trajectory. A stop or yield
    event also passes: both are deceleration by definition, and accepting
    them keeps this predicate consistent with _verify_stop/_verify_yield
    on trajectories where the drop straddles the threshold."""
    speed_drop = features.initial_speed_mps - features.min_speed_mps
    evidence = {"initial_speed_mps": features.initial_speed_mps,
                "min_speed_mps": features.min_speed_mps,
                "speed_drop_mps": speed_drop,
                "stop_event": features.stop_event, "yield_event": features.yield_event,
                "decelerate_min_speed_drop_mps": thresholds.decelerate_min_speed_drop_mps}
    if (features.stop_event or features.yield_event
            or speed_drop >= thresholds.decelerate_min_speed_drop_mps):
        return CommitmentVerdict(
            claim, Verdict.PASS, "speed_drop", evidence,
            f"slowed by {speed_drop:.2f} m/s "
            f"(from {features.initial_speed_mps:.2f} to min {features.min_speed_mps:.2f})",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "speed_drop", evidence,
        f"speed drop {speed_drop:.2f} m/s below "
        f"{thresholds.decelerate_min_speed_drop_mps} m/s and no stop/yield event",
    )


def _verify_accelerate(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Accelerate" passes on EITHER the classifier's mean-accel rule
    (shared threshold, tier 1) OR an end-to-end speed gain — the OR matters
    because a rollout that cruises first and accelerates late has its mean
    accel diluted toward zero, yet plainly did accelerate. mean accel uses
    the native action tensor when the harvester captured it (see
    TrajectoryFeatures.accel_source), which is why it's preferred as the
    first clause rather than derived speed deltas alone."""
    speed_gain = features.final_speed_mps - features.initial_speed_mps
    evidence = {"mean_acceleration_mps2": features.mean_acceleration_mps2,
                "accel_source": features.accel_source,
                "speed_gain_mps": speed_gain,
                "accelerate_mean_accel_mps2": thresholds.accelerate_mean_accel_mps2,
                "accelerate_min_speed_gain_mps": thresholds.accelerate_min_speed_gain_mps}
    if (features.mean_acceleration_mps2 >= thresholds.accelerate_mean_accel_mps2
            or speed_gain >= thresholds.accelerate_min_speed_gain_mps):
        return CommitmentVerdict(
            claim, Verdict.PASS, "mean_accel_or_speed_gain", evidence,
            f"mean accel {features.mean_acceleration_mps2:.2f} m/s^2 ({features.accel_source}), "
            f"speed gain {speed_gain:.2f} m/s",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "mean_accel_or_speed_gain", evidence,
        f"mean accel {features.mean_acceleration_mps2:.2f} m/s^2 and "
        f"speed gain {speed_gain:.2f} m/s both below thresholds",
    )


def _verify_adapt_speed(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Adapt/adjust speed" is the corpus's vaguest commitment — it never
    says which direction. Verified as ANY meaningful longitudinal response:
    stop/yield event, or end-to-end change, or a transient dip (initial
    minus min — covers slow-then-recover, which end-to-end change misses).
    Deliberately generous: the claim itself is weak, so weak evidence
    satisfies it, and the strictness belongs to specific claims like
    "decelerate" instead."""
    end_to_end_change = abs(features.final_speed_mps - features.initial_speed_mps)
    transient_dip = features.initial_speed_mps - features.min_speed_mps
    largest_change = max(end_to_end_change, transient_dip)
    evidence = {"end_to_end_change_mps": end_to_end_change,
                "transient_dip_mps": transient_dip,
                "stop_event": features.stop_event, "yield_event": features.yield_event,
                "adapt_speed_min_change_mps": thresholds.adapt_speed_min_change_mps}
    if (features.stop_event or features.yield_event
            or largest_change >= thresholds.adapt_speed_min_change_mps):
        return CommitmentVerdict(
            claim, Verdict.PASS, "any_speed_response", evidence,
            f"speed changed by {largest_change:.2f} m/s",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "any_speed_response", evidence,
        f"largest speed change {largest_change:.2f} m/s below "
        f"{thresholds.adapt_speed_min_change_mps} m/s; no stop/yield event",
    )


def _verify_proceed(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Proceed" = kept moving: ends above the proceed floor (== the stop
    rule's recovery speed, i.e. "not stopped" by the stop rule's own
    definition) with no stop_event. A yield_event does NOT fail this —
    "proceed after yielding" is a common corpus pattern and slowing then
    continuing is still proceeding."""
    evidence = {"final_speed_mps": features.final_speed_mps,
                "stop_event": features.stop_event,
                "proceed_min_final_speed_mps": thresholds.proceed_min_final_speed_mps}
    if not features.stop_event and features.final_speed_mps >= thresholds.proceed_min_final_speed_mps:
        return CommitmentVerdict(
            claim, Verdict.PASS, "kept_moving", evidence,
            f"ended at {features.final_speed_mps:.2f} m/s with no stop event",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "kept_moving", evidence,
        f"stop_event={features.stop_event}, final speed {features.final_speed_mps:.2f} m/s",
    )


# ---------------------------------------------------------------------------
# Lateral predicates. Sign convention throughout (from classify_maneuvers.py,
# ISO 8855): POSITIVE final_lateral_offset_m / total_heading_change_deg mean
# LEFT. A claim with direction=None (the parser found no left/right word in
# the direction window) is verified on magnitude alone — absence of a parsed
# direction is a parser limitation, not a claim the model got wrong.
# ---------------------------------------------------------------------------


def _direction_consistent(direction: str | None, signed_value: float) -> bool:
    """Whether `signed_value`'s sign agrees with a claimed direction
    (positive == left). direction=None is vacuously consistent — see the
    section comment above."""
    if direction is None:
        return True
    return signed_value > 0 if direction == "left" else signed_value < 0


def _verify_nudge(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Nudge" = a deliberate lateral shift in the half-open band
    [nudge_min, lane_change threshold), sign matching the claimed
    direction. Exceeding the band FAILS rather than passes: a "nudge" that
    displaced a full lane width is a lane change the model mislabeled, and
    this verifier's job is exactly to catch stated-vs-actual mismatches,
    not to grade generosity of movement."""
    lateral = features.final_lateral_offset_m
    evidence = {"final_lateral_offset_m": lateral, "claim_direction": claim.direction,
                "nudge_min_lateral_offset_m": thresholds.nudge_min_lateral_offset_m,
                "lane_change_lateral_offset_m": thresholds.lane_change_lateral_offset_m}
    in_band = thresholds.nudge_min_lateral_offset_m <= abs(lateral) < thresholds.lane_change_lateral_offset_m
    if in_band and _direction_consistent(claim.direction, lateral):
        return CommitmentVerdict(
            claim, Verdict.PASS, "lateral_band", evidence,
            f"lateral offset {lateral:+.2f} m in nudge band, direction consistent",
        )
    if in_band:
        return CommitmentVerdict(
            claim, Verdict.FAIL, "lateral_band", evidence,
            f"nudge magnitude ok but direction wrong: claimed {claim.direction}, "
            f"offset {lateral:+.2f} m (positive=left)",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "lateral_band", evidence,
        f"lateral offset {lateral:+.2f} m outside nudge band "
        f"[{thresholds.nudge_min_lateral_offset_m}, {thresholds.lane_change_lateral_offset_m})",
    )


def _verify_lane_change(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """Lane change = the classifier's own rule verbatim (tier-1 thresholds):
    lateral offset at least the lane-change threshold AND total heading
    change BELOW the turn threshold (a large heading change means the
    trajectory turned, whatever the lateral offset says), sign matching the
    claimed direction."""
    lateral = features.final_lateral_offset_m
    heading = features.total_heading_change_deg
    evidence = {"final_lateral_offset_m": lateral, "total_heading_change_deg": heading,
                "claim_direction": claim.direction,
                "lane_change_lateral_offset_m": thresholds.lane_change_lateral_offset_m,
                "turn_heading_change_deg": thresholds.turn_heading_change_deg}
    if (abs(lateral) >= thresholds.lane_change_lateral_offset_m
            and abs(heading) < thresholds.turn_heading_change_deg
            and _direction_consistent(claim.direction, lateral)):
        return CommitmentVerdict(
            claim, Verdict.PASS, "lane_change_rule", evidence,
            f"lateral offset {lateral:+.2f} m with heading change {heading:+.1f} deg",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "lane_change_rule", evidence,
        f"lateral offset {lateral:+.2f} m / heading change {heading:+.1f} deg "
        f"inconsistent with a {claim.direction or 'stated'} lane change",
    )


def _verify_merge(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Merge" is verified more loosely than lane_change: real merges onto
    a ramp/lane can show anywhere from nudge-scale to full lane-change
    lateral displacement depending on where in the merge the clip window
    falls, so the floor is the nudge minimum, with a direction-sign check
    when a direction was stated. Without lane geometry (Phase 0: none in
    the dataset) this is the strongest check ego kinematics support."""
    lateral = features.final_lateral_offset_m
    evidence = {"final_lateral_offset_m": lateral, "claim_direction": claim.direction,
                "nudge_min_lateral_offset_m": thresholds.nudge_min_lateral_offset_m}
    if (abs(lateral) >= thresholds.nudge_min_lateral_offset_m
            and _direction_consistent(claim.direction, lateral)):
        return CommitmentVerdict(
            claim, Verdict.PASS, "lateral_shift", evidence,
            f"lateral offset {lateral:+.2f} m consistent with a merge",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "lateral_shift", evidence,
        f"lateral offset {lateral:+.2f} m shows no "
        f"{claim.direction or ''} merge movement".replace("  ", " "),
    )


def _verify_turn(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Turn" = the classifier's turn rule verbatim (tier-1 threshold):
    total heading change at least turn_heading_change_deg, sign matching
    the claimed direction (positive=left)."""
    heading = features.total_heading_change_deg
    evidence = {"total_heading_change_deg": heading, "claim_direction": claim.direction,
                "turn_heading_change_deg": thresholds.turn_heading_change_deg}
    if (abs(heading) >= thresholds.turn_heading_change_deg
            and _direction_consistent(claim.direction, heading)):
        return CommitmentVerdict(
            claim, Verdict.PASS, "turn_rule", evidence,
            f"heading change {heading:+.1f} deg",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "turn_rule", evidence,
        f"heading change {heading:+.1f} deg inconsistent with a "
        f"{claim.direction or 'stated'} turn "
        f"(threshold {thresholds.turn_heading_change_deg} deg)",
    )


def _verify_keep_lane(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Keep lane" = stayed within the keep-lane lateral band AND below the
    turn heading threshold. The heading clause matters on its own: a turn
    can end near lateral offset zero in this t=0-heading frame while
    plainly not keeping its lane."""
    lateral = features.final_lateral_offset_m
    heading = features.total_heading_change_deg
    evidence = {"final_lateral_offset_m": lateral, "total_heading_change_deg": heading,
                "keep_lane_max_lateral_offset_m": thresholds.keep_lane_max_lateral_offset_m,
                "turn_heading_change_deg": thresholds.turn_heading_change_deg}
    if (abs(lateral) <= thresholds.keep_lane_max_lateral_offset_m
            and abs(heading) < thresholds.turn_heading_change_deg):
        return CommitmentVerdict(
            claim, Verdict.PASS, "keep_lane_band", evidence,
            f"lateral offset {lateral:+.2f} m, heading change {heading:+.1f} deg",
        )
    return CommitmentVerdict(
        claim, Verdict.FAIL, "keep_lane_band", evidence,
        f"lateral offset {lateral:+.2f} m / heading change {heading:+.1f} deg "
        f"outside keep-lane bounds",
    )


def _abstain_needs_lane_geometry(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Enter"/"exit" name a MAP FEATURE (ramp, freeway, roundabout,
    driveway) that ego supposedly moved into or out of. Phase 0 established
    the dataset has no lane geometry at all, so whether the trajectory
    actually entered that feature is unknowable here — ABSTAIN, same
    stance as _abstain_needs_other_agent. (Unlike keep_distance/create_gap
    these will NOT become decidable when obstacle.offline lands; they'd
    need map data the dataset simply doesn't have.)"""
    return CommitmentVerdict(
        claim, Verdict.ABSTAIN, "needs_lane_geometry", {},
        f"'{claim.maneuver}' refers to a map feature; dataset has no lane "
        "geometry ground truth (Phase 0 finding)",
    )


def _abstain_needs_other_agent(
    claim: CommitmentClaim, features: TrajectoryFeatures, thresholds: VerifierThresholds
) -> CommitmentVerdict:
    """"Keep distance" / "create a gap" are claims about ego's position
    RELATIVE TO ANOTHER AGENT — truth depends on where that agent was,
    which ego kinematics cannot say (decelerating is neither necessary nor
    sufficient: the gap also changes when the other agent moves). These
    become decidable once obstacle.offline actor tracks are integrated;
    until then, ABSTAIN, per the module docstring's 'undecidable is not
    FAIL' stance."""
    return CommitmentVerdict(
        claim, Verdict.ABSTAIN, "needs_other_agent_track", {},
        f"'{claim.maneuver}' is relative to another agent; "
        "not decidable from ego kinematics alone (needs obstacle.offline)",
    )


# ---------------------------------------------------------------------------
# Dispatch + orchestration.
# ---------------------------------------------------------------------------

# One entry per canonical maneuver key the parser can emit. The assertion
# below makes this an ENFORCED invariant at import time: if a new maneuver
# is added to the parser's MANEUVER_PATTERNS without a verifier entry here,
# the first import of this module fails loudly instead of that maneuver's
# claims silently taking some default path — same "no silent gaps" stance
# as the parser's unparsed_spans.
MANEUVER_VERIFIERS = {
    "lane_change": _verify_lane_change,
    "keep_lane": _verify_keep_lane,
    "nudge": _verify_nudge,
    "merge": _verify_merge,
    "turn": _verify_turn,
    "enter": _abstain_needs_lane_geometry,
    "exit": _abstain_needs_lane_geometry,
    "adapt_speed": _verify_adapt_speed,
    "accelerate": _verify_accelerate,
    "decelerate": _verify_decelerate,
    "keep_distance": _abstain_needs_other_agent,
    "create_gap": _abstain_needs_other_agent,
    "stop": _verify_stop,
    "yield": _verify_yield,
    "wait": _verify_wait,
    "proceed": _verify_proceed,
}

_PARSER_MANEUVERS = {name for name, _axis, _profile, _pattern in MANEUVER_PATTERNS}
assert MANEUVER_VERIFIERS.keys() == _PARSER_MANEUVERS, (
    "commitment_verifier's dispatch table is out of sync with "
    f"coc_claim_parser.MANEUVER_PATTERNS: missing={_PARSER_MANEUVERS - MANEUVER_VERIFIERS.keys()}, "
    f"stale={MANEUVER_VERIFIERS.keys() - _PARSER_MANEUVERS}"
)


def verify_commitment(
    claim: CommitmentClaim,
    features: TrajectoryFeatures,
    thresholds: VerifierThresholds | None = None,
) -> CommitmentVerdict:
    """Verify one CommitmentClaim against one rollout's TrajectoryFeatures.

    An unknown maneuver key (can only happen if a claim was produced by a
    NEWER parser than this verifier, since same-version sync is asserted at
    import) gets ABSTAIN rather than raising: mid-migration, "we can't
    judge this yet" is true and FAIL would be wrong."""
    thresholds = thresholds or VerifierThresholds()
    verifier = MANEUVER_VERIFIERS.get(claim.maneuver)
    if verifier is None:
        return CommitmentVerdict(
            claim, Verdict.ABSTAIN, "unknown_maneuver", {},
            f"no verifier registered for maneuver '{claim.maneuver}'",
        )
    return verifier(claim, features, thresholds)


def verify_trace_commitments(
    trace: ParsedCoCTrace,
    features: TrajectoryFeatures,
    thresholds: VerifierThresholds | None = None,
) -> list[CommitmentVerdict]:
    """Verify every commitment in one parsed trace against the feature row
    of the SAME rollout, in claim order.

    Raises ValueError on a scene_id/rollout_id mismatch instead of
    verifying anyway: a trace checked against another rollout's kinematics
    would produce well-formed, plausible-looking verdicts that are pure
    noise — the one failure mode a reward pipeline must never let through
    silently. IDs that are None on the trace (parse_coc_trace allows that
    for ad-hoc strings) skip the check, since there is nothing to compare."""
    if trace.scene_id is not None and trace.scene_id != features.scene_id:
        raise ValueError(
            f"trace scene_id {trace.scene_id!r} != features scene_id {features.scene_id!r}"
        )
    if trace.rollout_id is not None and trace.rollout_id != features.rollout_id:
        raise ValueError(
            f"trace rollout_id {trace.rollout_id!r} != features rollout_id {features.rollout_id!r}"
        )
    return [verify_commitment(c, features, thresholds) for c in trace.commitments]
