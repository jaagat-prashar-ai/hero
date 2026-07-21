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

from code_as_a_reward.coc_claim_parser import CommitmentClaim


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
