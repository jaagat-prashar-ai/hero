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
