# SPDX-License-Identifier: Apache-2.0
"""
perceptual_verifier.py — checks PerceptualClaims ("I see X", parsed by
coc_claim_parser.py) against the scene's obstacle.offline actor tracks
(obstacle_tracks.py). Together with commitment_verifier.py this completes
the two ground-truth-checkable claim types; CausalClaims compose the two
(see the reward-aggregation module).

A perceptual claim has two separately-decidable parts, and this module
keeps their verdicts separate on the result:

  * EXISTENCE — "was an actor of this kind near ego during the rollout's
    window at all?" Decidable exactly when the parser's entity key maps
    onto the dataset's 6-class label vocabulary (automobile / person /
    rider / other_vehicle / trailer / protruding_object). Phase 0's
    taxonomy: ~62% of corpus claims are of mappable kinds; ~27% (cones,
    signals, work zones, road furniture...) have NO ground truth in the
    dataset and must ABSTAIN — never FAIL, since penalizing the model for
    OUR missing labels is the exact failure mode the taxonomy warned about.
  * STATE — the predicate attached to the entity ("crossing", "ahead",
    "approaching"). Each state key is decidable, undecidable, or
    conditionally decidable given rig-frame (ego-relative) geometry — see
    STATE_CHECKS. Notably "stopped" is UNDECIDABLE here even though it
    sounds easy: track positions are ego-relative snapshots, so a parked
    car sweeps past at ego's own speed (obstacle_tracks.apparent_speed_mps
    documents this trap); deciding it needs egomotion differencing, which
    is future work, not a threshold away.

The combined `verdict` is conservative: FAIL if either part fails, PASS
only if existence passes and the state (if any was claimed) passes,
ABSTAIN otherwise. The parts stay on the verdict so reward aggregation
can weight "entity real, state unverifiable" differently from "nothing
verifiable at all" without re-running anything.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from code_as_a_reward.coc_claim_parser import ENTITY_PATTERNS, PerceptualClaim
from code_as_a_reward.commitment_verifier import Verdict
from code_as_a_reward.obstacle_tracks import OBSTACLE_LABEL_CLASSES, SceneObstacles


def split_scene_id(scene_id: str) -> tuple[str, int]:
    """scene_id -> (clip_id, t0_us). Inverse of rollout_harvester.py's
    f"{clip_id}_{t0_us}" naming, which every downstream artifact
    (feature rows, reasoning reports, parsed traces) carries."""
    clip_id, _, t0_str = scene_id.rpartition("_")
    if not clip_id or not t0_str.isdigit():
        raise ValueError(f"scene_id {scene_id!r} is not of the form <clip_id>_<t0_us>")
    return clip_id, int(t0_str)


# Parser entity key -> the dataset label classes that count as that entity
# existing. None means the dataset has NO ground truth for this entity kind
# (Phase 0's unverifiable ~27%: cones, signals, zones, road geometry,
# weather) -> existence ABSTAIN. Choices that lose information on purpose,
# documented so they're revisited deliberately rather than rediscovered:
#   * workers -> person: the dataset can confirm PEOPLE were present but
#     not that they were workers; existence-of-person is the checkable
#     part of the claim, and the lost "worker" attribute is exactly the
#     kind of thing the unparsed/abstain bookkeeping shouldn't hide — so
#     it's noted in the verdict evidence.
#   * emergency_vehicle -> None, NOT automobile: unlike workers, the
#     qualifying attribute is the entire claim (an ordinary car being
#     present says nothing about an emergency vehicle), so mapping it to
#     automobile would manufacture false PASSes.
#   * barricades -> None for now: the rare 'protruding_object' class MIGHT
#     cover some barricades/barrels, but one fixture clip isn't evidence
#     enough to hang verdicts on; candidate for a later data audit.
ENTITY_TO_CLASSES: dict[str, frozenset[str] | None] = {
    "pedestrian": frozenset({"person"}),
    "cyclist": frozenset({"rider"}),
    "workers": frozenset({"person"}),
    "lead_vehicle": frozenset({"automobile", "other_vehicle", "trailer"}),
    "stopped_vehicle": frozenset({"automobile", "other_vehicle", "trailer"}),
    "cutin_vehicle": frozenset({"automobile", "other_vehicle", "trailer"}),
    "vehicle_generic": frozenset({"automobile", "other_vehicle", "trailer"}),
    "cross_traffic": frozenset({"automobile", "other_vehicle", "trailer", "rider"}),
    "oncoming_traffic": frozenset({"automobile", "other_vehicle", "trailer", "rider"}),
    "emergency_vehicle": None,
    "construction_cones": None,
    "barricades": None,
    "crosswalk": None,
    "signal": None,
    "work_zone": None,
    "roundabout": None,
    "gate": None,
    "ramp_or_freeway": None,
    "curve": None,
    "shoulder_or_median": None,
    "weather_or_surface": None,
    "intersection": None,
    "speed_hump": None,
    "speed_limit_sign": None,
    "lane": None,
}

# Same import-time sync enforcement as commitment_verifier's dispatch
# table: every parser entity key must have an explicit mapping decision
# (even if that decision is None/"unverifiable"), and every mapped class
# must be a real dataset class.
_PARSER_ENTITIES = {name for name, _pattern in ENTITY_PATTERNS}
assert ENTITY_TO_CLASSES.keys() == _PARSER_ENTITIES, (
    "perceptual_verifier's entity mapping is out of sync with "
    f"coc_claim_parser.ENTITY_PATTERNS: missing={_PARSER_ENTITIES - ENTITY_TO_CLASSES.keys()}, "
    f"stale={ENTITY_TO_CLASSES.keys() - _PARSER_ENTITIES}"
)
assert all(
    classes <= OBSTACLE_LABEL_CLASSES for classes in ENTITY_TO_CLASSES.values() if classes
), "ENTITY_TO_CLASSES maps to a class name that is not in the dataset vocabulary"


@dataclasses.dataclass
class PerceptualThresholds:
    """Tunable knobs for perceptual verification. All initial judgment
    calls (there is no classifier counterpart to inherit from, unlike
    commitment_verifier's tier-1), to be recalibrated against hand-labeled
    verdicts alongside the commitment thresholds."""

    # An actor only makes "I see X" true if it was actually near ego —
    # a person 200m down a side street shouldn't verify "pedestrian
    # crossing". 50m ~ the distance over which reasoning-relevant actors
    # appear in the corpus's scene descriptions.
    presence_max_distance_m: float = 50.0
    # Fewer than this many detections in the window is treated as the
    # actor not (reliably) present — single-frame autolabel blips are the
    # dominant false-positive shape (see actors_present).
    min_samples: int = 2
    # "blocking"-family states: actor's mean position must be ahead of ego
    # (x > 0) inside a corridor of this half-width, within this range.
    # Half-width ~ half a lane; range chosen so a car two blocks ahead
    # doesn't count as blocking.
    corridor_half_width_m: float = 1.75
    corridor_max_ahead_m: float = 40.0
    # "approaching"/"pulling_away": planar ego-distance must change by at
    # least this much between the window's first and last sample —
    # smaller drifts are autolabel jitter, not relative motion.
    approach_min_delta_m: float = 2.0


@dataclasses.dataclass
class PerceptualVerdict:
    """One perceptual claim's verification result. `existence` and `state`
    are the separately-decidable parts (state is None when the claim
    carried no state predicate); `verdict` is their conservative
    combination (see module docstring). Evidence/rule/reason mirror
    CommitmentVerdict's audit-trail contract."""

    claim: PerceptualClaim
    verdict: Verdict
    existence: Verdict
    state: Verdict | None
    rule: str
    evidence: dict[str, Any]
    reason: str

    def to_row_dict(self) -> dict[str, Any]:
        return {
            "claim_text": self.claim.text,
            "claim_entity": self.claim.entity,
            "claim_state": self.claim.state,
            "verdict": self.verdict.value,
            "existence": self.existence.value,
            "state": self.state.value if self.state is not None else None,
            "rule": self.rule,
            "evidence": self.evidence,
            "reason": self.reason,
        }
