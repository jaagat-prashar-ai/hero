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

import numpy as np

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


# Entities whose parser key bundles an ATTRIBUTE the dataset's class
# vocabulary cannot confirm (there is no "stopped"/"lead"/"cut-in" label,
# only automobile/person/...). Verification is deliberately ASYMMETRIC for
# these: existence can still FAIL — if no vehicle of any kind was near ego,
# "a stopped vehicle" is false regardless of the attribute, and catching
# that hallucination is this project's core purpose — but a would-be PASS
# is capped at ABSTAIN, because certifying the claim would mean certifying
# an attribute nobody checked. (workers is here too: people being present
# doesn't make them workers.) vehicle_generic/pedestrian/cyclist are NOT
# here: their parser keys assert nothing beyond the base class.
ATTRIBUTE_UNVERIFIED_ENTITIES: dict[str, str] = {
    "stopped_vehicle": "stopped",
    "lead_vehicle": "lead",
    "cutin_vehicle": "cut-in",
    "cross_traffic": "crossing ego's path",
    "oncoming_traffic": "oncoming",
    "workers": "worker",
}


# ---------------------------------------------------------------------------
# State checks. Each maps a parser state key to a predicate over the
# WINDOWED tracks of the claim's mapped classes; a state passes if AT LEAST
# ONE such actor satisfies it ("a pedestrian is crossing" needs one
# crossing pedestrian, not all of them). States not in STATE_CHECKS are
# undecidable from rig-frame geometry (signal colors, lane-relative
# notions, world-frame motion like "stopped" — see module docstring) and
# ABSTAIN. All geometry is planar (x fwd, y left), consistent with
# obstacle_tracks' helpers.
# ---------------------------------------------------------------------------


def _check_ahead(tracks, thresholds):
    xs = [t.mean_bearing()[0] for t in tracks]
    ok = any(x > 0 for x in xs)
    return ok, {"mean_forward_m": xs}, "an actor's mean position is ahead of ego"


def _check_crossing(tracks, thresholds):
    # Crossing ego's path = the actor's lateral (y) coordinate changes sign
    # across the window: it was on one side of ego's heading line and ended
    # on the other. Ego-relative jitter can't produce a sign flip for any
    # actor that isn't actually near ego's path, so no extra threshold.
    flips = [bool(t.centers_m[0, 1] * t.centers_m[-1, 1] < 0) for t in tracks]
    return any(flips), {"lateral_sign_flip": flips}, "an actor's lateral offset changed sign"


def _distance_delta(track) -> float:
    first = float(np.linalg.norm(track.centers_m[0, :2]))
    last = float(np.linalg.norm(track.centers_m[-1, :2]))
    return last - first


def _check_approaching(tracks, thresholds):
    deltas = [_distance_delta(t) for t in tracks]
    ok = any(d <= -thresholds.approach_min_delta_m for d in deltas)
    return ok, {"distance_delta_m": deltas}, "an actor's ego-distance decreased over the window"


def _check_pulling_away(tracks, thresholds):
    deltas = [_distance_delta(t) for t in tracks]
    ok = any(d >= thresholds.approach_min_delta_m for d in deltas)
    return ok, {"distance_delta_m": deltas}, "an actor's ego-distance increased over the window"


def _check_in_corridor(tracks, thresholds):
    # "blocking"/"encroaching": mean position inside ego's forward corridor
    # (a straight-ahead approximation — without lane geometry, ego's
    # heading line is the only notion of "our path" available).
    bearings = [t.mean_bearing() for t in tracks]
    ok = any(
        0.0 < x <= thresholds.corridor_max_ahead_m and abs(y) <= thresholds.corridor_half_width_m
        for x, y in bearings
    )
    return ok, {"mean_bearing_m": bearings}, "an actor sits in ego's forward corridor"


def _check_nearby(tracks, thresholds):
    # actors_present already filtered by presence_max_distance_m, so any
    # surviving track IS nearby; the check exists so "nearby" reads as
    # decided-by-geometry in verdicts rather than falling into the
    # undecidable bucket.
    return bool(tracks), {"n_tracks": len(tracks)}, "an actor is within presence distance"


STATE_CHECKS = {
    "ahead": _check_ahead,
    "crossing": _check_crossing,
    "approaching": _check_approaching,
    "pulling_away": _check_pulling_away,
    "blocking": _check_in_corridor,
    "encroaching": _check_in_corridor,
    "nearby": _check_nearby,
}


def verify_perceptual(
    claim: PerceptualClaim,
    scene: SceneObstacles,
    t0_us: int,
    horizon_us: int,
    thresholds: PerceptualThresholds | None = None,
) -> PerceptualVerdict:
    """Verify one PerceptualClaim against one scene's obstacle tracks over
    the rollout window [t0_us, t0_us + horizon_us]. See the module
    docstring for the existence/state split and combination rule, and
    ATTRIBUTE_UNVERIFIED_ENTITIES for the asymmetric-PASS cap."""
    thresholds = thresholds or PerceptualThresholds()
    classes = ENTITY_TO_CLASSES[claim.entity]  # KeyError impossible: sync asserted at import

    if classes is None:
        state = Verdict.ABSTAIN if claim.state is not None else None
        return PerceptualVerdict(
            claim, Verdict.ABSTAIN, Verdict.ABSTAIN, state, "no_ground_truth",
            {"entity": claim.entity},
            f"dataset has no ground truth for '{claim.entity}' (Phase 0 unverifiable class)",
        )

    tracks = scene.actors_present(
        t0_us, t0_us + horizon_us,
        classes=set(classes),
        max_distance_m=thresholds.presence_max_distance_m,
        min_samples=thresholds.min_samples,
    )
    evidence: dict[str, Any] = {
        "mapped_classes": sorted(classes),
        "n_matching_tracks": len(tracks),
        "track_ids": [t.track_id for t in tracks],
        "window_us": [t0_us, t0_us + horizon_us],
    }
    existence = Verdict.PASS if tracks else Verdict.FAIL

    # State part. Checked only when actors exist: a state can't be
    # evaluated on actors that aren't there, and existence FAIL already
    # decides the combined verdict.
    state: Verdict | None = None
    if claim.state is not None:
        if existence is Verdict.FAIL:
            state = Verdict.ABSTAIN
        elif claim.state in STATE_CHECKS:
            ok, state_evidence, how = STATE_CHECKS[claim.state](tracks, thresholds)
            evidence.update(state_evidence)
            state = Verdict.PASS if ok else Verdict.FAIL
            evidence["state_check"] = how
        else:
            state = Verdict.ABSTAIN
            evidence["state_check"] = f"'{claim.state}' undecidable from rig-frame geometry"

    # Conservative combination (module docstring), then the asymmetric cap
    # for attribute-carrying entities.
    if existence is Verdict.FAIL or state is Verdict.FAIL:
        combined = Verdict.FAIL
    elif existence is Verdict.PASS and state in (None, Verdict.PASS):
        combined = Verdict.PASS
    else:
        combined = Verdict.ABSTAIN

    attribute = ATTRIBUTE_UNVERIFIED_ENTITIES.get(claim.entity)
    if attribute is not None and combined is Verdict.PASS:
        combined = Verdict.ABSTAIN
        evidence["attribute_unverified"] = attribute

    reason_bits = [f"existence={existence.value} ({len(tracks)} matching tracks)"]
    if state is not None:
        reason_bits.append(f"state[{claim.state}]={state.value}")
    if attribute is not None:
        reason_bits.append(f"'{attribute}' attribute has no ground truth")
    return PerceptualVerdict(
        claim, combined, existence, state, "actor_presence", evidence, "; ".join(reason_bits)
    )


def verify_trace_perceptual(
    trace,
    scene: SceneObstacles,
    horizon_us: int = 6_400_000,
    thresholds: PerceptualThresholds | None = None,
) -> list[PerceptualVerdict]:
    """Verify every perceptual claim in one parsed trace. t0 comes from the
    trace's own scene_id (rollout_harvester's <clip_id>_<t0_us> naming);
    the default horizon is the rollout future length every harvest in this
    repo uses (64 waypoints at 10Hz). Same mispairing stance as
    verify_trace_commitments: a clip mismatch raises rather than producing
    plausible nonsense verdicts."""
    if trace.scene_id is None:
        raise ValueError("trace has no scene_id; cannot locate its rollout window")
    clip_id, t0_us = split_scene_id(trace.scene_id)
    if clip_id != scene.clip_id:
        raise ValueError(f"trace clip {clip_id!r} != scene clip {scene.clip_id!r}")
    return [
        verify_perceptual(p, scene, t0_us, horizon_us, thresholds) for p in trace.perceptual
    ]
