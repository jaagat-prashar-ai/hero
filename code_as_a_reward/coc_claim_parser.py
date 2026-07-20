# SPDX-License-Identifier: Apache-2.0
"""
coc_claim_parser.py — rule-based parser that turns an Alpamayo rollout's
chain-of-causation (CoC) reasoning text into typed claims:

  * perceptual  — an entity the model says is present, optionally with a
                  state ("construction cones" + "blocking")
  * causal      — a maneuver explained by a cause ("nudge left" <- "due to"
                  <- "stopped car blocking the right side of our lane")
  * commitment  — a planned maneuver and/or speed profile ("nudge left",
                  "accelerate", "adapt speed", "stop")

This is a prerequisite for the pref-pairs faithfulness project's claim
verifier (see pref_pairs/trajectory_features.py's module docstring: "the
downstream claim verifier (checking CoC claims against kinematics)"): once
a CoC string is broken into a commitment claim ("nudge left") and the
perceptual claim that justifies it ("construction cones blocking our
lane"), the commitment side can be checked against a rollout's
TrajectoryFeatures (e.g. final_lateral_offset_m for "nudge left") and the
perceptual side can be checked against upstream scene-state (e.g.
obstacle.offline actor tracks) -- neither of which this module does itself.

Every lexicon and regex below was written against real coc_text samples in
pref_pairs/results/scene_reasoning/*.md (grep '^> ' across that directory
to see the corpus this was tuned on), not invented grammar. That corpus
ranges from single-clause strings ("Nudge left due to construction cones
blocking the right side of our lane") to long multi-maneuver run-ons
("Change lanes to the right and enter the freeway on-ramp because, after
yielding to the crossing scooter, the right-turn signal permits movement
and a safe gap exists to merge onto the ramp behind traffic, then
accelerate to match ramp speed..."). This is a rule-based best-effort
parser, not a full NL parser -- see `ParsedCoCTrace.unparsed_spans` for the
mechanism that surfaces what it failed to attribute to any claim, rather
than silently dropping it (same "no silent gaps" instinct as
trajectory_features.py's `accel_source` field: be honest about what was
actually derived vs. approximated).
"""

from __future__ import annotations

import dataclasses
from enum import Enum


class ManeuverAxis(str, Enum):
    """Which control axis a commitment claim's maneuver acts on. LATERAL
    covers lane changes/nudges/turns/merges; LONGITUDINAL covers
    speed-profile maneuvers (accelerate/decelerate/stop/yield/adapt speed).
    Kept as a plain (str, Enum) rather than a bare str field so downstream
    code can exhaustively branch on it, while still round-tripping through
    json.dumps as a plain string (no custom encoder needed)."""

    LATERAL = "lateral"
    LONGITUDINAL = "longitudinal"


@dataclasses.dataclass
class CommitmentClaim:
    """A planned maneuver and/or speed profile the model states it will
    execute, e.g. "nudge left", "adapt speed", "accelerate and turn right"
    (the last of which yields two CommitmentClaims, one per verb -- see
    module docstring's compound-commitment corpus examples)."""

    text: str  # verbatim source substring this claim was extracted from
    maneuver: str  # canonical key, e.g. "lane_change_left", "accelerate", "stop"
    axis: ManeuverAxis
    speed_profile: str | None  # "accelerate" | "decelerate" | "maintain" | "adapt" | None
    direction: str | None  # "left" | "right" | None
    span: tuple[int, int]  # char offsets into the raw CoC text


@dataclasses.dataclass
class PerceptualClaim:
    """An entity the model asserts is present, optionally with a state
    predicate, e.g. entity="construction_cones", state="blocking" from
    "construction cones blocking the right side of our lane". `state` is
    None when an entity is mentioned with no nearby state predicate this
    module's lexicon recognizes (see _pair_entities_with_states) -- that is
    a real "didn't find one" signal, not evidence none exists in the text."""

    text: str
    entity: str  # canonical key, e.g. "construction_cones", "stopped_vehicle"
    state: str | None  # canonical key, e.g. "blocking", "narrowing", "green"
    span: tuple[int, int]


@dataclasses.dataclass
class CausalClaim:
    """One "X because Y" link: one or more commitment claims (the effects,
    plural because a single cause clause can justify a compound commitment
    like "accelerate and turn right") explained by the perceptual claims
    found in the cause clause. `cause` can be an empty list -- that means a
    causal connective was found but no entity in it matched the perceptual
    lexicon, which is a real parse gap (see PARSE gap note in
    parse_coc_trace), not "no cause was stated"."""

    text: str  # verbatim beat this claim was split from (effect + connective + cause)
    connective: str  # the causal marker matched, e.g. "because", "due to", "since", "for"
    effects: list[CommitmentClaim]
    cause: list[PerceptualClaim]
    span: tuple[int, int]


@dataclasses.dataclass
class ParsedCoCTrace:
    """Everything extracted from one rollout's CoC text. `unparsed_spans`
    lists (start, end) substrings of `raw_text` that weren't attributed to
    any claim above and weren't filtered out as connective/filler
    boilerplate -- read it before trusting `commitments`/`perceptual`/
    `causal` to be a complete account of the text (they are a best-effort
    lower bound, per module docstring)."""

    raw_text: str
    scene_id: str | None
    rollout_id: int | None
    commitments: list[CommitmentClaim]
    perceptual: list[PerceptualClaim]
    causal: list[CausalClaim]
    unparsed_spans: list[tuple[int, int]]
