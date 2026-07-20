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
import re
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


# ---------------------------------------------------------------------------
# Lexicons. Every entry below was chosen by grepping coc_text frequency
# across pref_pairs/results/scene_reasoning/*.md (2000+ unique CoC strings)
# rather than guessed -- see the counts noted inline for the higher-value
# entries. This is deliberately NOT exhaustive: rare entities/states that
# didn't clear a "worth a lexicon entry" bar (e.g. "animal", "debris",
# "pothole", one-off garbled model output) fall through to
# ParsedCoCTrace.unparsed_spans instead of a silent misclassification.
# ---------------------------------------------------------------------------

# The corpus mixes ASCII '-' with the unicode non-breaking hyphen U+2011 and
# en-dash U+2013 for the *same* word ("work-zone" vs "work‑zone", 80+
# instances of U+2011 alone) -- normalizing once up front means every regex
# below only has to spell one hyphen, not double every hyphenated
# alternative. Single-char-for-single-char, so span offsets computed
# against the normalized text stay valid against the (also normalized)
# `raw_text` stored on ParsedCoCTrace -- see parse_coc_trace.
_PUNCTUATION_NORMALIZATION = str.maketrans(
    {"‑": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}
)


def _normalize_punctuation(text: str) -> str:
    return text.translate(_PUNCTUATION_NORMALIZATION)


# (maneuver key, axis, speed_profile, compiled regex). Order matters only in
# that later entries never get a chance to claim characters a *lower-index*
# entry already matched (see _extract_commitments' consumed-span tracking);
# it does not encode priority among independent matches -- a compound
# commitment ("Accelerate and turn right...") is meant to yield one claim
# per verb, not just the first.
#
# "turn" excludes a leading "right-"/"left-" via negative lookbehind because
# the corpus also uses "right-turn"/"left-turn" as an *adjective* on nouns
# like "right-turn traffic light" / "right-turn lane" -- that is a
# perceptual-entity description, not an ego commitment, and would otherwise
# be misread as one (checked: 8/2031 unique corpus lines have this adjective
# form; all 8 are on the cause side of a connective and so would be excluded
# from commitment-clause scanning anyway, but the lookbehind is cheap
# insurance against a commitment-clause instance we haven't seen).
MANEUVER_PATTERNS: list[tuple[str, ManeuverAxis, str | None, re.Pattern[str]]] = [
    ("lane_change", ManeuverAxis.LATERAL, None, re.compile(r"\bchang(?:e|es|ed|ing)\b", re.I)),
    ("keep_lane", ManeuverAxis.LATERAL, None, re.compile(r"\bkeep(?:s|ing)?\s+lane\b", re.I)),
    ("nudge", ManeuverAxis.LATERAL, None, re.compile(r"\bnudg(?:e|es|ed|ing)\b", re.I)),
    ("merge", ManeuverAxis.LATERAL, None, re.compile(r"\bmerg(?:e|es|ed|ing)\b", re.I)),
    (
        "turn",
        ManeuverAxis.LATERAL,
        None,
        re.compile(r"(?<!right-)(?<!left-)\bturn(?:s|ed|ing)?\b", re.I),
    ),
    ("enter", ManeuverAxis.LATERAL, None, re.compile(r"\benter(?:s|ing|ed)?\b", re.I)),
    ("exit", ManeuverAxis.LATERAL, None, re.compile(r"\bexit(?:s|ing|ed)?\b", re.I)),
    (
        "adapt_speed",
        ManeuverAxis.LONGITUDINAL,
        "adapt",
        re.compile(r"\b(?:adapt|adjust)(?:s|ing|ed)?\s+speed\b", re.I),
    ),
    (
        "accelerate",
        ManeuverAxis.LONGITUDINAL,
        "accelerate",
        re.compile(r"\baccelerat\w*\b", re.I),
    ),
    (
        "decelerate",
        ManeuverAxis.LONGITUDINAL,
        "decelerate",
        re.compile(r"\bdecelerat\w*\b|\bslow(?:s|ing)?\s+down\b|\bbrak\w*\b", re.I),
    ),
    (
        # "Keep distance"/"maintain a safe gap"/"maintain following distance" --
        # NOT "maintain lane" (that's keep_lane, LATERAL, matched above and
        # so already consumed before this entry runs).
        "keep_distance",
        ManeuverAxis.LONGITUDINAL,
        "maintain",
        re.compile(
            r"\bkeep(?:s|ing)?\s+distance\b"
            r"|\bmaintain(?:s|ing)?\s+(?:a\s+|the\s+)?(?:safe\s+)?"
            r"(?:distance|gap|following\s+distance|progress)\b",
            re.I,
        ),
    ),
    (
        "create_gap",
        ManeuverAxis.LONGITUDINAL,
        None,
        re.compile(r"\bcreat(?:e|es|ing)\s+(?:a\s+|an\s+)?(?:usable\s+)?gap\b", re.I),
    ),
    ("stop", ManeuverAxis.LONGITUDINAL, "decelerate", re.compile(r"\bstop(?:s|ping)?\b", re.I)),
    ("yield", ManeuverAxis.LONGITUDINAL, "decelerate", re.compile(r"\byield(?:s|ing)?\b", re.I)),
    ("wait", ManeuverAxis.LONGITUDINAL, "decelerate", re.compile(r"\bwait(?:s|ing)?\b", re.I)),
    ("proceed", ManeuverAxis.LONGITUDINAL, None, re.compile(r"\bproceed(?:s|ing)?\b", re.I)),
]

# Searched in a window immediately after (never before -- see
# _extract_commitments) each maneuver match to resolve CommitmentClaim.direction.
DIRECTION_PATTERN = re.compile(r"\b(left|right)\b", re.I)
# How far past a maneuver match's end to look for a direction word before
# giving up (chosen from corpus spot-checks like "change to the left lane",
# "merge back right after clearing them" -- direction is close by, this is
# generous headroom, not a precisely fit bound).
DIRECTION_WINDOW_CHARS = 40

# (entity key, compiled regex). Longer/more specific alternatives are listed
# first within each pattern so a phrase like "stopped emergency vehicle"
# prefers the more specific match its regex is written to prefer (Python's
# re picks the first alternative that matches at the leftmost position, so
# ordering within a single pattern's alternation *does* matter, unlike the
# ordering of ENTITY_PATTERNS entries themselves).
ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "emergency_vehicle",
        re.compile(r"\bemergency\s+vehicles?\b", re.I),
    ),
    (
        "construction_cones",
        re.compile(r"\b(?:construction|traffic|roadwork|work-zone)\s+cones?\b|\bcones?\b", re.I),
    ),
    (
        "barricades",
        re.compile(
            r"\b(?:construction\s+)?barricades?\b"
            r"|\b(?:construction\s+)?barriers?\b"
            r"|\b(?:concrete\s+)?barriers?\b"
            r"|\bconstruction\s+barrels?\b"
            r"|\bbarrels?\b",
            re.I,
        ),
    ),
    (
        "cutin_vehicle",
        re.compile(r"\bcut-in\s+(?:vehicle|car|truck|van|pickup)\b", re.I),
    ),
    (
        "lead_vehicle",
        re.compile(
            r"\blead\s+(?:vehicle|car|truck|bus|motorcycle|scooter|van|pickup)\b", re.I
        ),
    ),
    (
        "stopped_vehicle",
        re.compile(
            r"\bstopped\s+(?:police\s+|parked\s+)?(?:car|cars|vehicle|vehicles|"
            r"truck|trucks|van|bus|suv)\b"
            r"|\bstalled\s+(?:car|vehicle|truck)\b",
            re.I,
        ),
    ),
    ("pedestrian", re.compile(r"\bpedestrians?\b|\bperson\s+standing\b", re.I)),
    (
        "cyclist",
        re.compile(
            r"\bscooters?\b|\bscooter\s+riders?\b|\bcyclists?\b|\bbicyclists?\b|\bbikers?\b",
            re.I,
        ),
    ),
    ("crosswalk", re.compile(r"\bcrosswalks?\b", re.I)),
    (
        "cross_traffic",
        re.compile(r"\bcross[- ]traffic\b|\bcrossing\s+(?:traffic|vehicle)\b", re.I),
    ),
    (
        "oncoming_traffic",
        re.compile(r"\boncoming\b|\bopposing\s+(?:lane|traffic|van)\b", re.I),
    ),
    ("signal", re.compile(r"\btraffic\s+light\b|\bsignals?\b", re.I)),
    (
        "work_zone",
        re.compile(r"\bwork-zone\b|\bwork\s+zone\b|\bconstruction\s+zone\b|\broadwork\b", re.I),
    ),
    ("roundabout", re.compile(r"\broundabouts?\b", re.I)),
    ("gate", re.compile(r"\bgates?\b|\bdriveways?\b", re.I)),
    ("workers", re.compile(r"\bworkers?\b", re.I)),
    (
        "ramp_or_freeway",
        re.compile(r"\bfreeways?\b|\bon-ramps?\b|\boff-ramps?\b|\bramps?\b", re.I),
    ),
    ("curve", re.compile(r"\bcurves?\b|\bcurvature\b|\bbends?\b", re.I)),
    (
        "shoulder_or_median",
        re.compile(r"\bshoulders?\b|\bmedians?\b|\bislands?\b|\bguardrails?\b", re.I),
    ),
    (
        "weather_or_surface",
        re.compile(
            r"\bsnowy\b|\bsnow\b|\bwet\s+surface\b|\bwet\s+roadway\b|\bsun\s+glare\b|\bglare\b"
            r"|\blow-friction\b|\bicy\s+surface\b",
            re.I,
        ),
    ),
    ("intersection", re.compile(r"\bintersections?\b", re.I)),
    (
        "speed_hump",
        re.compile(r"\bspeed\s+humps?\b|\bspeed\s+bumps?\b", re.I),
    ),
    (
        "speed_limit_sign",
        re.compile(r"\bspeed\s+limit(?:\s+sign)?\b", re.I),
    ),
    # Generic vehicle mention with no lead/stopped/cut-in qualifier (a plain
    # "car"/"vehicle"/"truck"/"SUV"/"construction equipment" the reasoning
    # names directly, e.g. "a car pulling out from the right side"). Listed
    # after the more specific vehicle entities above so those get first
    # claim on any span this would otherwise also match.
    (
        "vehicle_generic",
        re.compile(
            r"\b(?:parked\s+|police\s+|construction\s+)?(?:vehicles?|cars?|trucks?|suvs?|"
            r"equipment)\b",
            re.I,
        ),
    ),
    ("lane", re.compile(r"\blanes?\b", re.I)),
]

# (state key, compiled regex). Paired to the *nearest* entity match by
# _pair_entities_with_states, not necessarily every predicate that could
# describe that entity -- see that function's docstring for why picking one
# is good enough here.
STATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("blocking", re.compile(r"\bblock(?:s|ing|ed)?\b", re.I)),
    (
        "narrowing",
        re.compile(r"\bnarrow(?:s|ed|ing)?\b|\bconstrict(?:s|ed|ing)?\b|\btaper(?:s|ing)?\b", re.I),
    ),
    (
        "clearing",
        re.compile(r"\bclear(?:s|ed|ing|ance)?\b|\bhas\s+cleared\b", re.I),
    ),
    ("stopped", re.compile(r"\bstopped\b|\bstalled\b", re.I)),
    ("crossing", re.compile(r"\bcrossing\b", re.I)),
    ("ahead", re.compile(r"\bahead\b", re.I)),
    ("closed", re.compile(r"\bclosed\b", re.I)),
    ("open", re.compile(r"\bopen\b", re.I)),
    ("green", re.compile(r"\bgreen\b", re.I)),
    ("red", re.compile(r"\bred\b", re.I)),
    ("encroaching", re.compile(r"\bencroaching\b", re.I)),
    (
        "yield_controlled",
        re.compile(r"\byield[- ]controlled\b|\byield\s+sign\b", re.I),
    ),
    ("approaching", re.compile(r"\bapproaching\b", re.I)),
    ("pulling_away", re.compile(r"\bpulling\s+away\b", re.I)),
    ("pulling_out", re.compile(r"\bpulling\s+out\b", re.I)),
    ("permits_movement", re.compile(r"\bpermits?\s+movement\b", re.I)),
    ("nearby", re.compile(r"\bnearby\b|\bpresent\b", re.I)),
    # "merging"/"exiting" describe ANOTHER agent's action (e.g. "a vehicle
    # merging from the right"), unrelated to MANEUVER_PATTERNS' "merge"/
    # "exit" entries -- those only ever scan commitment_text (the ego's own
    # stated action), this only ever scans a cause clause (see
    # _extract_perceptual_claims), so there is no risk of the two lexicons
    # cross-matching the same span.
    ("merging", re.compile(r"\bmerging\b", re.I)),
    ("exiting", re.compile(r"\bexiting\b", re.I)),
]

# Backward-explanatory connectives only ("X because Y" -- Y is the cause of
# X): tried in this tier first, leftmost match in the beat wins. "requiring"
# was deliberately left out despite being common enough to lexicon
# (~10 corpus hits) -- it runs FORWARD ("crossing pedestrians, requiring a
# speed reduction": cause is BEFORE "requiring", effect after), the opposite
# of every other connective here, and folding in a reversed-direction case
# would complicate _split_beat for a minority pattern rather than clarify it.
STRONG_CONNECTIVES = re.compile(
    r"\b(?:because\s+of|because|due\s+to|since|as)\b"
    r"(?!\s+(?:needed|well|confirmed|alongside|a\s+result))",
    re.I,
)
# Fallback tier, only consulted when no STRONG_CONNECTIVES match exists
# anywhere in the beat (see _split_beat) -- both are common in non-causal
# roles elsewhere in English ("wait FOR the van", "merge AFTER the ramp"
# read causally here only because this corpus's beats almost always state
# a reason), so they're deliberately not tried first.
WEAK_CONNECTIVES = re.compile(r"\b(?:after|for)\b", re.I)

# Splits a CoC string into sequential "beats" -- ';', an explicit "then", or
# a sentence-ending period followed by more text (a handful of corpus
# entries use one, most don't use periods at all). Each beat is expected to
# carry at most one causal connective (see _split_beat).
BEAT_DELIMITER = re.compile(r";|,?\s+then\b|\.\s+(?=\S)", re.I)


def _split_beats(text: str) -> list[tuple[int, int]]:
    """Split `text` into sequential beat spans on BEAT_DELIMITER. Adjacent
    delimiters (e.g. "...lane; then merge...", where ';' and ' then' both
    match) produce a zero-length beat between them -- dropped here rather
    than in the caller, since an empty/whitespace-only beat can never carry
    a claim regardless of what calls this."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in BEAT_DELIMITER.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_beat(
    text: str, beat_span: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int] | None, str | None]:
    """Within one beat, split on the first causal connective into
    (commitment_span, cause_span, connective_text) -- all as offsets into
    the ORIGINAL `text`, not into the beat substring, so callers never have
    to re-add an offset. STRONG_CONNECTIVES is tried before WEAK_CONNECTIVES
    (searched only if no strong match exists ANYWHERE in the beat, not just
    earlier in it -- e.g. "Create a usable gap for a left lane change
    because cones..." must split on "because", not the earlier "for", so
    trying WEAK_CONNECTIVES first and taking a leftmost-overall match would
    be wrong here). Returns (beat_span, None, None) when the beat states no
    reason at all -- a real, fairly common outcome (e.g. "Keep lane"), not
    a parse failure.
    """
    start, end = beat_span
    beat = text[start:end]
    match = STRONG_CONNECTIVES.search(beat) or WEAK_CONNECTIVES.search(beat)
    if match is None:
        return beat_span, None, None
    commitment_span = (start, start + match.start())
    cause_span = (start + match.end(), end)
    return commitment_span, cause_span, match.group()


def _extract_commitments(text: str, span: tuple[int, int]) -> list[CommitmentClaim]:
    """Find every maneuver mention inside `span`, which must be a
    commitment clause (the text before a beat's causal connective, or a
    whole beat with no connective) -- never a cause clause. That scoping is
    what keeps MANEUVER_PATTERNS' "turn" entry from misfiring on an
    adjectival "right-turn traffic light"/"right-turn lane" mention on the
    cause side (see that pattern's comment); it also means a maneuver word
    used to describe another agent in a cause clause (e.g. "vehicle
    merging from the right") is never mistaken for ego's own commitment.

    A compound commitment ("Accelerate and turn right...") yields one
    CommitmentClaim per verb. MANEUVER_PATTERNS entries are tried in list
    order and a later entry may not re-claim characters an earlier one
    already matched -- in practice the patterns' vocabularies barely
    overlap, but this keeps that an enforced invariant rather than an
    accident of which words happen to differ.
    """
    start, end = span
    region = text[start:end]
    claimed: list[tuple[int, int]] = []
    raw_matches: list[tuple[int, int, str, ManeuverAxis, str | None]] = []
    for name, axis, speed_profile, pattern in MANEUVER_PATTERNS:
        for m in pattern.finditer(region):
            m_start, m_end = start + m.start(), start + m.end()
            if any(m_start < c_end and c_start < m_end for c_start, c_end in claimed):
                continue
            claimed.append((m_start, m_end))
            raw_matches.append((m_start, m_end, name, axis, speed_profile))
    raw_matches.sort(key=lambda t: t[0])

    claims: list[CommitmentClaim] = []
    for i, (m_start, m_end, name, axis, speed_profile) in enumerate(raw_matches):
        # Direction is searched only AFTER this maneuver's own match, and
        # only up to wherever the NEXT maneuver match starts -- so in
        # "change lanes to the right and enter the freeway on-ramp",
        # "right" is attributed to "change lanes" (the nearer verb) and
        # "enter" correctly gets no direction of its own, rather than both
        # verbs claiming the same single direction word.
        window_end = min(m_end + DIRECTION_WINDOW_CHARS, end)
        if i + 1 < len(raw_matches):
            window_end = min(window_end, raw_matches[i + 1][0])
        direction_match = DIRECTION_PATTERN.search(text[m_end:window_end])
        direction = direction_match.group(1).lower() if direction_match else None
        claims.append(
            CommitmentClaim(
                text=text[m_start:m_end],
                maneuver=name,
                axis=axis,
                speed_profile=speed_profile,
                direction=direction,
                span=(m_start, m_end),
            )
        )
    return claims


def _nearest_state_key(
    entity_span: tuple[int, int], state_matches: list[tuple[str, int, int]]
) -> str | None:
    """The state key whose match is closest to `entity_span` (0 if they
    touch or overlap), or None if `state_matches` is empty. Picks exactly
    one state per entity even when several plausibly apply (e.g. "the
    closed gate blocking the driveway" -- "closed" and "blocking" both
    describe "gate") -- good enough for a first pass at this task's
    granularity (claim verification against kinematics doesn't need every
    predicate that could apply, just a representative one), but a real
    simplification worth knowing about before trusting `state` as
    exhaustive.
    """
    best_distance: int | None = None
    best_key: str | None = None
    for key, s_start, s_end in state_matches:
        if s_end <= entity_span[0]:
            distance = entity_span[0] - s_end
        elif s_start >= entity_span[1]:
            distance = s_start - entity_span[1]
        else:
            distance = 0  # overlapping/adjacent
        if best_distance is None or distance < best_distance:
            best_distance, best_key = distance, key
    return best_key


def _extract_perceptual_claims(text: str, span: tuple[int, int]) -> list[PerceptualClaim]:
    """Find every entity mention inside `span` (typically a cause clause,
    but callers may pass any span -- see parse_coc_trace, which also scans
    commitment clauses so an entity named there isn't missed just because
    no causal connective happened to follow it) and pair each with the
    nearest state predicate via _nearest_state_key. ENTITY_PATTERNS entries
    are tried in list order with the same no-reclaiming-a-matched-span rule
    as _extract_commitments, so e.g. "stopped police car" is captured whole
    by the stopped_vehicle entry rather than vehicle_generic also matching
    just "car" inside it.
    """
    start, end = span
    region = text[start:end]

    claimed: list[tuple[int, int]] = []
    entity_matches: list[tuple[int, int, str]] = []
    for key, pattern in ENTITY_PATTERNS:
        for m in pattern.finditer(region):
            m_start, m_end = start + m.start(), start + m.end()
            if any(m_start < c_end and c_start < m_end for c_start, c_end in claimed):
                continue
            claimed.append((m_start, m_end))
            entity_matches.append((m_start, m_end, key))
    entity_matches.sort(key=lambda t: t[0])

    state_matches = _find_state_matches(text, span)

    return [
        PerceptualClaim(
            text=text[m_start:m_end],
            entity=key,
            state=_nearest_state_key((m_start, m_end), state_matches),
            span=(m_start, m_end),
        )
        for m_start, m_end, key in entity_matches
    ]


def _find_state_matches(text: str, span: tuple[int, int]) -> list[tuple[str, int, int]]:
    """Every STATE_PATTERNS match inside `span`, as (key, start, end).
    Factored out of _extract_perceptual_claims so parse_coc_trace can also
    use it directly: a state word that matched but wasn't the *nearest* one
    to any entity (so didn't end up on a PerceptualClaim) still shouldn't
    be reported as unparsed -- the parser recognized it, it just wasn't the
    winning pairing. See parse_coc_trace's unparsed-span bookkeeping.
    """
    start, end = span
    region = text[start:end]
    return [
        (key, start + m.start(), start + m.end())
        for key, pattern in STATE_PATTERNS
        for m in pattern.finditer(region)
    ]


# Function words/connectives that, on their own, don't represent a missed
# claim -- a gap consisting only of these (plus whitespace/punctuation) is
# not reported in ParsedCoCTrace.unparsed_spans. Deliberately narrow: this
# is a denylist of grammatical glue, not an attempt to guess what content
# words are "unimportant" (those still get surfaced).
_FILLER_TOKEN = re.compile(
    r"^(?:and|the|a|an|our|to|with|of|in|on|at|it|is|are|be|being|for|after|"
    r"because|due|since|as|then|that|this|we|us|there|they|them)$",
    re.I,
)


def _is_boilerplate_gap(gap_text: str) -> bool:
    """True if every word in `gap_text` is filler -- i.e. this gap is not
    worth surfacing as an unparsed span. A gap with no words at all (pure
    whitespace/punctuation, e.g. the ", " left between two adjacent claim
    spans) counts as boilerplate too."""
    words = re.findall(r"[A-Za-z']+", gap_text)
    return all(_FILLER_TOKEN.match(w) for w in words)


def _compute_unparsed_spans(
    text: str, covered_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Complement of `covered_spans` within `text`, merging overlaps first,
    then dropping any gap _is_boilerplate_gap calls filler. What's left is
    the parser's honest account of what it did NOT attribute to any claim
    (see ParsedCoCTrace's docstring) -- e.g. a rare entity/state outside
    the lexicons, or a sub-clause like "to create space" that isn't itself
    a maneuver this module recognizes.
    """
    merged: list[tuple[int, int]] = []
    for s, e in sorted(covered_spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gaps: list[tuple[int, int]] = []
    pos = 0
    for s, e in merged:
        if s > pos:
            gaps.append((pos, s))
        pos = max(pos, e)
    if pos < len(text):
        gaps.append((pos, len(text)))

    return [(s, e) for s, e in gaps if not _is_boilerplate_gap(text[s:e])]


def parse_coc_trace(
    text: str, *, scene_id: str | None = None, rollout_id: int | None = None
) -> ParsedCoCTrace:
    """Parse one rollout's CoC reasoning string into typed claims.

    Each beat (see _split_beats) is processed independently: perceptual
    claims are extracted from the WHOLE beat (not just its cause clause) --
    e.g. in "Adapt speed for the roundabout since a yield-controlled entry
    is ahead", the entity "roundabout" sits in the commitment-side text
    ("for the roundabout") while its state "yield_controlled" sits in the
    cause-side text (after "since"); only scanning the whole beat lets
    _nearest_state_key pair them (see this module's perceptual-extraction
    commit message for why cause-only scanning gets this wrong). A
    CausalClaim's `cause` list is then just the subset of that beat's
    perceptual claims whose span falls after the connective -- not a
    second, separate extraction pass.
    """
    normalized = _normalize_punctuation(text)
    commitments: list[CommitmentClaim] = []
    perceptual: list[PerceptualClaim] = []
    causal: list[CausalClaim] = []
    state_spans: list[tuple[int, int]] = []  # covers matched-but-unpaired state words too

    for beat_span in _split_beats(normalized):
        commitment_span, cause_span, connective = _split_beat(normalized, beat_span)
        beat_commitments = _extract_commitments(normalized, commitment_span)
        beat_perceptual = _extract_perceptual_claims(normalized, beat_span)
        commitments.extend(beat_commitments)
        perceptual.extend(beat_perceptual)
        state_spans.extend((s, e) for _key, s, e in _find_state_matches(normalized, beat_span))

        if connective is not None:
            cause_start, cause_end = cause_span
            cause_claims = [
                p for p in beat_perceptual if cause_start <= p.span[0] and p.span[1] <= cause_end
            ]
            causal.append(
                CausalClaim(
                    text=normalized[beat_span[0] : beat_span[1]],
                    connective=connective,
                    effects=beat_commitments,
                    cause=cause_claims,
                    span=beat_span,
                )
            )

    covered_spans = [c.span for c in commitments] + [p.span for p in perceptual] + state_spans
    unparsed_spans = _compute_unparsed_spans(normalized, covered_spans)

    return ParsedCoCTrace(
        raw_text=normalized,
        scene_id=scene_id,
        rollout_id=rollout_id,
        commitments=commitments,
        perceptual=perceptual,
        causal=causal,
        unparsed_spans=unparsed_spans,
    )
