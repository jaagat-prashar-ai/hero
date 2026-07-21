# SPDX-License-Identifier: Apache-2.0
"""
coc_claim_parser_test.py — unit tests for coc_claim_parser.py, built from
real coc_text strings pulled out of pref_pairs/results/scene_reasoning/*.md
(not synthetic sentences), so a passing suite reflects the actual corpus
this parser has to handle rather than grammar invented for the test.
"""

from __future__ import annotations

from pathlib import Path

from code_as_a_reward.coc_claim_parser import ManeuverAxis, parse_coc_trace, parse_scene_reasoning_md

_SCENE_REASONING_FIXTURE = (
    Path(__file__).parent.parent
    / "pref_pairs/results/scene_reasoning/f0d61901-cfa0-46a4-8992-ab9ea553fc35_12988806_reasoning.md"
)


def test_single_clause_pairs_entity_and_state_with_the_commitment():
    parsed = parse_coc_trace(
        "Nudge left due to construction cones blocking the right side of our lane"
    )
    assert len(parsed.commitments) == 1
    commitment = parsed.commitments[0]
    assert commitment.maneuver == "nudge"
    assert commitment.axis is ManeuverAxis.LATERAL
    assert commitment.direction == "left"

    assert len(parsed.causal) == 1
    causal = parsed.causal[0]
    assert causal.connective == "due to"
    assert causal.effects == [commitment]
    assert [(c.entity, c.state) for c in causal.cause] == [
        ("construction_cones", "blocking"),
        ("lane", "blocking"),
    ]


def test_compound_commitment_yields_one_claim_per_verb():
    parsed = parse_coc_trace("Accelerate and turn right because of a clear intersection ahead")
    maneuvers = [(c.maneuver, c.axis, c.direction) for c in parsed.commitments]
    assert maneuvers == [
        ("accelerate", ManeuverAxis.LONGITUDINAL, None),
        ("turn", ManeuverAxis.LATERAL, "right"),
    ]
    # both verbs share the one stated cause
    assert len(parsed.causal) == 1
    assert [c.maneuver for c in parsed.causal[0].effects] == ["accelerate", "turn"]
    assert [(c.entity, c.state) for c in parsed.causal[0].cause] == [("intersection", "clearing")]


def test_weak_connective_for_is_used_only_when_no_strong_connective_present():
    parsed = parse_coc_trace("Adapt speed for the construction cones narrowing our lane ahead")
    assert parsed.causal[0].connective == "for"
    assert ("construction_cones", "narrowing") in [
        (c.entity, c.state) for c in parsed.causal[0].cause
    ]


def test_strong_connective_wins_over_an_earlier_weak_one_in_the_same_beat():
    # "for" (weak) appears before "because" (strong) here -- the split must
    # still happen at "because", not the earlier "for" (see _split_beat).
    parsed = parse_coc_trace(
        "Create a usable gap for a left lane change because cones and work trucks "
        "block our lane ahead and a car is behind in our left lane"
    )
    assert parsed.causal[0].connective == "because"
    # the cause clause must start after "because", not the earlier "for"
    assert "construction_cones" in {c.entity for c in parsed.causal[0].cause}


def test_entity_and_state_can_pair_across_the_connective_split():
    # "roundabout" sits before "since", "yield-controlled" sits after it --
    # only scanning the whole beat (not the cause clause alone) pairs them.
    parsed = parse_coc_trace("Adapt speed for the roundabout since a yield-controlled entry is ahead")
    roundabout_claims = [p for p in parsed.perceptual if p.entity == "roundabout"]
    assert len(roundabout_claims) == 1
    assert roundabout_claims[0].state == "yield_controlled"
    # the state word itself must not show up as unparsed just because it
    # wasn't inside the entity's own PerceptualClaim.span
    unparsed_text = " ".join(parsed.raw_text[s:e] for s, e in parsed.unparsed_spans)
    assert "yield-controlled" not in unparsed_text


def test_sequential_beats_split_on_semicolon_and_each_get_their_own_cause():
    parsed = parse_coc_trace(
        "Change lanes to the left due to emergency vehicles blocking the right side "
        "of our lane; then merge back right after clearing them, as the opposing lane "
        "is clear and the right shoulder is partially blocked by stopped emergency "
        "vehicles with pedestrians nearby, making a left pass the safest way to "
        "maintain progress."
    )
    assert [(c.maneuver, c.direction) for c in parsed.commitments] == [
        ("lane_change", "left"),
        ("merge", "right"),
    ]
    assert len(parsed.causal) == 2
    assert parsed.causal[0].connective == "due to"
    assert [c.maneuver for c in parsed.causal[0].effects] == ["lane_change"]
    assert parsed.causal[1].connective == "as"
    assert [c.maneuver for c in parsed.causal[1].effects] == ["merge"]


def test_direction_is_not_shared_across_two_different_maneuvers_in_one_clause():
    parsed = parse_coc_trace(
        "Change lanes to the right and enter the freeway on-ramp because the "
        "signal permits movement"
    )
    lane_change, enter = parsed.commitments
    assert lane_change.maneuver == "lane_change" and lane_change.direction == "right"
    assert enter.maneuver == "enter" and enter.direction is None


def test_no_connective_beat_still_extracts_a_chained_commitment_only_claim():
    # "Stop to yield ... wait ... before proceeding." has no
    # because/since/due to/for/after connective at all in this corpus's
    # sense -- every verb here is a real ego commitment, not a stated cause.
    parsed = parse_coc_trace(
        "Stop to yield to the pedestrian walking across our lane ahead, blocking "
        "our path; wait until they clear before proceeding."
    )
    assert [c.maneuver for c in parsed.commitments] == ["stop", "yield", "wait", "proceed"]
    assert parsed.causal == []
    assert any(p.entity == "pedestrian" for p in parsed.perceptual)


def test_adjectival_right_turn_on_the_cause_side_is_not_a_turn_commitment():
    parsed = parse_coc_trace(
        "Accelerate to proceed through the intersection since the right-turn "
        "traffic light turns green"
    )
    assert [c.maneuver for c in parsed.commitments] == ["accelerate", "proceed"]
    assert ("signal", "green") in [(p.entity, p.state) for p in parsed.perceptual]


def test_unicode_hyphen_variant_is_normalized_and_still_matched():
    # U+2011 (non-breaking hyphen), not ASCII '-' -- real model output uses
    # both for the same word across the corpus.
    parsed = parse_coc_trace("Adapt speed for the narrowed work‑zone lane ahead")
    assert any(p.entity == "work_zone" for p in parsed.perceptual)


def test_unrecognized_prose_is_reported_as_unparsed_not_silently_dropped():
    parsed = parse_coc_trace(
        "Change lanes to the left and enter the left side street due to the "
        "straight-ahead lane being closed by construction barricades, slowing "
        "for the tight entry and then accelerating after clearing the obstruction"
    )
    assert parsed.unparsed_spans, "expected some genuinely unmatched prose in this long beat"
    for start, end in parsed.unparsed_spans:
        assert 0 <= start < end <= len(parsed.raw_text)


def test_parse_scene_reasoning_md_extracts_scene_id_and_all_rollouts():
    traces = parse_scene_reasoning_md(_SCENE_REASONING_FIXTURE)
    assert len(traces) == 100
    assert all(t.scene_id == "f0d61901-cfa0-46a4-8992-ab9ea553fc35_12988806" for t in traces)
    assert [t.rollout_id for t in traces] == list(range(100))
    # every rollout in this fixture states either a nudge or an adapt_speed
    # commitment (see the report's "## lane_change_left (100 rollouts)" --
    # a lane-change maneuver class, but individual rollouts phrase the same
    # underlying action either way)
    assert all(t.commitments for t in traces)
    assert {t.commitments[0].maneuver for t in traces} <= {"nudge", "adapt_speed"}


def test_closed_beat_claims_are_stable_under_prefix_truncation():
    # Answers notes.txt's open question ("does this also work for the
    # prefix idea?"): when the parser sees a PREFIX of a CoC string (a
    # partial, still-generating reasoning trace, the fixed_reasoning /
    # _rollout_prefix setting), claims extracted from CLOSED beats — beats
    # a delimiter has already moved past — must be exactly what the full
    # text's parse yields for those spans. The final beat of a prefix is
    # provisional by construction (a later strong connective can re-split
    # it, e.g. "Stop for the stop sign" vs "...stop sign BECAUSE ...", and
    # its cause clause may be half-generated), so it is excluded — that
    # exclusion IS the prefix-scoring protocol this test encodes.
    #
    # Verified corpus-wide before being locked in here as a regression
    # test: 0 crashes and 0 closed-beat disagreements across all 33,136
    # word-boundary prefixes of all 2,032 unique corpus strings
    # (2026-07-21). Running the full sweep is too slow for a unit test, so
    # this uses the same 100-rollout fixture the ingestion test reads.
    import re

    from code_as_a_reward.coc_claim_parser import _split_beats

    def closed_beat_claims(trace, limit):
        return {
            ("c", c.maneuver, c.span) for c in trace.commitments if c.span[1] <= limit
        } | {
            ("p", p.entity, p.span) for p in trace.perceptual if p.span[1] <= limit
        } | {
            ("x", x.connective.lower(), x.span) for x in trace.causal if x.span[1] <= limit
        }

    for trace in parse_scene_reasoning_md(_SCENE_REASONING_FIXTURE):
        text = trace.raw_text
        for m in re.finditer(r"\S+", text):
            cut = m.end()
            if cut == len(text):
                continue
            partial = parse_coc_trace(text[:cut])  # totality: must not raise
            beats = _split_beats(partial.raw_text)
            closed_limit = beats[-1][0] if len(beats) > 1 else 0
            assert closed_beat_claims(partial, closed_limit) == closed_beat_claims(
                trace, closed_limit
            ), f"closed-beat instability at cut={cut} of {text!r}"
