# SPDX-License-Identifier: Apache-2.0
"""
rule_based_generator.py — mechanical (no-LLM) alternative to
dpo_pairs.perturbation_generator_v2. Free, deterministic, offline.

Rationale: dpo_pairs.perturbation_generator_v2.select_targets() already
reduces most of its taxonomy to closed-vocabulary swaps that are spelled out
verbatim in that module's own LLM system prompt (left<->right,
blocking<->clearing, pedestrian<->cyclist, ...) and coc_claim_parser's
lexicons (ENTITY_PATTERNS, STATE_PATTERNS, DIRECTION_PATTERN). Picking a
replacement from a fixed table needs no generation. This module covers the
four types where that holds:

  commitment_direction_flip   left <-> right
  perceptual_state_flip       antonym from a small fixed table
  perceptual_entity_swap      confusable entity from a small fixed table
  quantity_edit               numeric value x10 (unit unchanged)

commitment_maneuver_swap and causal_cause_substitution are deliberately NOT
implemented here: both need grammatical generation (verb conjugation, a
fluent replacement clause) rather than a lookup, which is exactly where
perturbation_generator_v2's LLM still earns its keep.

Reuses select_targets/validate_perturbation (dpo_pairs.perturbation_generator_v2)
and extract_ground_truth_traces/write_perturbations_jsonl (pref_pairs.
perturbation_generator) so the output is the same schema run.py/mine_pairs.py
already consume — this is a drop-in alternative corpus source, not a new
pipeline stage.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any

from code_as_a_reward.coc_claim_parser import DIRECTION_PATTERN, ENTITY_PATTERNS, STATE_PATTERNS
from dpo_pairs.perturbation_generator_v2 import select_targets, validate_perturbation
from pref_pairs.perturbation_generator import extract_ground_truth_traces, write_perturbations_jsonl

logger = logging.getLogger(__name__)

TAXONOMY_VERSION = "rule_v1"

SUPPORTED_TYPES: tuple[str, ...] = (
    "commitment_direction_flip",
    "perceptual_state_flip",
    "perceptual_entity_swap",
    "quantity_edit",
)

DIRECTION_FLIP = {"left": "right", "right": "left"}

# Only antonym pairs that are both themselves recognized STATE_PATTERNS keys
# and unambiguous opposites — a subset of the LLM prompt's list, deliberately
# not exhaustive (e.g. "stopped" has no clean single-word opposite key here).
STATE_FLIP = {
    "blocking": "clearing",
    "clearing": "blocking",
    "closed": "open",
    "open": "closed",
    "red": "green",
    "green": "red",
}

# Only visually-confusable pairs that are both recognized ENTITY_PATTERNS
# keys with an unambiguous canonical replacement word (e.g. "debris" isn't a
# lexicon entry at all, so cones<->debris from the LLM prompt has no
# mechanical equivalent here).
ENTITY_SWAP = {
    "pedestrian": "cyclist",
    "cyclist": "pedestrian",
}

_STATE_PATTERN_BY_KEY = dict(STATE_PATTERNS)
_ENTITY_PATTERN_BY_KEY = dict(ENTITY_PATTERNS)

_QUANTITY_VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)(\s*)(.*)$")


def _match_case(replacement: str, original: str) -> str:
    """Mirror the original span's capitalization (style-camouflage rule)."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


def _substitute(trace: str, start: int, end: int, replacement: str) -> dict[str, str]:
    return {
        "original_span": trace[start:end],
        "perturbed_span": replacement,
        "perturbed_trace": trace[:start] + replacement + trace[end:],
    }


def _nearest_match(
    trace: str, pattern: re.Pattern[str], anchor: tuple[int, int],
) -> re.Match | None:
    """The pattern match closest to the anchor span. A claim's own span
    doesn't always contain the word being replaced — e.g. a commitment
    claim's span is just the maneuver verb ("Nudge"); its direction word
    ("left") is resolved by coc_claim_parser from a window AFTER it, not
    inside the claim span — so search the whole trace and pick nearest
    rather than assuming containment."""
    a_start, a_end = anchor
    best, best_dist = None, None
    for m in pattern.finditer(trace):
        if m.end() <= a_start:
            dist = a_start - m.end()
        elif m.start() >= a_end:
            dist = m.start() - a_end
        else:
            dist = 0
        if best_dist is None or dist < best_dist:
            best, best_dist = m, dist
    return best


def _apply_direction_flip(trace: str, target: dict[str, Any]) -> dict[str, str] | None:
    direction = target["details"]["direction"]
    m = _nearest_match(trace, DIRECTION_PATTERN, tuple(target["span"]))
    if m is None or m.group(0).lower() != direction:
        return None
    replacement = _match_case(DIRECTION_FLIP[direction], m.group(0))
    return _substitute(trace, m.start(), m.end(), replacement)


def _apply_state_flip(trace: str, target: dict[str, Any]) -> dict[str, str] | None:
    state = target["details"]["state"]
    flipped = STATE_FLIP.get(state)
    if flipped is None:
        return None
    pattern = _STATE_PATTERN_BY_KEY[state]
    m = _nearest_match(trace, pattern, tuple(target["span"]))
    # Some STATE_PATTERNS regexes match several inflections of one key (e.g.
    # "blocking"'s pattern also matches bare "block"/"blocked"/"blocks");
    # STATE_FLIP's replacement word is a fixed gerund, so require the actual
    # match to be that exact spelling too, else e.g. adjective "clear" (-> a
    # gerund-suffix pattern) would turn into the ungrammatical "is blocking".
    if m is None or m.group(0).lower() != state:
        return None
    replacement = _match_case(flipped, m.group(0))
    return _substitute(trace, m.start(), m.end(), replacement)


# ENTITY_PATTERNS' cyclist regex lists "scooters?" as an earlier alternative
# than "scooter\s+riders?" -- Python's alternation takes the first
# alternative that matches at a position, not the longest, so on the
# corpus's common "scooter rider" phrasing the match is just "scooter",
# stranding "rider" and turning a swap into "pedestrian rider". Skip that
# one continuation; bare "scooter ahead" (no trailing "rider") swaps fine.
_STRANDED_TAIL_RE = re.compile(r"\s+riders?\b", re.I)


def _apply_entity_swap(trace: str, target: dict[str, Any]) -> dict[str, str] | None:
    entity = target["details"]["entity"]
    swapped = ENTITY_SWAP.get(entity)
    if swapped is None:
        return None
    pattern = _ENTITY_PATTERN_BY_KEY[entity]
    m = _nearest_match(trace, pattern, tuple(target["span"]))
    if m is None or _STRANDED_TAIL_RE.match(trace, m.end()):
        return None
    replacement = swapped + "s" if m.group(0).lower().endswith("s") else swapped
    replacement = _match_case(replacement, m.group(0))
    return _substitute(trace, m.start(), m.end(), replacement)


def _apply_quantity_edit(trace: str, target: dict[str, Any]) -> dict[str, str] | None:
    start, end = target["span"]
    original = trace[start:end]
    m = _QUANTITY_VALUE_RE.match(original)
    if m is None:
        return None
    num_str, sep, unit = m.groups()
    value = float(num_str) * 10
    if "." in num_str:
        decimals = len(num_str.split(".")[1])
        new_num = f"{value:.{decimals}f}"
    else:
        new_num = str(int(value))
    return _substitute(trace, start, end, new_num + sep + unit)


_APPLY_BY_TYPE = {
    "commitment_direction_flip": _apply_direction_flip,
    "perceptual_state_flip": _apply_state_flip,
    "perceptual_entity_swap": _apply_entity_swap,
    "quantity_edit": _apply_quantity_edit,
}


def generate_perturbation_rule_based(
    trace: str, perturbation_type: str, target: dict[str, Any],
) -> dict[str, str] | None:
    """{original_span, perturbed_span, perturbed_trace} or None if this type
    isn't mechanically supported, or the fixed table has no entry for this
    trace's specific claim (e.g. a state/entity outside STATE_FLIP/
    ENTITY_SWAP) — same graceful-skip convention as v2's applicable-types
    filtering, just resolved earlier."""
    apply = _APPLY_BY_TYPE.get(perturbation_type)
    if apply is None:
        return None
    result = apply(trace, target)
    if result is None:
        return None
    if validate_perturbation(trace, result) is not None:
        return None
    return result


def generate_all_perturbations_rule_based(
    ground_truth_traces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """(rows, n_skipped) — one row per (scene, type) that had both an
    applicable target AND a mechanical rule that produced a valid
    substitution."""
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for gt in ground_truth_traces:
        targets = select_targets(gt["trace"])
        for ptype in SUPPORTED_TYPES:
            target = targets.get(ptype)
            if target is None:
                continue
            perturbation = generate_perturbation_rule_based(gt["trace"], ptype, target)
            if perturbation is None:
                n_skipped += 1
                continue
            rows.append({
                "scene_id": gt["scene_id"],
                "event_cluster": gt.get("event_cluster"),
                "ground_truth_trace": gt["trace"],
                "trace_id": f"{gt['scene_id']}__{ptype}",
                "perturbation_type": ptype,
                "target_claim_type": target["claim_type"],
                "target_claim_span": target["span"],
                "taxonomy_version": TAXONOMY_VERSION,
                "generator": "rule_based",
                **perturbation,
            })
    return rows, n_skipped


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene_reasoning_dir",
                    default="pref_pairs/results/fixed_reasoning/scene_reasoning")
    ap.add_argument("--out_path", default="dpo_mechanical/results/perturbations.jsonl")
    ap.add_argument("--max_scenes", type=int, default=None)
    args = ap.parse_args()

    ground_truth_traces = extract_ground_truth_traces(args.scene_reasoning_dir)
    if args.max_scenes is not None:
        ground_truth_traces = ground_truth_traces[: args.max_scenes]

    rows, n_skipped = generate_all_perturbations_rule_based(ground_truth_traces)
    out_path = write_perturbations_jsonl(rows, args.out_path)
    logger.info(
        "%d scenes -> %d perturbations written to %s (%d skipped: unsupported type, "
        "no fixed-table entry, or failed mechanical validation) — no API cost",
        len(ground_truth_traces), len(rows), out_path, n_skipped,
    )


if __name__ == "__main__":
    main()
