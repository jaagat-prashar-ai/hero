# SPDX-License-Identifier: Apache-2.0
"""
build_reasoning_matched_pairs.py — flatten pref_pairs/results/perturbation_actions/
ground_truth_actions.json (one nested object per scene) into one DPO-style
preference-pair row per perturbation.

"Reasoning-matched" pair, per the project brief: SAME action, reasoning differs
in correctness -- chosen=ground_truth_trace (the trace that actually produced
the action), rejected=perturbed_trace (a semantically corrupted variant that
never ran through the model; see pref_pairs.perturbation_generator). The
action is identical on both sides of a pair by construction (we didn't re-run
inference on the perturbed text -- see build_ground_truth_action_dataset.py's
module docstring for why that's the correct comparison for this pair type).

One scene with N perturbations yields N pairs, all sharing that scene's
ground_truth_action -- 717 total pairs across 120 scenes (matches
perturbations.jsonl's row count exactly, since this is a pure reshape/join,
not a filter).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PERTURBATION_META_FIELDS = [
    "perturbation_type", "original_span", "perturbed_span",
    "semantic_delta", "decision_impact", "plausibility_rationale",
]


def load_ground_truth_actions(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        return json.load(fh)


def build_pairs(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One flat pair per (scene, perturbation) -- chosen/rejected trace text
    plus the scene's single shared ground_truth_action, duplicated onto every
    pair from that scene (small redundancy, but keeps each row self-contained
    for a DPO data loader that reads one line at a time)."""
    pairs: list[dict[str, Any]] = []
    for scene in scenes:
        action = scene["ground_truth_action"]
        for pert in scene["perturbations"]:
            pairs.append({
                "pair_id": pert["trace_id"],
                "scene_id": scene["scene_id"],
                "event_cluster": scene["event_cluster"],
                "chosen_trace": scene["ground_truth_trace"],
                "rejected_trace": pert["perturbed_trace"],
                "action": action,
                **{field: pert[field] for field in _PERTURBATION_META_FIELDS},
            })
    return pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in_path",
        default="pref_pairs/results/perturbation_actions/ground_truth_actions.json",
    )
    ap.add_argument(
        "--out_path",
        default="pref_pairs/results/reasoning_matched_pairs/reasoning_matched_pairs.jsonl",
    )
    args = ap.parse_args()

    scenes = load_ground_truth_actions(args.in_path)
    pairs = build_pairs(scenes)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair) + "\n")
    logger.info("Wrote %d pairs from %d scenes to %s", len(pairs), len(scenes), out_path)


if __name__ == "__main__":
    main()
