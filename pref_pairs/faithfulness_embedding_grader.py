# SPDX-License-Identifier: Apache-2.0
"""
faithfulness_embedding_grader.py -- distilled, LLM-free reasoning/action
faithfulness scorer for Alpamayo CoT reasoning traces.

rl_posttrain/rewards/llm_judge.py already scores "does this reasoning trace
justify this trajectory" by sending the trace + a waypoint table to Claude
Fable 5 for every rollout. That's the validated ground truth (calibrated
against pref_pairs/results/judged_pairs/judged_pairs.jsonl -- 717 pairs,
84.6% agreement with construction labels) but costs one API round-trip per
rollout.

This module distills that judgment into a local, no-API scorer: a frozen
sentence-embedding text encoder embeds the reasoning trace, and a small
learned MLP ("projection head") maps the rollout's raw waypoint tensor into
the SAME embedding space. Faithfulness is then just cosine similarity
between the two vectors -- one forward pass, no network call, and no
templating of the trajectory into text anywhere in the pipeline.

The projection head is trained via contrastive (triplet) learning directly
on judged_pairs.jsonl: for each of the 717 judged pairs there is a SINGLE
trajectory (from reasoning_matched_pairs.jsonl, joined by pair_id) paired
with a `chosen_trace` (faithful) and a `rejected_trace` (a corrupted,
unfaithful variant of the same trace). Training pushes the trajectory's
projected embedding closer to `chosen_trace`'s text embedding than to
`rejected_trace`'s -- i.e. it never needs the judge's absolute 0-10 scores,
only the (chosen, rejected) ranking that construction + judge agreement
already gives us for free.

This is a proxy for the Claude judge, not a replacement: use it to cheaply
pre-filter or rank rollouts (e.g. skip the API call for obviously-faithful
or obviously-unfaithful traces) rather than as the sole GRPO reward signal,
since it has not been independently validated the way llm_judge.py has.

Per the project's no-fake-model-tests preference (see llm_judge.py's module
docstring): only the pure helpers in this file (load_training_triplets,
waypoints_to_feature_vector, the loss math) get pytest coverage. The frozen
text encoder and the trained projection head are verified by the held-out
pairwise-accuracy eval this module's CLI runs against judged_pairs.jsonl
itself, not by a mocked-model test.

Built incrementally -- see git history for this file for the commit-per-piece
breakdown (dataset join -> feature extraction -> model -> training loop ->
inference API -> CLI/tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JUDGED_PAIRS_PATH = _REPO_ROOT / "pref_pairs/results/judged_pairs/judged_pairs.jsonl"
_DEFAULT_MATCHED_PAIRS_PATH = (
    _REPO_ROOT / "pref_pairs/results/reasoning_matched_pairs/reasoning_matched_pairs.jsonl"
)


def load_training_triplets(
    judged_pairs_path: Path | str = _DEFAULT_JUDGED_PAIRS_PATH,
    matched_pairs_path: Path | str = _DEFAULT_MATCHED_PAIRS_PATH,
) -> list[dict[str, Any]]:
    """Join judged_pairs.jsonl with reasoning_matched_pairs.jsonl on pair_id.

    judged_pairs.jsonl (judge_reasoning_pairs.py's output) has the chosen/
    rejected trace text and the judge's scores, but NOT the trajectory
    itself. reasoning_matched_pairs.jsonl (the judge's own input) has the
    same chosen/rejected trace text plus the `action` dict (waypoints +
    kinematic summary) they were judged against. Joining on `pair_id`
    recovers exactly the (chosen_trace, rejected_trace, action) triplets
    this module trains on.

    Only pairs where the judge agreed with construction
    (`judge_agrees_with_construction: true`) are kept -- the ~15%
    disagreements are exactly the ones judge_reasoning_pairs.py flags as
    ambiguous or mislabeled, and training a ranking objective on a wrong
    label would teach the projection head backwards.

    Returns:
        List of dicts: {"pair_id", "chosen_trace", "rejected_trace", "action"}.
    """
    judged_by_pair_id = {}
    with open(judged_pairs_path) as f:
        for line in f:
            row = json.loads(line)
            if row["judge_agrees_with_construction"]:
                judged_by_pair_id[row["pair_id"]] = row

    triplets = []
    with open(matched_pairs_path) as f:
        for line in f:
            row = json.loads(line)
            if row["pair_id"] not in judged_by_pair_id:
                continue
            triplets.append(
                {
                    "pair_id": row["pair_id"],
                    "chosen_trace": row["chosen_trace"],
                    "rejected_trace": row["rejected_trace"],
                    "action": row["action"],
                }
            )
    return triplets
