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

import torch

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


def waypoints_to_feature_vector(action: dict[str, Any]) -> torch.Tensor:
    """Flatten a trajectory's (x, y) waypoints into a fixed-size numeric vector.

    Args:
        action: an `action` dict as stored on reasoning_matched_pairs.jsonl
            rows -- must have "waypoints", a (T, 3) xyz sequence (z is held
            constant by Alpamayo's action space, see
            unicycle_accel_curvature.py, so only x, y carry maneuver info,
            the same convention pref_pairs/trajectory_features.py relies on).

    Returns:
        1-D float32 tensor of length T*2 (x, y per waypoint, in time order).
        This is the raw flattened trajectory -- not any text description or
        hand-picked summary stat -- so WaypointProjectionHead (added in the
        next increment) is free to learn which parts of the trajectory
        shape matter for faithfulness, rather than inheriting our guesses.
    """
    xy = torch.tensor(action["waypoints"], dtype=torch.float32)[:, :2]
    return xy.reshape(-1)


class WaypointProjectionHead(torch.nn.Module):
    """Small MLP projecting a flattened waypoint vector into text-embedding space.

    Input is waypoints_to_feature_vector's output (num_waypoints * 2 floats);
    output is a unit-norm vector in the SAME space as the frozen text
    encoder's trace embeddings, so faithfulness is a plain dot product /
    cosine similarity between the two. Output normalization lives here (not
    in the caller) so every consumer -- triplet loss, inference, eval --
    gets the same geometry for free.

    Defaults match all-MiniLM-L6-v2 (384-dim), the intended frozen encoder
    (added with the inference API increment). With only ~600 training
    triplets (717 judged pairs minus judge/construction disagreements),
    the head is kept deliberately small; if held-out pairwise accuracy
    shows overfitting, shrink hidden_dim before reaching for regularizers.
    """

    def __init__(
        self,
        num_waypoints: int = 64,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_waypoints * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, waypoint_features: torch.Tensor) -> torch.Tensor:
        """Project (..., num_waypoints*2) features to unit-norm (..., embedding_dim)."""
        return torch.nn.functional.normalize(self.net(waypoint_features), dim=-1)


def cosine_triplet_loss(
    trajectory_embeddings: torch.Tensor,
    chosen_embeddings: torch.Tensor,
    rejected_embeddings: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin ranking loss on cosine similarity, batched over triplets.

    All three inputs are (batch, embedding_dim) and assumed unit-norm
    (WaypointProjectionHead normalizes its output; embed_traces asks the
    sentence encoder for normalized embeddings), so cosine similarity is a
    row-wise dot product. The loss for a triplet is

        relu(margin - sim(traj, chosen) + sim(traj, rejected))

    i.e. zero once the faithful trace beats the corrupted one by at least
    `margin`, so already-separated triplets stop pulling on the head and
    training effort concentrates on the still-confused ones.
    """
    sim_chosen = (trajectory_embeddings * chosen_embeddings).sum(dim=-1)
    sim_rejected = (trajectory_embeddings * rejected_embeddings).sum(dim=-1)
    return torch.relu(margin - sim_chosen + sim_rejected).mean()


def split_triplets(
    triplets: list[dict[str, Any]],
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic (train, holdout) split of load_training_triplets' output.

    Shuffles with a seeded torch generator so the same seed always yields the
    same split -- the held-out pairwise accuracy reported by the CLI must be
    reproducible, and the holdout must never leak into training between runs.
    """
    n_holdout = int(len(triplets) * holdout_fraction)
    order = torch.randperm(
        len(triplets), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    holdout_idx = set(order[:n_holdout])
    train = [t for i, t in enumerate(triplets) if i not in holdout_idx]
    holdout = [t for i, t in enumerate(triplets) if i in holdout_idx]
    return train, holdout


def pairwise_accuracy(
    head: WaypointProjectionHead,
    feature_vectors: torch.Tensor,
    chosen_embeddings: torch.Tensor,
    rejected_embeddings: torch.Tensor,
) -> float:
    """Fraction of triplets where the projected trajectory is closer to the
    faithful trace than to the corrupted one.

    This is the same pairwise-ranking metric the Claude judge's 84.6%
    construction-agreement figure is expressed in, so the two numbers are
    directly comparable.
    """
    with torch.no_grad():
        traj = head(feature_vectors)
        sim_chosen = (traj * chosen_embeddings).sum(dim=-1)
        sim_rejected = (traj * rejected_embeddings).sum(dim=-1)
        return (sim_chosen > sim_rejected).float().mean().item()


def train_projection_head(
    feature_vectors: torch.Tensor,
    chosen_embeddings: torch.Tensor,
    rejected_embeddings: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 0.2,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[WaypointProjectionHead, list[float]]:
    """Train a WaypointProjectionHead on precomputed (traj, chosen, rejected) rows.

    Takes tensors, not triplet dicts, so this stays decoupled from the
    sentence encoder: the caller embeds the traces once up front (they're
    frozen -- re-embedding per epoch would be pure waste) and this function
    is plain torch, runnable in tests with synthetic embeddings.

    The full dataset is ~600 triplets so everything sits in memory; epochs
    are full passes in shuffled minibatches. Returns the trained head and
    the per-epoch mean loss history (a non-decreasing tail is the CLI's cue
    that `epochs` is set too high or lr too low).
    """
    torch.manual_seed(seed)
    head = WaypointProjectionHead(
        num_waypoints=feature_vectors.shape[-1] // 2,
        embedding_dim=chosen_embeddings.shape[-1],
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(feature_vectors))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            loss = cosine_triplet_loss(
                head(feature_vectors[idx]),
                chosen_embeddings[idx],
                rejected_embeddings[idx],
                margin=margin,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(sum(epoch_losses) / len(epoch_losses))
    return head, history
