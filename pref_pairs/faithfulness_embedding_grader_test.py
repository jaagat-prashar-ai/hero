# SPDX-License-Identifier: Apache-2.0
"""
faithfulness_embedding_grader_test.py — tests for the pure helpers only
(dataset join, feature extraction, loss math, split, checkpoint round-trip).
Per the module docstring: the frozen text encoder and the trained head are
validated by the CLI's held-out pairwise-accuracy eval, not mocked here, so
nothing in this file needs sentence-transformers, a GPU, or the network.
"""

from __future__ import annotations

import json

import torch

from pref_pairs.faithfulness_embedding_grader import (
    FaithfulnessEmbeddingGrader,
    WaypointProjectionHead,
    cosine_triplet_loss,
    load_training_triplets,
    pairwise_accuracy,
    save_checkpoint,
    split_triplets,
    train_projection_head,
    waypoints_to_feature_vector,
)


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_training_triplets_joins_and_filters(tmp_path):
    judged = tmp_path / "judged.jsonl"
    matched = tmp_path / "matched.jsonl"
    _write_jsonl(
        judged,
        [
            {"pair_id": "a", "judge_agrees_with_construction": True},
            {"pair_id": "b", "judge_agrees_with_construction": False},
            # judge_reasoning_pairs.py emits None when the judge abstained --
            # must be dropped exactly like an explicit disagreement.
            {"pair_id": "c", "judge_agrees_with_construction": None},
            # judged but missing from matched -- must not crash the join.
            {"pair_id": "orphan", "judge_agrees_with_construction": True},
        ],
    )
    _write_jsonl(
        matched,
        [
            {"pair_id": p, "chosen_trace": "good", "rejected_trace": "bad", "action": {"waypoints": []}}
            for p in ("a", "b", "c")
        ],
    )
    triplets = load_training_triplets(judged, matched)
    assert [t["pair_id"] for t in triplets] == ["a"]
    assert set(triplets[0]) == {"pair_id", "chosen_trace", "rejected_trace", "action"}


def test_waypoints_to_feature_vector_drops_z_keeps_time_order():
    action = {"waypoints": [[1.0, 2.0, 9.0], [3.0, 4.0, 9.0], [5.0, 6.0, 9.0]]}
    vec = waypoints_to_feature_vector(action)
    assert vec.dtype == torch.float32
    assert vec.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_cosine_triplet_loss_zero_when_separated_beyond_margin():
    traj = torch.tensor([[1.0, 0.0]])
    chosen = torch.tensor([[1.0, 0.0]])  # sim 1.0
    rejected = torch.tensor([[0.0, 1.0]])  # sim 0.0
    assert cosine_triplet_loss(traj, chosen, rejected, margin=0.2).item() == 0.0


def test_cosine_triplet_loss_penalizes_violations():
    traj = torch.tensor([[1.0, 0.0]])
    chosen = torch.tensor([[0.0, 1.0]])  # sim 0.0
    rejected = torch.tensor([[1.0, 0.0]])  # sim 1.0
    # relu(0.2 - 0.0 + 1.0) = 1.2
    assert abs(cosine_triplet_loss(traj, chosen, rejected, margin=0.2).item() - 1.2) < 1e-6


def test_split_triplets_deterministic_and_disjoint():
    triplets = [{"pair_id": str(i)} for i in range(10)]
    train_a, holdout_a = split_triplets(triplets, holdout_fraction=0.2, seed=0)
    train_b, holdout_b = split_triplets(triplets, holdout_fraction=0.2, seed=0)
    assert (train_a, holdout_a) == (train_b, holdout_b)
    assert len(holdout_a) == 2 and len(train_a) == 8
    ids = {t["pair_id"] for t in train_a} | {t["pair_id"] for t in holdout_a}
    assert len(ids) == 10


def test_train_projection_head_learns_learnable_relationship():
    # Synthetic data where faithfulness IS learnable: chosen embeddings are a
    # fixed linear function of the features, rejected are unrelated noise.
    torch.manual_seed(0)
    features = torch.randn(80, 16)  # 8 waypoints
    projection = torch.randn(16, 32)
    chosen = torch.nn.functional.normalize(features @ projection, dim=-1)
    rejected = torch.nn.functional.normalize(torch.randn(80, 32), dim=-1)

    head, history = train_projection_head(
        features, chosen, rejected, epochs=40, seed=0
    )
    assert history[-1] < history[0]
    assert pairwise_accuracy(head, features, chosen, rejected) >= 0.9


def test_checkpoint_round_trip_restores_exact_outputs(tmp_path):
    head = WaypointProjectionHead(num_waypoints=8, embedding_dim=32, hidden_dim=16)
    path = tmp_path / "head.pt"
    save_checkpoint(head, "some/encoder", path)

    grader = FaithfulnessEmbeddingGrader.load(path)
    assert grader.text_encoder_name == "some/encoder"
    features = torch.randn(4, 16)
    with torch.no_grad():
        torch.testing.assert_close(grader.head(features), head(features))
