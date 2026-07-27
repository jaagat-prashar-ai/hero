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

import argparse
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


_DEFAULT_TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def embed_traces(traces: list[str], model_name: str = _DEFAULT_TEXT_ENCODER) -> torch.Tensor:
    """Embed reasoning traces with the frozen sentence encoder, unit-normalized.

    sentence_transformers is imported lazily so that everything above this
    line (the pure helpers + the head, i.e. all the pytest-covered surface)
    stays importable with torch alone -- the Bazel test target doesn't have
    sentence-transformers in @python_deps, and doesn't need it.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        traces, convert_to_tensor=True, normalize_embeddings=True
    ).cpu()


def save_checkpoint(head: WaypointProjectionHead, text_encoder_name: str, path: Path | str) -> None:
    """Persist the trained head + the encoder identity it was trained against.

    The encoder name travels WITH the weights because the head is only
    meaningful in that specific encoder's embedding space -- loading it and
    scoring against a different encoder's embeddings would silently produce
    garbage similarities.
    """
    torch.save(
        {
            "state_dict": head.state_dict(),
            "num_waypoints": head.net[0].in_features // 2,
            "embedding_dim": head.net[-1].out_features,
            "hidden_dim": head.net[0].out_features,
            "text_encoder_name": text_encoder_name,
        },
        path,
    )


class FaithfulnessEmbeddingGrader:
    """Local, no-API faithfulness scorer: cosine(trace embedding, projected trajectory).

    Usage:
        grader = FaithfulnessEmbeddingGrader.load("head.pt")
        score = grader.score(trace_text, action_dict)   # in [-1, 1], higher = more faithful

    Scores are relative, not calibrated probabilities: use them to RANK
    rollouts or threshold against percentiles measured on judged data, per
    the module docstring's pre-filter-not-replacement caveat.
    """

    def __init__(self, head: WaypointProjectionHead, text_encoder_name: str) -> None:
        self.head = head.eval()
        self.text_encoder_name = text_encoder_name

    @classmethod
    def load(cls, checkpoint_path: Path | str) -> "FaithfulnessEmbeddingGrader":
        ckpt = torch.load(checkpoint_path, weights_only=True)
        head = WaypointProjectionHead(
            num_waypoints=ckpt["num_waypoints"],
            embedding_dim=ckpt["embedding_dim"],
            hidden_dim=ckpt["hidden_dim"],
        )
        head.load_state_dict(ckpt["state_dict"])
        return cls(head, ckpt["text_encoder_name"])

    def score(self, trace: str, action: dict[str, Any]) -> float:
        """Faithfulness of one reasoning trace to one trajectory."""
        return self.score_batch([trace], [action])[0]

    def score_batch(self, traces: list[str], actions: list[dict[str, Any]]) -> list[float]:
        """Batched scoring -- one encoder call for all traces, one head pass."""
        trace_embs = embed_traces(traces, self.text_encoder_name)
        features = torch.stack([waypoints_to_feature_vector(a) for a in actions])
        with torch.no_grad():
            traj_embs = self.head(features)
        return (trace_embs * traj_embs).sum(dim=-1).tolist()


def _triplets_to_tensors(
    triplets: list[dict[str, Any]], text_encoder_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(features, chosen_embeddings, rejected_embeddings) for a triplet list."""
    features = torch.stack([waypoints_to_feature_vector(t["action"]) for t in triplets])
    chosen = embed_traces([t["chosen_trace"] for t in triplets], text_encoder_name)
    rejected = embed_traces([t["rejected_trace"] for t in triplets], text_encoder_name)
    return features, chosen, rejected


def main() -> None:
    """Train the projection head and report held-out pairwise accuracy.

    Writes head.pt (the deployable checkpoint FaithfulnessEmbeddingGrader.load
    consumes) and eval_report.json (config + train/holdout accuracy) to
    --out-dir. The holdout accuracy in that report is THE validation number
    for this module -- compare it against llm_judge.py's 84.6% construction
    agreement before trusting the grader as a pre-filter.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--judged-pairs", type=Path, default=_DEFAULT_JUDGED_PAIRS_PATH)
    parser.add_argument("--matched-pairs", type=Path, default=_DEFAULT_MATCHED_PAIRS_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "pref_pairs/results/faithfulness_embedding_grader",
    )
    parser.add_argument("--text-encoder", default=_DEFAULT_TEXT_ENCODER)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    triplets = load_training_triplets(args.judged_pairs, args.matched_pairs)
    train, holdout = split_triplets(triplets, args.holdout_fraction, args.seed)
    print(f"{len(triplets)} judge-agreed triplets -> {len(train)} train / {len(holdout)} holdout")

    train_tensors = _triplets_to_tensors(train, args.text_encoder)
    holdout_tensors = _triplets_to_tensors(holdout, args.text_encoder)

    head, history = train_projection_head(
        *train_tensors,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    report = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_triplets": len(triplets),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "loss_first_epoch": history[0],
        "loss_last_epoch": history[-1],
        "train_pairwise_accuracy": pairwise_accuracy(head, *train_tensors),
        "holdout_pairwise_accuracy": pairwise_accuracy(head, *holdout_tensors),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(head, args.text_encoder, args.out_dir / "head.pt")
    with open(args.out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
