# SPDX-License-Identifier: Apache-2.0
"""GPU worker: samples a real Alpamayo rollout group per target clip via
rollout_sampler.py, writes one <clip_id>.json per clip. Runs INSIDE the
bootstrapped alpamayo1_5 Python 3.12 venv (ood_eval.bootstrap_venv), invoked
as a subprocess by run_real_rollout_gen.py -- same process-boundary split as
code_as_a_reward/ood_eval/run.py + worker.py, and for the same reason: torch/
alpamayo1_5 aren't importable in Lilypad's base Python 3.10 env.

No LLM calls here (keeps Anthropic/OpenAI SDKs out of this venv); the
base-env driver runs the LLM generation+gate loop afterward, reading these
files (see run_prototype.py's module docstring).

Resumable: skips clip_ids whose output file already exists, so a preempted
node (or a re-run after run_real_rollout_gen.py restores prior progress
from S3) doesn't resample clips it already has.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)
ROLLOUT_SCHEMA_VERSION = "clipgen.rollouts.v2"


def _clip_seed(base_seed: int, clip_id: str) -> int:
    digest = hashlib.sha256(clip_id.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def _tensor_waypoints(value) -> list[list[float]]:
    """Flatten the official ego_future_xyz leading singleton dimensions."""
    arr = value.detach().float().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"ego_future_xyz has unexpected shape {arr.shape}")
    return arr.tolist()


def _valid_existing(
    path: str,
    *,
    clip_id: str,
    t0_us: int,
    generation_group_size: int,
    holdout_group_size: int,
    base_seed: int,
    model_checkpoint: str,
    model_revision: str,
    top_p: float,
    temperature: float,
    max_generation_length: int,
) -> bool:
    try:
        with open(path) as f:
            doc = json.load(f)
        groups = doc.get("groups", {})
        generation = groups.get("generation", [])
        holdout = groups.get("holdout", [])
        provenance = doc.get("provenance", {})
        generation_seed = _clip_seed(base_seed, clip_id)
        holdout_seed = (generation_seed + 1) % (2**31 - 1)
        generation_ids = [
            rollout.get("rollout_id")
            for rollout in generation
            if isinstance(rollout, dict)
        ]
        holdout_ids = [
            rollout.get("rollout_id")
            for rollout in holdout
            if isinstance(rollout, dict)
        ]
        return (
            doc.get("schema_version") == ROLLOUT_SCHEMA_VERSION
            and doc.get("clip_id") == clip_id
            and int(doc.get("t0_us")) == int(t0_us)
            and len(generation) == generation_group_size
            and len(holdout) == holdout_group_size
            and len(generation_ids) == generation_group_size
            and len(set(generation_ids)) == len(generation_ids)
            and len(holdout_ids) == holdout_group_size
            and len(set(holdout_ids)) == len(holdout_ids)
            and len(doc.get("gt_waypoints", [])) > 1
            and provenance.get("model") == model_checkpoint
            and provenance.get("model_revision") == model_revision
            and float(provenance.get("top_p")) == float(top_p)
            and float(provenance.get("temperature")) == float(temperature)
            and int(provenance.get("max_generation_length")) == int(max_generation_length)
            and int(provenance.get("base_seed")) == int(base_seed)
            and int(provenance.get("generation_seed")) == generation_seed
            and int(provenance.get("holdout_seed")) == holdout_seed
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_json_dump(path: str, doc: dict) -> None:
    parent = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", dir=parent, delete=False) as f:
        json.dump(doc, f)
        f.flush()
        os.fsync(f.fileno())
        tmp_path = f.name
    os.replace(tmp_path, path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets_json", required=True, help='JSON list of {"clip_id","t0_us"}')
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--generation_group_size", type=int, default=12)
    ap.add_argument("--holdout_group_size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    from code_as_a_reward.clipgen.rollout_sampler import (
        MAX_GENERATION_LENGTH,
        MODEL_CHECKPOINT,
        MODEL_REVISION,
        TEMPERATURE,
        TOP_P,
        fetch_clip_data,
        load_model,
        sample_rollout_group,
    )

    with open(args.targets_json) as f:
        targets = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    # A resumed smoke/full shard commonly has every requested rollout JSON
    # restored from S3. Validate against the pinned checkpoint revision
    # before allocating/loading a 10B model; the previous implementation
    # spent several GPU-minutes loading it only to skip every clip.
    configured_revision = MODEL_REVISION or "unreported"
    all_cached = all(
        _valid_existing(
            os.path.join(args.output_dir, f"{t['clip_id']}.json"),
            clip_id=t["clip_id"],
            t0_us=t["t0_us"],
            generation_group_size=args.generation_group_size,
            holdout_group_size=args.holdout_group_size,
            base_seed=args.seed,
            model_checkpoint=MODEL_CHECKPOINT,
            model_revision=configured_revision,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            max_generation_length=MAX_GENERATION_LENGTH,
        )
        for t in targets
    )
    if all_cached:
        logger.info(
            "done: 0 ok, 0 failed, %d already-done skipped (model load avoided)",
            len(targets),
        )
        return

    logger.info("loading model ...")
    model = load_model()
    logger.info("model loaded")
    resolved_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None)
        or MODEL_REVISION
        or "unreported"
    )

    n_ok = n_skipped = n_failed = 0
    for t in targets:
        clip_id, t0_us = t["clip_id"], t["t0_us"]
        out_path = os.path.join(args.output_dir, f"{clip_id}.json")
        if os.path.exists(out_path) and _valid_existing(
            out_path,
            clip_id=clip_id,
            t0_us=t0_us,
            generation_group_size=args.generation_group_size,
            holdout_group_size=args.holdout_group_size,
            base_seed=args.seed,
            model_checkpoint=MODEL_CHECKPOINT,
            model_revision=resolved_revision,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            max_generation_length=MAX_GENERATION_LENGTH,
        ):
            n_skipped += 1
            continue
        try:
            data = fetch_clip_data(clip_id, t0_us)
            generation_seed = _clip_seed(args.seed, clip_id)
            holdout_seed = (generation_seed + 1) % (2**31 - 1)
            generation = sample_rollout_group(
                model,
                data,
                n=args.generation_group_size,
                seed=generation_seed,
            )
            # A separate stochastic forward pass. This group is never shown
            # to the generator and is evaluated once for final acceptance.
            holdout = sample_rollout_group(
                model,
                data,
                n=args.holdout_group_size,
                seed=holdout_seed,
            )
            doc = {
                "schema_version": ROLLOUT_SCHEMA_VERSION,
                "clip_id": clip_id,
                "t0_us": int(t0_us),
                # This is the exact expert future consumed by the training
                # dataset, already expressed in the keyframe ego frame.
                "gt_waypoints": _tensor_waypoints(data["ego_future_xyz"]),
                "groups": {"generation": generation, "holdout": holdout},
                "provenance": {
                    "model": MODEL_CHECKPOINT,
                    "model_revision": resolved_revision,
                    "top_p": TOP_P,
                    "temperature": TEMPERATURE,
                    "max_generation_length": MAX_GENERATION_LENGTH,
                    "base_seed": int(args.seed),
                    "generation_seed": generation_seed,
                    "holdout_seed": holdout_seed,
                },
            }
            _atomic_json_dump(out_path, doc)
            logger.info(
                "clip %s: sampled %d generation + %d holdout rollouts",
                clip_id,
                len(generation),
                len(holdout),
            )
            n_ok += 1
        except Exception:
            logger.exception("clip %s: rollout sampling failed, skipping", clip_id)
            n_failed += 1

    logger.info("done: %d ok, %d failed, %d already-done skipped", n_ok, n_failed, n_skipped)
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
