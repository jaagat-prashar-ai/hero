# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint for the pref_pairs rollout-harvest +
maneuver-classification pipeline.

For every scene in a clip manifest (built by masking.data.sample_clips --
see pref_pairs/configs/sample_clips_all.json for the "every OOD clip"
variant), this:
  1. Harvests K sampled rollouts from Alpamayo 1.5
     (pref_pairs.rollout_harvester.RolloutHarvester.harvest_scene).
  2. Extracts kinematic features and classifies a maneuver_class for each
     rollout (pref_pairs.trajectory_features / pref_pairs.classify_maneuvers).
  3. Appends one JSONL row per rollout (rollout fields + features +
     maneuver_class) to this rank's results file.

This mirrors masking/training/run.py's Lilypad conventions -- distributed
sharding via RANK/WORLD_SIZE env vars, append-only JSONL with resume -- but
is NOT built on that file's data-acquisition path. masking_loop calls
`iter_snapshots(shards)` after downloading full shards; as of this writing
`iter_snapshots` no longer exists anywhere in masking/data/wds_dataset.py
(verified: its only top-level functions are iter_clip_events,
iter_clip_events_from_manifest, and the internal helpers around them) --
that file appears to predate wds_dataset.py's rewrite to the current
raw-clip-data format and would fail at import time if run today. Flagging
this here since it's a real, separate bug in a file this module doesn't
own; NOT fixing it as part of this change.

pref_pairs' scene selection is inherently OOD-scoped (it needs the
ood_reasoning.parquet lookup that only masking.data.sample_clips does), so
this entrypoint consumes a pre-built manifest via
iter_clip_events_from_manifest -- the same range-read-only, no-full-download
path pref_pairs.rollout_harvester.harvest_dataset already uses -- rather
than mirroring masking_loop's full-shard-download-then-iterate-everything
approach.

Full config reference (all keys optional, defaults shown):
    manifest_path:     "pref_pairs/configs/sample_clips_all.json"
    bucket:             "research-datasets-chicago"
    checkpoint:         "nvidia/Alpamayo-1.5-10B"
    k:                  20
    seed:               0
    temperature:        0.6
    top_p:              0.98
    top_k:              null
    thresholds_config:  "pref_pairs/configs/maneuver_thresholds.yaml"
    resume:             false
    outdir:             "/tmp/pref_pairs_results"
    max_scenes:         null
    wandb_project:       "pref-pairs"
    wandb_entity:        "research"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "manifest_path": "pref_pairs/configs/sample_clips_all.json",
    "bucket": "research-datasets-chicago",
    "checkpoint": "nvidia/Alpamayo-1.5-10B",
    "k": 20,
    "seed": 0,
    "temperature": 0.6,
    "top_p": 0.98,
    "top_k": None,
    "thresholds_config": "pref_pairs/configs/maneuver_thresholds.yaml",
    "resume": False,
    "outdir": "/tmp/pref_pairs_results",
    "max_scenes": None,
    "wandb_project": "pref-pairs",
    "wandb_entity": "research",
    "rank": 0,
    "world_size": 1,
}


def _distributed_context(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from Lilypad env vars or config.

    Same pattern as masking/training/run.py's _distributed_context --
    reimplemented here (3 lines) rather than imported, since importing a
    masking.training helper would pull pref_pairs' model-facing code back
    toward a masking dependency, which rollout_harvester.py already went
    out of its way to avoid (see that module's docstring).
    """
    rank = int(os.environ.get("RANK", cfg.get("rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", cfg.get("world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def _scene_owner(scene_id: str, world_size: int) -> int:
    """Stable scene->rank assignment so resume/requeue always maps a scene
    to the same rank -- same hashing approach as masking's _shard_owner,
    just keyed on scene_id (already "{clip_id}_{t0_us}", see
    rollout_harvester.harvest_dataset) instead of a (clip_id, t0_us) pair."""
    digest = hashlib.md5(scene_id.encode()).hexdigest()
    return int(digest, 16) % world_size


def _results_path(outdir: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return outdir / "pref_pairs_rollouts.jsonl"
    return outdir / f"pref_pairs_rollouts_rank{rank:02d}.jsonl"


def _load_done_scenes(path: Path) -> set[str]:
    """Scenes already present in this rank's results file. Resume skips a
    scene entirely if ANY of its rollouts were already written -- scenes
    are processed atomically (all K rollouts written together, one flush),
    so a scene_id present at all means that scene finished, not partially."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                done.add(row["scene_id"])
            except Exception:
                pass
    return done


def _resolve_device(local_rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def pref_pairs_loop(training_fn_config: dict[str, Any], experiment_tracker: Any) -> None:
    """Lilypad-compatible entrypoint. See module docstring for the full
    config reference and design rationale."""
    cfg = {**_DEFAULTS, **training_fn_config}

    rank, world_size, local_rank = _distributed_context(cfg)
    device = _resolve_device(local_rank)

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(outdir, rank, world_size)
    logger.info(
        "Distributed context: rank=%d world_size=%d local_rank=%d device=%s",
        rank, world_size, local_rank, device,
    )
    logger.info("Writing results to %s", results_path)

    done: set[str] = set()
    if cfg["resume"]:
        done = _load_done_scenes(results_path)
        logger.info("Resuming: %d scenes already done on rank %d", len(done), rank)

    # Imported here, not at module load time, for the same reason
    # rollout_harvester.harvest_dataset lazily imports masking.data.wds_dataset:
    # keeps this module importable (e.g. for its pure helper functions above)
    # without requiring alpamayo1_5/transformers/masking's data deps installed.
    from masking.data.wds_dataset import iter_clip_events_from_manifest

    from pref_pairs.classify_maneuvers import FeatureConfig, ManeuverConfig, classify
    from pref_pairs.rollout_harvester import RolloutHarvester
    from pref_pairs.trajectory_features import extract_features

    with open(cfg["thresholds_config"]) as fh:
        raw_thresholds = yaml.safe_load(fh)
    feature_config = FeatureConfig.from_dict(raw_thresholds)
    maneuver_config = ManeuverConfig.from_dict(raw_thresholds)

    logger.info("Loading model %s on %s ...", cfg["checkpoint"], device)
    harvester = RolloutHarvester.load(checkpoint=cfg["checkpoint"], device=device)

    n_success = n_skipped = n_error = n_other_rank = n_scenes = 0

    with open(results_path, "a") as out_f:
        for event in iter_clip_events_from_manifest(cfg["manifest_path"], cfg["bucket"]):
            if cfg["max_scenes"] is not None and n_scenes >= int(cfg["max_scenes"]):
                logger.info("Reached max_scenes=%s, stopping.", cfg["max_scenes"])
                break
            n_scenes += 1

            # scene_id matches rollout_harvester.harvest_dataset's convention
            # exactly, so scene JSON files written by that standalone path
            # and JSONL rows written by this Lilypad path are cross-referenceable.
            scene_id = f"{event['clip_id']}_{event['t0_us']}"

            if _scene_owner(scene_id, world_size) != rank:
                n_other_rank += 1
                continue
            if scene_id in done:
                n_skipped += 1
                continue

            logger.info(
                "[rank %d] scene=%s cluster=%s", rank, scene_id, event.get("event_cluster"),
            )
            try:
                records = harvester.harvest_scene(
                    event["model_inputs"],
                    scene_id=scene_id,
                    k=int(cfg["k"]),
                    seed=int(cfg["seed"]),
                    top_p=float(cfg["top_p"]),
                    top_k=cfg["top_k"],
                    temperature=float(cfg["temperature"]),
                    ground_truth_coc=event.get("event_coc") or None,
                )

                written_classes: list[str] = []
                for record in records:
                    features = extract_features(
                        record.waypoints,
                        hz=record.hz,
                        scene_id=record.scene_id,
                        rollout_id=record.rollout_id,
                        config=feature_config,
                        native_accel_mps2=record.native_accel_mps2,
                    )
                    result = classify(features, maneuver_config)

                    # One flat row per rollout: raw rollout fields + kinematic
                    # features + the classified maneuver -- everything Task 6
                    # (pair miner, not built yet) will need is in one place,
                    # rather than split across the per-scene JSON files and a
                    # separate maneuver_labels.parquet the standalone CLI path
                    # produces.
                    row = record.to_json_dict()
                    row.update(features.to_row_dict())
                    row["maneuver_class"] = result.maneuver_class
                    row["boundary_margins"] = result.boundary_margins
                    row["event_cluster"] = event.get("event_cluster")
                    row["rank"] = rank
                    row["world_size"] = world_size

                    out_f.write(json.dumps(row) + "\n")
                    written_classes.append(result.maneuver_class)
                out_f.flush()
                n_success += 1
                logger.info("  wrote %d rollouts, classes=%s", len(records), written_classes)

            except Exception as exc:
                logger.error("scene %s: %s", scene_id, exc)
                traceback.print_exc()
                n_error += 1

    logger.info(
        "Done rank %d/%d: %d scenes succeeded  %d skipped  %d failed  %d owned by other ranks",
        rank, world_size, n_success, n_skipped, n_error, n_other_rank,
    )
    logger.info("Results: %s", results_path)
