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
    max_batch_size:     null
    log_scene_summary:  false
    log_detailed_scenes: 0
    wandb_project:       "pref-pairs"
    wandb_entity:        "research"

RESULT RETRIEVAL -- log_scene_summary / log_detailed_scenes:
    outdir is a plain local path on whichever machine the Lilypad job
    actually runs on. When that path isn't reachable from wherever you'd
    read results afterward (e.g. the job ran in a different region/cluster
    than your workstation's storage mount, even if the path LOOKS the
    same -- confirmed to happen with /mnt/work between this project's
    workstation and its us-chicago-1 Lilypad cluster), the outdir file is
    simply unreachable, full stop. These two flags are an alternate,
    opt-in retrieval path THROUGH the log stream instead:
      - log_scene_summary=true: after each scene, additionally logs one
        compact JSON line (marked with SCENE_SUMMARY_LOG_MARKER) --
        action_space_variance.per_clip_variance's summary row for that
        scene, with the bulky per-waypoint arrays stripped out. Safe to
        enable for every scene in a large job; each line is small.
      - log_detailed_scenes=N: for the first N scenes per rank, ALSO logs
        one JSON line per rollout (marked with ROLLOUT_FULL_LOG_MARKER)
        containing the COMPLETE row -- waypoints, full coc_text, native
        action, features. Deliberately bounded (not "every scene") since
        this is much larger per scene; N is a per-rank cap so total volume
        stays proportional to N x world_size, not to the manifest size.
    pref_pairs/fetch_from_logs.py reconstructs local DataFrames from
    these markers (via `lilypad workload logs --content-filter`) and feeds
    them straight into the existing action_space_variance /
    scene_reasoning_report report builders -- no new report logic, just an
    alternate data source. Both default to off/0, so existing runs that
    don't need this (outdir already reachable) are unaffected.
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
    # None = single batched call (original behavior). Set this (e.g. 20,
    # the largest batch this pipeline has run in production) when k is
    # large enough to CUDA-OOM a single call -- see rollout_harvester.py's
    # harvest_scene docstring for why 20 is the recommended value.
    "max_batch_size": None,
    # See module docstring's "RESULT RETRIEVAL" section.
    "log_scene_summary": False,
    "log_detailed_scenes": 0,
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


# See module docstring's "RESULT RETRIEVAL" section and fetch_from_logs.py,
# which greps for these exact markers.
SCENE_SUMMARY_LOG_MARKER = "PREF_PAIRS_SCENE_SUMMARY"
ROLLOUT_FULL_LOG_MARKER = "PREF_PAIRS_ROLLOUT_FULL"

# action_space_variance.per_clip_variance's per-waypoint arrays (length T
# each) -- dropped from the scene-summary log line since the per-cluster
# report only ever consumes the scalar _std_mean_/_range_max_ summaries
# derived from them, not the raw per-waypoint detail. Keeping hundreds of
# scene-summary log lines small matters when the log stream is the ONLY
# retrieval path (see RESULT RETRIEVAL).
_SUMMARY_LOG_DROP_FIELDS = (
    "accel_per_waypoint_mean", "accel_per_waypoint_std",
    "accel_per_waypoint_min", "accel_per_waypoint_max",
    "curvature_per_waypoint_mean", "curvature_per_waypoint_std",
    "curvature_per_waypoint_min", "curvature_per_waypoint_max",
)


def _build_scene_summary_log_line(scene_rows: list[dict[str, Any]], expected_k: int | None = None) -> str:
    """One line summarizing a single scene's cross-rollout variance/range,
    via action_space_variance.per_clip_variance (scene_rows all share one
    scene_id, so that function's groupby produces exactly one row here).
    Pure function -- pandas/action_space_variance imported lazily inside,
    same reasoning as pref_pairs_loop's own lazy imports, so this stays
    testable without a GPU or the model.
    """
    import pandas as pd

    from pref_pairs.action_space_variance import per_clip_variance

    summary = per_clip_variance(pd.DataFrame(scene_rows), expected_k=expected_k).iloc[0].to_dict()
    for field in _SUMMARY_LOG_DROP_FIELDS:
        summary.pop(field, None)
    return f"{SCENE_SUMMARY_LOG_MARKER} {json.dumps(summary)}"


def _build_detailed_log_lines(scene_rows: list[dict[str, Any]]) -> list[str]:
    """One log line per rollout with the row verbatim -- full waypoints,
    full coc_text, native action, features. Only ever called for a small,
    bounded number of scenes (pref_pairs_loop's log_detailed_scenes cap)."""
    return [f"{ROLLOUT_FULL_LOG_MARKER} {json.dumps(row)}" for row in scene_rows]


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
    n_detailed_logged = 0

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
                    max_batch_size=cfg["max_batch_size"],
                )

                scene_rows: list[dict[str, Any]] = []
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
                    scene_rows.append(row)

                for row in scene_rows:
                    out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_success += 1
                logger.info(
                    "  wrote %d rollouts, classes=%s",
                    len(scene_rows), [r["maneuver_class"] for r in scene_rows],
                )

                # Optional: ALSO surface results through the log stream, not
                # just the outdir file -- see module docstring's "RESULT
                # RETRIEVAL" note. Needed when outdir isn't reachable from
                # wherever fetch_from_logs.py runs (e.g. a different-region
                # storage mount than the one submitting the job).
                if cfg["log_scene_summary"]:
                    logger.info(_build_scene_summary_log_line(scene_rows, expected_k=int(cfg["k"])))
                if n_detailed_logged < int(cfg["log_detailed_scenes"]):
                    for line in _build_detailed_log_lines(scene_rows):
                        logger.info(line)
                    n_detailed_logged += 1

            except Exception as exc:
                logger.error("scene %s: %s", scene_id, exc)
                traceback.print_exc()
                n_error += 1

    logger.info(
        "Done rank %d/%d: %d scenes succeeded  %d skipped  %d failed  %d owned by other ranks",
        rank, world_size, n_success, n_skipped, n_error, n_other_rank,
    )
    logger.info("Results: %s", results_path)


# Fixed-reasoning / diffusion-only mode (pref_pairs.fixed_reasoning_rollout):
# freeze one CoC reasoning per scene, draw num_draws diffusion-only samples
# against it. Same manifest-iteration / sharding / resume / logging shape as
# pref_pairs_loop above -- only the harvesting call differs -- so this reuses
# every module-level helper above unchanged. Separate defaults dict rather
# than folding into _DEFAULTS: num_draws/seed_start replace k/seed/
# max_batch_size, which don't apply to this mode (nothing is batched, see
# fixed_reasoning_rollout.py's module docstring).
_FIXED_REASONING_DEFAULTS: dict[str, Any] = {
    "manifest_path": "pref_pairs/configs/sample_clips_all.json",
    "bucket": "research-datasets-chicago",
    "checkpoint": "nvidia/Alpamayo-1.5-10B",
    "num_draws": 100,
    "seed_start": 0,
    "temperature": 0.6,
    "top_p": 0.98,
    "top_k": None,
    "thresholds_config": "pref_pairs/configs/maneuver_thresholds.yaml",
    "resume": False,
    "outdir": "/tmp/fixed_reasoning_results",
    "max_scenes": None,
    "log_scene_summary": False,
    "log_detailed_scenes": 0,
    "wandb_project": "pref-pairs-fixed-reasoning",
    "wandb_entity": "research",
    "rank": 0,
    "world_size": 1,
}


def fixed_reasoning_loop(training_fn_config: dict[str, Any], experiment_tracker: Any) -> None:
    """Lilypad-compatible entrypoint for the fixed-reasoning/diffusion-only
    mode. See _FIXED_REASONING_DEFAULTS above and
    pref_pairs.fixed_reasoning_rollout's module docstring for the full
    config reference and design rationale. Structurally identical to
    pref_pairs_loop -- manifest iteration, rank/resume gating, row
    assembly, logging -- differing only in which harvester is loaded and
    called; see that function for inline comments on the shared logic."""
    cfg = {**_FIXED_REASONING_DEFAULTS, **training_fn_config}

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

    from masking.data.wds_dataset import iter_clip_events_from_manifest

    from pref_pairs.classify_maneuvers import FeatureConfig, ManeuverConfig, classify
    from pref_pairs.fixed_reasoning_rollout import FixedReasoningHarvester
    from pref_pairs.trajectory_features import extract_features

    with open(cfg["thresholds_config"]) as fh:
        raw_thresholds = yaml.safe_load(fh)
    feature_config = FeatureConfig.from_dict(raw_thresholds)
    maneuver_config = ManeuverConfig.from_dict(raw_thresholds)

    logger.info("Loading model %s on %s ...", cfg["checkpoint"], device)
    harvester = FixedReasoningHarvester.load(checkpoint=cfg["checkpoint"], device=device)

    n_success = n_skipped = n_error = n_other_rank = n_scenes = 0
    n_detailed_logged = 0

    with open(results_path, "a") as out_f:
        for event in iter_clip_events_from_manifest(cfg["manifest_path"], cfg["bucket"]):
            if cfg["max_scenes"] is not None and n_scenes >= int(cfg["max_scenes"]):
                logger.info("Reached max_scenes=%s, stopping.", cfg["max_scenes"])
                break
            n_scenes += 1

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
                    num_draws=int(cfg["num_draws"]),
                    seed_start=int(cfg["seed_start"]),
                    top_p=float(cfg["top_p"]),
                    top_k=cfg["top_k"],
                    temperature=float(cfg["temperature"]),
                    ground_truth_coc=event.get("event_coc") or None,
                )

                scene_rows: list[dict[str, Any]] = []
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

                    row = record.to_json_dict()
                    row.update(features.to_row_dict())
                    row["maneuver_class"] = result.maneuver_class
                    row["boundary_margins"] = result.boundary_margins
                    row["event_cluster"] = event.get("event_cluster")
                    row["rank"] = rank
                    row["world_size"] = world_size
                    scene_rows.append(row)

                for row in scene_rows:
                    out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_success += 1
                logger.info(
                    "  wrote %d rollouts, classes=%s",
                    len(scene_rows), [r["maneuver_class"] for r in scene_rows],
                )

                if cfg["log_scene_summary"]:
                    logger.info(_build_scene_summary_log_line(scene_rows, expected_k=int(cfg["num_draws"])))
                if n_detailed_logged < int(cfg["log_detailed_scenes"]):
                    for line in _build_detailed_log_lines(scene_rows):
                        logger.info(line)
                    n_detailed_logged += 1

            except Exception as exc:
                logger.error("scene %s: %s", scene_id, exc)
                traceback.print_exc()
                n_error += 1

    logger.info(
        "Done rank %d/%d: %d scenes succeeded  %d skipped  %d failed  %d owned by other ranks",
        rank, world_size, n_success, n_skipped, n_error, n_other_rank,
    )
    logger.info("Results: %s", results_path)
