# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint for the CoT-masking experiment.

Reads WebDataset shards from S3 (downloading to a local cache dir first),
processes each clip's snapshots through the masked Alpamayo 1.5 model, and
writes results to a JSONL file.  Compatible with the Lilypad training_fn
contract: accepts a flat config dict and an ExperimentTracker.

Experiments
-----------
experiment=a  Masked vs unmasked ADE across all clips (batch experiment A).
experiment=b  Per-word steering salience, top-20 ranked words (batch experiment B).
experiment=c  Prefix/suffix threshold sweep: ADE vs. CoT word position (batch experiment C).
experiment=d  Commitment/perceptual clause-order reversal via forced CoC decode (batch experiment D).

Resuming
--------
The JSONL output is append-only. Pass resume=true to skip (clip_id, t0_us) pairs
already present in this rank's output file.

Distributed (num_gpus > 1)
--------------------------
Lilypad sets RANK / WORLD_SIZE / LOCAL_RANK. Each GPU loads its own model copy,
processes snapshots where hash(clip_id, t0_us) % world_size == rank, and writes
batch_experiment_{experiment}_rank{RR}.jsonl. Rank 0 downloads WDS shards from S3;
other ranks poll until shards appear in local_data_dir.

Full config reference (all keys optional, defaults shown):
    local_data_dir:   "/tmp/wds_cache"   # directory containing shard_*.tar files
    skip_download:    false              # set true to use local_data_dir as-is (no S3)
    s3_bucket:        "PLACEHOLDER_BUCKET"
    s3_prefix:        "PLACEHOLDER_PREFIX/wds"
    max_shards:       null              # limit number of shards (for smoke tests)
    checkpoint:       "nvidia/Alpamayo-1.5-10B"
    experiment:       "a"              # "a" or "b"
    seed:             0
    concepts:         "pedestrian,person,cyclist,crosswalk,vehicle,stop,red,light"
    resume:           false
    outdir:           "/tmp/masking_results"
    wandb_project:    "masking-cot"
    wandb_entity:     "research"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Defaults used when a key is absent from training_fn_config
_DEFAULTS: dict[str, Any] = {
    "local_data_dir": "/tmp/wds_cache",
    "skip_download":  False,
    "s3_bucket":      "PLACEHOLDER_BUCKET",
    "s3_prefix":      "PLACEHOLDER_PREFIX/wds",
    "max_shards":     None,
    "checkpoint":     "nvidia/Alpamayo-1.5-10B",
    "experiment":     "a",
    "seed":           0,
    "concepts":       "pedestrian,person,cyclist,crosswalk,vehicle,stop,red,light",
    "resume":              False,
    "outdir":              "/tmp/masking_results",
    "wandb_project":       "masking-cot",
    "wandb_entity":        "research",
    "threshold_words":     [0, 5, 10, 20, 30, 50],  # prefix/suffix sweep thresholds (exp c)
    "rank":                0,
    "world_size":          1,
    "download_wait_seconds": 3600,
}


def _distributed_context(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from Lilypad env vars or config."""
    rank = int(os.environ.get("RANK", cfg.get("rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", cfg.get("world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def _shard_owner(clip_id: str, t0_us: int, world_size: int) -> int:
    """Stable shard assignment so resume/requeue always maps a snapshot to one rank."""
    digest = hashlib.md5(f"{clip_id}:{t0_us}".encode()).hexdigest()
    return int(digest, 16) % world_size


def _results_path(outdir: Path, experiment: str, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return outdir / f"batch_experiment_{experiment}.jsonl"
    return outdir / f"batch_experiment_{experiment}_rank{rank:02d}.jsonl"


def _load_done_rows(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                done.add((row["clip_id"], row["t0_us"]))
            except Exception:
                pass
    return done


def _acquire_shards(cfg: dict[str, Any], local_data: Path, rank: int) -> list[Path]:
    from masking.data.s3_download import download_shards, shard_paths

    if cfg["skip_download"]:
        shards = shard_paths(local_data)
        logger.info("skip_download=true: using %d existing shards in %s",
                    len(shards), local_data)
        return shards

    if rank == 0:
        logger.info("Downloading WDS shards from s3://%s/%s -> %s",
                    cfg["s3_bucket"], cfg["s3_prefix"], local_data)
        download_shards(
            bucket=cfg["s3_bucket"],
            prefix=cfg["s3_prefix"],
            local_dir=local_data,
            max_shards=cfg["max_shards"],
        )
        return shard_paths(local_data)

    wait_seconds = int(cfg["download_wait_seconds"])
    logger.info("rank %d waiting for rank 0 to download shards into %s", rank, local_data)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        shards = shard_paths(local_data)
        if shards:
            logger.info("rank %d found %d shards in %s", rank, len(shards), local_data)
            return shards
        time.sleep(10)

    raise RuntimeError(
        f"Timed out after {wait_seconds}s waiting for rank 0 to download shards into {local_data}"
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: str = "cuda"):
    """Load MaskedAlpamayo1_5 from a HuggingFace checkpoint."""
    from masking.bootstrap import ensure_alpamayo1_5

    ensure_alpamayo1_5()
    from masking.masked_model import MaskedAlpamayo1_5
    model = MaskedAlpamayo1_5.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model


def _resolve_device(local_rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def _build_inputs(model, item: dict, device: str = "cuda") -> dict:
    """Convert a WDS snapshot item to model inputs on the specified device."""
    from masking.bootstrap import ensure_alpamayo1_5

    ensure_alpamayo1_5()
    from alpamayo1_5 import helper

    data = item["model_inputs"]
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
        nav_text=None,
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


# ---------------------------------------------------------------------------
# Per-snapshot inference
# ---------------------------------------------------------------------------

def _run_experiment_a(model, model_inputs: dict, seed: int) -> dict:
    """Masked vs unmasked trajectory ADE for one snapshot."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = model.compare_conditions(
            model_inputs,
            {"none": {"mode": "none"}, "reasoning": {"mode": "reasoning"}},
            seed=seed,
        )

    cot_raw = res["cot"]
    if isinstance(cot_raw, dict):
        seqs = cot_raw.get("cot", [])
        cot = seqs[0].strip() if seqs else "<no cot>"
    else:
        cot = str(cot_raw).strip()

    none_xyz   = res["conditions"]["none"]["pred_xyz"][0, 0, 0].numpy()    # (T, 3)
    masked_xyz = res["conditions"]["reasoning"]["pred_xyz"][0, 0, 0].numpy()
    T          = min(len(none_xyz), len(masked_xyz))
    delta_xy   = np.linalg.norm(masked_xyz[:T, :2] - none_xyz[:T, :2], axis=-1)

    return {
        "cot":           cot,
        "n_masked_cols": int(res["conditions"]["reasoning"]["n_masked_cols"]),
        "ade_m":         float(delta_xy.mean()),
        "endpoint_m":    float(delta_xy[-1]),
        "cot_len_chars": len(cot),
        # Raw action-chunk waypoints (rounded to keep JSONL small) so a
        # dashboard can render the actual masked-vs-unmasked path, not just
        # the scalar ADE/endpoint summary.
        "traj_none_xy":   none_xyz[:T, :2].round(4).tolist(),
        "traj_masked_xy": masked_xyz[:T, :2].round(4).tolist(),
    }


def _run_experiment_b(model, model_inputs: dict, seed: int, concepts: list[str]) -> dict:
    """Per-word steering salience + concept-set ablation for one snapshot."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sal  = model.salience_leave_one_word_out(model_inputs, seed=seed)
        conc = model.compare_conditions(
            model_inputs,
            {"none": {"mode": "none"}, "concept": {"mode": "concept", "concepts": concepts}},
            seed=seed,
        )

    none_ctrl = conc["conditions"]["none"]["controls"]
    conc_ctrl = conc["conditions"]["concept"]["controls"]

    bc = none_ctrl["curvature"][0].numpy()
    cc = conc_ctrl["curvature"][0].numpy()
    ba = none_ctrl["accel"][0].numpy()
    ca = conc_ctrl["accel"][0].numpy()

    cot_raw = sal.get("cot", "")
    if isinstance(cot_raw, dict):
        seqs = cot_raw.get("cot", [])
        cot  = seqs[0].strip() if seqs else "<no cot>"
    else:
        cot = str(cot_raw).strip()

    return {
        "cot":                      cot,
        "concepts_ablated":         concepts,
        "n_concept_cols_masked":    int(conc["conditions"]["concept"]["n_masked_cols"]),
        "concept_d_curvature_mean": float(np.abs(cc - bc).mean()),
        "concept_d_curvature_max":  float(np.abs(cc - bc).max()),
        "concept_d_accel_mean":     float(np.abs(ca - ba).mean()),
        # Baseline path + per-word masked path (traj_xy) so a caller can
        # render "click a word, see the trajectory shift" instead of only
        # the scalar salience metrics.
        "traj_baseline_xy":         sal["baseline_xy"],
        "per_word_salience_top20":  sal["words"][:20],
    }


def _run_experiment_c(
    model, model_inputs: dict, seed: int, threshold_words: list[int]
) -> dict:
    """Prefix/suffix threshold sweep: ADE vs. CoT word position.

    For each threshold n, we run two conditions:
      prefix n  — expert sees only the first n words; words[n:] are masked
      suffix n  — expert sees only words from position n onward; words[:n] are masked

    A baseline (no masking) is included at key "baseline". All conditions share
    a single VLM rollout so the CoT text is identical across thresholds.

    Output fields:
      cot              — generated reasoning text
      n_words_total    — total whole-word count in the reasoning span
      traj_baseline_xy — (T, 2) unmasked waypoint path every branch is compared against
      prefix_sweep     — list of {n, n_masked_cols, ade_m, traj_xy} sorted by n
      suffix_sweep     — list of {n, n_masked_cols, ade_m, traj_xy} sorted by n
    """
    conditions: dict[str, dict] = {"baseline": {"mode": "none"}}
    for n in threshold_words:
        conditions[f"prefix_{n}w"] = {"mode": "prefix", "n": n, "unit": "words"}
        conditions[f"suffix_{n}w"] = {"mode": "suffix", "n": n, "unit": "words"}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = model.compare_conditions(model_inputs, conditions, seed=seed)

    cot_raw = res["cot"]
    if isinstance(cot_raw, dict):
        seqs = cot_raw.get("cot", [])
        cot = seqs[0].strip() if seqs else "<no cot>"
    else:
        cot = str(cot_raw).strip()

    n_words_total = len(res.get("words", []))

    baseline_xyz = res["conditions"]["baseline"]["pred_xyz"][0, 0, 0].numpy()  # (T, 3)

    def branch_vs_baseline(cond_name: str) -> tuple[float, list[list[float]]]:
        xyz = res["conditions"][cond_name]["pred_xyz"][0, 0, 0].numpy()
        T = min(len(xyz), len(baseline_xyz))
        ade = float(np.linalg.norm(xyz[:T, :2] - baseline_xyz[:T, :2], axis=-1).mean())
        return ade, xyz[:T, :2].round(4).tolist()

    prefix_sweep = []
    for n in sorted(threshold_words):
        ade, traj_xy = branch_vs_baseline(f"prefix_{n}w")
        prefix_sweep.append({
            "n": n,
            "n_masked_cols": int(res["conditions"][f"prefix_{n}w"]["n_masked_cols"]),
            "ade_m": ade,
            "traj_xy": traj_xy,
        })
    suffix_sweep = []
    for n in sorted(threshold_words):
        ade, traj_xy = branch_vs_baseline(f"suffix_{n}w")
        suffix_sweep.append({
            "n": n,
            "n_masked_cols": int(res["conditions"][f"suffix_{n}w"]["n_masked_cols"]),
            "ade_m": ade,
            "traj_xy": traj_xy,
        })

    return {
        "cot":            cot,
        "n_words_total":  n_words_total,
        # Root of the tree: the unmasked full-reasoning trajectory every
        # prefix/suffix branch's traj_xy is compared against.
        "traj_baseline_xy": baseline_xyz[:, :2].round(4).tolist(),
        "prefix_sweep":   prefix_sweep,
        "suffix_sweep":   suffix_sweep,
    }


# ---------------------------------------------------------------------------
# Lilypad entrypoint
# ---------------------------------------------------------------------------

def masking_loop(
    training_fn_config: dict[str, Any],
    experiment_tracker: Any,
) -> None:
    """Lilypad-compatible entrypoint.

    Downloads WDS shards from S3, iterates over all snapshots, runs the
    chosen masking experiment, and writes results to a JSONL file.
    """
    cfg = {**_DEFAULTS, **training_fn_config}

    rank, world_size, local_rank = _distributed_context(cfg)
    device = _resolve_device(local_rank)

    outdir     = Path(cfg["outdir"])
    local_data = Path(cfg["local_data_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    local_data.mkdir(parents=True, exist_ok=True)

    experiment       = str(cfg["experiment"]).lower()
    seed             = int(cfg["seed"])
    concepts         = [c.strip() for c in str(cfg["concepts"]).split(",") if c.strip()]
    threshold_words  = list(cfg["threshold_words"])

    results_path = _results_path(outdir, experiment, rank, world_size)
    logger.info("Distributed context: rank=%d world_size=%d local_rank=%d device=%s",
                rank, world_size, local_rank, device)
    logger.info("Writing results to %s", results_path)

    # ── 1. Acquire shards (rank 0 downloads; other ranks wait) ───────────────
    shards = _acquire_shards(cfg, local_data, rank)

    if cfg["max_shards"] is not None:
        shards = shards[:int(cfg["max_shards"])]
        logger.info("max_shards=%d: using %d shards", cfg["max_shards"], len(shards))

    if not shards:
        raise RuntimeError(
            f"No shards found in {local_data}. "
            f"Check s3_bucket={cfg['s3_bucket']} and s3_prefix={cfg['s3_prefix']}."
        )

    # ── 2. Load already-done rows if resuming ────────────────────────────────
    done: set[tuple[str, int]] = set()
    if cfg["resume"]:
        done = _load_done_rows(results_path)
        logger.info("Resuming: %d events already done on rank %d", len(done), rank)

    # ── 3. Load model once on this GPU ───────────────────────────────────────
    logger.info("Loading model %s on %s …", cfg["checkpoint"], device)
    model = _load_model(cfg["checkpoint"], device=device)

    # ── 4. Iterate WDS snapshots assigned to this rank ───────────────────────
    from masking.data.wds_dataset import iter_snapshots

    n_success = 0
    n_skipped = 0
    n_error   = 0
    n_other_rank = 0

    with open(results_path, "a") as out_f:
        for item in iter_snapshots(shards):
            clip_id = item["clip_id"]
            t0_us   = item["t0_us"]

            if _shard_owner(clip_id, t0_us, world_size) != rank:
                n_other_rank += 1
                continue

            if (clip_id, t0_us) in done:
                n_skipped += 1
                continue

            logger.info("[rank %d] clip=%.8s t0=%.2fs  cluster=%s",
                        rank, clip_id, t0_us / 1e6, item["event_cluster"])
            try:
                model_inputs = _build_inputs(model, item, device=device)

                if experiment == "a":
                    result = _run_experiment_a(model, model_inputs, seed)
                elif experiment == "b":
                    result = _run_experiment_b(model, model_inputs, seed, concepts)
                elif experiment == "c":
                    result = _run_experiment_c(model, model_inputs, seed, threshold_words)
                elif experiment == "d":
                    from masking.training.experiment_d_reversal import run_experiment_d
                    result = run_experiment_d(model, model_inputs, seed)
                else:
                    raise ValueError(f"Unknown experiment: {experiment!r}")

                result.update({
                    "clip_id":       clip_id,
                    "t0_us":         t0_us,
                    "rank":          rank,
                    "world_size":    world_size,
                    "event_cluster": item["event_cluster"],
                    "group":         item["group"],
                    "event_idx":     item["event_idx"],
                    "was_clamped":   item["was_clamped"],
                    "event_coc":     item["event_coc"],
                })

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_success += 1

                if experiment == "c":
                    sweep_adel = [r["ade_m"] for r in result.get("prefix_sweep", [])]
                    logger.info("  n_words=%d  prefix_ade_range=[%.4f, %.4f]",
                                result.get("n_words_total", 0),
                                min(sweep_adel, default=0.0), max(sweep_adel, default=0.0))
                elif experiment == "d":
                    logger.info("  beats_reversed=%d/%d  ade_control=%.4f  ade_reversal=%s",
                                result.get("n_beats_reversed", 0), result.get("n_beats", 0),
                                result.get("ade_control_m", 0.0),
                                ("%.4f" % result["ade_reversal_m"])
                                if result.get("ade_reversal_m") is not None else "n/a")
                else:
                    logger.info("  ade_m=%.4f  n_masked=%d",
                                result.get("ade_m", 0.0),
                                result.get("n_masked_cols", result.get("n_concept_cols_masked", 0)))

            except Exception as exc:
                logger.error("clip %s t0=%d: %s", clip_id, t0_us, exc)
                traceback.print_exc()
                n_error += 1

    logger.info(
        "Done rank %d/%d: %d succeeded  %d skipped  %d failed  %d owned by other ranks",
        rank, world_size, n_success, n_skipped, n_error, n_other_rank,
    )
    logger.info("Results: %s", results_path)
