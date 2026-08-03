# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint for the dpo_pairs counterfactual MEASUREMENT run:
for every scene that has perturbations in the corpus, force each perturbed
CoC into Alpamayo 1.5 and sample N autoregressive trajectory-token
continuations per condition (see ar_forced_rollout.py for why the AR pathway
and not the diffusion expert).

Conditions per scene (all share seed_start; chunk c of every condition is
seeded seed_start + c — common-random-numbers at chunk granularity):

  control_rawids       forced text = None (the template's own reasoning ids
                       spliced verbatim) — this pathway's forced_orig-style
                       machinery control: ADE(clean, control_rawids) is the
                       re-tokenization error floor, NOT a perturbation effect.
  clean                forced text = the scene's archived ground-truth trace
                       (chosen-side source AND the per-scene AR noise floor).
  perturbed__<type>    forced text = each perturbation's perturbed_trace
                       (rejected-side source), one condition per perturbation
                       row the corpus has for this scene.

Structural clone of counterfactual/run.py::counterfactual_sweep_loop — same
manifest iteration, md5 scene→rank sharding, resume markers, bf16 autocast
discipline, and the same three-channel result delivery (all of BUGS.md's
"outdir is node-local and dies with the pod" lesson):
  1. per-rank append-only JSONL under outdir (resume tracking + local debug),
  2. one `DPO_MEASURE <json>` log line per (scene, condition) row — the
     lilypad-logs retrieval path (dpo_pairs/fetch_from_logs.py),
  3. if results_s3_prefix is set: one JSON per scene put_object'd to S3
     (put_object with an in-memory body, NEVER upload_file — OCI's S3-compat
     endpoint rejects s3transfer's chunked UploadPart, see
     masking/training/run.py::_upload_results).

Optional diffusion cross-check (diffusion_crosscheck_allowlist): for a small
scene subset, ALSO force clean + each perturbed text through the diffusion
expert (experiment D's rollout_forced_cot + _denoise_with_mask, 1 seed) and
log per-perturbation ADE between them — quantifies whether the AR head and
the diffusion expert agree on which perturbations matter (plan risk: an
effect present in one pathway may not exist in the other).

Full config reference (defaults shown):
    manifest_path:       "pref_pairs/configs/sample_clips_n100_unstratified.json"
    bucket:              "research-datasets-chicago"
    checkpoint:          "nvidia/Alpamayo-1.5-10B"
    perturbations_path:  "pref_pairs/results/perturbations/perturbations.jsonl"
    max_scenes:          1        # null/omit to cover every scene with perturbations
    n_samples:           20       # AR trajectory samples per condition
    sample_batch_size:   5        # samples per generate call (KV-memory bound)
    seed_start:          0
    template_seed:       0        # seeds the one template generation per scene
    temperature:         0.6      # matches rl_posttrain's rollout sampling params
    top_p:               0.98     #   (alpamayo_rvla_rl_code_reward.toml [rollout])
    top_k:               null
    outdir:              "/mnt/work/tmp/dpo_pairs_measure"
    results_s3_prefix:   null     # e.g. "dpo_pairs/measure/run1" -> per-scene JSON on S3
    resume:              false
    diffusion_crosscheck_allowlist: null   # list of scene_ids, or null to disable
    rank / world_size:   auto-set by Lilypad via RANK/WORLD_SIZE env vars
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

MEASURE_LOG_MARKER = "DPO_MEASURE "

_DEFAULTS: dict[str, Any] = {
    "manifest_path": "pref_pairs/configs/sample_clips_n100_unstratified.json",
    "bucket": "research-datasets-chicago",
    "checkpoint": "nvidia/Alpamayo-1.5-10B",
    "perturbations_path": "pref_pairs/results/perturbations/perturbations.jsonl",
    "max_scenes": 1,
    "n_samples": 20,
    "sample_batch_size": 5,
    "seed_start": 0,
    "template_seed": 0,
    "temperature": 0.6,
    "top_p": 0.98,
    "top_k": None,
    "outdir": "/mnt/work/tmp/dpo_pairs_measure",
    "results_s3_prefix": None,
    "resume": False,
    "diffusion_crosscheck_allowlist": None,
    "rank": 0,
    "world_size": 1,
}


def _distributed_context(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Same pattern as counterfactual/run.py — see its docstring for why this
    is reimplemented per-experiment rather than shared."""
    rank = int(os.environ.get("RANK", cfg.get("rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", cfg.get("world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def _scene_owner(scene_id: str, world_size: int) -> int:
    digest = hashlib.md5(scene_id.encode()).hexdigest()
    return int(digest, 16) % world_size


def _results_path(outdir: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return outdir / "dpo_measure_rows.jsonl"
    return outdir / f"dpo_measure_rows_rank{rank:02d}.jsonl"


def _load_done_scenes(path: Path) -> set[str]:
    """Scenes already FULLY measured on this rank. Only `scene_done` marker
    rows count — per-condition rows are written incrementally, so their
    presence alone doesn't imply the scene finished (same atomicity
    convention as counterfactual/run.py's resume markers)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                if row.get("kind") == "scene_done":
                    done.add(row["scene_id"])
            except Exception:
                pass
    return done


def load_perturbations_by_scene(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Index a perturbations JSONL (v1 pref_pairs schema or v2 superset) by
    scene_id. Every row must carry scene_id / perturbation_type /
    perturbed_trace / ground_truth_trace; rows missing any of those are
    skipped with a warning rather than aborting the run (graceful-skip
    convention, same as fetch_from_logs.parse_marked_lines)."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    required = ("scene_id", "perturbation_type", "perturbed_trace", "ground_truth_trace")
    n_bad = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if any(k not in row for k in required):
                n_bad += 1
                continue
            by_scene.setdefault(row["scene_id"], []).append(row)
    if n_bad:
        logger.warning("%s: skipped %d malformed/incomplete perturbation rows", path, n_bad)
    return by_scene


def _resolve_device(local_rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def _load_model(checkpoint: str, device: str):
    """MaskedAlpamayo1_5, not plain Alpamayo1_5: ar_forced_rollout needs its
    _reasoning_span/_cot_special_ids (splice_reasoning's contract), and the
    optional diffusion cross-check needs _rollout_prefix/_denoise_with_mask —
    same weights either way. attn_implementation="sdpa" is REQUIRED (the
    default flash_attention_2 is not installed — masking/training/run.py)."""
    from masking.bootstrap import ensure_alpamayo1_5

    ensure_alpamayo1_5()
    from masking.masked_model import MaskedAlpamayo1_5

    model = MaskedAlpamayo1_5.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model


def _emit_row(row: dict[str, Any], results_path: Path) -> None:
    """Channel 1 + 2: local JSONL append and the marked log line."""
    payload = json.dumps(row)
    with open(results_path, "a") as fh:
        fh.write(payload + "\n")
    logger.info(MEASURE_LOG_MARKER + "%s", payload)


def _upload_scene_json(
    scene_id: str, rows: list[dict[str, Any]], bucket: str, s3_prefix: str, rank: int,
) -> None:
    """Channel 3: one JSON per scene on S3. put_object with an in-memory
    body — see module docstring for why never upload_file here."""
    import boto3

    s3 = boto3.client("s3")
    key = f"{s3_prefix.rstrip('/')}/rank{rank:02d}/{scene_id}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(rows).encode())


def _diffusion_crosscheck(
    model, data: dict[str, Any], template: dict[str, Any],
    clean_text: str, perturbations: list[dict[str, Any]], seed: int,
) -> dict[str, Any]:
    """Force clean + each perturbed text through the DIFFUSION expert (1 seed
    each, common random numbers) and return per-perturbation ADE against the
    clean diffusion trajectory. Reuses experiment D's forced-decode machinery
    verbatim; runs only for allowlisted scenes (each text costs a full
    re-prefill + denoise)."""
    import numpy as np

    from masking.training.experiment_d_reversal import rollout_forced_cot, splice_reasoning

    def _forced_xy(text: str | None) -> "np.ndarray":
        forced_seq = splice_reasoning(model, template["seq0"], text)
        prefix = rollout_forced_cot(model, data, forced_seq)
        xyz, _, _ = model._denoise_with_mask(prefix, None, seed=seed)
        return xyz[0, 0, 0].float().cpu().numpy()[:, :2]

    clean_xy = _forced_xy(clean_text)
    out: dict[str, Any] = {"ade_diffusion_m": {}, "clean_xy": clean_xy.round(4).tolist()}
    for pert in perturbations:
        pert_xy = _forced_xy(pert["perturbed_trace"])
        t = min(len(clean_xy), len(pert_xy))
        ade = float(np.linalg.norm(clean_xy[:t] - pert_xy[:t], axis=-1).mean())
        out["ade_diffusion_m"][pert["perturbation_type"]] = ade
    return out


def dpo_measure_loop(training_fn_config: dict[str, Any], experiment_tracker: Any) -> None:
    """Lilypad-compatible entrypoint. See module docstring for the config
    reference and the three result-delivery channels."""
    cfg = {**_DEFAULTS, **training_fn_config}
    rank, world_size, local_rank = _distributed_context(cfg)
    device = _resolve_device(local_rank)
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Distributed context: rank=%d world_size=%d local_rank=%d device=%s",
                rank, world_size, local_rank, device)

    from masking.data.wds_dataset import iter_clip_events_from_manifest
    from pref_pairs.rollout_harvester import build_tokenized_inputs

    from dpo_pairs.ar_forced_rollout import build_template_sequence, sample_traj_tokens_given_coc

    by_scene = load_perturbations_by_scene(cfg["perturbations_path"])
    logger.info("perturbation corpus: %d scenes, %d rows (%s)",
                len(by_scene), sum(len(v) for v in by_scene.values()), cfg["perturbations_path"])

    logger.info("Loading model %s on %s ...", cfg["checkpoint"], device)
    model = _load_model(cfg["checkpoint"], device=device)

    n_samples = int(cfg["n_samples"])
    sample_batch_size = int(cfg["sample_batch_size"])
    seed_start = int(cfg["seed_start"])
    template_seed = int(cfg["template_seed"])
    max_scenes = cfg["max_scenes"]
    sampling_kwargs = {
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "top_k": cfg["top_k"],
    }
    crosscheck_allowlist = (
        set(cfg["diffusion_crosscheck_allowlist"])
        if cfg["diffusion_crosscheck_allowlist"] else set()
    )
    sampling_params = {
        **sampling_kwargs, "n_samples": n_samples,
        "sample_batch_size": sample_batch_size, "seed_start": seed_start,
        "template_seed": template_seed, "pathway": "ar_token",
    }

    results_path = _results_path(outdir, rank, world_size)
    done_scenes: set[str] = set()
    if cfg["resume"]:
        done_scenes = _load_done_scenes(results_path)
        logger.info("Resuming: %d scene(s) already done on rank %d", len(done_scenes), rank)

    n_done = 0
    n_skipped_no_pert = 0
    n_skipped_other_rank = 0
    n_skipped_resume = 0
    for event in iter_clip_events_from_manifest(cfg["manifest_path"], cfg["bucket"]):
        if max_scenes is not None and n_done >= int(max_scenes):
            logger.info("Reached max_scenes=%s, stopping.", max_scenes)
            break

        scene_id = f"{event['clip_id']}_{event['t0_us']}"
        perturbations = by_scene.get(scene_id)
        if not perturbations:
            n_skipped_no_pert += 1
            continue
        if _scene_owner(scene_id, world_size) != rank:
            n_skipped_other_rank += 1
            continue
        if scene_id in done_scenes:
            n_skipped_resume += 1
            continue

        clean_text = perturbations[0]["ground_truth_trace"]
        logger.info("=== scene %s: %d perturbations (rank %d, %d done so far) ===",
                    scene_id, len(perturbations), rank, n_done)

        data = build_tokenized_inputs(model, event["model_inputs"], device)

        # (condition, forced text, perturbation row or None). clean forces
        # the archived TEXT — not the raw ids — so chosen/rejected tokenize
        # through the identical path; control_rawids isolates exactly that
        # re-tokenization machinery error (see ar_forced_rollout docstring).
        conditions: list[tuple[str, str | None, dict[str, Any] | None]] = [
            ("control_rawids", None, None),
            ("clean", clean_text, None),
        ] + [
            (f"perturbed__{p['perturbation_type']}", p["perturbed_trace"], p)
            for p in perturbations
        ]

        scene_rows: list[dict[str, Any]] = []
        base = {
            "scene_id": scene_id,
            "event_cluster": event.get("event_cluster"),
            "clip_id": event["clip_id"],
            "t0_us": event["t0_us"],
            "sampling_params": sampling_params,
            "model_version": cfg["checkpoint"],
            "rank": rank,
            "world_size": world_size,
        }
        with torch.autocast(device, dtype=torch.bfloat16):
            # Wrapping everything model-touching in autocast(bfloat16) is the
            # caller's job in this codebase — the missing wrapper was a real
            # crash on counterfactual/run.py's first smoke run.
            template = build_template_sequence(
                model, data, seed=template_seed, **sampling_kwargs,
            )

            for condition, text, pert in conditions:
                result = sample_traj_tokens_given_coc(
                    model, template, text,
                    n_samples=n_samples, seed_start=seed_start,
                    sample_batch_size=sample_batch_size, **sampling_kwargs,
                )
                row = {
                    **base,
                    "kind": "condition",
                    "condition": condition,
                    "perturbation_type": pert["perturbation_type"] if pert else None,
                    "trace_id": pert.get("trace_id") if pert else None,
                    "coc_text": text if text is not None else template["self_generated_coc"],
                    "coc_token_ids": result["coc_token_ids"],
                    "forced_len": result["forced_len"],
                    "samples": result["samples"],
                    "self_generated_coc": template["self_generated_coc"],
                    "prompt_len": template["prompt_len"],
                }
                _emit_row(row, results_path)
                scene_rows.append(row)

            if scene_id in crosscheck_allowlist:
                logger.info("diffusion cross-check for %s ...", scene_id)
                check = _diffusion_crosscheck(
                    model, data, template, clean_text, perturbations, seed=seed_start,
                )
                row = {**base, "kind": "diffusion_crosscheck", **check}
                _emit_row(row, results_path)
                scene_rows.append(row)

        if cfg["results_s3_prefix"]:
            try:
                _upload_scene_json(
                    scene_id, scene_rows, cfg["bucket"], cfg["results_s3_prefix"], rank,
                )
            except Exception:
                # The log-marker channel already has every row — an S3 hiccup
                # must not kill a multi-hour sweep.
                logger.exception("S3 upload failed for %s (rows still in logs)", scene_id)

        # scene_done written only after every condition (and the optional
        # cross-check + S3 upload attempt) succeeded — a scene marked done is
        # genuinely done, never partially processed.
        with open(results_path, "a") as fh:
            fh.write(json.dumps({"kind": "scene_done", "scene_id": scene_id}) + "\n")
        n_done += 1

    logger.info(
        "Done rank %d/%d: %d scene(s) measured, %d skipped (resume), "
        "%d skipped (other rank), %d skipped (no perturbations).",
        rank, world_size, n_done, n_skipped_resume, n_skipped_other_rank, n_skipped_no_pert,
    )
