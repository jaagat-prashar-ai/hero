# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint for a SMOKE TEST of counterfactual.py's
CounterfactualTokenAnalyzer, which had never been run against the real model
(no tests, no prior cluster run, no results) before this file existed.

For each scene in a clip manifest, this runs all three of that module's
analyses on the SAME sampled reasoning trace (seeded so the baseline
generation is byte-identical across all three calls, not three independent
resamples):
  1. token_alternative_map  — the model's own logit distribution at every
     generated reasoning token: the top-K alternatives it almost said
     instead, ranked purely by probability.
  2. single_token_swap_sweep (Option A) — isolated single-token swap + cheap
     KV-cache re-forward (rest of the trace stays byte-identical).
  3. counterfactual_sweep (Option B) — forces the alternative token during
     generation so the rest of the reasoning coherently re-samples around it
     (more realistic, less isolated than Option A).

position_selection="all" for both sweeps: every reasoning token position is
tested (not filtered down to a heuristic subset like "entropy" or
"prob_gap"), per an explicit request to cover every position's top-K
alternatives ranked by logit score, not a curated subset of positions.

Reuses pref_pairs.rollout_harvester's manifest-driven scene iteration
(iter_clip_events_from_manifest + build_tokenized_inputs) rather than
masking.training.run's shard-download path. That path calls
iter_snapshots(shards), which no longer exists anywhere in
masking/data/wds_dataset.py (see pref_pairs/training/run.py's own docstring
for the same finding, made independently in an earlier session) — it would
fail at import time if run today.

Cost note: Option B (counterfactual_sweep) re-runs FULL VLM generation for
every (position, alternative) pair — roughly n_reasoning_tokens *
top_k_alternatives full regenerations. Option A is far cheaper (one VLM
forward pass per swap, no regeneration). max_scenes defaults to 1 and
top_k_alternatives defaults to a modest 4 specifically because this is a
smoke test, not a scaled sweep — scale both up only after confirming this
runs correctly.

RESULT RETRIEVAL: same confirmed issue as masking/pref_pairs' own cluster
runs — outdir is a plain local path on whichever machine the job actually
runs on and is not reliably reachable from the submitting workstation
afterward. The three log lines per scene (marked COUNTERFACTUAL_TOKEN_MAP /
COUNTERFACTUAL_SWAP_A / COUNTERFACTUAL_SWAP_B) are the real retrieval path —
fetch via `lilypad workload logs <id> --content-filter <marker>
--start-time ... --end-time ...` (see pref_pairs/fetch_from_logs.py and
pref_pairs/render_trajectory_overlay.py's get_workload_time_window for the
established pattern; this file does not yet have its own fetch script since
that's premature before the smoke test itself has run once).

Full config reference (all keys optional, defaults shown):
    manifest_path:       "pref_pairs/configs/sample_clips_n100_unstratified.json"
    bucket:               "research-datasets-chicago"
    checkpoint:           "nvidia/Alpamayo-1.5-10B"
    max_scenes:           1
    top_k_alternatives:   4
    reasoning_seed:       0   # seeds torch RNG before each of the 3 baseline generations
    diffusion_seed:       0   # passed to single_token_swap_sweep / counterfactual_sweep
    rollout_kwargs:       {}  # extra kwargs forwarded to _extended_rollout_prefix (temperature, top_p, top_k)
    outdir:               "/mnt/work/tmp/counterfactual_smoke"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

TOKEN_MAP_LOG_MARKER = "COUNTERFACTUAL_TOKEN_MAP "
SWAP_A_LOG_MARKER = "COUNTERFACTUAL_SWAP_A "
SWAP_B_LOG_MARKER = "COUNTERFACTUAL_SWAP_B "

_DEFAULTS: dict[str, Any] = {
    "manifest_path": "pref_pairs/configs/sample_clips_n100_unstratified.json",
    "bucket": "research-datasets-chicago",
    "checkpoint": "nvidia/Alpamayo-1.5-10B",
    "max_scenes": 1,
    "top_k_alternatives": 4,
    "reasoning_seed": 0,
    "diffusion_seed": 0,
    "rollout_kwargs": {},
    "outdir": "/mnt/work/tmp/counterfactual_smoke",
}


def _load_model(checkpoint: str, device: str):
    """Load CounterfactualTokenAnalyzer. Same from_pretrained pattern as
    masking.training.run._load_model and pref_pairs.rollout_harvester.load_alpamayo
    -- just a different (leaf) subclass, so the loaded instance also has
    token_alternative_map / single_token_swap_sweep / counterfactual_sweep."""
    from masking.bootstrap import ensure_alpamayo1_5

    ensure_alpamayo1_5()
    from counterfactual.counterfactual import CounterfactualTokenAnalyzer

    model = CounterfactualTokenAnalyzer.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_reasoning_rng(seed: int) -> None:
    """_extended_rollout_prefix's VLM generation is do_sample=True with no
    internal seed control (same caveat as rollout_harvester.py's
    _harvest_batch) -- the caller must seed torch's global RNG immediately
    before each call. Called once before EACH of the three top-level
    analyses below with the SAME seed, so all three see the byte-identical
    sampled reasoning trace for a given scene, not three independent
    resamples that would make the per-position results incomparable."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _position_result_to_json(position_result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one position's dict (as returned by single_token_swap_sweep /
    counterfactual_sweep, containing a list of CounterfactualResult
    dataclasses) into a plain JSON-safe dict."""
    return {
        "step": position_result["step"],
        "col": position_result["col"],
        "sampled_token": position_result["sampled_token"],
        "sampled_prob": position_result["sampled_prob"],
        "entropy": position_result["entropy"],
        "alternatives": [
            {
                "token": cf.forced_token.text,
                "token_prob": cf.forced_token.prob,
                "d_curvature_mean": cf.d_curvature_mean,
                "d_curvature_max": cf.d_curvature_max,
                "endpoint_shift_m": cf.endpoint_shift_m,
                "traj_ade_m": cf.traj_ade_m,
                "forced_cot": cf.forced_cot,
            }
            for cf in position_result["alternatives"]
        ],
    }


def counterfactual_smoke_loop(training_fn_config: dict[str, Any], experiment_tracker: Any) -> None:
    """Lilypad-compatible entrypoint. See module docstring for the full
    config reference and retrieval instructions."""
    cfg = {**_DEFAULTS, **training_fn_config}
    device = _resolve_device()
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    from masking.data.wds_dataset import iter_clip_events_from_manifest
    from pref_pairs.rollout_harvester import build_tokenized_inputs

    logger.info("Loading model %s on %s ...", cfg["checkpoint"], device)
    model = _load_model(cfg["checkpoint"], device=device)

    top_k_alternatives = int(cfg["top_k_alternatives"])
    reasoning_seed = int(cfg["reasoning_seed"])
    diffusion_seed = int(cfg["diffusion_seed"])
    rollout_kwargs = dict(cfg["rollout_kwargs"])
    max_scenes = cfg["max_scenes"]

    n_done = 0
    for event in iter_clip_events_from_manifest(cfg["manifest_path"], cfg["bucket"]):
        if max_scenes is not None and n_done >= int(max_scenes):
            logger.info("Reached max_scenes=%s, stopping.", max_scenes)
            break

        scene_id = f"{event['clip_id']}_{event['t0_us']}"
        logger.info("=== scene %s (%d/%s) ===", scene_id, n_done + 1, max_scenes)
        data = build_tokenized_inputs(model, event["model_inputs"], device)

        # --- 1. Pure logit inspection: every reasoning token's top-K alternatives ---
        # torch.autocast(bfloat16) wraps every model call below, matching
        # masking/training/run.py's three experiment runners and
        # pref_pairs/rollout_harvester.py's _harvest_batch -- neither
        # Alpamayo1_5 nor MaskedAlpamayo1_5 exposes a .device attribute or
        # self-wraps in autocast internally (confirmed by grep), so the
        # CALLER is always responsible for establishing this context before
        # calling into the model. counterfactual.py's three public methods
        # never got this treatment: token_alternative_map happened to work
        # without it (pure logit inspection, never touches the diffusion
        # expert), but single_token_swap_sweep/counterfactual_sweep both
        # call _denoise_with_mask immediately and crashed with "mat1 and
        # mat2 must have the same dtype, but got Float and BFloat16" on the
        # very first real smoke-test run -- caught here, not fixed inside
        # counterfactual.py itself, to keep that file's untested-but-now-
        # partially-verified internals otherwise unchanged.
        with torch.autocast(device, dtype=torch.bfloat16):
            _seed_reasoning_rng(reasoning_seed)
            alt_map = model.token_alternative_map(
                data, top_k=top_k_alternatives + 1, **rollout_kwargs
            )
        n_positions = alt_map["summary"]["n_reasoning_tokens"]
        logger.info(
            TOKEN_MAP_LOG_MARKER + "%s",
            json.dumps({
                "scene_id": scene_id,
                "cot": alt_map["cot"],
                "n_reasoning_tokens": n_positions,
                "mean_entropy": alt_map["summary"]["mean_entropy"],
                "positions": [
                    {
                        "step": p.step, "col": p.col,
                        "sampled": p.sampled_text, "sampled_prob": p.sampled_prob,
                        "entropy": p.entropy,
                        "top_k": [{"text": t.text, "prob": t.prob} for t in p.top_k],
                    }
                    for p in alt_map["positions"]
                ],
            }),
        )

        # --- 2. Option A: isolated single-token swap, every position, every top-K alt ---
        logger.info("single_token_swap_sweep: %d positions x %d alternatives ...",
                     n_positions, top_k_alternatives)
        with torch.autocast(device, dtype=torch.bfloat16):
            _seed_reasoning_rng(reasoning_seed)
            sweep_a = model.single_token_swap_sweep(
                data, top_k_alternatives=top_k_alternatives, max_positions=n_positions,
                position_selection="all", seed=diffusion_seed, **rollout_kwargs,
            )
        logger.info(
            SWAP_A_LOG_MARKER + "%s",
            json.dumps({
                "scene_id": scene_id,
                "baseline_cot": sweep_a["baseline"]["cot"],
                "positions": [_position_result_to_json(p) for p in sweep_a["positions"]],
            }),
        )

        # --- 3. Option B: forced token + coherent re-sampled continuation ---
        logger.info("counterfactual_sweep: %d positions x %d alternatives ...",
                     n_positions, top_k_alternatives)
        with torch.autocast(device, dtype=torch.bfloat16):
            _seed_reasoning_rng(reasoning_seed)
            sweep_b = model.counterfactual_sweep(
                data, top_k_alternatives=top_k_alternatives, max_positions=n_positions,
                position_selection="all", seed=diffusion_seed, **rollout_kwargs,
            )
        logger.info(
            SWAP_B_LOG_MARKER + "%s",
            json.dumps({
                "scene_id": scene_id,
                "baseline_cot": sweep_b["baseline"]["cot"],
                "positions": [_position_result_to_json(p) for p in sweep_b["positions"]],
            }),
        )

        # Also write to outdir for completeness, even though it's unlikely
        # to be reachable from the submitting workstation afterward (see
        # module docstring) -- the three log lines above are the real
        # retrieval path.
        (outdir / f"{scene_id}.json").write_text(json.dumps({
            "token_alternative_map": {"cot": alt_map["cot"], "summary": alt_map["summary"]},
            "single_token_swap_sweep": [_position_result_to_json(p) for p in sweep_a["positions"]],
            "counterfactual_sweep": [_position_result_to_json(p) for p in sweep_b["positions"]],
        }, default=str, indent=2))

        n_done += 1

    logger.info("Done: %d scene(s) processed.", n_done)
