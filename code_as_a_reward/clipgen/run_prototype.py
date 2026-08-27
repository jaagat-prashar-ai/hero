# SPDX-License-Identifier: Apache-2.0
"""ClipGen-v2 reward builder and legacy rollout-diagnostic harness.

Production corpus construction is GT-only: recorded scene observations,
NVIDIA CoC, and NVIDIA action -> constrained reward spec -> GT-derived
semantic counterfactual gate -> versioned cached function. Policy rollouts
are neither loaded nor sampled by that path. Argmax/top-k selection and the
same-pair perturbation gate run later, inside GRPO.

The legacy diagnostic mode retains rollout-group gates for explicit
ablations. It is not the offline corpus builder.

Manifest (JSON list, one entry per clip):
    [{"clip_id": "...",
      "obstacle_parquet": "path/to/<clip>.obstacle.offline.parquet",
      "egomotion_parquet": "path/to/<clip>.egomotion.parquet",   # or:
      "waypoints_npy": "path/to/<clip>.waypoints.npy",           # (N,2)
      "gt_coc": "path/to/<clip>.coc.txt",
      "hz": 10.0}, ...]

Usage:
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir rollouts_dir
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir --dry-run
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir rollouts_dir --backend openai

rollouts_dir (local dir or s3://bucket/prefix) holds one <clip_id>.json per
clip (see load_rollout_groups), written by run_real_rollout_gen.py's GPU
worker phase -- real Alpamayo rollouts, not GT.

--dry-run builds dossiers only (no API calls, no rollout group required) --
use it to sanity-check the data before spending the generation budget.
--backend openai uses gpt-4o (2026-08-04 smoke runs); default anthropic
(claude-opus-5) is kept for the later code-generation-quality comparison.

``clipgen_offline_entrypoint`` is the production Lilypad entrypoint;
``clipgen_entrypoint`` is the rollout-diagnostic entrypoint.

The accepted artifact is a declarative reward specification compiled by a
deterministic compiler. Executable Python is never authored by the LLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen import dossier as dossier_mod
from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen.generate import (
    BudgetExceeded,
    CostTracker,
    GenerationRefused,
    PROMPT_VERSION,
    generate_reward_fn,
)
from code_as_a_reward.clipgen.reward_spec import compile_reward_spec_to_source
from code_as_a_reward.clipgen.sandbox import RewardFnError
from code_as_a_reward.clipgen.target_contract import (
    derive_target_contract,
    validate_gt_target,
    validate_spec_against_target,
)
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.obstacle_tracks import SceneObstacles

MAX_ATTEMPTS = 7
PIPELINE_VERSION = "clipgen-v3-behavior-contracts"
# Bumped from 3 (2026-08-10): the corpus352 run's real-corpus cost was
# only ~$0.07/clip (~5.4 calls/clip avg) -- real budget headroom to give
# the self-correction loop more room, especially now that gate.py's
# NAMED CULPRIT COMPONENTS feedback gives retries a sharper signal to act
# on (02fd6a8f self-corrected exactly this way on its 3rd attempt already).

# Alpamayo's trajectory head has a fixed output length -- every real
# rollout sampled so far is exactly this many waypoints (rollout_sampler.py
# has no horizon parameter; this falls straight out of the model
# architecture). The GT reference below is truncated to exactly this
# window so generation only ever sees what a rollout can be compared
# against.
ROLLOUT_HORIZON_WP = 64


def _curve_calibration_candidates(spec: dict[str, Any]):
    """Yield deterministic execution-curve variants for adaptive group A.

    The LLM chooses reward semantics, but retrying it cannot repair numeric
    saturation because ``calibrate_spec_against_target`` overwrites those
    numbers on every attempt.  Group A may therefore tune only the monotonic
    curve shape; sealed group B remains unseen and decides generalization.
    """

    seen: set[str] = set()
    for full_scale in (1.5, 2.0, 2.5, 3.0):
        for power in (0.30, 0.40, 0.50, 0.70, 1.0):
            candidate = copy.deepcopy(spec)
            changed = False
            for component in candidate.get("components", []):
                if component.get("claim", {}).get("kind") != "commitment":
                    continue
                rule = component.get("trajectory")
                if not isinstance(rule, dict):
                    continue
                # The GT calibrator's base full point is 1.20x the target
                # magnitude (1.05 for bounded stop-quality features). Expand
                # it without changing the independently anchored floor.
                rule["full"] = float(rule["full"]) * (full_scale / 1.20)
                rule["power"] = power
                changed = True
            if not changed:
                continue
            digest = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            if digest in seen:
                continue
            seen.add(digest)
            yield candidate


def _search_group_curve_calibration(
    *,
    clip_id: str,
    hz: float,
    rollouts: list[dict[str, Any]],
    spec: dict[str, Any],
    gt_cases: list[Any],
    target_contract: Any,
    top_k: int,
    min_score_std: float,
    min_score_range: float,
    min_unique_scores: int,
    max_saturation_fraction: float,
):
    """Return the first strict group-A/GT-passing curve, or ``None``."""

    for candidate in _curve_calibration_candidates(spec):
        source = compile_reward_spec_to_source(candidate)
        if not gate_mod.run_gate(source, gt_cases).passed:
            continue
        quality = agr.validate_rollout_group(
            clip_id,
            f"{clip_id}_generation_curve_calibration",
            hz,
            rollouts,
            source,
            top_k=top_k,
            min_score_std=min_score_std,
            min_score_range=min_score_range,
            min_unique_scores=min_unique_scores,
            max_saturation_fraction=max_saturation_fraction,
            target_contract=target_contract,
            reward_spec=candidate,
        )
        if quality.passed:
            return candidate, source, quality
    return None


def summarize_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    """Produce stable stage-level rates for canary/full-corpus reporting."""

    clips = list((report.get("clips") or {}).values())
    total = len(clips)

    def count(predicate) -> int:
        return sum(1 for clip in clips if predicate(clip))

    def attempts(clip: dict[str, Any]) -> list[dict[str, Any]]:
        return [a for a in clip.get("attempts", []) if isinstance(a, dict)]

    stage_counts = {
        "total": total,
        "generated_valid_spec": count(
            lambda c: any(a.get("reward_spec") is not None for a in attempts(c))
        ),
        "valid_gt_target": count(
            lambda c: (c.get("gt_target_validation") or {}).get("passed") is True
        ),
        "invalid_gt_target": count(
            lambda c: (c.get("gt_target_validation") or {}).get("passed") is False
        ),
        "gt_empirical_gate": count(
            lambda c: any(a.get("gt_gate_passed") is True for a in attempts(c))
        ),
        "generation_has_target_eligible_rollout": count(
            lambda c: any(bool(a.get("eligible_rollout_ids")) for a in attempts(c))
        ),
        "no_valid_rollout": count(
            lambda c: any(a.get("outcome") == "NO_VALID_ROLLOUT" for a in attempts(c))
        ),
        "generation_gate": count(
            lambda c: any(
                a.get("stage") == "generation_group" and a.get("passed") is True
                for a in attempts(c)
            )
        ),
        "sealed_holdout": count(lambda c: (c.get("holdout") or {}).get("passed") is True),
        "cross_scene": count(lambda c: (c.get("cross_scene") or {}).get("passed") is True),
        "published": count(lambda c: c.get("passed") is True),
    }
    rates = {
        name: (value / total if total else 0.0)
        for name, value in stage_counts.items()
        if name != "total"
    }
    valid_gt = stage_counts["valid_gt_target"]
    rates["published_over_valid_gt"] = (
        stage_counts["published"] / valid_gt if valid_gt else 0.0
    )
    target_eligible = stage_counts["generation_has_target_eligible_rollout"]
    rates["published_over_target_eligible"] = (
        stage_counts["published"] / target_eligible if target_eligible else 0.0
    )

    families: dict[str, dict[str, int | float]] = {}
    for clip in clips:
        target = clip.get("target_contract") or {}
        labels = [*target.get("speed_profiles", []), *target.get("lateral_maneuvers", [])]
        family = "+".join(sorted(set(labels))) or "unclassified"
        row = families.setdefault(
            family, {"total": 0, "published": 0, "no_valid_rollout": 0}
        )
        row["total"] = int(row["total"]) + 1
        row["published"] = int(row["published"]) + int(clip.get("passed") is True)
        row["no_valid_rollout"] = int(row["no_valid_rollout"]) + int(
            any(a.get("outcome") == "NO_VALID_ROLLOUT" for a in attempts(clip))
        )
    for row in families.values():
        row["published_rate"] = int(row["published"]) / int(row["total"])

    return {"counts": stage_counts, "rates": rates, "action_families": families}


def summarize_offline_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    """Stage counts for GT-only reward construction.

    Policy rollout quality is deliberately absent: offline publication means
    that the GT target is internally consistent, the constrained reward spec
    matches that target, and the intact NVIDIA pair reaches POS_MIN.
    """

    clips = list((report.get("clips") or {}).values())
    total = len(clips)

    def attempts(clip: dict[str, Any]) -> list[dict[str, Any]]:
        return [a for a in clip.get("attempts", []) if isinstance(a, dict)]

    counts = {
        "total": total,
        "valid_gt_target": sum(
            (c.get("gt_target_validation") or {}).get("passed") is True for c in clips
        ),
        "invalid_gt_target": sum(
            (c.get("gt_target_validation") or {}).get("passed") is False for c in clips
        ),
        "generated_valid_spec": sum(
            any(a.get("reward_spec") is not None for a in attempts(c)) for c in clips
        ),
        "gt_semantic_gate": sum(
            any(a.get("stage") == "offline_gt_semantic_gate" and a.get("passed") is True
                for a in attempts(c))
            for c in clips
        ),
        "published": sum(c.get("passed") is True for c in clips),
    }
    rates = {
        name: value / total if total else 0.0
        for name, value in counts.items()
        if name != "total"
    }
    valid_gt = counts["valid_gt_target"]
    rates["published_over_valid_gt"] = counts["published"] / valid_gt if valid_gt else 0.0
    return {"counts": counts, "rates": rates}


def _load_clip(entry: dict, rollout_doc: dict[str, Any] | None = None) -> dict:
    clip_id = entry["clip_id"]
    if "t0_us" not in entry:
        raise ValueError(
            f"clip {clip_id}: manifest is missing required t0_us; refusing to "
            "build a dossier at clip start while rollouts use an event keyframe"
        )
    obstacle_path = entry.get("obstacle_parquet")
    if obstacle_path:
        scene = SceneObstacles.from_dataframe(pd.read_parquet(obstacle_path), clip_id)
    else:
        scene = SceneObstacles(
            clip_id=clip_id,
            tracks=[],
            availability_note=(
                "The NVIDIA feature inventory reports obstacle.offline=False; "
                "no actor-distance claims are inferred from missing labels."
            ),
        )
    hz = float(entry.get("hz", 10.0))
    if "waypoints_npy" in entry:
        waypoints = np.load(entry["waypoints_npy"])
    else:
        waypoints = dossier_mod.waypoints_from_egomotion(
            pd.read_parquet(entry["egomotion_parquet"]), hz=hz
        )
    gt_coc = Path(entry["gt_coc"]).read_text().strip()
    overlay = entry.get("overlay_jpeg")

    # The window is anchored at the TRAINING KEYFRAME (entry["t0_us"],
    # clip-relative microseconds -- the same instant the RL pipeline's
    # dataset samples rollouts from, get_clip_key_frame). No search for
    # where the action is (the pre-2026-08-14 find_rollout_anchor_s
    # design): the model predicts 6.4s of future from the keyframe, and
    # the GT future over that SAME window is the reference. gt_traj is
    # truncated at the horizon so the whole generation reference
    # (trajectory numbers AND obstacle tracks, see build_dossier) contains
    # nothing a rollout cannot be compared against; if the scene's
    # decisive maneuver completes after the window, the dossier
    # deliberately shows only its in-window beginning.
    anchor_s = float(entry["t0_us"]) / 1e6
    i0 = int(round(anchor_s * hz))
    gt_source = "egomotion_reframed_fallback"
    if rollout_doc is not None and rollout_doc.get("gt_waypoints"):
        # Exact training reference from the same physical_ai_av fetch that
        # produced the rollouts. It is already in the keyframe ego frame.
        window_waypoints = np.asarray(rollout_doc["gt_waypoints"], dtype=np.float64)
        gt_source = "official_ego_future_xyz"
    else:
        reframed = dossier_mod.reframe_waypoints_at_keyframe(waypoints, i0)
        window_waypoints = reframed[i0 : i0 + ROLLOUT_HORIZON_WP]
    if len(window_waypoints) < 2:
        raise ValueError(
            f"clip {clip_id}: prediction window contains only {len(window_waypoints)} waypoints"
        )

    return {
        "clip_id": clip_id,
        "scene": scene,
        "waypoints": waypoints,  # full clip, t=0 = the clip's own start
        "window_waypoints": window_waypoints,  # t=0 = keyframe; ends at the horizon
        "rollout_anchor_s": anchor_s,
        "hz": hz,
        "gt_coc": gt_coc,
        "gt_claims": parse_coc_trace(gt_coc, scene_id=clip_id),
        "gt_traj": dossier_mod.features_from_waypoints(window_waypoints, hz, clip_id),
        "gt_waypoints": window_waypoints,
        "gt_trajectory_source": gt_source,
        # Scene grounding for the generator (camera frame + projected GT
        # waypoints, see build_overlays.py); optional so older manifests
        # still run text-only. The frame is fetched at the clip's default
        # t0; with the keyframe anchor these are usually close, but not
        # exactly time-synced -- known approximation.
        "overlay_jpeg": Path(overlay).read_bytes() if overlay else None,
    }


def _no_finite_rollout_feedback(scored: list[dict[str, Any]]) -> str:
    """Every rollout scored non-finite -- surface the ACTUAL captured
    exceptions (select_and_verify's clipgen_error) instead of a generic
    message, so a retry (or a human) can see WHY, not just THAT."""
    errors = sorted({r["clipgen_error"] for r in scored if r.get("clipgen_error")})
    header = (
        "every real rollout in this clip's group raised an exception or scored"
        " non-finite with this function -- it must handle real, non-GT"
        " trajectories and CoC phrasing, not just the ground-truth pair."
    )
    if not errors:
        return header + " (no exception captured -- every score was a non-finite value the function itself returned)"
    lines = "\n".join(f"  - {e}" for e in errors[:5])
    return f"{header} Captured exception(s):\n{lines}"


def _cross_scene_specificity(
    clip_id: str,
    hz: float,
    source: str,
    own_scored: list[dict[str, Any]],
    rollout_groups: dict[str, dict[str, Any]],
    n_negative_scenes: int,
    min_margin: float = 0.10,
) -> dict[str, Any]:
    """Compare a clip's sealed holdout scores with unrelated scene rollouts."""
    if n_negative_scenes <= 0:
        return {"passed": True, "skipped": True, "reason": "disabled"}
    candidates = [cid for cid in rollout_groups if cid != clip_id]
    candidates.sort(
        key=lambda cid: hashlib.sha256(f"{clip_id}:{cid}".encode()).hexdigest()
    )
    chosen = candidates[:n_negative_scenes]
    if len(chosen) < n_negative_scenes:
        return {
            "passed": False,
            "failure": f"only {len(chosen)} cross-scene negatives available, needs {n_negative_scenes}",
            "negative_clip_ids": chosen,
        }
    own_scores = [
        float(r["clipgen_score"])
        for r in own_scored
        if np.isfinite(r.get("clipgen_score", np.nan))
    ]
    unrelated_scores: list[float] = []
    per_scene: dict[str, list[float]] = {}
    for other_id in chosen:
        other_rollouts = (rollout_groups[other_id].get("groups") or {}).get("holdout") or []
        result = agr.select_and_verify(
            other_id,
            f"{other_id}_cross_negative_for_{clip_id}",
            hz,
            other_rollouts,
            source,
        )
        finite = [
            float(r["clipgen_score"])
            for r in result.scored
            if np.isfinite(r.get("clipgen_score", np.nan))
        ]
        per_scene[other_id] = finite
        unrelated_scores.extend(finite)
    if not own_scores or not unrelated_scores:
        return {
            "passed": False,
            "failure": "no finite own or cross-scene scores",
            "negative_clip_ids": chosen,
        }
    own_p90 = float(np.percentile(own_scores, 90))
    unrelated_p90 = float(np.percentile(unrelated_scores, 90))
    margin = own_p90 - unrelated_p90
    passed = margin + 1e-9 >= min_margin
    return {
        "passed": passed,
        "own_p90": own_p90,
        "unrelated_p90": unrelated_p90,
        "margin": margin,
        "required_margin": min_margin,
        "negative_clip_ids": chosen,
        "negative_scores": per_scene,
        "failure": None
        if passed
        else f"cross-scene p90 margin {margin:.3f} needs >= {min_margin:.3f}",
    }


def run(
    manifest_path: str,
    out_dir: str,
    rollout_groups: dict[str, dict[str, Any]],
    dry_run: bool = False,
    backend: str = "anthropic",
    wandb_logger: Any = None,
    min_generation_rollouts: int = 12,
    min_holdout_rollouts: int = 12,
    strict_rollouts: bool = True,
    holdout_top_k: int = 2,
    cross_scene_negatives: int = 3,
    min_cross_scene_margin: float = 0.10,
    holdout_min_score_std: float = 0.05,
    holdout_min_score_range: float = 0.15,
    holdout_min_unique_scores: int = 3,
    holdout_max_saturation_fraction: float = 0.25,
    offline_gt_only: bool = False,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict:
    """Build reward artifacts in GT-only or legacy rollout-gated mode.

    ``offline_gt_only`` is the production corpus-builder contract: reward
    generation sees only the recorded scene/observations, NVIDIA CoC, and
    NVIDIA action. Policy rollouts are used later by the GRPO reward worker.
    The rollout-gated path remains available for diagnostics and ablations.
    """
    out = Path(out_dir)
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    (out / "reward_fns").mkdir(parents=True, exist_ok=True)
    (out / "reward_specs").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    manifest_entries = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest_entries, list):
        raise ValueError("manifest must be a JSON list")

    # Validate every clip/group join before spending any LLM budget. This
    # turns missing/mismatched rollout files into a workload failure rather
    # than a deceptively successful 0-function run.
    validation_errors: list[str] = []
    if not dry_run and not offline_gt_only:
        for entry in manifest_entries:
            clip_id = entry.get("clip_id")
            doc = rollout_groups.get(clip_id)
            if doc is None:
                validation_errors.append(f"{clip_id}: missing rollout document")
                continue
            if doc.get("schema_version") != "clipgen.rollouts.v2":
                validation_errors.append(
                    f"{clip_id}: rollout schema {doc.get('schema_version')!r} is not clipgen.rollouts.v2"
                )
            if doc.get("clip_id") != clip_id:
                validation_errors.append(
                    f"{clip_id}: rollout document identifies {doc.get('clip_id')!r}"
                )
            if "t0_us" not in entry or int(doc.get("t0_us", -1)) != int(entry.get("t0_us", -2)):
                validation_errors.append(
                    f"{clip_id}: manifest t0_us={entry.get('t0_us')} != "
                    f"rollout t0_us={doc.get('t0_us')}"
                )
            groups = doc.get("groups") or {}
            for group_name in ("generation", "holdout"):
                group_rollouts = groups.get(group_name) or []
                rollout_ids = [
                    rollout.get("rollout_id")
                    for rollout in group_rollouts
                    if isinstance(rollout, dict)
                ]
                if len(rollout_ids) != len(group_rollouts) or len(set(rollout_ids)) != len(
                    rollout_ids
                ):
                    validation_errors.append(
                        f"{clip_id}: rollout_id values must exist and be unique within "
                        f"the {group_name} group"
                    )
            if len(groups.get("generation") or []) < min_generation_rollouts:
                validation_errors.append(
                    f"{clip_id}: generation group has {len(groups.get('generation') or [])}, "
                    f"needs >= {min_generation_rollouts}"
                )
            if len(groups.get("holdout") or []) < min_holdout_rollouts:
                validation_errors.append(
                    f"{clip_id}: holdout group has {len(groups.get('holdout') or [])}, "
                    f"needs >= {min_holdout_rollouts}"
                )
            if not doc.get("gt_waypoints"):
                validation_errors.append(f"{clip_id}: rollout document has no official gt_waypoints")
            else:
                gt = np.asarray(doc["gt_waypoints"], dtype=np.float64)
                if gt.ndim != 2 or gt.shape[0] < 2 or gt.shape[1] < 2 or not np.isfinite(gt).all():
                    validation_errors.append(
                        f"{clip_id}: official gt_waypoints must be a finite N-by-D array"
                    )
            provenance = doc.get("provenance") or {}
            if provenance.get("generation_seed") == provenance.get("holdout_seed"):
                validation_errors.append(
                    f"{clip_id}: generation and holdout seeds must be distinct"
                )
        if validation_errors and strict_rollouts:
            raise ValueError("invalid ClipGen rollout inputs:\n- " + "\n- ".join(validation_errors))

    clips = [
        _load_clip(
            e,
            None
            if offline_gt_only or dry_run
            else rollout_groups.get(e["clip_id"]),
        )
        for e in manifest_entries
    ]

    client, tracker = None, CostTracker()
    if not dry_run:
        if backend == "openai":
            from code_as_a_reward.clipgen.generate import _PRICE_OPENAI, OpenAIChat

            client = OpenAIChat()
            tracker = CostTracker(prices=_PRICE_OPENAI)
        elif backend == "anthropic":
            import anthropic

            client = anthropic.Anthropic()
        else:
            raise ValueError(f"unknown backend {backend!r} (openai | anthropic)")

    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    parser_path = Path(__file__).resolve().parent.parent / "coc_claim_parser.py"
    parser_sha256 = hashlib.sha256(parser_path.read_bytes()).hexdigest()
    report: dict = {
        "schema_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "manifest_sha256": manifest_sha256,
        "parser_sha256": parser_sha256,
        "clips": {},
        "model": None,
        "input_validation_errors": validation_errors,
        "pipeline_mode": "offline_gt_only" if offline_gt_only else "rollout_diagnostic",
    }
    for clip in clips:
        clip_id = clip["clip_id"]
        rollout_doc = rollout_groups.get(clip_id)
        groups = (rollout_doc or {}).get("groups") or {}
        rollouts = groups.get("generation")
        holdout_rollouts = groups.get("holdout")
        # clip["gt_traj"] IS the prediction window (see _load_clip: sliced at
        # the training keyframe, truncated at ROLLOUT_HORIZON_WP), so the
        # dossier needs no separate rollout-horizon section -- every number
        # in it is already something a real rollout can be compared against.
        text = dossier_mod.build_dossier(
            clip["scene"],
            clip["gt_traj"],
            clip["gt_coc"],
            rollout_anchor_s=clip["rollout_anchor_s"],
        )
        (out / f"{clip_id}.dossier.txt").write_text(text + "\n")
        entry: dict = {
            "n_generation_rollouts": len(rollouts) if rollouts else 0,
            "n_holdout_rollouts": len(holdout_rollouts) if holdout_rollouts else 0,
            "gt_coc": clip["gt_coc"],
            "t0_us": int(round(clip["rollout_anchor_s"] * 1e6)),
            "gt_trajectory_source": clip["gt_trajectory_source"],
            "rollout_provenance": (rollout_doc or {}).get("provenance"),
            "has_overlay": clip["overlay_jpeg"] is not None,
            "obstacle_labels_available": clip["scene"].availability_note is None,
            "attempts": [],
            "passed": False,
        }
        target_contract = derive_target_contract(clip["gt_claims"], clip["gt_traj"])
        entry["target_contract"] = target_contract.to_dict()
        gt_target_failures = validate_gt_target(target_contract, clip["gt_traj"])
        entry["gt_target_validation"] = {
            "passed": not gt_target_failures,
            "failures": gt_target_failures,
        }
        report["clips"][clip_id] = entry
        if dry_run:
            continue
        if gt_target_failures:
            entry["error"] = "invalid or unverifiable NVIDIA GT target"
            entry["attempts"].append(
                {
                    "attempt": 0,
                    "stage": "gt_input_validation",
                    "outcome": "INVALID_GT_TARGET",
                    "passed": False,
                    "failures": gt_target_failures,
                }
            )
            print(
                f"[clipgen] {clip_id}: INVALID_GT_TARGET: "
                + "; ".join(gt_target_failures),
                flush=True,
            )
            continue
        if not offline_gt_only and (not rollouts or not holdout_rollouts):
            entry["error"] = f"no sampled rollout group for clip {clip_id}"
            print(f"[clipgen] {clip_id}: {entry['error']}, skipping", flush=True)
            continue

        transcript, feedback = None, None
        for attempt in range(1, max_attempts + 1):
            try:
                result = generate_reward_fn(
                    client,
                    text,
                    gt_claims=clip["gt_claims"],
                    feedback=feedback,
                    prior_transcript=transcript,
                    tracker=tracker,
                    overlay_jpeg=clip["overlay_jpeg"],
                    gt_traj_facts=gate_mod._traj_facts(clip["gt_traj"]),
                    target_contract=target_contract,
                )
            except BudgetExceeded as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                report["aborted"] = f"budget ceiling: {e}"
                break
            except (RewardFnError, GenerationRefused) as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                print(f"[clipgen] {clip_id} attempt {attempt} invalid reply: {e}", flush=True)
                # generate_reward_fn attaches the transcript it built (even
                # on a first, never-successful attempt) to RewardFnError --
                # use it so the retry critiques ITS OWN mistake in
                # conversation instead of regenerating blind. Only a bare
                # GenerationRefused (no attached transcript) falls back to
                # starting over with no feedback.
                carried_transcript = getattr(e, "transcript", None)
                if carried_transcript:
                    transcript = carried_transcript
                    feedback = f"the reply was invalid: {e}"
                else:
                    feedback, transcript = None, None
                continue
            except RuntimeError as e:
                # Provider/network exhaustion must be persisted in the
                # report, not lose the whole run before S3 sync.
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                entry["error"] = f"generation backend failed: {e}"
                print(f"[clipgen] {clip_id}: {entry['error']}", flush=True)
                break
            transcript = result.transcript
            report["model"] = result.model

            # A generated rubric first has to recognize the NVIDIA GT pair
            # and reject target-specific GT counterfactuals. Policy samples
            # are not allowed to define their own positive before this
            # independent anchor passes.
            target_failures = (
                validate_spec_against_target(result.spec, target_contract)
                if result.spec is not None
                else []
            )
            if target_failures:
                feedback_text = "GT target-contract validation failed:\n- " + "\n- ".join(
                    target_failures
                )
                entry["attempts"].append(
                    {
                        "attempt": attempt,
                        "stage": "gt_target_contract",
                        "passed": False,
                        "error": feedback_text,
                        "source": result.source,
                        "reward_spec": result.spec,
                    }
                )
                (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                    json.dumps(result.transcript, indent=2)
                )
                feedback = feedback_text
                continue

            if offline_gt_only:
                # Offline corpus construction ends at the recorded NVIDIA
                # pair and GT-derived semantic counterfactuals. These prove
                # reasoning/action discrimination without exposing the
                # generator to any policy rollout; live argmax/top-k checks
                # remain exclusively in the GRPO worker.
                gt_cases = gate_mod.build_perturbations(
                    clip_id,
                    clip["gt_claims"],
                    np.asarray(clip["gt_waypoints"], dtype=np.float64),
                    clip["hz"],
                    tag="nvidia_gt_offline",
                    reward_spec=result.spec,
                )
                try:
                    gt_gate = gate_mod.run_gate(result.source, gt_cases)
                except RewardFnError as exc:
                    gt_gate = None
                    feedback_text = f"offline GT semantic gate failed to execute: {exc}"
                else:
                    feedback_text = gt_gate.feedback()
                gt_gate_passed = bool(gt_gate is not None and gt_gate.passed)
                entry["attempts"].append(
                    {
                        "attempt": attempt,
                        "stage": "offline_gt_semantic_gate",
                        "passed": gt_gate_passed,
                        "pos_score": None if gt_gate is None else gt_gate.pos_score,
                        "max_pert": None if gt_gate is None else gt_gate.max_pert,
                        "scores": {} if gt_gate is None else gt_gate.scores,
                        "components": {} if gt_gate is None else gt_gate.components,
                        "failures": [] if gt_gate is None else gt_gate.failures,
                        "source": result.source,
                        "reward_spec": result.spec,
                        "gate_feedback": None if gt_gate_passed else feedback_text,
                    }
                )
                (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                    json.dumps(result.transcript, indent=2)
                )
                if not gt_gate_passed:
                    feedback = feedback_text
                    continue

                header = (
                    f"# {PIPELINE_VERSION}; clip {clip_id}; attempt {attempt}/{max_attempts}; "
                    f"offline GT-only PASS; pos {gt_gate.pos_score:.2f}; "
                    f"max_pert {gt_gate.max_pert:.2f}\n"
                )
                (out / "reward_fns" / f"{clip_id}.py").write_text(header + result.source)
                if result.spec is not None:
                    spec_artifact = {
                        "schema_version": PIPELINE_VERSION,
                        "pipeline_mode": "offline_gt_only",
                        "clip_id": clip_id,
                        "spec": result.spec,
                        "provenance": {
                            "t0_us": entry["t0_us"],
                            "gt_trajectory_source": clip["gt_trajectory_source"],
                            "parser_sha256": parser_sha256,
                            "prompt_version": PROMPT_VERSION,
                            "generator_model": result.model,
                            "manifest_sha256": manifest_sha256,
                            "policy_rollouts_used": False,
                        },
                        "validation": {
                            "gt_target": entry["gt_target_validation"],
                            "gt_semantic_gate": entry["attempts"][-1],
                        },
                    }
                    (out / "reward_specs" / f"{clip_id}.json").write_text(
                        json.dumps(spec_artifact, indent=2, default=str)
                    )
                entry["passed"] = True
                break

            gt_cases = gate_mod.build_perturbations(
                clip_id,
                clip["gt_claims"],
                np.asarray(clip["gt_waypoints"], dtype=np.float64),
                clip["hz"],
                tag="nvidia_gt",
                reward_spec=result.spec,
            )
            gt_gate = gate_mod.run_gate(result.source, gt_cases)
            if not gt_gate.passed:
                feedback_text = (
                    "The rubric failed the NVIDIA GT anchor before any policy rollout was "
                    "considered:\n" + gt_gate.feedback()
                )
                entry["attempts"].append(
                    {
                        "attempt": attempt,
                        "stage": "gt_empirical_gate",
                        "passed": False,
                        "pos_score": gt_gate.pos_score,
                        "max_pert": gt_gate.max_pert,
                        "scores": gt_gate.scores,
                        "components": gt_gate.components,
                        "source": result.source,
                        "reward_spec": result.spec,
                        "gate_feedback": feedback_text,
                    }
                )
                (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                    json.dumps(result.transcript, indent=2)
                )
                feedback = feedback_text
                continue

            # Score the cached group, but only an independently GT-eligible
            # rollout may serve as the empirical positive. A group with no
            # such rollout is a sampling failure, not an invitation to make
            # the reward looser.
            select = agr.select_and_verify(
                clip_id,
                f"{clip_id}_realrollout",
                clip["hz"],
                rollouts,
                result.source,
                target_contract=target_contract,
                reward_spec=result.spec,
            )
            gate_result = select.argmax_gate
            if select.selection_status == "no_target_eligible_rollout":
                feedback_text = (
                    "NO_VALID_ROLLOUT: none of the generation rollouts independently matched "
                    "the GT target action, relevant entity, and tolerant execution. The reward "
                    "passed its GT gate and must not be repaired toward this bad batch."
                )
                entry["attempts"].append(
                    {
                        "attempt": attempt,
                        "stage": "generation_group",
                        "outcome": "NO_VALID_ROLLOUT",
                        "passed": False,
                        "gt_gate_passed": True,
                        "eligible_rollout_ids": [],
                        "source": result.source,
                        "reward_spec": result.spec,
                        "gate_feedback": feedback_text,
                    }
                )
                entry["error"] = "generation group contains no independently valid positive"
                (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                    json.dumps(result.transcript, indent=2)
                )
                break
            if gate_result is None:
                gate_passed = False
                pos_score = max_pert = float("nan")
                feedback_text = _no_finite_rollout_feedback(select.scored)
                scores, components, failures = {}, {}, [feedback_text]
            else:
                gate_passed = gate_result.passed
                pos_score, max_pert = gate_result.pos_score, gate_result.max_pert
                scores, components, failures = (
                    gate_result.scores,
                    gate_result.components,
                    gate_result.failures,
                )
                feedback_text = gate_result.feedback()
            ranking_margin = (
                select.best_eligible_score - select.best_ineligible_score
                if np.isfinite(select.best_eligible_score)
                and np.isfinite(select.best_ineligible_score)
                else float("inf")
            )
            if ranking_margin < 0.05:
                ranking_failure = (
                    f"target ranking margin {ranking_margin:.3f} needs >= 0.050; an "
                    "independently wrong rollout ties or outranks the valid positive"
                )
                gate_passed = False
                failures = [*failures, ranking_failure]
                feedback_text = feedback_text + "\n" + ranking_failure

            # The sealed holdout used to be the first place we checked GRPO
            # rank quality. That cannot repair a binary, flat, saturated, or
            # top-1-only reward: group B is intentionally never fed back.
            # Apply the identical checks to adaptive group A first, then open
            # group B only after the candidate has demonstrated useful
            # within-group advantages and top-k robustness.
            generation_quality = None
            curve_calibration_applied = False
            if gate_passed:
                generation_quality = agr.validate_rollout_group(
                    clip_id,
                    f"{clip_id}_generation_quality",
                    clip["hz"],
                    rollouts,
                    result.source,
                    top_k=holdout_top_k,
                    min_score_std=holdout_min_score_std,
                    min_score_range=holdout_min_score_range,
                    min_unique_scores=holdout_min_unique_scores,
                    max_saturation_fraction=holdout_max_saturation_fraction,
                    target_contract=target_contract,
                    reward_spec=result.spec,
                )
                if not generation_quality.passed:
                    recalibrated = _search_group_curve_calibration(
                        clip_id=clip_id,
                        hz=clip["hz"],
                        rollouts=rollouts,
                        spec=result.spec,
                        gt_cases=gt_cases,
                        target_contract=target_contract,
                        top_k=holdout_top_k,
                        min_score_std=holdout_min_score_std,
                        min_score_range=holdout_min_score_range,
                        min_unique_scores=holdout_min_unique_scores,
                        max_saturation_fraction=holdout_max_saturation_fraction,
                    )
                    if recalibrated is not None:
                        result.spec, result.source, generation_quality = recalibrated
                        select = generation_quality.selection
                        gate_result = select.argmax_gate
                        gate_passed = gate_result is not None and gate_result.passed
                        pos_score = gate_result.pos_score if gate_result else float("nan")
                        max_pert = gate_result.max_pert if gate_result else float("nan")
                        scores = gate_result.scores if gate_result else {}
                        components = gate_result.components if gate_result else {}
                        failures = gate_result.failures if gate_result else []
                        ranking_margin = (
                            select.best_eligible_score - select.best_ineligible_score
                            if np.isfinite(select.best_eligible_score)
                            and np.isfinite(select.best_ineligible_score)
                            else float("inf")
                        )
                        curve_calibration_applied = True
                    else:
                        quality_feedback = (
                            "The argmax corruption gate passed, but generation group A "
                            "does not provide a usable GRPO ranking signal:\n- "
                            + "\n- ".join(generation_quality.failures)
                        )
                        gate_passed = False
                        failures = [*failures, *generation_quality.failures]
                        feedback_text = quality_feedback
            selected_rollout = next(
                (r for r in select.scored if r["rollout_id"] == select.argmax_rollout_id),
                None,
            )
            if not gate_passed and selected_rollout is not None:
                feedback_text += (
                    "\nIndependently target-eligible rollout used for this check:"
                    f"\n  rollout_id={selected_rollout['rollout_id']}"
                    f"\n  CoC={selected_rollout['coc_text']!r}"
                    f"\n  eligibility_failures={selected_rollout['target_eligibility_failures']}"
                )

            # Persist EVERYTHING the attempt saw/produced: the source, the
            # argmax rollout selected, and the verifier feedback -- used to
            # live only inside transcripts (source) or nowhere (feedback) --
            # report_html.py renders these.
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "stage": "generation_group",
                    "argmax_rollout_id": select.argmax_rollout_id,
                    "eligible_rollout_ids": select.eligible_rollout_ids,
                    "target_ranking_margin": ranking_margin,
                    "gt_gate_passed": True,
                    "pos_score": pos_score,
                    "max_pert": max_pert,
                    "passed": gate_passed,
                    "scores": scores,
                    "components": components,
                    "score_std": None
                    if generation_quality is None
                    else generation_quality.score_std,
                    "score_range": None
                    if generation_quality is None
                    else generation_quality.score_range,
                    "unique_scores": None
                    if generation_quality is None
                    else generation_quality.unique_scores,
                    "saturation_fraction": None
                    if generation_quality is None
                    else generation_quality.saturation_fraction,
                    "curve_calibration_applied": curve_calibration_applied,
                    "source": result.source,
                    "reward_spec": result.spec,
                    "gate_feedback": None if gate_passed else feedback_text,
                }
            )
            (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                json.dumps(result.transcript, indent=2)
            )
            if wandb_logger is not None:
                try:
                    wandb_logger.log_attempt(
                        clip_id=clip_id,
                        attempt=attempt,
                        rollouts=select.scored,
                        argmax_rollout_id=select.argmax_rollout_id,
                        gate_result=gate_result,
                        source=result.source,
                    )
                except Exception as e:
                    print(
                        f"[clipgen-wandb] {clip_id}: logging disabled after "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    wandb_logger = None
            if gate_passed:
                # The holdout group is evaluated exactly once and its
                # failure details are never returned to the LLM. This is a
                # final generalization test, not another adaptive prompt.
                heldout_validation = agr.validate_rollout_group(
                    clip_id,
                    f"{clip_id}_heldout",
                    clip["hz"],
                    holdout_rollouts,
                    result.source,
                    top_k=holdout_top_k,
                    min_score_std=holdout_min_score_std,
                    min_score_range=holdout_min_score_range,
                    min_unique_scores=holdout_min_unique_scores,
                    max_saturation_fraction=holdout_max_saturation_fraction,
                    target_contract=target_contract,
                    reward_spec=result.spec,
                )
                heldout = heldout_validation.selection
                heldout_gate = heldout.argmax_gate
                heldout_passed = heldout_validation.passed
                entry["holdout"] = {
                    "passed": heldout_passed,
                    "argmax_rollout_id": heldout.argmax_rollout_id,
                    "pos_score": None if heldout_gate is None else heldout_gate.pos_score,
                    "max_pert": None if heldout_gate is None else heldout_gate.max_pert,
                    "scores": {} if heldout_gate is None else heldout_gate.scores,
                    "failures": heldout_validation.failures,
                    "score_std": heldout_validation.score_std,
                    "score_range": heldout_validation.score_range,
                    "unique_scores": heldout_validation.unique_scores,
                    "saturation_fraction": heldout_validation.saturation_fraction,
                    "top_k_verified": len(heldout_validation.top_gates),
                }
                if not heldout_passed:
                    entry["error"] = "candidate passed adaptive gate but failed sealed holdout"
                    break
                specificity = _cross_scene_specificity(
                    clip_id,
                    clip["hz"],
                    result.source,
                    heldout.scored,
                    rollout_groups,
                    n_negative_scenes=cross_scene_negatives,
                    min_margin=min_cross_scene_margin,
                )
                entry["cross_scene"] = specificity
                if not specificity["passed"]:
                    entry["error"] = "candidate failed sealed cross-scene specificity check"
                    break
                header = (
                    f"# {PIPELINE_VERSION}; clip {clip_id}; attempt {attempt}/{max_attempts}; "
                    "generation+holdout PASS; "
                    f"pos {pos_score:.2f}; max_pert {max_pert:.2f}; "
                    f"generation_argmax {select.argmax_rollout_id}\n"
                )
                (out / "reward_fns" / f"{clip_id}.py").write_text(header + result.source)
                if result.spec is not None:
                    spec_artifact = {
                        "schema_version": PIPELINE_VERSION,
                        "clip_id": clip_id,
                        "spec": result.spec,
                        "provenance": {
                            "t0_us": entry["t0_us"],
                            "gt_trajectory_source": clip["gt_trajectory_source"],
                            "parser_sha256": parser_sha256,
                            "prompt_version": PROMPT_VERSION,
                            "generator_model": result.model,
                            "manifest_sha256": manifest_sha256,
                            "rollouts": entry["rollout_provenance"],
                        },
                        "validation": {
                            "generation": entry["attempts"][-1],
                            "holdout": entry["holdout"],
                            "cross_scene": entry["cross_scene"],
                        },
                    }
                    (out / "reward_specs" / f"{clip_id}.json").write_text(
                        json.dumps(spec_artifact, indent=2, default=str)
                    )
                entry["passed"] = True
                break
            feedback = feedback_text

    n_pass = sum(1 for e in report["clips"].values() if e["passed"])
    report["summary"] = (
        f"{n_pass}/{len(clips)} clips published by "
        + ("GT-only offline construction" if offline_gt_only else "rollout diagnostic gates")
    )
    report["acceptance"] = (
        summarize_offline_acceptance(report)
        if offline_gt_only
        else summarize_acceptance(report)
    )
    report["api_cost_usd"] = round(tracker.spent_usd, 4)
    report["api_calls"] = tracker.calls
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _sync_out_to_s3(out_dir: Path, bucket: str, prefix: str) -> int:
    """Upload every file under out_dir to s3://bucket/prefix/ (the node's
    disk is ephemeral; S3 is how results leave a lilypad workload).
    put_object, NOT upload_file: the OCI S3-compat endpoint rejects
    boto3's chunked transfer encoding ("AWS chunked encoding not
    supported" -- killed run y6uw60's sync); same limitation run.py's
    _CheckpointUploader documents. All clipgen outputs are tiny (<1 MB),
    so whole-body put_object is fine."""
    import boto3

    s3 = boto3.client("s3")
    n = 0
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(out_dir)}"
            s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
            n += 1
    return n


def load_rollout_groups(rollouts_dir: str) -> dict[str, dict[str, Any]]:
    """Read v2 rollout documents from a local directory or S3 prefix."""

    def add(out: dict[str, dict[str, Any]], doc: dict[str, Any], source: str) -> None:
        clip_id = doc.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"{source}: rollout document has no clip_id")
        if clip_id in out:
            raise ValueError(f"duplicate rollout document for {clip_id}: {source}")
        out[clip_id] = doc

    if rollouts_dir.startswith("s3://"):
        import boto3

        bucket, _, prefix = rollouts_dir[len("s3://") :].partition("/")
        client = boto3.client("s3")
        out: dict[str, dict[str, Any]] = {}
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                doc = json.loads(body)
                add(out, doc, f"s3://{bucket}/{obj['Key']}")
        return out
    out = {}
    for path in sorted(Path(rollouts_dir).glob("*.json")):
        doc = json.loads(path.read_text())
        add(out, doc, str(path))
    return out


def clipgen_entrypoint(config: dict) -> None:
    """Lilypad training-workload entrypoint (see
    configs/clipgen_real_rollout_smoke.yaml). Loads the real rollout groups
    a prior GPU worker phase sampled (config["rollouts_dir"]), runs the
    prototype, prints the FULL report into the workload logs (so results
    survive even if the S3 sync fails -- learned from y6uw60, which lost
    every per-attempt detail to a sync crash), then syncs the whole out/
    tree (report.json, dossiers, transcripts, reward_fns) to S3 for local
    inspection/report_html.py."""
    out_dir = config.get("out_dir", "/tmp/clipgen_out")
    rollout_groups = load_rollout_groups(config["rollouts_dir"])
    wandb_logger = None
    if config.get("wandb_images", True) and not config.get("dry_run", False):
        try:
            from code_as_a_reward.clipgen.wandb_log import ClipgenWandbLogger

            wandb_logger = ClipgenWandbLogger(
                project=config.get("wandb_project", "code-as-reward-clipgen"),
                entity=config.get("wandb_entity"),
                run_name=config.get("name"),
            )
        except Exception as e:
            print(
                f"[clipgen-wandb] initialization failed; continuing without W&B "
                f"({type(e).__name__}: {e})",
                flush=True,
            )
    result = run(
        config["manifest"],
        out_dir,
        rollout_groups,
        dry_run=bool(config.get("dry_run", False)),
        backend=config.get("backend", "openai"),
        wandb_logger=wandb_logger,
        min_generation_rollouts=int(config.get("min_generation_rollouts", 12)),
        min_holdout_rollouts=int(config.get("min_holdout_rollouts", 12)),
        strict_rollouts=bool(config.get("strict_rollouts", True)),
        holdout_top_k=int(config.get("holdout_top_k", 2)),
        cross_scene_negatives=int(config.get("cross_scene_negatives", 3)),
        min_cross_scene_margin=float(config.get("min_cross_scene_margin", 0.10)),
        holdout_min_score_std=float(config.get("holdout_min_score_std", 0.05)),
        holdout_min_score_range=float(config.get("holdout_min_score_range", 0.15)),
        holdout_min_unique_scores=int(config.get("holdout_min_unique_scores", 3)),
        holdout_max_saturation_fraction=float(
            config.get("holdout_max_saturation_fraction", 0.25)
        ),
    )
    print("CLIPGEN_REPORT_JSON_BEGIN")
    print(json.dumps(result, default=str))
    print("CLIPGEN_REPORT_JSON_END")
    n = _sync_out_to_s3(Path(out_dir), config["s3_bucket"], config["s3_prefix"].rstrip("/"))
    print(f"synced {n} files to s3://{config['s3_bucket']}/{config['s3_prefix']}")
    if wandb_logger is not None:
        try:
            wandb_logger.finish()
        except Exception as e:
            print(
                f"[clipgen-wandb] finish failed ({type(e).__name__}: {e})",
                flush=True,
            )


def clipgen_offline_entrypoint(config: dict) -> None:
    """Lilypad entrypoint for GT-only cached reward construction.

    This entrypoint intentionally has no ``rollouts_dir`` and never imports
    the Alpamayo rollout worker. It consumes the staged manifest's recorded
    observations, NVIDIA CoC, and NVIDIA action, then publishes versioned
    reward specs/functions for later GRPO-time verification.
    """

    out_dir = config.get("out_dir", "/tmp/clipgen_offline_out")
    result = run(
        config["manifest"],
        out_dir,
        {},
        dry_run=bool(config.get("dry_run", False)),
        backend=config.get("backend", "openai"),
        offline_gt_only=True,
        max_attempts=int(config.get("max_attempts", 3)),
    )
    print("CLIPGEN_REPORT_JSON_BEGIN")
    print(json.dumps(result, default=str))
    print("CLIPGEN_REPORT_JSON_END")
    n = _sync_out_to_s3(
        Path(out_dir), config["s3_bucket"], config["s3_prefix"].rstrip("/")
    )
    print(f"synced {n} files to s3://{config['s3_bucket']}/{config['s3_prefix']}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    backend = "anthropic"
    if "--backend" in argv:
        i = argv.index("--backend")
        backend = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    args = [a for a in argv if a != "--dry-run"]
    rollout_groups = load_rollout_groups(args[2]) if len(args) > 2 else {}
    result = run(args[0], args[1], rollout_groups, dry_run="--dry-run" in argv, backend=backend)
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
    for cid, e in result["clips"].items():
        print(f"{cid}: passed={e['passed']} attempts={len(e['attempts'])}")
