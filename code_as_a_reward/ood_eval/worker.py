# SPDX-License-Identifier: Apache-2.0
"""
worker.py — runs the code-as-a-reward verifiers (commitment_verifier.py +
perceptual_verifier.py, aggregated by trace_reward.py) over a manifest of
OOD events in TWO modes per event, so results are directly comparable:

  * ground truth  — the dataset's own annotated CoC text + the clip's real
    recorded future trajectory. Tests whether the verifier logic itself is
    well-calibrated (a FAIL/ABSTAIN here is a verifier bug or dataset gap,
    never a model mistake).
  * model rollout  — Alpamayo 1.5 run directly on the same clip (single
    forward pass via sample_trajectories_from_data_with_vlm_rollout,
    num_traj_samples=1) -- NOT pref_pairs/rollout_harvester.py's k=20
    preference-pair machinery, which is a different experiment.

Both branches need physical_ai_av (Python >= 3.11) and, for the model
branch, torch + alpamayo1_5 (Python == 3.12) -- neither importable in this
repo's base Python 3.10 env (see obstacle_tracks.py's module docstring for
the same split). This module is meant to run INSIDE the bootstrapped venv
run.py builds (bootstrap_venv.py), not directly under the project env --
except for `_verify_one`/`verify_trace`/serialization, which are plain
dataclasses/numpy and are exactly what worker_test.py exercises under the
project's normal env using cached fixture data, without needing any of
that.

Output is one JSON object per line (append-only, flushed per row) with the
full audit trail the code-as-a-reward pipeline already produces: parsed
claims, every commitment/perceptual/causal verdict with its evidence, the
full TrajectoryFeatures row (including per-waypoint speed/heading/lateral-
offset series), and the aggregate TraceReward -- for BOTH branches, plus a
flat comparison block. Resumable: re-running with the same --output skips
(clip_id, t0_us) pairs already present, so a preempted cluster job picks up
where it left off instead of restarting from zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from enum import Enum

import numpy as np

from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.commitment_verifier import Verdict, verify_trace_commitments
from code_as_a_reward.ood_eval.manifest import OODEvent
from code_as_a_reward.trace_reward import RewardConfig, TraceReward, TraceVerification, score_trace
from pref_pairs.trajectory_features import extract_features

logger = logging.getLogger(__name__)

MODEL_CHECKPOINT = "nvidia/Alpamayo-1.5-10B"
HZ = 10.0  # load_physical_aiavdataset's time_step=0.1s convention


# ---------------------------------------------------------------------------
# Pure verification + serialization -- no physical_ai_av/torch import here,
# so this half is exercised directly by worker_test.py under the project env.
# ---------------------------------------------------------------------------


def verify_trace(trace, features, scene):
    """score_trace when `scene` is available; degrades to commitment-only
    verification when it's not, mirroring code_reward_entry.py's _load_scene
    degraded path -- a missing obstacle.offline must never silently vanish
    a clip's result, so the caller also gets an explicit `scene_available`
    flag. Returns (TraceVerification, scene_available)."""
    if scene is not None:
        return score_trace(trace, features, scene), True

    commitment_verdicts = verify_trace_commitments(trace, features)
    n_pass = sum(v.verdict is Verdict.PASS for v in commitment_verdicts)
    n_fail = sum(v.verdict is Verdict.FAIL for v in commitment_verdicts)
    n_abstain = sum(v.verdict is Verdict.ABSTAIN for v in commitment_verdicts)
    decided = n_pass + n_fail
    atomic_precision = (n_pass / decided) if decided else None
    n_claims = len(commitment_verdicts)
    decided_fraction = (decided / n_claims) if n_claims else 0.0
    unparsed_chars = sum(e - s for s, e in trace.unparsed_spans)
    unparsed_fraction = (unparsed_chars / len(trace.raw_text)) if trace.raw_text else 0.0
    reward = None
    if atomic_precision is not None:
        reward = max(0.0, atomic_precision - RewardConfig().unparsed_penalty * unparsed_fraction)

    tv = TraceVerification(
        trace=trace,
        commitment_verdicts=commitment_verdicts,
        perceptual_verdicts=[],
        causal_verdicts=[],
        reward=TraceReward(
            scene_id=trace.scene_id,
            rollout_id=trace.rollout_id,
            n_pass={"commitment": n_pass},
            n_fail={"commitment": n_fail},
            n_abstain={"commitment": n_abstain},
            atomic_precision=atomic_precision,
            causal_precision=None,
            decided_fraction=decided_fraction,
            unparsed_char_fraction=unparsed_fraction,
            reward=reward,
        ),
    )
    return tv, False


def verify_one(coc_text: str, waypoints, event: OODEvent, scene, rollout_id: int):
    """Parse + extract features + verify one (coc_text, trajectory) pair
    against `event`'s scene window. `waypoints` is (T, 2 or 3) in the
    ego-frame-at-t0 convention (trajectory_features.py's module docstring) --
    true of both the dataset's ego_future_xyz and Alpamayo's decoded rollouts.
    Returns (TrajectoryFeatures, TraceVerification, scene_available)."""
    scene_id = event.scene_id()
    features = extract_features(waypoints, hz=HZ, scene_id=scene_id, rollout_id=rollout_id)
    trace = parse_coc_trace(coc_text, scene_id=scene_id, rollout_id=rollout_id)
    tv, scene_available = verify_trace(trace, features, scene)
    return features, tv, scene_available


def _json_default(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _verification_to_dict(tv: TraceVerification, features) -> dict:
    return {
        "trace": {
            "raw_text": tv.trace.raw_text,
            "commitments": [dataclasses.asdict(c) for c in tv.trace.commitments],
            "perceptual": [dataclasses.asdict(c) for c in tv.trace.perceptual],
            "causal": [dataclasses.asdict(c) for c in tv.trace.causal],
            "unparsed_spans": tv.trace.unparsed_spans,
        },
        "features": dataclasses.asdict(features),
        "commitment_verdicts": [dataclasses.asdict(v) for v in tv.commitment_verdicts],
        "perceptual_verdicts": [dataclasses.asdict(v) for v in tv.perceptual_verdicts],
        "causal_verdicts": [dataclasses.asdict(v) for v in tv.causal_verdicts],
        "reward": dataclasses.asdict(tv.reward),
    }


def build_row(
    event: OODEvent,
    gt_features,
    gt_tv: TraceVerification,
    gt_scene_available: bool,
    model_result: tuple | None = None,
) -> dict:
    """One JSONL row: full ground-truth verification always, full model
    verification when `model_result` (features, tv, scene_available,
    pred_cot) is given, plus a flat comparison block for quick aggregation
    without re-parsing the nested detail."""
    row: dict = {
        "clip_id": event.clip_id,
        "t0_us": event.t0_us,
        "scene_id": event.scene_id(),
        "event_cluster": event.event_cluster,
        "rank_in_clip": event.rank_in_clip,
        "ground_truth": {
            "coc_text": event.gt_coc,
            "scene_available": gt_scene_available,
            **_verification_to_dict(gt_tv, gt_features),
        },
        "model": None,
        "comparison": {
            "reward_gt": gt_tv.reward.reward,
            "atomic_precision_gt": gt_tv.reward.atomic_precision,
            "decided_fraction_gt": gt_tv.reward.decided_fraction,
            "reward_model": None,
            "atomic_precision_model": None,
            "decided_fraction_model": None,
        },
    }
    if model_result is not None:
        model_features, model_tv, model_scene_available, pred_cot = model_result
        row["model"] = {
            "coc_text": pred_cot,
            "scene_available": model_scene_available,
            **_verification_to_dict(model_tv, model_features),
        }
        row["comparison"]["reward_model"] = model_tv.reward.reward
        row["comparison"]["atomic_precision_model"] = model_tv.reward.atomic_precision
        row["comparison"]["decided_fraction_model"] = model_tv.reward.decided_fraction
    return row


# ---------------------------------------------------------------------------
# Fetch + model layer -- physical_ai_av/torch/alpamayo1_5 imports live INSIDE
# these functions (lazy), so importing this module never requires them; only
# actually calling one does. Real per-clip work, only runs in the
# bootstrapped venv.
# ---------------------------------------------------------------------------


def load_scene_cached(clip_id: str, cache: dict, cache_dir: str = "code_as_a_reward/testdata"):
    """SceneObstacles for `clip_id`, cached across this run's events (a clip
    can have more than one event). A load failure is cached as None too --
    Phase 0 measured obstacle.offline present for 97.4% of clips, so ~2.6%
    are an expected, not exceptional, degraded-path case (see verify_trace);
    caching the failure avoids retrying it once per event of that clip."""
    if clip_id in cache:
        return cache[clip_id]
    from code_as_a_reward.obstacle_tracks import load_obstacle_tracks

    try:
        scene = load_obstacle_tracks(clip_id, cache_dir=cache_dir)
    except Exception:
        logger.exception("clip %s: obstacle.offline load failed, scoring commitments only", clip_id)
        scene = None
    cache[clip_id] = scene
    return scene


def fetch_clip_data(clip_id: str, t0_us: int, avdi=None):
    """One HF fetch (image frames + egomotion history/future, already
    transformed to the ego-frame-at-t0 convention) -- see
    third_party/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py."""
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    return load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)


def load_model():
    """Load Alpamayo-1.5-10B once for the whole run. attn_implementation=
    "eager" skips flash-attn (which compiles from source, 20-40+ min) --
    same tradeoff perplexity/cluster_worker.py made for AlpamayoR1."""
    import torch
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    return Alpamayo1_5.from_pretrained(
        MODEL_CHECKPOINT, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")


def run_model_rollout(model, data: dict) -> tuple[np.ndarray, str]:
    """One forward pass -> (pred_xyz (T, 3) numpy in the ego-frame-at-t0
    convention, pred_cot reasoning text). Mirrors
    third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py's example
    exactly (the vendored package's own documented minimal call), not
    pref_pairs/rollout_harvester.py's k-rollout preference-pair harness."""
    import torch
    from alpamayo1_5 import helper

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    pred_cot = extra["cot"][0]
    pred_xyz_np = pred_xyz.detach().float().cpu().numpy()[0, 0, 0]  # (T, 3), the one sample
    return pred_xyz_np, pred_cot


# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------


def _load_done_keys(output_path: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping unparseable line in existing output (truncated write?)")
                continue
            done.add((row["clip_id"], row["t0_us"]))
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest_json", required=True, help="JSON list of OODEvent dicts")
    ap.add_argument("--output", required=True, help="JSONL output path, appended to (resumable)")
    ap.add_argument("--skip_model", action="store_true", help="ground-truth branch only, no GPU model")
    ap.add_argument("--cache_dir", default="code_as_a_reward/testdata")
    args = ap.parse_args()

    with open(args.manifest_json) as f:
        events = [OODEvent(**e) for e in json.load(f)]

    done = _load_done_keys(args.output)
    logger.info("%d events in manifest, %d already done, %d remaining", len(events), len(done), len(events) - len(done))

    model = None
    if not args.skip_model:
        logger.info("loading %s ...", MODEL_CHECKPOINT)
        model = load_model()
        logger.info("model loaded")

    scene_cache: dict[str, object] = {}
    n_ok = n_failed = n_skipped = 0
    with open(args.output, "a") as out_f:
        for event in events:
            if (event.clip_id, event.t0_us) in done:
                n_skipped += 1
                continue
            try:
                data = fetch_clip_data(event.clip_id, event.t0_us)
                scene = load_scene_cached(event.clip_id, scene_cache, args.cache_dir)

                gt_waypoints = data["ego_future_xyz"].cpu().numpy()[0, 0]
                gt_features, gt_tv, gt_scene_ok = verify_one(
                    event.gt_coc, gt_waypoints, event, scene, rollout_id=0
                )

                model_result = None
                if model is not None:
                    pred_xyz_np, pred_cot = run_model_rollout(model, data)
                    model_features, model_tv, model_scene_ok = verify_one(
                        pred_cot, pred_xyz_np, event, scene, rollout_id=1
                    )
                    model_result = (model_features, model_tv, model_scene_ok, pred_cot)

                row = build_row(event, gt_features, gt_tv, gt_scene_ok, model_result)
                out_f.write(json.dumps(row, default=_json_default) + "\n")
                out_f.flush()
                n_ok += 1
            except Exception:
                logger.exception("clip %s t0=%d: failed, skipping this event", event.clip_id, event.t0_us)
                n_failed += 1

    logger.info("done: %d ok, %d failed, %d already-done skipped", n_ok, n_failed, n_skipped)


if __name__ == "__main__":
    main()
