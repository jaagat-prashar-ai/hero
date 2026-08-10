# SPDX-License-Identifier: Apache-2.0
"""Five-clip prototype harness: dossier -> generate -> gate -> report.

Gate semantics (2026-08-09 redesign -- fixes the GT-overfit bug found by
d50ad0's select-then-verify diagnostic): the generator is free-form (no
prescribed score rubric); the gate is verified against a REAL Alpamayo
rollout group's argmax, not GT. Gating against GT alone let a function pass
by hardcoding GT's exact numbers (e.g. "heading change within 1 deg of
18.1") -- such a function then scores every real GRPO rollout identically,
since none reproduce GT's exact numbers (see gate.py's module docstring and
BUGS.md). Now: for each clip, a real rollout group is sampled ONCE (real
Alpamayo inference, expensive) via code_as_a_reward.clipgen.rollout_sampler;
each of the up-to-MAX_ATTEMPTS candidate functions is scored against that
SAME cached group (analyze_group_rollouts.select_and_verify -- cheap,
local, no resampling), takes the argmax, and gate-verifies the argmax
against corruptions of ITSELF. GT is still used for the dossier/prompt
context (what the scene-understanding step reasons about); only the gate's
reference case changed.

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

Also exposes clipgen_entrypoint(config) for a lilypad training workload
(see configs/clipgen_real_rollout_smoke.yaml): loads the rollout groups a
prior GPU worker phase sampled (see run_real_rollout_gen.py), then run(),
then out_dir syncs to S3.

Success criterion (fixed up front): >= 4 of 5 clips produce a gate-passing
function within MAX_ATTEMPTS, and the source survives human reading.
"""

from __future__ import annotations

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
    generate_reward_fn,
)
from code_as_a_reward.clipgen.sandbox import RewardFnError
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.obstacle_tracks import SceneObstacles

MAX_ATTEMPTS = 3
# Alpamayo's trajectory head has a fixed output length -- every real
# rollout sampled so far is exactly this many waypoints (rollout_sampler.py
# has no horizon parameter; this falls straight out of the model
# architecture). Used both to decide how much of the dossier a real rollout
# can actually cover, and where to re-anchor t=0 to (see
# dossier.find_rollout_anchor_s).
ROLLOUT_HORIZON_WP = 64


def _load_clip(entry: dict) -> dict:
    clip_id = entry["clip_id"]
    scene = SceneObstacles.from_dataframe(pd.read_parquet(entry["obstacle_parquet"]), clip_id)
    hz = float(entry.get("hz", 10.0))
    if "waypoints_npy" in entry:
        waypoints = np.load(entry["waypoints_npy"])
    else:
        waypoints = dossier_mod.waypoints_from_egomotion(
            pd.read_parquet(entry["egomotion_parquet"]), hz=hz
        )
    gt_coc = Path(entry["gt_coc"]).read_text().strip()
    overlay = entry.get("overlay_jpeg")

    # Re-anchor to wherever this clip's real decisive event actually is,
    # rather than trusting the OOD event's own start timestamp -- the two
    # can be 10+ seconds apart (see find_rollout_anchor_s's docstring;
    # confirmed on the real 352-clip training corpus). anchored_waypoints
    # is what a real rollout sampled from that new t0 would actually cover;
    # gt_traj below is built from it so the WHOLE dossier (obstacle tracks +
    # trajectory) is on one consistent clock.
    anchor_s, _, _ = dossier_mod.find_rollout_anchor_s(waypoints, hz, ROLLOUT_HORIZON_WP)
    anchored_waypoints = waypoints[int(round(anchor_s * hz)) :]

    return {
        "clip_id": clip_id,
        "scene": scene,
        "waypoints": waypoints,  # ORIGINAL, unshifted (t=0 = OOD event start)
        "anchored_waypoints": anchored_waypoints,  # t=0 = anchor_s into the original clip
        "rollout_anchor_s": anchor_s,
        "hz": hz,
        "gt_coc": gt_coc,
        "gt_claims": parse_coc_trace(gt_coc, scene_id=clip_id),
        "gt_traj": dossier_mod.features_from_waypoints(anchored_waypoints, hz, clip_id),
        # Scene grounding for the generator (camera frame + projected GT
        # waypoints, see build_overlays.py); optional so older manifests
        # still run text-only. NOTE: this image is still built relative to
        # the ORIGINAL t0, not the re-anchored one -- not yet regenerated to
        # match (would need re-fetching the camera frame at the new
        # timestamp via build_overlays.py); known gap, not fixed here.
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


def run(
    manifest_path: str,
    out_dir: str,
    rollout_groups: dict[str, list[dict[str, Any]]],
    dry_run: bool = False,
    backend: str = "anthropic",
    wandb_logger: Any = None,
) -> dict:
    """rollout_groups: clip_id -> list of real Alpamayo rollouts (rollout_id/
    coc_text/waypoints), sampled ONCE per clip by run_real_rollout_gen.py's
    GPU worker phase (real inference is expensive; re-scoring a cached group
    against each retry's candidate function is not). A clip missing from
    rollout_groups is skipped with an error entry rather than silently
    falling back to GT -- gating against GT is exactly the bug this redesign
    fixes."""
    out = Path(out_dir)
    (out / "reward_fns").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    clips = [_load_clip(e) for e in json.loads(Path(manifest_path).read_text())]

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

    report: dict = {"clips": {}, "model": None}
    for clip in clips:
        clip_id = clip["clip_id"]
        rollouts = rollout_groups.get(clip_id)
        # Re-extract GT's own (already re-anchored, see _load_clip) trajectory
        # over just ROLLOUT_HORIZON_WP waypoints, so the dossier and the
        # generation prompt only ever cite numbers a real rollout could
        # actually reproduce. Computed from the fixed horizon constant, NOT
        # from `rollouts`' own length -- this must stay valid even before any
        # rollout has been sampled at the (possibly new) anchor.
        rollout_horizon_traj = None
        if len(clip["anchored_waypoints"]) > ROLLOUT_HORIZON_WP:
            rollout_horizon_traj = dossier_mod.features_from_waypoints(
                clip["anchored_waypoints"][:ROLLOUT_HORIZON_WP], clip["hz"], clip_id
            )
        text = dossier_mod.build_dossier(
            clip["scene"],
            clip["gt_traj"],
            clip["gt_coc"],
            rollout_horizon_traj=rollout_horizon_traj,
            rollout_anchor_s=clip["rollout_anchor_s"],
        )
        (out / f"{clip_id}.dossier.txt").write_text(text + "\n")
        entry: dict = {
            "n_rollouts": len(rollouts) if rollouts else 0,
            "gt_coc": clip["gt_coc"],
            "has_overlay": clip["overlay_jpeg"] is not None,
            "attempts": [],
            "passed": False,
        }
        report["clips"][clip_id] = entry
        if dry_run:
            continue
        if not rollouts:
            entry["error"] = f"no sampled rollout group for clip {clip_id}"
            print(f"[clipgen] {clip_id}: {entry['error']}, skipping", flush=True)
            continue

        transcript, feedback = None, None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = generate_reward_fn(
                    client,
                    text,
                    gt_claims=clip["gt_claims"],
                    feedback=feedback,
                    prior_transcript=transcript,
                    tracker=tracker,
                    overlay_jpeg=clip["overlay_jpeg"],
                    gt_traj_facts=gate_mod._traj_facts(rollout_horizon_traj or clip["gt_traj"]),
                )
            except BudgetExceeded as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                report["aborted"] = f"budget ceiling: {e}"
                break
            except (RewardFnError, GenerationRefused) as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                print(f"[clipgen] {clip_id} attempt {attempt} invalid reply: {e}", flush=True)
                # No usable transcript to critique when the failure predates a
                # successful exchange -- regenerate from scratch instead of
                # crashing generate_reward_fn's retry precondition.
                if not transcript:
                    feedback, transcript = None, None
                else:
                    feedback = f"the reply was invalid: {e}"
                continue
            transcript = result.transcript
            report["model"] = result.model

            # The fix: re-score the SAME cached real rollout group with this
            # attempt's candidate (cheap, local, no resampling), take its
            # argmax, and gate-verify the argmax against corruptions of
            # ITSELF -- not GT. select_and_verify is shared with
            # analyze_group_rollouts.py's offline diagnostic.
            select = agr.select_and_verify(
                clip_id, f"{clip_id}_realrollout", clip["hz"], rollouts, result.source
            )
            gate_result = select.argmax_gate
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

            # Persist EVERYTHING the attempt saw/produced: the source, the
            # argmax rollout selected, and the verifier feedback -- used to
            # live only inside transcripts (source) or nowhere (feedback) --
            # report_html.py renders these.
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "argmax_rollout_id": select.argmax_rollout_id,
                    "pos_score": pos_score,
                    "max_pert": max_pert,
                    "passed": gate_passed,
                    "scores": scores,
                    "components": components,
                    "source": result.source,
                    "gate_feedback": None if gate_passed else feedback_text,
                }
            )
            (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                json.dumps(result.transcript, indent=2)
            )
            if wandb_logger is not None:
                wandb_logger.log_attempt(
                    clip_id=clip_id,
                    attempt=attempt,
                    rollouts=select.scored,
                    argmax_rollout_id=select.argmax_rollout_id,
                    gate_result=gate_result,
                    source=result.source,
                )
            if gate_passed:
                header = (
                    f'"""clip {clip_id} - attempt {attempt}/{MAX_ATTEMPTS} - gate PASS '
                    f'(pos {pos_score:.2f}, max pert {max_pert:.2f}, real rollout argmax'
                    f' {select.argmax_rollout_id})"""\n'
                )
                (out / "reward_fns" / f"{clip_id}.py").write_text(header + result.source)
                entry["passed"] = True
                break
            feedback = feedback_text

    n_pass = sum(1 for e in report["clips"].values() if e["passed"])
    report["summary"] = f"{n_pass}/{len(clips)} clips passed the gate (success bar: >=4/5)"
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


def load_rollout_groups(rollouts_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Read `<clip_id>.json` files produced by run_real_rollout_gen.py's GPU
    worker phase -- each `{"clip_id", "t0_us", "rollouts": [...]}` -- into
    {clip_id: rollouts}. Local dir or s3://bucket/prefix (same convention as
    analyze_group_rollouts.iter_dump_files)."""
    if rollouts_dir.startswith("s3://"):
        import boto3

        bucket, _, prefix = rollouts_dir[len("s3://") :].partition("/")
        client = boto3.client("s3")
        out: dict[str, list[dict[str, Any]]] = {}
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                doc = json.loads(body)
                out[doc["clip_id"]] = doc["rollouts"]
        return out
    out = {}
    for path in sorted(Path(rollouts_dir).glob("*.json")):
        doc = json.loads(path.read_text())
        out[doc["clip_id"]] = doc["rollouts"]
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
        from code_as_a_reward.clipgen.wandb_log import ClipgenWandbLogger

        wandb_logger = ClipgenWandbLogger(
            project=config.get("wandb_project", "code-as-reward-clipgen"),
            entity=config.get("wandb_entity"),
            run_name=config.get("name"),
        )
    result = run(
        config["manifest"],
        out_dir,
        rollout_groups,
        dry_run=bool(config.get("dry_run", False)),
        backend=config.get("backend", "openai"),
        wandb_logger=wandb_logger,
    )
    print("CLIPGEN_REPORT_JSON_BEGIN")
    print(json.dumps(result, default=str))
    print("CLIPGEN_REPORT_JSON_END")
    n = _sync_out_to_s3(Path(out_dir), config["s3_bucket"], config["s3_prefix"].rstrip("/"))
    print(f"synced {n} files to s3://{config['s3_bucket']}/{config['s3_prefix']}")
    if wandb_logger is not None:
        wandb_logger.finish()


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
