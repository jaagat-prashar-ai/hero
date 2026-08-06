# SPDX-License-Identifier: Apache-2.0
"""Five-clip prototype harness: dossier -> generate -> gate -> report.

Gate semantics (2026-08-05 redesign): the generator is free-form (no
prescribed score rubric); the gate checks the GT pair scores >= POS_MIN and
that corrupted variants of that same pair (reversed/flattened trajectory,
gutted claims) each score at least MIN_DROP below it. Cross-clip negatives
are gone -- semantically similar clips made them unwinnable. In the full
pipeline the same perturbation battery verifies the argmax rollout of a
group at selection time.

Manifest (JSON list, one entry per clip):
    [{"clip_id": "...",
      "obstacle_parquet": "path/to/<clip>.obstacle.offline.parquet",
      "egomotion_parquet": "path/to/<clip>.egomotion.parquet",   # or:
      "waypoints_npy": "path/to/<clip>.waypoints.npy",           # (N,2)
      "gt_coc": "path/to/<clip>.coc.txt",
      "hz": 10.0}, ...]

Usage:
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir --dry-run
    python -m code_as_a_reward.clipgen.run_prototype manifest.json out_dir --backend openai

--dry-run builds dossiers and gate cases only (no API calls) -- use it to
sanity-check the data before spending the generation budget.
--backend openai uses gpt-4o (2026-08-04 smoke runs); default anthropic
(claude-opus-5) is kept for the later code-generation-quality comparison.

Also exposes clipgen_entrypoint(config) for a lilypad generic workload
(see configs/clipgen_smoke.yaml): same run(), then out_dir syncs to S3.

Success criterion (fixed up front): >= 4 of 5 clips produce a gate-passing
function within MAX_ATTEMPTS, and the source survives human reading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    return {
        "clip_id": clip_id,
        "scene": scene,
        "waypoints": waypoints,
        "hz": hz,
        "gt_coc": gt_coc,
        "gt_claims": parse_coc_trace(gt_coc, scene_id=clip_id),
        "gt_traj": dossier_mod.features_from_waypoints(waypoints, hz, clip_id),
        # Scene grounding for the generator (camera frame + projected GT
        # waypoints, see build_overlays.py); optional so older manifests
        # still run text-only.
        "overlay_jpeg": Path(overlay).read_bytes() if overlay else None,
    }


def run(manifest_path: str, out_dir: str, dry_run: bool = False, backend: str = "anthropic") -> dict:
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
        text = dossier_mod.build_dossier(clip["scene"], clip["gt_traj"], clip["gt_coc"])
        (out / f"{clip_id}.dossier.txt").write_text(text + "\n")
        # GT stands in for the rollout: the battery is the GT pair plus
        # corrupted variants of it (see gate.build_perturbations). The
        # selection-time flow reuses the same battery on the argmax rollout.
        cases = gate_mod.build_perturbations(
            clip_id, clip["gt_claims"], clip["waypoints"], clip["hz"]
        )
        entry: dict = {
            "n_gate_cases": len(cases),
            "gate_cases": [{"name": c.name, "kind": c.kind} for c in cases],
            "gt_coc": clip["gt_coc"],
            "has_overlay": clip["overlay_jpeg"] is not None,
            "attempts": [],
            "passed": False,
        }
        report["clips"][clip_id] = entry
        if dry_run:
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
            gate_result = gate_mod.run_gate(result.source, cases)
            # Persist EVERYTHING the attempt saw/produced: the source and the
            # verifier feedback used to live only inside transcripts (source)
            # or nowhere (feedback) -- report_html.py renders these.
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "pos_score": gate_result.pos_score,
                    "max_pert": gate_result.max_pert,
                    "passed": gate_result.passed,
                    "scores": gate_result.scores,
                    "components": gate_result.components,
                    "source": result.source,
                    "gate_feedback": None if gate_result.passed else gate_result.feedback(),
                }
            )
            (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                json.dumps(result.transcript, indent=2)
            )
            if gate_result.passed:
                header = (
                    f'"""clip {clip_id} - attempt {attempt}/{MAX_ATTEMPTS} - gate PASS '
                    f'(pos {gate_result.pos_score:.2f}, max pert {gate_result.max_pert:.2f})"""\n'
                )
                (out / "reward_fns" / f"{clip_id}.py").write_text(header + result.source)
                entry["passed"] = True
                break
            feedback = gate_result.feedback()

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


def clipgen_entrypoint(config: dict) -> None:
    """Lilypad generic-workload entrypoint (see configs/clipgen_smoke.yaml).
    Runs the prototype, prints the FULL report into the workload logs
    (so results survive even if the S3 sync fails -- learned from y6uw60,
    which lost every per-attempt detail to a sync crash), then syncs the
    whole out/ tree (report.json, dossiers, transcripts, reward_fns) to S3
    for local inspection/report_html.py."""
    out_dir = config.get("out_dir", "/tmp/clipgen_out")
    result = run(
        config["manifest"],
        out_dir,
        dry_run=bool(config.get("dry_run", False)),
        backend=config.get("backend", "openai"),
    )
    print("CLIPGEN_REPORT_JSON_BEGIN")
    print(json.dumps(result, default=str))
    print("CLIPGEN_REPORT_JSON_END")
    n = _sync_out_to_s3(Path(out_dir), config["s3_bucket"], config["s3_prefix"].rstrip("/"))
    print(f"synced {n} files to s3://{config['s3_bucket']}/{config['s3_prefix']}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    backend = "anthropic"
    if "--backend" in argv:
        i = argv.index("--backend")
        backend = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    args = [a for a in argv if a != "--dry-run"]
    result = run(args[0], args[1], dry_run="--dry-run" in argv, backend=backend)
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
    for cid, e in result["clips"].items():
        print(f"{cid}: passed={e['passed']} attempts={len(e['attempts'])}")
