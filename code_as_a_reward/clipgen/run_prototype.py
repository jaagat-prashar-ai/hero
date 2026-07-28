# SPDX-License-Identifier: Apache-2.0
"""Five-clip prototype harness: dossier -> generate -> gate -> report.

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

--dry-run builds dossiers and gate cases only (no API calls) -- use it to
sanity-check the data before spending the ~$3 generation budget.

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
    return {
        "clip_id": clip_id,
        "scene": scene,
        "waypoints": waypoints,
        "hz": hz,
        "gt_coc": gt_coc,
        "gt_claims": parse_coc_trace(gt_coc, scene_id=clip_id),
        "gt_traj": dossier_mod.features_from_waypoints(waypoints, hz, clip_id),
    }


def run(manifest_path: str, out_dir: str, dry_run: bool = False) -> dict:
    out = Path(out_dir)
    (out / "reward_fns").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    clips = [_load_clip(e) for e in json.loads(Path(manifest_path).read_text())]

    client, tracker = None, CostTracker()
    if not dry_run:
        import anthropic

        client = anthropic.Anthropic()

    report: dict = {"clips": {}, "model": None}
    for clip in clips:
        clip_id = clip["clip_id"]
        text = dossier_mod.build_dossier(clip["scene"], clip["gt_traj"], clip["gt_coc"])
        (out / f"{clip_id}.dossier.txt").write_text(text + "\n")
        others = [
            (c["gt_claims"], c["waypoints"]) for c in clips if c["clip_id"] != clip_id
        ]
        cases = gate_mod.build_cases(clip_id, clip["gt_claims"], clip["waypoints"], clip["hz"], others)
        entry: dict = {"n_gate_cases": len(cases), "attempts": [], "passed": False}
        report["clips"][clip_id] = entry
        if dry_run:
            continue

        transcript, feedback = None, None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = generate_reward_fn(
                    client, text, feedback=feedback, prior_transcript=transcript, tracker=tracker
                )
            except BudgetExceeded as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                report["aborted"] = f"budget ceiling: {e}"
                break
            except (RewardFnError, GenerationRefused) as e:
                entry["attempts"].append({"attempt": attempt, "error": str(e)})
                feedback, transcript = f"the reply was invalid: {e}", transcript or []
                continue
            transcript = result.transcript
            report["model"] = result.model
            gate_result = gate_mod.run_gate(result.source, cases)
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "pos_score": gate_result.pos_score,
                    "neg_p95": gate_result.neg_p95,
                    "passed": gate_result.passed,
                    "scores": gate_result.scores,
                }
            )
            (out / "transcripts" / f"{clip_id}.attempt{attempt}.json").write_text(
                json.dumps(result.transcript, indent=2)
            )
            if gate_result.passed:
                header = (
                    f'"""clip {clip_id} - attempt {attempt}/{MAX_ATTEMPTS} - gate PASS '
                    f'(pos {gate_result.pos_score:.2f}, neg p95 {gate_result.neg_p95:.2f})"""\n'
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


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    result = run(args[0], args[1], dry_run="--dry-run" in sys.argv)
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
    for cid, e in result["clips"].items():
        print(f"{cid}: passed={e['passed']} attempts={len(e['attempts'])}")
