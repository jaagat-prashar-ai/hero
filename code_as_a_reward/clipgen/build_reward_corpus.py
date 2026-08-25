# SPDX-License-Identifier: Apache-2.0
"""Publish validated ClipGen run outputs as a versioned GRPO corpus.

The production path accepts GT-only offline artifacts whose target and
semantic counterfactual gates passed and whose provenance proves policy
rollouts were not used. The earlier rollout-gated format remains readable
only for explicit diagnostic/ablation corpora. Existing legacy reward_fns
are never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from code_as_a_reward.clipgen.reward_spec import (
    compile_reward_spec_to_source,
    reward_spec_digest,
    validate_reward_spec,
)


def action_family(spec: dict[str, Any]) -> str:
    commitment = [
        c for c in spec["components"] if c["claim"]["kind"] == "commitment"
    ]
    primary = max(commitment, key=lambda c: float(c["weight"]))
    feature = primary["trajectory"]["feature"]
    if feature in {
        "speed_drop",
        "speed_reduction_fraction",
        "stationary_quality",
        "stop_dwell_fraction",
        "late_stationary_quality",
    }:
        return "decelerate_or_hold"
    if feature == "speed_gain":
        return "accelerate"
    if feature in {"heading_left", "lateral_left"}:
        return "lateral_left"
    if feature in {"heading_right", "lateral_right"}:
        return "lateral_right"
    raise ValueError(f"unclassified primary feature {feature!r}")


def _artifact_is_publishable(artifact: dict[str, Any]) -> bool:
    validation = artifact.get("validation") or {}
    mode = artifact.get("pipeline_mode")
    if mode == "offline_gt_only":
        provenance = artifact.get("provenance") or {}
        gt_target = validation.get("gt_target") or {}
        semantic = validation.get("gt_semantic_gate") or {}
        return bool(
            provenance.get("policy_rollouts_used") is False
            and gt_target.get("passed") is True
            and semantic.get("passed") is True
            and float(semantic.get("pos_score", -1.0)) >= 0.7
            and float(semantic.get("pos_score", -1.0))
            - float(semantic.get("max_pert", float("inf")))
            >= 0.4 - 1e-9
        )

    # Backward-compatible diagnostic format. It must never be mistaken for
    # GT-only provenance, but remains useful for controlled ablations.
    generation = validation.get("generation", {})
    return bool(
        generation.get("passed")
        and generation.get("gt_gate_passed") is True
        and validation.get("holdout", {}).get("passed")
        and validation.get("cross_scene", {}).get("passed")
    )


def build_corpus(run_dirs: list[str], out_dir: str) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for run_dir_raw in run_dirs:
        run_dir = Path(run_dir_raw)
        for path in sorted((run_dir / "reward_specs").glob("*.json")):
            artifact = json.loads(path.read_text())
            clip_id = str(artifact["clip_id"])
            validation = artifact.get("validation") or {}
            if not _artifact_is_publishable(artifact):
                continue
            spec = validate_reward_spec(artifact["spec"])
            source = compile_reward_spec_to_source(spec)
            digest = reward_spec_digest(spec)
            prior = rows.get(clip_id)
            if prior and prior["spec_sha256"] != digest:
                raise ValueError(f"clip {clip_id}: conflicting passing specs across runs")
            (out / f"{clip_id}.py").write_text(source)
            shutil.copy2(path, out / f"{clip_id}.spec.json")
            rows[clip_id] = {
                "clip_id": clip_id,
                "action_family": action_family(spec),
                "spec_sha256": digest,
                "pipeline_mode": artifact.get("pipeline_mode", "rollout_diagnostic"),
                "provenance": artifact.get("provenance"),
                "validation": validation,
            }
    manifest = {
        "schema_version": "clipgen.corpus.v2",
        "n_clips": len(rows),
        "clips": [rows[cid] for cid in sorted(rows)],
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    manifest = build_corpus(args.run_dirs, args.out_dir)
    print(json.dumps({"out_dir": args.out_dir, "n_clips": manifest["n_clips"]}, indent=2))


if __name__ == "__main__":
    main()
