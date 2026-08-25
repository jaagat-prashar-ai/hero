# SPDX-License-Identifier: Apache-2.0
"""Asynchronous, held-out repair for ClipGen rewards rejected during GRPO.

The GRPO worker only appends ``clipgen.repair.v1`` records. This module is
run between training boundaries: it groups repeated failures for the same
frozen reward version, exposes development groups to the LLM, keeps one or
more groups sealed, revalidates against the NVIDIA GT semantic gate, and
writes a *proposal*. It never overwrites an active reward function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen import dossier as dossier_mod
from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen.generate import CostTracker, generate_reward_fn
from code_as_a_reward.clipgen.reward_spec import compile_reward_spec_to_source
from code_as_a_reward.clipgen.run_prototype import _load_clip
from code_as_a_reward.clipgen.target_contract import (
    calibrate_spec_against_target,
    derive_target_contract,
    validate_spec_against_target,
)


@dataclass(frozen=True)
class RepairBatch:
    clip_id: str
    parent_sha256: str
    development: list[dict[str, Any]]
    holdout: list[dict[str, Any]]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    source = Path(path)
    if source.is_dir():
        lines = [p.read_text() for p in sorted(source.glob("*.json"))]
    else:
        lines = source.read_text().splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != "clipgen.repair.v1":
            raise ValueError(f"line {line_no}: unsupported repair schema")
        if not row.get("clip_id") or not row.get("reward_source_sha256"):
            raise ValueError(f"line {line_no}: missing clip/version identity")
        records.append(row)
    return records


def make_batches(
    records: list[dict[str, Any]], *, min_groups: int = 3, holdout_groups: int = 1
) -> list[RepairBatch]:
    """Deduplicate scenes and form deterministic development/holdout splits."""

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in records:
        key = (str(row["clip_id"]), str(row["reward_source_sha256"]))
        scene_id = str(row.get("scene_id") or "")
        if scene_id:
            grouped.setdefault(key, {})[scene_id] = row
    batches = []
    for (clip_id, parent_sha), by_scene in sorted(grouped.items()):
        rows = sorted(
            by_scene.values(),
            key=lambda r: hashlib.sha256(str(r["scene_id"]).encode()).hexdigest(),
        )
        if len(rows) < min_groups or len(rows) <= holdout_groups:
            continue
        batches.append(
            RepairBatch(
                clip_id=clip_id,
                parent_sha256=parent_sha,
                development=rows[:-holdout_groups],
                holdout=rows[-holdout_groups:],
            )
        )
    return batches


def development_feedback(batch: RepairBatch) -> str:
    """Compact evidence shown to the LLM; sealed groups are absent."""

    parts = [
        "The frozen reward failed live GRPO verification on multiple development groups. "
        "Repair general semantics, not rollout IDs, exact coordinates, or these literal strings."
    ]
    for row in batch.development:
        parts.append(f"scene {row['scene_id']}: " + "; ".join(row.get("failures") or []))
        for rollout_id, gate in sorted((row.get("top_gates") or {}).items()):
            parts.append(
                f"  selected rollout {rollout_id}: positive={gate.get('pos_score')}, "
                f"max_pert={gate.get('max_pert')}, delta={gate.get('delta')}"
            )
    parts.append(
        "A separate live group is sealed and will be evaluated once. Keep thresholds broad, "
        "monotonic, and tied to canonical reasoning/action semantics."
    )
    return "\n".join(parts)


def _validate_group(source: str, row: dict[str, Any]) -> list[str]:
    result = agr.validate_rollout_group(
        str(row["clip_id"]),
        str(row["scene_id"]),
        float(row.get("hz") or 10.0),
        list(row.get("rollouts") or []),
        source,
        top_k=int(row.get("verify_top_k") or 1),
    )
    return list(result.failures)


def _latest_transcript(corpus_dir: Path, clip_id: str) -> list[dict[str, Any]]:
    candidates = sorted((corpus_dir / "transcripts").glob(f"{clip_id}.attempt*.json"))
    if not candidates:
        raise FileNotFoundError(f"no generation transcript for {clip_id}")
    return json.loads(candidates[-1].read_text())


def repair_batch(
    batch: RepairBatch,
    *,
    manifest_entry: dict[str, Any],
    corpus_dir: str | Path,
    out_dir: str | Path,
    client: Any,
    tracker: CostTracker,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Generate a proposal; activate it later at an epoch/checkpoint boundary."""

    corpus = Path(corpus_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip = _load_clip(manifest_entry, None)
    dossier_path = corpus / f"{batch.clip_id}.dossier.txt"
    dossier = (
        dossier_path.read_text()
        if dossier_path.exists()
        else dossier_mod.build_dossier(
            clip["scene"], clip["gt_traj"], clip["gt_coc"],
            rollout_anchor_s=clip["rollout_anchor_s"],
        )
    )
    transcript = _latest_transcript(corpus, batch.clip_id)
    target = derive_target_contract(clip["gt_claims"], clip["gt_traj"])
    feedback = development_feedback(batch)
    report: dict[str, Any] = {
        "schema_version": "clipgen.repair-proposal.v1",
        "clip_id": batch.clip_id,
        "parent_sha256": batch.parent_sha256,
        "development_scene_ids": [r["scene_id"] for r in batch.development],
        "holdout_scene_ids": [r["scene_id"] for r in batch.holdout],
        "status": "rejected",
        "activation": "proposal_only",
        "attempts": [],
    }

    candidate = None
    for attempt in range(1, max_attempts + 1):
        result = generate_reward_fn(
            client,
            dossier,
            gt_claims=clip["gt_claims"],
            feedback=feedback,
            prior_transcript=transcript,
            tracker=tracker,
            overlay_jpeg=clip["overlay_jpeg"],
            gt_traj_facts=gate_mod._traj_facts(clip["gt_traj"]),
        )
        transcript = result.transcript
        spec = calibrate_spec_against_target(result.spec, target)
        source = compile_reward_spec_to_source(spec)
        failures = validate_spec_against_target(spec, target)
        gt_gate = gate_mod.run_gate(
            source,
            gate_mod.build_perturbations(
                batch.clip_id,
                clip["gt_claims"],
                np.asarray(clip["gt_waypoints"], dtype=np.float64),
                clip["hz"],
                tag="repair_gt",
                reward_spec=spec,
            ),
        )
        failures.extend(gt_gate.failures)
        for row in batch.development:
            failures.extend(f"{row['scene_id']}: {f}" for f in _validate_group(source, row))
        report["attempts"].append({"attempt": attempt, "failures": failures})
        if not failures:
            candidate = (result, spec, source, gt_gate)
            break
        feedback = development_feedback(batch) + "\nCandidate failures:\n- " + "\n- ".join(failures)

    if candidate is None:
        (out / f"{batch.clip_id}.{batch.parent_sha256[:12]}.json").write_text(
            json.dumps(report, indent=2)
        )
        return report

    result, spec, source, gt_gate = candidate
    # Open the sealed split exactly once and never feed its result back.
    holdout_failures = []
    for row in batch.holdout:
        holdout_failures.extend(f"{row['scene_id']}: {f}" for f in _validate_group(source, row))
    report["sealed_holdout_failures"] = holdout_failures
    if not holdout_failures:
        candidate_sha = hashlib.sha256(source.encode()).hexdigest()
        stem = f"{batch.clip_id}.{candidate_sha[:12]}"
        (out / f"{stem}.py").write_text(source)
        report.update(
            {
                "status": "accepted_proposal",
                "candidate_sha256": candidate_sha,
                "model": result.model,
                "spec": spec,
                "gt_pos_score": gt_gate.pos_score,
                "gt_max_pert": gt_gate.max_pert,
            }
        )
    (out / f"{batch.clip_id}.{batch.parent_sha256[:12]}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue")
    parser.add_argument("manifest")
    parser.add_argument("corpus_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--backend", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--min-groups", type=int, default=3)
    args = parser.parse_args()

    if args.backend == "openai":
        from code_as_a_reward.clipgen.generate import OpenAIChat, _PRICE_OPENAI

        client, tracker = OpenAIChat(), CostTracker(prices=_PRICE_OPENAI)
    else:
        import anthropic

        client, tracker = anthropic.Anthropic(), CostTracker()
    entries = {str(e["clip_id"]): e for e in json.loads(Path(args.manifest).read_text())}
    reports = []
    for batch in make_batches(load_records(args.queue), min_groups=args.min_groups):
        if batch.clip_id not in entries:
            continue
        reports.append(
            repair_batch(
                batch,
                manifest_entry=entries[batch.clip_id],
                corpus_dir=args.corpus_dir,
                out_dir=args.out_dir,
                client=client,
                tracker=tracker,
            )
        )
    print(json.dumps({"proposals": reports, "api_cost_usd": tracker.spent_usd}, default=str))


if __name__ == "__main__":
    main()
