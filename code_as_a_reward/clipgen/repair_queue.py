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
import difflib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen import dossier as dossier_mod
from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen.generate import CostTracker, generate_reward_fn
from code_as_a_reward.clipgen.run_prototype import _load_clip
from code_as_a_reward.clipgen.target_contract import (
    derive_target_contract,
    target_contract_from_dict,
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
        if row.get("repair_eligible") is False:
            continue
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
        parts.append(
            f"scene {row['scene_id']}: "
            + "; ".join(row.get("reward_failures") or row.get("failures") or [])
        )
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


def _evaluate_group(source: str, row: dict[str, Any]) -> agr.GroupValidationResult:
    contract_payload = row.get("target_contract")
    return agr.validate_rollout_group(
        str(row["clip_id"]),
        str(row["scene_id"]),
        float(row.get("hz") or 10.0),
        list(row.get("rollouts") or []),
        source,
        top_k=int(row.get("verify_top_k") or 1),
        target_contract=(
            target_contract_from_dict(contract_payload)
            if isinstance(contract_payload, dict)
            else None
        ),
        reward_spec=row.get("reward_spec"),
    )


def _validate_group(source: str, row: dict[str, Any]) -> list[str]:
    # A candidate repair is judged only on rubric defects. The immutable
    # cached sampler evidence cannot be repaired by changing the reward.
    return list(_evaluate_group(source, row).reward_failures)


def _gate_payload(gate: gate_mod.GateResult) -> dict[str, Any]:
    return {
        "passed": gate.passed,
        "pos_score": gate.pos_score,
        "max_pert": gate.max_pert,
        "delta": gate.pos_score - gate.max_pert,
        "scores": gate.scores,
        "components": gate.components,
        "failures": gate.failures,
    }


def _group_payload(scene_id: str, result: agr.GroupValidationResult) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "passed": result.passed,
        "argmax_rollout_id": result.selection.argmax_rollout_id,
        "score_std": result.score_std,
        "score_range": result.score_range,
        "unique_scores": result.unique_scores,
        "saturation_fraction": result.saturation_fraction,
        "failures": result.failures,
        "sample_failures": result.sample_failures,
        "reward_failures": result.reward_failures,
        "top_gates": {
            str(rollout_id): _gate_payload(gate)
            for rollout_id, gate in result.top_gates.items()
        },
    }


def repair_audit_rows(reports: list[dict[str, Any]]) -> list[list[Any]]:
    """Flatten the complete source -> feedback -> candidate chain."""

    rows: list[list[Any]] = []
    for report in reports:
        initial = report.get("initial_reward_source", "")
        for attempt in report.get("attempts") or []:
            gt_gate = attempt.get("gt_gate") or {}
            development = attempt.get("development_results") or []
            rows.append(
                [
                    report.get("clip_id"),
                    report.get("parent_sha256"),
                    attempt.get("attempt"),
                    report.get("status"),
                    attempt.get("candidate_sha256"),
                    attempt.get("model"),
                    attempt.get("api_cost_usd"),
                    gt_gate.get("pos_score"),
                    gt_gate.get("max_pert"),
                    gt_gate.get("delta"),
                    sum(bool(row.get("passed")) for row in development),
                    len(development),
                    initial,
                    attempt.get("feedback_sent", ""),
                    attempt.get("candidate_source", ""),
                    attempt.get("source_diff", ""),
                    json.dumps(attempt.get("llm_transcript") or [], default=str),
                    "\n".join(attempt.get("failures") or []),
                ]
            )
    return rows


_REPAIR_AUDIT_COLUMNS = [
    "clip_id",
    "parent_sha256",
    "attempt",
    "final_status",
    "candidate_sha256",
    "model",
    "attempt_api_cost_usd",
    "gt_positive_score",
    "gt_max_perturbation",
    "gt_delta",
    "development_groups_passed",
    "development_groups_total",
    "initial_reward_source",
    "feedback_sent_to_llm",
    "candidate_reward_source",
    "source_diff",
    "llm_transcript_json",
    "candidate_failures",
]


def log_repair_reports_to_wandb(
    reports: list[dict[str, Any]],
    *,
    api_cost_usd: float,
    project: str,
    entity: str | None,
    run_name: str | None,
) -> str:
    """Log an auditable repair timeline and attempt-wise improvement curves."""

    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name or f"clipgen-repair-audit-{time.strftime('%Y%m%d%H%M%S')}",
        job_type="clipgen-reward-repair",
    )
    step = 0
    for report in reports:
        for attempt in report.get("attempts") or []:
            gate = attempt.get("gt_gate") or {}
            run.log(
                {
                    "repair/attempt": attempt.get("attempt"),
                    "repair/gt_positive_score": gate.get("pos_score"),
                    "repair/gt_delta": gate.get("delta"),
                    "repair/failure_count": len(attempt.get("failures") or []),
                    "repair/development_pass_rate": (
                        sum(
                            bool(row.get("passed"))
                            for row in attempt.get("development_results") or []
                        )
                        / len(attempt.get("development_results") or [])
                        if attempt.get("development_results")
                        else 0.0
                    ),
                },
                step=step,
            )
            step += 1
    rows = repair_audit_rows(reports)
    run.log(
        {
            "repair_timeline": wandb.Table(columns=_REPAIR_AUDIT_COLUMNS, data=rows),
            "repair/rewards_considered": len(reports),
            "repair/accepted_proposals": sum(
                report.get("status") == "accepted_proposal" for report in reports
            ),
            "repair/api_cost_usd": api_cost_usd,
        }
    )
    run_url = run.url
    run.finish()
    return run_url


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
    initial_source = str(batch.development[0].get("reward_source") or "")
    report: dict[str, Any] = {
        "schema_version": "clipgen.repair-proposal.v1",
        "clip_id": batch.clip_id,
        "parent_sha256": batch.parent_sha256,
        "development_scene_ids": [r["scene_id"] for r in batch.development],
        "holdout_scene_ids": [r["scene_id"] for r in batch.holdout],
        "status": "rejected",
        "activation": "proposal_only",
        "initial_reward_source": initial_source,
        "initial_feedback": feedback,
        "development_evidence": [
            {
                "scene_id": row.get("scene_id"),
                "failures": row.get("failures") or [],
                "top_gates": row.get("top_gates") or {},
            }
            for row in batch.development
        ],
        "attempts": [],
    }

    candidate = None
    for attempt in range(1, max_attempts + 1):
        feedback_sent = feedback
        cost_before = tracker.spent_usd
        result = generate_reward_fn(
            client,
            dossier,
            gt_claims=clip["gt_claims"],
            feedback=feedback,
            prior_transcript=transcript,
            tracker=tracker,
            overlay_jpeg=clip["overlay_jpeg"],
            gt_traj_facts=gate_mod._traj_facts(clip["gt_traj"]),
            target_contract=target,
        )
        transcript = result.transcript
        spec = result.spec
        source = result.source
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
        development_results = []
        for row in batch.development:
            group_result = _evaluate_group(source, row)
            development_results.append(_group_payload(str(row["scene_id"]), group_result))
            failures.extend(f"{row['scene_id']}: {f}" for f in group_result.failures)
        candidate_sha = hashlib.sha256(source.encode()).hexdigest()
        source_diff = "\n".join(
            difflib.unified_diff(
                initial_source.splitlines(),
                source.splitlines(),
                fromfile=f"parent-{batch.parent_sha256[:12]}.py",
                tofile=f"candidate-{candidate_sha[:12]}.py",
                lineterm="",
            )
        )
        report["attempts"].append(
            {
                "attempt": attempt,
                "model": result.model,
                "api_cost_usd": tracker.spent_usd - cost_before,
                "feedback_sent": feedback_sent,
                "llm_transcript": result.transcript,
                "candidate_sha256": candidate_sha,
                "candidate_spec": spec,
                "candidate_source": source,
                "source_diff": source_diff,
                "gt_gate": _gate_payload(gt_gate),
                "development_results": development_results,
                "failures": failures,
            }
        )
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
    holdout_results = []
    for row in batch.holdout:
        group_result = _evaluate_group(source, row)
        holdout_results.append(_group_payload(str(row["scene_id"]), group_result))
        holdout_failures.extend(f"{row['scene_id']}: {f}" for f in group_result.failures)
    report["sealed_holdout_failures"] = holdout_failures
    report["sealed_holdout_results"] = holdout_results
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
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="alpamayo-rl")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or "research")
    parser.add_argument("--wandb-run-name")
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
    wandb_url = None
    if not args.no_wandb:
        try:
            wandb_url = log_repair_reports_to_wandb(
                reports,
                api_cost_usd=tracker.spent_usd,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=args.wandb_run_name,
            )
        except Exception as exc:
            print(f"repair W&B logging failed (continuing): {type(exc).__name__}: {exc}")
    print(
        json.dumps(
            {
                "proposals": reports,
                "api_cost_usd": tracker.spent_usd,
                "wandb_url": wandb_url,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
