# SPDX-License-Identifier: Apache-2.0
"""Re-gate published ClipGen rewards against an existing W&B rollout table.

This is a development diagnostic: it reuses already-sampled rollout evidence
and never mutates or activates a reward function.  Reports are recovered from
the final JSON line in each Lilypad workload's logs so the same sealed source
that was published by the offline GT job is evaluated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen.reward_spec import compile_reward_spec_to_source
from code_as_a_reward.clipgen.sandbox import compile_reward_module, run_reward_fn
from code_as_a_reward.clipgen.target_contract import TargetContract
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.commitment_verifier import Verdict, verify_trace_commitments
from pref_pairs.trajectory_features import extract_features


def _report_from_workload(workload_id: str, lilypad: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [lilypad, "workload", "logs", workload_id],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    rows = [line for line in output.splitlines() if '"summary":' in line]
    if not rows:
        raise RuntimeError(f"no final report JSON found in {workload_id}")
    raw = rows[-1]
    return json.loads(raw[raw.index("{") :])


def _contract(payload: dict[str, Any]) -> TargetContract:
    values = dict(payload)
    for key in (
        "entities",
        "speed_profiles",
        "lateral_maneuvers",
        "lateral_directions",
        "behavior_maneuvers",
    ):
        values[key] = frozenset(values.get(key) or [])
    for key in (
        "gt_reference_speed_mps",
        "gt_reference_lateral_m",
        "gt_reference_heading_deg",
    ):
        values[key] = tuple(values.get(key) or [])
    return TargetContract(**values)


def _published_rewards(
    reports: list[dict[str, Any]],
) -> dict[str, tuple[str, TargetContract, dict[str, Any] | None]]:
    rewards = {}
    for report in reports:
        for clip_id, record in (report.get("clips") or {}).items():
            if record.get("passed") is not True:
                continue
            attempt = next(
                (
                    item
                    for item in reversed(record.get("attempts") or [])
                    if item.get("passed") is True and item.get("source")
                ),
                None,
            )
            if attempt is not None:
                reward_spec = attempt.get("reward_spec")
                # This mirrors build_reward_corpus.py, which deterministically
                # recompiles the sealed JSON spec when constructing the live
                # training corpus. Scoring the historical source text here
                # would test an older evaluator than training will load.
                source = (
                    compile_reward_spec_to_source(reward_spec)
                    if reward_spec is not None
                    else str(attempt["source"])
                )
                rewards[str(clip_id)] = (
                    source,
                    _contract(record["target_contract"]),
                    reward_spec,
                )
    return rewards


def _rollout_groups(
    table: dict[str, Any], reward_clip_ids: set[str]
) -> dict[tuple[Any, str, str], list[dict[str, Any]]]:
    index = {name: i for i, name in enumerate(table["columns"])}
    groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table["data"]:
        clip_id = str(row[index["clip_id"]])
        if clip_id not in reward_clip_ids:
            continue
        key = (row[index["group"]], str(row[index["scene_id"]]), clip_id)
        groups[key].append(
            {
                "rollout_id": int(row[index["rollout_id"]]),
                "coc_text": str(row[index["coc_text"]]),
                "waypoints": json.loads(row[index["trajectory_waypoints_json"]]),
            }
        )
    return groups


def analyze(
    reports: list[dict[str, Any]], table: dict[str, Any]
) -> dict[str, Any]:
    rewards = _published_rewards(reports)
    groups = _rollout_groups(table, set(rewards))
    evaluated: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = defaultdict(int)

    for (_group, scene_id, clip_id), rollouts in groups.items():
        source, contract, reward_spec = rewards[clip_id]
        try:
            result = agr.validate_rollout_group(
                clip_id,
                scene_id,
                10.0,
                rollouts,
                source,
                top_k=1,
                target_contract=contract,
                reward_spec=reward_spec,
            )
            fn, _ = compile_reward_module(source)
            scores: list[float] = []
            consistencies: list[float] = []
            rollout_ids: list[int] = []
            for rollout in rollouts:
                rollout_id = int(rollout["rollout_id"])
                trace = parse_coc_trace(
                    rollout["coc_text"], scene_id=scene_id, rollout_id=rollout_id
                )
                features = extract_features(
                    rollout["waypoints"],
                    hz=10.0,
                    scene_id=scene_id,
                    rollout_id=rollout_id,
                )
                scores.append(float(run_reward_fn(fn, trace, features)))
                verdicts = verify_trace_commitments(trace, features)
                n_pass = sum(v.verdict is Verdict.PASS for v in verdicts)
                consistencies.append(n_pass / len(verdicts) if verdicts else 0.0)
                rollout_ids.append(rollout_id)

            score_array = np.asarray(scores, dtype=np.float64)
            consistency_array = np.asarray(consistencies, dtype=np.float64)
            argmax_index = int(np.argmax(score_array))
            for failure in result.failures:
                if failure.startswith("NO_VALID_ROLLOUT"):
                    key = "no_valid_rollout"
                elif failure.startswith("rollout score std"):
                    key = "low_std"
                elif failure.startswith("rollout score range"):
                    key = "low_range"
                elif failure.startswith("only ") and "distinct scores" in failure:
                    key = "low_unique"
                elif "failed perturbation gate" in failure:
                    key = "perturbation"
                elif failure.startswith("target ranking margin"):
                    key = "target_margin"
                else:
                    key = "other"
                failure_counts[key] += 1
            evaluated.append(
                {
                    "passed": bool(result.passed),
                    # A group with no independently GT-compatible rollout is
                    # a sampler/model-coverage failure, not evidence that the
                    # frozen reward itself is flat. Keep it in the overall
                    # diagnostic denominator, but do not mix it into the
                    # reward-quality go/no-go statistics below.
                    "has_target_eligible_rollout": bool(
                        result.selection.selection_status
                        not in {"no_target_eligible_rollout", "no_finite_rollout"}
                    ),
                    "scores": score_array,
                    "consistencies": consistency_array,
                    "score_std": float(np.std(score_array)),
                    "score_range": float(np.ptp(score_array)),
                    "argmax_consistency": float(consistency_array[argmax_index]),
                    "mean_consistency": float(np.mean(consistency_array)),
                    "argmax_rollout_id": rollout_ids[argmax_index],
                }
            )
        except Exception as exc:  # retain the denominator and failure type
            failure_counts[f"exception:{type(exc).__name__}"] += 1

    def pooled_correlation(rows: list[dict[str, Any]]) -> float:
        centered_scores: list[float] = []
        centered_consistencies: list[float] = []
        for group in rows:
            centered_scores.extend(
                (group["scores"] - np.mean(group["scores"])).tolist()
            )
            centered_consistencies.extend(
                (
                    group["consistencies"]
                    - np.mean(group["consistencies"])
                ).tolist()
            )
        if (
            centered_scores
            and np.std(centered_scores) > 0
            and np.std(centered_consistencies) > 0
        ):
            return float(
                np.corrcoef(centered_scores, centered_consistencies)[0, 1]
            )
        return float("nan")

    total = len(groups)
    evaluated_total = len(evaluated)
    passed = sum(group["passed"] for group in evaluated)
    exact_flat = sum(group["score_range"] <= 1e-9 for group in evaluated)
    low_resolution = sum(group["score_std"] < 0.05 for group in evaluated)
    target_eligible = [
        group for group in evaluated if group["has_target_eligible_rollout"]
    ]
    eligible_total = len(target_eligible)
    eligible_exact_flat = sum(
        group["score_range"] <= 1e-9 for group in target_eligible
    )
    eligible_low_resolution = sum(
        group["score_std"] < 0.05 for group in target_eligible
    )
    eligible_argmax_lift = (
        float(
            np.mean(
                [
                    group["argmax_consistency"] - group["mean_consistency"]
                    for group in target_eligible
                ]
            )
        )
        if eligible_total
        else float("nan")
    )
    return {
        "published_rewards": len(rewards),
        "unique_matched_clips": len({key[2] for key in groups}),
        "matched_cached_groups": total,
        "successfully_evaluated_groups": evaluated_total,
        "group_gate_pass": passed,
        "group_gate_rate": passed / total if total else 0.0,
        "exact_flat_groups": exact_flat,
        "exact_flat_rate": exact_flat / total if total else 0.0,
        "low_resolution_groups_std_lt_005": low_resolution,
        "low_resolution_rate": low_resolution / total if total else 0.0,
        "within_group_reward_action_consistency_corr": pooled_correlation(evaluated),
        "argmax_consistency_lift": float(
            np.mean(
                [
                    group["argmax_consistency"] - group["mean_consistency"]
                    for group in evaluated
                ]
            )
        )
        if evaluated_total
        else float("nan"),
        "target_eligible_groups": eligible_total,
        "no_target_eligible_groups": evaluated_total - eligible_total,
        "eligible_exact_flat_groups": eligible_exact_flat,
        "eligible_exact_flat_rate": (
            eligible_exact_flat / eligible_total if eligible_total else 0.0
        ),
        "eligible_low_resolution_groups_std_lt_005": eligible_low_resolution,
        "eligible_low_resolution_rate": (
            eligible_low_resolution / eligible_total if eligible_total else 0.0
        ),
        "eligible_within_group_reward_action_consistency_corr": pooled_correlation(
            target_eligible
        ),
        "eligible_argmax_consistency_lift": eligible_argmax_lift,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def regate_s3_loop(training_fn_config: dict[str, Any], experiment_tracker=None) -> None:
    """Lilypad entrypoint for a sealed, read-only cached-rollout re-gate."""

    import boto3
    import wandb

    bucket = str(training_fn_config.get("s3_bucket", "research-datasets-chicago"))
    prefixes = [str(value).rstrip("/") for value in training_fn_config["s3_prefixes"]]
    s3 = boto3.client("s3")
    reports = [
        json.loads(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/report.json")["Body"].read()
        )
        for prefix in prefixes
    ]

    artifact_name = str(training_fn_config["wandb_table_artifact"])
    table_root = Path("/tmp/clipgen_regate_wandb_table")
    artifact = wandb.Api().artifact(artifact_name)
    artifact.download(root=str(table_root))
    table_paths = sorted(table_root.glob("*.table.json"))
    if len(table_paths) != 1:
        raise RuntimeError(
            f"expected exactly one W&B table in {artifact_name}, found {table_paths}"
        )
    result = analyze(reports, json.loads(table_paths[0].read_text()))
    result.update(
        {
            "source_s3_prefixes": prefixes,
            "wandb_table_artifact": artifact_name,
            "diagnostic_kind": "cached_development_regate",
        }
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True).encode()
    output_key = str(training_fn_config["output_s3_key"])
    s3.put_object(Bucket=bucket, Key=output_key, Body=payload)

    run = wandb.init(
        entity=str(training_fn_config.get("wandb_entity", "research")),
        project=str(training_fn_config.get("wandb_project", "code-as-reward-clipgen")),
        name=str(training_fn_config.get("wandb_name", "clipgen-cached-regate")),
        job_type="cached-development-regate",
    )
    scalars = {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    run.log(scalars)
    run.summary.update(scalars)
    run.summary["output_s3_key"] = output_key
    run.summary["failure_counts_json"] = json.dumps(
        result["failure_counts"], sort_keys=True
    )
    run.finish()
    print(json.dumps(result, sort_keys=True, allow_nan=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("workload_ids", nargs="+")
    parser.add_argument("--lilypad", default="lilypad")
    args = parser.parse_args()
    reports = [
        _report_from_workload(workload_id, args.lilypad)
        for workload_id in args.workload_ids
    ]
    table = json.loads(args.table.read_text())
    print(json.dumps(analyze(reports, table), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
