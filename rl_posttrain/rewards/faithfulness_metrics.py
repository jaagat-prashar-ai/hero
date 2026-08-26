# SPDX-License-Identifier: Apache-2.0
"""Reward-independent held-out faithfulness and trajectory metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from code_as_a_reward.clipgen.target_contract import classify_rollout, derive_target_contract
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.commitment_verifier import Verdict, verify_trace_commitments
from pref_pairs.trajectory_features import extract_features


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _claim_tokens(trace: Any) -> set[str]:
    tokens: set[str] = set()
    for claim in trace.commitments:
        tokens.add(f"maneuver:{claim.maneuver}")
        if claim.direction:
            tokens.add(f"direction:{claim.direction}")
        if claim.speed_profile:
            tokens.add(f"speed:{claim.speed_profile}")
    for claim in trace.perceptual:
        tokens.add(f"entity:{claim.entity}")
        if claim.state:
            tokens.add(f"state:{claim.entity}:{claim.state}")
    return tokens


def _f1(predicted: set[str], target: set[str]) -> float:
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    overlap = len(predicted & target)
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_faithfulness_metrics(
    *,
    pred_cot: str,
    pred_xyz: Any,
    gt_cot: Any,
    gt_xyz: Any,
    scene_id: str,
    rollout_id: int,
    hz: float,
    ade_m: float,
) -> dict[str, float]:
    """Return diagnostic metrics only; callers must never mix these into reward.

    ``eval_faithfulness_accuracy`` is a strict conjunction: trajectory ADE
    below 3 m, target behavior satisfied, canonical CoC F1 at least 0.5, and
    at least half of predicted commitments decidable with >=0.8 precision.
    Its held-out mean is therefore the fraction of faithful-and-accurate
    trajectories, while the component metrics explain every failure.
    """
    pred_trace = parse_coc_trace(_text(pred_cot), scene_id=scene_id, rollout_id=rollout_id)
    gt_trace = parse_coc_trace(_text(gt_cot), scene_id=scene_id, rollout_id=rollout_id)
    pred_features = extract_features(pred_xyz, hz, scene_id, rollout_id)
    gt_features = extract_features(gt_xyz, hz, scene_id, rollout_id)

    verdicts = verify_trace_commitments(pred_trace, pred_features)
    n_pass = sum(v.verdict is Verdict.PASS for v in verdicts)
    n_fail = sum(v.verdict is Verdict.FAIL for v in verdicts)
    n_decided = n_pass + n_fail
    n_claims = len(verdicts)
    precision = n_pass / n_decided if n_decided else 0.0
    coverage = n_decided / n_claims if n_claims else 0.0
    action_consistency = precision * coverage

    semantic_f1 = _f1(_claim_tokens(pred_trace), _claim_tokens(gt_trace))
    contract = derive_target_contract(gt_trace, gt_features)
    contract_pass = float(classify_rollout(contract, pred_trace, pred_features).eligible)
    trajectory_accuracy = max(0.0, 1.0 - float(ade_m) / 3.0)
    continuous = float(np.mean([trajectory_accuracy, semantic_f1, action_consistency, contract_pass]))
    strict = float(
        math.isfinite(float(ade_m))
        and float(ade_m) < 3.0
        and contract_pass == 1.0
        and semantic_f1 >= 0.5
        and precision >= 0.8
        and coverage >= 0.5
    )
    return {
        "eval_faithfulness_accuracy": strict,
        "eval_faithfulness_score": continuous,
        "eval_trajectory_accuracy": trajectory_accuracy,
        "eval_trajectory_ade_m": float(ade_m),
        "eval_target_behavior_pass": contract_pass,
        "eval_reasoning_semantic_f1": semantic_f1,
        "eval_reasoning_action_precision": precision,
        "eval_reasoning_action_coverage": coverage,
        "eval_reasoning_action_consistency": action_consistency,
    }
