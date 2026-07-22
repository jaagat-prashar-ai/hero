# SPDX-License-Identifier: Apache-2.0
"""
aggregated_reward_llm_judge.py -- drop-in replacement for the vendored
recipe's rewards/aggregated_reward_with_reasoning.py that swaps its
reasoning grader for our trajectory-grounded LLM judge
(rl_posttrain.rewards.llm_judge).

Why a copy-with-one-block-changed rather than a patch: the vendored
submodule is never edited (rl_posttrain convention), and the recipe's entry
script hardcodes its reward import -- so the supported extension point is
"bring your own compute_reward with the same signature" wired through our
own entry script (llm_judge_entry.py).

What is deliberately IDENTICAL to the vendored reasoning reward (so the
already-validated GRPO behavior is preserved):
  - the compute_reward signature and (reward, reward_dict) return contract,
  - trajectory decode (decode_rollout_trajectory) + ADE (calculate_ade),
  - comfort scoring (compute_comfort, shifted to [-1, 0]),
  - the continuous mixing formula, its TOML weight keys under
    [custom.alpamayo.reward], the ade_threshold = 3.0 /
    reasoning_threshold = -0.4 gates, and the -1.0 floor when gates fail.

What is different -- the reasoning component only:
  - Vendored: LingoJudgeGrader.score(pred_cot, gt_cot) -- a cached local
    text-similarity model comparing predicted CoC to the ground-truth CoC
    annotation. (Never wired up in our runner: needs a downloaded grader
    checkpoint, and similarity-to-reference is not the faithfulness signal
    this project is after.)
  - Ours: llm_judge.judge_trace(pred_cot, predicted trajectory) -- scores
    whether the rollout's stated reasoning is consistent with the trajectory
    that SAME rollout produced. This is exactly the trajectory-grounded
    rubric validated offline on the 717 judged pairs. Consequences:
      * no gt_cot dependence: clips without a CoC annotation still get a
        reasoning score (the vendored variant fell back to the -1.0 penalty
        path for those), and reward-hacking toward parroting the reference
        text is structurally impossible;
      * ADE remains the only component anchoring to ground truth, and the
        judge the only component scoring text -- clean separation.

Score-scale note: judge_trace returns 0-10; normalize_score maps it to
[-1, 0] exactly where the vendored grader's `sigmoid - 1.0` lands, so the
recipe's reasoning_threshold = -0.4 corresponds to judge score 6 (between
the judged-pairs corrupted-trace median of 1 and chosen-trace median of 7).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

# Reward code executes inside cosmos-rl worker processes whose sys.path is
# the recipe venv's -- our repo modules aren't installed there, so resolve
# them relative to this file (same pattern as llm_judge.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "reasoning_weight",
]


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward].
    Verbatim from the vendored aggregated_reward_with_reasoning."""
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "Reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_REWARD_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_REWARD_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing key(s) in [custom.alpamayo.reward]: {missing}")

    return {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}


# Upper bound on concurrent judge API calls per reward task. 8 comfortably
# covers a 12-rollout GRPO group without approaching Anthropic rate limits
# even with several groups in flight; override via LLM_JUDGE_MAX_CONCURRENCY.
_DEFAULT_JUDGE_CONCURRENCY = 8


def _run_judges_parallel(
    jobs: list[tuple[str | None, Any]],
    judge_fn: Callable[[str, Any], int],
    max_workers: int | None = None,
) -> list[int | None]:
    """Fans judge calls out over a thread pool. `jobs` is an ordered list of
    (pred_cot, waypoints_xyz) pairs, one per rollout; entries whose pred_cot
    is empty/None are skipped (score None) without spending an API call.
    Returns the raw 0-10 scores (or None) in input order.

    Pure fan-out with judge_fn injected so it's unit-testable without the
    network, per the project's no-fake-model-tests convention. Threads, not
    processes: judge_fn is a stateless HTTPS call (llm_judge.judge_trace),
    which is exactly the latency-hiding the shared anthropic client's thread
    safety exists for. A failed judgment propagates JudgeRewardError out of
    .result() -- same fail-loud policy as the serial path."""
    if max_workers is None:
        max_workers = int(
            os.environ.get("LLM_JUDGE_MAX_CONCURRENCY", str(_DEFAULT_JUDGE_CONCURRENCY))
        )
    scores: list[int | None] = [None] * len(jobs)
    runnable = [(i, cot, xyz) for i, (cot, xyz) in enumerate(jobs) if cot and cot.strip()]
    if not runnable:
        return scores
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(runnable)))) as pool:
        futures = [(i, pool.submit(judge_fn, cot, xyz)) for i, cot, xyz in runnable]
        for i, fut in futures:
            scores[i] = fut.result()
    return scores


def compute_reward_batch(
    to_be_evaluated_list: list[str],
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[list[float], list[dict[str, float]]]:
    """Scores one prompt's whole rollout group in a single call -- the shape
    cosmos-rl uses when [train.train_policy].group_reward_calculation is on.

    Why this exists: the default per-rollout reward loop is serial, so a
    12-rollout group paid 12 x (judge API latency) back to back -- slow enough
    to starve train batches past cosmos-rl's 10-min NCCL watchdog (canary
    alpamayo-rl-llm-judge-canary-u0j67p, 2026-07-22). Here the GPU-local work
    (trajectory decode, ADE, comfort, CoC extraction) stays serial -- it's
    fast, and tokenizer thread-safety is not something to gamble on -- while
    the judge's HTTPS calls, the actual bottleneck, run concurrently via
    _run_judges_parallel.

    Per-component semantics are identical to compute_reward (which now
    delegates here with a batch of one)."""
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

    from rl_posttrain.rewards.llm_judge import judge_trace, normalize_score

    w = _get_reward_cfg(config)
    gt_fut_xyz = reference["ego_future_xyz"]

    # Stage 1 (serial, GPU-local): geometry + CoC extraction per rollout.
    per_rollout: list[tuple[float, float, str, Any]] = []
    for to_be_evaluated in to_be_evaluated_list:
        predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
            to_be_evaluated,
            reference["ego_history_xyz"],
            reference["ego_history_rot"],
            tokenizer=tokenizer,
            traj_tokenizer=traj_tokenizer,
            model_config=model_config,
        )

        l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])

        comfort_dict_t = compute_comfort(
            predicted_fut_xyz[:, None, None, ...],
            predicted_fut_rot[:, None, None, ...],
        )
        comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
        comfort_score = comfort_score - 1.0

        pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
        logger.debug(f"[compute_reward] Pred_cot: {pred_cot}")

        # predicted_fut_xyz is (1, T, 3) on the worker's device; the judge
        # wants a plain (T, 3) CPU array.
        pred_xyz_np = predicted_fut_xyz[0].detach().float().cpu().numpy()
        per_rollout.append((float(l2_dist), comfort_score, pred_cot, pred_xyz_np))

    # Stage 2 (parallel): the API-bound judge calls. Rollouts with no decoded
    # CoC keep score None (the -1.0 "missing" value below, matching the
    # vendored missing-CoC penalty) -- no point paying an API call to confirm
    # empty text is unfaithful.
    judge_raws = _run_judges_parallel(
        [(pred_cot, pred_xyz_np) for (_, _, pred_cot, pred_xyz_np) in per_rollout],
        judge_fn=judge_trace,
    )

    # Stage 3 (serial): mix components -- thresholds/formula verbatim from
    # the vendored variant.
    ade_threshold = 3.0
    reasoning_threshold = -0.4

    rewards: list[float] = []
    reward_dicts: list[dict[str, float]] = []
    for (l2_dist, comfort_score, pred_cot, _), judge_raw in zip(per_rollout, judge_raws):
        reasoning_score = -1.0 if judge_raw is None else normalize_score(judge_raw)
        pred_cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)

        if pred_cot_decoded and reasoning_score > reasoning_threshold and l2_dist < ade_threshold:
            final_reward = (
                -w["traj_l2_weight"] * (l2_dist / ade_threshold)
                + w["comfort_weight"] * comfort_score
                + w["reasoning_weight"] * (reasoning_score / reasoning_threshold)
            )
        else:
            final_reward = -1.0

        logger.debug(
            f"[compute_reward] l2={l2_dist:.3f} judge_raw={judge_raw} reasoning={reasoning_score:.3f} "
            f"cot_decoded={pred_cot_decoded} final={final_reward:.4f}"
        )
        rewards.append(float(final_reward))
        reward_dicts.append(
            {
                "traj_L2": float(l2_dist),
                "comfort_reward": float(comfort_score),
                "reasoning_score": float(reasoning_score),
                "reward": float(final_reward),
            }
        )

    return rewards, reward_dicts


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Aggregate traj, comfort, and LLM-judge reasoning into one scalar reward
    for a single rollout -- the shape cosmos-rl's default (non-group) reward
    loop calls. Thin wrapper over compute_reward_batch with a batch of one;
    all semantics live there."""
    rewards, reward_dicts = compute_reward_batch(
        [to_be_evaluated],
        reference,
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        config=config,
        model_config=model_config,
    )
    return rewards[0], reward_dicts[0]
