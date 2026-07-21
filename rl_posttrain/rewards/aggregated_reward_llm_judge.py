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

import sys
from pathlib import Path
from typing import Any

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


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Aggregate traj, comfort, and LLM-judge reasoning into one scalar reward.

    Trajectory and comfort match the vendored
    alpamayo1_x_rl.rewards.aggregated_reward_with_reasoning.compute_reward
    line for line; the reasoning score comes from judging the decoded CoC
    against the rollout's own predicted trajectory (see module docstring).
    """
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

    from rl_posttrain.rewards.llm_judge import judge_trace, normalize_score

    w = _get_reward_cfg(config)

    gt_fut_xyz = reference["ego_future_xyz"]
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

    # --- the one block that differs from the vendored variant ---
    # Judge the decoded CoC against the trajectory this rollout itself
    # produced. Skipped entirely (score stays at the -1.0 "missing" value,
    # which also fails the gate below, matching the vendored missing-CoC
    # penalty) when no CoC decoded -- no point paying an API call to confirm
    # empty text is unfaithful.
    reasoning_score = -1.0
    judge_raw = None
    if pred_cot and pred_cot.strip():
        # predicted_fut_xyz is (1, T, 3) on the worker's device; the judge
        # wants a plain (T, 3) CPU array.
        pred_xyz_np = predicted_fut_xyz[0].detach().float().cpu().numpy()
        judge_raw = judge_trace(pred_cot, pred_xyz_np)
        reasoning_score = normalize_score(judge_raw)

    # Continuous reward: each component contributes independently, no hard
    # gates -- thresholds/formula verbatim from the vendored variant.
    ade_threshold = 3.0
    reasoning_threshold = -0.4
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
    reward_dict: dict[str, float] = {
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "reasoning_score": float(reasoning_score),
        "reward": float(final_reward),
    }

    return reward_dict["reward"], reward_dict
