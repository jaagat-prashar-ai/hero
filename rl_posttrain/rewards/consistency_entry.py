# SPDX-License-Identifier: Apache-2.0
"""
consistency_entry.py -- the CoC-action consistency GRPO integration, in ONE
file: both the cosmos-rl entry script (run.py launches this for
reward_mode="consistency") and the reward mixing it wires in.

Reward = ADE + comfort + Lingo-Judge reasoning + r_consistency -- the
Alpamayo-R1 paper's joint recipe (Sec. 5.3.2): the reasoning reward pushes
CoC quality up, and the binary consistency reward stops it from drifting
into fluent-but-ungrounded traces (the paper measures reasoning-reward-alone
DROPPING consistency below the SFT baseline, 0.62 -> 0.53).

Composition, relative to the validated in-repo rewards:

  * ADE + comfort: identical to aggregated_reward_llm_judge /
    code_reward_entry (decode_rollout_trajectory -> calculate_ade,
    compute_comfort, ade_threshold = 3.0).
  * Reasoning: the vendored Lingo-Judge grader
    (get_reasoning_grader_from_config -- a local HF sequence classifier, no
    API calls), with the FIXED mixing term `1 - score/threshold` (the
    vendored `score/threshold` was inverted -- full weight at barely-passing,
    zero at perfect; BUGS.md 2026-07-30).
  * Consistency: alpamayo1_x_rl.rewards.coc_consistency_reward (implemented
    2026-08-25 in the vendored recipe; classifies the predicted controls
    into meta-actions, parses the CoC into the closed decision set, checks
    the compatibility table). Enters as a penalty
    `consistency_weight * (r - 1)`: 0 when consistent, -w when not, so the
    reward stays on the same <= 0 scale as the other terms. It applies
    OUTSIDE the pass/fail gates -- a rollout that passes ADE + reasoning but
    contradicts its own stated plan still loses the penalty, which is the
    entire point of the term.
  * Failure band: _graded_failure_reward from aggregated_reward_llm_judge
    (all-fail GRPO groups keep advantage variance).

Everything in the reward is GPU/CPU-local and deterministic (Lingo-Judge is
a small local model; consistency is pure numpy + regex) -- no network, no
API keys, per-rollout latency is milliseconds, so group_reward_calculation
stays FALSE in the TOML and only the per-rollout shape is implemented.

W&B per-component keys: traj_L2, comfort_reward, reasoning_score,
consistency, consistency_unparseable (parse-failure rate, distinct from
genuine inconsistency), reward. At step ~0 the `consistency` mean IS the
un-post-trained SFT checkpoint's consistency rate (rollouts come from the
frozen initial policy), so the training curve doubles as the baseline
comparison (paper: SFT 0.62 -> RL 0.85).
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Reward code executes inside cosmos-rl worker processes whose sys.path is
# the recipe venv's -- our repo modules aren't installed there, so resolve
# them relative to this file (same pattern as code_reward_entry).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward half
# ---------------------------------------------------------------------------

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "reasoning_weight",
    "consistency_weight",
]

_ADE_THRESHOLD = 3.0
_REASONING_THRESHOLD = -0.4


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward]."""
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


def mix_reward(
    *,
    l2_dist: float,
    comfort_score: float,
    reasoning_score: float,
    consistency_score: float,
    cot_decoded: bool,
    w: dict[str, float],
) -> float:
    """Pure mixing formula (unit-testable without torch/cosmos).

    Passing gates (cot decoded, reasoning above threshold, ADE below
    threshold): weighted sum with the fixed reasoning term plus the
    consistency penalty. Failing: the graded failure band, additionally
    shifted down by the consistency penalty so an inconsistent near-miss
    ranks below a consistent one within an all-fail group.
    """
    from rl_posttrain.rewards.aggregated_reward_llm_judge import _graded_failure_reward

    consistency_penalty = w["consistency_weight"] * (consistency_score - 1.0)
    if (
        cot_decoded
        and reasoning_score > _REASONING_THRESHOLD
        and l2_dist < _ADE_THRESHOLD
    ):
        return (
            -w["traj_l2_weight"] * (l2_dist / _ADE_THRESHOLD)
            + w["comfort_weight"] * comfort_score
            + w["reasoning_weight"] * (1.0 - reasoning_score / _REASONING_THRESHOLD)
            + consistency_penalty
        )
    return (
        _graded_failure_reward(
            l2_dist,
            reasoning_score,
            ade_threshold=_ADE_THRESHOLD,
            reasoning_threshold=_REASONING_THRESHOLD,
            cot_decoded=cot_decoded,
        )
        + consistency_penalty
    )


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """ADE + comfort + Lingo-Judge reasoning + consistency for one rollout."""
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards.coc_consistency_reward import (
        compute_consistency_from_completion,
    )
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.light_weight_reasoning_grading_model import (
        get_reasoning_grader_from_config,
    )
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

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
    l2_dist = float(calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0]))

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t)) - 1.0

    pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
    gt_cot = reference.get("cot", "")
    cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)

    reasoning_score = -1.0
    if cot_decoded and gt_cot:
        grader = get_reasoning_grader_from_config(config)
        reasoning_score = float(grader.score(pred_cot, gt_cot).item()) - 1.0

    consistency_score, consistency_diag = compute_consistency_from_completion(
        to_be_evaluated,
        reference,
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    final_reward = mix_reward(
        l2_dist=l2_dist,
        comfort_score=comfort_score,
        reasoning_score=reasoning_score,
        consistency_score=consistency_score,
        cot_decoded=cot_decoded,
        w=w,
    )

    logger.debug(
        f"[consistency_reward] l2={l2_dist:.3f} reasoning={reasoning_score:.3f} "
        f"consistency={consistency_score:.0f} diag={consistency_diag} "
        f"cot_decoded={cot_decoded} final={final_reward:.4f}"
    )
    reward_dict: dict[str, float] = {
        "traj_L2": l2_dist,
        "comfort_reward": comfort_score,
        "reasoning_score": reasoning_score,
        "consistency": float(consistency_score),
        "consistency_unparseable": float(bool(consistency_diag.get("unparseable", False))),
        "reward": float(final_reward),
    }
    return reward_dict["reward"], reward_dict


# ---------------------------------------------------------------------------
# Entry half (mirrors the vendored reasoning entry's ModelSpec launch)
# ---------------------------------------------------------------------------


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    """Same shape as the other rl_posttrain entries' reward fns; re-imports
    this module by its canonical package name so the compute code resolves
    identically in processes that only saw this file as __main__."""
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    import alpamayo1_x_rl.state as alp_state
    from rl_posttrain.rewards.consistency_entry import compute_reward as _compute

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    return _compute(
        to_be_evaluated,
        reference,
        tokenizer=alp_state.get_tokenizer(),
        traj_tokenizer=alp_state.get_traj_tokenizer(),
        config=config,
        model_config=alp_state.get_ckpt_cfg(),
    )


def main() -> None:
    os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
    os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

    pai_reasoning_local_dir = os.getenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR")
    if not pai_reasoning_local_dir:
        raise RuntimeError(
            "Missing required env var ALPAMAYO_PAI_REASONING_LOCAL_DIR "
            "(expected PAI reasoning dataset root)."
        )

    from cosmos_rl.utils.logging import logger as cosmos_logger  # pyright: ignore[reportMissingImports]

    try:
        from vllm import ModelRegistry as vllm_model_registry

        from alpamayo1_x_rl.models.reasoning_vla.vllm_wrapper import ReasoningVLAModelForVLLM

        vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
    except Exception as e:
        cosmos_logger.warning(f"Failed to register ReasoningVLA model with vLLM: {e}")

    from alpamayo1_x_rl.models._spec import ModelSpec
    from alpamayo1_x_rl.models.reasoning_vla.cosmos_wrapper import ReasoningVLACosmos
    from alpamayo1_x_rl.models.reasoning_vla.data_packer import RVLADataPacker
    from alpamayo1_x_rl.models.reasoning_vla.rollout import ReasoningVLAVllmRollout  # noqa: F401 (Cosmos registry)
    from alpamayo1_x_rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401 (Cosmos registry)
    from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper

    spec = ModelSpec(
        cosmos_wrapper=ReasoningVLACosmos,
        weight_mapper=ReasoningVLAWeightMapper,
        data_packer_cls=RVLADataPacker,
        reward_fn=_reasoning_vla_reward_fn,
        hydra_config_path="hydra_configs",
        hydra_config_name="alpamayo1_5_rvla_rl_pai",
        hydra_overrides=[
            f"data.train.dataset.local_dir={pai_reasoning_local_dir}",
            "data.train.dataset.clip_index_metadata=clip_index_reasoning_mini.parquet",
            "data.train.dataset.features_metadata=features.csv",
            "data.train.dataset.use_default_keyframe=False",
            "data.train.dataset.reasoning_metadata=reasoning/ood_reasoning.parquet",
        ],
    )
    spec.launch()


if __name__ == "__main__":
    main()
