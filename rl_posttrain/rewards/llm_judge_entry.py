# SPDX-License-Identifier: Apache-2.0
"""ReasoningVLA RL post-training entry point with the LLM-judge reward.

Derived verbatim from the vendored recipe's
models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py
with exactly one functional change: the reward function imports
rl_posttrain.rewards.aggregated_reward_llm_judge instead of the recipe's
aggregated_reward_with_reasoning (whose Lingo-Judge grader compares
predicted CoC to reference CoC and needs a cached local model -- see the
reward module's docstring for the full rationale).

Everything else -- env-var contract (ALPAMAYO_PAI_REASONING_LOCAL_DIR),
vLLM registration, ModelSpec components, hydra config/overrides -- is kept
identical to the vendored reasoning entry, because that composition is what
the recipe's GRPO pipeline was validated against. cosmos-rl executes this
file by path (not as an installed package), so the repo root is inserted on
sys.path both here and inside the reward fn (the fn may be serialized into
worker processes where this module's import-time side effects never ran).
"""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

# rl_posttrain/rewards/llm_judge_entry.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

_PAI_REASONING_LOCAL_DIR = os.getenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR")
if not _PAI_REASONING_LOCAL_DIR:
    raise RuntimeError(
        "Missing required env var ALPAMAYO_PAI_REASONING_LOCAL_DIR "
        "(expected PAI reasoning dataset root, e.g. /path/to/PAI_Reasoning_mini)."
    )
# Fail fast on a missing Anthropic credential at launch time, not thousands
# of GPU-seconds later when the first reward is scored.
if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
    raise RuntimeError(
        "Missing ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN -- the LLM-judge "
        "reward scores every rollout via the Anthropic API."
    )

# ---------------------------------------------------------------------------
# vLLM registration (verbatim from the vendored reasoning entry)
# ---------------------------------------------------------------------------
from cosmos_rl.utils.logging import logger

try:
    from vllm import ModelRegistry as vllm_model_registry

    from alpamayo1_x_rl.models.reasoning_vla.vllm_wrapper import ReasoningVLAModelForVLLM

    vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
except Exception as e:
    logger.warning(f"Failed to register ReasoningVLA model with vLLM: {e}")

# ---------------------------------------------------------------------------
# Model spec components (verbatim from the vendored reasoning entry)
# ---------------------------------------------------------------------------
from alpamayo1_x_rl.models._spec import ModelSpec
from alpamayo1_x_rl.models.reasoning_vla.cosmos_wrapper import ReasoningVLACosmos
from alpamayo1_x_rl.models.reasoning_vla.data_packer import RVLADataPacker
from alpamayo1_x_rl.models.reasoning_vla.rollout import ReasoningVLAVllmRollout  # noqa: F401 (Cosmos registry)
from alpamayo1_x_rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401 (Cosmos registry)
from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    """Compute aggregated reward for a single ReasoningVLA rollout.

    Same shape as the vendored entry's reward fn; the compute_reward import
    is the one swapped line. sys.path is re-ensured here because cosmos-rl
    may run this function in a process that never imported this module."""
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    import alpamayo1_x_rl.state as alp_state
    from rl_posttrain.rewards.aggregated_reward_llm_judge import compute_reward

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    return compute_reward(
        to_be_evaluated,
        reference,
        tokenizer=alp_state.get_tokenizer(),
        traj_tokenizer=alp_state.get_traj_tokenizer(),
        config=config,
        model_config=alp_state.get_ckpt_cfg(),
    )


REASONING_VLA_SPEC = ModelSpec(
    cosmos_wrapper=ReasoningVLACosmos,
    weight_mapper=ReasoningVLAWeightMapper,
    data_packer_cls=RVLADataPacker,
    reward_fn=_reasoning_vla_reward_fn,
    hydra_config_path="hydra_configs",
    hydra_config_name="alpamayo1_5_rvla_rl_pai",
    hydra_overrides=[
        f"data.train.dataset.local_dir={_PAI_REASONING_LOCAL_DIR}",
        "data.train.dataset.clip_index_metadata=clip_index_reasoning_mini.parquet",
        "data.train.dataset.features_metadata=features.csv",
        "data.train.dataset.use_default_keyframe=False",
        "data.train.dataset.reasoning_metadata=reasoning/ood_reasoning.parquet",
    ],
)

if __name__ == "__main__":
    REASONING_VLA_SPEC.launch()
