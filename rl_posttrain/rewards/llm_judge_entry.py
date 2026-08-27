# SPDX-License-Identifier: Apache-2.0
"""ReasoningVLA RL post-training entry point with the LLM-judge reward.

Derived verbatim from the vendored recipe's
models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py
with exactly three functional changes:
  - the reward function imports rl_posttrain.rewards.aggregated_reward_llm_judge
    instead of the recipe's aggregated_reward_with_reasoning (whose
    Lingo-Judge grader compares predicted CoC to reference CoC and needs a
    cached local model -- see the reward module's docstring);
  - launch goes through _launch_with_scene_reference (a copy of the vendored
    launcher) so the dataset is SceneReferenceDataset, whose reference dicts
    carry the scene frame + calibration the scene-grounded judge requires.
  - the trainer's LR scheduler is rebuilt on the first training step using
    the controller's real run horizon. The pinned Cosmos revision otherwise
    constructs fractional schedules against its 1,000,000-step default,
    leaving short runs at an effectively zero learning rate.

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
# Fail fast on a missing judge credential at launch time, not thousands of
# GPU-seconds later when the first reward is scored. Routed by
# LLM_JUDGE_MODEL exactly like judge_trace: "mock" needs no key at all
# (zero-API infra smoke mode), "gpt-*" needs an OpenAI key, everything else
# is the Anthropic path. (Before 2026-07-30 this unconditionally required
# the Anthropic key, which would have killed a gpt-4o run at entry.)
_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "claude-fable-5")
if _JUDGE_MODEL == "mock":
    pass
elif _JUDGE_MODEL.startswith("gpt"):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"Missing OPENAI_API_KEY -- LLM_JUDGE_MODEL={_JUDGE_MODEL} scores "
            "every rollout via the OpenAI API."
        )
elif not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
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
    """Compute aggregated reward for ReasoningVLA rollouts.

    Same shape as the vendored entry's reward fn; the compute_reward import
    is the one swapped line. sys.path is re-ensured here because cosmos-rl
    may run this function in a process that never imported this module.

    With [train.train_policy].group_reward_calculation on (our llm_judge
    TOML), cosmos-rl hands the prompt's WHOLE rollout group as a list in one
    call and expects (list, list-of-dicts) back -- that's the batched path,
    which parallelizes the judge API calls. A single str still gets the
    vendored one-rollout contract, so the fn stays drop-in either way."""
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    import alpamayo1_x_rl.state as alp_state
    from rl_posttrain.rewards.aggregated_reward_llm_judge import (
        compute_reward,
        compute_reward_batch,
    )

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    fn = compute_reward_batch if isinstance(to_be_evaluated, list) else compute_reward
    return fn(
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


def _launch_with_scene_reference(spec: ModelSpec) -> None:
    """Copy of the vendored alpamayo1_x_rl.launcher.launch_alpamayo_model
    with exactly one functional change: launch_worker gets
    SceneReferenceDataset instead of AlpamayoCosmosDataset, so every
    reference dict carries the judge's scene payload (frame + calibration).
    The vendored launcher hardcodes its dataset class -- same
    copy-with-one-block-changed convention as the reward module itself."""
    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from cosmos_rl.policy.model.base import ModelRegistry

    import alpamayo1_x_rl.state as alp_state
    from alpamayo1_x_rl.launcher import _read_ckpt_path_from_toml

    from rl_posttrain.rewards.scene_reference_dataset import SceneReferenceDataset

    ckpt_path = _read_ckpt_path_from_toml()

    alp_state.init_once(
        ckpt_path,
        hydra_config_path=spec.hydra_config_path,
        hydra_config_name=spec.hydra_config_name,
        overrides=spec.hydra_overrides,
    )
    from rl_posttrain.rewards.heldout_validation import (
        install_cosmos_validation_compat,
        install_heldout_validation_split,
    )

    train_n, val_n = install_heldout_validation_split(alp_state.get_dataloaders())
    install_cosmos_validation_compat()
    logger.warning("[llm_judge] deterministic dataset split: train=%d val=%d", train_n, val_n)

    # Keep the judge optimizer schedule identical to the code/global arms.
    # The pinned Cosmos trainer creates this scheduler with total_steps=1e6;
    # step_training is the first point that receives the real controller
    # horizon. Without this wrapper the live judge arm reached only 1.73e-9
    # at step 25 instead of ~1.91e-6.
    from rl_posttrain.rewards.code_reward_entry import _patch_lr_scheduler_total_steps

    _patch_lr_scheduler_total_steps(ReasoningVLAGRPOTrainer)

    ModelRegistry.register_model(
        spec.cosmos_wrapper,
        spec.weight_mapper,
        data_packer_cls=spec.data_packer_cls,
    )

    launch_worker(
        dataset=lambda config: SceneReferenceDataset(split="train"),
        data_packer=spec.data_packer_cls(),
        reward_fns=[spec.reward_fn],
        val_dataset=lambda config: SceneReferenceDataset(split="val"),
        val_data_packer=spec.data_packer_cls(),
        val_reward_fns=[spec.reward_fn],
    )


if __name__ == "__main__":
    # Same instrumentation that root-caused the code-mode ~51-min stalls
    # (BUGS.md 2026-07-28/30): the judge full-run crash loop (every attempt
    # dead at steps 137-149, ~2h40m cadence) is still unnamed, and an
    # uninstrumented crash teaches nothing. _StackSampler leaves repeated
    # frames pointing at the hang site; the retry-ladder cap turns a silent
    # ~53-min rollout-report stall into an ERROR-level exception log.
    from rl_posttrain.rewards.code_reward_entry import _StackSampler

    _sample_s = float(os.getenv("CODE_REWARD_STACK_SAMPLE_S", "300"))
    if _sample_s > 0:
        _StackSampler(interval_s=_sample_s).start()
        logger.warning("[llm_judge] stack sampler armed, every %.0fs", _sample_s)
    try:
        from cosmos_rl.utils import constant as _cosmos_constant

        _prev = _cosmos_constant.COSMOS_HTTP_RETRY_CONFIG.max_retries
        _cap = int(os.getenv("CODE_REWARD_HTTP_MAX_RETRIES", "20"))
        _cosmos_constant.COSMOS_HTTP_RETRY_CONFIG.max_retries = _cap
        logger.warning(
            "[llm_judge] COSMOS_HTTP_RETRY_CONFIG.max_retries %d -> %d "
            "(caps the rollout-report stall at ~1.9 min instead of ~52.9 min)",
            _prev,
            _cap,
        )
    except Exception:
        logger.exception("[llm_judge] could not cap COSMOS_HTTP_RETRY_CONFIG.max_retries")

    _launch_with_scene_reference(REASONING_VLA_SPEC)
