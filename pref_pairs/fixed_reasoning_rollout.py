# SPDX-License-Identifier: Apache-2.0
"""
fixed_reasoning_rollout.py — a second rollout mode for the pref-pairs
faithfulness project: generate a scene's chain-of-causation (CoC) reasoning
ONCE, freeze it (keep its KV cache), then draw K diffusion-only trajectory
samples conditioned on that SAME fixed reasoning, varying only the diffusion
seed each draw.

Why this is a separate mode from rollout_harvester.py's harvest_scene: that
method resamples BOTH reasoning and diffusion independently per rollout, so
its K-rollout variance conflates two different noise sources. This module
isolates diffusion-only noise -- how much does the action vary when the
model's stated reasoning is held perfectly fixed -- as its own measurement,
directly comparable against rollout_harvester's compound-noise numbers on
the same scenes.

Design notes:

  * INDEPENDENT OF masking.masked_model.MaskedAlpamayo1_5 ON PURPOSE, same
    reason as rollout_harvester.py (see that module's docstring): this is a
    separate experiment, not a subclass/import of masking's model code.
    MaskedAlpamayo1_5's `_rollout_prefix` / `_denoise_with_mask` already do
    almost exactly this reasoning/diffusion split (for a different purpose:
    per-token attention masking, not reasoning-freezing), and everything
    they call to do it is a plain method/attribute already on the upstream
    `Alpamayo1_5` class (self.vlm, self._find_eos_offset,
    self._build_expert_pos_ids_and_attn_mask, self.expert,
    self.action_in_proj, self.action_out_proj, self.diffusion,
    self.action_space -- none of it masking-specific). generate_fixed_reasoning
    and sample_trajectory_given_fixed_reasoning below are independent ports
    of that same logic (mask-application code deleted, since nothing here
    ever masks anything -- attention is always full over the frozen
    reasoning), operating on a plain Alpamayo1_5 instance via composition,
    matching rollout_harvester.py's RolloutHarvester shape exactly. This
    duplicates real logic rather than importing it -- a bigger instance of
    the same trade-off rollout_harvester.py already makes with
    `_denorm_accel_curvature` (reimplementing 4 lines rather than importing
    masking's version), just applied to a larger, more load-bearing piece
    of code because there's no smaller upstream primitive to lean on.
  * SEQUENTIAL, NOT BATCHED. rollout_harvester.py's k-at-once batching works
    by generating K *different* reasoning sequences up front, then one
    combined diffusion call. Here there is exactly ONE reasoning sequence
    reused K times -- no batching trick for that shape exists anywhere in
    this codebase (checked). Each draw calls the diffusion expert once,
    then crops the KV cache back to its pristine post-reasoning length
    (cache.crop(prefill_seq_len)) so the next draw starts from the same
    frozen state -- same crop-and-reuse pattern as
    MaskedAlpamayo1_5.compare_conditions's per-condition loop.
  * NATIVE ACTION CAPTURE is direct here, no monkey-patching needed. Unlike
    rollout_harvester.py's `_sample_with_captured_action` (which has to spy
    on `action_to_traj` because it calls an opaque upstream convenience
    method), sample_trajectory_given_fixed_reasoning IS the diffusion step
    -- the raw normalized (accel, curvature) action tensor is already a
    local variable before action_to_traj is called, so it's just returned.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

import einops
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from masking.bootstrap import ensure_alpamayo1_5

ensure_alpamayo1_5()

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5, ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

from pref_pairs.rollout_harvester import (
    DEFAULT_CHECKPOINT,
    RolloutRecord,
    _denorm_accel_curvature,
    build_tokenized_inputs,
    load_alpamayo,
)

logger = logging.getLogger(__name__)


def generate_fixed_reasoning(
    model: Alpamayo1_5,
    tokenized_inputs: dict[str, Any],
    *,
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    max_generation_length: int | None = None,
) -> dict[str, Any]:
    """Generate a scene's CoC reasoning ONCE and return everything needed to
    denoise repeatedly against it. Independent port of
    masking/masked_model.py's `_rollout_prefix` with all mask-construction
    dropped (see module docstring) -- asserts B==1, num_traj_samples==1 for
    the same reason `_rollout_prefix` does: token columns must align with
    KV-cache columns for the crop-and-reuse trick in
    sample_trajectory_given_fixed_reasoning to work.
    """
    data = copy.deepcopy(tokenized_inputs)
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    if n_traj_group != 1:
        raise ValueError(f"Only one trajectory group supported, got n_traj_group={n_traj_group}")
    if B != 1:
        raise ValueError(
            f"generate_fixed_reasoning assumes B==1 so token columns align with "
            f"KV-cache columns. Got B={B}."
        )

    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
    )
    device = input_ids.device

    if max_generation_length is None:
        max_generation_length = model.config.tokens_per_future_traj
    gen = model.vlm.generation_config
    gen.top_p, gen.temperature, gen.top_k = top_p, temperature, top_k
    gen.do_sample = True
    gen.num_return_sequences = 1
    gen.max_new_tokens = max_generation_length
    gen.output_logits = True
    gen.return_dict_in_generate = True
    gen.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_proc = LogitsProcessorList(
        [ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )]
    )
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids, generation_config=gen,
        stopping_criteria=stopping, logits_processor=logits_proc, **tokenized_data,
    )
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id, pad_token_id=model.tokenizer.pad_token_id,
    )
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star = vlm_outputs.sequences.shape[0]
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]

    offset = model._find_eos_offset(
        sequences=vlm_outputs.sequences, eos_token_id=eos_token_id, device=device
    )
    prefix_mask = tokenized_data.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, 1, dim=0)
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=prefill_seq_len, n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star, device=device, prefix_mask=prefix_mask,
    )

    cot = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)

    return {
        "prompt_cache": prompt_cache,
        "prefill_seq_len": prefill_seq_len,
        "n_diffusion_tokens": n_diffusion_tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "B": B,
        "device": device,
        "cot": cot,
    }


@torch.no_grad()
def sample_trajectory_given_fixed_reasoning(
    model: Alpamayo1_5,
    prefix: dict[str, Any],
    seed: int | None,
    diffusion_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One diffusion-only draw against a frozen reasoning prefix (from
    generate_fixed_reasoning). Returns (pred_xyz, pred_rot, action_raw) --
    action_raw is the normalized [accel, curvature] tensor, shape
    (B, 1, 1, n_waypoints, 2), matching rollout_harvester.py's
    _harvest_batch indexing convention (B=1, one trajectory set, one
    trajectory sample per call). Independent port of
    masking/masked_model.py's `_denoise_with_mask` with mask_cols removed --
    attention is always the unmodified prefix mask, nothing is masked here.
    """
    device = prefix["device"]
    cache = prefix["prompt_cache"]
    prefill = prefix["prefill_seq_len"]
    n_dt = prefix["n_diffusion_tokens"]
    pos = prefix["position_ids"]
    am = prefix["attention_mask"]
    dims = model.action_space.get_action_space_dims()

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        fte = model.action_in_proj(x, t)
        if fte.dim() == 2:
            fte = fte.view(b, n_dt, -1)
        out = model.expert(
            inputs_embeds=fte, position_ids=pos, past_key_values=cache,
            attention_mask=am, use_cache=True, **forward_kwargs,
        )
        cache.crop(prefill)  # restore cache length so the prefix is reusable next draw
        last_hidden = out.last_hidden_state[:, -n_dt:]
        return model.action_out_proj(last_hidden).view(-1, *dims)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    sampled = model.diffusion.sample(
        batch_size=prefix["B"], step_fn=step_fn, device=device,
        return_all_steps=False, **(diffusion_kwargs or {}),
    )

    hist_xyz = einops.repeat(prefix["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=1)
    hist_rot = einops.repeat(prefix["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=1)
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled, hist_xyz, hist_rot)
    return pred_xyz, pred_rot, sampled


class FixedReasoningHarvester:
    """Loads Alpamayo 1.5 once and, per scene, generates one fixed reasoning
    then draws num_draws diffusion-only samples against it."""

    def __init__(self, model: Alpamayo1_5, device: str = "cuda") -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(cls, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda") -> "FixedReasoningHarvester":
        return cls(model=load_alpamayo(checkpoint, device), device=device)

    def harvest_scene(
        self,
        model_inputs: dict[str, Any],
        scene_id: str,
        *,
        num_draws: int = 100,
        seed_start: int = 0,
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        ground_truth_coc: str | None = None,
    ) -> list[RolloutRecord]:
        """Sample num_draws diffusion-only rollouts sharing ONE fixed
        reasoning for this scene. Returns one RolloutRecord per draw, with
        coc_text identical across all of them (that's the point) --
        RolloutRecord's schema needs no change for this, see module
        docstring. sampling_params carries reasoning_fixed=True and
        num_draws so a downstream reader can distinguish this mode's rows
        from rollout_harvester's compound-noise rows if files were ever mixed.
        """
        tokenized_inputs = build_tokenized_inputs(self.model, model_inputs, self.device)

        with torch.autocast(self.device, dtype=torch.bfloat16):
            prefix = generate_fixed_reasoning(
                self.model, tokenized_inputs, top_p=top_p, top_k=top_k, temperature=temperature,
            )

        # extract_text_tokens (called inside generate_fixed_reasoning) returns
        # dict[str, list[str]] -- {"cot": [...], "meta_action": [...], "answer": [...]},
        # one string per sequence in the batch. num_return_sequences=1 above,
        # so there's exactly one entry to pull out here.
        coc_text = prefix["cot"]["cot"][0]
        hz = 10.0  # same fixed convention as rollout_harvester.py -- see its comment
        model_version = getattr(self.model.config, "_name_or_path", DEFAULT_CHECKPOINT)
        sampling_params = {
            "seed_start": seed_start,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_draws": num_draws,
            "reasoning_fixed": True,
        }

        records: list[RolloutRecord] = []
        for i in range(num_draws):
            with torch.autocast(self.device, dtype=torch.bfloat16):
                pred_xyz, _pred_rot, action_raw = sample_trajectory_given_fixed_reasoning(
                    self.model, prefix, seed=seed_start + i,
                )
            # pred_xyz shape (1, T, 3): unlike _denoise_with_mask (which
            # rearranges "(b ns nj) ... -> b ns nj ..." for multi-sample
            # batching), this port never batches multiple samples in one
            # diffusion.sample() call -- B==1, one draw at a time -- so
            # there's no ns/nj dims to split out, just the plain batch dim.
            waypoints = pred_xyz[0].float().cpu()  # (T, 3)
            accel_t, kappa_t = _denorm_accel_curvature(self.model.action_space, action_raw)

            records.append(
                RolloutRecord(
                    scene_id=scene_id,
                    rollout_id=i,
                    coc_text=coc_text,
                    waypoints=waypoints.tolist(),
                    hz=hz,
                    sampling_params=sampling_params,
                    model_version=model_version,
                    ground_truth_coc=ground_truth_coc,
                    native_accel_mps2=accel_t[0].float().cpu().tolist(),
                    native_curvature_per_m=kappa_t[0].float().cpu().tolist(),
                )
            )
        return records
