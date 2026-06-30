# SPDX-License-Identifier: Apache-2.0
"""
counterfactual_tokens.py — Logit-based counterfactual token analysis for CoT reasoning.

Two analyses:
  token_alternative_map(data)   — pure logit inspection, no re-running.
  counterfactual_sweep(data)    — forces top-K alternatives via LogitsProcessor,
                                   re-samples subsequent reasoning, compares trajectories.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from masking.masked_model import MaskedAlpamayo1_5

logger = logging.getLogger(__name__)


class ForcedTokenAtStep(LogitsProcessor):
    """Zero-out all logits except `token_id` at generation step `step`.

    Step is 0-indexed relative to the first NEW token (not prompt tokens).
    Append LAST in LogitsProcessorList so ExpertLogitsProcessor runs first.
    """

    def __init__(self, step: int, token_id: int) -> None:
        self.step = step
        self.token_id = token_id
        self._counter = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self._counter == self.step:
            scores[:] = torch.finfo(scores.dtype).min
            scores[:, self.token_id] = 0.0
        self._counter += 1
        return scores


# --------------------------------------------------------------------------- #
# Output data classes                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class AlternativeToken:
    token_id: int
    text: str
    prob: float   # softmax probability in the original (unforced) distribution


@dataclass
class ReasoningPosition:
    """Logit statistics for one generated reasoning token."""

    step: int            # 0-indexed generation step (= col - prompt_len)
    col: int             # absolute column in vlm_outputs.sequences
    sampled_id: int
    sampled_text: str
    sampled_prob: float  # rank-0 probability in the original distribution
    entropy: float       # Shannon entropy in nats
    top_k: list[AlternativeToken]  # sorted by prob descending; rank-0 = sampled token


@dataclass
class CounterfactualResult:
    """Trajectory delta from forcing one alternative token at one reasoning step."""

    forced_token: AlternativeToken
    forced_cot: str          # full CoT text when this token was forced
    d_curvature_mean: float  # mean |Δcurvature| over waypoints vs. baseline
    d_curvature_max: float
    endpoint_shift_m: float  # L2 distance of final waypoint vs. baseline (metres)
    traj_ade_m: float        # average displacement error over full trajectory (metres)


# --------------------------------------------------------------------------- #
# Analyzer                                                                      #
# --------------------------------------------------------------------------- #

class CounterfactualTokenAnalyzer(MaskedAlpamayo1_5):
    """Extends MaskedAlpamayo1_5 with logit-based counterfactual token analysis."""

    @torch.no_grad()
    def _extended_rollout_prefix(
        self,
        data: dict[str, Any],
        extra_logits_processors: list[LogitsProcessor] | None = None,
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        max_generation_length: int | None = None,
    ) -> dict[str, Any]:
        """Like _rollout_prefix but also captures `logits` and `prompt_len`.

        extra_logits_processors are appended AFTER ExpertLogitsProcessor so
        they take final precedence over the logit distribution.

        Extra keys in the returned dict vs. _rollout_prefix:
          logits     — tensor (n_gen_steps, vocab_size), post-processor, B=1 squeezed
          prompt_len — number of prompt tokens before generation started
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group supported."
        assert B == 1 and n_samples_total == 1, (
            "Analysis path assumes B==1 and num_traj_samples==1. "
            "Got B=%d n_samples=%d" % (B, n_samples_total)
        )

        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        prompt_len = int(input_ids.shape[1])
        input_ids = self.fuse_traj_tokens(
            input_ids,
            {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
        )
        device = input_ids.device

        if max_generation_length is None:
            max_generation_length = self.config.tokens_per_future_traj
        gen = self.vlm.generation_config
        gen.top_p, gen.temperature, gen.top_k = top_p, temperature, top_k
        gen.do_sample = True
        gen.num_return_sequences = num_traj_samples
        gen.max_new_tokens = max_generation_length
        gen.output_logits = True
        gen.return_dict_in_generate = True
        gen.pad_token_id = self.tokenizer.pad_token_id

        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        processors: list[LogitsProcessor] = [
            ExpertLogitsProcessor(
                traj_token_offset=self.config.traj_token_start_idx,
                traj_vocab_size=self.config.traj_vocab_size,
            )
        ]
        if extra_logits_processors:
            processors.extend(extra_logits_processors)
        logits_proc = LogitsProcessorList(processors)

        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=gen,
            stopping_criteria=stopping,
            logits_processor=logits_proc,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()
        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]

        offset = self._find_eos_offset(
            sequences=vlm_outputs.sequences, eos_token_id=eos_token_id, device=device
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        # Stack logits tuple of (B, V) → (n_gen_steps, V) for B==1.
        logits_tensor: torch.Tensor | None = None
        if vlm_outputs.logits:
            logits_tensor = torch.stack(vlm_outputs.logits, dim=0)  # (T, B, V)
            if logits_tensor.dim() == 3:
                logits_tensor = logits_tensor[:, 0, :]              # (T, V)

        return {
            "sequences": vlm_outputs.sequences,
            "prompt_cache": prompt_cache,
            "prefill_seq_len": prefill_seq_len,
            "n_diffusion_tokens": n_diffusion_tokens,
            "position_ids": position_ids,
            "attention_mask_base": attention_mask,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "B": B,
            "n_samples_total": n_samples_total,
            "num_traj_sets": num_traj_sets,
            "num_traj_samples": num_traj_samples,
            "device": device,
            "cot": extract_text_tokens(self.tokenizer, vlm_outputs.sequences),
            "logits": logits_tensor,   # (n_gen_steps, vocab_size) or None
            "prompt_len": prompt_len,
        }

    # ------------------------------------------------------------------ #
    # Logit-analysis helpers                                               #
    # ------------------------------------------------------------------ #

    def _reasoning_positions_with_logits(
        self,
        prefix: dict[str, Any],
        top_k: int = 5,
    ) -> list[ReasoningPosition]:
        """Build a ReasoningPosition for every token inside the CoT span.

        Maps absolute sequence columns to generation-step indices via:
            gen_step = col - prompt_len
        Columns that fall inside the prompt (gen_step < 0) are skipped —
        guards against off-by-one bugs if the reasoning span marker sits
        at the boundary of prompt vs. generated tokens.
        """
        seq0 = prefix["sequences"][0]   # (seq_len,)
        logits = prefix["logits"]       # (n_gen_steps, V)
        prompt_len = prefix["prompt_len"]

        if logits is None:
            raise RuntimeError(
                "logits not available — use _extended_rollout_prefix, not _rollout_prefix"
            )

        rs, re = self._reasoning_span(seq0)
        positions: list[ReasoningPosition] = []

        for col in range(rs, re):
            gen_step = col - prompt_len
            if gen_step < 0 or gen_step >= logits.shape[0]:
                continue

            step_logits = logits[gen_step]                              # (V,)
            probs = F.softmax(step_logits, dim=-1)
            entropy = float(-torch.sum(probs * torch.log(probs.clamp(min=1e-12))))

            top_probs, top_ids = probs.topk(min(top_k, probs.shape[-1]))
            top_tokens = [
                AlternativeToken(
                    token_id=int(tid),
                    text=self.tokenizer.decode([int(tid)], skip_special_tokens=False),
                    prob=float(p),
                )
                for tid, p in zip(top_ids.tolist(), top_probs.tolist())
            ]

            sampled_id = int(seq0[col])
            positions.append(ReasoningPosition(
                step=gen_step,
                col=col,
                sampled_id=sampled_id,
                sampled_text=self.tokenizer.decode([sampled_id], skip_special_tokens=False),
                sampled_prob=float(probs[sampled_id]),
                entropy=entropy,
                top_k=top_tokens,
            ))

        return positions

    @staticmethod
    def _select_positions(
        positions: list[ReasoningPosition],
        max_positions: int,
        method: str,
    ) -> list[ReasoningPosition]:
        """Return up to max_positions entries ranked by the given heuristic."""
        if method == "entropy":
            key = lambda p: p.entropy
        elif method == "prob_gap":
            # Runner-up (rank-1) probability: high → model nearly chose something else.
            key = lambda p: p.top_k[1].prob if len(p.top_k) > 1 else 0.0
        elif method == "all":
            return positions[:max_positions]
        else:
            raise ValueError(f"unknown position_selection: {method!r}")
        return sorted(positions, key=key, reverse=True)[:max_positions]
