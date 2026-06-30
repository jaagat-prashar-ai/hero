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
        """Like _rollout_prefix but also captures extra state needed for analysis.

        extra_logits_processors are appended AFTER ExpertLogitsProcessor so
        they take final precedence over the logit distribution.

        Extra keys in the returned dict vs. _rollout_prefix:
          logits          — tensor (n_gen_steps, vocab_size), post-processor, B=1 squeezed.
                            Indexed by generation step: logits[i] corresponds to
                            sequences[:, prompt_len + i].
          prompt_len      — number of prompt tokens before generation started. Used to
                            convert an absolute sequence column to a generation step index
                            via: gen_step = col - prompt_len.
          fused_input_ids — the prompt token IDs AFTER fuse_traj_tokens has been applied
                            (shape: 1 × prompt_len). Stored so Option-A single-token-swap
                            can reconstruct the full modified sequence for a VLM re-forward
                            without needing to re-fuse trajectory tokens.
          generate_kwargs — a snapshot of tokenized_data (minus input_ids, which was
                            popped) at the moment generate() was called. Contains the
                            attention mask, pixel values, and any other modality inputs
                            that must be passed again during a VLM re-forward in Option A.
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

        # Snapshot the fused prompt ids and the remaining generate inputs now,
        # before vlm.generate() consumes them. Both are needed by Option-A's
        # _reforward_with_single_swap to reconstruct the full modified sequence.
        fused_input_ids = input_ids.clone()          # (1, prompt_len) — prompt after trajectory fusion
        generate_kwargs = dict(tokenized_data)       # attention_mask, pixel_values, etc.

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
            "logits": logits_tensor,        # (n_gen_steps, vocab_size) or None
            "prompt_len": prompt_len,
            # --- Option-A re-forward state ---
            # These two fields are only used by _reforward_with_single_swap.
            # They are not needed for masking-style experiments.
            "fused_input_ids": fused_input_ids,   # (1, prompt_len)
            "generate_kwargs": generate_kwargs,   # attention_mask + vision inputs
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

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def token_alternative_map(
        self,
        data: dict[str, Any],
        top_k: int = 5,
        **rollout_kwargs: Any,
    ) -> dict[str, Any]:
        """Pure logit analysis — runs generation once, no counterfactual re-runs.

        For every generated reasoning token, returns the top-K candidate tokens
        and their softmax probabilities, plus per-step Shannon entropy. Use this
        first to survey which positions are "close calls" before committing to
        the more expensive counterfactual_sweep.

        Returns:
            {
              "cot": str,
              "positions": list[ReasoningPosition],
              "summary": {
                "n_reasoning_tokens": int,
                "mean_entropy": float,
                "highest_entropy_position": ReasoningPosition | None,
                "strongest_runner_up_position": ReasoningPosition | None,
              },
            }
        """
        prefix = self._extended_rollout_prefix(data, **rollout_kwargs)
        positions = self._reasoning_positions_with_logits(prefix, top_k=top_k)

        summary: dict[str, Any] = {
            "n_reasoning_tokens": len(positions),
            "mean_entropy": (
                float(sum(p.entropy for p in positions) / len(positions))
                if positions else 0.0
            ),
            "highest_entropy_position": (
                max(positions, key=lambda p: p.entropy) if positions else None
            ),
            "strongest_runner_up_position": (
                max(
                    positions,
                    key=lambda p: p.top_k[1].prob if len(p.top_k) > 1 else 0.0,
                )
                if positions else None
            ),
        }
        return {"cot": prefix["cot"], "positions": positions, "summary": summary}

    @torch.no_grad()
    def counterfactual_sweep(
        self,
        data: dict[str, Any],
        top_k_alternatives: int = 3,
        max_positions: int = 5,
        position_selection: str = "entropy",
        seed: int = 0,
        **rollout_kwargs: Any,
    ) -> dict[str, Any]:
        """For selected reasoning positions, force each top-K alternative and
        measure the trajectory delta vs. the baseline.

        For each (position, alternative) pair:
          1. Re-run VLM generation with ForcedTokenAtStep at that step.
             All subsequent reasoning tokens are re-sampled conditioned on the
             forced choice, so the counterfactual CoT is coherent rather than
             a splice.
          2. Run the diffusion expert with `seed` (common-random-numbers), so
             trajectory deltas are purely from the changed VLM KV-cache, not
             diffusion noise.

        Args:
            top_k_alternatives:  Runner-up tokens to test per position.
                                  The sampled token (rank 0) is always excluded.
            max_positions:       How many reasoning positions to analyse.
            position_selection:  "entropy" | "prob_gap" | "all"
            seed:                Diffusion seed shared across baseline and all CFs.

        Returns:
            {
              "baseline": {
                "cot": str,
                "controls": {"accel": Tensor, "curvature": Tensor},
                "pred_xyz": Tensor,
              },
              "positions": [
                {
                  "step": int,
                  "col": int,
                  "sampled_token": str,
                  "sampled_prob": float,
                  "entropy": float,
                  "alternatives": list[CounterfactualResult],
                },
                ...
              ],
            }
        """
        # --- baseline ---
        logger.info("counterfactual_sweep: running baseline generation...")
        prefix = self._extended_rollout_prefix(data, **rollout_kwargs)
        base_xyz, _, base_act = self._denoise_with_mask(prefix, mask_cols=None, seed=seed)
        base_controls = {k: v.float().cpu() for k, v in self.denorm_action(base_act).items()}
        base_xy = base_xyz[..., :2].float().cpu()

        # --- select positions ---
        # Request top_k_alternatives + 1 so the sampled token occupies rank 0
        # and we still have top_k_alternatives runner-ups available.
        all_positions = self._reasoning_positions_with_logits(
            prefix, top_k=top_k_alternatives + 1
        )
        selected = self._select_positions(all_positions, max_positions, position_selection)
        logger.info(
            "Selected %d/%d reasoning positions via '%s'",
            len(selected), len(all_positions), position_selection,
        )

        # --- per-position counterfactuals ---
        out_positions: list[dict[str, Any]] = []
        for pos in selected:
            alts = [t for t in pos.top_k if t.token_id != pos.sampled_id][:top_k_alternatives]
            cf_results: list[CounterfactualResult] = []

            for alt in alts:
                logger.info(
                    "  step=%d col=%d | forcing '%s' (p=%.3f) instead of '%s' (p=%.3f)",
                    pos.step, pos.col,
                    alt.text.strip(), alt.prob,
                    pos.sampled_text.strip(), pos.sampled_prob,
                )
                forcer = ForcedTokenAtStep(step=pos.step, token_id=alt.token_id)
                cf_prefix = self._extended_rollout_prefix(
                    data, extra_logits_processors=[forcer], **rollout_kwargs
                )
                cf_xyz, _, cf_act = self._denoise_with_mask(
                    cf_prefix, mask_cols=None, seed=seed
                )
                cf_controls = {
                    k: v.float().cpu() for k, v in self.denorm_action(cf_act).items()
                }
                cf_xy = cf_xyz[..., :2].float().cpu()

                d_curv = (cf_controls["curvature"] - base_controls["curvature"]).abs()
                cf_results.append(CounterfactualResult(
                    forced_token=alt,
                    forced_cot=cf_prefix["cot"],
                    d_curvature_mean=float(d_curv.mean()),
                    d_curvature_max=float(d_curv.max()),
                    endpoint_shift_m=float(
                        (cf_xy[..., -1, :] - base_xy[..., -1, :]).norm(dim=-1).mean()
                    ),
                    traj_ade_m=float((cf_xy - base_xy).norm(dim=-1).mean()),
                ))

            out_positions.append({
                "step": pos.step,
                "col": pos.col,
                "sampled_token": pos.sampled_text,
                "sampled_prob": pos.sampled_prob,
                "entropy": pos.entropy,
                "alternatives": cf_results,
            })

        return {
            "baseline": {
                "cot": prefix["cot"],
                "controls": base_controls,
                "pred_xyz": base_xyz.float().cpu(),
            },
            "positions": out_positions,
        }


# --------------------------------------------------------------------------- #
# Display helpers                                                               #
# --------------------------------------------------------------------------- #

def print_alternative_map(result: dict[str, Any], top_n: int = 10) -> None:
    """Pretty-print the output of token_alternative_map."""
    print(f"\n=== CoT ===\n{result['cot']}\n")
    s = result["summary"]
    print(
        f"Reasoning tokens : {s['n_reasoning_tokens']}\n"
        f"Mean entropy     : {s['mean_entropy']:.3f} nats"
    )
    if (p := s["highest_entropy_position"]):
        print(f"Highest entropy  : step {p.step}  '{p.sampled_text.strip()}'  H={p.entropy:.3f}")
    if (p := s["strongest_runner_up_position"]) and len(p.top_k) > 1:
        ru = p.top_k[1]
        print(
            f"Strongest runner-up: step {p.step}  "
            f"'{ru.text.strip()}' p={ru.prob:.3f}  "
            f"(sampled '{p.sampled_text.strip()}' p={p.sampled_prob:.3f})"
        )

    header = f"{'step':>5}  {'col':>5}  {'H':>6}  {'sampled':>22}  {'p_s':>6}  alternatives"
    print(f"\n{header}")
    print("-" * len(header))
    by_entropy = sorted(result["positions"], key=lambda p: p.entropy, reverse=True)[:top_n]
    for pos in by_entropy:
        alts = "  ".join(
            f"'{t.text.strip()}'({t.prob:.3f})"
            for t in pos.top_k
            if t.token_id != pos.sampled_id
        )
        print(
            f"{pos.step:>5}  {pos.col:>5}  {pos.entropy:>6.3f}  "
            f"{pos.sampled_text.strip():>22}  {pos.sampled_prob:>6.3f}  {alts}"
        )


def print_counterfactual_sweep(result: dict[str, Any]) -> None:
    """Pretty-print the output of counterfactual_sweep."""
    print(f"\n=== Baseline CoT ===\n{result['baseline']['cot']}\n")
    for pos in result["positions"]:
        print(
            f"\n── step {pos['step']}  col {pos['col']}  "
            f"sampled: '{pos['sampled_token'].strip()}'  "
            f"p={pos['sampled_prob']:.3f}  H={pos['entropy']:.3f}"
        )
        for cf in pos["alternatives"]:
            print(
                f"   → '{cf.forced_token.text.strip()}' (p={cf.forced_token.prob:.3f}) | "
                f"Δκ_mean={cf.d_curvature_mean:.4f}  "
                f"Δκ_max={cf.d_curvature_max:.4f}  "
                f"endpoint={cf.endpoint_shift_m:.3f} m  "
                f"ADE={cf.traj_ade_m:.3f} m"
            )
            preview = cf.forced_cot.replace("\n", " ")[:120]
            print(f"     CoT: {preview}...")
