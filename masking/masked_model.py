# SPDX-License-Identifier: Apache-2.0
"""
masked_model.py — CoT-masked Alpamayo 1.5 for open-loop semantic-action alignment.

- Imported:
    - alpamayo1_5.models.alpamayo1_5.Alpamayo1_5, ExpertLogitsProcessor
    - alpamayo1_5.models.token_utils.{to_special_token, StopAfterEOS,
          replace_padding_after_eos, extract_text_tokens}

 
    - We split the "sample_trajectories_from_data_with_vlm_rollout_ into "_rollout_prefix" and "_denoise_with_mask"
    - A mask can be injected into the expert's cross-attention

    # Other changes (have to review this later)
    - whole-WORD masking of the reasoning span (fixes the per-subword-token
      substring bug in the original draft)
    - returns the raw [accel, curvature] action so steering/throttle deltas are
      read directly (curvature == steering, accel == throttle/brake)
    - `compare_conditions` generates the reasoning ONCE and applies every mask to
      the SAME generation → conditions differ only in the expert's attention,
      not in sampled reasoning text (clean controlled comparison)
    - `salience_leave_one_word_out` for per-word steering salience (experiment b)

review this before we actually state results:
`mask_spec="none"` is a fork of upstream, not upstream itself. Confirm it
reproduces the stock `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout`
numerically (same seed -> same trajectory) before reporting any deltas. See
run_masked_openloop.py --validate.

# Analysis paths assuem that num_traj_samples == 1 and B == 1 so that the token positions in 
# "sequences" align 1:1 with KV-cache columns (we prevent the left-adding offset). asserts this where it matters.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import einops
import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5, ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class MaskedAlpamayo1_5(Alpamayo1_5):
    """Alpamayo 1.5 with reasoning/word knockout in the diffusion expert's attention."""

    # Reasoning-span / word bookkeeping (operate on generated token ids)  #
    def _cot_special_ids(self) -> dict[str, int]:
        return {
            name: self.tokenizer.convert_tokens_to_ids(to_special_token(name))
            for name in ("cot_start", "cot_end", "traj_future_start")
        }

    def _reasoning_span(self, seq: torch.Tensor) -> tuple[int, int]:
        """[start, end) token columns holding the chain-of-causation CONTENT.

        Strictly between <|cot_start|> and <|cot_end|> (markers themselves stay
        visible). Falls back to <|traj_future_start|> if <|cot_end|> is absent.
        """
        sid = self._cot_special_ids()
        cs = (seq == sid["cot_start"]).nonzero(as_tuple=True)[0]
        ce = (seq == sid["cot_end"]).nonzero(as_tuple=True)[0]
        ts = (seq == sid["traj_future_start"]).nonzero(as_tuple=True)[0]
        start = int(cs[0]) + 1 if len(cs) else 0
        end = int(ce[0]) if len(ce) else (int(ts[0]) if len(ts) else int(seq.shape[0]))
        return start, max(start, end)

    def _reasoning_columns(self, seq: torch.Tensor) -> torch.Tensor:
        start, end = self._reasoning_span(seq)
        return torch.arange(start, end, device=seq.device)

    def _word_groups(self, seq: torch.Tensor) -> list[dict[str, Any]]:
        """Group the reasoning span's sub-word tokens into WHOLE words.

        Byte-level BPE marks a new word with a leading space when a single token
        is decoded. We group consecutive tokens until the next token begins a new
        word. Returns one dict per word: {"text", "cols" (LongTensor of columns)}.
        """
        start, end = self._reasoning_span(seq)
        words: list[dict[str, Any]] = []
        cur_text, cur_cols = "", []
        for j in range(start, end):
            piece = self.tokenizer.decode([int(seq[j])], skip_special_tokens=False)
            starts_word = len(piece) > 0 and piece[0].isspace()
            if starts_word and cur_cols:
                words.append(
                    {"text": cur_text, "cols": torch.tensor(cur_cols, device=seq.device)}
                )
                cur_text, cur_cols = "", []
            cur_text += piece
            cur_cols.append(j)
        if cur_cols:
            words.append(
                {"text": cur_text, "cols": torch.tensor(cur_cols, device=seq.device)}
            )
        for w in words:
            w["norm"] = w["text"].strip().lower()
        return words

    def _concept_columns(self, seq: torch.Tensor, concepts: list[str]) -> torch.Tensor:
        """All reasoning columns belonging to whole words matching any concept.

        Whole-word substring match (so "pedestrian" also catches "pedestrians").
        """
        targets = [c.strip().lower() for c in concepts if c.strip()]
        cols: list[int] = []
        for w in self._word_groups(seq):
            if any(t in w["norm"] for t in targets):
                cols.extend(int(c) for c in w["cols"])
        return torch.tensor(sorted(set(cols)), device=seq.device, dtype=torch.long)

    def _cols_for_spec(self, seq: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor | None:
        mode = spec.get("mode", "none")
        if mode == "none":
            return None
        if mode == "reasoning":
            return self._reasoning_columns(seq)
        if mode == "concept":
            return self._concept_columns(seq, spec.get("concepts", []))
        if mode == "explicit":  # caller supplies columns directly (leave-one-out)
            return spec["cols"]
        if mode == "prefix":
            return self._prefix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"))
        if mode == "suffix":
            return self._suffix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"))
        raise ValueError(f"unknown mask mode: {mode}")

    # ------------------------------------------------------------------ #
    # Rollout split: generate prompt ONCE, denoise many times             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _rollout_prefix(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        max_generation_length: int | None = None,
    ) -> dict[str, Any]:
        """Faithful fork of the upstream rollout UP TO mask construction.

        Runs the VLM reasoning generation and builds the expert position ids +
        base (unmasked) attention mask. Everything needed to denoise repeatedly.
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group supported."
        assert B == 1 and n_samples_total == 1, (
            "Analysis path assumes B==1 and num_traj_samples==1 so token columns "
            "align with KV-cache columns. Got B=%d n_samples=%d" % (B, n_samples_total)
        )

        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
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
        logits_proc = LogitsProcessorList(
            [ExpertLogitsProcessor(
                traj_token_offset=self.config.traj_token_start_idx,
                traj_vocab_size=self.config.traj_vocab_size,
            )]
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids, generation_config=gen,
            stopping_criteria=stopping, logits_processor=logits_proc, **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
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
            offset=offset, rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len, n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star, device=device, prefix_mask=prefix_mask,
        )

        cot = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)

        return {
            "sequences": vlm_outputs.sequences,
            "prompt_cache": prompt_cache,
            "prefill_seq_len": prefill_seq_len,
            "n_diffusion_tokens": n_diffusion_tokens,
            "position_ids": position_ids,
            "attention_mask_base": attention_mask,  # DO NOT mutate; clone per condition
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "B": B,
            "n_samples_total": n_samples_total,
            "num_traj_sets": num_traj_sets,
            "num_traj_samples": num_traj_samples,
            "device": device,
            "cot": cot,
        }

    @torch.no_grad()
    def _denoise_with_mask(
        self,
        prefix: dict[str, Any],
        mask_cols: torch.Tensor | None,
        seed: int | None = 0,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the diffusion expert with `mask_cols` knocked out of its attention.

        Returns (pred_xyz, pred_rot, action_raw) where action_raw is the NORMALIZED
        [accel, curvature] tensor of shape (B*, n_waypoints, 2) before action_to_traj.
        """
        device = prefix["device"]
        cache = prefix["prompt_cache"]
        prefill = prefix["prefill_seq_len"]
        n_dt = prefix["n_diffusion_tokens"]
        pos = prefix["position_ids"]
        dims = self.action_space.get_action_space_dims()

        am = prefix["attention_mask_base"].clone()
        if mask_cols is not None and len(mask_cols) > 0:
            neg = torch.finfo(am.dtype).min
            am[:, :, :, mask_cols] = neg  # b_star==1 assumed

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            fte = self.action_in_proj(x, t)
            if fte.dim() == 2:
                fte = fte.view(b, n_dt, -1)
            out = self.expert(
                inputs_embeds=fte, position_ids=pos, past_key_values=cache,
                attention_mask=am, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)  # restore cache length so prefix is reusable
            last_hidden = out.last_hidden_state[:, -n_dt:]
            return self.action_out_proj(last_hidden).view(-1, *dims)

        if seed is not None:  # common-random-numbers across conditions
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        total_batch = prefix["B"] * prefix["n_samples_total"]
        sampled = self.diffusion.sample(
            batch_size=total_batch, step_fn=step_fn, device=device,
            return_all_steps=False, **(diffusion_kwargs or {}),
        )

        hist_xyz = einops.repeat(
            prefix["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=prefix["n_samples_total"]
        )
        hist_rot = einops.repeat(
            prefix["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=prefix["n_samples_total"]
        )
        pred_xyz, pred_rot = self.action_space.action_to_traj(sampled, hist_xyz, hist_rot)
        ns, nj = prefix["num_traj_sets"], prefix["num_traj_samples"]
        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=ns, nj=nj)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=ns, nj=nj)
        return pred_xyz, pred_rot, sampled

    # ------------------------------------------------------------------ #
    # Physical-unit controls (curvature == steering, accel == long.)      #
    # ------------------------------------------------------------------ #
    def denorm_action(self, action_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map normalized [accel, curvature] -> physical units, per waypoint."""
        a = self.action_space
        accel = action_raw[..., 0] * a.accel_std.to(action_raw) + a.accel_mean.to(action_raw)
        kappa = action_raw[..., 1] * a.curvature_std.to(action_raw) + a.curvature_mean.to(action_raw)
        return {"accel": accel, "curvature": kappa}  # curvature = steering proxy [1/m]

    # ------------------------------------------------------------------ #
    # Public analysis API                                                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compare_conditions(
        self,
        data: dict[str, Any],
        conditions: dict[str, dict[str, Any]],
        seed: int = 0,
        **rollout_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate reasoning ONCE, then evaluate each named masking condition.

        conditions example:
            {"none": {"mode": "none"},
             "reasoning": {"mode": "reasoning"},
             "no_vru": {"mode": "concept",
                        "concepts": ["pedestrian", "cyclist", "crosswalk"]}}

        Returns dict with per-condition pred_xyz/pred_rot/action + the shared CoT
        text and word list, so every condition shares identical reasoning.
        """
        prefix = self._rollout_prefix(data, **rollout_kwargs)
        seq0 = prefix["sequences"][0]
        words = self._word_groups(seq0)
        out: dict[str, Any] = {"cot": prefix["cot"], "words": words, "conditions": {}}
        for name, spec in conditions.items():
            cols = self._cols_for_spec(seq0, spec)
            pxyz, prot, act = self._denoise_with_mask(prefix, cols, seed=seed)
            out["conditions"][name] = {
                "spec": spec,
                "n_masked_cols": 0 if cols is None else int(len(cols)),
                "pred_xyz": pxyz.float().cpu(),
                "pred_rot": prot.float().cpu(),
                "controls": {k: v.float().cpu() for k, v in self.denorm_action(act).items()},
            }
        return out

    @torch.no_grad()
    def salience_leave_one_word_out(
        self,
        data: dict[str, Any],
        seed: int = 0,
        **rollout_kwargs: Any,
    ) -> dict[str, Any]:
        """Per-word steering salience: baseline + drop each reasoning word once.

        Reasoning is generated once; for each word we re-run ONLY the diffusion
        expert with that word's full token span knocked out. Word importance =
        change in steering (curvature) / trajectory it induces.
        """
        prefix = self._rollout_prefix(data, **rollout_kwargs)
        seq0 = prefix["sequences"][0]
        words = self._word_groups(seq0)

        base_xyz, _, base_act = self._denoise_with_mask(prefix, None, seed=seed)
        base = self.denorm_action(base_act)
        base_curv = base["curvature"].float().cpu()  # (1, T)
        base_xy = base_xyz[..., :2].float().cpu()     # (1,1,1,T,2)

        ranked = []
        for w in words:
            pxyz, _, act = self._denoise_with_mask(prefix, w["cols"], seed=seed)
            c = self.denorm_action(act)["curvature"].float().cpu()
            xy = pxyz[..., :2].float().cpu()
            delta_xy = (xy - base_xy).norm(dim=-1)  # (1,1,1,T)
            ranked.append({
                "word": w["text"].strip(),
                "n_tokens": int(len(w["cols"])),
                "d_curvature_mean_abs": float((c - base_curv).abs().mean()),
                "d_curvature_max_abs": float((c - base_curv).abs().max()),
                "endpoint_shift_m": float((xy[..., -1, :] - base_xy[..., -1, :]).norm(dim=-1).mean()),
                "traj_ade_m": float(delta_xy.mean()),
                # (T, 2) waypoint path with this word masked out, so a caller
                # can render it against baseline_xy instead of only the
                # scalar deltas above.
                "traj_xy": xy[0, 0, 0].tolist(),
                # Per-waypoint distance to baseline (length T) -- traj_ade_m/
                # endpoint_shift_m are just this array's mean and last value.
                "delta_xy_per_waypoint": delta_xy[0, 0, 0].tolist(),
            })
        ranked.sort(key=lambda r: r["d_curvature_mean_abs"], reverse=True)
        return {
            "cot": prefix["cot"],
            "baseline_curvature": base_curv,
            "baseline_xy": base_xy[0, 0, 0].tolist(),
            "words": ranked,
        }

# Prefix / suffix threshold masking approach:

    def _prefix_mask_columns(
        self, seq: torch.Tensor, n: int, unit: str = "tokens"
    ) -> torch.Tensor:
        """Columns to MASK so that the expert sees only the first n tokens/words of reasoning

        The function returns reasoning columns after position n. With n==0, the full reasoning span is 
        masked; with n >= span length, nothing is masked (returns empty tensor).

        unit: "tokens" - n counts sub-word tokens within the reasoning span
               "words" - n counts whole words (as we did with initial masking approach), all tokens of words[n:] are masked

        # Locate the inclusive [start, end) column range of the reasoning span in 'seq'
        # 'start' is the column right after <|cot_start|>; "end" is the column of <|cot_end|> (or <|traj_future_start|> if the end marker is absent).

        
        """

        start, end = self._reasoning_span(seq)

        if unit == "tokens":
            # The cutoff colimn is start + n sub-word tokens into the reasoning span 
            # 'min(..., end)' clamps so we never go past the end of the reasoning span
            # which could silently mask tokens outside the CoT section
            cutoff = min(start+n, end)
            # Return every column from "cutoff" to "end," these are the tokens after 
            # the n-prefix token that we want to hide from the expert's attention
            return torch.arange(cutoff, end, device=seq.device)

        if unit == "words":
            # Re-group the reasoning sub-word tokens into whole words so that the cutoff falls 
            # on a clean word boundary rather than mid-word
            words = self._word_groups(seq)
            # If n is at least as large as the total number of words, entire prefix is "visible" and nothing is left to mask 
            # i.e., base case
            if n >= len(words):
                return torch.tensor([], device=seq.device, dtype=torch.long)
            # Collect every sub-word token column belonging to words[n:], the suffix that comes after the n-word prefix we want to keep visible

            cols: list[int] = []
            for w in words[n:]: # iterate over the suffix words 
                cols.extend(int(c) for c in w["cols"]) # flatten their token columns

            # Sort to keep columns in ascending order (requried by attention mask indexing in _denoise_with_mask) and return as a LongTensor.

            return torch.tensor(sorted(cols), device=seq.device, dtype=torch.long)  
        raise ValueError(f"unknown unit: {unit!r}")

    def _suffix_mask_columns(
        self, seq: torch.Tensor, n: int, unit: str = "tokens"
    ) -> torch.Tensor:
        """ Coilumns to MASK so that the expert sees only reasoning tokens/words from n onward

        Returns reasoning columns BEFORE position n. Complement of "_prefix_mask_columns".
        """

        # Locate the [start, end) column range of the reasoning span, same as above. 

        start, end = self._reasoning_span(seq)

        if unit == "tokens":
            # The cutoff column is the start + n tokens into the reasoning span 
            # min(..., end) prevents going past the reasoning span boundary
            cutoff = min(start+n, end)
            # Return columns from "start" up to (not including) "cutoff" - i.e., the 
            # first n tokens of the reasoning span that we want the expert to not see 
            # returns a 1D tensor
            return torch.arange(start, cutoff, device=seq.device)

        if unit == "words":
            words = self._word_groups(seq)
            n_clamp = min(n, len(words))
            if n_clamp == 0:
                return torch.tensor([], device=seq.device, dtype=torch.long)
            cols: list[int] = []
            for w in words[:n_clamp]:
                cols.extend(int(c) for c in w["cols"])
            return torch.tensor(sorted(cols), device=seq.device, dtype=torch.long)

        raise ValueError(f"unknown unit: {unit!r}")
