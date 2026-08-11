# SPDX-License-Identifier: Apache-2.0
"""
masked_model_a2.py — CoT-masked Alpamayo 2 Super for open-loop semantic-action alignment.

Port of masking/masked_model.py (MaskedAlpamayo1_5) to the Alpamayo 2 Super
release (NVlabs/alpamayo2, nvidia/Alpamayo2-Super: 32B Cosmos VLM + 2B
diffusion Action Expert). The rollout structure is the same as 1.5 — VLM
generates the Chain-of-Causation, then the diffusion expert denoises action
chunks while cross-attending over the VLM's KV cache through an explicit
additive attention mask — so the masking method carries over unchanged:
knock reasoning columns out of that mask and re-denoise.

What moved between 1.5 and 2 (why this is a separate class, not a patch):
  - expert components live under `self.expert.*` (ExpertModel wrapper):
    action_space, diffusion, action_in_proj, action_out_proj, and the expert
    LLM itself at `self.expert.expert`
  - fuse_traj_tokens / find_eos_offset / build_expert_pos_ids_and_attn_mask /
    replace_padding_after_eos are free functions, not methods
  - trajectory special-token ids come from `self.config.traj_ids` (built off
    SPECIAL_TOKENS), not to_special_token()
  - generation prefills the prompt once via `_generate_with_shared_prefill`
  - upstream masks text-EOS ids during generation (_append_text_eos_mask)

The public analysis API is IDENTICAL to MaskedAlpamayo1_5, so
masking/training/run.py's experiment functions and experiment_d_reversal.py
run against either class:
  _rollout_prefix / _denoise_with_mask / compare_conditions /
  salience_leave_one_word_out / denorm_action / _reasoning_span /
  _cot_special_ids, plus shims (`action_space`, `_find_eos_offset`,
  `_build_expert_pos_ids_and_attn_mask`) covering the 1.5 methods that
  experiment_d_reversal.py calls.

Reasoning-span note: Alpamayo 2's training template wraps assistant CoT in
<|cot_start|>...<|cot_end|>, so generated sequences should carry the same
markers as 1.5. If a checkpoint generates without markers, the span falls
back to [prompt_len, <|traj_future_start|>) — prompt_len is threaded through
the prefix dict. With neither markers nor prompt_len we raise instead of
guessing (a wrong span would silently mask the prompt).

Same as the 1.5 fork, review before stating results: `mask_spec="none"` is a
fork of upstream, not upstream itself. Confirm it reproduces stock
`Alpamayo2Super.sample_trajectories_from_data` numerically (same seed ->
same trajectory) before reporting any deltas.

Analysis paths assume num_traj_samples == 1 and B == 1 so token positions in
`sequences` align 1:1 with KV-cache columns; asserted where it matters.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import einops
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo2_super.models.alpamayo2_super import (
    Alpamayo2Super,
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.expert_utils import (
    StopAfterEOS,
    build_expert_pos_ids_and_attn_mask,
    find_eos_offset,
    replace_padding_after_eos,
)
from alpamayo2_super.models.utils import SPECIAL_TOKENS, fuse_traj_tokens

logger = logging.getLogger(__name__)


class MaskedAlpamayo2Super(Alpamayo2Super):
    """Alpamayo 2 Super with reasoning/word knockout in the diffusion expert's attention."""

    # ------------------------------------------------------------------ #
    # 1.5-API shims used by experiment_d_reversal.py                      #
    # ------------------------------------------------------------------ #
    @property
    def action_space(self):
        """1.5 exposed action_space on the model; 2 nests it in ExpertModel."""
        return self.expert.action_space

    @staticmethod
    def _find_eos_offset(*, sequences, eos_token_id, device):
        return find_eos_offset(sequences=sequences, eos_token_id=eos_token_id, device=device)

    @staticmethod
    def _build_expert_pos_ids_and_attn_mask(
        *, offset, rope_deltas, kv_cache_seq_len, n_diffusion_tokens, b_star, device,
        prefix_mask=None,
    ):
        return build_expert_pos_ids_and_attn_mask(
            offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=kv_cache_seq_len,
            n_diffusion_tokens=n_diffusion_tokens, b_star=b_star, device=device,
            prefix_mask=prefix_mask,
        )

    # ------------------------------------------------------------------ #
    # Reasoning-span / word bookkeeping (operate on generated token ids)  #
    # ------------------------------------------------------------------ #
    def _cot_special_ids(self) -> dict[str, int]:
        return {
            name: self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[name])
            for name in ("cot_start", "cot_end", "traj_future_start")
        }

    def _reasoning_span(self, seq: torch.Tensor, prompt_len: int | None = None) -> tuple[int, int]:
        """[start, end) token columns holding the chain-of-causation CONTENT.

        Strictly between <|cot_start|> and <|cot_end|> (markers themselves stay
        visible). End falls back to <|traj_future_start|>. Start falls back to
        prompt_len (everything generated before the future-traj marker is CoT);
        with neither markers nor prompt_len we raise — a 0-start fallback like
        1.5's would silently include the whole multi-camera prompt.
        """
        sid = self._cot_special_ids()
        cs = (seq == sid["cot_start"]).nonzero(as_tuple=True)[0]
        ce = (seq == sid["cot_end"]).nonzero(as_tuple=True)[0]
        ts = (seq == sid["traj_future_start"]).nonzero(as_tuple=True)[0]
        if len(cs):
            start = int(cs[0]) + 1
        elif prompt_len is not None:
            start = int(prompt_len)
        else:
            raise ValueError(
                "No <|cot_start|> marker in sequence and no prompt_len provided; "
                "refusing to guess the reasoning span."
            )
        end = int(ce[0]) if len(ce) else (int(ts[0]) if len(ts) else int(seq.shape[0]))
        return start, max(start, end)

    def _reasoning_columns(
        self, seq: torch.Tensor, prompt_len: int | None = None
    ) -> torch.Tensor:
        start, end = self._reasoning_span(seq, prompt_len)
        return torch.arange(start, end, device=seq.device)

    def _word_groups(
        self, seq: torch.Tensor, prompt_len: int | None = None
    ) -> list[dict[str, Any]]:
        """Group the reasoning span's sub-word tokens into WHOLE words.

        Byte-level BPE marks a new word with a leading space when a single token
        is decoded (holds for the Qwen-family tokenizer Alpamayo 2 uses, same as
        1.5). Returns one dict per word: {"text", "cols", "norm"}.
        """
        start, end = self._reasoning_span(seq, prompt_len)
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

    def _concept_columns(
        self, seq: torch.Tensor, concepts: list[str], prompt_len: int | None = None
    ) -> torch.Tensor:
        """All reasoning columns belonging to whole words matching any concept."""
        targets = [c.strip().lower() for c in concepts if c.strip()]
        cols: list[int] = []
        for w in self._word_groups(seq, prompt_len):
            if any(t in w["norm"] for t in targets):
                cols.extend(int(c) for c in w["cols"])
        return torch.tensor(sorted(set(cols)), device=seq.device, dtype=torch.long)

    def _prefix_mask_columns(
        self, seq: torch.Tensor, n: int, unit: str = "tokens", prompt_len: int | None = None
    ) -> torch.Tensor:
        """Columns to MASK so the expert sees only the first n tokens/words of reasoning."""
        start, end = self._reasoning_span(seq, prompt_len)
        if unit == "tokens":
            cutoff = min(start + n, end)
            return torch.arange(cutoff, end, device=seq.device)
        if unit == "words":
            words = self._word_groups(seq, prompt_len)
            if n >= len(words):
                return torch.tensor([], device=seq.device, dtype=torch.long)
            cols: list[int] = []
            for w in words[n:]:
                cols.extend(int(c) for c in w["cols"])
            return torch.tensor(sorted(cols), device=seq.device, dtype=torch.long)
        raise ValueError(f"unknown unit: {unit!r}")

    def _suffix_mask_columns(
        self, seq: torch.Tensor, n: int, unit: str = "tokens", prompt_len: int | None = None
    ) -> torch.Tensor:
        """Columns to MASK so the expert sees only reasoning tokens/words from n onward."""
        start, end = self._reasoning_span(seq, prompt_len)
        if unit == "tokens":
            cutoff = min(start + n, end)
            return torch.arange(start, cutoff, device=seq.device)
        if unit == "words":
            words = self._word_groups(seq, prompt_len)
            n_clamp = min(n, len(words))
            if n_clamp == 0:
                return torch.tensor([], device=seq.device, dtype=torch.long)
            cols: list[int] = []
            for w in words[:n_clamp]:
                cols.extend(int(c) for c in w["cols"])
            return torch.tensor(sorted(cols), device=seq.device, dtype=torch.long)
        raise ValueError(f"unknown unit: {unit!r}")

    def _cols_for_spec(
        self, seq: torch.Tensor, spec: dict[str, Any], prompt_len: int | None = None
    ) -> torch.Tensor | None:
        mode = spec.get("mode", "none")
        if mode == "none":
            return None
        if mode == "reasoning":
            return self._reasoning_columns(seq, prompt_len)
        if mode == "concept":
            return self._concept_columns(seq, spec.get("concepts", []), prompt_len)
        if mode == "explicit":  # caller supplies columns directly (leave-one-out)
            return spec["cols"]
        if mode == "prefix":
            return self._prefix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"), prompt_len)
        if mode == "suffix":
            return self._suffix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"), prompt_len)
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
        """Faithful fork of Alpamayo2Super.sample_trajectories_from_data UP TO
        mask construction. Runs the VLM CoC generation and builds the expert
        position ids + base (unmasked) attention mask.
        """
        if not self.config.enable_expert:
            raise ValueError("Masking analysis requires an expert-enabled checkpoint.")

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

        tokenized_data = dict(data["tokenized_data"])
        traj_data = {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        input_ids = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )
        tokenized_data["input_ids"] = input_ids
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])

        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.max_new_tokens = max_generation_length or max(
            256, self.config.tokens_per_future_traj
        )
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=min(
                        self.config.traj_ids["history_id0"],
                        self.config.traj_ids["future_id0"],
                    ),
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        eos_token_id = self.config.traj_ids["future_start"]
        _append_text_eos_mask(
            logits_processor,
            generation_config.eos_token_id,
            preserved_token_id=eos_token_id,
        )
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        vlm_outputs = self._generate_with_shared_prefill(
            tokenized_data,
            generation_config=generation_config,
            n_samples_total=n_samples_total,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()
        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.expert.action_space.get_action_space_dims()[0]

        offset = find_eos_offset(
            sequences=vlm_outputs.sequences, eos_token_id=eos_token_id, device=device
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        # Decode the reasoning span directly rather than going through
        # extract_text_tokens(): its assistant-text split leaves the literal
        # "<|cot_start|>"/"<|cot_end|>" marker strings inside the cot field.
        seq0 = vlm_outputs.sequences[0]
        rs, re_ = self._reasoning_span(seq0, prompt_len)
        cot_text = self.tokenizer.decode(seq0[rs:re_], skip_special_tokens=False).strip()

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
            "prompt_len": prompt_len,
            "cot": {"cot": [cot_text]},  # same dict shape run.py already parses
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

        Returns (pred_xyz, pred_rot, action_raw) where action_raw is the
        NORMALIZED [accel, curvature] tensor before action_to_traj.
        """
        device = prefix["device"]
        cache = prefix["prompt_cache"]
        prefill = prefix["prefill_seq_len"]
        n_dt = prefix["n_diffusion_tokens"]
        pos = prefix["position_ids"]
        dims = self.expert.action_space.get_action_space_dims()

        am = prefix["attention_mask_base"].clone()
        if mask_cols is not None and len(mask_cols) > 0:
            neg = torch.finfo(am.dtype).min
            am[:, :, :, mask_cols] = neg  # b_star==1 assumed

        forward_kwargs = {}
        if self.expert.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            fte = self.expert.action_in_proj(x, t)
            if fte.dim() == 2:
                fte = fte.view(b, n_dt, -1)
            out = self.expert.expert(
                inputs_embeds=fte, position_ids=pos, past_key_values=cache,
                attention_mask=am, use_cache=True, **forward_kwargs,
            )
            cache.crop(prefill)  # restore cache length so prefix is reusable
            pred = self.expert.action_out_proj(out.last_hidden_state)
            return pred.view(-1, *dims)

        if seed is not None:  # common-random-numbers across conditions
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        total_batch = prefix["B"] * prefix["n_samples_total"]
        sampled = self.expert.diffusion.sample(
            batch_size=total_batch, step_fn=step_fn, device=device,
            return_all_steps=False, **(diffusion_kwargs or {}),
        )

        hist_xyz = einops.repeat(
            prefix["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=prefix["n_samples_total"]
        )
        hist_rot = einops.repeat(
            prefix["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=prefix["n_samples_total"]
        )
        pred_xyz, pred_rot = self.expert.action_space.action_to_traj(sampled, hist_xyz, hist_rot)
        ns, nj = prefix["num_traj_sets"], prefix["num_traj_samples"]
        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=ns, nj=nj)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=ns, nj=nj)
        return pred_xyz, pred_rot, sampled

    # ------------------------------------------------------------------ #
    # Physical-unit controls (curvature == steering, accel == long.)      #
    # ------------------------------------------------------------------ #
    def denorm_action(self, action_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map normalized [accel, curvature] -> physical units, per waypoint."""
        a = self.expert.action_space
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
        """Generate reasoning ONCE, then evaluate each named masking condition."""
        prefix = self._rollout_prefix(data, **rollout_kwargs)
        seq0 = prefix["sequences"][0]
        prompt_len = prefix["prompt_len"]
        words = self._word_groups(seq0, prompt_len)
        out: dict[str, Any] = {"cot": prefix["cot"], "words": words, "conditions": {}}
        for name, spec in conditions.items():
            cols = self._cols_for_spec(seq0, spec, prompt_len)
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
        """Per-word steering salience: baseline + drop each reasoning word once."""
        prefix = self._rollout_prefix(data, **rollout_kwargs)
        seq0 = prefix["sequences"][0]
        words = self._word_groups(seq0, prefix["prompt_len"])

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
                "traj_xy": xy[0, 0, 0].tolist(),
                "delta_xy_per_waypoint": delta_xy[0, 0, 0].tolist(),
            })
        ranked.sort(key=lambda r: r["d_curvature_mean_abs"], reverse=True)
        return {
            "cot": prefix["cot"],
            "baseline_curvature": base_curv,
            "baseline_xy": base_xy[0, 0, 0].tolist(),
            "words": ranked,
        }
