# SPDX-License-Identifier: Apache-2.0
"""
experiment_d_reversal.py — Experiment D: commitment/perceptual ORDER reversal.

Motivation (follows directly from Experiment C): the corpus CoC is
commitment-first ("Nudge left due to construction cones blocking ..."), and
Experiment C showed the diffusion expert's use of the reasoning is heavily
front-loaded — the first ~10 words recover ~80% of the full-reasoning effect.
If the expert is merely reading the maneuver off the OPENING of the CoT
(position), moving the commitment to the END of each beat should hurt as much
as suffix-masking the first words did (~0.135 m at n=5). If the expert
actually parses the semantics, a cause-fronted paraphrase ("Due to
construction cones blocking ..., nudge left") should barely move the
trajectory. That gap is what this experiment measures.

Why masking can't do this: reordering changes the token SEQUENCE itself, so
instead of knocking columns out of the expert's attention we must re-prefill
the VLM on a rewritten CoC ("forced decode") and then denoise. Three
conditions per snapshot, all denoised with the same seed (common random
numbers, same convention as masked_model.compare_conditions):

  generated     — the normal rollout (VLM generates the CoC, expert denoises)
  forced_orig   — the SAME token ids re-prefilled through the forced path.
                  Pure control: any ade vs `generated` is machinery error
                  (forward-pass vs generate numerics), NOT a reasoning effect.
                  This plays the same role --validate played for the masking
                  fork: without it, "reversal moved the trajectory" and "the
                  forcing path moved the trajectory" are indistinguishable.
  reversed      — every beat with a causal connective rewritten cause-first;
                  beats without a connective are left verbatim.

Primary metric: ade_reversal_m = ADE(reversed, forced_orig). Comparing against
forced_orig rather than `generated` cancels the forcing-path error common to
both forced conditions.

The beat rewriter reuses code_as_a_reward.coc_claim_parser's segmentation
(_split_beats / _split_beat) rather than re-deriving it: those regexes were
tuned against 2000+ real corpus CoC strings, and reusing them guarantees the
reversal splits beats exactly where the claim parser does.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

# Parser internals are imported at module scope on purpose: they are pure
# stdlib (re/dataclasses), so this cannot pull torch-heavy deps, and an
# import error (e.g. code_as_a_reward not packaged) should fail the workload
# loudly at import time, not per-snapshot inside the try/except in run.py.
from code_as_a_reward.coc_claim_parser import (
    _normalize_punctuation,
    _split_beat,
    _split_beats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text transform: commitment-first  ->  cause-first
# ---------------------------------------------------------------------------

def reverse_coc_text(text: str) -> dict[str, Any]:
    """Rewrite each causal beat of a CoC string cause-first.

    "Nudge left due to cones blocking the lane, then accelerate"
      -> "Due to cones blocking the lane, nudge left, then accelerate"

    Inter-beat delimiter text (";", " then", ". ") is preserved verbatim by
    slicing the gaps between beat spans rather than re-joining with a fixed
    separator — so the ONLY difference between input and output is the
    within-beat clause order. Beats with no connective pass through unchanged
    (they have no cause to front).

    Returns {"text", "n_beats", "n_beats_reversed"}. When n_beats_reversed
    is 0 the output text equals the input (modulo punctuation normalization)
    and the caller should skip the reversed condition rather than burn a
    forced prefill on an identity rewrite.
    """
    norm = _normalize_punctuation(text)
    parts: list[str] = []
    pos = 0
    n_reversed = 0
    beat_spans = _split_beats(norm)
    for span in beat_spans:
        s, e = span
        parts.append(norm[pos:s])  # delimiter gap before this beat, verbatim
        commitment_span, cause_span, connective = _split_beat(norm, span)
        if connective is None:
            parts.append(norm[s:e])
        else:
            commitment = norm[commitment_span[0] : commitment_span[1]].strip().rstrip(",")
            # strip(" ,") both ends: the corpus writes "because, after ..."
            # (comma directly after the connective), which would otherwise
            # front as "Because , after ...".
            cause = norm[cause_span[0] : cause_span[1]].strip(" ,")
            # Sentence-case bookkeeping: if the beat opened with a capital
            # (typical corpus style: "Nudge left due to ..."), the fronted
            # connective inherits the capital and the commitment is
            # lowercased — unless it starts with something like an acronym
            # (two leading capitals), which we leave alone.
            beat_text = norm[s:e].lstrip()
            if beat_text[:1].isupper():
                connective = connective[0].upper() + connective[1:]
                if commitment[:1].isupper() and not commitment[:2].isupper():
                    commitment = commitment[0].lower() + commitment[1:]
            parts.append(f"{connective} {cause}, {commitment}")
            n_reversed += 1
        pos = e
    parts.append(norm[pos:])
    return {
        "text": "".join(parts),
        "n_beats": len(beat_spans),
        "n_beats_reversed": n_reversed,
    }


# ---------------------------------------------------------------------------
# Forced-CoC rollout: re-prefill the VLM on a chosen token sequence
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_forced_cot(model, data: dict[str, Any], forced_seq: torch.Tensor) -> dict[str, Any]:
    """Build a `_denoise_with_mask`-compatible prefix dict from a FORCED
    token sequence instead of a generated one.

    `forced_seq` is (1, L): the full fused prompt + <|cot_start|> + reasoning
    tokens + <|cot_end|> + <|traj_future_start|>, ending exactly at the eos
    the expert offsets from. It is spliced from a real rollout's `sequences`
    (see splice_reasoning), so all special/fused-trajectory tokens are
    positionally identical to the generate path — only reasoning tokens
    differ.

    Mirrors masked_model._rollout_prefix after the generate() call: one VLM
    forward pass populates the KV cache (prefill), rope_deltas are read off
    the model exactly as the upstream rollout does, and the expert position
    ids / base attention mask are built with the same upstream helper.
    """
    # deepcopy for the same reason _rollout_prefix does: we pop input_ids /
    # attention_mask out of tokenized_data, and the caller reuses `data`
    # for the next condition.
    data = copy.deepcopy(data)
    tokenized_data = data["tokenized_data"]
    tokenized_data.pop("input_ids")  # replaced wholesale by forced_seq
    # The prompt-length mask is still what _build_expert_pos_ids_and_attn_mask
    # expects as prefix_mask (upstream passes the same thing on the generate
    # path); the full-length mask below is for the prefill forward only.
    prompt_mask = tokenized_data.pop("attention_mask", None)

    device = forced_seq.device
    assert forced_seq.shape[0] == 1, "forced path assumes B==1 like the analysis path"
    full_mask = torch.ones_like(forced_seq)

    out = model.vlm(
        input_ids=forced_seq,
        attention_mask=full_mask,
        use_cache=True,
        return_dict=True,
        **tokenized_data,  # pixel_values / image_grid_thw etc. — image token
                           # positions in forced_seq are untouched by the
                           # reasoning splice, so visual fusion lands where
                           # it did on the generate path
    )
    # Same source _rollout_prefix reads rope_deltas from after generate():
    # the VLM computes and caches them during prefill when position_ids
    # aren't supplied.
    rope_deltas = model.vlm.model.rope_deltas
    prompt_cache = out.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]

    eos_token_id = model._cot_special_ids()["traj_future_start"]
    offset = model._find_eos_offset(
        sequences=forced_seq, eos_token_id=eos_token_id, device=device
    )
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=1,
        device=device,
        prefix_mask=prompt_mask,
    )

    return {
        "sequences": forced_seq,
        "prompt_cache": prompt_cache,
        "prefill_seq_len": prefill_seq_len,
        "n_diffusion_tokens": n_diffusion_tokens,
        "position_ids": position_ids,
        "attention_mask_base": attention_mask,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "B": 1,
        "n_samples_total": 1,
        "num_traj_sets": 1,
        "num_traj_samples": 1,
        "device": device,
        "cot": None,  # caller already has the text; keep the key for parity
    }


def splice_reasoning(model, seq: torch.Tensor, new_cot_text: str | None) -> torch.Tensor:
    """Return a (1, L) forced sequence: `seq` (1D generated token ids) with
    its reasoning span replaced by the tokenization of `new_cot_text`, then
    truncated at the first <|traj_future_start|> so the sequence ends exactly
    at the eos `_find_eos_offset` keys on (drops any post-eos padding
    replace_padding_after_eos left behind).

    new_cot_text=None means "keep the original reasoning token ids verbatim"
    — used for the forced_orig control, where re-TOKENIZING the decoded text
    could itself introduce id-level drift (tokenizers don't guarantee
    decode->encode round-trips); splicing the raw ids keeps the control a
    pure test of the forward-pass machinery.
    """
    rs_start, rs_end = model._reasoning_span(seq)
    if new_cot_text is None:
        new_ids = seq[rs_start:rs_end]
    else:
        # Match the leading-space convention of the generated span: BPE
        # tokenizes " Due" and "Due" differently, and the span as generated
        # virtually always starts with a space-prefixed word piece.
        original_text = model.tokenizer.decode(seq[rs_start:rs_end], skip_special_tokens=False)
        if original_text.startswith(" ") and not new_cot_text.startswith(" "):
            new_cot_text = " " + new_cot_text
        new_ids = torch.tensor(
            model.tokenizer(new_cot_text, add_special_tokens=False).input_ids,
            device=seq.device,
            dtype=seq.dtype,
        )
    forced = torch.cat([seq[:rs_start], new_ids, seq[rs_end:]])
    eos_token_id = model._cot_special_ids()["traj_future_start"]
    eos_pos = (forced == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos):
        forced = forced[: int(eos_pos[0]) + 1]
    return forced.unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-snapshot experiment
# ---------------------------------------------------------------------------

def run_experiment_d(model, model_inputs: dict, seed: int) -> dict:
    """Reversal experiment for one snapshot. See module docstring.

    Output fields:
      cot / cot_reversed        — original + rewritten reasoning text
      n_beats / n_beats_reversed— rewriter bookkeeping (0 reversed => control-only row)
      ade_control_m             — ADE(forced_orig, generated): forcing-path error
      ade_reversal_m            — ADE(reversed, forced_orig): the effect (null if identity rewrite)
      ade_reversed_vs_generated_m, endpoint_reversal_m
      traj_generated_xy / traj_forced_orig_xy / traj_reversed_xy — (T,2) paths
      delta_xy_per_waypoint     — |reversed - forced_orig| per waypoint
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # 1. Normal rollout: generate the CoC once; denoise unmasked.
        prefix = model._rollout_prefix(model_inputs)
        gen_xyz, _, _ = model._denoise_with_mask(prefix, None, seed=seed)
        seq0 = prefix["sequences"][0]

        rs_start, rs_end = model._reasoning_span(seq0)
        cot_text = model.tokenizer.decode(
            seq0[rs_start:rs_end], skip_special_tokens=False
        ).strip()

        # 2. Rewrite cause-first.
        rev = reverse_coc_text(cot_text)

        # 3. Control: identical token ids through the forced-prefill path.
        forced_orig_seq = splice_reasoning(model, seq0, None)
        ctrl_prefix = rollout_forced_cot(model, model_inputs, forced_orig_seq)
        ctrl_xyz, _, _ = model._denoise_with_mask(ctrl_prefix, None, seed=seed)

        # 4. Reversed condition — skipped when the rewrite is an identity
        #    (no causal connective anywhere), since forcing the same tokens
        #    again would just reproduce the control.
        rev_xyz = None
        if rev["n_beats_reversed"] > 0:
            reversed_seq = splice_reasoning(model, seq0, rev["text"])
            rev_prefix = rollout_forced_cot(model, model_inputs, reversed_seq)
            rev_xyz, _, _ = model._denoise_with_mask(rev_prefix, None, seed=seed)

    gen_xy = gen_xyz[0, 0, 0].float().cpu().numpy()[:, :2]
    ctrl_xy = ctrl_xyz[0, 0, 0].float().cpu().numpy()[:, :2]

    def _ade(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
        T = min(len(a), len(b))
        d = np.linalg.norm(a[:T] - b[:T], axis=-1)
        return float(d.mean()), d

    ade_control, _ = _ade(ctrl_xy, gen_xy)

    row: dict[str, Any] = {
        "cot": cot_text,
        "cot_reversed": rev["text"].strip(),
        "n_beats": rev["n_beats"],
        "n_beats_reversed": rev["n_beats_reversed"],
        "ade_control_m": ade_control,
        "ade_reversal_m": None,
        "ade_reversed_vs_generated_m": None,
        "endpoint_reversal_m": None,
        "traj_generated_xy": gen_xy.round(4).tolist(),
        "traj_forced_orig_xy": ctrl_xy.round(4).tolist(),
        "traj_reversed_xy": None,
        "delta_xy_per_waypoint": None,
    }

    if rev_xyz is not None:
        rev_xy = rev_xyz[0, 0, 0].float().cpu().numpy()[:, :2]
        ade_reversal, delta = _ade(rev_xy, ctrl_xy)
        ade_rev_vs_gen, _ = _ade(rev_xy, gen_xy)
        row.update({
            "ade_reversal_m": ade_reversal,
            "ade_reversed_vs_generated_m": ade_rev_vs_gen,
            "endpoint_reversal_m": float(delta[-1]),
            "traj_reversed_xy": rev_xy.round(4).tolist(),
            "delta_xy_per_waypoint": delta.round(4).tolist(),
        })

    return row
