# SPDX-License-Identifier: Apache-2.0
"""
ar_forced_rollout.py — force a (possibly perturbed) chain-of-causation text
into Alpamayo 1.5's reasoning span and sample N AUTOREGRESSIVE trajectory-token
continuations from it, decoded to xyz via the model's own DeltaTrajectoryTokenizer.

This is the Stage-2 core of the dpo_pairs counterfactual pipeline (see
.claude plan "Counterfactual Perturbation Generator → DPO Pairs"): the repo's
existing perturbation corpus (pref_pairs/results/perturbations/) has never
been fed back into the model — reasoning_matched_pairs.jsonl carries the
IDENTICAL action on both sides of every pair. This module closes that loop.

WHY THE AR TOKEN PATHWAY AND NOT THE DIFFUSION EXPERT: the RL recipe
(third_party/alpamayo-recipes/recipes/alpamayo1_x_rl) trains ONLY the
autoregressive trajectory-token head — GRPO logprobs are computed over
completion token sequences, and a future DPO trainer will too. A DPO pair
must therefore BE an AR token sequence. Measuring counterfactual effects on
the diffusion expert (masking/'s machinery) and re-encoding via
DeltaTrajectoryTokenizer.encode would (a) risk the tokenizer's ±4 m/step
clamp saturating silently and (b) measure a pathway the trainer never
touches — an effect present in the diffusion expert may simply not exist in
the AR head's distribution. dpo_pairs/run.py optionally runs a small
diffusion cross-check per scene to quantify exactly that gap.

Mechanics reused rather than reinvented:
  * masking.training.experiment_d_reversal.splice_reasoning builds the
    forced sequence — prompt + <|cot_start|> + tokenize(coc_text) +
    <|cot_end|>, truncated at the first <|traj_future_start|> — including
    its leading-space BPE convention fix and its `None` = keep-raw-ids mode
    (used here as the `control_rawids` condition, this pathway's analogue of
    experiment D's forced_orig machinery control).
  * The template generation below is the same recipe as
    pref_pairs.fixed_reasoning_rollout.generate_fixed_reasoning's generate
    call (fuse_traj_tokens + StopAfterEOS(traj_future_start) +
    ExpertLogitsProcessor), minus everything KV-cache/diffusion-related:
    here the sequence itself is the product, the cache is not reused.
  * alpamayo1_5.models.token_utils.extract_traj_tokens does the trajectory
    token extraction — it keys on the LAST <|traj_future_start|> in the
    sequence, so the fused history-trajectory tokens earlier in the prompt
    cannot confuse it, and it offset-normalizes into [0, traj_vocab_size).

TOKENIZATION SYMMETRY (load-bearing for DPO): the `clean` condition forces
the archived ground-truth text through splice_reasoning exactly like every
`perturbed__*` condition does — it does NOT reuse the template's raw
reasoning ids. Decode→encode round-trips are not guaranteed by the
tokenizer, so forcing text on both sides keeps chosen and rejected
completions tokenized by the identical path; the raw-ids control condition
exists precisely to measure the residual machinery error of that choice.

SEEDING: AR sampling has no per-call seed parameter — like
counterfactual/run.py's _seed_reasoning_rng, the caller seeds torch's global
RNG immediately before each generate call. Samples are drawn in chunks of
`sample_batch_size` (one generate call each, num_return_sequences=chunk) to
bound KV memory at N=20; chunk c of every condition is seeded with
seed_start + c, so conditions share RNG streams chunk-for-chunk
(common-random-numbers across conditions at chunk granularity — sampling is
still stochastic per token, so Stage-3 gating is distributional, never
per-sample-paired).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import einops
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from dpo_pairs.token_math import (  # noqa: F401 — re-exported for callers
    TrajTokenOnlyProcessor,
    assemble_completion_ids,
    count_generated_traj_tokens,
)
from masking.bootstrap import ensure_alpamayo1_5

ensure_alpamayo1_5()

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    extract_traj_tokens,
    replace_padding_after_eos,
    to_special_token,
)

from masking.training.experiment_d_reversal import splice_reasoning

logger = logging.getLogger(__name__)


def _special_ids(model) -> dict[str, int]:
    """cot/traj marker ids. MaskedAlpamayo1_5._cot_special_ids covers the cot
    side; traj_future_end is resolved the same way fixed_reasoning_rollout
    resolves traj_future_start."""
    ids = dict(model._cot_special_ids())
    for name in ("traj_future_start", "traj_future_end"):
        if name not in ids:
            ids[name] = model.tokenizer.convert_tokens_to_ids(to_special_token(name))
    return ids


def _seed_rng(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_template_sequence(
    model,
    tokenized_inputs: dict[str, Any],
    *,
    seed: int,
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
) -> dict[str, Any]:
    """One seeded VLM generation stopped at <|traj_future_start|>, returning
    the positional skeleton every forced condition splices into, plus the
    multimodal kwargs needed to re-generate from a forced prefix.

    Same recipe as fixed_reasoning_rollout.generate_fixed_reasoning (which is
    itself the independent port of masking's _rollout_prefix) with the
    KV-cache/diffusion plumbing dropped: the SEQUENCE is the product here.
    Asserts B==1 for the same reason those do.

    Returns dict with:
      seq0                (L,) token ids: fused prompt + generated reasoning,
                          ending at <|traj_future_start|> after padding strip
      mm_kwargs           tokenized_data minus input_ids — pixel values etc.,
                          reused verbatim by every forced generate call
      ego_history_xyz/rot original history tensors (decode is history-conditioned)
      self_generated_coc  the template's own CoC text — logged for drift QC
                          against the archived ground-truth trace (risk R7)
      prompt_len          fused prompt length (tokens before generation)
    """
    data = copy.deepcopy(tokenized_inputs)
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    if B != 1 or n_traj_group != 1:
        raise ValueError(
            f"build_template_sequence assumes B==1, n_traj_group==1 "
            f"(got B={B}, n_traj_group={n_traj_group}) — same constraint as "
            f"generate_fixed_reasoning, for the same alignment reasons."
        )

    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
    )
    prompt_len = int(input_ids.shape[1])

    gen = model.vlm.generation_config
    gen.top_p, gen.temperature, gen.top_k = top_p, temperature, top_k
    gen.do_sample = True
    gen.num_return_sequences = 1
    gen.max_new_tokens = model.config.tokens_per_future_traj
    gen.return_dict_in_generate = True
    gen.pad_token_id = model.tokenizer.pad_token_id

    sid = _special_ids(model)
    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=sid["traj_future_start"])])
    logits_proc = LogitsProcessorList(
        [ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )]
    )
    _seed_rng(seed)
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids, generation_config=gen,
        stopping_criteria=stopping, logits_processor=logits_proc, **tokenized_data,
    )
    sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=sid["traj_future_start"], pad_token_id=model.tokenizer.pad_token_id,
    )

    cot = extract_text_tokens(model.tokenizer, sequences)
    self_generated_coc = (cot.get("cot") or [""])[0].strip() if isinstance(cot, dict) else str(cot).strip()

    # attention_mask in tokenized_data covers only the prompt — forced
    # generate calls rebuild it at full forced-sequence length (B==1, no
    # padding, so all-ones is exact).
    mm_kwargs = {k: v for k, v in tokenized_data.items() if k != "attention_mask"}

    return {
        "seq0": sequences[0],
        "mm_kwargs": mm_kwargs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "self_generated_coc": self_generated_coc,
        "prompt_len": prompt_len,
    }


@torch.no_grad()
def sample_traj_tokens_given_coc(
    model,
    template: dict[str, Any],
    coc_text: str | None,
    *,
    n_samples: int,
    seed_start: int,
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    sample_batch_size: int = 5,
) -> dict[str, Any]:
    """Force coc_text into the template's reasoning span (None = keep raw ids,
    the control condition) and draw n_samples AR trajectory-token
    continuations. Returns
      {"coc_token_ids": [...],       # the spliced reasoning-span ids
       "forced_len": int,
       "samples": [{"sample_idx", "seed", "traj_token_ids",  # offset-normalized
                    "xyz",           # (T, 3) list, ego frame @ t0
                    "n_traj_tokens", "hit_traj_future_end"}]}
    Samples are drawn in ceil(n_samples / sample_batch_size) generate calls;
    chunk c is seeded seed_start + c (see module docstring's SEEDING note).
    """
    sid = _special_ids(model)
    device = template["seq0"].device

    forced_seq = splice_reasoning(model, template["seq0"], coc_text)  # (1, L)
    forced_len = int(forced_seq.shape[1])
    rs_start, rs_end = model._reasoning_span(forced_seq[0])
    coc_token_ids = forced_seq[0, rs_start:rs_end].tolist()

    gen = model.vlm.generation_config
    gen.top_p, gen.temperature, gen.top_k = top_p, temperature, top_k
    gen.do_sample = True
    gen.max_new_tokens = model.config.tokens_per_future_traj + 1  # traj tokens + end marker
    gen.return_dict_in_generate = True
    gen.pad_token_id = model.tokenizer.pad_token_id

    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=sid["traj_future_end"])])
    logits_proc = LogitsProcessorList(
        [TrajTokenOnlyProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
            traj_future_end_id=sid["traj_future_end"],
        )]
    )
    attention_mask = torch.ones_like(forced_seq)

    samples: list[dict[str, Any]] = []
    chunk_starts = list(range(0, n_samples, sample_batch_size))
    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk_n = min(sample_batch_size, n_samples - chunk_start)
        gen.num_return_sequences = chunk_n
        chunk_seed = seed_start + chunk_idx
        _seed_rng(chunk_seed)
        vlm_outputs = model.vlm.generate(
            input_ids=forced_seq, attention_mask=attention_mask,
            generation_config=gen, stopping_criteria=stopping,
            logits_processor=logits_proc, **template["mm_kwargs"],
        )
        sequences = vlm_outputs.sequences  # (chunk_n, forced_len + new)

        traj_tokens = extract_traj_tokens(
            sequences,
            special_token_ids=sid,
            tokens_per_future_traj=model.config.tokens_per_future_traj,
            future_token_start_idx=model.config.traj_token_start_idx,
            traj_tokenizer_vocab_size=model.config.traj_vocab_size,
        )  # (chunk_n, tokens_per_future_traj), offset-normalized

        hist_xyz = einops.repeat(
            template["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=chunk_n
        ).to(device)
        hist_rot = einops.repeat(
            template["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=chunk_n
        ).to(device)
        fut_xyz, _fut_rot, _ = model.traj_tokenizer.decode(
            hist_xyz=hist_xyz.float(), hist_rot=hist_rot.float(), tokens=traj_tokens,
        )  # (chunk_n, T, 3)

        for j in range(chunk_n):
            gen_part = sequences[j, forced_len:]
            n_traj, hit_end = count_generated_traj_tokens(
                gen_part,
                model.config.traj_token_start_idx,
                model.config.traj_vocab_size,
                sid["traj_future_end"],
            )
            samples.append({
                "sample_idx": chunk_start + j,
                "seed": chunk_seed,
                "traj_token_ids": traj_tokens[j].tolist(),
                "xyz": fut_xyz[j].float().cpu().round(decimals=4).tolist(),
                "n_traj_tokens": n_traj,
                "hit_traj_future_end": hit_end,
            })

    return {"coc_token_ids": coc_token_ids, "forced_len": forced_len, "samples": samples}
