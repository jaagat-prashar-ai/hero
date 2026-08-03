# SPDX-License-Identifier: Apache-2.0
"""
token_math.py — the pure token-level pieces of dpo_pairs.ar_forced_rollout,
split into their own module so unit tests can import them WITHOUT pulling in
masking/alpamayo1_5 (whose module-level imports need the pinned
transformers==4.57.1 environment and, transitively, the model code). Imports
here are torch + transformers.LogitsProcessor only — runnable in any venv
with torch, which is what makes local `pytest dpo_pairs/` possible on a box
that can't load the model (same motivation as pref_pairs' _FakeModel test
pattern, solved by module layout instead of mocks).
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class TrajTokenOnlyProcessor(LogitsProcessor):
    """Inverse of alpamayo1_5's ExpertLogitsProcessor: that one masks the
    trajectory-token block OUT (so CoC generation can't emit trajectory
    tokens); this one masks everything EXCEPT the trajectory-token block and
    <|traj_future_end|> out, so a continuation from a forced
    ...<|traj_future_start|> prefix can only emit trajectory tokens and then
    terminate. Without this, a forced-perturbed prefix occasionally continues
    in TEXT (the model 'reopens' reasoning), which would silently decode to
    the zero-token trajectory."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int, traj_future_end_id: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size
        self.traj_future_end_id = traj_future_end_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        allowed[self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = True
        allowed[self.traj_future_end_id] = True
        scores[:, ~allowed] = float("-inf")
        return scores


def count_generated_traj_tokens(
    generated_ids: list[int] | torch.Tensor,
    traj_token_offset: int,
    traj_vocab_size: int,
    traj_future_end_id: int,
) -> tuple[int, bool]:
    """(n_traj_tokens_before_end, hit_traj_future_end) for ONE sample's
    generated-part ids. Counts only tokens strictly before the first
    <|traj_future_end|>; tokens after it (possible, since
    TrajTokenOnlyProcessor keeps allowing trajectory tokens after the end
    marker) are deliberately not counted — extract_traj_tokens ignores them
    too, so this count matches what actually gets decoded."""
    if isinstance(generated_ids, torch.Tensor):
        generated_ids = generated_ids.tolist()
    n = 0
    for tok in generated_ids:
        if tok == traj_future_end_id:
            return n, True
        if traj_token_offset <= tok < traj_token_offset + traj_vocab_size:
            n += 1
    return n, False


def assemble_completion_ids(
    coc_token_ids: list[int],
    traj_token_ids_norm: list[int],
    special_ids: dict[str, int],
    traj_token_offset: int,
) -> list[int]:
    """Assemble the full DPO completion token sequence:
    [cot_start] + coc ids + [cot_end, traj_future_start] + raw traj ids +
    [traj_future_end]. traj_token_ids_norm is extract_traj_tokens' output
    (offset-normalized into [0, traj_vocab_size)), so the vocabulary offset
    is added back here."""
    return (
        [special_ids["cot_start"]]
        + list(coc_token_ids)
        + [special_ids["cot_end"], special_ids["traj_future_start"]]
        + [t + traj_token_offset for t in traj_token_ids_norm]
        + [special_ids["traj_future_end"]]
    )
