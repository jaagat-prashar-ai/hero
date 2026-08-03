# SPDX-License-Identifier: Apache-2.0
"""
token_math_test.py — unit tests for dpo_pairs.token_math, the pure
token-level pieces of the counterfactual measurement harness. No model, no
GPU, no network (per the project's no-fake-model-tests rule, the model path
in ar_forced_rollout.py is verified by a real max_scenes=1 smoke run, not
mocked here — these tests cover exactly the arithmetic this repo adds).

Toy vocabulary used throughout: 20 ids total, trajectory block = [10, 14)
(offset 10, vocab 4), traj_future_end = 17.
"""

from __future__ import annotations

import torch

from dpo_pairs.token_math import (
    TrajTokenOnlyProcessor,
    assemble_completion_ids,
    count_generated_traj_tokens,
)

_OFFSET, _VOCAB, _END = 10, 4, 17
_SPECIAL = {"cot_start": 1, "cot_end": 2, "traj_future_start": 3, "traj_future_end": _END}


class TestTrajTokenOnlyProcessor:
    def _scores(self) -> torch.Tensor:
        return torch.zeros(2, 20)  # batch of 2, all logits equal pre-mask

    def test_only_traj_block_and_end_survive(self):
        proc = TrajTokenOnlyProcessor(_OFFSET, _VOCAB, _END)
        out = proc(torch.zeros(2, 5, dtype=torch.long), self._scores())
        finite = torch.isfinite(out)
        expected = torch.zeros(20, dtype=torch.bool)
        expected[_OFFSET : _OFFSET + _VOCAB] = True
        expected[_END] = True
        assert torch.equal(finite[0], expected)
        assert torch.equal(finite[1], expected)  # same mask on every batch row

    def test_masked_positions_are_neg_inf_not_just_small(self):
        proc = TrajTokenOnlyProcessor(_OFFSET, _VOCAB, _END)
        out = proc(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 20))
        assert out[0, 0] == float("-inf")  # text token
        assert out[0, _OFFSET + _VOCAB] == float("-inf")  # first id past the block
        assert out[0, _OFFSET - 1] == float("-inf")  # last id before the block

    def test_allowed_scores_unchanged(self):
        proc = TrajTokenOnlyProcessor(_OFFSET, _VOCAB, _END)
        scores = torch.arange(20, dtype=torch.float).unsqueeze(0)
        out = proc(torch.zeros(1, 5, dtype=torch.long), scores.clone())
        assert torch.equal(out[0, _OFFSET : _OFFSET + _VOCAB], scores[0, _OFFSET : _OFFSET + _VOCAB])
        assert out[0, _END] == scores[0, _END]


class TestCountGeneratedTrajTokens:
    def test_clean_termination(self):
        ids = [10, 11, 12, 13, _END]
        assert count_generated_traj_tokens(ids, _OFFSET, _VOCAB, _END) == (4, True)

    def test_no_end_marker(self):
        ids = [10, 11, 12]
        assert count_generated_traj_tokens(ids, _OFFSET, _VOCAB, _END) == (3, False)

    def test_tokens_after_end_not_counted(self):
        # TrajTokenOnlyProcessor keeps allowing trajectory tokens after the
        # end marker — extract_traj_tokens ignores them, and so must this
        # count, or well-formedness stats would disagree with what decoded.
        ids = [10, 11, _END, 12, 13]
        assert count_generated_traj_tokens(ids, _OFFSET, _VOCAB, _END) == (2, True)

    def test_early_end_is_degenerate_but_reported(self):
        assert count_generated_traj_tokens([_END], _OFFSET, _VOCAB, _END) == (0, True)

    def test_accepts_tensor_input(self):
        ids = torch.tensor([10, 11, _END])
        assert count_generated_traj_tokens(ids, _OFFSET, _VOCAB, _END) == (2, True)


class TestAssembleCompletionIds:
    def test_layout_and_offset_restoration(self):
        out = assemble_completion_ids(
            coc_token_ids=[5, 6, 7],
            traj_token_ids_norm=[0, 3, 1],
            special_ids=_SPECIAL,
            traj_token_offset=_OFFSET,
        )
        assert out == [1, 5, 6, 7, 2, 3, 10, 13, 11, _END]

    def test_empty_traj_still_wellformed(self):
        out = assemble_completion_ids([5], [], _SPECIAL, _OFFSET)
        assert out == [1, 5, 2, 3, _END]

    def test_roundtrip_with_count(self):
        # The assembled completion's post-<|traj_future_start|> segment must
        # agree with count_generated_traj_tokens — this is the invariant
        # mine_pairs relies on when it rebuilds completions from stored rows.
        traj_norm = [2, 0, 1, 3]
        out = assemble_completion_ids([5, 6], traj_norm, _SPECIAL, _OFFSET)
        start = out.index(_SPECIAL["traj_future_start"])
        n, hit = count_generated_traj_tokens(out[start + 1 :], _OFFSET, _VOCAB, _END)
        assert (n, hit) == (len(traj_norm), True)
