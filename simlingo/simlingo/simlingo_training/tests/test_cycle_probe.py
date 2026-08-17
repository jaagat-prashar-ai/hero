"""Tests for the Stage-0 cycle probe pieces: delta-token span selection and
its interaction with the grouped ranking loss."""
import torch

from simlingo_training.models.utils import group_delta_spans, grouped_rank_cycle_loss


def _t(*vals):
    return torch.tensor(vals, dtype=torch.long)


def test_shared_prefix_suffix_trimmed():
    g = [_t(5, 6, 1, 2, 9, 9), _t(5, 6, 3, 9, 9), _t(5, 6, 4, 4, 4, 9, 9)]
    spans = group_delta_spans(g)
    assert spans == [(2, 4), (2, 3), (2, 5)]
    # the delta spans are exactly the discriminative clauses
    assert [g[i][s:e].tolist() for i, (s, e) in enumerate(spans)] == [[1, 2], [3], [4, 4, 4]]


def test_identical_candidates_fall_back_to_full_span():
    g = [_t(7, 8, 9), _t(7, 8, 9)]
    assert group_delta_spans(g) == [(0, 3), (0, 3)]


def test_strict_prefix_candidate_never_gets_empty_span():
    g = [_t(1, 2, 3), _t(1, 2, 3, 4)]
    spans = group_delta_spans(g)
    assert all(e > s for s, e in spans)
    assert spans[1] == (3, 4)  # the extra token IS the delta


def test_no_shared_structure_keeps_full_spans():
    g = [_t(1, 2), _t(3, 4)]
    assert group_delta_spans(g) == [(0, 2), (0, 2)]


def test_singleton_group():
    assert group_delta_spans([_t(1, 2, 3)]) == [(0, 3)]
    assert group_delta_spans([]) == []


def test_realistic_template_group():
    base, tail = [101, 102, 103], [201, 202]
    g = [
        _t(*base, 11, 12, *tail),
        _t(*base, 13, *tail),
        _t(*base, 14, 15, 16, *tail),
        _t(*base, 17, *tail),
    ]
    assert group_delta_spans(g) == [(3, 5), (3, 4), (3, 6), (3, 4)]


def test_delta_scoring_beats_full_span_dilution():
    """Identical shared-token CE plus a small on-match delta-token bonus:
    mean over the full span dilutes the signal, mean over the delta span
    preserves it. Both must still rank correctly in the noiseless case."""
    K, shared_len, delta_len, match_bonus = 4, 40, 3, 0.3
    rows, cols, ce_full, ce_delta = [], [], [], []
    for i in range(K):
        for j in range(K):
            rows.append(i)
            cols.append(j)
            delta_ce = 2.0 - (match_bonus if i == j else 0.0)
            ce_full.append((2.0 * shared_len + delta_ce * delta_len) / (shared_len + delta_len))
            ce_delta.append(delta_ce)
    args = (torch.tensor(rows), torch.tensor(cols), K)
    loss_f, count_f, acc_f = grouped_rank_cycle_loss(torch.tensor(ce_full), *args)
    loss_d, count_d, acc_d = grouped_rank_cycle_loss(torch.tensor(ce_delta), *args)
    assert acc_f == 1.0 and acc_d == 1.0
    assert count_f.sum() == K and count_d.sum() == K
    # the delta-span margin (softmax confidence in the correct candidate) is
    # strictly larger, i.e. the same signal produces a stronger gradient
    p_full = torch.softmax(-torch.tensor(ce_full[:K]), 0)[0]
    p_delta = torch.softmax(-torch.tensor(ce_delta[:K]), 0)[0]
    assert p_delta > p_full
    assert loss_d.sum() < loss_f.sum()
