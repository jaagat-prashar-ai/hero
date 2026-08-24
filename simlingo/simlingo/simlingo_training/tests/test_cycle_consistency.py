"""
Unit tests for the grouped inverse-cycle ranking loss.

Run standalone (only needs torch, not the full training stack):
    python simlingo_training/tests/test_cycle_consistency.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlingo_training.models.utils import grouped_rank_cycle_loss


def make_pairs(group_sizes):
    pair_row, pair_col = [], []
    start = 0
    for k in group_sizes:
        idx = list(range(start, start + k))
        for i in idx:
            pair_row.extend([i] * k)
            pair_col.extend(idx)
        start += k
    return torch.tensor(pair_row), torch.tensor(pair_col)


def test_true_pairing_lowest_ce_full_accuracy():
    pair_row, pair_col = make_pairs([3, 2])
    # true instruction (row == col) gets CE 0.1, siblings get 3.0
    ce = torch.where(pair_row == pair_col, torch.tensor(0.1), torch.tensor(3.0))
    loss, count, acc = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=5)
    assert count.tolist() == [1] * 5
    assert acc is not None and acc.item() == 1.0
    assert loss.mean().item() < 0.2


def test_inverted_pairing_high_loss_zero_accuracy():
    pair_row, pair_col = make_pairs([3])
    ce = torch.where(pair_row == pair_col, torch.tensor(3.0), torch.tensor(0.1))
    loss_bad, _, acc = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=3)
    ce_good = torch.where(pair_row == pair_col, torch.tensor(0.1), torch.tensor(3.0))
    loss_good, _, _ = grouped_rank_cycle_loss(ce_good, pair_row, pair_col, batch_size=3)
    assert acc.item() == 0.0
    assert loss_bad.mean() > loss_good.mean() + 1.0


def test_singletons_and_missing_pairs_skipped():
    # sample 2 has no siblings: only its self-pair exists
    pair_row = torch.tensor([0, 0, 1, 1, 2])
    pair_col = torch.tensor([0, 1, 0, 1, 2])
    ce = torch.tensor([0.1, 3.0, 3.0, 0.1, 0.5])
    loss, count, acc = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=3)
    assert count.tolist() == [1, 1, 0]
    assert loss[2].item() == 0.0
    assert acc.item() == 1.0


def test_temperature_scales_confidence():
    pair_row, pair_col = make_pairs([2])
    ce = torch.tensor([0.5, 1.0, 1.0, 0.5])
    loss_sharp, _, _ = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=2, temperature=0.1)
    loss_soft, _, _ = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=2, temperature=10.0)
    # same (correct) ranking, but low temperature should be far more confident
    assert loss_sharp.mean() < loss_soft.mean()


def test_per_instruction_prior_does_not_decide_the_ranking():
    """
    Regression for the 2026-08-16 smoke failure: mean-token CE is dominated by
    how cheap each instruction is to say, an offset identical down its column.
    A raw row softmax ranks by that prior alone -> the same winner for every
    trajectory, exactly 1/K correct, loss above ln(K). The objective must key
    on the trajectory-conditional part instead.
    """
    k = 4
    pair_row, pair_col = make_pairs([k])
    prior = torch.tensor([2.0, 3.5, 5.0, 6.5])  # per-instruction, same for every trajectory
    signal = torch.full((k, k), 0.30)
    signal.fill_diagonal_(0.0)  # true pair is only 0.3 nats better
    ce = (prior.unsqueeze(0) + signal).flatten()

    loss, count, acc = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=k)
    assert count.tolist() == [1] * k
    assert acc.item() == 1.0, f"prior swamped the match signal, acc={acc.item()}"
    assert loss.mean().item() < torch.log(torch.tensor(float(k))).item()

    # the prior must be what is ignored: shifting it must not move the ranking
    shifted = (prior.unsqueeze(0) * 7.0 - 4.0 + signal).flatten()
    _, _, acc_shifted = grouped_rank_cycle_loss(shifted, pair_row, pair_col, batch_size=k)
    assert acc_shifted.item() == 1.0


def test_symmetric_both_directions_contribute():
    """Trajectory->instruction and instruction->trajectory are both scored."""
    k = 3
    pair_row, pair_col = make_pairs([k])
    ce = torch.full((k, k), 2.0)
    ce.fill_diagonal_(0.5)
    loss_sym, _, acc = grouped_rank_cycle_loss(ce.flatten(), pair_row, pair_col, batch_size=k)
    assert acc.item() == 1.0
    # break only the column direction: instruction 0 explains trajectory 1 better
    # than its own trajectory 0, while row 0 still prefers instruction 0.
    ce_col_broken = ce.clone()
    ce_col_broken[1, 0] = 0.1
    loss_broken, _, _ = grouped_rank_cycle_loss(ce_col_broken.flatten(), pair_row, pair_col, batch_size=k)
    assert loss_broken.sum() > loss_sym.sum(), "column direction is not penalised"


def test_summarise_losses_zero_count_backward_is_finite():
    # regression: torch.where backprops through BOTH branches, so an unclamped
    # v.sum()/0 emitted NaN grads whenever a rank had a zero-count aux loss
    # tied into the graph (the 08-21 fleet poisoning; anomaly: DivBackward0
    # at summarise_losses). The zero-count loss must yield exactly-zero grads.
    from simlingo_training.models.utils import summarise_losses

    param = torch.randn(4, requires_grad=True)
    live_loss = (param * 2.0).abs()                      # count > 0
    tied_zero = (param * 0.0)                            # graph-tied, count == 0
    out = summarise_losses({
        "task_loss": (live_loss, torch.ones(4, dtype=torch.long)),
        "cycle_loss": (tied_zero, torch.zeros(4, dtype=torch.long)),
    })
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert torch.isfinite(param.grad).all(), "NaN leaked through the zero-count branch"


def test_gradient_flows_to_ce():
    pair_row, pair_col = make_pairs([2])
    ce = torch.tensor([0.5, 1.0, 1.0, 0.5], requires_grad=True)
    loss, _, _ = grouped_rank_cycle_loss(ce, pair_row, pair_col, batch_size=2)
    loss.sum().backward()
    assert ce.grad is not None and ce.grad.abs().sum() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
