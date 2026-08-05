"""
Unit tests for the intra-scene counterfactual contrastive alignment loss.

Run standalone (only needs torch, not the full training stack):
    python simlingo_training/tests/test_contrastive_alignment.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlingo_training.models.utils import intra_scene_contrastive_loss


def make_embeddings(n, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=g), dim=-1)


def test_perfect_alignment_low_loss_full_accuracy():
    # z_text == z_traj with distinct directions: diagonal dominates every group
    z = make_embeddings(6)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    loss, count, acc = intra_scene_contrastive_loss(z, z, group_ids, temperature=0.07)
    assert loss.shape == (6,) and count.tolist() == [1] * 6
    assert acc is not None and acc.item() == 1.0
    assert loss.mean().item() < 0.1, f"aligned loss should be near 0, got {loss.mean().item()}"


def test_shuffled_pairing_high_loss():
    z = make_embeddings(4)
    group_ids = torch.tensor([0, 0, 0, 0])
    loss_aligned, _, _ = intra_scene_contrastive_loss(z, z, group_ids)
    loss_shuffled, _, acc = intra_scene_contrastive_loss(z, z[[1, 0, 3, 2]], group_ids)
    assert loss_shuffled.mean() > loss_aligned.mean() + 1.0
    assert acc.item() < 1.0


def test_singleton_groups_are_skipped():
    z = make_embeddings(3)
    group_ids = torch.tensor([0, 1, 2])  # e.g. driving samples: no counterfactual siblings
    loss, count, acc = intra_scene_contrastive_loss(z, z, group_ids)
    assert loss.abs().sum().item() == 0.0
    assert count.tolist() == [0, 0, 0]
    assert acc is None


def test_mixed_batch_only_groups_contribute():
    z = make_embeddings(5)
    group_ids = torch.tensor([0, 1, 1, 1, 2])  # one dreamer group of 3 among singletons
    loss, count, acc = intra_scene_contrastive_loss(z, z, group_ids)
    assert count.tolist() == [0, 1, 1, 1, 0]
    assert loss[0].item() == 0.0 and loss[4].item() == 0.0
    assert acc is not None


def test_gradients_flow_to_both_sides():
    z_text = make_embeddings(4, seed=1).requires_grad_(True)
    z_traj = make_embeddings(4, seed=2).requires_grad_(True)
    group_ids = torch.tensor([0, 0, 0, 0])
    loss, count, _ = intra_scene_contrastive_loss(z_text, z_traj, group_ids)
    (loss.sum() / count.sum()).backward()
    assert z_text.grad is not None and z_text.grad.abs().sum() > 0
    assert z_traj.grad is not None and z_traj.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
