"""
Unit tests for the Stage-0 probe rigor controls (apply_traj_controls).

Run standalone (only needs torch, not the full training stack):
    python simlingo_training/tests/test_traj_controls.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlingo_training.models.utils import apply_traj_controls


def make_traj(b=6, n=11, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def test_defaults_are_identity():
    traj = make_traj()
    out = apply_traj_controls(traj)
    assert out is traj  # no copy, no change


def test_noise_perturbs_at_requested_scale():
    traj = make_traj(b=64, n=20)
    torch.manual_seed(1234)
    out = apply_traj_controls(traj, noise_m=0.3)
    delta = out - traj
    assert not torch.equal(out, traj)
    # empirical std of the added noise ~= 0.3 (loose 10% band, 2560 points)
    assert abs(delta.std().item() - 0.3) < 0.03
    assert out.shape == traj.shape


def test_shuffle_permutes_batch_rows_exactly():
    traj = make_traj(b=8)
    torch.manual_seed(7)
    out = apply_traj_controls(traj, shuffle=True)
    assert out.shape == traj.shape
    # every output row is some input row, each used exactly once
    matches = [
        [torch.equal(out[i], traj[j]) for j in range(traj.size(0))]
        for i in range(traj.size(0))
    ]
    assert all(sum(row) == 1 for row in matches)
    assert all(sum(col) == 1 for col in zip(*matches))


def test_shuffle_redraws_every_call():
    # the control must re-randomize per step; a fixed permutation would be
    # learnable and defeat the point. Over 20 draws on b=32, at least one
    # pair of draws must differ (P[all equal] ~ (1/32!)^19).
    traj = make_traj(b=32)
    torch.manual_seed(42)
    perms = set()
    for _ in range(20):
        out = apply_traj_controls(traj, shuffle=True)
        order = tuple(
            next(j for j in range(32) if torch.equal(out[i], traj[j]))
            for i in range(32)
        )
        perms.add(order)
    assert len(perms) > 1


def test_noise_then_shuffle_compose():
    traj = make_traj(b=8)
    torch.manual_seed(3)
    out = apply_traj_controls(traj, noise_m=0.3, shuffle=True)
    assert out.shape == traj.shape
    # after noise, no output row should exactly equal any input row
    assert not any(
        torch.equal(out[i], traj[j]) for i in range(8) for j in range(8)
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
