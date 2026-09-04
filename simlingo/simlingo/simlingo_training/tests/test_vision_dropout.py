"""Unit tests for dual-pass vision-dropout scheduling and Gaussian KL."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlingo_training.models.utils import (  # noqa: E402
    diagonal_gaussian_kl,
    periodic_auxiliary_active,
)


def test_periodic_quarter_schedule_is_ddp_deterministic():
    active = [periodic_auxiliary_active(step, 0.25) for step in range(12)]
    assert active == [True, False, False, False] * 3


def test_periodic_schedule_endpoints():
    assert not any(periodic_auxiliary_active(step, 0.0) for step in range(5))
    assert all(periodic_auxiliary_active(step, 1.0) for step in range(5))


def test_equal_gaussians_have_zero_kl():
    mean = torch.randn(4, 10, 2)
    kl = diagonal_gaussian_kl(mean, mean, sigma_p=1.0, sigma_q=1.0)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)


def test_kl_penalizes_mean_mismatch():
    p = torch.zeros(3, 10, 2, requires_grad=True)
    q = torch.ones_like(p)
    near = diagonal_gaussian_kl(p, p.detach(), sigma_p=0.5, sigma_q=2.0)
    far = diagonal_gaussian_kl(p, q, sigma_p=0.5, sigma_q=2.0)
    assert torch.all(far > near)
    far.mean().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_detached_text_prior_receives_no_kl_gradient():
    p = torch.zeros(2, 4, 2, requires_grad=True)
    q = torch.ones(2, 4, 2, requires_grad=True)
    diagonal_gaussian_kl(
        p, q, sigma_p=0.5, sigma_q=2.0, detach_q=True
    ).sum().backward()
    assert p.grad is not None
    assert q.grad is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

