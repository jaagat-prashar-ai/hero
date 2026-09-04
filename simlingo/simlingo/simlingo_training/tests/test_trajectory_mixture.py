"""Unit tests for the K-mode trajectory mixture: NLL, variational mixture KL,
and the DrivingAdaptor mixture head."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlingo_training.models.utils import (  # noqa: E402
    mixture_kl_variational,
    mixture_trajectory_nll,
)


def _single_mode(mean, sigma):
    """Wrap a [B,N,D] point trajectory as a K=1 mixture with constant sigma."""
    means = mean.unsqueeze(1)
    log_sigmas = torch.full_like(means, math.log(sigma))
    logits = torch.zeros(mean.size(0), 1)
    return means, log_sigmas, logits


def test_single_mode_nll_reduces_to_scaled_l2():
    """With K=1 and fixed sigma, NLL must equal 0.5*||err/sigma||^2 + const."""
    torch.manual_seed(0)
    target = torch.randn(4, 30, 2)
    pred = target + 0.3
    sigma = 0.5
    nll = mixture_trajectory_nll(*_single_mode(pred, sigma), target)
    n_terms = 30 * 2
    expected = 0.5 * ((pred - target) / sigma).square().flatten(1).sum(-1) + n_terms * (
        math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    )
    assert torch.allclose(nll, expected, atol=1e-4)


def test_nll_prefers_nearest_mode_not_the_average():
    """The MDN property the fixed-sigma KL lacked: with GT at one of two valid
    plans, a two-mode mixture on the plans beats a point estimate at their mean."""
    target = torch.zeros(1, 10, 2)
    target[..., 1] = 1.0  # GT: the "left" plan
    left = target.clone()
    right = target.clone()
    right[..., 1] = -1.0
    means = torch.stack([left, right], dim=1)
    log_sigmas = torch.full_like(means, math.log(0.5))
    logits = torch.zeros(1, 2)
    nll_mixture = mixture_trajectory_nll(means, log_sigmas, logits, target)
    average = (left + right) / 2.0
    nll_average = mixture_trajectory_nll(*_single_mode(average, 0.5), target)
    assert nll_mixture.item() < nll_average.item()


def test_mixture_kl_is_zero_for_identical_mixtures():
    torch.manual_seed(1)
    means = torch.randn(3, 4, 10, 2)
    log_sigmas = torch.randn(3, 4, 10, 2) * 0.1
    logits = torch.randn(3, 4)
    kl = mixture_kl_variational(means, log_sigmas, logits, means, log_sigmas, logits)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_mixture_kl_zero_forcing_direction():
    """Leaving the text support costs far more than sharpening inside it.

    Note the sharpening itself is NOT free: the log sigma-ratio accrues per
    dimension, a mild constant pressure against over-confidence. What must
    hold is that mean displacement out of the support dominates it.
    """
    base = torch.zeros(1, 1, 10, 2)
    sharp = torch.full_like(base, math.log(0.2))
    broad = torch.full_like(base, math.log(2.0))
    logits = torch.zeros(1, 1)
    inside = mixture_kl_variational(base, sharp, logits, base, broad, logits)
    matched = mixture_kl_variational(base, broad, logits, base, broad, logits)
    far = mixture_kl_variational(base + 6.0, sharp, logits, base, broad, logits)
    assert matched.item() < 1e-5
    assert far.item() > 2.0 * inside.item() > 0.0


def test_mixture_kl_matches_any_licensed_mode():
    """Camera plan agreeing with EITHER text mode stays cheap -- the failure of
    the point-estimate KL (pull toward the average) must not reappear."""
    left = torch.zeros(1, 1, 10, 2)
    left[..., 1] = 1.0
    right = -left
    text_means = torch.cat([left, right], dim=1)
    text_log_sigmas = torch.full_like(text_means, math.log(0.5))
    text_logits = torch.zeros(1, 2)
    cam_sig = torch.full_like(left, math.log(0.3))
    cam_logits = torch.zeros(1, 1)
    kl_on_right = mixture_kl_variational(
        right, cam_sig, cam_logits, text_means, text_log_sigmas, text_logits
    )
    kl_on_average = mixture_kl_variational(
        (left + right) / 2.0, cam_sig, cam_logits, text_means, text_log_sigmas, text_logits
    )
    assert kl_on_right.item() < kl_on_average.item()


def test_mixture_kl_detach_q_blocks_text_gradient():
    means_p = torch.zeros(2, 2, 5, 2, requires_grad=True)
    means_q = torch.ones(2, 2, 5, 2, requires_grad=True)
    log_sig = torch.zeros(2, 2, 5, 2)
    logits = torch.zeros(2, 2)
    mixture_kl_variational(
        means_p, log_sig, logits, means_q, log_sig.clone(), logits.clone(), detach_q=True
    ).sum().backward()
    assert means_p.grad is not None and torch.isfinite(means_p.grad).all()
    assert means_q.grad is None


def test_driving_adaptor_mixture_shapes_and_argmax_output():
    from simlingo_training.models.adaptors.adaptors import DrivingAdaptor

    torch.manual_seed(2)
    adaptor = DrivingAdaptor(
        hidden_size=32, mlp_dim=16, predict_route_as_wps=True,
        speed_wps_mode='2d', mixture_k=3,
    )
    n_total = adaptor.sizes['route'] + adaptor.sizes['speed_wps']
    features = torch.randn(2, n_total, 32)
    means, log_sigmas, logits = adaptor.mixture_params(features)
    assert means.shape == (2, 3, n_total, 2)
    assert log_sigmas.shape == (2, 3, n_total, 2)
    assert logits.shape == (2, 3)
    # zero-init sigma head: every sigma starts exactly at sigma_init
    assert torch.allclose(log_sigmas.exp(), torch.full_like(log_sigmas, 0.5), atol=1e-5)
    # zero-init logit head: uniform modes at start
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    preds = adaptor.get_predictions(features)
    assert preds['route'].shape == (2, adaptor.sizes['route'], 2)
    assert preds['speed_wps'].shape == (2, adaptor.sizes['speed_wps'], 2)


def test_driving_adaptor_mixture_requires_2d():
    from simlingo_training.models.adaptors.adaptors import DrivingAdaptor

    try:
        DrivingAdaptor(hidden_size=32, speed_wps_mode='1d', mixture_k=3)
    except ValueError:
        return
    raise AssertionError("mixture_k with speed_wps_mode='1d' must raise")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
