"""Tests for TrajectoryEncoder, the sequence encoder that turns cycle-pass
trajectory coordinates into LLM token embeddings."""
import pytest
import torch

from simlingo_training.models.adaptors.adaptors import TrajectoryEncoder


def _enc(token_size=32, n_segments=2, d_model=16, n_layers=1, n_heads=2):
    torch.manual_seed(0)
    return TrajectoryEncoder(token_size, n_segments, d_model=d_model, n_layers=n_layers, n_heads=n_heads)


def _traj(b=3, n=30):
    torch.manual_seed(1)
    return torch.randn(b, n, 2) * 5.0


def test_output_shape_and_dtype():
    enc = _enc()
    out = enc(_traj(), [10, 20])
    assert out.shape == (3, 30, 32)
    assert out.dtype == torch.float32


def test_every_parameter_receives_gradient_with_all_segments():
    enc = _enc()
    enc(_traj(), [10, 20]).sum().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert missing == []


def test_every_parameter_receives_gradient_with_one_segment_present():
    # a missing segment (e.g. no route) must not leave its embedding row out of
    # the graph, or DDP's allreduce param set diverges across ranks
    enc = _enc(n_segments=2)
    enc(_traj(n=10), [10]).sum().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert missing == []
    assert enc.seg_emb.weight.grad is not None


def test_gradient_reaches_input_coordinates():
    enc = _enc()
    x = _traj().requires_grad_(True)
    enc(x, [10, 20]).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_not_permutation_equivariant():
    # a bag-of-points model satisfies enc(x[perm]) == enc(x)[perm]; the
    # position and delta features must break that
    enc = _enc(n_segments=1).eval()
    x = _traj(n=10)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(3))
    assert not torch.allclose(enc(x[:, perm], [10]), enc(x, [10])[:, perm], atol=1e-5)


def test_segment_embedding_is_used():
    enc = _enc(n_segments=2).eval()
    x = _traj(n=20)
    before = enc(x, [10, 10])
    with torch.no_grad():
        enc.seg_emb.weight.copy_(enc.seg_emb.weight.flip(0))
    assert not torch.allclose(before, enc(x, [10, 10]), atol=1e-5)


def test_tokens_carry_context_from_other_points():
    enc = _enc().eval()
    x = _traj()
    y = x.clone()
    y[:, 25] += 3.0
    a, b = enc(x, [10, 20]), enc(y, [10, 20])
    # a per-point MLP would leave every other token untouched; attention must not
    assert not torch.allclose(a[:, :25], b[:, :25], atol=1e-5)


def test_input_features_are_xy_plus_delta_from_previous_point():
    enc = _enc(n_segments=1)
    captured = {}

    def grab_input(module, inputs, output):  # returns None so the hook does not replace the output
        captured['x'] = inputs[0].detach()

    enc.in_proj.register_forward_hook(grab_input)
    x = _traj(n=10)
    enc(x, [10])
    feats = captured['x']
    assert feats.shape == (3, 10, 4)
    assert torch.allclose(feats[..., :2], x)
    assert torch.allclose(feats[:, 0, 2:], x[:, 0])  # first delta: offset from the ego origin
    assert torch.allclose(feats[:, 1:, 2:], x[:, 1:] - x[:, :-1])


def test_bad_seg_lens_raise():
    enc = _enc()
    with pytest.raises(ValueError):
        enc(_traj(), [10, 10])


def test_odd_d_model_rejected():
    with pytest.raises(ValueError):
        TrajectoryEncoder(32, 2, d_model=15, n_heads=1)


def test_half_precision_cast_path():
    enc = _enc().to(torch.bfloat16)
    out = enc(_traj().to(enc.out_proj.weight.dtype), [10, 20])
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_segment_longer_than_max_len_raises():
    enc = TrajectoryEncoder(32, 1, d_model=16, n_heads=2, max_len=8)
    with pytest.raises(ValueError):
        enc(_traj(n=9), [9])
