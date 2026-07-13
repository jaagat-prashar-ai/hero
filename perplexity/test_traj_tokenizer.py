# T1.2 "Done when" check: tokenize_traj/detokenize_traj both run on one real
# dataset sample without error, and the round trip is sane (small but nonzero
# error -- exact reconstruction is NOT expected, see detokenize_traj docstring).

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

from traj_tokenizer import detokenize_traj, tokenize_traj

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # same clip used in alpamayo's test_inference.py


def test_tokenize_detokenize_roundtrip() -> None:
    data = load_physical_aiavdataset(CLIP_ID, t0_us=5_100_000)
    hist_xyz, hist_rot = data["ego_history_xyz"], data["ego_history_rot"]
    fut_xyz, fut_rot = data["ego_future_xyz"], data["ego_future_rot"]

    ids = tokenize_traj(hist_xyz, hist_rot, fut_xyz, fut_rot)
    assert ids.shape == (1, 128)
    assert ids.dtype.is_floating_point is False
    assert ids.min() >= 0 and ids.max() <= 2999  # num_bins - 1

    recon_xyz, recon_rot = detokenize_traj(ids, hist_xyz, hist_rot)
    assert recon_xyz.shape == fut_xyz.shape
    assert recon_rot.shape == fut_rot.shape

    xy_err = (fut_xyz[..., :2] - recon_xyz[..., :2]).norm(dim=-1)
    mean_err, max_err = xy_err.mean().item(), xy_err.max().item()
    print(f"roundtrip xy error (m): mean={mean_err:.4f} max={max_err:.4f}")

    # Sanity bounds, not exactness: quantization (3000 bins over [-10,10] std
    # units) and Tikhonov smoothing both lose information by design.
    assert mean_err < 0.5, f"mean roundtrip error {mean_err}m looks too large -- something is wrong"
    assert max_err < 2.0, f"max roundtrip error {max_err}m looks too large -- something is wrong"


if __name__ == "__main__":
    test_tokenize_detokenize_roundtrip()
    print("T1.2 OK: tokenize_traj/detokenize_traj ran on one dataset sample without error.")
