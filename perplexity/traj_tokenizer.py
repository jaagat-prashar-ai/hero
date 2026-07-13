# T1.2: tokenize / detokenize ground-truth trajectories.
#
# This module does NOT reimplement any of the math. Every step below (the
# Tikhonov-regularized unicycle fit, the uniform quantizer, the bin<->control
# mapping) already exists in NVIDIA's vendored code at
# alpamayo/src/alpamayo_r1/action_space/. We just:
#   1. build the exact DiscreteTrajectoryTokenizer the nvidia/Alpamayo-R1-10B
#      checkpoint was trained with, by reading its own config.json instead of
#      hardcoding constants, and
#   2. expose two thin, documented functions over it.
#
# Note on "ids": these are the tokenizer's native bin indices in
# [0, num_bins - 1] = [0, 2999], NOT LM vocabulary token ids. To get the vocab
# ids the model's logits actually live at, add config.traj_token_start_idx
# (151669 for this checkpoint) -- that step belongs to the perplexity code
# (T1.3+), not here.

import json

import hydra.utils as hyu
import torch
from huggingface_hub import hf_hub_download

from alpamayo_r1.action_space.discrete_action_space import DiscreteTrajectoryTokenizer

CHECKPOINT = "nvidia/Alpamayo-R1-10B"

_traj_tokenizer: DiscreteTrajectoryTokenizer | None = None


def _get_traj_tokenizer() -> DiscreteTrajectoryTokenizer:
    """Build (once) the checkpoint's real DiscreteTrajectoryTokenizer.

    Downloads only config.json (a few KB, no model weights) and instantiates
    the tokenizer from its own `traj_tokenizer_cfg` block -- the same recipe
    alpamayo_r1.models.base_model.ReasoningVLA uses internally
    (`hyu.instantiate(config.traj_tokenizer_cfg, load_weights=False)`) -- so
    the accel/curvature normalization stats, bin count, and value range are
    guaranteed to match the checkpoint instead of being copy-pasted by hand.
    """
    global _traj_tokenizer
    if _traj_tokenizer is None:
        config_path = hf_hub_download(repo_id=CHECKPOINT, filename="config.json")
        with open(config_path) as f:
            traj_tokenizer_cfg = json.load(f)["traj_tokenizer_cfg"]
        _traj_tokenizer = hyu.instantiate(traj_tokenizer_cfg, load_weights=False)
    return _traj_tokenizer


def tokenize_traj(
    hist_xyz: torch.Tensor,
    hist_rot: torch.Tensor,
    fut_xyz: torch.Tensor,
    fut_rot: torch.Tensor,
) -> torch.LongTensor:
    """Ground-truth future trajectory -> 128 discrete bin-index ids.

    History is required, not optional: the unicycle fit needs an initial
    velocity v0, which UnicycleAccelCurvatureActionSpace.estimate_t0_states
    estimates from `hist_xyz`/`hist_rot` (see traj_to_action in
    action_space/unicycle_accel_curvature.py). You cannot tokenize a future
    trajectory in isolation.

    Args:
        hist_xyz: (..., T_hist, 3) history positions, ego frame, last step = t0.
        hist_rot: (..., T_hist, 3, 3) history rotation matrices.
        fut_xyz: (..., 64, 3) future positions to encode.
        fut_rot: (..., 64, 3, 3) future rotation matrices to encode.

    Returns:
        ids: (..., 128) long tensor, values in [0, 2999]. Order is
            [accel_0, kappa_0, accel_1, kappa_1, ..., accel_63, kappa_63].
    """
    tokenizer = _get_traj_tokenizer()
    return tokenizer.encode(hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot)


def detokenize_traj(
    ids: torch.LongTensor,
    hist_xyz: torch.Tensor,
    hist_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """128 bin-index ids -> reconstructed future trajectory.

    This is the inverse path T2.1's validation gate needs: ids are dequantized
    to bin centers (`id / (num_bins - 1) * (dims_max - dims_min) + dims_min`),
    giving back (accel, kappa) controls, which are then rolled out through the
    same unicycle kinematic model (action_to_traj) anchored on `hist_xyz`/
    `hist_rot` -- the same history used when tokenizing, since the rollout
    also needs v0.

    Reconstruction is not exact: quantization to 3000 bins and the Tikhonov
    smoothing in the forward fit both lose information. Expect on the order of
    a few cm of xy error versus the original ground truth, not zero.

    Args:
        ids: (..., 128) long tensor of bin indices, as returned by
            tokenize_traj.
        hist_xyz: (..., T_hist, 3) same history used to tokenize.
        hist_rot: (..., T_hist, 3, 3) same history used to tokenize.

    Returns:
        fut_xyz: (..., 64, 3) reconstructed future positions.
        fut_rot: (..., 64, 3, 3) reconstructed future rotation matrices.
    """
    tokenizer = _get_traj_tokenizer()
    fut_xyz, fut_rot, _ = tokenizer.decode(hist_xyz=hist_xyz, hist_rot=hist_rot, tokens=ids)
    return fut_xyz, fut_rot
