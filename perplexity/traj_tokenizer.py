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

CHECKPOINT = "nvidia/Alpamayo-R1-10B" # SFT checkpoint, cross-entropy loss.

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

# thin wrapper over alpamayo_r1's own DiscreteTrajectoryTokenizer. The tokenizer is built by downloading the checkpoint's config.json and reading its traj_tokenizer_cfg block directly (same pattern the model itself uses internally).
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

# Note: 
# ids are bin indices (0-2999), not vocab token ids. Adding 151669 (the checkpoint's traj_token_start_idx) to get lM vocabulary ids is separate, later step - deliberately not done here, since T1.2 is purelyy about th ediscrete action space, not the language model's vocabulary. 
# Tokenizing requires history, not just future waypoints. The unicycle fit needs an initial velocity v0, estimated from hist_xyz/hist_rot. There's no way to tokenize a future trajectory in isolation - that's inherent to the underlying NVIDIA code, not a wrapper limitation. 
# The round trip is lossy by design - ~2cm mean / 3.6cm max xy error on this sample, from 3000-bin quanitzation and the Tikhonov smoothing in the original fit. That's expected. 

# The "original" = the forward direction, and it's not incidental, it's the actual ground-truth-target pipeline. 
# Perplexity needs, at each of the model's 128 discrete-action positions, the answer to "what probability did the model assign to the actual ground-truth token?"
# To ask that question, you need the ground-truth token - i.e., you need to run a real waypoint trajectory through exactly the same fit-and-quantize pipeline NVIDIA used when building training labels (Tikhonov unicycle fit -> accel/kappa -> uniform quantize -> 128 bin ids). 
# tokenize_traj is that pipeline. It's not a redundant convenience - it's the thing that produces the target ids you'll index into the model's logits with, at the very last step of the whole experiment. 
# Nothing about the perplexity computation works without it. 

## July 14th 

# Idea: transformer produces one prediction per position, simultaneously, and each position's prediction is only allowed to see positions before it (
# (enforced by the causal mask - verified in modeling_qwen3_vl.py), where is_causal=True for text vs False for image patches).

# Position p's output is always "what comes at p+1" computed from 0..p only. Since we're scoring a real logged trajectory, the true token at p+1 is already sitting in input_ids. 
# We just read off the probability the model assigned to the token that's actually there. One forward pass gives you every position's "guess" for free. 

# Out of 3, 154 total positions, score.py only reads the 128 that predict action tokens. Reasoning tokens are pure conditioning, never scored themselves. 
# Question is "does the reasoning predict the action" not "does the reasoning sound plausible"
# Scoring reasoning tokens may just measure sentence fluency 

# outputs = model.vlm(input_ids=input_ids, attention_mask=attention_mask, pixel_values=aux["pixel_values"], image_grid_thw=aux["image_grid_thw"])

# logits = outputs.logits # (1, 3154, 155697) - one 155, 697-way score per position. // what is 3154?

# pred_logits = logits[0, action_start-1; action_start-1+128, :]
# log_probs = torch.log_softmax(pred_logits, dim=-1) # raw scores, real probabilities

# targets = input_ids[0, action_start : action_start+128] # the ground truth we inserted
# nlls = -log_probs.gather(-1, targets.unsqueeze(-1)) # pick out probe(True token)

# Same clip,s ame forward pass. 

# traj_tokenizer.py since 

# dump_input_template.build_prompt() calls the model's own helper.create_message() + processor.apply_chat_template() 
    # standard Qwen3-VL processing: 

    # Images (data["image_frames"]) get turned into pixel_values + image_grid_thw, and the processor expands placeholder 
    # image_token_id tokens into input_ids at the vision span (this is the non-causal part: vision encoder attends every patch to every other patch).

    # Ego-history is not real numbers yet at this stage: it's 48 copies of a single placeholder token 
    # model.fuse_traj_tokens(...), called right after, replaces those 48 placeholders with real quantized history bin tokens from ego_history_xyz/rot. 
    # "scene" in the final sequence = vision tokens (from images, via the processor) + fused ego-history bins (from real telemetry) + surrounding instruction text, ending at <|cot_start|>. 
    # None of this is hand-built - it's the literal prefix test_inference.py uses for real generation, just captured instead of thrown away. 


    # [prefix_ids] is the (scene + ego_history + instructions, ends <|cot_start|>) + 
    # [reasoning_ids] is the (your CoT text, real/none/shuffled - the experiment variable) + 
    # [<traj_future_start|>] + 
    # [128 GT action tokens] 
        # (accel_0, kappa_0, accel_1, kappa_1, ..., accel_63, kappa_63)

        # traj_toklenizer.tokenizer_traj() produces those last 128 as bin indices in [0, 2999] - one accel and one kappa per bin timestep. 
        # 64 timesteps, 128 tokens. 
        # score.py shifts each bin index into vocab space (traj_token_start_idx + bin_id) before appending. 

        # at every position, we feed the modle the real/true token as context for predicting the next one (never the model's own guess), even if the guess would ahve been wrong. 



# Generate actions
# either through discrte path 
# continuous actions, through diffusion path
# do these two generated actions look the same?

# measure alignment between discrete and the continuous action path 
# backbone for translating analysis 

# GT tokens 
# get the discrete trajecotry during inference

# if you want to restrict comparison to gt tokens
# for these gt tokens, there should be an equivalent ocntinous value that it would produce conditioned on reasoning and kv cache

# does the continous path the model produces look similar to the GT path?

# during test time
# using images, etc. check if the discrete and continous are the same 

# Poutine, AutoVLA 
# give more concrete updates on this soon. 

# but as a precursor, we must ensure the discrete and continuous are the same 
# we are still conditioning on the perturbed reaosning

# 1. run the score comparison to compare the discrete and continuous trjaectories from the validation dataset
    # during inference, check the comparison 
    # Have an update on this by the end of the day, should be straightforward for comparing trajectories

# 2. do perturbed reasoning and look at the influence  
    # understand the impact on the actions 
    # what are other ways of measuring alignment?
    # this is more so trying to understand if the data is good, this is the GT data. 
    # But we want to focus on the actual predicted data, think about the problem more carefully


# 3. updates for poutine and autovla
    # run experiments for this as well alongside. 
    # for poutine, text space, so should be fairly straightforward. 

# Timestamps fix: 
    # reuses build_webdataset.py's existing HF/S3 retry helpers and clip list/split logic (so shard indexing lines up with what's already on S3), but per clip it only opens clip_files_in_zip["frame_timestamps"] and skips "video" entirely.
    # No MP4 download, no transcode. 

    # Output goes to a new path: 
    # s3_clip_loader.py gets updated to pull that file alongside the existing extract, pass real timestamps= into SeekVideoReader, and swap decode_last_n_frames -> decode_images_from_timestamps 

    # function decode_last_n_frames was a guess, invented specifically because we didn't have timestamps, and it needs to go away entirely once we do 
    # What "last N frames" was actually standing in for 
    
    # load_physical_aiavdataset.py: 164-168
    # image_timestamps = np.array(
    # [t0_us - (num_frames - 1 -i) * int[time_step * 1_000_000) for i in range(num_frames)] ],
    # dtype=np.int64,
    # )


    # decode_last_n_frames picks by index because index was the only thing we could use without timestamps. it was never trying to approximate "last N frames" as a goal in itself, it was a stand-in for "frames at t0" under the constraint of missing metadata. 
    # Once frame_timestamps exists, there's no more reason for the index-based guess: you can ask for the actual frames 
    # decode_last_n_frames was never trying to approximate "last N frames" as a goal in itself, it was a stand-in for "frames at t0" under the constraint of missing metadata. 
    # Once frame_timestamps exists, there's no more reason for the index-based guess: you can ask for the actual frames at t0-0.3s...t0, which is what decode_images_from_timestamps does and what decode_last_n_frames was always trying to approximate. 



    

