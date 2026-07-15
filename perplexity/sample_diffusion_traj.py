# Fixed-reasoning counterpart to sample_discrete_traj.py: draw
# num_traj_samples diffusion trajectory samples off a FROZEN reasoning_text,
# instead of Route B's real production entrypoint
# (sample_trajectories_from_data_with_vlm_rollout), which always generates
# its OWN new reasoning per call and has no way to hold reasoning fixed
# across multiple diffusion draws -- its num_traj_sets parameter looks like
# it should mean "diffusion draws per reasoning trace" but isn't wired to a
# matching batch size anywhere in that function (checked: every call in this
# codebase, including the demo, only ever uses the default of 1). So there
# is no existing call that produces K diffusion draws sharing one reasoning
# trace -- this is genuinely new, not a rebuild of something that exists.
# Same problem, same solution shape as this workspace's
# pref_pairs/fixed_reasoning_rollout.py, which already solved this for the
# sibling alpamayo1_5 checkpoint.
#
# Mechanism, recomposed (not reimplemented) from alpamayo_r1.py's own public
# building blocks (self.expert/self.action_in_proj/self.action_out_proj/
# self.diffusion/self.action_space, all attributes that function already
# uses directly): a plain forward pass over
# prefix + reasoning_text + <|traj_future_start|> -- same reconstruction
# sample_discrete_action_tokens already does for the discrete side, so both
# sides build their conditioning context the same way -- gives the same KV
# cache / rope_deltas a generate() call would have produced for that exact
# token sequence (confirmed via the installed transformers source,
# qwen3_vl/modeling_qwen3_vl.py:899,1200-1207: rope_deltas is cached as a
# side effect of ANY forward pass through the base model, not something
# generate() computes specially). That cache is then fed to
# self.expert/self.diffusion exactly as
# sample_trajectories_from_data_with_vlm_rollout's own step_fn does.
#
# Simplification specific to the fixed-reasoning case: the original
# function's per-row offset/attention_mask logic exists to handle batches
# where different rows generated DIFFERENT reasoning lengths (so sequences
# get padded to a common length and the padding gap must be masked out).
# Here every one of the num_traj_samples diffusion draws shares one
# identical, single known-length reasoning (the KV cache is simply
# batch-repeated) -- there is no padding gap, offset is a constant equal to
# the prefill length, and an all-zero (fully-attending) additive mask is
# exactly what the original formula reduces to when offset == prefill_seq_len
# for every row.

import einops
import torch

from dump_input_template import build_prompt


def sample_diffusion_trajectories_given_fixed_reasoning(
    model,
    sample: dict,
    reasoning_text: str,
    num_traj_samples: int = 10,
    diffusion_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw num_traj_samples diffusion-expert trajectories conditioned on a FIXED reasoning_text.

    Args:
        model: AlpamayoR1, already .to("cuda") in bf16.
        sample: dict with image_frames, ego_history_xyz/rot (same schema
            build_prompt expects).
        reasoning_text: the reasoning to hold fixed -- pass the exact same
            string given to sample_discrete_action_tokens for a fair,
            confound-free comparison.
        num_traj_samples: number of independent diffusion draws off this one
            fixed reasoning (varies only the diffusion's own noise/seed, not
            the reasoning).
        diffusion_kwargs: forwarded to model.diffusion.sample, same as
            alpamayo_r1.py's own diffusion_kwargs param.

    Returns:
        pred_xyz: (num_traj_samples, 64, 3)
        pred_rot: (num_traj_samples, 64, 3, 3)
    """
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    prompt = build_prompt(model, sample)
    prefix_ids = prompt["fused_input_ids"][0].tolist()  # ends in <|cot_start|>
    reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False)["input_ids"]
    end_marker_id = model.config.traj_token_ids["future_start"]  # <|traj_future_start|>

    full_ids = prefix_ids + reasoning_ids + [end_marker_id]
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask_prefill = torch.ones_like(input_ids)
    aux = prompt["aux"]

    # One forward pass over the FIXED, already-known sequence -- no
    # generate(), since nothing here is being sampled at this step (same
    # "plain forward, not generate()" reasoning score.py documents, there
    # because every action token was already known too).
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_outputs = model.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask_prefill,
            pixel_values=aux["pixel_values"],
            image_grid_thw=aux["image_grid_thw"],
            use_cache=True,
        )
    rope_deltas = model.vlm.model.rope_deltas  # (1,), cached as a forward-pass side effect
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    assert prefill_seq_len == len(full_ids)

    # Repeat the ONE fixed-reasoning KV cache across the batch dim so
    # num_traj_samples diffusion draws can share it -- same cache, same
    # reasoning, different diffusion noise per row.
    prompt_cache.batch_repeat_interleave(num_traj_samples)

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]  # 64 future timesteps

    # offset is constant (== prefill_seq_len) for every row here, unlike the
    # real generate()-based path where different rows can generate different
    # reasoning lengths and need per-row offsets.
    offset = prefill_seq_len
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=num_traj_samples).clone()
    position_ids += rope_deltas.to(device) + offset

    # No padding gap to mask (every row shares the identical fixed
    # reasoning), so a fully-attending (all-zero) additive mask is correct
    # here, unlike the real path's per-row masked interval.
    attention_mask = torch.zeros(
        (num_traj_samples, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b_star = x.shape[0]
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        # crop the prompt cache back down -- otherwise each diffusion step
        # would keep appending the newly-added future tokens on top of the
        # last, same as alpamayo_r1.py's own step_fn does.
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(
            -1, *model.action_space.get_action_space_dims()
        )

    if diffusion_kwargs is None:
        diffusion_kwargs = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sampled_action = model.diffusion.sample(
            batch_size=num_traj_samples,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

    hist_xyz = sample["ego_history_xyz"][:, -1].to(device)  # (1, T_hist, 3)
    hist_rot = sample["ego_history_rot"][:, -1].to(device)
    hist_xyz_rep = einops.repeat(hist_xyz, "b ... -> (b k) ...", k=num_traj_samples)
    hist_rot_rep = einops.repeat(hist_rot, "b ... -> (b k) ...", k=num_traj_samples)

    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)
    return pred_xyz, pred_rot
