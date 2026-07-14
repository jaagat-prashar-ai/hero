# Step 1 of the discrete-vs-continuous equivalence question: actually SAMPLE
# discrete action tokens from the model's own discrete action head, rather
# than only ever scoring ground-truth tokens (score.py's Route A, a single
# teacher-forced forward pass) or sampling continuous trajectories via the
# diffusion expert (Route B, alpamayo_r1.py's
# sample_trajectories_from_data_with_vlm_rollout).
#
# Why this didn't already exist: alpamayo_r1.py's ExpertLogitsProcessor masks
# OUT the action-token vocab range (sets it to -inf) so that generate() can
# only ever produce reasoning/CoT text -- by design, since Route B never
# needs the model to emit its own action tokens. We need the mirror image:
# ban everything EXCEPT the action-token vocab range, and force exactly
# tokens_per_future_traj (128) new tokens via min_new_tokens=max_new_tokens,
# since there's no natural stop token inside the action vocab itself to key
# a stopping criterion off of (unlike ExpertLogitsProcessor's generation,
# which stops at <|traj_future_start|>).
#
# Reasoning is passed in as a literal string (reasoning_text), same contract
# as score.py -- the caller is expected to hold it FIXED (e.g. one real
# generated CoT sample, reused for both this discrete path and a diffusion
# draw) so that any downstream trajectory disagreement reflects the two
# heads disagreeing, not two independently-sampled reasoning traces.

import copy

import torch
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from dump_input_template import build_prompt


class ActionOnlyLogitsProcessor(LogitsProcessor):
    """Mirror image of alpamayo_r1.ExpertLogitsProcessor.

    That processor bans the action-token vocab range so generate() only ever
    emits reasoning text. This processor bans everything OUTSIDE the
    action-token vocab range, so generate() can only ever emit action tokens.
    """

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        traj_end = self.traj_token_offset + self.traj_vocab_size
        scores[:, : self.traj_token_offset] = float("-inf")
        scores[:, traj_end:] = float("-inf")
        return scores


def sample_discrete_action_tokens(
    model,
    sample: dict,
    reasoning_text: str,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.98,
    top_k: int | None = None,
) -> torch.LongTensor:
    """Autoregressively sample the model's own 128 discrete action-bin ids.

    Continues generation from prefix + reasoning_text + <|traj_future_start|>
    (same prefix/reasoning-splice contract as score.py), constrained to the
    action-token vocab via ActionOnlyLogitsProcessor.

    Args:
        model: AlpamayoR1, already .to("cuda") in bf16.
        sample: dict with image_frames, ego_history_xyz/rot (same schema
            build_prompt/score expect; ego_future_* is not needed here since
            we are SAMPLING actions, not scoring ground truth).
        reasoning_text: the reasoning to hold fixed and condition on -- same
            role as score()'s reasoning_text argument.
        do_sample/temperature/top_p/top_k: standard sampling controls, same
            defaults sample_trajectories_from_data_with_vlm_rollout uses for
            its own generation call, so reasoning-generation and
            action-generation are sampled comparably.

    Returns:
        (1, 128) long tensor of bin ids in [0, num_bins-1], in the fixed
        order [accel_0, kappa_0, ..., accel_63, kappa_63] -- directly
        feedable into traj_tokenizer.detokenize_traj (same schema
        tokenize_traj's output uses).
    """
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    prompt = build_prompt(model, sample)
    prefix_ids = prompt["fused_input_ids"][0].tolist()  # ends in <|cot_start|>
    reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False)["input_ids"]
    end_marker_id = model.config.traj_token_ids["future_start"]  # <|traj_future_start|>

    input_ids = torch.tensor(
        [prefix_ids + reasoning_ids + [end_marker_id]], device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)
    prefix_len = input_ids.shape[1]

    n_action = model.config.tokens_per_future_traj  # 128

    # Clone rather than mutate model.vlm.generation_config in place --
    # alpamayo_r1.py's own generation call mutates that shared object
    # directly, which is fine there since it always sets the same fields for
    # its one use case; we don't want this call to leave stray min/max
    # new_tokens settings behind for some other caller.
    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = do_sample
    generation_config.temperature = temperature
    generation_config.top_p = top_p
    generation_config.top_k = top_k
    generation_config.num_return_sequences = 1
    generation_config.min_new_tokens = n_action
    generation_config.max_new_tokens = n_action
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = False

    logits_processor = LogitsProcessorList(
        [
            ActionOnlyLogitsProcessor(
                traj_token_offset=model.config.traj_token_start_idx,
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )

    aux = prompt["aux"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sequences = model.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=aux["pixel_values"],
            image_grid_thw=aux["image_grid_thw"],
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

    # min_new_tokens == max_new_tokens == n_action guarantees exactly n_action
    # new tokens with no padding, so we can slice by length directly instead
    # of hunting for a stop token the way replace_padding_after_eos does for
    # the reasoning-generation call.
    generated_vocab_ids = sequences[0, prefix_len:]
    assert generated_vocab_ids.shape[0] == n_action, (
        f"expected exactly {n_action} generated tokens, got {generated_vocab_ids.shape[0]}"
    )

    bin_ids = generated_vocab_ids - model.config.traj_token_start_idx
    if bool(((bin_ids < 0) | (bin_ids >= model.config.traj_vocab_size)).any()):
        raise ValueError(
            "generate() produced a token outside the action-token vocab range despite "
            "ActionOnlyLogitsProcessor -- masking bug, investigate before trusting output."
        )
    return bin_ids.unsqueeze(0)