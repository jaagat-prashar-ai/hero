# T1.4: score(sample, reasoning_text) -> 128 per-token NLLs for the GT
# discrete future-trajectory tokens.
#
# This is Route A from the very first design discussion: perplexity under the
# model's discrete action head, via ordinary next-token cross-entropy -- NOT
# the flow-matching expert (Route B), which this never touches.
#
# Reuses, doesn't reimplement:
#   - dump_input_template.build_prompt(model, sample)  -- T1.3's exact prefix
#     (ends at <|cot_start|>, vision+ego-history tokens already fused in)
#   - traj_tokenizer.tokenize_traj(...)                -- T1.2's GT action
#     bin indices (native tokenizer output, [0, num_bins-1])
#
# The only new piece is: prefix + reasoning_text + <|traj_future_start|> +
# GT action tokens, one forward pass, then read off log p(true next token) at
# each of the 128 action positions via the standard causal-LM shift-by-one
# (logits at position p predict the token at position p+1).
#
# reasoning_text is deliberately just a string, not auto-wrapped with
# <|cot_end|>/<|meta_action_start|>/<|meta_action_end|> -- the whole point of
# this project is comparing real vs. no vs. shuffled reasoning conditions, so
# the caller decides what goes in (including embedding those markers as
# literal text if they want the true generation format; the tokenizer maps
# those substrings back to their real special-token ids since they were
# registered as such).

import numpy as np
import torch

from dump_input_template import build_prompt
from traj_tokenizer import tokenize_traj


def score(model, sample: dict, reasoning_text: str) -> np.ndarray:
    """Per-token NLLs (nats) for the 128 GT discrete action tokens, given reasoning_text as context.

    Args:
        model: AlpamayoR1, already .to("cuda") in bf16 (as from_pretrained(..., dtype=torch.bfloat16)).
        sample: dict with image_frames, ego_history_xyz/rot, ego_future_xyz/rot
            (same schema load_physical_aiavdataset / load_clip_from_s3_extract return).
        reasoning_text: the text to condition on, inserted between the prefix's
            trailing <|cot_start|> and <|traj_future_start|>.

    Returns:
        (128,) float32 array of NLLs, in the fixed token order
        [accel_0, kappa_0, accel_1, kappa_1, ..., accel_63, kappa_63].
    """
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    prompt = build_prompt(model, sample)
    prefix_ids = prompt["fused_input_ids"][0].tolist()  # ends in <|cot_start|>

    reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False)["input_ids"]
    end_marker_id = model.config.traj_token_ids["future_start"]  # <|traj_future_start|>

    action_bin_ids = tokenize_traj(
        sample["ego_history_xyz"],
        sample["ego_history_rot"],
        sample["ego_future_xyz"],
        sample["ego_future_rot"],
    )[0].tolist()
    # action_bin_ids are in [0, num_bins-1] = [0, 2999] per channel (accel,
    # kappa) -- that's the size of the meaningful range for a discrete action
    # token. It is NOT the size of the softmax denominator below: every
    # position still normalizes over the full model vocabulary (155,697
    # tokens), never just these 3,000. A model can be -- and in the
    # mechanism artifact's Case C, is -- sharply confident about the wrong
    # one of those 155,697 options.
    action_vocab_ids = [model.config.traj_token_start_idx + b for b in action_bin_ids]
    n_action = len(action_vocab_ids)  # 128

    full_ids = prefix_ids + reasoning_ids + [end_marker_id] + action_vocab_ids
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    aux = prompt["aux"]
    # A plain forward call -- model.vlm(input_ids=...), the same __call__ any
    # PyTorch/transformers module exposes -- not model.vlm.generate(...).
    # generate() is a different method entirely, built for sampling new
    # tokens one at a time; it has no reason to exist here since every token
    # in full_ids is already known. Calling it would mean re-deriving what
    # forward already gives us directly.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=aux["pixel_values"],
            image_grid_thw=aux["image_grid_thw"],
            use_cache=False,
        )
    # Nothing conversational comes out of this call. logits is the only
    # thing model.vlm() returns here -- no decoded text, no chat turn -- and
    # everything below just turns it into 128 floating-point numbers.
    logits = outputs.logits  # (1, L, vocab_size)

    action_start = len(full_ids) - n_action
    # logits at position p predict the token at position p+1 -- shift by one.
    pred_logits = logits[0, action_start - 1 : action_start - 1 + n_action, :].float()
    targets = input_ids[0, action_start : action_start + n_action]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    nlls = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nlls.cpu().numpy()


