# Done-when check for sample_discrete_action_tokens: generation actually
# produces exactly 128 tokens, every one of them lands inside the
# action-token vocab range (proves ActionOnlyLogitsProcessor's masking is
# working, not just that the assert-on-count happens to pass), and
# detokenizing them through the existing traj_tokenizer round-trips to a
# plausible (non-exploding) xyz trajectory.
#
# Per this project's real-model-only testing convention: no mocking, run
# against the real checkpoint on GPU, same clips/reasoning_text test_score.py
# already uses so results are comparable to that file's numbers.

import time

import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from s3_clip_loader import load_clip_from_s3_extract
from sample_discrete_traj import sample_discrete_action_tokens
from traj_tokenizer import detokenize_traj


CHECKPOINT = "nvidia/Alpamayo-R1-10B"
CLIP_DIR = (
    "/tmp/claude-4035/-home-jaagat-prashar-workspace-research-project-template-main-perplexity/"
    "64d82f1e-0858-4573-8406-df96512c11e1/scratchpad/s3/clip"
)
T0_US = 5_100_000
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # same clip test_score.py/test_inference.py use
# Same real CoC text test_score.py uses for this clip, so this file's numbers
# are directly comparable to that file's.
REASONING_TEXT = "Nudge to the left to increase clearance from the construction cones encroaching into the lane."


def main() -> None:
    model = AlpamayoR1.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    sample = load_clip_from_s3_extract(CLIP_DIR, CLIP_ID, t0_us=T0_US)

    torch.cuda.synchronize()
    t0 = time.time()
    bin_ids = sample_discrete_action_tokens(model, sample, REASONING_TEXT)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    assert bin_ids.shape == (1, 128), f"expected shape (1, 128), got {bin_ids.shape}"
    assert bin_ids.dtype.is_floating_point is False
    assert bin_ids.min() >= 0 and bin_ids.max() <= 2999, (
        f"bin id out of [0, 2999]: min={bin_ids.min().item()} max={bin_ids.max().item()}"
    )

    # bin_ids comes back on the model's device (cuda) since generate() needs
    # input_ids there; detokenize_traj/action_to_traj do no internal device
    # management and expect everything on one device, matching
    # test_traj_tokenizer.py's existing CPU-only convention for this utility.
    hist_xyz, hist_rot = sample["ego_history_xyz"], sample["ego_history_rot"]
    fut_xyz, _ = detokenize_traj(bin_ids.cpu(), hist_xyz, hist_rot)

    # ego_history_xyz (and thus fut_xyz) carries a leading n_traj_group dim
    # in this dataset's schema: (B, n_traj_group, T, 3), not (B, T, 3).
    assert fut_xyz.shape == (1, 1, 64, 3), f"expected shape (1, 1, 64, 3), got {fut_xyz.shape}"
    assert bool(torch.isfinite(fut_xyz).all()), "non-finite values in detokenized trajectory"

    xy_dist_from_origin = fut_xyz[0, 0, :, :2].norm(dim=-1)
    print(
        f"{CLIP_ID}: {elapsed:.2f}s, sampled 128 action tokens, "
        f"detokenized xy range=[{xy_dist_from_origin.min():.2f}, {xy_dist_from_origin.max():.2f}]m"
    )
    # Loose sanity bound, not a precision check: a plausible 64-step (6.4s
    # @10Hz) future trajectory shouldn't cover hundreds of meters. This just
    # catches "the sampled tokens/detokenization are producing garbage",
    # not fine-grained trajectory quality.
    assert xy_dist_from_origin.max() < 200.0, (
        f"detokenized trajectory travels {xy_dist_from_origin.max():.1f}m -- looks like garbage, "
        "check ActionOnlyLogitsProcessor masking and bin-id conversion"
    )

    print("Discrete-sampling OK: 128 valid action-vocab tokens generated and detokenized to a finite, plausible trajectory.")


if __name__ == "__main__":
    main()