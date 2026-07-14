# Compares the model's discrete action head against its diffusion expert on
# the SAME real reasoning trace, for one scene.
#
# The diffusion side is the REAL, unmodified, already-existing entrypoint --
# alpamayo_r1/test_inference.py's own demo call to
# model.sample_trajectories_from_data_with_vlm_rollout(..., return_extra=True)
# -- nothing about that call is reimplemented here. That single call already
# generates a reasoning trace AND returns the actual generated text
# (extra["cot"]/extra["meta_action"]), so there is no need to separately
# extract or replay any KV cache: the generated text IS the shared
# conditioning, reused as-is for the discrete side.
#
# The discrete side reuses sample_discrete_traj.sample_discrete_action_tokens
# (already built/verified) with reasoning_text reconstructed from those exact
# same extra["cot"]/extra["meta_action"] strings, joined with the real
# special-token markers as literal text -- same convention score.py's
# docstring already establishes (registered special tokens round-trip
# exactly through the tokenizer, so this reproduces the true generated
# token sequence, not an approximation of it).

import numpy as np
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from dump_input_template import build_prompt
from s3_clip_loader import load_clip_from_s3_extract
from sample_discrete_traj import sample_discrete_action_tokens
from traj_tokenizer import detokenize_traj

CHECKPOINT = "nvidia/Alpamayo-R1-10B"
CLIP_DIR = (
    "/tmp/claude-4035/-home-jaagat-prashar-workspace-research-project-template-main-perplexity/"
    "64d82f1e-0858-4573-8406-df96512c11e1/scratchpad/s3/clip"
)
T0_US = 5_100_000
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # same clip test_inference.py/test_score.py use


def ade_fde(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Average/final displacement error (xy only) between two (T, 3) trajectories."""
    xy_err = (a[..., :2] - b[..., :2]).norm(dim=-1)
    return xy_err.mean().item(), xy_err[-1].item()


def main() -> None:
    model = AlpamayoR1.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    sample = load_clip_from_s3_extract(CLIP_DIR, CLIP_ID, t0_us=T0_US)

    # 1) Diffusion side: the real, unmodified demo call. This generates its
    # own reasoning trace via the model's own generate() -- we don't control
    # or duplicate that; we just read back what it actually generated.
    prompt = build_prompt(model, sample)
    tokenized_data = {"input_ids": prompt["raw_input_ids"], **prompt["aux"]}
    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": sample["ego_history_xyz"].to("cuda"),
        "ego_history_rot": sample["ego_history_rot"].to("cuda"),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    cot_text = extra["cot"][0, 0, 0]
    meta_action_text = extra["meta_action"][0, 0, 0]
    print(f"generated cot: {cot_text!r}")
    print(f"generated meta_action: {meta_action_text!r}")

    # Reconstruct the literal generated span (cot + meta_action, with the
    # real markers in between) as text -- this is what actually sat in the
    # diffusion path's conditioning context, not just the cot substring.
    reasoning_text = (
        f"{cot_text}<|cot_end|><|meta_action_start|>{meta_action_text}<|meta_action_end|>"
    )

    xyz_diffusion = pred_xyz[0, 0, 0]  # (64, 3)

    # 2) Discrete side: same reasoning_text, already-verified sampler.
    bin_ids = sample_discrete_action_tokens(model, sample, reasoning_text)
    hist_xyz, hist_rot = sample["ego_history_xyz"], sample["ego_history_rot"]
    xyz_discrete, _ = detokenize_traj(bin_ids.cpu(), hist_xyz, hist_rot)
    xyz_discrete = xyz_discrete[0, 0].to(xyz_diffusion.device)  # (64, 3)

    mean_err, final_err = ade_fde(xyz_discrete.float(), xyz_diffusion.float())
    print(
        f"discrete xy range=[{xyz_discrete[:, :2].norm(dim=-1).min():.2f}, "
        f"{xyz_discrete[:, :2].norm(dim=-1).max():.2f}]m"
    )
    print(
        f"diffusion xy range=[{xyz_diffusion[:, :2].norm(dim=-1).min():.2f}, "
        f"{xyz_diffusion[:, :2].norm(dim=-1).max():.2f}]m"
    )
    print(f"discrete vs diffusion: mean_xy_err={mean_err:.3f}m final_xy_err={final_err:.3f}m")


if __name__ == "__main__":
    main()