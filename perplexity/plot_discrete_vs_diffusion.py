# Visual discrete-vs-continuous comparison: one real generated reasoning
# trace, held fixed, fed into BOTH decode heads -- the discrete head once
# (sample_discrete_action_tokens -> detokenize_traj) and the diffusion
# expert num_traj_samples times
# (sample_diffusion_trajectories_given_fixed_reasoning) -- plotted top-down
# so the discrete trajectory's position relative to the diffusion samples'
# own spread can be read by eye. If the discrete trajectory sits inside the
# cloud of diffusion draws, the two heads agree within the diffusion head's
# own noise; if it sits clearly outside, they disagree beyond that noise.
#
# Colors follow this repo's dataviz convention (fixed categorical slots, not
# eyeballed): diffusion samples use categorical slot 1 (blue, #2a78d6,
# recedes as a cloud of many thin lines) and the single discrete trajectory
# uses slot 8 (orange, #eb6834, bold, stands out against the cloud) --
# a standard high-contrast, colorblind-safe pairing.

import matplotlib.pyplot as plt
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from dump_input_template import build_prompt
from s3_clip_loader import load_clip_from_s3_extract
from sample_diffusion_traj import sample_diffusion_trajectories_given_fixed_reasoning
from sample_discrete_traj import sample_discrete_action_tokens
from traj_tokenizer import detokenize_traj

CHECKPOINT = "nvidia/Alpamayo-R1-10B"
CLIP_DIR = (
    "/tmp/claude-4035/-home-jaagat-prashar-workspace-research-project-template-main-perplexity/"
    "64d82f1e-0858-4573-8406-df96512c11e1/scratchpad/s3/clip"
)
T0_US = 5_100_000
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # same clip test_inference.py/test_score.py use
NUM_DIFFUSION_SAMPLES = 10
OUT_PATH = "results/discrete_vs_diffusion.png"

DIFFUSION_COLOR = "#2a78d6"  # dataviz categorical slot 1 (blue)
DISCRETE_COLOR = "#eb6834"  # dataviz categorical slot 8 (orange)


def main() -> None:
    model = AlpamayoR1.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to("cuda")
    sample = load_clip_from_s3_extract(CLIP_DIR, CLIP_ID, t0_us=T0_US)

    # 1) Get one real generated reasoning trace, the same way
    # compare_discrete_continuous.py does -- the real, unmodified demo call.
    prompt = build_prompt(model, sample)
    tokenized_data = {"input_ids": prompt["raw_input_ids"], **prompt["aux"]}
    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": sample["ego_history_xyz"].to("cuda"),
        "ego_history_rot": sample["ego_history_rot"].to("cuda"),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    cot_text = extra["cot"][0, 0, 0]
    meta_action_text = extra["meta_action"][0, 0, 0]
    reasoning_text = (
        f"{cot_text}<|cot_end|><|meta_action_start|>{meta_action_text}<|meta_action_end|>"
    )
    print(f"reasoning (held fixed for both heads): {cot_text!r}")

    # 2) Discrete head, once.
    bin_ids = sample_discrete_action_tokens(model, sample, reasoning_text)
    hist_xyz, hist_rot = sample["ego_history_xyz"], sample["ego_history_rot"]
    xyz_discrete, _ = detokenize_traj(bin_ids.cpu(), hist_xyz, hist_rot)
    xyz_discrete = xyz_discrete[0, 0].cpu().float().numpy()  # (64, 3)

    # 3) Diffusion expert, K times, off the SAME reasoning_text.
    pred_xyz, _ = sample_diffusion_trajectories_given_fixed_reasoning(
        model, sample, reasoning_text, num_traj_samples=NUM_DIFFUSION_SAMPLES
    )
    xyz_diffusion = pred_xyz.cpu().float().numpy()  # (K, 64, 3)

    # 4) Plot top-down (x, y), diffusion cloud first so the discrete line
    # draws on top of it.
    fig, ax = plt.subplots(figsize=(7, 7))
    for k in range(NUM_DIFFUSION_SAMPLES):
        ax.plot(
            xyz_diffusion[k, :, 0],
            xyz_diffusion[k, :, 1],
            color=DIFFUSION_COLOR,
            alpha=0.5,
            linewidth=1.5,
            label=f"diffusion samples (K={NUM_DIFFUSION_SAMPLES})" if k == 0 else None,
        )
    ax.plot(
        xyz_discrete[:, 0],
        xyz_discrete[:, 1],
        color=DISCRETE_COLOR,
        linewidth=2.5,
        label="discrete head (detokenized)",
        zorder=5,
    )
    ax.scatter([0], [0], color="#0b0b0b", s=25, zorder=6, label="t0 (ego origin)")
    ax.set_xlabel("x (m, ego frame)")
    ax.set_ylabel("y (m, ego frame)")
    ax.set_title(f"Discrete vs. diffusion trajectories, same reasoning\nclip {CLIP_ID}")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
