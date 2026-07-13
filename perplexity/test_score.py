# T1.4 "Done when" check: score() returns 128 finite floats for 3 different
# samples, at a wall time roughly comparable to one prefill of alpamayo's own
# test_inference.py demo (that script's whole run, including a 128-token
# *generation* loop, took well under a minute on this GPU -- score() does a
# single forward pass, no generation, so should be much faster per call).

import time

import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from s3_clip_loader import load_clip_from_s3_extract
from score import score

CHECKPOINT = "nvidia/Alpamayo-R1-10B"
CLIP_DIR = (
    "/tmp/claude-4035/-home-jaagat-prashar-workspace-research-project-template-main-perplexity/"
    "64d82f1e-0858-4573-8406-df96512c11e1/scratchpad/s3/clip"
)
T0_US = 5_100_000
CLIP_IDS = [
    "030c760c-ae38-49aa-9ad8-f5650a545d26",  # alpamayo's own demo clip
    "00032edf-c04b-428c-b3ee-21377dd70a80",
    "001f9280-2240-4183-80c8-ce1b4b3c40eb",
]
# Real CoC text for the first clip (from Step 1's test_inference.py run). Not
# real generated reasoning for the other two -- T1.4 only needs score() to
# work correctly for different samples, not real per-clip reasoning yet.
REASONING_TEXT = "Nudge to the left to increase clearance from the construction cones encroaching into the lane."


def main() -> None:
    model = AlpamayoR1.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to("cuda")

    for clip_id in CLIP_IDS:
        sample = load_clip_from_s3_extract(CLIP_DIR, clip_id, t0_us=T0_US)

        torch.cuda.synchronize()
        t0 = time.time()
        nlls = score(model, sample, REASONING_TEXT)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        assert nlls.shape == (128,), f"expected shape (128,), got {nlls.shape}"
        assert bool((nlls == nlls).all()), f"NaN in NLLs for {clip_id}"  # NaN != NaN
        assert bool((abs(nlls) < float("inf")).all()), f"Inf in NLLs for {clip_id}"

        ppl = float(torch.tensor(nlls).mean().exp())
        print(
            f"{clip_id}: {elapsed:.2f}s, mean_nll={nlls.mean():.4f}, "
            f"perplexity={ppl:.2f}, min={nlls.min():.4f}, max={nlls.max():.4f}"
        )

    print("T1.4 OK: 128 finite NLLs returned for 3 different samples.")


if __name__ == "__main__":
    main()
