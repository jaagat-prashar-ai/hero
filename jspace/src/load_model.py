"""Load Alpamayo-R1 and expose its text backbone as a jlens LensModel.

The J-lens only needs the language decoder (residual blocks, final norm,
unembedding); jlens.from_hf locates them inside the Qwen3-VL wrapper via its
`model.language_model` layout, and the lm_head already covers the extended
vocab (<i*> trajectory tokens), so lens readouts can surface them.
"""

import sys
from pathlib import Path

import torch

# alpamayo_r1 lives in the sibling experiment dir, not on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perplexity" / "alpamayo" / "src"))

import jlens  # noqa: E402
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: E402

CHECKPOINT = "nvidia/Alpamayo-R1-10B"


def load_lens_model(
    checkpoint: str = CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AlpamayoR1, "jlens.hf.HFLensModel"]:
    """Return (alpamayo, lens_model).

    Keep the AlpamayoR1 handle: jlens.from_hf freezes all params in place, so
    this model instance is for lens work only — don't reuse it for training.
    """
    alpamayo = AlpamayoR1.from_pretrained(checkpoint, dtype=dtype).to(device)
    lens_model = jlens.from_hf(alpamayo.vlm, alpamayo.tokenizer)
    return alpamayo, lens_model


def traj_token_range(alpamayo: AlpamayoR1) -> tuple[int, int]:
    """Vocab-id range [start, end) of future-trajectory tokens <i0>..<iN-1>."""
    start = alpamayo.tokenizer.traj_token_start_idx
    return start, start + alpamayo.config.traj_vocab_size
