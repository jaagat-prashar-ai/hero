"""Load Alpamayo 1.5 and expose its text backbone as a jlens LensModel.

The J-lens only needs the language decoder (residual blocks, final norm,
unembedding); jlens.from_hf locates them inside the Qwen3-VL wrapper via its
`model.language_model` layout, and the lm_head already covers the extended
vocab (<i*> trajectory tokens), so lens readouts can surface them.

Targets nvidia/Alpamayo-1.5-10B (Cosmos-Reason2-8B backbone: 36 layers,
d_model 4096) — the checkpoint the rest of this repo runs on — via the
vendored third_party/alpamayo1.5 source, same as masking/ and pref_pairs/.
"""

import sys
from pathlib import Path

import torch

# Repo root on sys.path for masking.bootstrap (vendored alpamayo1.5 source).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from masking.bootstrap import ensure_alpamayo1_5  # noqa: E402

ensure_alpamayo1_5()

import jlens  # noqa: E402
from alpamayo1_5.models import base_model as _base_model  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

CHECKPOINT = "nvidia/Alpamayo-1.5-10B"


def _tie_weights(self, **kwargs) -> None:
    """transformers>=5 passes recompute_mapping; forward it to the nested VLM."""
    if hasattr(self.vlm, "tie_weights"):
        self.vlm.tie_weights(**kwargs)


# Vendored alpamayo1.5 targets transformers 4.x; jlens needs >=5.5.
_base_model.ReasoningVLA.tie_weights = _tie_weights


def load_lens_model(
    checkpoint: str = CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Alpamayo1_5, "jlens.hf.HFLensModel"]:
    """Return (alpamayo, lens_model).

    Keep the Alpamayo1_5 handle: jlens.from_hf freezes all params in place, so
    this model instance is for lens work only — don't reuse it for training.
    """
    alpamayo = Alpamayo1_5.from_pretrained(checkpoint, dtype=dtype).to(device)
    lens_model = jlens.from_hf(alpamayo.vlm, alpamayo.tokenizer)
    return alpamayo, lens_model


def traj_token_range(alpamayo: Alpamayo1_5) -> tuple[int, int]:
    """Vocab-id range [start, end) of future-trajectory tokens <i0>..<iN-1>."""
    start = alpamayo.config.traj_token_start_idx
    return start, start + alpamayo.config.traj_vocab_size
