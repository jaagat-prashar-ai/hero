"""Fit J_l for the Alpamayo backbone over the driving corpus.

Wraps jlens.fit with the driving prompt corpus and sane defaults for the
36-layer / d_model-4096 backbone. The running sum is checkpointed after every
prompt (atomic write), so a killed run resumes where it left off.

Usage:
  python fit_lens.py --out out/lens.pt                 # all built-in prompts
  python fit_lens.py --out out/lens.pt --n-prompts 12  # quicker
  python fit_lens.py --out out/lens.pt --parquet .../ood_reasoning.parquet \
      --n-prompts 100                                  # top up with real coc

Cost scales as ceil(d_model / dim_batch) backward passes per prompt,
independent of how many source layers are fitted; more layers only cost
checkpoint size (d_model^2 * 4 bytes ~= 64 MB per layer fp32).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jlens  # noqa: E402
from load_model import load_lens_model  # noqa: E402
from prompts import corpus  # noqa: E402

# Every 4th layer of 36, spanning ~6%-94% depth: brackets the ~33%-92% Claude
# workspace band (README question 3) instead of assuming it transfers.
DEFAULT_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="path to save the fitted lens (.pt)")
    ap.add_argument("--checkpoint", default=None, help="resumable running-sum path (default: <out>.ckpt)")
    ap.add_argument("--n-prompts", type=int, default=None, help="truncate the corpus (default: all)")
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--target-layer", type=int, default=None, help="default: final layer")
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--parquet", default=None, help="ood_reasoning.parquet to top up the corpus")
    args = ap.parse_args()

    prompts = corpus(args.parquet)
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]
    checkpoint = args.checkpoint or args.out + ".ckpt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"fitting J_l at layers {args.layers} over {len(prompts)} prompts", flush=True)
    _, model = load_lens_model()
    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=args.layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        resume=True,
    )
    lens.save(args.out)
    print(f"fit {len(prompts)} prompts in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
