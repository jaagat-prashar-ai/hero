"""Apply a fitted J-lens to prompts and dump layer x position readouts.

One JSONL record per (prompt, layer, position):
  {prompt_idx, layer, pos, tok, next_tok, lens_top: [[tok, p], ...],
   model_top: [[tok, p], ...], traj_mass}
where lens_top is the J-lens readout softmax(W_U norm(J_l h)) top-k,
model_top is the model's actual final logits at the same position (same in
every layer's record, kept for self-contained analysis), and traj_mass is the
lens probability mass on the <i*> trajectory-token range — the README's
question 2 signal ("is the maneuver held in mind before the block is
emitted").

Usage:
  python apply_lens.py --lens out/lens.pt --out out/readouts.jsonl
  python apply_lens.py --lens out/lens.pt --out r.jsonl --prompt "..." --layers 14 22
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jlens import JacobianLens  # noqa: E402
from load_model import load_lens_model, traj_token_range  # noqa: E402

# Held out from prompts.DRIVING_PROMPTS. The last one ends in a literal
# trajectory-token block (the <i*> strings tokenize to single vocab tokens),
# so text positions *before* the block probe pre-emission traj disposition.
EVAL_PROMPTS = [
    "A school bus ahead has stopped with its red lights flashing and the stop"
    " arm extended. The ego vehicle must stop behind it and wait until the"
    " lights stop flashing before proceeding.",
    "A garbage truck is double-parked blocking our lane while workers load it."
    " The ego vehicle should signal left, wait for a gap in oncoming traffic,"
    " and pass slowly with wide clearance.",
    "A pedestrian has stepped into the crosswalk ahead, so the ego vehicle"
    " should brake smoothly and stop before the crosswalk."
    "<i2000><i2010><i2020><i2030><i2040><i2050>",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", required=True, help="fitted lens .pt from fit_lens.py")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--prompt", action="append", default=None, help="repeatable; default: EVAL_PROMPTS")
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all fitted layers")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    prompts = args.prompt or EVAL_PROMPTS
    lens = JacobianLens.load(args.lens)
    alpamayo, model = load_lens_model()
    traj_start, traj_end = traj_token_range(alpamayo)
    tokenizer = model.tokenizer
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def decode(tok_id: int) -> str:
        return tokenizer.decode([tok_id])

    n_records = 0
    with open(args.out, "w") as f:
        for p_idx, prompt in enumerate(prompts):
            lens_logits, model_logits, input_ids = lens.apply(
                model, prompt, layers=args.layers, max_seq_len=args.max_seq_len
            )
            ids = input_ids.squeeze(0).tolist()
            model_probs = model_logits.float().softmax(-1)
            m_top = torch.topk(model_probs, args.topk, dim=-1)
            for layer, logits in sorted(lens_logits.items()):
                probs = logits.float().softmax(-1)
                traj_mass = probs[:, traj_start:traj_end].sum(-1)
                top = torch.topk(probs, args.topk, dim=-1)
                for pos in range(len(ids)):
                    rec = {
                        "prompt_idx": p_idx,
                        "layer": layer,
                        "pos": pos,
                        "tok": decode(ids[pos]),
                        "next_tok": decode(ids[pos + 1]) if pos + 1 < len(ids) else None,
                        "lens_top": [
                            [decode(i), round(p, 5)]
                            for i, p in zip(top.indices[pos].tolist(), top.values[pos].tolist())
                        ],
                        "model_top": [
                            [decode(i), round(p, 5)]
                            for i, p in zip(m_top.indices[pos].tolist(), m_top.values[pos].tolist())
                        ],
                        "traj_mass": round(traj_mass[pos].item(), 6),
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_records += 1
    print(f"wrote {n_records} records for {len(prompts)} prompts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
