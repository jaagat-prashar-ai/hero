"""J-space decomposition: sparse nonnegative fit of residuals on J-lens vectors.

Not in the reference repo (README). For layer l the per-token J-lens vectors
are the rows of W_U J_l (unembedding with the final RMSNorm gain folded in,
composed with the transport). Each position's residual h_l is decomposed as

    h_l ~= sum_{y in S} c_y v_y,   c_y >= 0,  |S| <= k   (k ~= 25)

by nonnegative gradient pursuit (Blumensath & Davies 2008: greedy atom
selection + a line-searched gradient step on the active set, coefficients
clamped to >= 0). The explained variance fraction ||h||^2 - ||r||^2 over
||h||^2 is the position's J-space fraction; the paper reports J-space
carrying <= ~10% of activation variance, so small numbers are expected.

One JSONL record per (prompt, layer, position):
  {prompt_idx, layer, pos, tok, explained_var, atoms: [[tok, coef], ...]}
plus a stdout table of per-layer mean explained variance.

Usage:
  python jspace_decomp.py --lens out/lens.pt --out out/jspace.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jlens import ActivationRecorder, JacobianLens  # noqa: E402
from apply_lens import EVAL_PROMPTS  # noqa: E402
from load_model import load_lens_model  # noqa: E402


def gradient_pursuit(
    H: torch.Tensor, A: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose rows of H [P, d] on unit-norm dictionary A [n_atoms, d].

    Returns (sel [P, k] atom indices, coef [P, k] >= 0, explained [P]).
    Greedy: at each step pick the atom with the largest positive correlation
    with the residual (excluding already-selected), then take one exact
    line-search gradient step on the active set and clamp coefficients to
    >= 0. Positions whose best correlation is <= 0 stop growing.
    """
    P, d = H.shape
    R = H.clone()
    sel = torch.zeros(P, k, dtype=torch.long, device=H.device)
    coef = torch.zeros(P, k, device=H.device)
    used = torch.zeros(P, k, dtype=torch.bool, device=H.device)
    for i in range(k):
        corr = R @ A.T  # [P, n_atoms]
        if i > 0:
            corr.scatter_(1, sel[:, :i], float("-inf"))
        best, idx = corr.max(dim=1)
        grow = best > 0  # stop growing exhausted positions
        sel[:, i] = idx
        used[:, i] = grow
        active = A[sel[:, : i + 1]]  # [P, i+1, d]
        g = torch.einsum("pid,pd->pi", active, R) * used[:, : i + 1]
        step = torch.einsum("pi,pid->pd", g, active)  # A_S g
        denom = step.pow(2).sum(-1)
        alpha = torch.where(
            denom > 0, (R * step).sum(-1) / denom.clamp_min(1e-12), torch.zeros_like(denom)
        )
        coef[:, : i + 1] = (coef[:, : i + 1] + alpha[:, None] * g).clamp_min(0)
        R = H - torch.einsum("pi,pid->pd", coef[:, : i + 1] * used[:, : i + 1], active)
    coef = coef * used
    explained = 1 - R.pow(2).sum(-1) / H.pow(2).sum(-1).clamp_min(1e-12)
    return sel, coef, explained


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", required=True, help="fitted lens .pt from fit_lens.py")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--prompt", action="append", default=None, help="repeatable; default: EVAL_PROMPTS")
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all fitted layers")
    ap.add_argument("--k", type=int, default=25, help="max atoms per position")
    ap.add_argument("--top-atoms", type=int, default=8, help="atoms recorded per position")
    ap.add_argument("--skip-first", type=int, default=16, help="drop early positions, as in fitting")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    prompts = args.prompt or EVAL_PROMPTS
    lens = JacobianLens.load(args.lens)
    layers = args.layers or lens.source_layers
    _, model = load_lens_model()
    tokenizer = model.tokenizer
    device = model.input_device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Effective linear readout: RMSNorm gain folded into the unembedding.
    W_eff = model._lm_head.weight.float() * model._final_norm.weight.float()  # [vocab, d]

    per_layer_ev: dict[int, list[float]] = {l: [] for l in layers}
    with open(args.out, "w") as f:
        for p_idx, prompt in enumerate(prompts):
            input_ids = model.encode(prompt, max_length=args.max_seq_len)
            with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(input_ids)
            ids = input_ids.squeeze(0).tolist()
            positions = list(range(args.skip_first, len(ids)))
            for layer in layers:
                J = lens.jacobians[layer].to(device=device, dtype=torch.float32)
                A = W_eff @ J  # [vocab, d]: rows are per-token J-lens vectors
                A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
                H = rec.activations[layer].squeeze(0)[positions].float()
                sel, coef, explained = gradient_pursuit(H, A, args.k)
                order = coef.argsort(dim=1, descending=True)[:, : args.top_atoms]
                for row, pos in enumerate(positions):
                    atoms = [
                        [tokenizer.decode([sel[row, j].item()]), round(coef[row, j].item(), 4)]
                        for j in order[row].tolist()
                        if coef[row, j] > 0
                    ]
                    f.write(
                        json.dumps(
                            {
                                "prompt_idx": p_idx,
                                "layer": layer,
                                "pos": pos,
                                "tok": tokenizer.decode([ids[pos]]),
                                "explained_var": round(explained[row].item(), 5),
                                "atoms": atoms,
                            }
                        )
                        + "\n"
                    )
                per_layer_ev[layer] += explained.tolist()
                del A, J

    print(f"\nJ-space explained variance fraction (k={args.k}), by layer:")
    for layer in layers:
        vals = per_layer_ev[layer]
        mean = sum(vals) / max(len(vals), 1)
        print(f"  layer {layer:3d}: {mean:.4f}  (n={len(vals)})")
    print(f"records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
