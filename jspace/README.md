# J-space analysis of Alpamayo-R1

Apply Anthropic's Jacobian lens (J-lens) to the Alpamayo-R1 language backbone
(Qwen3-VL-2B) to look for a "global workspace" (J-space) in its mid-layers,
and to ask driving-specific questions:

1. Do driving concepts ("pedestrian", "brake", "yield", "merge") appear as
   silent J-space readouts in mid-layers *before* the CoT verbalizes them?
2. Are future trajectory tokens (`<i*>`, vocab ids 151669+) disposed in
   J-space before the trajectory block is emitted — i.e. does the model
   "hold the maneuver in mind" while still reasoning in text?
3. Where does the workspace band sit in a 2B VLA fine-tuned model vs. the
   ~33%-92% depth band reported for Claude models?

## Method (from the paper)

- J_l = E[dh_final,t' / dh_l,t], averaged over positions t, later positions
  t' >= t, and a prompt corpus (paper: ~1000 prompts x 128 tokens; ~100 is
  usable). First-order causal effect of a layer-l residual on final states.
- Lens readout: lens_l(h) = softmax(W_U · norm(J_l h)) — what the activation
  is disposed to make the model say later. Per-token J-lens vectors are the
  rows of W_U J_l.
- J-space: activations expressible as a sparse nonnegative combination
  (k ~= 25, gradient pursuit) of J-lens vectors; the orthogonal remainder is
  non-J-space. Reported to carry <=10% of activation variance.

## References

- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Reference implementation (we depend on it): https://github.com/anthropics/jacobian-lens
- Announcement: https://www.anthropic.com/research/global-workspace

## Layout

- `src/load_model.py` — load the Alpamayo-R1 checkpoint, expose the text
  backbone as a `jlens` LensModel (`jlens.from_hf(model.vlm, tokenizer)`).
- `src/prompts.py` — driving-domain fitting/eval prompt corpus.
- `src/fit_lens.py` — fit J_l over the corpus (checkpointed, resumable).
- `src/apply_lens.py` — dump layer x position lens readouts to JSONL.
- `src/jspace_decomp.py` — gradient-pursuit sparse decomposition +
  per-layer J-space variance fraction (not in the reference repo).

## Caveats

- Phase 1 is text-only (`input_ids`): CoT text and `<i*>` tokens work; real
  camera-frame inputs need a custom apply via `inputs_embeds` (later phase).
- The lens only surfaces concepts that map to single vocabulary tokens.
