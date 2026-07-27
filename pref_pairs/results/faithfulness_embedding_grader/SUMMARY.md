# Faithfulness embedding grader — status summary (2026-07-27)

Distilled, LLM-free faithfulness scorer: frozen MiniLM embeds the reasoning
trace, a small MLP ("head") projects raw waypoints into the same space,
faithfulness = cosine similarity. Module: `pref_pairs/faithfulness_embedding_grader.py`.

## What happened, in order

1. **Original training** (`eval_report.json`, `head.pt`): trained on 594
   judge-agreed (chosen, corrupted) pairs, 83.9% holdout pairwise accuracy —
   looked comparable to the Claude judge's 84.6%.
2. **Baselines killed that eval** (`baselines.json`): a constant text-space
   direction — no trajectory input at all — scores **95.8%** on the same eval.
   The corruptions leave a text fingerprint, so chosen-vs-corrupted accuracy
   does not measure trajectory-conditioned faithfulness.
3. **But the head does read the trajectory** (`trajectory_sensitivity.json`):
   its outputs are far from constant, and it prefers the true trajectory over
   a random one for clean traces. (Caveat: that 80.3% figure is
   scene-contaminated — see 4.)
4. **Swap-negative retrain + honest eval** (`swap_negatives_eval.json`,
   `head_swap_negatives.pt`): negatives = clean traces from other scenes;
   eval = 10-way retrieval (1 true + 9 other-scene traces), scene-level
   96/24 split over the **120 unique scenes** the dataset contains.
   - Constant-direction control: 10.2% top-1 = exactly chance → eval is
     corruption-proof. This is the metric to optimize.
   - Swap-trained head: **25.3% top-1 holdout** vs 86.3% on train scenes →
     real cross-modal signal, but heavy memorization of the 96 train scenes.
   - Old head "62.2%" on this eval is scene-contaminated (its triplet-level
     split trained on nearly all scenes) — not comparable.

## Where to pick up

- **Binding constraint is data: 120 unique scenes.** Mine more
  (trajectory, trace) pairs (e.g. clean_reasoning_actions, rollout harvests)
  before touching architecture.
- With current data: smaller/regularized head + k-fold CV over scenes
  (24 holdout scenes is too noisy for model selection).
- Retire chosen-vs-corrupted accuracy as a headline metric; report 10-way
  swap retrieval instead.
- Perturbation generator follow-up: make corruptions edit verifiable
  quantities while preserving style (see scratch note in
  perturbation_generator.py), so the corrupted-pair dataset stops being
  text-only-solvable. This also matters for the LLM judge's calibration set —
  its 84.6% may partly ride the same corruption fingerprint.
