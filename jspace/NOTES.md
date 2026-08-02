# J-space run notes

## 2026-08-02 — fit_lens full run results (jspace-fit-lens-ff9pr0)

Full J-lens fit on the Alpamayo 1.5 text backbone completed 2026-07-31
(23 min wall, ~12.5 min fitting, 1×A100). Lens artifact:
`s3://research-datasets-chicago/jspace/lens_full_v1/lens.pt` (~604 MB).
W&B: https://appliedintuition.wandb.io/research/jspace/runs/jspace-fit-lens-ff9pr0_7b69f73b62f_02000000_0

Config: 9 source layers [2, 6, 10, 14, 18, 22, 26, 30, 34] → target L35,
n_layers=36, d_model=4096, CoC prompts (`use_coc=True`), max_seq_len=128,
skip_first=16, chunk_size=5.

### Findings

1. **J-lens is numerically healthy on this backbone.** `max‖J‖/√d` stayed
   in 1.6–4.5 across all prompts and layers — order-1 causal influence,
   no vanishing or exploding Jacobians. Checkpoint size matches 9 dense
   4096² matrices (9 × 64 MB ≈ 604 MB), so all layers were genuinely fit.
   The precondition for the workspace analysis holds on a 10B VLA
   fine-tune, not just Claude-family text models.

2. **Length-dependence of causal reach (main signal).** Long prompts
   (seq_len 34–42) show ~10× larger `max_d_mean` (0.3–0.6 vs 0.02–0.05)
   and 2–3× larger Jacobian norms than short prompts (seq_len 18–23).
   Direction is a priori surprising: short prompts only have *nearby*
   (t, t′) pairs, which usually couple most strongly, yet show the weaker
   effect. Plausible reading: in reasoning-rich CoC prompts, early-token
   activations keep propagating into distant future states — long-horizon
   information persistence, the behavior research question 2
   ("holds the maneuver in mind") is probing. Caveats: short prompts had
   n_valid of only 1–6 pairs (noisy), and prompt content (caption vs
   reasoning) is confounded with length. Suggestive, not conclusive.

3. **Effective corpus is 41/100 prompts.** 59 prompts were skipped as
   too short (`seq_len` must exceed 17 because skip_first=16). Reference
   paper uses ~1000 prompts (~100 usable). Expect wide error bars on the
   decomposition outputs, especially the workspace depth-band location.
   The length filter also biases the corpus toward long reasoning
   prompts — arguably the distribution of interest, but not neutral.

4. **No fit-quality metric exists at this stage.** The fit only produces
   J_l matrices; all three research questions (silent mid-layer readouts,
   `<i*>` disposition before the trajectory block, depth band vs Claude's
   33–92%) are answered by `apply_lens` + `jspace_decomp`, not yet run.

### Log-reading notes (non-issues)

- A stack trace at 23:49 UTC is an idle-thread stack dump captured during
  an S3 checkpoint upload, not an error.
- The `ConnectionResetError` at shutdown is a benign wandb atexit
  teardown race; exit code 0 was recorded before it.

### Next steps

- Run `apply_lens` preferentially on long CoC prompts (10× better
  displacement signal-to-noise).
- Optional cheap re-fit with corpus pre-filtered to seq_len > 17 to
  recover the 59 lost slots (~7.5 s/prompt observed → ~13 min A100).
- Then `jspace_decomp` → per-layer J-space variance fraction → report.
