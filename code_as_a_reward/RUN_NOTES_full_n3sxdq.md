# Run note: alpamayo-rl-code-reward-full-n3sxdq

**Status: EXPERIMENT_COMPLETED** — 2026-07-31 14:36 → 19:10 PDT (4h 33m), 8×A100, 0m queue.
First full code-reward run to finish end-to-end (rovn5p's disk-eviction fix held).

## Outcome

- Trained all **264/264 steps** in a single attempt. Final checkpoint `step_264`
  plus W&B files synced to
  `s3://research-datasets-chicago/alpamayo_rl/checkpoints/code_reward_full/20260731220943/`
  (355 files, ckpt-uploader final sync clean — no marker race).
- No OOM, no crash loop, no disk eviction, no 51-min stalls.

## Reward trend (train/reward_mean, from controller logs)

| steps | mean reward |
|---|---|
| 1–20 | −0.459 |
| 21–60 | −0.334 |
| 61–120 | −0.296 |
| 121–180 | −0.276 |
| 181–240 | −0.316 |
| 241–264 | −0.317 |

Clear improvement over the first ~half (−0.46 → −0.28), then a plateau with mild
regression in the back half. Best logged step: 94 (−0.037); final step 264 landed
at −0.454 (noisy per-step, std ~0.34 throughout). Worth checking W&B for LR
schedule / KL behavior around step ~180 before deciding whether to extend training.

## Issues observed (all non-fatal)

- **No overlay images for this run**: `TypeError: 'LoggingConfig' object is not
  subscriptable` recurred 83× (`[code_reward] W&B overlay logging failed (continuing)`).
  Expected — the run launched 14:36 PDT, before the fix in `2ed26ab` (16:13 PDT).
  Already logged in BUGS.md (2026-07-31, 2nd entry). Next run should have overlays.
- **9 distinct clips** failed obstacle scene load ("scoring commitments only" fallback).
- At least one clip dropped rows with unknown label class `stroller`
  (OBSTACLE_LABEL_CLASSES vocabulary gap).

Logs pulled via OCI Log Analytics 2026-08-02 (73k lines, full run coverage
21:36 UTC Jul 31 → 02:10 UTC Aug 1).
