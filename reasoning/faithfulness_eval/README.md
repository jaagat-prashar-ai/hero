# Faithfulness eval (2026-08-27) + misalignment figure

- `judge.py` — the gpt-4o trajectory-vs-text judge (4 arms x 100 events x 2 rollouts).
- `judgments*.jsonl`, `gt_pairs.jsonl` — raw judge outputs; NOT reproducible without re-paying, keep in git.
- `rollouts_<arm>.jsonl` — local copies of S3 `alpamayo_rl/track1_submissions/fleet1/rollouts_<arm>.jsonl` (not in git).
- `test_egomotion/` — egomotion parquets streamed out of the S3 test WDS shards
  (`wds/test/shard_000_0000{0..4}.tar` + `wds/test_egomotion_fill/`) (not in git).
- `make_misalignment_figure.py` — builds `misalignment_figure.png`:
  (a) per-event mean judge score vs mean ADE against GT egomotion (r=0.04, n=100),
  (b) "Stop due to pedestrians in the crosswalk" rollout that accelerates into it.
