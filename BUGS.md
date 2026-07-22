# Bug tracker

Log of confirmed, non-obvious bugs found and fixed in this repo — what broke, why,
and how it was confirmed. Newest first. Scope: bugs worth remembering months from
now, not routine typos.

---

## 2026-07-22 — full OOD run crashed at training start: `t0_us must be greater than the history time range`

**Symptom:** `alpamayo-rl-llm-judge-full-5ieeuh` (first 352-clip-scale llm-judge
run) hit `EXPERIMENT_FAILED` 33 min in, seconds after vLLM came up — rank0
raised `AssertionError: t0_us must be greater than the history time range`
from `alpamayo_r1/load_physical_aiavdataset.py:98` via the prefetch server.
The fail-fast wrapper surfaced the real traceback immediately.

**Root cause:** boundary disagreement between two vendored components. The
recipe's `pai_utils.filter_clips_by_event_t0s` keeps OOD events with
`t0 >= start_safe_margin_seconds` (1.6 s, `>=`), but the loader asserts
STRICTLY `t0_us > num_history_steps * time_step` = 16 x 0.1 s = the same
1.6 s. **295 of 1731 OOD clips have their first surviving event at exactly
1,600,000 µs** (timestamps evidently clamped to the margin when the dataset
was built), so any of them crashes the loader on first touch. Small random
canaries (16 clips, seed 42) never sampled one; scale did — 17% odds per
clip.

**Fix:** `2f4628c` — select_dense_ood_chunks.py reproduces the runtime's
event-margin filter and drops clips whose first kept event fails the strict
assert (the data packer always reads `sample_index_in_clip=0`). The margin
isn't reachable via hydra overrides (the dataset ctor doesn't expose it), so
selection-time filtering is the only non-vendored-edit fix. Densest-100
config: 392 -> 352 clips.

Same-day sibling fix: `9bc5bd5` — the S3 warm-cache upload failed with
`NotImplemented: AWS chunked encoding not supported` (OCI S3 compat);
uploads must use put_object + payload signing, never boto3 upload_file
(build_wds already knew this — its `_OCI_BOTO_CONFIG` comment documents it).

## 2026-07-22 — llm-judge canary died at step 3: NCCL watchdog killed a reward-starved policy

**Symptom:** `alpamayo-rl-llm-judge-canary-u0j67p` (8-GPU a100 node, reward_mode
`llm_judge`) hit `EXPERIMENT_FAILED` after 41 min. Steps 1/6 and 2/6 trained
normally (iteration time ~22-25s, real judge scores flowing), then all policy
ranks crashed with:
```
[Worker] Task <_Task ... timeout_ms=600000> done | timed_out=True
[NCCL] Aborted communicator idx=0
TimeoutError: NCCL: non-blocking enqueue timed out
```
followed by torchrun `ChildFailedError` → launcher `Process 1 failed with
return code 1` → our fail-fast wrapper (c3c7ed9) dumped per-rank logs and
tore the job down as designed.

**Root cause:** three compounding throughput problems, no code crash at all.
(1) cosmos-rl's NCCL watchdog (`COSMOS_NCCL_TIMEOUT_MS`, default 600 000 ms,
see `cosmos_rl/utils/pynccl.py` at the pinned rev 747d1bd) aborts any
communicator whose pending collective exceeds 10 min — after step 2 the
policy ranks sat in exactly such a collective waiting for the step-3 batch.
(2) The default reward path scores a 12-rollout GRPO group **serially**, one
blocking Anthropic API call per rollout (~1-7s each), so reward throughput
lagged rollout production — controller backlog grew 24 → 96 pending over the
run. (3) Most groups failed the ADE/reasoning gates uniformly and got the
flat -1.0 reward → zero within-group advantage variance → GRPO discards
them, so filling a 48-rollout step with usable groups took even longer.
Steps 1→2 already took 3.7 min; step 3 never made it under 10.

**Fix (three commits):**
- `c71cc05` — raise `COSMOS_NCCL_TIMEOUT_MS` to 1h (+ `COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT`
  to 3600s) in `run.py`, scoped to reward_mode=llm_judge only.
- `b377b8f` — `group_reward_calculation = true` in the llm_judge TOML +
  `compute_reward_batch` fans judge HTTPS calls over a thread pool
  (`LLM_JUDGE_MAX_CONCURRENCY`, default 8); GPU-local decode stays serial.
- `2692180` — gate-failing rollouts get a graded reward in [-1.0, -0.5]
  (`_graded_failure_reward`) instead of the flat -1.0, restoring advantage
  variance in all-fail groups; missing-CoC keeps flat -1.0.

Verification: 27 pure-helper tests pass, and relaunch canary
`alpamayo-rl-llm-judge-canary-grq1cf` (2026-07-22, W&B run
`research/alpamayo-rl/runs/20260722185529`) confirmed end-to-end:
EXPERIMENT_COMPLETED, all 45 steps, median inter-step gap 29s (vs ~3.7 min
serial before), reward_std 0.19-0.35 per step (advantage variance restored).
One 733s mid-run stall would still have tripped the old 600s watchdog —
the 1h timeout absorbed it, so both throughput AND timeout fixes were needed.
Diagnosis details: default `lilypad workload logs` window (last 4h) missed
the original run entirely — pass `--start-time/--end-time` around the
`workload info` Created/Finished timestamps. Also: cosmos-rl replica stdout
only lands in OCI at teardown, so a quiet log stream during training is
normal; use W&B for live progress.

## 2026-07-02 — masking experiment C failed on every clip: `unknown mask mode: prefix`

**Symptom:** `masking_loop` with `experiment=c` produced zero successful rows —
every single event logged `ERROR:masking.training.run:clip <id> t0=<t>: unknown
mask mode: prefix` and was counted as a failure, while experiments A and B ran
correctly against the same code/data.

**Root cause:** `run.py::_run_experiment_c()` builds conditions like
`{"mode": "prefix", "n": n, "unit": "words"}` and passes them to
`MaskedAlpamayo1_5.compare_conditions()`, which resolves each condition's mask
columns via `_cols_for_spec()`. That dispatcher only handled
`"none"/"reasoning"/"concept"/"explicit"` and raised `ValueError` for anything
else. `masked_model.py` already had fully-implemented
`_prefix_mask_columns()`/`_suffix_mask_columns()` methods (matching signature:
`(seq, n, unit)`) sitting unused right below `compare_conditions` — they were
just never wired into the dispatch. This looks like the two pieces were
written in the same pass but the connecting `if` branches were never added;
nothing about it depended on data or environment, so it would have failed
identically the very first time experiment C was ever run.

**Fix:** added the two missing branches to `_cols_for_spec()`:
```python
if mode == "prefix":
    return self._prefix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"))
if mode == "suffix":
    return self._suffix_mask_columns(seq, spec["n"], spec.get("unit", "tokens"))
```
Verification pending re-launch (an unrelated results-storage bug found at the
same time required stopping the first post-fix run before it finished).

## 2026-07-01 — 34/100 build-physicalai-wds ranks failed with HF `/whoami-v2` 429 at launch

**Symptom:** Relaunching the WDS build at `world_size=100` (`build-wds-parallel
100 1`) produced 34 `EXPERIMENT_FAILED` ranks (scattered across the full
0-99 range, e.g. `p0`, `p50`, `p90`), each within ~4-8 minutes of submission.
All had the same traceback:
```
huggingface_hub.errors.HfHubHTTPError: You've hit the rate limit for the
/whoami-v2 endpoint, which is intentionally strict for security reasons.
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url
'https://huggingface.co/api/whoami-v2'
```
The 8-way smoke test at the same code version did not trigger this — 8
simultaneous logins didn't trip the throttle, 100 did.

**Root cause:** `build_webdataset.py:main()` called
`huggingface_hub.login(token=args.hf_token, add_to_git_credential=False)`
unconditionally on every rank. `login()` validates the token via a call to
`/whoami-v2` before caching it — an endpoint with a much stricter rate limit
than the general resolver endpoints (the already-known 5000 req/5min limit
in [[reference_lilypad_cluster_ops]] does not apply here). `build-wds-parallel`
launches all `WORLD_SIZE` jobs back-to-back (~2s apart), so at world_size=100
all 100 `login()` calls landed within the same few-minute window.

**Fix:** [ff4eebe](../../commit/ff4eebe) removed the explicit `--hf_token`
argv plumbing first (unrelated cleanup, done same session); this fix replaces
the `login()` call with `os.environ.setdefault("HF_TOKEN", args.hf_token)`.
Every downstream HF call (including `PhysicalAIAVDatasetInterface()`, which
takes no explicit token) resolves its token via `huggingface_hub.get_token()`,
which checks `HF_TOKEN` before the login-cache file — so setting the env var
is sufficient and skips the `/whoami-v2` network round-trip entirely, rather
than just staggering it.

Also fixed two related launcher bugs found while diagnosing this, both in
`build_wds/configs/launch.sh`:
- `build-wds-parallel`'s default `WORKERS` was `2`, contradicting
  `cluster.yaml`'s own `workers: 1` comment (concurrent chunk-ZIP downloads
  OOM the ~30GB head node). Default changed to `1`.
- `build-wds-staggered` hardcoded `world_size=50` regardless of the actual
  run's world_size — reusing it to relaunch ranks from a `world_size=100` run
  would have silently broken `chunk_id % world_size == rank` partitioning
  (and outright errored for any rank ≥ 50). `world_size` is now an explicit
  argument.

**How this was found:** user asked why some of the 100 relaunched jobs
failed; `lilypad workload logs` on 3 sample failed ranks (`p0`, `p90`, `p26`)
showed the identical `/whoami-v2` 429 traceback in each.

---

## 2026-07-01 — S3 shard uploads silently failing on OCI (100% failure rate, 17h+ undetected)

**Symptom:** All 8 parallel `build-physicalai-wds-p0..p7` Lilypad jobs ran for
17h+ reporting healthy-looking `Progress: N ok / M err` counters (e.g.
`400 ok / 8 err`), but a direct listing of
`s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/{train,val}/`
showed **zero** rank-prefixed shard tars (`shard_XXX_YYYYY.tar`) from any of
them — only pre-existing, unrelated test artifacts. Every clip processed by
every partition was silently discarded; none of it ever reached S3.

**Root cause (two compounding bugs, both in `build_wds/data/build_webdataset.py`):**

1. `S3ShardWriter._flush()` uploaded shard tars with `boto3`'s `upload_file()`,
   which routes through `s3transfer`'s multipart `TransferManager`. That path
   always issues `UploadPart` requests with AWS chunked transfer-encoding,
   which OCI's S3-compatible endpoint rejects outright:
   `botocore.exceptions.ClientError: An error occurred (NotImplemented) when
   calling the UploadPart operation: AWS chunked encoding not supported.`
   The existing `payload_signing_enabled=True` / `request_checksum_calculation=
   when_required` client config (added specifically to work around OCI's lack
   of chunked-encoding support) only affects single-shot `PutObject` calls —
   it does nothing for `s3transfer`'s multipart path. `upload_metadata_parquets()`
   in the same file already worked around this correctly by using `put_object`
   with an in-memory buffer instead — that code comment was the tell. Shard
   tars are bounded in size (~125MB for 50 clips) and hit `upload_file`'s
   16MB multipart threshold every time, so **every single shard upload since
   this job started failed**, with `NotImplemented` not on the transient-error
   allowlist in `_s3_retry`, so it failed on the first attempt with no retry.

2. `S3ShardWriter._flush()`'s `finally` block unconditionally deleted the local
   tempfile and advanced `_shard_idx` / reset `_count` — even when the upload
   raised. Combined with `main()`'s `process()` only marking the *one* clip
   whose `write()` call happened to trigger the flush as failed (the other
   ~49 clips in the same shard had already been counted into `n_ok` by their
   own earlier, individually-successful `write()` calls), this meant a failed
   shard's data vanished with no retry and no accurate accounting — the
   `n_ok`/`n_err` progress counters looked fine while ~98% of the "successful"
   clips in each failing shard were actually being thrown away.

**Fix:** [728494d](../../commit/728494d), [4950da4](../../commit/4950da4)
- Switch `_flush()` to `put_object` with the tar buffered in memory (proven
  pattern from `upload_metadata_parquets`), eliminating the multipart/chunked-
  encoding path entirely. Verified against the real OCI endpoint with a 50MB
  in-memory payload (well above the old 16MB multipart threshold) — succeeds.
- Added `ShardUploadFailed(clips_lost=...)`, raised from `_flush()` on
  permanent failure. `process()` now catches it specifically and moves *all*
  `clips_lost` clips from `n_ok` into `n_err`, so the counters can no longer
  lie about data having landed.

**Known residual risk (not fixed, low priority since root cause is gone):**
`--resume_file` records a clip's ID as done immediately after its own
`write()` call returns, before the shard containing it is flushed. If a
shard upload still fails for some other reason in the future, the ~49
clips already recorded in the resume file will be permanently skipped on
a resumed run even though their data was lost. Acceptable for now because
the chunked-encoding failure mode that caused 100% of observed losses is
fixed at the root; revisit only if shard upload failures reappear.

**How this was found:** user asked to inspect logs for
`build-physicalai-wds-p7-vuvz8a` and check S3 upload status. Log inspection
via `lilypad workload logs` surfaced the `S3UploadFailedError` tracebacks;
cross-checking `aws s3api list-objects-v2` against the actual bucket (not
just the job's self-reported counters) is what revealed the 100% real
failure rate. All 8 sibling jobs (`p0`-`p7`) were stopped via `lilypad
workload stop` once confirmed to share the same code and same bug.

---

## 2026-06-30 — WDS rank partitioning crash when shard count < world_size

**Symptom:** Training crashed with `No samples found in dataset; perhaps you
have fewer shards than workers` on most ranks whenever the number of WDS
shards was smaller than `world_size` (e.g. 2 shards, 8 ranks) — seen in
`masking-cot-cluster-jcyksk` logs.

**Root cause:** `masking/data/wds_dataset.py`'s `iter_snapshots()` passed
`nodesplitter=wds.split_by_node` to `WebDataset`, which slices the *shard
list itself* by rank/world_size — on top of `masking_loop`'s own independent
sample-level rank partitioning (`_shard_owner()`, hash-based, expects every
rank to see the full shard list). Any rank whose index fell outside the
shard-count-sized slice got zero shards and crashed.

**Fix:** [0375203](../../commit/0375203) — replaced `wds.split_by_node` with
a no-op `_no_node_split` nodesplitter so every rank sees every shard; sample-
level partitioning in `masking_loop` is left to do the actual work division.

---

## Format for new entries

```
## YYYY-MM-DD — one-line symptom

**Symptom:** what was observed (error text, log line, incorrect behavior).
**Root cause:** why it actually happened — the non-obvious mechanism.
**Fix:** [commit-sha](../../commit/sha) — what changed and why that's correct.
**How this was found:** (optional but valuable) what investigation surfaced it.
```
