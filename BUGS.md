# Bug tracker

Log of confirmed, non-obvious bugs found and fixed in this repo — what broke, why,
and how it was confirmed. Newest first. Scope: bugs worth remembering months from
now, not routine typos.

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
