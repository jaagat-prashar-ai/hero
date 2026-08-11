# Bug tracker

Log of confirmed, non-obvious bugs found and fixed in this repo — what broke, why,
and how it was confirmed. Newest first. Scope: bugs worth remembering months from
now, not routine typos.

---

## 2026-08-09 (2nd) — run_real_rollout_gen.py's S3 sync used upload_file, hitting the same known chunked-encoding rejection

**Symptom:** `clipgen-real-rollout-smoke-05vvru` failed after successfully
sampling all 3 clips' real rollout groups: `botocore.exceptions.ClientError:
An error occurred (NotImplemented) when calling the PutObject operation: AWS
chunked encoding not supported`, raised from `boto3.exceptions.S3UploadFailedError`
inside the periodic sync thread, which killed the whole Ray Train trial.

**Root cause:** `run_real_rollout_gen.py`'s `_sync_dir` used boto3's managed
`s3.upload_file(...)`, which always uses chunked transfer encoding regardless
of file size — the OCI S3-compat endpoint used by this cluster rejects that
encoding. This exact constraint was already documented and worked around
elsewhere in this repo (`code_as_a_reward/clipgen/run_prototype.py`'s
`_sync_out_to_s3`, `rl_posttrain/rewards/code_reward_entry.py`'s
`_CheckpointUploader` — both use `put_object` with the whole body read into
memory instead), but `run_real_rollout_gen.py` was modeled after
`code_as_a_reward/ood_eval/run.py`'s `_upload`, which has the SAME
`upload_file` call and is therefore latently exposed to the same failure
(not yet fixed there — out of scope for this session, flagging for later).

**Fix:** committed alongside this entry — `_sync_dir` now reads each file's
bytes and calls `s3.put_object(Bucket=..., Key=..., Body=...)`. Rollout JSON
files are tiny (well under 1 MB), so whole-body put_object is fine.

**How this was found:** direct log inspection after relaunching the smoke
with the `extra["cot"]` indexing fix below — the new failure surfaced
immediately on the very next attempt.

---

## 2026-08-09 (1st) — rollout_sampler.py indexed Alpamayo's extra["cot"] as a flat list; broke for num_traj_samples > 1

**Symptom:** `clipgen-real-rollout-smoke-f2e7vq` completed (EXPERIMENT_COMPLETED)
but every one of its 3 clips failed rollout sampling with `IndexError: index 1
is out of bounds for axis 0 with size 1`, logged from `rollout_worker.py`.

**Root cause:** `code_as_a_reward/clipgen/rollout_sampler.py`'s
`sample_rollout_group` was adapted from `code_as_a_reward/ood_eval/worker.py`'s
`run_model_rollout`, which calls Alpamayo's
`sample_trajectories_from_data_with_vlm_rollout` with `num_traj_samples=1` and
indexes `extra["cot"][0]` directly. That only worked by coincidence: per
`third_party/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`'s actual
implementation, `extra["cot"]` is reshaped to a `(B, num_traj_sets,
num_traj_samples)` **numpy array**, not a flat list — with `num_traj_samples=1`
every axis is size 1, so `[0]`/`[i]`-style indexing silently "worked" for the
wrong reason. Bumping `num_traj_samples` to 12 (to sample a whole rollout
group in one forward pass) immediately exposed the real shape.

**Fix:** [ba406d9](../../commit/ba406d9) — reshape both `pred_xyz` and
`extra["cot"]` to a flat `n`-length leading axis (`.reshape(n, *pred_xyz.shape[-2:])`
/ `.reshape(-1)`) instead of hand-indexing specific axis positions; robust
regardless of the exact `(B, num_traj_sets, num_traj_samples)` axis
convention since `B = num_traj_sets = 1` always holds for a single-clip call.

**How this was found:** first real-GPU smoke test of the new real-rollout
gate loop (`code_as_a_reward/clipgen/run_prototype.py`'s 2026-08-09 redesign,
see its module docstring) — direct traceback inspection via `lilypad workload
logs ... --role all`.

---

## 2026-08-07 (4th) — simlingo smoke_baseline.yaml still had the batch_size that OOMed on the contrastive arm

**Symptom:** none observed in a live run — found while preparing to launch
the baseline (w=0) arm for comparison against the now-healthy contrastive
(w=0.5) smoke run. `smoke_baseline.yaml`'s header comment claims it's
"identical to smoke_contrastive.yaml except the contrastive loss weight is
0," but its `data_module.batch_size` was still `4`.

**Root cause:** `54c63e0` (2026-08-06) halved `smoke_contrastive.yaml`'s
`batch_size` to 2 after `va9ixh`'s CUDA OOM (batch_size=4 x contrastive
K=4 fan-out ~= 16-sample effective forward batch, ~72GB allocated on an
80GB A100) but only edited that one file — `smoke_baseline.yaml` was never
touched, so it silently drifted from its own stated invariant. Launching
it as-is would almost certainly OOM on the first forward pass (identical
fan-out math), and even if it happened to survive, a different batch size
than the contrastive arm breaks the clean ablation the two runs are
supposed to provide (same batches, only the loss differs).

**Fix:** [649694d](../../commit/649694d) — `smoke_baseline.yaml`'s
`data_module.batch_size` set to 2, matching `smoke_contrastive.yaml`.

**How this was found:** re-reading the baseline config immediately before
launching it, prompted by the user asking for a baseline run to compare
against the contrastive arm — the header comment's claim of "identical
except loss weight" made the batch_size mismatch a one-line diff-catch
rather than something requiring reproduction.

---

## 2026-08-07 (3rd) — build_webdataset.py never stored per-frame timestamps, silently corrupting camera-frame timing downstream

**Symptom:** while building WDS shards for the 214 Track 1 OOD test-split
clips (`build_test_split.py`), `perplexity/s3_clip_loader.py`'s existing
documented caveat — that S3-sourced camera frames land ~15-20s off the
requested moment — turned out to be caused by data missing from the WDS
shards themselves, not a loader bug. `decode_last_n_frames()` has to fall
back to "last N frames by index" specifically because no per-frame
timestamp data was ever stored for it to align against.

**Root cause:** `avdi.features.get_clip_files_in_zip(clip_id, feature)`
returns three entries per camera — `video`, `frame_timestamps`, and
`blurred_boxes` — but `_read_camera_mp4_bytes()` (the only reader
`build_clip_sample` called) only ever extracted `video`. The
`frame_timestamps` parquet sitting right next to it in the same chunk zip
was never read or written into any WDS shard, for train, val, or test.

**Fix:** [aaf23d9](../../commit/aaf23d9) — renamed the reader to
`_read_camera_zip_entries`, returning `(video_bytes, timestamps_bytes)`
from a single zip open, and `build_clip_sample` now writes
`{clip_id}.{cam_key}.timestamps.parquet` alongside each camera's `.mp4`.
Applied before the test-split build ran, so those new `wds/test/` shards
have it from the start. The existing `train/`/`val/` shards on S3 were
built before this fix and still lack it — not backfilled here, scope was
the test-split build only.

**How this was found:** investigating why `s3_clip_loader.py`'s frame-timing
caveat existed at all, while scoping a chain-of-causation generation task
that needs visually-grounded inference over these same clips — traced it
to `get_clip_files_in_zip`'s actual return value rather than assuming it
was a downstream decoding limitation.

---

## 2026-08-07 — simlingo contrastive smoke crashed on dynamically-imported conversation.py missing `get_conv_template`

**Symptom:** `simlingo-contrastive-smoke-ujuu6j` (the arial.ttf relaunch)
trained ~28 min then a rank-0 DataLoader worker (worker 7) raised
`AttributeError: module 'get_conv_template' has no attribute
'get_conv_template'. Did you mean: '_return_value'?` from
`internvl2_utils.py:129`, killing the whole job (`RuntimeError: train.py
exited with code 1 on rank 0`). The crash landed in the same few seconds as
a burst of `Crashed routes`/dataset-init scan prints across many workers —
consistent with an epoch-boundary DataLoader worker respawn hammering the
same cache path at once.

**Root cause:** `get_custom_chat_template` dynamically re-downloads
(`snapshot_download`, only if missing) and re-imports
(`importlib.util.spec_from_file_location` + `exec_module`) the encoder's
`conversation.py` from scratch on **every single collate call**, across all
8 ranks x 8 dataloader workers, for the whole run — thousands of redundant
disk reads/execs with no caching. Confirmed the upstream file itself is
fine (fetched `OpenGVLab/InternVL2-1B`'s current `conversation.py` — it
still defines `get_conv_template` and registers `'internlm2-chat'`), so
this wasn't an upstream API change. The exact trigger (a transient
partial/inconsistent read during a 64-worker respawn burst) couldn't be
pinned down further without the on-disk state at the crash instant —
this run's own earlier, unlogged crash #1 was 8 TorchTrainer workers
racing on shared `/mnt/work` tarballs, the same shared-filesystem-race
class — but the massive unnecessary read/exec surface is what made any
such transient race possible in the first place.

**Fix:** [0862928](../../commit/0862928) — cache the imported module in a
process-local dict keyed by `model_path` so each worker loads/execs the
file at most once instead of every batch; serialize the first
download+import behind a `filelock.FileLock` (already a transitive dep via
`huggingface_hub`) so concurrent workers can't race on the same path at
startup/epoch boundaries; assert `hasattr(conv_module, 'get_conv_template')`
right after exec with a clear message instead of letting a bad load surface
as a cryptic `AttributeError` deep in the dataloader. Note:
`team_code/agent_simlingo.py` (CARLA closed-loop eval agent, not on the
training path) has the identical unguarded pattern at lines 595-615 — not
fixed here since it doesn't block training, but the same fix should be
applied before it's used for eval.

**How this was found:** `lilypad workload logs ujuu6j --role all
--start-time/--end-time`, grepped for `File "` to find the real traceback
under the Ray/Lightning wrapper noise; the `internvl2_utils.py:129` frame
led to the actual `AttributeError`.

---

## 2026-08-07 (2nd) — simlingo checkpoint shards could silently scatter across ranks: `wandb_name` (-> `hydra.run.dir` -> checkpoint dir) is a per-process timestamp, never pinned

**Symptom:** none observed yet in a live run — found during the deep
investigation after the bug above, before relaunching. `train.py`'s
`ModelCheckpoint` dirpath is the relative `./checkpoints`, and
`simlingo_seed1.yaml`/`simlingo_seed2.yaml` set `hydra.run.dir:
outputs/${wandb_name}_${name}`. Each of the 8 ranks is an
independently-launched `train.py` subprocess (`simlingo_lilypad/run.py`), so
each resolves `${wandb_name}` from its own process's config load.

**Root cause:** `simlingo_training/config.py:152`'s `wandb_name` field
defaults to `f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"`, evaluated
independently per rank at that rank's own `train.py` startup, and no
`hydra_overrides` entry pins it. Not just low-probability skew: `run.py`
has ranks 1-7 block in `_wait_for_marker`, polling every 20s for rank 0 to
finish the S3 dataset extraction before they launch `train.py` — so a
multi-second gap between rank 0's and every other rank's `train.py` launch
is close to guaranteed on every run, at 1-second timestamp resolution. Each
rank's DeepSpeed checkpoint save (`./checkpoints` relative to its own
`hydra.run.dir`) would then land in a different directory — no crash or
error, just an unreconstructable checkpoint the first time anyone tries to
resume from one (a full checkpoint needs every rank's model+optimizer shard
in the same directory).

**Fix:** [49c8869](../../commit/49c8869) — `simlingo_lilypad/run.py`'s `_get_shared_run_name()`:
rank 0 writes a timestamp to `<workdir>/.run_name` (once; reused across
requeues on the same node like the existing `.extract_done` marker), every
other rank polls for and reads that same file, and the resulting value is
passed as an explicit `wandb_name=<value>` hydra override to every rank's
`train.py` invocation — so `hydra.run.dir` (and the checkpoint dir under
it) is byte-identical across all 8 ranks regardless of launch skew.

**How this was found:** deep-read of `train.py`/`config.py`/the experiment
yamls specifically looking for anything untested by the 9 crashed smoke
attempts (all died before reaching a checkpoint save); traced
`hydra.run.dir`'s `${wandb_name}` interpolation back to its dataclass
default to confirm it really is per-process wall-clock, not a shared value.

---

## 2026-08-06 — simlingo train-viz callback killed contrastive smoke run: wrong arial.ttf path, no try/except

**Symptom:** `simlingo-contrastive-smoke-6njfmd` (the batch_size=2 relaunch
after va9ixh's OOM) died 44 min in with `OSError: cannot open resource` from
`PIL ImageFont.truetype` inside `VisualiseCallback.on_train_batch_end`
(rank 0), status EXPERIMENT_FAILED. Earlier `visualise_training_examples
cannot open resource` lines at 19:49/19:50 UTC were the same failure caught
on the validation path.

**Root cause:** `visualise_waypoints` built the font path as
`get_original_cwd() + "/simlingo_training/arial.ttf"`, but the font lives at
the package root `simlingo/simlingo/arial.ttf` — the path doesn't exist even
in a local checkout, so any run reaching the train-viz interval crashes.
Validation viz is wrapped in try/except (only logs), but
`on_train_batch_end` called `_visualise_training_examples` bare, so the
OSError propagated and killed training. Masked until now because earlier
smoke runs OOMed before reaching the viz step.

**Fix:** [e7885cb](../../commit/e7885cb) — resolve the font relative to `__file__`
(`Path(__file__).resolve().parents[2] / "arial.ttf"`) with a
`ImageFont.load_default()` fallback, and wrap train-time viz in the same
try/except as the validation path so visualization can never kill a run.

**How this was found:** `lilypad workload logs 6njfmd` traceback:
`visualise.py:220` → `ImageFont.truetype` → `OSError: cannot open resource`.

## 2026-08-05 — clipgen lateral_offset_m is road geometry on curving clips, not in-lane position

**Symptom:** clipgen smoke runs (62eytm/g9349h/ejodln): every generated
reward function with a lateral check failed on the ground truth itself;
b7f37a71's positive stuck at 0.5 for six attempts across two runs. Gate
feedback facts showed `lateral final -99.53 m` on fe20b8b9's GT.

**Root cause:** `lateral_offset_m` is raw y in the FROZEN t=0-heading frame
(`pref_pairs/trajectory_features.py:269`). On curving roads it accumulates
the road's geometry: measured 91 deg turn -> 22 m "offset" (b7f37a71),
22 deg sweep -> 95 m (fe20b8b9), vs ~0.3 m for a real in-lane nudge. Also
verified NOT a frame bug: egomotion initial headings are within 1.4 deg of
zero on all 5 smoke clips (dossier.py's old "extract_features handles the
rotation" comment was wrong — it never rotates; the data just complies).
In-lane position is unrecoverable from egomotion alone (no lane reference),
so the fix is disclosure, not reconstruction: dossier/API/feedback now warn
the generator off lateral checks on curvy clips (e137df1).

## 2026-08-05 — clipgen gate was unpassable by construction: prescribed additive rubric + GT-claims-carrying negatives

**Symptom:** gpt-4o smoke 62eytm scored 0/5 with every trajectory negative
floored at exactly 0.6 (perception 0.3 + commitment 0.3 from the prompt's
prescribed "~0.3/~0.3/~0.4" split — negatives carry GT claims, so 2x the
0.3 ceiling before the trajectory is inspected). Separately, cross-clip
negatives from the 5-clip decel-heavy pool were sometimes unwinnable:
01340cf8's `other_claims_3` donor parses to the SAME canonical claim keys
as its GT (no function on (claims, traj) can separate them), scored 1.0
all 3 attempts. Plus 3/15 attempts lost to `if <numpy array>` ValueError.

**Fix (960fd11, e946615):** free-form generation (no rubric; gate stated
numerically: pos >= 0.7, each corruption >= 0.4 below), battery rebuilt as
corruptions of the SAME rollout (reversed/flattened traj, gutted/
no-commitment claims) replacing cross-clip negatives, numpy-truthiness
warning in the API reference, and measured per-case trajectory facts
appended to gate feedback (g9349h showed retries were blind without them:
monotonicity checks failing on the noisy GT itself, three identical
attempts). Timing named as the reversal discriminator — speed-magnitude
checks provably cannot separate time-symmetric profiles (fe20b8b9: same
speeds, min at t=5.6 s vs t=14.3 s).

---

## 2026-08-03 — regex_excludes silently dead in every launch config: `^dir/` can never match the SDK's `/`-prefixed paths

**Symptom:** every `lilypad/launch.py` submit uploaded a ~5 GB code zip
at ~1 MB/s (~85 min), making fresh launches look "invisible" on lilypad
for over an hour. The zip was dominated by `perplexity/` clip caches
(2.53 GB), the root `.venv` (1.20 GB), and `jspace/` (1.12 GB, mostly
its own venv) — all things the exclude list was supposed to keep out.
Actual code needed by a job: ~32 MB.

**Root cause:** the lilypad SDK matches each candidate path as
`"/" + path.relative_to(root)` (`lilypad/public/sdk_py/packaging_utils.py:48`,
`_get_excludes`) — so the string under test is `/perplexity/clip_cache/...`.
Every pattern in every config was anchored as `^perplexity/`, which can
never match a string starting with `/`. All excludes across all 18
launch configs were silently ignored on every launch to date (including
completed runs n3sxdq and fbbpdd — they trained fine, just uploaded
~50× the payload).

**Fix:** anchor all patterns with a leading slash (`^/perplexity/`),
escape the dot in `alpamayo1\.5`, and add previously-missing excludes
for `/\.venv/` (unanchored — catches root and `jspace/.venv`),
`^/jspace/`, and `^/\.git/` to the code-reward full config. Verified by
replaying the SDK's own match logic over the repo: included payload
drops 5 GB → 32.4 MB.: step_255 checkpoint mostly lost to max_keep=1 pruning; 24eb078 doesn't cover multi-GB payload

**Symptom:** post-run S3 audit of `alpamayo-rl-llm-judge-mock-4dgpaq`
(otherwise fully successful: 264/264 steps in one attempt, disk-eviction
fix held, final checkpoint intact). Beyond the marker losses already
logged in the entry below, `step_255/policy/` has only **6 of ~21
objects** on S3 (no `model_rank_0/3`, no `optimizer_rank_0/1/3`, no
`cosmos_config`, one marker) — not resumable. `step_75` has all 20
payload files but no `.rank_3_complete`. All 16 other intermediate
checkpoints and final `step_264` are complete, including all four
markers (the final checkpoint is never pruned, so it escapes the race).

**Root cause:** this run predates [24eb078](../../commit/24eb078) (the
entry below was written FROM this run's logs), but the mechanism it
exposes is one 24eb078 does not fix: with `max_keep = 1`
([684f315](../../commit/684f315)), cosmos-rl prunes checkpoint N in
full the moment N+1 finishes saving. step_264's save landed at 01:18
UTC while the uploader was still shipping step_255's multi-GB shards;
everything not yet uploaded vanished mid-pass, and rescan-based retry
can never recover a deleted file (the 01:19 "(will retry)" ERROR lines
were false promises). 24eb078 snapshots only small files into memory —
model/optimizer shards still upload from disk over a multi-minute pass,
so the second-to-last checkpoint of every run remains exposed. The two
disk fixes interact: max_keep=1 (needed to stop pod eviction) made
pruning aggressive enough to widen the race from markers to payload.

**Fix:** none yet — final checkpoint is what the real-API run needs and
it verifies complete, so this only bites a run that must resume from
the second-to-last checkpoint after a late crash. If that matters:
have the uploader pin/rename a checkpoint dir aside until its pass
completes, or make pruning wait for upload confirmation.

**How this was found:** log grep for `ckpt-uploader`/`FileNotFoundError`
(10 hits) cross-checked object-by-object against
`aws --profile oci.chi s3 ls` for every affected step prefix.

---

## 2026-07-31 (3rd) — ckpt-uploader loses race vs cosmos-rl marker cleanup: every S3 checkpoint missing .rank_3_complete, one lost a 10 GB optimizer shard (both reward modes)

**Symptom:** In `alpamayo-rl-llm-judge-mock-4dgpaq` logs, `ckpt-uploader:
upload failed ... (will retry)` + `FileNotFoundError` for
`.rank_3_complete` at *every* checkpoint (steps 15/30/45/60/75) and once
for `step_60/policy/optimizer_rank_1.pth` (10 GB) — none of these ever
reached S3, so step_60's S3 copy is not fully resumable. Same rank-3
marker gap in `code-reward-full-n3sxdq` step_30 (silently — no error
when deletion beats the scan itself).

**Root cause:** cosmos-rl deletes its `.rank_N_complete` markers shortly
after all ranks finish saving. `_CheckpointUploader` scans, then uploads
sequentially in directory order — a pass shipping multi-GB shards runs for
minutes, so files deleted after the scan hit `FileNotFoundError` at their
upload turn. Rank 3 saves last, so its marker systematically lost the
race. The "(will retry)" log line was also false for this case: retry
works by rescanning, and a deleted file never reappears.

**Fix:** [24eb078](../../commit/24eb078) — small checkpoint files are
snapshotted into memory at scan time (uploaded from the snapshot if since
deleted), each pass uploads smallest-first so markers don't queue behind
optimizer shards, and vanished files log an honest "NOT retrying" error.

**How this was found:** log inspection of both live 2026-07-31 full runs;
grep showed `.rank_0/1/2_complete` uploaded for every step but `.rank_3`
for none.

---

## 2026-07-31 (2nd) — code-reward overlay images never logged: LoggingConfig treated as a dict

**Symptom:** `alpamayo-rl-code-reward-full-n3sxdq` logged `[code_reward]
W&B overlay logging failed (continuing)` with `TypeError: 'LoggingConfig'
object is not subscriptable` on every sampled group (54×) — zero overlay
images, sibling W&B run never created. Training itself unaffected.

**Root cause:** `_get_overlay_run` in `rl_posttrain/rewards/code_reward_entry.py`
read `logging_cfg["experiment_name"]` / `["project_name"]`, but cosmos-rl's
`config.logging` is a `LoggingConfig` object (attribute access only). The
llm-judge overlay path (`aggregated_reward_llm_judge.py`) already used
`config.logging.project_name` correctly — the dict style crept in only in
the code-reward sibling-run variant, and the blanket `except` around
overlay logging hid it from the smoke run.

**Fix:** [2ed26ab](../../commit/2ed26ab) — attribute access, matching the
working llm-judge path and the runtime config dump.

---

## 2026-07-31 — checkpoints fill the node's root disk → kubelet evicts the pod at the 3rd save: EVERY full-run attempt died 44-45 min in ("reaper"/NodeDiedError, both reward modes)

**Symptom:** `alpamayo-rl-code-reward-full-rovn5p` (12h26m) ended
`EXPERIMENT_FAILED` with `ray.exceptions.NodeDiedError` from `ray.get` at
`run.py:1084`. OCI logs show SEVEN training attempts across six recreated
Ray clusters, each starting from step 0 (no cross-requeue resume), each
dying within ~1 min of a checkpoint-upload burst. The parallel
`alpamayo-rl-llm-judge-mock-qq9r6y` died in lockstep (every W&B run for
both workloads lasted 44-45 min). Training itself never crashed once.
Fluentd sidecars shut down gracefully at each death — pod eviction, not
hardware failure.

**Root cause:** disk arithmetic. `/mnt/work` lives on the node's 992 GB
root filesystem, whose baseline after setup is ~764 GB (77%: 570 GB
dataset warm cache + venv + HF and converted models). cosmos-rl saves a
~60 GB checkpoint every 15 steps (4x5 GB model + 4x10 GB optimizer);
`_CheckpointUploader` shipped them to S3 but never deleted local copies,
and `max_keep = 10` meant cosmos-rl wouldn't prune until 10 (600 GB)
accumulated. W&B system metrics for the final attempt: 77.0% flat →
step_30 save +56 GB → 82.7% → step_45 save begins → **85.3%** at
09:57:32, node declared dead at 09:57:38 — kubelet's default
`imagefs.available<15%` DiskPressure hard-eviction threshold. At ~48
s/step the third save lands ~44 min in, hence the metronomic lifetime.
This also retro-explains the llm-judge-full "step ~148 reaper" crash
loop: at judge step cadence the disk wall arrives near checkpoint 9-10 ≈
step 148. Three earlier attributions (idle-GPU reaper, HF throttling,
reward stalls) were wrong.

**Fix:** [52859d9](../../commit/52859d9) — `_CheckpointUploader` deletes
each checkpoint tensor file (>1 MiB, under `checkpoints/`) after a
successful `put_object`, capping the local footprint at one in-flight
checkpoint (~83% peak, verified against kubelet's 85% line); failed
uploads keep the file for retry. [684f315](../../commit/684f315) —
`max_keep = 1` as the in-framework backstop. Headroom is still only ~19
GB at peak — if a future run grows the baseline (bigger dataset, extra
caches), revisit trimming the warm cache after per-clip pre-extraction.

---

## 2026-07-30 (2nd) — vendored reasoning term INVERTED inside the passing band: full credit at barely-passing, zero at perfect

**Symptom:** none observable in aggregate curves — found by re-deriving the
passing-branch arithmetic while answering "how will the coverage blend move
the gate pass-rate". With `reasoning_weight = 0.3` and
`reasoning_threshold = -0.4`, the passing branch computed
`+ w * (reasoning_score / reasoning_threshold)`: perfect reasoning
(score 0) earned **0.0** while barely-passing reasoning (score -0.39)
earned **+0.29**. The term DECREASES in reasoning quality.

**Root cause:** the vendored recipe's `aggregated_reward_with_reasoning.py`
divides a negative score by the negative threshold, producing a ratio that
is 1 at the gate and 0 at perfect — the mirror of what a reward should be.
The l2 term (`-w * l2/ade`) and comfort term point the right way; reasoning
is the only inverted component, which is why it reads as plausible. Both of
our entries (`aggregated_reward_llm_judge.py`, `code_reward_entry.py`)
copied it verbatim under the "change only the reasoning source" principle,
so EVERY llm-judge and code-reward run to date trained with a
within-passing-band gradient pushing reasoning quality DOWN toward the
gate. Impact was bounded because most rollouts fail the gate (where
`_graded_failure_reward`'s ordering is correct) and GRPO advantage is
group-normalized — but the model's best-reward strategy inside the band
was "be barely faithful enough", and with the coverage blend an
all-abstain trace (score -0.2 → +0.15) out-earned a fully-verified one
(score ~0 → ~0) at the final-reward level even after the gate-ordering
fix in the entry below.

**Fix:** [683dd4d](../../commit/683dd4d) — passing-branch reasoning term mirrored to
`w * (1 - reasoning_score / reasoning_threshold)` in BOTH entries: 0 at
the gate, full weight at perfect. The passing floor stays -0.2 (now at
barely-passing instead of, absurdly, at perfect reasoning), so graded
failures still rank strictly below every passing rollout; the ceiling
rises from 0 to +0.3 (harmless under GRPO's within-group normalization).
The vendored submodule itself is untouched, per rl_posttrain convention;
comparability caveat: reward curves before/after this date are not
directly comparable in either reward mode.

**How this was found:** user asked how the coverage blend would compress
reasoning scores and shift the gate pass-rate; plugging the band endpoints
into the mixing formula exposed the inversion.

---

## 2026-07-30 — code-reward W&B metrics silently biased/NaN-poisoned + all-abstain traces out-scored half-verified ones

**Symptom:** bugs28.txt open question ("zero decided claims? what is this
failure case about?"). `code_reward_raw`/`code_atomic_precision` were written
as `math.nan` when a rollout decided nothing, and the no-CoC branch emitted an
aux dict containing only `code_scene_available`.

**Root cause (three compounding, all around zero-decided rollouts):**

1. cosmos-rl's `aggregate_report_data` (`cosmos_rl/utils/util.py:1387`)
   reduces per-rollout report_metrics into per-step W&B values with
   `np.mean([data.get(k, 0) ...])`. A single NaN rollout therefore NaNs the
   whole step's metric (latent — zero NaN steps in all 69 logged steps
   across a1npli/eivn91/vtx3ys, because multi-claim rollouts rarely
   all-abstain).
2. The same `data.get(k, 0)` treats a *missing* key as literal 0 — for a
   precision metric, "every claim failed". The no-CoC branch's one-key aux
   dict hit exactly this: vtx3ys had one step whose `reward_min` is exactly
   -1.0 (the flat no-CoC penalty), so that step's code metrics are biased
   low in W&B.
3. Reward semantics: a fully-undecided trace took the fixed neutral -0.2
   and PASSED the reasoning gate while a half-verified trace (r=0.5 →
   -0.5) FAILED it — "say only unverifiable things" strictly beat "be
   half right". Not hypothetical phrasing: the all-abstain shape is
   "keep a safe distance from the lead vehicle" (keep_distance commitment
   abstains as relative-to-agent; the 'lead' attribute has no ground
   truth), and 5.2% of the 1434 judged_pairs traces have no decidable
   commitment at all.

**Fix:** [8a69ac3](../../commit/8a69ac3) — every rollout emits the full aux
key set; undecided rollouts carry the group's decided-only mean instead of
NaN (mean-neutral by construction); new `code_undecided_cnt` /
`code_no_cot_cnt` counters (summed per step via the aggregator's `_cnt`
suffix convention) keep the fill-in rate visible. Reasoning credit is now
coverage-blended — `r_eff = decided_fraction·r + (1-df)·0.8` — so credit is
proportional to how much of the trace was verifiable; a fully-undecided
trace keeps exactly the old -0.2 minus the unparsed penalty that
`score_trace` could never apply to it (r=None short-circuited the penalty).
Verified: 110 tests pass; on a real clipgen scene, all-abstain scores
-0.263 < partially-verified GT -0.145, and a false "stop" claim still
gate-fails at -0.600. Note: the blend compresses reasoning scores toward
-0.2, so `reasoning_threshold = -0.4` no longer reads "60% of decided
claims verified" — recalibrate against the next run's gate pass-rate.

**How this was found:** code-reading `aggregate_report_data` after
bugs28.txt flagged the NaN sentinel; parse-sweep over 1434 judged_pairs
traces + full-path scoring of the 5 clipgen GT clips (which also surfaced
the canonical all-abstain trace shape); W&B history scan of all three
code-reward runs for NaN steps and flat -1.0 `reward_min` rollouts.

---

## 2026-07-29 — code-reward ~51-min stalls, for real this time: cosmos-rl's HTTP retry ladder exhausting on a failed rollout-report POST

**Symptom:** the obstacle-extraction fix below (`e901539`) didn't change the
stall shape. Direct profiling against the real dataset put end-to-end
`_load_scene` at ~0.5s (0.003s to open a 53.9 MB chunk zip, 0.08s to read the
one member, 0.01-0.40s to parse) — an order of magnitude too fast to explain
a 51-minute stall, and `run.py` was never configuring its logger, so all 16h
of stall time in run `a1npli` left zero diagnostic records. Two prior fixes
(the COSMOS_NCCL_TIMEOUT_MS/rollout-wait bump to 1h, and `e901539`) had both
targeted the wrong thing — the 1h timeout bump was actively hiding the
failure by letting each stall get absorbed instead of aborting loudly.

**Root cause:** with logging fixed and a `_StackSampler` daemon thread added
(dumps every thread's stack every 300s in both replica processes, so the hang
need not be in the reward path at all), a short canary (`eivn91`) caught it
directly: the rollout was parked in `post_rollout_completion ->
make_request_with_retry -> time.sleep(jitter)` (`client.py:570`,
`network_util.py:135`) while all 4 policy ranks spun in `grouped_send ->
nccl_group_end -> time.sleep(0.001)` (`rl_worker.py:410`, `pynccl.py:228`)
waiting on the weight-sync collective for a rollout that could never report
back. One failed localhost HTTP POST was idling all 5 GPUs. The ~51-minute
quantum is cosmos-rl's stock retry ladder running to exhaustion, not any
computation: `max_retries=60`, backoff x2 capped at 60s, jitter
`(1+random())*delay`, initial delay 1s → expected total 3172s = 52.9 min.
`a1npli`'s 14 slow steps measured 50.0-54.2 min in clean 1x/2x/3x multiples
(19 quanta total, ~16 of the run's 16h53m); `train/iteration_time` stayed
flat at 22s throughout because the stall is entirely outside the training
step. Separately, `_CosmosLogTailer`'s 80-line-per-tick cap (keeps the tail,
drops the middle) was discarding the stack-sample header on every tick —
`eivn91` dropped 4900 lines in 25 minutes — so even with the sampler running,
the evidence was getting truncated away.

**Fix:** [b29dfbc](../../commit/b29dfbc) — caches
`cosmos_rl.utils.constant.COSMOS_HTTP_RETRY_CONFIG.max_retries` down to 20
(`CODE_REWARD_HTTP_MAX_RETRIES`, ~1.9 min budget — still enough to ride out a
real transient blip), patched at `ApiClient` construction time in `main()`
before `launch_worker` builds the client. This both caps the stall and lets
`post_rollout_completion`'s own `except` (`client.py:578`) finally run and
log the real underlying error at ERROR instead of the DEBUG level
`network_util` was using per attempt. Also raises `_CosmosLogTailer`'s cap
80 → 1200 lines/tick so that log stops getting truncated. Supersedes the
obstacle-extraction attribution below: `e901539` is real and harmless (avoids
a ~0.5s-per-cold-clip zip touch) but was never the cause of the 51-minute
stalls.

**Validated** (run `vtx3ys` vs. `a1npli`, same 45-step config): wall clock
16h42m → 62.5 min; steps completed 35 → 32; steps over 20 min 14 → 0; worst
step 158 min → 6.1 min; median step ~0.5 min → 0.57 min;
`train/iteration_time` flat at 22s in both, confirming the win is entirely
the stall going away, not a training speedup.

**How this was found:** debugrun.txt notes record the pushback that started
this — profiling `_load_scene` directly against production data disproved
the obstacle-parse theory, which prompted turning `run.py`'s logger back on
and adding the stack sampler instead of guessing a fourth explanation.

---

## 2026-07-28 — code-reward ~51-min reward-time stalls: obstacle.offline chunk-zip access, now pre-extracted at setup

**Symptom:** `alpamayo-rl-code-reward-a1npli` (first clean-finish code-reward
run) completed only 35 optimizer steps in 16h53m. W&B `_runtime` deltas show
bimodal step times: ~30 s normally, but ~19 stall events quantized at ~51 min
(1x/2x/3x back-to-back: 51/~102/~158 min) consuming ~16 h of the run. Same
stall previously killed `ketkv3` outright via the stock 600 s NCCL watchdog
(bumped to 1 h for this mode as a workaround, which is why a1npli survived
but crawled).

**Root cause (localized, low-level mechanism still open):** the stall is
specific to the code reward path — llm-judge-full runs on the identical
dataset/dataloader/node do ~149 steps per 2h45m (~66 s/step, no stalls), and
the only extra I/O code mode does is `_load_scene` →
`avdi.get_clip_feature(clip_id, "obstacle.offline")`. NOTE: an earlier
"whole-chunk parquet parse" theory (echoed in old run.py comments) is
FALSIFIED — features.csv shows obstacle.offline ships as
`obstacle.offline.chunk_NNNN.zip` holding one small parquet PER CLIP
(~250 KB), so reads amplify at the zip/chunk level (multi-GB zip access at
reward time while all ranks wait on the NCCL barrier), not via pandas
parsing. Why one chunk touch costs ~51 min (storage contention from the
background 570 GB warm-cache upload? per-open cost on /mnt/work?) was not
pinned down.

**Fix:** stop touching chunk zips at reward time entirely. run.py
`_extract_obstacles_by_clip` unzips every per-clip member into
`obstacles_by_clip/` during the CPU-only setup phase (idempotent, runs
before GPUs are reserved; code mode only), and `_load_scene` reads
`obstacles_by_clip/<clip_id>.obstacle.offline.parquet` directly, falling
back to the old avdi path when the extraction is absent. Verified via
`code_as_a_reward` suite (63 passed) plus a smoke test loading the testdata
clip through the new path (41 tracks, no alp_state needed). Self-diagnosing:
if a future run still stalls ~51 min/step, scene loading is exonerated and
the cause is elsewhere (rollout/infra).

---

## 2026-07-27 (3rd) — masking-run9-d node died mid-setup: no manifest → full WDS prefix mirrored to /mnt/work

**Symptom:** `masking-run9-d-cr0skx` (third attempt, on the restored run.py)
died 25m50s in with `ray.exceptions.ActorDiedError: The actor died because
its node has died` — no Python traceback anywhere. OCI log export showed the
worker's entire lifetime was rank 0's `masking.data.s3_download` bulk-
downloading shard after shard (`shard_005_*`, `shard_006_*`, …) of the whole
`wds/train/` prefix; not one clip was ever processed.

**Root cause:** launched without `sample_clips_manifest`, so run.py's
`_acquire_shards` mirrors EVERYTHING under `s3_prefix` onto node-local
/mnt/work. The full WDS mirror is orders of magnitude larger than the node
disk, which eventually killed the node (hence a node death, not a process
error). Runs 8 a/b/c never hit this because they were launched with the
52-clip `masking/configs/sample_clips.json` manifest passed as a `-o`
override — manifest mode skips shard acquisition entirely and pulls each
clip via S3 range reads at iteration time.

**Fix:** `sample_clips_manifest: masking/configs/sample_clips.json` and
`results_s3_prefix: masking_results/run9` are now defaults in
`masking/configs/cluster.yaml` instead of launch-time overrides someone has
to remember. Relaunched as `masking-run9-d-mhycqz` with both set.

**Moral:** a config default that downloads-the-world is a footgun when the
safe behavior lives only in somebody's shell history; promote required
overrides into the yaml.

---

## 2026-07-27 (later) — masking-run9-d relaunch died iterating data: run.py was a stale pre-rewrite copy

**Symptom:** `masking-run9-d-o8x0zw` (relaunch after the requirements fix
below) failed 6m47s in — model load now succeeded, but every Ray train
worker raised `ImportError: cannot import name 'iter_snapshots' from
'masking.data.wds_dataset'` as soon as data iteration began. Confirmed via
full OCI Log Analytics export: exactly one failure mode, 4 identical
tracebacks.

**Root cause:** the same rogue checkpoint commit as the requirements bug —
`b79f73b` ("checkpoint pending changes") committed a STALE working-tree copy
of `masking/training/run.py`, reverting it to the pre-680ac17 "snapshots"
era. That wiped out: (1) the `iter_clip_events`/`iter_clip_events_from_manifest`
API usage (wds_dataset.py stopped exporting `iter_snapshots` in the 2026-07-02
rewrite — hence the crash), (2) `results_s3_prefix` + `_upload_results`
(without which the results JSONL dies with the pod — the run8-A data-loss
failure mode), (3) `sample_clips_manifest` range-read feeding, and (4) the
`delta_xy_per_waypoint` output arrays the dashboard renders. Experiment D
(`c00a088`) was then built on the regressed file, and since D had never been
launched, nothing exercised the import until today.

**Fix:** restored `run.py` wholesale from `b79f73b^` (the last good
revision) and re-applied `c00a088`'s three experiment-D additions (docstring
line, `elif experiment == "d"` dispatch, D log line) — verified the result
differs from last-good by exactly those 10 added lines. Relaunched with
`results_s3_prefix` set so D's results are durably uploaded.

**Moral:** `b79f73b` bundled unrelated stale files; treat every path it
touched as suspect. requirements.txt and run.py are confirmed casualties —
if another masking file misbehaves, diff it against `b79f73b^` first.

---

## 2026-07-27 — masking-run9-d died at model load: transformers pin silently lost from requirements.txt

**Symptom:** `masking-run9-d-dz71co` (first-ever launch of experiment D, the
commitment/perceptual reversal) went `EXPERIMENT_FAILED` 5 minutes in. Every
rank raised the same error at `_load_model`:
`ImportError: cannot import name 'Qwen3VLConfig' from 'transformers'
(/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)` — i.e.
transformers resolved from the cluster image's old system copy, which predates
Qwen3-VL support (added in 4.57).

**Root cause:** commit `b79f73b` ("checkpoint pending changes") replaced
`masking/requirements.txt` with a scratch-notes version, silently dropping
most of the working pin set from `f47a187`: `transformers==4.57.1`,
`accelerate`, `hydra-core`, `s3transfer`, `boto3`, `pyarrow`, `scipy`, `av`,
and the `torch==2.7.1` pin. The same-day fix `3b4d219` restored only the
torch pin (the one line the launch validator checks) and commented out the
prose. The `--dry-run` preflight passed because uv only resolves the packages
*listed* — it cannot know the workload imports transformers. Transformers
merely failed first; hydra-core/accelerate/av/scipy would have failed next.

**Fix:** restored the full `f47a187` dependency set (merged with the current
huggingface_hub/pandas floors) and moved the scratch notes out of the file
(they remain in git history at `b79f73b`).

**Moral:** a requirements.txt that passes `uv` resolution proves nothing
about import-time completeness; when a "checkpoint pending changes" commit
touches a requirements file, diff it against the last known-good run's
revision before launching.

---

## 2026-07-24 (later) — `ignore_idle_reaper: true` didn't fix the mid-training reaper kills; it only hid them

**Symptom:** After the fix below shipped (commit `3f24da6`), both relaunched
attempts — `alpamayo-rl-code-reward-23i8uk` and
`alpamayo-rl-llm-judge-full-f6pj1y` — sat at `EXPERIMENT_RUNNING` in
`lilypad workload list` for 10+ hours, which looked healthy. It wasn't:
`research/alpamayo-rl`'s W&B runs (queried directly, since Ray Application
Log stdout only flushes at teardown) showed `llm_judge_full` crashing and
restarting from a **new** run ID (step 0, `resume=false`) every ~2h45m
throughout the whole window (`...193143` crashed@step149, `...221727`
crashed@step148, `...011821` crashed@step148, `...040343` running); `code_reward`
hadn't produced a single new W&B run in over 10h despite the workload
supposedly running.

**Root cause:** `ignore_idle_reaper: true` does not do what the 2026-07-24
(earlier) entry below assumed. Direct log inspection around a crash boundary
(llm-judge-full, 21:47:26 UTC) showed the exact same signature as before the
flag: worker pod `...-6zq9q` logs `Received graceful stop` alone (head pod
untouched), and a **new** worker pod (`...-dkn6q`) appears minutes later
running the full setup sequence (apt/venv/checkpoint-convert) from scratch.
The flag only stopped Lilypad from flipping the *workload* to
`EXPERIMENT_FAILED` — the reaper still evicts the lone idle worker pod, Ray
now auto-relaunches inside the same workload, and training silently restarts
from step 0 forever, burning 8xA100-hours with zero net progress while
`workload list`/`workload info` report `EXPERIMENT_RUNNING` the entire time.
Confirmed by reading the installed `lilypad_py==2.27.0` SDK source directly
(`~/.local/lib/python3.10/site-packages/lilypad/public/schemas/workload_config.py:693`):
the field's own docstring is `"Opt out of the idle GPU reaper **alert**"` —
it silences a notification, not the termination action.

**Fix:** Do what the earlier entry explicitly declined to do (extend
`_GpuKeepalive` into the reward path) — that uncertainty is resolved now
that `ignore_idle_reaper` is confirmed not to work. `run.py`'s
`_run_on_gpu_node` no longer stops the keepalive thread before launching
`cosmos-rl` for `reward_mode in ("llm_judge", "code")`; it keeps nudging all
GPUs every 5s for the whole subprocess lifetime (`try`/`finally` around
`_launch_cosmos_rl`), only stopping once cosmos-rl exits. Kept
`ignore_idle_reaper: true` in the cluster configs too (harmless, still
silences the alert noise) but it is not load-bearing for correctness anymore.
`reasoning`/motion modes are untouched (stop keepalive before launch as
before) since they aren't reward-latency-bound and don't need the extra
GPU contention risk.

**Still open:** `alpamayo-rl-code-reward-23i8uk` and
`alpamayo-rl-llm-judge-full-f6pj1y` are still running the OLD code
(pre-this-fix) and will keep crash-looping — they need to be stopped and
relaunched on the new master once this commit lands. See
[[project_code_as_reward_not_started]] / [[project_llm_judge_full_run]].

**How this was found:** `lilypad workload list`/`workload info` alone were
misleading (both showed healthy `EXPERIMENT_RUNNING`); the real signal was
querying `research/alpamayo-rl` directly via the `wandb` Python API
(`api.runs(...)`, ordered by `-created_at`) for run `state`/`_step`/
`created_at`, which exposed the crash-restart cadence, then reading
`lilypad workload logs --start-time/--end-time` around one exact crash
boundary and the `lilypad_py` SDK source itself for the flag's real
semantics.

---

## 2026-07-24 — llm-judge-full eigzof: idle-GPU reaper struck mid-training, not just setup

**Symptom:** `alpamayo-rl-llm-judge-full-eigzof` went `EXPERIMENT_FAILED` after
11h1m (`preemptible: never`, confirmed via `workload info`). The worker pod
was replaced four times at an almost perfectly regular ~2h40m cadence
(`cjrws` 05:40-08:16, `srfrq` 08:27-11:08, `v8rz5` 11:10-13:52, `r5kzx`
13:55-16:35), the last incarnation raising `ray.exceptions.NodeDiedError`
and failing the job. Zero step/reward/CUDA content ever reached Ray
Application Logs across the whole 11h.

**Root cause:** each transition showed the worker pod's fluentd sidecar log
`Received graceful stop` alone (head pod unaffected, still logging
afterward) — the exact lone-worker signature the 2026-07-23 entry below
attributes to external preemption, distinct from a manual `workload stop`
(which stops head+worker simultaneously, see that entry's `12qhs5`
sibling). But this job ran `preemptible: never`, ruling out cloud
preemption — leaving Lilypad's idle-GPU reaper (already confirmed to
strike twice during setup, see the 2026-07-22 entry below) as the only
mechanism that kills a lone worker pod regardless of the preemptible flag.
`_GpuKeepalive` in `run.py` only nudges the GPUs during the CPU-only setup
phase and explicitly `.stop()`s right before `cosmos-rl` launches
(`run.py:872`) — it has no reach once training starts. `llm_judge`'s
reward is Anthropic-API-latency-bound (`aggregated_reward_llm_judge.py`'s
`_run_judges_parallel`), which can leave the whole node's GPUs
idle-but-reserved for long stretches during a rollout group's judge calls
— the same shape as the setup-phase problem, just relocated to training.

**Fix:** `ignore_idle_reaper: true` added to `llm_judge_full_cluster.yaml`,
`llm_judge_cluster.yaml`, and (preemptively, same rationale as the
NCCL-timeout extension in the 2026-07-23 entry below) `code_reward_cluster.yaml`
— the top-level config flag `perplexity/configs/cluster.yaml` already uses
for its own CPU-only-setup idle-reaper problem. Simpler and more robust
than extending `_GpuKeepalive` into the reward function: unclear whether
the reaper watches per-GPU or per-node utilization, and other ranks sit
idle on an NCCL barrier during the same stall regardless of which rank is
making the API call — opting the whole job out sidesteps needing to answer
that.

**How this was found:** `lilypad workload info --show-diff` confirmed
`Preemptible: No` was live at launch (ruling out a stale config); per-
transition-window `lilypad workload logs --start-time/--end-time` around
each worker-pod handoff (not just the failure tail) surfaced the
`Received graceful stop` line on the worker pod alone, matching the
signature already documented for q00jjc/12qhs5.

---

## 2026-07-23 — llm-judge-full q00jjc could never finish: preemption cadence (~2h15m) shorter than run length, no training resume

**Symptom:** `alpamayo-rl-llm-judge-full-q00jjc` went `EXPERIMENT_FAILED` after
14h48m. W&B shows 6 crashed `reasoning_vla_llm_judge_full` attempts inside the
one workload, every one dying at almost exactly `_runtime` ≈ 8100 s at step
148-149 of 264 — constant *time* of death with slightly varying *step*, across
different per-step speeds (34-46 s/it) and different nodes.

**Root cause:** external preemption, not a code bug. The job driver logged
`ray.exceptions.NodeDiedError` for a different worker node each cycle, and the
worker pod's fluentd sidecar logged `Received graceful stop` (Kubernetes
SIGTERM) at the exact crash instants (confirmed 08:56 and 21:02 UTC crashes in
OCI Log Analytics). The config ran `preemptible: always` +
`requeue_if_preempted: true`, betting the S3 warm cache made requeues cheap —
but cosmos-rl runs with `train.resume: false` and a freshly timestamped
`output_dir` per attempt, so its every-50-step checkpoints are never resumed
and each requeue restarts training at step 0 after ~35 min of env rebuild.
With the us-chicago-1 A100 pool preempting this worker every ~2h15m and a full
run needing ~2.5-3h, the run mathematically could not complete; requeueing
just burned 8 GPUs for 15 hours.

**Fix:** [314a11a](../../commit/314a11a), [4efba47](../../commit/4efba47) —
`preemptible: never` in `llm_judge_full_cluster.yaml` and
`code_reward_cluster.yaml` (waits for non-preemptible quota instead of
being fed to the preemption cycle). Follow-up worth doing if long runs grow
past quota patience: wire cosmos-rl checkpoint resume across requeues (stable
`output_dir` + `train.resume: true`), which would make `preemptible: always`
viable again.

**How this was found:** W&B run list showed every recent run `crashed` with
near-identical `_runtime`; `lilypad workload logs` only carries fluentd/env
noise, so the tracebacks were pulled from OCI Log Analytics (`Ray Application
Logs` source) around each crash timestamp. The 08:56 crash bonus-logged
"Raylet is terminated ... SIGKILL by the user or system OOM killer" and the
sidecar's simultaneous graceful stop pinned it as pod termination rather than
in-process death. (Sibling evidence: `alpamayo-rl-code-reward-12qhs5` trained
healthily on current master until head+worker both got graceful stops at
19:20 UTC simultaneously = deliberate `workload stop`, status
`EXPERIMENT_STOPPED` — distinct signature from preemption, which kills only
the worker.)

---

## 2026-07-23 — code-reward canary died on NCCL watchdog: same failure as llm_judge, fix never extended to `code` mode

**Symptom:** `alpamayo-rl-code-reward-ketkv3` went `EXPERIMENT_FAILED` at 26
min. Default `lilypad workload logs` window (last 4h) missed the run entirely
(it ran hours earlier); `--start-time/--end-time` around `workload info`'s
Created/Finished timestamps, plus `--content-filter ERROR`, surfaced the real
traceback out of thousands of non-error lines: `[rank0]`/`[rank1]`/`[rank3]`
all raised `TimeoutError: NCCL: non-blocking enqueue timed out`, then
`[cosmos] ERROR - Process 1 failed with return code 1`.

**Root cause:** identical fingerprint to `alpamayo-rl-llm-judge-canary-u0j67p`
(2026-07-22): cosmos-rl's NCCL watchdog (`COSMOS_NCCL_TIMEOUT_MS`, default
600000 ms) aborts any communicator whose pending collective runs past 10 min.
That entry's fix bumped the timeout to 1h, but scoped it to
`reward_mode == "llm_judge"` only (`run.py`), reasoning that only the judge's
Anthropic API latency could stall a rollout group that long. `code_reward_entry.py`
shares the exact same TOML (`group_reward_calculation = true`) by design, and
while its own docstring calls the reward math itself "cheap, not
load-bearing," `_load_scene()` is an LRU-cached parse of each clip's
`obstacle.offline` chunk that its own docstring flags as "the expensive
part" — on a fresh canary every clip in the first several rollout groups is
a cold cache miss, which can stall the collective past 600s the same way the
judge's API calls did. `code` mode never got the timeout bump.

**Fix:** extended the `COSMOS_NCCL_TIMEOUT_MS`/`COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT`
bump in `run.py`'s `_run_on_gpu_node` to `reward_mode in ("llm_judge", "code")`,
separating it from the `ANTHROPIC_API_KEY` check (which stays llm_judge-only).
Relaunched as `alpamayo-rl-code-reward-ketkv3`'s successor.

---

## 2026-07-23 — llm-judge full run died 3h in on ONE truncated judge response

**Symptom:** `alpamayo-rl-llm-judge-full-mhebtx` went `EXPERIMENT_FAILED` at
3h 1m (after surviving a node preemption + rebuild): rank0 raised
`JudgeRewardError: invalid judgment after retry: Unterminated string
starting at: line 1 column 55 (char 54)` and the whole cosmos-rl job tore
down.

**Root cause:** the judge's JSON response was cut off immediately after
`"one_line_rationale": "` (column 55 is that exact position). The score
integer earlier in the object was complete — but `_parse_single_judgment`
does a strict `json.loads`, the content-retry budget was a single fresh
call, and the same truncation shape repeating twice hit the fail-loud
raise. At ~14k judge calls per full run, a rare per-call failure shape is a
per-run certainty.

**Fix:** `8e20390` — `_salvage_score` recovers the (complete, unambiguous)
score integer from truncated response text before burning a retry — it's
the judge's actual emitted score, only the log-only rationale is lost, so
the no-placeholder-rewards policy holds; `stop_reason == "max_tokens"`
doubles the token budget on retry instead of re-rolling; content retries
1 → 3. Fail-loud raise unchanged after that.

---

## 2026-07-23 — code-reward canary crashed at controller start: `KeyError: 'COSMOS_CONFIG'`

**Symptom:** `alpamayo-rl-code-reward-b2wwha` (first code-as-a-reward
canary) went `EXPERIMENT_FAILED` 15 min in; every cosmos-rl process died
immediately with `KeyError: 'COSMOS_CONFIG'` in
`code_reward_entry._read_ckpt_path_from_toml`.

**Root cause:** the entry's ckpt-path helper was copied from **run.py's**
`_read_ckpt_path_from_toml` (head-node convention: `COSMOS_CONFIG` env
var) instead of the **vendored launcher's** same-named helper. cosmos-rl
invokes entry scripts as `python entry.py --port ... --config
/tmp/<patched>.toml` and does not export `COSMOS_CONFIG` — the config
path only exists in argv. Two same-named helpers with different
contracts; the wrong one was mirrored.

**Fix:** `f06ecc0` — parse `--config <path>` from `sys.argv` (the vendored
launcher's behavior), keep the env var only as a fallback for head-node
style invocation.

---

## 2026-07-22 — full OOD run SIGINT'd at 61 min: idle-GPU reaper vs. GPU-free setup phase

**Symptom:** `alpamayo-rl-llm-judge-full-lmhb35` went `EXPERIMENT_STOPPED` at
1h 1m with zero application errors — "terminated gracefully with SIGINT" at
22:48:30 while the ~570 GB dataset download (started 21:54) was still
running. No W&B run was ever created; training never started.

**Root cause:** the entire setup phase — venv build, model conversion,
`snapshot_download` — is CPU/network-only, and Lilypad's idle-GPU reaper
watches GPU *utilization*, not ray reservations (`num_gpus=8` doesn't
count). Second confirmed strike: canary `xgo36t` (2026-07-21) was killed the
same way after a replica died and the survivors idled 96 min. Run `5ieeuh`
survived only because its node downloaded fast enough (~21 min) to reach
vLLM before the threshold; `lmhb35`'s node was slower and crossed ~60 min
idle.

**Fix:** `2e6f26f` — `_GpuKeepalive` daemon thread in run.py runs a tiny
matmul burst (1024x1024, ~4 MB) on every visible GPU every 5 s from task
start until just before cosmos-rl launches, then frees memory and empties
the CUDA cache. Relaunched as `alpamayo-rl-llm-judge-full-mhebtx`.

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

## 2026-08-11 — a1 (Alpamayo R1) model load rejects attn_implementation="sdpa"

**Symptom:** masking-run10-a1-a-gp3mxp died in `_load_model` with
`ValueError: MaskedAlpamayoR1 does not support an attention implementation
through torch.nn.functional.scaled_dot_product_attention yet` (transformers
4.57 `_sdpa_can_dispatch` check inside `from_pretrained`).

**Root cause:** NVlabs/alpamayo (R1) predates the per-class attention-support
flags; its `ReasoningVLA` never sets `_supports_sdpa`, so transformers'
init-time capability check refuses the sdpa request. The 1.5 repo's otherwise
near-identical `base_model.py` sets `_supports_sdpa = True` /
`_supports_flash_attn = True`, which is why the same load path always worked
for MaskedAlpamayo1_5.

**Fix:** set `_supports_sdpa = True` and `_supports_flash_attn = True` on the
`MaskedAlpamayoR1` fork class (masking/masked_model_a1.py) — same Qwen3-VL
family stack 1.5 declares SDPA-safe, and sdpa is the implementation the 1.5
masking results were produced with.

**How this was found:** first a1 smoke run after the a1 port; traceback in
workload logs, then `grep _supports_sdpa` across both vendored repos showed
the 1.5-only flags.

---

## Format for new entries

```
## YYYY-MM-DD — one-line symptom

**Symptom:** what was observed (error text, log line, incorrect behavior).
**Root cause:** why it actually happened — the non-obvious mechanism.
**Fix:** [commit-sha](../../commit/sha) — what changed and why that's correct.
**How this was found:** (optional but valuable) what investigation surfaced it.
```
