# sample_scenario_clips.py -- sample 15 clips from EACH of the 9 event_cluster
# scenario types in reasoning/ood_reasoning.parquet's TRAIN split, resolve
# each to its real S3 WDS shard location, extract the files an inference run
# actually needs, and write a manifest. This is prep for a Lilypad cluster
# job that will run AlpamayoR1 inference on the resulting (up to) 135 clips.
#
# Reuses, doesn't reimplement:
#   - masking/data/sample_clips.py's scan_shard_for_clips() -- the actual
#     technically-hard part (walking one shard's tar headers via HTTP Range
#     reads to find OOD clips without downloading the shard). We do NOT call
#     that module's higher-level find_available_ood_clips() wrapper, though:
#     benchmarked live against this bucket, a single shard's header walk
#     costs ~180-190s serialized (500-600 tiny 512B range-GETs, one per tar
#     member -- there's no way to skip a member without first reading its
#     header to learn its size, so this cost is inherent to the tar format,
#     not a scan_shard_for_clips inefficiency). Some of these 9 clusters are
#     rare enough (as low as ~18-22 total rows dataset-wide) that finding 15
#     of them requires scanning a large fraction of the ~4200 shards under
#     this prefix -- and find_available_ood_clips() only returns once
#     either every cluster is satisfied or literally every shard has been
#     scanned, with no intermediate checkpointing. At the sizes involved
#     here that could be a multi-hour blocking call with nothing to show if
#     it needs to be interrupted. The loop in main() below is a thin
#     orchestration replacement -- same ThreadPoolExecutor-over-shards
#     structure, same scan_shard_for_clips() calls -- that additionally
#     writes the manifest incrementally as each new accepted clip is found,
#     so a partial run is never a wasted one.
#   - the tar-walk-from-a-known-group-start-offset technique already solved
#     in pref_pairs/render_trajectory_overlay.py's fetch_clip_files() and
#     masking/data/s3_clip_extract.py's extract_clip_members() -- both do
#     exactly this "walk forward from group_start_offset until the clip_id
#     prefix changes, range-GET only the members you want" pattern. Neither
#     is imported directly (one writes into an in-memory dict shaped for a
#     different caller, the other only fetches 1 camera + calibration), but
#     extract_clip_files() below is the same walk, just parameterized for
#     the 4 camera keys + egomotion s3_clip_loader.py needs, streaming each
#     member straight to disk instead of buffering all of them in memory.
#
# t0_us selection: same convention as sample_ood_clips.py -- the clip's
# FIRST event (from ood_reasoning.parquet's own "events" column, not the
# WDS shard's per-clip json blob) with event_start_timestamp > MIN_T0_US.
# Clips whose events never clear that bar are skipped and a different
# candidate is drawn instead, same as sample_ood_clips.py's redraw-on-skip
# loop.
#
# Per-cluster target is 15. Two of the 9 clusters (ANIMALS_BIRDS_ROADKILL,
# ROAD_DEBRIS_OR_SAFETY_TRACES) have only 22/18 total rows in the train
# split to begin with, so 15 is only reachable if nearly all of them are
# already uploaded to S3 -- if a cluster's real S3-available (and
# t0-valid) candidate pool is smaller than 15, we take whatever IS
# available and print the shortfall rather than padding or blocking.

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import time

import boto3
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# masking/data/sample_clips.py lives in a sibling top-level package, not
# under perplexity/ -- other perplexity/*.py scripts (sample_ood_clips.py,
# score.py, ...) use flat imports and are run with perplexity/ itself as the
# working directory / sys.path root, so the repo root isn't on sys.path by
# default here. Add it explicitly rather than moving/duplicating
# scan_shard_for_clips/list_shards.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from masking.data.sample_clips import list_shards, scan_shard_for_clips  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
BUCKET = "research-datasets-chicago"
PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train"
S3_ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com"
MIN_T0_US = 1_700_000  # same buffer sample_ood_clips.py uses ahead of the history window
N_PER_CLUSTER = 15
TAR_BLOCK = 512

# find_available_ood_clips's default max_workers=40 clocked a clean,
# SUSTAINED 0.148 shards/sec against this bucket (measured live, two
# back-to-back 80-shard batches with no degradation over ~540s). Higher
# concurrency was tried and reverted: a 100-worker *short* batch looked
# promising (1-2 shards/sec after an initial ~340s connection-warmup
# window), but a 150-worker run measured over a longer, sustained window
# collapsed to ~2 shards completed in 693s -- i.e. more concurrency
# eventually triggers something (almost certainly server-side throttling
# that a brief burst test doesn't surface) that makes things much WORSE,
# not better. 40 is the one concurrency level actually confirmed stable
# over a sustained run -- not raising it further without new evidence.
SCAN_MAX_WORKERS = 40

# How often (in shards completed) and how long (in seconds) between
# manifest checkpoint writes -- see module docstring: some clusters may
# require scanning most/all of the ~4200 shards to reach n_per_cluster, so
# this can be a long-running job. Checkpointing means a kill/interrupt at
# any point still leaves a valid, up-to-date manifest on disk, not just
# whatever was true at the last full scan completion.
CHECKPOINT_EVERY_SHARDS = 20
CHECKPOINT_EVERY_SECONDS = 45.0

# Hard wall-clock cap on the scan phase. If some clusters still haven't
# reached n_per_cluster when this elapses, we stop and report exactly what
# was found in the shards actually scanned by then -- a real, if partial,
# result instead of an open-ended multi-hour block. See main()'s
# per-cluster SHORTFALL logging, which distinguishes this case (scan
# stopped early, more may exist unscanned) from a confirmed-exhausted one
# (full scan completed and this is genuinely everything currently in S3).
TIME_BUDGET_SECONDS = 3 * 3600

# Matches s3_clip_loader.py's load_clip_from_s3_extract() default camera_keys
# exactly -- that's the only consumer of these extracted files, so the set
# fetched here must match it 1:1. calibration.json is deliberately NOT
# fetched: s3_clip_loader.py never reads it.
CAMERA_KEYS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)

_HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(_HERE, "clip_cache")
MANIFEST_PATH = os.path.join(_HERE, "configs", "scenario_sample_manifest.json")


def load_train_ood_df() -> pd.DataFrame:
    """Download reasoning/ood_reasoning.parquet and restrict to the train
    split (indexed by clip_id, one row per OOD clip, event_cluster column
    has exactly the 9 target scenario types)."""
    path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="reasoning/ood_reasoning.parquet")
    df = pd.read_parquet(path)
    return df[df["split"] == "train"]


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def extract_clip_files(s3, bucket: str, shard_key: str, group_start_offset: int, clip_id: str, clip_dir: str) -> None:
    """Walk one shard's tar headers starting at the clip's known
    group_start_offset (as returned by find_available_ood_clips's
    scan_shard_for_clips) and range-GET just egomotion.parquet + the 4
    camera mp4s, writing each straight to `{clip_dir}/{clip_id}.<suffix>` --
    the exact filenames s3_clip_loader.load_clip_from_s3_extract expects.

    Same "walk forward until the clip_id prefix changes" technique as
    pref_pairs/render_trajectory_overlay.py's fetch_clip_files() /
    masking/data/s3_clip_extract.py's extract_clip_members(), just streaming
    to disk instead of returning an in-memory bytes dict (135 clips x 4
    videos held in memory at once isn't worth it when we're about to write
    them to disk anyway).
    """
    wanted = {f"{clip_id}.egomotion.parquet": "egomotion.parquet"}
    for cam in CAMERA_KEYS:
        wanted[f"{clip_id}.{cam}.mp4"] = f"{cam}.mp4"

    pos = group_start_offset
    zero_blocks = 0
    while wanted:
        hdr = _range_get(s3, bucket, shard_key, pos, TAR_BLOCK)
        if len(hdr) < TAR_BLOCK or hdr == b"\x00" * TAR_BLOCK:
            zero_blocks += 1
            pos += TAR_BLOCK
            if zero_blocks >= 2:
                break
            continue
        zero_blocks = 0
        name = hdr[0:100].split(b"\x00")[0].decode(errors="replace")
        size_field = hdr[124:136].split(b"\x00")[0].strip()
        size = int(size_field, 8) if size_field else 0
        data_start = pos + TAR_BLOCK
        is_pax = "@PaxHeader" in name

        if name in wanted:
            suffix = wanted.pop(name)
            data = _range_get(s3, bucket, shard_key, data_start, size)
            with open(os.path.join(clip_dir, f"{clip_id}.{suffix}"), "wb") as f:
                f.write(data)
        elif not is_pax and not name.startswith(clip_id):
            break  # walked past this clip's own member group into the next clip's

        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK

    if wanted:
        raise RuntimeError(
            f"clip {clip_id}: could not find {sorted(wanted.values())} in "
            f"{shard_key} starting at offset {group_start_offset}"
        )


def pick_t0_and_coc(ood_df: pd.DataFrame, clip_id: str) -> tuple[int, str] | None:
    """First event (from ood_reasoning.parquet's own events column) with
    event_start_timestamp > MIN_T0_US -- same convention sample_ood_clips.py
    uses. Returns None if the clip has no events at all, or none clear the
    threshold, so the caller can skip and draw a different candidate."""
    row = ood_df.loc[clip_id]
    events = row["events"]
    # Missing "events" shows up as various not-really-a-value markers
    # depending on pandas/pyarrow's mood (float nan, None, or pandas' own
    # pd.NA singleton for nullable/pyarrow-backed columns -- confirmed live:
    # a bare `isinstance(events, float) and pd.isna(events)` check missed
    # the pd.NA case and crashed with "TypeError: 'NAType' object is not
    # iterable" on the very first real clip that had one). The only two
    # valid non-missing shapes are a JSON string or an already-parsed list,
    # so checking for NOT those is more robust than trying to enumerate
    # every possible missing-value sentinel.
    if not isinstance(events, (str, list)):
        return None  # a small number of OOD rows have no events (known upstream gap)
    events = json.loads(events) if isinstance(events, str) else events
    valid = [e for e in events if e["event_start_timestamp"] > MIN_T0_US]
    if not valid:
        return None
    return int(valid[0]["event_start_timestamp"]), valid[0]["coc"]


def write_manifest(manifest: list[dict]) -> None:
    """Overwrite the manifest file with the current accepted-clip list.
    Called repeatedly (see main()'s checkpoint cadence) so the file on disk
    is never more than CHECKPOINT_EVERY_SECONDS/SHARDS stale -- if this
    script gets killed mid-scan, whatever's on disk is still a valid,
    directly-usable manifest of whatever was found so far."""
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)  # atomic: never leaves a half-written manifest


def main(
    n_per_cluster: int = N_PER_CLUSTER,
    seed: int = 42,
    max_workers: int = SCAN_MAX_WORKERS,
    time_budget_s: float = TIME_BUDGET_SECONDS,
) -> None:
    os.makedirs(CLIP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    # scan_shard_for_clips/list_shards reach for a bare boto3.client("s3")
    # with no explicit profile/endpoint, relying on env vars for both (see
    # pref_pairs/render_trajectory_overlay.py's identical fallback) -- set
    # them here rather than editing that shared module.
    os.environ.setdefault("AWS_PROFILE", "oci.chi")
    os.environ.setdefault("AWS_ENDPOINT_URL_S3", S3_ENDPOINT)

    ood_df = load_train_ood_df()
    all_clusters = sorted(ood_df["event_cluster"].unique())
    ood_ids = set(ood_df.index.astype(str))
    cluster_of = ood_df["event_cluster"].to_dict()
    logger.info("Loaded %d train-split OOD clips across %d clusters", len(ood_df), len(all_clusters))

    # Own S3 client (explicit profile/endpoint, matching this session's
    # already-verified credentials) for the extraction step below.
    session = boto3.Session(profile_name="oci.chi")
    s3 = session.client("s3", endpoint_url=S3_ENDPOINT)

    keys = list_shards(BUCKET, PREFIX)
    # Shuffle (seeded) before submitting -- list_shards returns shards in
    # lexicographic key order, which is very likely correlated with
    # collection time/session/rank. Scanning in that order risks the
    # early-found candidates for any cluster skewing toward whatever
    # happens to collect first, rather than a representative sample --
    # same justification masking/data/sample_clips.py's own
    # find_first_n_ood_clips() uses for shuffling before its scan.
    keys = list(keys)
    np.random.default_rng(seed).shuffle(keys)
    logger.info(
        "Scanning up to %d shards (max_workers=%d, time_budget=%.0fs) for %d candidates per cluster across %d clusters",
        len(keys), max_workers, time_budget_s, n_per_cluster, len(all_clusters),
    )

    # Resume from a previous (partial or interrupted) run's checkpoint if
    # one is already on disk, rather than discarding real, already-paid-for
    # scan results -- given how long a full scan can take (see module
    # docstring), restarting from zero on every crash/interrupt would be
    # wasteful. Shard scan order is reproduced identically on a rerun with
    # the same seed, so already-accepted clips will simply be seen again
    # and skipped quickly (already in seen_clip_ids) rather than
    # re-extracted.
    manifest: list[dict] = []
    accepted_counts: dict[str, int] = {c: 0 for c in all_clusters}
    seen_clip_ids: set[str] = set()  # guards against the same clip surfacing twice (shouldn't happen, but cheap to guard)
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        for entry in manifest:
            accepted_counts[entry["event_cluster"]] = accepted_counts.get(entry["event_cluster"], 0) + 1
            seen_clip_ids.add(entry["clip_id"])
        logger.info("Resumed %d previously-accepted clips from %s: %s", len(manifest), MANIFEST_PATH, accepted_counts)

    start_t = time.time()
    last_checkpoint_t = start_t
    shards_done = 0
    shards_failed = 0
    stopped_reason = "all shards scanned"

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(scan_shard_for_clips, BUCKET, k, ood_ids): k for k in keys}
    try:
        for fut in concurrent.futures.as_completed(futures):
            shards_done += 1
            shard_key = futures[fut]
            try:
                results = fut.result()
            except Exception:
                logger.exception("shard scan failed for %s -- skipping (not retried)", shard_key)
                shards_failed += 1
                results = []

            for clip_id, found_shard_key, _meta, offset in results:
                cluster = cluster_of.get(clip_id)
                if cluster is None or accepted_counts[cluster] >= n_per_cluster:
                    continue  # not a tracked cluster, or that cluster is already full
                if clip_id in seen_clip_ids:
                    continue
                seen_clip_ids.add(clip_id)

                t0_and_coc = pick_t0_and_coc(ood_df, clip_id)
                if t0_and_coc is None:
                    continue  # no event clears MIN_T0_US -- draw a different candidate (i.e. just skip)
                t0_us, coc = t0_and_coc

                try:
                    extract_clip_files(s3, BUCKET, found_shard_key, offset, clip_id, CLIP_DIR)
                except Exception:
                    logger.exception("failed to extract %s from %s@%d -- skipping", clip_id, found_shard_key, offset)
                    continue

                manifest.append({
                    "clip_id": clip_id,
                    "event_cluster": cluster,
                    "t0_us": t0_us,
                    "coc": coc,
                    "clip_dir": CLIP_DIR,
                })
                accepted_counts[cluster] += 1
                logger.info(
                    "accepted %s (%s) -- cluster now %d/%d [%d/%d shards scanned]",
                    clip_id, cluster, accepted_counts[cluster], n_per_cluster, shards_done, len(keys),
                )

            all_satisfied = all(v >= n_per_cluster for v in accepted_counts.values())
            now = time.time()
            if (
                shards_done % CHECKPOINT_EVERY_SHARDS == 0
                or now - last_checkpoint_t >= CHECKPOINT_EVERY_SECONDS
                or all_satisfied
            ):
                write_manifest(manifest)
                last_checkpoint_t = now
                logger.info(
                    "checkpoint: %d/%d shards scanned (%d failed), %.0fs elapsed, counts=%s",
                    shards_done, len(keys), shards_failed, now - start_t, accepted_counts,
                )

            if all_satisfied:
                stopped_reason = "all clusters satisfied"
                break
            if now - start_t > time_budget_s:
                stopped_reason = f"time budget ({time_budget_s:.0f}s) exceeded"
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    write_manifest(manifest)
    fully_scanned = stopped_reason in ("all clusters satisfied", "all shards scanned")
    logger.info(
        "STOPPED: %s. Scanned %d/%d shards (%d failed). Wrote %d/%d clips to %s",
        stopped_reason, shards_done, len(keys), shards_failed, len(manifest), n_per_cluster * len(all_clusters), MANIFEST_PATH,
    )
    for cluster in all_clusters:
        n = accepted_counts[cluster]
        if n < n_per_cluster:
            coverage_note = (
                "confirmed exhausted -- full shard scan completed" if fully_scanned
                else f"scan stopped early ({shards_done}/{len(keys)} shards) -- more may exist unscanned"
            )
            logger.warning("SHORTFALL %s: %d/%d (%s)", cluster, n, n_per_cluster, coverage_note)
    if all(v >= n_per_cluster for v in accepted_counts.values()):
        logger.info("All %d clusters reached the full %d-clip target.", len(all_clusters), n_per_cluster)


if __name__ == "__main__":
    main()
