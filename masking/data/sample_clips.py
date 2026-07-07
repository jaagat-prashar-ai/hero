# SPDX-License-Identifier: Apache-2.0
"""
sample_clips.py — pick N clips per OOD scenario type (event_cluster) for the
masking analysis, restricted to clips whose shard has already been uploaded
by the build_wds job (which is still running and has not covered the full
dataset yet).

Finding "which OOD clips are already in S3" without downloading every shard
(each ~1-12GB) is done via HTTP Range reads: we walk each shard's tar headers
directly (512-byte blocks), reading only the small `{clip_id}.json` payloads
and skipping over the multi-MB camera/parquet payloads by jumping straight to
the next header using the size field. This costs a handful of small requests
per clip instead of downloading ~3.5GB/shard.

Before scanning shards at all, candidate clips are filtered against
metadata/feature_presence.parquet (one boolean column per named sensor
stream -- verified directly against the dataset: 306,152 rows x 36 columns,
indexed by clip_id) to require the 4 camera views AND egomotion that
_expand_clip_to_events (masking/data/wds_dataset.py) actually needs. Without
this, a clip missing e.g. one camera view could still get selected into the
manifest here, only to have _expand_clip_to_events silently return []  for
it at harvest time -- quietly burning a slot out of the requested sample
size for a clip that could never have produced any scenes to begin with.

Usage:
    python -m masking.data.sample_clips \
        --bucket research-datasets-chicago \
        --prefix nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train \
        --hf_token $HF_TOKEN \
        --n_per_type 10 \
        --out masking/configs/sample_clips.json

    # --all: every currently-uploaded OOD clip (still feature_presence-filtered),
    # not a balanced per-category sample -- e.g. for pref_pairs/training/run.py's
    # Lilypad workload, which wants the full OOD set rather than a curated subset.
    python -m masking.data.sample_clips \
        --bucket research-datasets-chicago \
        --prefix nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train \
        --hf_token $HF_TOKEN \
        --all \
        --out pref_pairs/configs/sample_clips_all.json

    # --clusters: restrict sampling/scanning to specific event_cluster names
    # (comma-separated) -- e.g. pref_pairs' epsilon-calibration experiment,
    # which only wants a fixed 100-per-cluster sample across a known set of
    # scenario types, not every category in the OOD taxonomy.
    python -m masking.data.sample_clips \
        --hf_token $HF_TOKEN \
        --n_per_type 100 \
        --clusters PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY,WORK_ZONES_TEMP_TRAFFIC_CONTROL \
        --out pref_pairs/configs/sample_clips_action_variance.json

    # --n_total: unstratified sample of N total clips, ANY category, no
    # per-category balance -- stops as soon as N are found instead of
    # requiring every (or every requested) category to individually satisfy
    # a quota. Useful when some target categories are too rare to ever
    # reach n_per_type and waiting for a full scan isn't worth it.
    python -m masking.data.sample_clips \
        --hf_token $HF_TOKEN \
        --n_total 100 \
        --out pref_pairs/configs/sample_clips_n100_unstratified.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
TAR_BLOCK = 512

# Matches masking/data/wds_dataset.py's CAMERA_FEATURES (the 4 camera views
# _expand_clip_to_events actually decodes) plus "egomotion" (that function
# also hard-requires egomotion.parquet to build ego_history/ego_future --
# see its `if ego_bytes is None: ... return []` check). Verified column
# names directly against metadata/feature_presence.parquet (36 boolean
# columns, indexed by clip_id).
REQUIRED_FEATURES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "egomotion",
]


def load_ood_clips(hf_token: str | None) -> pd.DataFrame:
    """Download reasoning/ood_reasoning.parquet — one row per OOD clip."""
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename="reasoning/ood_reasoning.parquet",
        token=hf_token,
    )
    return pd.read_parquet(path)


def load_feature_presence(hf_token: str | None) -> pd.DataFrame:
    """Download metadata/feature_presence.parquet — one boolean row per clip,
    one column per named sensor/data stream (cameras, lidar, radar,
    egomotion, calibration, ...). Indexed by clip_id."""
    path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename="metadata/feature_presence.parquet",
        token=hf_token,
    )
    return pd.read_parquet(path)


def filter_clips_with_required_features(
    ood_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    required_features: list[str] = REQUIRED_FEATURES,
) -> pd.DataFrame:
    """Drop OOD clips missing any of `required_features` in feature_df,
    BEFORE we spend any shard-scanning effort on them.

    A clip absent from feature_df entirely (shouldn't normally happen, since
    feature_df's index is used elsewhere as the master clip list, but is
    handled defensively here) is treated as missing every feature, not
    silently kept -- reindex+fillna(False) below, rather than a join that
    would drop it in a way that's easy to miss.
    """
    aligned = feature_df.reindex(ood_df.index)[required_features].fillna(False)
    has_all_required = aligned.all(axis=1)

    n_before, n_after = len(ood_df), int(has_all_required.sum())
    if n_after < n_before:
        # Per-cluster breakdown so a category that gets hit hard by this
        # filter is visible, not just the aggregate count -- silent
        # truncation is exactly what this filter exists to prevent one step
        # earlier (missing cameras at harvest time), so it shouldn't
        # introduce a new silent truncation of its own.
        dropped = ood_df.loc[~has_all_required]
        logger.info(
            "feature_presence filter: dropped %d/%d OOD clips missing one of %s",
            n_before - n_after, n_before, required_features,
        )
        if "event_cluster" in dropped.columns:
            logger.info(
                "  dropped-by-cluster:\n%s",
                dropped["event_cluster"].value_counts().to_string(),
            )
    return ood_df.loc[has_all_required]


def list_shards(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].split("/")[-1].startswith("shard_"):
                keys.append(obj["Key"])
    return sorted(keys)


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    end = start + length - 1
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()


def scan_shard_for_clips(
    bucket: str, key: str, ood_ids: set[str]
) -> list[tuple[str, str, dict, int]]:
    """Walk one shard's tar headers, returning (clip_id, shard_key, json_meta,
    group_start_offset) for every clip in ood_ids found in this shard.

    group_start_offset is the offset of the FIRST tar member belonging to
    that clip, not the ".json" member's own offset -- webdataset.TarWriter
    writes each clip's files in alphabetical order by extension
    (calibration.json, camera_*.mp4, egomotion.parquet, json), so "json" is
    always the LAST member in a clip's group. Recording its own position and
    later trying to walk forward from there (see s3_clip_extract.py) would
    only ever re-find that same json and then hit the next clip's first
    header -- this tracks the true group start by watching for the clip_id
    prefix to change instead.
    """
    s3 = boto3.client("s3")
    pos = 0
    zero_blocks = 0
    found: list[tuple[str, str, dict, int]] = []
    current_clip_id: str | None = None
    current_clip_start = 0
    while True:
        hdr = _range_get(s3, bucket, key, pos, TAR_BLOCK)
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

        if not is_pax:
            this_clip_id = name.split(".", 1)[0] if "." in name else name
            if this_clip_id != current_clip_id:
                current_clip_id = this_clip_id
                current_clip_start = pos

        if name.endswith(".json") and not is_pax and not name.endswith("calibration.json"):
            clip_id = name[: -len(".json")]
            if clip_id in ood_ids:
                data = _range_get(s3, bucket, key, data_start, size)
                try:
                    found.append((clip_id, key, json.loads(data.decode()), current_clip_start))
                except Exception:
                    logger.warning("shard %s: failed to parse json for %s", key, clip_id)

        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
        pos = data_start + data_blocks * TAR_BLOCK
    return found


def find_available_ood_clips(
    bucket: str,
    prefix: str,
    ood_df: pd.DataFrame,
    n_per_type: int | None,
    max_workers: int = 40,
) -> dict[str, list[tuple[str, str, dict, int]]]:
    """Scan shards under prefix, grouping found OOD clips by event_cluster.

    n_per_type=None means "no cap" -- used to build a manifest of EVERY
    currently-uploaded OOD clip (see the --all CLI flag) rather than a
    balanced per-category sample. In that mode there is no "satisfied"
    condition to stop early on, so every shard under prefix gets scanned.

    With an integer n_per_type, stops as soon as every event_cluster has
    >= n_per_type candidates, rather than scanning every shard
    unconditionally — for common categories this resolves after a small
    fraction of shards. Categories too rare to reach n_per_type in what's
    uploaded so far will still require a full scan (there's no way to know
    a rare category is exhausted without looking everywhere), but that's
    inherent to how little of the rare clips exist yet, not scanner
    overhead.
    """
    ood_ids = set(ood_df.index.astype(str))
    cluster_of = ood_df["event_cluster"].to_dict()
    all_clusters = sorted(set(cluster_of.values()))

    keys = list_shards(bucket, prefix)
    if n_per_type is None:
        logger.info(
            "Scanning all %d shards under s3://%s/%s for all %d target OOD clips "
            "(--all mode: no per-category cap, no early stop)",
            len(keys), bucket, prefix, len(ood_ids),
        )
    else:
        logger.info(
            "Scanning up to %d shards under s3://%s/%s for %d target OOD clips "
            "(stopping early once all %d categories have >= %d candidates)",
            len(keys), bucket, prefix, len(ood_ids), len(all_clusters), n_per_type,
        )

    by_cluster: dict[str, list[tuple[str, str, dict, int]]] = defaultdict(list)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(scan_shard_for_clips, bucket, k, ood_ids): k for k in keys}
    try:
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                for clip_id, shard_key, meta, offset in fut.result():
                    by_cluster[cluster_of.get(clip_id, "UNKNOWN")].append((clip_id, shard_key, meta, offset))
            except Exception as exc:
                logger.error("Failed to scan %s: %s", key, exc)
            done += 1

            if n_per_type is None:
                if done % 20 == 0 or done == len(keys):
                    logger.info("  %d/%d shards scanned, %d clips found so far",
                                done, len(keys), sum(len(v) for v in by_cluster.values()))
                continue

            n_satisfied = sum(1 for c in all_clusters if len(by_cluster.get(c, [])) >= n_per_type)
            if done % 20 == 0 or n_satisfied == len(all_clusters):
                logger.info(
                    "  %d/%d shards scanned, %d/%d categories have >= %d candidates",
                    done, len(keys), n_satisfied, len(all_clusters), n_per_type,
                )
            if n_satisfied == len(all_clusters):
                logger.info("All categories satisfied after %d/%d shards -- stopping early",
                            done, len(keys))
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return by_cluster


def find_first_n_ood_clips(
    bucket: str,
    prefix: str,
    ood_df: pd.DataFrame,
    n_total: int,
    seed: int = 0,
    max_workers: int = 40,
) -> list[tuple[str, str, dict, int, str]]:
    """Scan shards, stopping as soon as n_total OOD clips have been found --
    ANY category, no per-category balance -- rather than find_available_ood_clips's
    balanced-per-cluster mode. For a quick, unstratified sample when waiting on
    a specific rare category's full quota isn't worth it (a handful of common
    clips will usually satisfy n_total long before a full scan finishes).

    Shard scan order is shuffled (seeded) before submission -- list_shards
    returns shards in lexicographic key order, which is very likely
    correlated with collection time/session/location. Stopping "as soon as
    n_total are found" against that UNshuffled order would silently sample
    only whatever's in the lexicographically-first shards actually reached
    (not a random subset of the dataset at all, just "whatever's first"),
    biasing the sample toward however that ordering happens to cluster.
    Shuffling first makes "first n_total found" approximate an actual
    random sample of the currently-available OOD clips.

    Each result also carries its event_cluster (unlike find_available_ood_clips's
    return value, which is already grouped by cluster) so the caller can
    record which category each sampled clip actually came from.
    """
    ood_ids = set(ood_df.index.astype(str))
    cluster_of = ood_df["event_cluster"].to_dict()

    keys = list_shards(bucket, prefix)
    np.random.default_rng(seed).shuffle(keys)
    logger.info(
        "Scanning up to %d shards under s3://%s/%s for the first %d OOD clips "
        "(any category, unstratified -- stopping as soon as %d are found)",
        len(keys), bucket, prefix, n_total, n_total,
    )

    found: list[tuple[str, str, dict, int, str]] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(scan_shard_for_clips, bucket, k, ood_ids): k for k in keys}
    try:
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                for clip_id, shard_key, meta, offset in fut.result():
                    found.append((clip_id, shard_key, meta, offset, cluster_of.get(clip_id, "UNKNOWN")))
            except Exception as exc:
                logger.error("Failed to scan %s: %s", key, exc)
            done += 1

            if done % 20 == 0 or len(found) >= n_total:
                logger.info("  %d/%d shards scanned, %d/%d clips found", done, len(keys), len(found), n_total)
            if len(found) >= n_total:
                logger.info(
                    "Found %d clips (>= target %d) after %d/%d shards -- stopping early",
                    len(found), n_total, done, len(keys),
                )
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return found


def sample_per_type(
    by_cluster: dict[str, list[tuple[str, str, dict, int]]],
    n_per_type: int | None,
    seed: int,
) -> list[dict]:
    """Sample up to n_per_type clips per event_cluster from what was found.

    n_per_type=None means take EVERY candidate found for every cluster --
    the --all mode. Sorted by clip_id for determinism in that case (there's
    nothing to sample, so the seed is unused -- taking "all" has no
    randomness to seed in the first place).
    """
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    for cluster in sorted(by_cluster):
        candidates = by_cluster[cluster]

        if n_per_type is None:
            selected = sorted(candidates, key=lambda c: c[0])  # sort by clip_id
        else:
            n_take = min(n_per_type, len(candidates))
            if n_take < n_per_type:
                logger.warning(
                    "event_cluster=%s: only %d/%d clips available in already-uploaded shards",
                    cluster, n_take, n_per_type,
                )
            idxs = rng.choice(len(candidates), size=n_take, replace=False)
            selected = [candidates[i] for i in idxs]

        for clip_id, shard_key, meta, offset in selected:
            manifest.append({
                "clip_id": clip_id, "event_cluster": cluster, "shard_key": shard_key,
                "offset": offset,
            })
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="research-datasets-chicago")
    ap.add_argument(
        "--prefix",
        default="nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/train",
    )
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument(
        "--n_per_type", type=int, default=10,
        help="Clips to sample per event_cluster. Ignored if --all is set.",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Take every currently-uploaded OOD clip (still filtered by "
             "feature_presence), instead of sampling n_per_type per category. "
             "Scans every shard -- no early stop.",
    )
    ap.add_argument(
        "--n_total", type=int, default=None,
        help="Sample this many OOD clips total, ANY category, unstratified "
             "(no per-category balance) -- stops as soon as n_total are "
             "found, rather than requiring every category to individually "
             "reach a quota. Each manifest entry still records its "
             "event_cluster. Mutually exclusive with --n_per_type/--all.",
    )
    ap.add_argument(
        "--clusters", default=None,
        help="Comma-separated event_cluster names to restrict sampling to "
             "(e.g. for a per-scenario-type experiment that doesn't need "
             "every OOD category). Restricts both scanning and the "
             "early-stop condition to just these clusters, instead of "
             "requiring every cluster in the full OOD taxonomy to satisfy "
             "n_per_type first. Omit to sample/scan every cluster (default).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="masking/configs/sample_clips.json")
    args = ap.parse_args()

    n_per_type = None if args.all else args.n_per_type

    ood_df = load_ood_clips(args.hf_token)

    feature_df = load_feature_presence(args.hf_token)
    ood_df = filter_clips_with_required_features(ood_df, feature_df)

    if args.clusters:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]
        n_before = len(ood_df)
        ood_df = ood_df[ood_df["event_cluster"].isin(clusters)]
        logger.info(
            "--clusters filter: restricted to %d/%d clips in %s",
            len(ood_df), n_before, clusters,
        )

    if args.n_total is not None:
        found = find_first_n_ood_clips(args.bucket, args.prefix, ood_df, args.n_total, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        if len(found) > args.n_total:
            # The last completed shard's batch can push slightly past
            # n_total (results arrive per-shard, not per-clip) -- trim back
            # down to exactly n_total via a random subset, not just
            # truncation, so which clips get dropped isn't biased toward
            # whichever shard happened to be scanned last.
            idxs = rng.choice(len(found), size=args.n_total, replace=False)
            found = [found[i] for i in idxs]
        manifest = [
            {"clip_id": clip_id, "event_cluster": cluster, "shard_key": shard_key, "offset": offset}
            for clip_id, shard_key, meta, offset, cluster in sorted(found, key=lambda f: f[0])
        ]
    else:
        by_cluster = find_available_ood_clips(args.bucket, args.prefix, ood_df, n_per_type)
        manifest = sample_per_type(by_cluster, n_per_type, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d sampled clips to %s", len(manifest), out_path)


if __name__ == "__main__":
    main()
