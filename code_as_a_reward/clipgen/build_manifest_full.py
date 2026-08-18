# SPDX-License-Identifier: Apache-2.0
"""Corpus-scale manifest builder for clipgen curation: given a JSON pool of
{clip_id, t0_us, gt_coc, chunk} (see build_manifest_full.pool_from_ood_reasoning),
downloads each touched chunk's obstacle.offline + egomotion.offline label
zips via the vendored scripts/download_pai.py, extracts each pool clip's own
parquet + writes its GT coc text, and uploads each clip's data to S3
IMMEDIATELY (shard index assigned up front, deterministically, so no
end-of-run accumulation step is needed) -- the same layout
run_real_rollout_gen.py's manifest_data_s3_prefix restore expects.

Run on a cluster node (not a local dev box): the label zips for ~700+
chunks run tens of GB, far more than this repo's usual dev-box headroom.
Chunks are processed and deleted one at a time to keep local disk bounded.

Crash safety: every clip's parquet/coc data is durable in S3 within
seconds of extraction -- nothing is held in memory or local disk waiting
for a final batch step. Each shard's manifest.json/targets.json is
re-uploaded (small, cheap) after every chunk, so a crash loses at most the
one in-flight chunk's shard-metadata updates, never any clip's actual data.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DOWNLOAD_PAI = "third_party/alpamayo-recipes/scripts/download_pai.py"


def pool_from_ood_reasoning(*, split: str = "train") -> list[dict]:
    """Every (clip_id, t0_us, gt_coc, chunk) triple in the train split whose
    first in-window event survives select_dense_ood_chunks.py's own margin +
    loader-history filters -- the exact same eligibility test the existing
    100-dense-chunk corpus was built with, just applied to the FULL pool
    instead of only the 100 densest chunks."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    START_MARGIN_US = int(1.6e6)
    END_MARGIN_US = int(6.4e6)
    CLIP_DURATION_US = 20_000_000
    HISTORY_RANGE_US = int(16 * 0.1 * 1e6)

    ood = pd.read_parquet(
        hf_hub_download(REPO_ID, "reasoning/ood_reasoning.parquet", repo_type="dataset")
    )
    clip_index = pd.read_parquet(hf_hub_download(REPO_ID, "clip_index.parquet", repo_type="dataset"))

    def events_nonempty(e):
        try:
            return e is not None and len(e) > 0
        except TypeError:
            return False

    def first_kept_event(events_cell):
        parsed = json.loads(events_cell) if isinstance(events_cell, str) else events_cell
        if parsed is None or not hasattr(parsed, "__iter__"):
            return None
        for ev in parsed:
            if not (isinstance(ev, dict) and "event_start_timestamp" in ev):
                continue
            t0 = int(ev["event_start_timestamp"])
            if t0 >= START_MARGIN_US and t0 + END_MARGIN_US <= CLIP_DURATION_US:
                return ev
        return None

    ood_ne = ood[ood["events"].map(events_nonempty)]
    ood_split = ood_ne[ood_ne["split"] == split] if split != "all" else ood_ne

    ci = clip_index[["chunk"]].copy()
    ci.index = ci.index.astype(str)
    chunk_map = ci["chunk"].to_dict()

    rows = []
    for clip_id, row in ood_split.iterrows():
        ev = first_kept_event(row["events"])
        if ev is None:
            continue
        t0 = int(ev["event_start_timestamp"])
        if t0 <= HISTORY_RANGE_US:
            continue
        chunk = chunk_map.get(str(clip_id))
        if chunk is None:
            continue
        rows.append(
            {
                "clip_id": str(clip_id),
                "t0_us": t0,
                "gt_coc": ev["coc"],
                "event_cluster": str(row.get("event_cluster")),
                "chunk": int(chunk),
            }
        )
    return rows


def _download_chunk_labels(repo_root: str, chunk: int, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        os.path.join(repo_root, DOWNLOAD_PAI),
        "--chunk-ids",
        str(chunk),
        "--labels",
        "egomotion.offline",
        "obstacle.offline",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def build(
    pool: list[dict],
    *,
    repo_root: str,
    work_dir: str,
    shard_size: int,
    s3_bucket: str,
    s3_prefix_root: str,
) -> dict:
    import boto3

    # Same endpoint convention as rl_posttrain/rewards/code_reward_entry.py's
    # _dump_client(): OCI's S3-compat endpoint isn't boto3's default, and
    # relying on the bare client silently falls back to AWS's real S3
    # (wrong bucket entirely) or fails auth outright off-cluster.
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    by_chunk: dict[int, list[dict]] = {}
    for r in pool:
        by_chunk.setdefault(r["chunk"], []).append(r)
    chunks = sorted(by_chunk)
    logger.info("pool: %d clips across %d chunks", len(pool), len(chunks))

    # Shard assignment is decided UP FRONT, deterministically, from the pool
    # order alone -- so each clip's data can be uploaded straight to its
    # final shard_N/ prefix the moment it's extracted, with no later
    # accumulate-then-shard pass. A skipped clip just leaves a small gap in
    # its shard; nothing needs re-numbering.
    n_shards = max(1, (len(pool) + shard_size - 1) // shard_size)
    shard_of_clip = {r["clip_id"]: i % n_shards for i, r in enumerate(pool)}
    manifest_by_shard: dict[int, list[dict]] = {i: [] for i in range(n_shards)}
    targets_by_shard: dict[int, list[dict]] = {i: [] for i in range(n_shards)}
    dirty_shards: set[int] = set()

    def flush_shard_metadata(shard_ids) -> None:
        for i in shard_ids:
            prefix = f"{s3_prefix_root}/shard_{i}"
            s3.put_object(Bucket=s3_bucket, Key=f"{prefix}/manifest.json", Body=json.dumps(manifest_by_shard[i]).encode())
            s3.put_object(Bucket=s3_bucket, Key=f"{prefix}/targets.json", Body=json.dumps(targets_by_shard[i]).encode())

    n_staged = 0
    skipped: list[str] = []

    for i, chunk in enumerate(chunks):
        chunk_dir = work / f"chunk_{chunk:04d}"
        try:
            _download_chunk_labels(repo_root, chunk, chunk_dir)
        except subprocess.CalledProcessError as e:
            logger.warning("chunk %d download failed (%s), skipping its %d clips", chunk, e, len(by_chunk[chunk]))
            skipped.extend(r["clip_id"] for r in by_chunk[chunk])
            shutil.rmtree(chunk_dir, ignore_errors=True)
            continue

        obs_zip = chunk_dir / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{chunk:04d}.zip"
        ego_zip = chunk_dir / "labels" / "egomotion.offline" / f"egomotion.offline.chunk_{chunk:04d}.zip"
        try:
            with zipfile.ZipFile(obs_zip) as zf_obs, zipfile.ZipFile(ego_zip) as zf_ego:
                for r in by_chunk[chunk]:
                    clip_id = r["clip_id"]
                    obs_name = f"{clip_id}.obstacle.offline.parquet"
                    ego_name = f"{clip_id}.egomotion.offline.parquet"
                    try:
                        obs_bytes = zf_obs.read(obs_name)
                        ego_bytes = zf_ego.read(ego_name)
                    except KeyError:
                        skipped.append(clip_id)
                        continue

                    shard_i = shard_of_clip[clip_id]
                    prefix = f"{s3_prefix_root}/shard_{shard_i}"
                    # Uploaded straight from the read bytes -- never touches
                    # local disk, never held in memory past this iteration.
                    s3.put_object(Bucket=s3_bucket, Key=f"{prefix}/{obs_name}", Body=obs_bytes)
                    s3.put_object(Bucket=s3_bucket, Key=f"{prefix}/{ego_name}", Body=ego_bytes)
                    s3.put_object(Bucket=s3_bucket, Key=f"{prefix}/{clip_id}.coc.txt", Body=r["gt_coc"].encode())

                    manifest_by_shard[shard_i].append(
                        {
                            "clip_id": clip_id,
                            "obstacle_parquet": obs_name,
                            "egomotion_parquet": ego_name,
                            "gt_coc": f"{clip_id}.coc.txt",
                            "hz": 10.0,
                        }
                    )
                    targets_by_shard[shard_i].append({"clip_id": clip_id, "t0_us": r["t0_us"]})
                    dirty_shards.add(shard_i)
                    n_staged += 1
        except (zipfile.BadZipFile, FileNotFoundError) as e:
            logger.warning("chunk %d zip read failed (%s), skipping its %d clips", chunk, e, len(by_chunk[chunk]))
            skipped.extend(r["clip_id"] for r in by_chunk[chunk])

        shutil.rmtree(chunk_dir, ignore_errors=True)

        # Re-upload manifest.json/targets.json for every shard touched by
        # this chunk -- small (KB-scale) so doing it every chunk is cheap,
        # and it's what makes a crash lose at most one chunk's worth of
        # shard-metadata bookkeeping, never any clip's actual parquet/coc.
        if dirty_shards:
            flush_shard_metadata(dirty_shards)
            dirty_shards.clear()

        if (i + 1) % 20 == 0:
            logger.info("processed %d/%d chunks, %d clips staged, %d skipped", i + 1, len(chunks), n_staged, len(skipped))

    logger.info("staged %d/%d clips (%d skipped: missing chunk/clip data)", n_staged, len(pool), len(skipped))
    return {"n_clips_staged": n_staged, "n_skipped": len(skipped), "n_shards": n_shards, "skipped_clip_ids": skipped}


def run_from_lilypad_config(training_fn_config: dict, experiment_tracker=None) -> None:
    """entrypoint_fn for a lilypad `workload_type: generic` job -- same
    (config, experiment_tracker) calling convention as
    rl_posttrain.training.run.rl_local_test_loop /
    code_as_a_reward.clipgen.run_real_rollout_gen.clipgen_real_rollout_loop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import boto3

    pool_s3 = training_fn_config["pool_s3"]  # "s3://bucket/key"
    bucket, key = pool_s3[len("s3://") :].split("/", 1)
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))
    pool = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = build(
        pool,
        repo_root=repo_root,
        work_dir=training_fn_config.get("work_dir", "/mnt/work/tmp/clipgen_manifest_build"),
        shard_size=training_fn_config.get("shard_size", 43),
        s3_bucket=training_fn_config.get("s3_bucket", "research-datasets-chicago"),
        s3_prefix_root=training_fn_config["s3_prefix_root"],
    )
    logger.info("done: %s", {k: v for k, v in result.items() if k != "skipped_clip_ids"})
    if result["skipped_clip_ids"]:
        logger.warning("skipped %d clips (missing chunk/clip data): %s", len(result["skipped_clip_ids"]), result["skipped_clip_ids"][:20])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-json", required=True, help="local path OR s3://bucket/key to the pool JSON")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--work-dir", default="/mnt/work/tmp/clipgen_manifest_build")
    ap.add_argument("--shard-size", type=int, default=43)
    ap.add_argument("--s3-bucket", default="research-datasets-chicago")
    ap.add_argument("--s3-prefix-root", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.pool_json.startswith("s3://"):
        import boto3

        bucket, key = args.pool_json[len("s3://") :].split("/", 1)
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))
        pool = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    else:
        pool = json.loads(Path(args.pool_json).read_text())

    result = build(
        pool,
        repo_root=args.repo_root,
        work_dir=args.work_dir,
        shard_size=args.shard_size,
        s3_bucket=args.s3_bucket,
        s3_prefix_root=args.s3_prefix_root,
    )
    logger.info("done: %s", {k: v for k, v in result.items() if k != "skipped_clip_ids"})
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
