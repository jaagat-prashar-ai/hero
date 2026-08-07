#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-off: build WDS shards for the 214 OOD test-split clips in
reasoning/ood_reasoning_test.parquet and upload them to
s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/test/,
matching the train/val shard layout build_webdataset.py already produced there.

ood_reasoning_test.parquet has no `coc` (ground-truth reasoning text) field —
NVIDIA withholds it for the held-out test split — so these shards carry
event location/timing/category only, same as every other field build_clip_sample
already writes.
"""
import concurrent.futures
import logging
import os
import sys
import threading

import huggingface_hub as hfh
import pandas as pd
import physical_ai_av

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from build_wds.data.build_webdataset import (
    HF_REPO, ShardUploadFailed, S3ShardWriter, _hf_retry, build_clip_sample,
)
from build_wds.data.video_transcode import ensure_ffmpeg_av1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("build_test_split")

BUCKET = "research-datasets-chicago"
PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds"
SPLIT = "test"
WORKERS = 8
CLIPS_PER_SHARD = 50
TEST_PARQUET = "reasoning/ood_reasoning_test.parquet"


def main() -> None:
    encoder = ensure_ffmpeg_av1()
    logger.info("AV1 encoder: %s", encoder)

    df = pd.read_parquet(TEST_PARQUET)
    clip_ids = [str(c) for c in df.index]
    logger.info("Loaded %d test-split clips from %s", len(clip_ids), TEST_PARQUET)

    fp_path = _hf_retry(lambda: hfh.hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename="metadata/feature_presence.parquet"))
    dc_path = _hf_retry(lambda: hfh.hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", filename="metadata/data_collection.parquet"))
    feature_by_clip = {str(cid): row.to_dict() for cid, row in pd.read_parquet(fp_path).iterrows()}
    collection_by_clip = {str(cid): row.to_dict() for cid, row in pd.read_parquet(dc_path).iterrows()}

    avdi = _hf_retry(physical_ai_av.PhysicalAIAVDatasetInterface)
    writer = S3ShardWriter(BUCKET, PREFIX, SPLIT, CLIPS_PER_SHARD, worker_rank=0)

    n_ok, n_err = 0, 0
    lock = threading.Lock()

    def process(clip_id: str) -> None:
        nonlocal n_ok, n_err
        ood_events = [df.loc[clip_id].to_dict()]
        try:
            sample = build_clip_sample(
                avdi, clip_id,
                collection_by_clip.get(clip_id),
                ood_events,
                feature_by_clip.get(clip_id, {}),
                skip_lidar=True,
                video_codec="av1", video_crf=32, video_preset=6,
            )
            writer.write(clip_id, sample)
            with lock:
                n_ok += 1
                logger.info("OK %s (%d/%d done)", clip_id, n_ok + n_err, len(clip_ids))
        except ShardUploadFailed as exc:
            with lock:
                n_ok -= exc.clips_lost - 1
                n_err += exc.clips_lost
            logger.error("FAIL %s: shard upload failed, %d clips lost", clip_id, exc.clips_lost)
        except Exception as exc:
            with lock:
                n_err += 1
            logger.error("FAIL %s: %s", clip_id, exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(process, clip_ids))

    writer.close()
    logger.info("Finished: %d ok / %d err out of %d", n_ok, n_err, len(clip_ids))
    logger.info("Shards at s3://%s/%s/%s/", BUCKET, PREFIX, SPLIT)


if __name__ == "__main__":
    main()
