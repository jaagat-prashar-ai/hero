"""Rank-0 upload of the ModelCheckpoint directory to S3.

Checkpoints written to node-local disk die with the node (preemption or
post-run recycling), so every completed validation epoch syncs the
checkpoint dir to S3. Must be registered AFTER ModelCheckpoint in the
callbacks list so its hooks run once the checkpoint files are on disk;
the strategy barrier makes every rank's DeepSpeed shard write visible
before rank 0 starts uploading.
"""
import os
from pathlib import Path

import boto3
from botocore.config import Config
from pytorch_lightning.callbacks import Callback


def _s3_client():
    # OCI's S3-compat endpoint rejects s3transfer's multipart chunked
    # encoding ("NotImplemented: AWS chunked encoding not supported") -- same
    # bug and fix as rl_posttrain/training/run.py's _pai_cache_client
    # (BUGS.md 2026-07-01). payload_signing_enabled=True disables chunking
    # for single-shot requests only; callers below must use put_object, never
    # the multipart upload_file (s3transfer always chunks regardless).
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"payload_signing_enabled": True},
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


class S3CheckpointUpload(Callback):
    def __init__(self, dirpath: str, s3_uri: str):
        self.dirpath = Path(dirpath)
        bucket_key = s3_uri.replace("s3://", "", 1)
        self.bucket, _, self.prefix = bucket_key.partition("/")
        self._uploaded = {}  # relpath -> (size, mtime_ns)

    def _sync(self, trainer):
        trainer.strategy.barrier()
        if not trainer.is_global_zero or not self.dirpath.exists():
            return
        s3 = _s3_client()
        uploaded = 0
        for f in sorted(self.dirpath.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(self.dirpath))
            stat = f.stat()
            sig = (stat.st_size, stat.st_mtime_ns)
            if self._uploaded.get(rel) == sig:
                continue
            with open(f, "rb") as fh:
                s3.put_object(
                    Bucket=self.bucket, Key=f"{self.prefix.rstrip('/')}/{rel}", Body=fh
                )
            self._uploaded[rel] = sig
            uploaded += 1
        print(f"[s3ckpt] uploaded {uploaded} files from {self.dirpath} "
              f"to s3://{self.bucket}/{self.prefix}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._sync(trainer)

    def on_train_end(self, trainer, pl_module):
        self._sync(trainer)
