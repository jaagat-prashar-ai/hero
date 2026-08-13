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
from pytorch_lightning.callbacks import Callback


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
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))
        uploaded = 0
        for f in sorted(self.dirpath.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(self.dirpath))
            stat = f.stat()
            sig = (stat.st_size, stat.st_mtime_ns)
            if self._uploaded.get(rel) == sig:
                continue
            s3.upload_file(str(f), self.bucket, f"{self.prefix.rstrip('/')}/{rel}")
            self._uploaded[rel] = sig
            uploaded += 1
        print(f"[s3ckpt] uploaded {uploaded} files from {self.dirpath} "
              f"to s3://{self.bucket}/{self.prefix}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._sync(trainer)

    def on_train_end(self, trainer, pl_module):
        self._sync(trainer)
