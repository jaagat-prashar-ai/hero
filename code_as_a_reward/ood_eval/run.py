# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad entrypoint (training_fn) for the OOD ground-truth vs.
model-rollout verifier comparison. Runs in Lilypad's BASE Python 3.10
environment -- cannot import physical_ai_av/alpamayo1_5 directly (see
bootstrap_venv.py's module docstring). Builds the event manifest here (base
env is enough: manifest.py only needs pandas + huggingface_hub), bootstraps
the isolated Python 3.12 venv, then runs worker.py -- the actual per-event
GT+model verification -- as a subprocess INSIDE that venv, same
process-boundary pattern as perplexity/training/run.py.

Durable output: worker.py appends to a NODE-LOCAL JSONL file (dies with the
pod otherwise -- masking/configs/cluster.yaml's exact prior lesson). A
background thread periodically uploads that file to S3 while the subprocess
runs, and a final upload happens in `finally` regardless of outcome, so a
preemption never loses more than one sync interval of progress. On startup,
any existing S3 object is downloaded to the local path FIRST, so worker.py's
own (clip_id, t0_us) skip-set resume logic (see worker.py's
`_load_done_keys`) spans across requeues, not just within one pod's
lifetime.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

S3_SYNC_INTERVAL_S = 60.0


def _download_if_exists(s3, bucket: str, key: str, local_path: str) -> None:
    import botocore.exceptions

    try:
        s3.download_file(bucket, key, local_path)
        logger.info("resumed: downloaded existing s3://%s/%s to %s", bucket, key, local_path)
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
        logger.info("no existing s3://%s/%s -- starting fresh", bucket, key)


def _upload(s3, bucket: str, key: str, local_path: str) -> None:
    if os.path.exists(local_path):
        s3.upload_file(local_path, bucket, key)


def _sync_loop(stop_event: threading.Event, s3, bucket: str, key: str, local_path: str) -> None:
    while not stop_event.wait(S3_SYNC_INTERVAL_S):
        try:
            _upload(s3, bucket, key, local_path)
        except Exception:
            logger.exception("periodic S3 sync failed (will retry next interval)")


def ood_verifier_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    from code_as_a_reward.ood_eval.bootstrap_venv import ensure_alpamayo15_venv
    from code_as_a_reward.ood_eval.manifest import build_manifest

    max_events = training_fn_config.get("max_events")
    clip_ids = training_fn_config.get("clip_ids")
    skip_model = bool(training_fn_config.get("skip_model", False))
    local_ood_parquet = training_fn_config.get("local_ood_parquet")
    venv_dir = training_fn_config.get("venv_dir", "/mnt/work/tmp/alpamayo15_venv")
    local_output = training_fn_config.get("local_output", "/mnt/work/tmp/ood_eval/results.jsonl")
    s3_bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    s3_key = training_fn_config["results_s3_key"]  # required: this run's durable output location
    cache_dir = training_fn_config.get("cache_dir", "code_as_a_reward/testdata")

    os.makedirs(os.path.dirname(local_output), exist_ok=True)

    import boto3

    s3 = boto3.client("s3")
    _download_if_exists(s3, s3_bucket, s3_key, local_output)

    logger.info(
        "building manifest (max_events=%s, clip_ids=%s) ...",
        max_events, "all" if clip_ids is None else len(clip_ids),
    )
    events = build_manifest(local_ood_parquet, max_events=max_events, clip_ids=clip_ids)
    logger.info("manifest built: %d events", len(events))
    if not events:
        raise RuntimeError("manifest resolved 0 events -- check the OOD parquet fetch / filters")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([dataclasses.asdict(e) for e in events], f)
        manifest_path = f.name

    python_bin = ensure_alpamayo15_venv(venv_dir, repo_root)

    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=_sync_loop, args=(stop_event, s3, s3_bucket, s3_key, local_output), daemon=True
    )
    sync_thread.start()

    cmd = [
        python_bin, "-m", "code_as_a_reward.ood_eval.worker",
        "--manifest_json", manifest_path,
        "--output", local_output,
        "--cache_dir", cache_dir,
    ]
    if skip_model:
        cmd.append("--skip_model")

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([repo_root, child_env.get("PYTHONPATH", "")])

    tail: deque[str] = deque(maxlen=80)
    try:
        proc = subprocess.Popen(
            cmd, cwd=repo_root, env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        returncode = proc.wait()
        logger.info("worker.py exited with code %d", returncode)
        if returncode != 0:
            raise RuntimeError(
                f"worker.py failed (exit {returncode}); last {len(tail)} output lines:\n{''.join(tail)}"
            )
    finally:
        stop_event.set()
        sync_thread.join(timeout=S3_SYNC_INTERVAL_S)
        _upload(s3, s3_bucket, s3_key, local_output)
        logger.info("final sync: s3://%s/%s", s3_bucket, s3_key)
        time.sleep(1)  # let the log line above actually ship before the process exits
