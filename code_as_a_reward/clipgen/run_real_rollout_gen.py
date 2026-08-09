# SPDX-License-Identifier: Apache-2.0
"""Lilypad training_fn entrypoint: samples real Alpamayo rollout groups for
clipgen's target clips (GPU subprocess inside the bootstrapped alpamayo1_5
venv -- mirrors code_as_a_reward/ood_eval/run.py's process-boundary split),
then runs the LLM-based reward-function generation+gate loop
(run_prototype.py) against those real rollouts instead of GT. See
run_prototype.py's module docstring for why GT-only gating is wrong.

Durable output for the rollouts themselves: rollout_worker.py writes
NODE-LOCAL JSON (dies with the pod otherwise). A background thread
periodically syncs that directory to S3 while the subprocess runs, and a
final sync happens in `finally` -- same pattern as ood_eval/run.py, so a
preemption never loses more than one sync interval, and existing S3 objects
are restored to the local dir FIRST so a re-run doesn't resample clips it
already has (rollout_worker.py's own skip-if-exists logic then spans across
requeues).
"""

from __future__ import annotations

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


def _sync_dir(s3, bucket: str, prefix: str, local_dir: str) -> None:
    if not os.path.isdir(local_dir):
        return
    for name in os.listdir(local_dir):
        path = os.path.join(local_dir, name)
        if os.path.isfile(path):
            s3.upload_file(path, bucket, f"{prefix.rstrip('/')}/{name}")


def _sync_dir_loop(stop_event: threading.Event, s3, bucket: str, prefix: str, local_dir: str) -> None:
    while not stop_event.wait(S3_SYNC_INTERVAL_S):
        try:
            _sync_dir(s3, bucket, prefix, local_dir)
        except Exception:
            logger.exception("periodic S3 sync failed (will retry next interval)")


def _restore_dir(s3, bucket: str, prefix: str, local_dir: str) -> None:
    import botocore.exceptions

    try:
        paginator = s3.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                found = True
                s3.download_file(bucket, obj["Key"], os.path.join(local_dir, os.path.basename(obj["Key"])))
        logger.info(
            "restored %s from s3://%s/%s", "existing rollouts" if found else "nothing (fresh start)", bucket, prefix
        )
    except botocore.exceptions.ClientError:
        logger.info("no existing s3://%s/%s -- starting fresh", bucket, prefix)


def clipgen_real_rollout_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    from code_as_a_reward.ood_eval.bootstrap_venv import ensure_alpamayo15_venv

    targets = training_fn_config["targets"]  # [{"clip_id","t0_us"}, ...]
    group_size = training_fn_config.get("group_size", 12)
    venv_dir = training_fn_config.get("venv_dir", "/mnt/work/tmp/alpamayo15_venv")
    rollouts_local = training_fn_config.get("rollouts_local", "/mnt/work/tmp/clipgen_rollouts")
    s3_bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    rollouts_s3_prefix = training_fn_config["rollouts_s3_prefix"]

    os.makedirs(rollouts_local, exist_ok=True)
    import boto3

    s3 = boto3.client("s3")
    _restore_dir(s3, s3_bucket, rollouts_s3_prefix, rollouts_local)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(targets, f)
        targets_path = f.name

    logger.info("bootstrapping alpamayo1.5 venv ...")
    python_bin = ensure_alpamayo15_venv(venv_dir, repo_root)

    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=_sync_dir_loop, args=(stop_event, s3, s3_bucket, rollouts_s3_prefix, rollouts_local), daemon=True
    )
    sync_thread.start()

    cmd = [
        python_bin, "-m", "code_as_a_reward.clipgen.rollout_worker",
        "--targets_json", targets_path,
        "--output_dir", rollouts_local,
        "--group_size", str(group_size),
    ]
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([repo_root, child_env.get("PYTHONPATH", "")])

    tail: deque[str] = deque(maxlen=80)
    try:
        proc = subprocess.Popen(
            cmd, cwd=repo_root, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        returncode = proc.wait()
        logger.info("rollout_worker.py exited with code %d", returncode)
        if returncode != 0:
            raise RuntimeError(
                f"rollout_worker.py failed (exit {returncode}); last {len(tail)} output lines:\n{''.join(tail)}"
            )
    finally:
        stop_event.set()
        sync_thread.join(timeout=S3_SYNC_INTERVAL_S)
        _sync_dir(s3, s3_bucket, rollouts_s3_prefix, rollouts_local)
        logger.info("rollout sampling done, synced to s3://%s/%s", s3_bucket, rollouts_s3_prefix)
        time.sleep(1)

    # Phase 2: LLM generation + real-rollout gate loop, in THIS (base, no
    # torch/GPU needed from here on) process -- reads the rollout files
    # rollout_worker.py (or a restored prior attempt) just wrote.
    from code_as_a_reward.clipgen.run_prototype import clipgen_entrypoint

    clipgen_entrypoint(
        {
            "manifest": training_fn_config["manifest"],
            "out_dir": training_fn_config.get("out_dir", "/mnt/work/tmp/clipgen_out"),
            "rollouts_dir": rollouts_local,
            "backend": training_fn_config.get("backend", "openai"),
            "s3_bucket": s3_bucket,
            "s3_prefix": training_fn_config["s3_prefix"],
            "wandb_project": training_fn_config.get("wandb_project", "code-as-reward-clipgen"),
            "wandb_entity": training_fn_config.get("wandb_entity"),
            "name": training_fn_config.get("name"),
        }
    )
