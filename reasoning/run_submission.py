# SPDX-License-Identifier: Apache-2.0
"""Lilypad entrypoint for the 284-event PhysicalAI AV Track-1 submission."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _run_streamed(cmd: list[str], *, cwd: Path, env: dict[str, str], on_line=None) -> None:
    """Stream child output and retain a diagnostic tail for durable status."""
    tail: deque[str] = deque(maxlen=80)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if on_line is not None:
            on_line(line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"child exited {returncode}; final output:\n{''.join(tail)}"
        )


def _download_prefix(s3, bucket: str, prefix: str, dest: Path, *, keep=None) -> int:
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for attempt in range(8):
        try:
            objects = [o for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/") for o in page.get("Contents", [])]
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))
    if not objects:
        raise RuntimeError(f"no S3 objects under s3://{bucket}/{prefix}")
    kept = 0
    for obj in objects:
        rel = obj["Key"][len(prefix.rstrip("/") + "/"):]
        if not rel or (keep is not None and not keep(rel)):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, obj["Key"], str(target))
        kept += 1
    return kept


def submission_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    import boto3
    from code_as_a_reward.ood_eval.bootstrap_venv import ensure_alpamayo15_venv

    bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    workspace = Path(training_fn_config.get("workspace_dir", "/mnt/work/tmp/code_consistency_submission"))
    checkpoint_prefix = training_fn_config["checkpoint_s3_prefix"].rstrip("/")
    base_prefix = training_fn_config.get("base_model_s3_prefix", "alpamayo_rl/model_cache/alpamayo15_converted").rstrip("/")
    test_prefix = training_fn_config.get("test_wds_s3_prefix", "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/test").rstrip("/")
    output_key = training_fn_config.get("output_s3_key", "alpamayo_rl/submissions/code_consistency_full/submission.json")
    partial_key = output_key.rsplit("/", 1)[0] + "/submission.partial.json"
    status_key = output_key.rsplit("/", 1)[0] + "/status.json"
    repo = Path(__file__).resolve().parents[1]
    workspace.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    cosmos = workspace / "cosmos_policy"
    base = workspace / "base_hf"
    shards = workspace / "test_shards"
    def status(stage: str, error: str | None = None) -> None:
        import json
        body = json.dumps({"stage": stage, "error": error, "updated_at": time.time()}).encode()
        s3.put_object(Bucket=bucket, Key=status_key, Body=body, ContentType="application/json")

    try:
        status("restore_checkpoint")
        checkpoint_keep = lambda rel: (
            rel.startswith("model_rank_")
            or rel.startswith(".rank_")
            or rel in {"cosmos_config"}
        )
        logger.info(
            "restoring checkpoint: %d required objects",
            _download_prefix(s3, bucket, checkpoint_prefix, cosmos, keep=checkpoint_keep),
        )
        status("restore_base_model")
        logger.info("restoring base model: %d objects", _download_prefix(s3, bucket, base_prefix, base))
        status("restore_test_shards")
        logger.info("restoring test shards: %d objects", _download_prefix(s3, bucket, test_prefix, shards))

        rank_markers = list(cosmos.glob(".rank_*_complete"))
        model_ranks = list(cosmos.glob("model_rank_*.pth"))
        if len(rank_markers) != 4 or len(model_ranks) != 4:
            raise RuntimeError(f"incomplete step-253 checkpoint: markers={len(rank_markers)} model_ranks={len(model_ranks)}")
        shard_paths = sorted(shards.glob("*.tar"))
        if len(shard_paths) != 5:
            raise RuntimeError(f"expected 5 test WDS shards, got {len(shard_paths)}")

        status("bootstrap")
        python = ensure_alpamayo15_venv(str(workspace / "venv"), str(repo))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([
            str(repo), str(repo / "third_party/alpamayo-recipes/src"), env.get("PYTHONPATH", "")
        ])
        merged = workspace / "merged_hf"
        inference = workspace / "inference_hf"
        status("convert_cosmos")
        subprocess.run([
        python, str(repo / "third_party/alpamayo-recipes/scripts/convert_cosmos_rl_checkpoint.py"),
        "--cosmos-policy-ckpt", str(cosmos), "--base-hf-ckpt", str(base),
        "--output-dir", str(merged), "--overwrite",
        ], check=True, env=env)
        status("convert_inference")
        subprocess.run([
        python, str(repo / "third_party/alpamayo-recipes/scripts/convert_checkpoint.py"), "to-a15",
        "--input", str(merged), "--output", str(inference), "--overwrite",
        ], check=True, env=env)

        output = workspace / "submission.json"
        try:
            s3.download_file(bucket, partial_key, str(output))
            logger.info("resumed partial submission from s3://%s/%s", bucket, partial_key)
        except Exception:
            logger.info("no prior partial submission; starting generation from zero")
        progress_count = 0

        def sync_progress(line: str) -> None:
            nonlocal progress_count
            if not line.startswith("submission progress:"):
                return
            progress_count += 1
            if progress_count % 10 == 0 and output.exists():
                s3.upload_file(str(output), bucket, partial_key)
                logger.info("synced partial submission after %d new events", progress_count)

        status("inference")
        _run_streamed([
            python, "-m", "reasoning.generate_submission", "--checkpoint", str(inference),
            "--shards", *map(str, shard_paths), "--output", str(output),
        ], cwd=repo, env=env, on_line=sync_progress)
        status("upload")
        s3.upload_file(str(output), bucket, output_key)
        alias = output.with_name("submissions.json")
        shutil.copy2(output, alias)
        s3.upload_file(str(alias), bucket, output_key.rsplit("/", 1)[0] + "/submissions.json")
        status("complete")
        logger.info("submission complete: s3://%s/%s", bucket, output_key)
    except Exception as exc:
        logger.exception("submission failed")
        try:
            if "output" in locals() and output.exists():
                s3.upload_file(str(output), bucket, partial_key)
        except Exception:
            logger.exception("failed to persist partial submission")
        try:
            status("failed", f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.exception("failed to persist submission failure status")
        raise
