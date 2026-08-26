# SPDX-License-Identifier: Apache-2.0
"""Lilypad entrypoints for ClipGen offline construction and diagnostics.

``clipgen_offline_gt_loop`` is the production corpus path. It stages only
recorded observations, NVIDIA CoC, and NVIDIA action, then builds cached
reward functions without importing Alpamayo or sampling policy rollouts.

``clipgen_real_rollout_loop`` is retained only for explicitly requested
rollout diagnostics/ablations. It samples Alpamayo rollout groups in a GPU
subprocess and must not be used to construct the offline reward corpus.

Durable output for diagnostic rollouts: rollout_worker.py writes
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


def merge_manifest_targets(manifest: list[dict], targets: list[dict]) -> list[dict]:
    """Return manifest rows with an asserted, authoritative ``t0_us``.

    Rollout sampling historically read targets.json while dossier building
    read manifest.json. Keeping the join implicit allowed the two phases to
    describe different moments of the same clip. Validate the one-to-one
    mapping before any GPU/API work and carry the keyframe in the runtime
    manifest consumed by ClipGen.
    """
    manifest_ids = [str(e["clip_id"]) for e in manifest]
    target_ids = [str(e["clip_id"]) for e in targets]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest contains duplicate clip_id rows")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("targets contains duplicate clip_id rows")
    missing_targets = sorted(set(manifest_ids) - set(target_ids))
    missing_manifest = sorted(set(target_ids) - set(manifest_ids))
    if missing_targets or missing_manifest:
        raise ValueError(
            "manifest/targets clip sets differ: "
            f"missing_targets={missing_targets[:10]}, "
            f"missing_manifest={missing_manifest[:10]}"
        )
    target_by_id = {str(t["clip_id"]): int(t["t0_us"]) for t in targets}
    merged = []
    for entry in manifest:
        clip_id = str(entry["clip_id"])
        t0_us = target_by_id[clip_id]
        if "t0_us" in entry and int(entry["t0_us"]) != t0_us:
            raise ValueError(
                f"clip {clip_id}: manifest t0_us={entry['t0_us']} != targets t0_us={t0_us}"
            )
        merged.append({**entry, "t0_us": t0_us})
    return merged


def _sync_dir(s3, bucket: str, prefix: str, local_dir: str) -> None:
    """put_object, NOT upload_file: the OCI S3-compat endpoint rejects
    boto3's managed-transfer chunked encoding ("AWS chunked encoding not
    supported" -- confirmed against a real 05vvru run, same limitation
    run_prototype.py's _sync_out_to_s3 and code_reward_entry.py's
    _CheckpointUploader already document/work around). Rollout JSON files
    are tiny (well under 1 MB), so whole-body put_object is fine."""
    if not os.path.isdir(local_dir):
        return
    for name in os.listdir(local_dir):
        path = os.path.join(local_dir, name)
        # rollout_worker atomically renames fully written documents to
        # *.json; never upload its transient *.tmp files.
        if name.endswith(".json") and os.path.isfile(path):
            with open(path, "rb") as f:
                s3.put_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/{name}", Body=f.read())


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


def _restore_prefix(s3, bucket: str, prefix: str, local_dir: str) -> int:
    """Download EVERY object under an S3 prefix into local_dir (flat, by
    basename) -- unlike _restore_dir (rollout dumps only, .json), this is
    for corpus-scale clipgen manifest data (obstacle/egomotion parquet +
    coc.txt + manifest.json + targets.json), too large to bundle via
    lilypad's code_assets zip (e.g. a 352-clip run is ~250MB total, split
    into per-shard prefixes of ~30MB each). Returns the number of files
    restored."""
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            s3.download_file(bucket, obj["Key"], os.path.join(local_dir, os.path.basename(obj["Key"])))
            n += 1
    return n


def clipgen_offline_gt_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    """Stage Full1050 inputs and build rewards without policy rollouts.

    This is intentionally separate from ``clipgen_real_rollout_loop`` so a
    configuration cannot accidentally bootstrap Alpamayo or sample rollout
    groups during offline reward-corpus generation.
    """

    import boto3

    s3_bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    s3 = boto3.client("s3")
    manifest_data_s3_prefix = training_fn_config.get("manifest_data_s3_prefix")
    if manifest_data_s3_prefix:
        manifest_local_dir = training_fn_config.get(
            "manifest_local_dir", "/mnt/work/tmp/clipgen_offline_manifest"
        )
        os.makedirs(manifest_local_dir, exist_ok=True)
        n = _restore_prefix(s3, s3_bucket, manifest_data_s3_prefix, manifest_local_dir)
        logger.info(
            "restored %d GT-only manifest files from s3://%s/%s",
            n,
            s3_bucket,
            manifest_data_s3_prefix,
        )
        manifest_path = os.path.join(manifest_local_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest_entries = json.load(f)
        for entry in manifest_entries:
            for key in (
                "obstacle_parquet",
                "egomotion_parquet",
                "gt_coc",
                "overlay_jpeg",
                "waypoints_npy",
            ):
                if entry.get(key) and not os.path.isabs(entry[key]):
                    entry[key] = os.path.join(manifest_local_dir, entry[key])
        with open(os.path.join(manifest_local_dir, "targets.json")) as f:
            targets = json.load(f)
    else:
        manifest_path = training_fn_config["manifest"]
        with open(manifest_path) as f:
            manifest_entries = json.load(f)
        targets = training_fn_config["targets"]

    manifest_entries = merge_manifest_targets(manifest_entries, targets)
    selection_key = training_fn_config.get("selection_report_s3_key")
    if selection_key:
        prior = json.loads(
            s3.get_object(Bucket=s3_bucket, Key=selection_key)["Body"].read()
        )
        prior_clips = prior.get("clips") or {}

        selection_mode = training_fn_config.get("selection_mode", "repair")

        def selected(entry: dict) -> bool:
            clip = prior_clips.get(entry["clip_id"])
            if selection_mode == "missing_prior":
                return not isinstance(clip, dict)
            if not isinstance(clip, dict) or clip.get("passed") is True:
                return False
            gt_validation = clip.get("gt_target_validation") or {}
            unsupported = any(
                "no currently verifiable discriminative action family" in str(reason)
                for reason in gt_validation.get("failures", [])
            )
            reward_generation_failure = gt_validation.get("passed") is True
            return unsupported or reward_generation_failure

        manifest_entries = [entry for entry in manifest_entries if selected(entry)]
        selected_ids = {entry["clip_id"] for entry in manifest_entries}
        targets = [target for target in targets if target["clip_id"] in selected_ids]
        logger.info(
            "selected %d clips in mode=%s using s3://%s/%s",
            len(manifest_entries),
            selection_mode,
            s3_bucket,
            selection_key,
        )
    with tempfile.NamedTemporaryFile("w", suffix=".offline.manifest.json", delete=False) as f:
        json.dump(manifest_entries, f)
        runtime_manifest = f.name

    from code_as_a_reward.clipgen.run_prototype import clipgen_offline_entrypoint

    clipgen_offline_entrypoint(
        {
            "manifest": runtime_manifest,
            "out_dir": training_fn_config.get(
                "out_dir", "/mnt/work/tmp/clipgen_offline_out"
            ),
            "backend": training_fn_config.get("backend", "openai"),
            "max_attempts": int(training_fn_config.get("max_attempts", 3)),
            "s3_bucket": s3_bucket,
            "s3_prefix": training_fn_config["s3_prefix"],
        }
    )


def clipgen_real_rollout_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    from code_as_a_reward.ood_eval.bootstrap_venv import ensure_alpamayo15_venv

    s3_bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    import boto3

    s3 = boto3.client("s3")

    manifest_data_s3_prefix = training_fn_config.get("manifest_data_s3_prefix")
    if manifest_data_s3_prefix:
        # Corpus-scale mode: manifest.json/targets.json/parquet/coc.txt for
        # this shard live in S3 (staged offline -- see
        # code_as_a_reward/clipgen/configs/ for the smoke configs' inline
        # `targets`/`manifest` convention, used only at small scale).
        # Restore once to local disk, then read manifest/targets from there.
        manifest_local_dir = training_fn_config.get("manifest_local_dir", "/mnt/work/tmp/clipgen_manifest_data")
        os.makedirs(manifest_local_dir, exist_ok=True)
        n = _restore_prefix(s3, s3_bucket, manifest_data_s3_prefix, manifest_local_dir)
        logger.info("restored %d manifest files from s3://%s/%s", n, s3_bucket, manifest_data_s3_prefix)
        manifest_path = os.path.join(manifest_local_dir, "manifest.json")
        # manifest.json's per-clip paths are bare filenames (correct
        # alongside the manifest when staged offline) -- rewrite them to
        # absolute paths now that they're restored to manifest_local_dir,
        # since run_prototype._load_clip opens them relative to the
        # process's cwd, not the manifest file's own directory (confirmed
        # by a real FileNotFoundError on shard0: 'No such file or
        # directory: <clip>.obstacle.offline.parquet').
        with open(manifest_path) as f:
            manifest_entries = json.load(f)
        for e in manifest_entries:
            for key in ("obstacle_parquet", "egomotion_parquet", "gt_coc", "overlay_jpeg", "waypoints_npy"):
                if e.get(key) and not os.path.isabs(e[key]):
                    e[key] = os.path.join(manifest_local_dir, e[key])
        with open(manifest_path, "w") as f:
            json.dump(manifest_entries, f)
        with open(os.path.join(manifest_local_dir, "targets.json")) as f:
            targets = json.load(f)
    else:
        manifest_path = training_fn_config["manifest"]
        with open(manifest_path) as f:
            manifest_entries = json.load(f)
        targets = training_fn_config["targets"]  # [{"clip_id","t0_us"}, ...]

    manifest_entries = merge_manifest_targets(manifest_entries, targets)
    # Never mutate a checked-in or restored source manifest in place. The
    # runtime copy is the exact contract handed to the generation phase.
    with tempfile.NamedTemporaryFile("w", suffix=".manifest.json", delete=False) as f:
        json.dump(manifest_entries, f)
        manifest_path = f.name

    generation_group_size = training_fn_config.get(
        "generation_group_size", training_fn_config.get("group_size", 12)
    )
    holdout_group_size = training_fn_config.get("holdout_group_size", 12)
    rollout_seed = int(training_fn_config.get("rollout_seed", 20260824))
    venv_dir = training_fn_config.get("venv_dir", "/mnt/work/tmp/alpamayo15_venv")
    rollouts_local = training_fn_config.get("rollouts_local", "/mnt/work/tmp/clipgen_rollouts")
    rollouts_s3_prefix = training_fn_config["rollouts_s3_prefix"]

    os.makedirs(rollouts_local, exist_ok=True)
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
        "--generation_group_size", str(generation_group_size),
        "--holdout_group_size", str(holdout_group_size),
        "--seed", str(rollout_seed),
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
            "manifest": manifest_path,
            "out_dir": training_fn_config.get("out_dir", "/mnt/work/tmp/clipgen_out"),
            "rollouts_dir": rollouts_local,
            "backend": training_fn_config.get("backend", "openai"),
            "min_generation_rollouts": generation_group_size,
            "min_holdout_rollouts": holdout_group_size,
            "holdout_top_k": training_fn_config.get("holdout_top_k", 2),
            "cross_scene_negatives": training_fn_config.get("cross_scene_negatives", 3),
            "min_cross_scene_margin": training_fn_config.get("min_cross_scene_margin", 0.10),
            "s3_bucket": s3_bucket,
            "s3_prefix": training_fn_config["s3_prefix"],
            "wandb_project": training_fn_config.get("wandb_project", "code-as-reward-clipgen"),
            "wandb_entity": training_fn_config.get("wandb_entity"),
            "name": training_fn_config.get("name"),
        }
    )
