# SPDX-License-Identifier: Apache-2.0
"""Lilypad entrypoint: build Track 1 (CoC generation) submission.json files
for a fleet of post-trained Alpamayo RL checkpoints, one arm per GPU rank.

Runs under workload_type "training" with num_gpus: N -- Lilypad starts N
replicas of this function (RANK/WORLD_SIZE env, one GPU each, same node for
a100.8). Rank r owns arms[r] end-to-end:

  resolve checkpoint step on S3 (optionally waiting for a still-training
  run's final save) -> download model_rank_*.pth -> merge to HF safetensors
  via third_party/alpamayo-recipes/scripts/convert_cosmos_rl_checkpoint.py
  (max 2 concurrent conversions node-wide, mkdir-slot semaphore: each rank
  holds ~45 GB of tensors in RAM while merging) -> run
  reasoning/track1_worker.py in the bootstrapped py3.12 venv on this rank's
  GPU -> build submission_<arm>.json (every key from
  reasoning/ood_reasoning_test.parquet; unanswerable events become 6 empty
  strings, loudly logged) -> put_object to S3.

Durability follows code_as_a_reward/ood_eval/run.py: the worker's JSONL is
restored from S3 on start, synced every 60 s while running, and uploaded in
`finally`. All S3 writes are put_object -- OCI's S3-compat endpoint rejects
s3transfer's chunked upload_file (BUGS.md / code_reward_entry._dump_client).

Arm config (training_fn_config["arms"], one per rank):
  {"name": "code_v2_main",
   "ckpt_prefix": "alpamayo_rl/checkpoints/code_reward_clipgen_v2/20260817200928",
   "step": 447}                      # null ckpt_prefix = base SFT model
  optional: "wait_max_min": poll until step's save is complete (final save of
  a still-running training job), falling back to the max available step.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

S3_SYNC_INTERVAL_S = 60.0
BASE_MODEL = "nvidia/Alpamayo-1.5-10B"
TEST_PARQUET_REPO_PATH = "reasoning/ood_reasoning_test.parquet"
TEST_SHARDS_PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds/test/"
CONVERT_SCRIPT = "third_party/alpamayo-recipes/scripts/convert_cosmos_rl_checkpoint.py"
GROUP_SIZE_DEFAULT = 6


# ── GPU keepalive (rl_posttrain/training/run.py's _GpuKeepalive, minimal) ──
class _GpuKeepalive(threading.Thread):
    """Tiny matmul burst on this rank's GPU every few seconds so Lilypad's
    idle-GPU reaper doesn't kill the node during the long CPU-only phase
    (shard download, checkpoint download, DTensor merge)."""

    def __init__(self, interval_s: float = 5.0):
        super().__init__(daemon=True, name="gpu-keepalive")
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("gpu-keepalive: no CUDA device visible")
                return
            mat = torch.randn(1024, 1024, device="cuda:0")
            while not self._stop_event.wait(self._interval_s):
                (mat @ mat).sum().item()
            del mat
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("gpu-keepalive died (non-fatal)")

    def stop(self) -> None:
        self._stop_event.set()


# ── small S3 helpers (put_object only for writes) ──
def _put_file(s3, bucket: str, key: str, local_path: str) -> None:
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read())


def _download_if_exists(s3, bucket: str, key: str, local_path: str) -> None:
    import botocore.exceptions

    try:
        s3.download_file(bucket, key, local_path)
        logger.info("resumed s3://%s/%s", bucket, key)
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise


def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


# ── shared-node coordination ──
def _wait_for_marker(path: str, timeout_s: float, what: str) -> None:
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timed out waiting for {what} ({path})")
        time.sleep(10)


class _ConvertSlot:
    """Node-wide semaphore via atomic mkdir: at most `slots` concurrent
    DTensor merges (each holds ~45 GB in RAM)."""

    def __init__(self, base_dir: str, slots: int = 2):
        self._base = base_dir
        self._slots = slots
        self._held: str | None = None
        os.makedirs(base_dir, exist_ok=True)

    def __enter__(self):
        while self._held is None:
            for i in range(self._slots):
                path = os.path.join(self._base, f"slot_{i}")
                try:
                    os.mkdir(path)
                    self._held = path
                    return self
                except FileExistsError:
                    continue
            time.sleep(15)
        return self

    def __exit__(self, *exc):
        if self._held:
            os.rmdir(self._held)
            self._held = None


# ── checkpoint resolution / conversion ──
def _available_steps(s3, bucket: str, ckpt_prefix: str) -> dict[int, list[str]]:
    """Map step -> that step's policy/ keys, for steps under <prefix>/checkpoints/."""
    steps: dict[int, list[str]] = {}
    for key in _list_keys(s3, bucket, f"{ckpt_prefix}/checkpoints/"):
        parts = key.split("/")
        for i, p in enumerate(parts):
            if p.startswith("step_") and i + 1 < len(parts) and parts[i + 1] == "policy":
                steps.setdefault(int(p.removeprefix("step_")), []).append(key)
                break
    return steps


def _step_is_complete(policy_keys: list[str]) -> bool:
    names = [k.rsplit("/", 1)[-1] for k in policy_keys]
    n_model = sum(1 for n in names if n.startswith("model_rank_") and n.endswith(".pth"))
    n_marker = sum(1 for n in names if n.startswith(".rank_") and n.endswith("_complete"))
    return n_model > 0 and n_marker >= n_model


def _resolve_step(s3, bucket: str, arm: dict) -> int:
    want = int(arm["step"])
    wait_max_min = float(arm.get("wait_max_min", 0))
    deadline = time.time() + wait_max_min * 60
    while True:
        steps = _available_steps(s3, bucket, arm["ckpt_prefix"])
        if want in steps and _step_is_complete(steps[want]):
            return want
        if time.time() >= deadline:
            complete = [s for s, keys in steps.items() if _step_is_complete(keys)]
            if not complete:
                raise RuntimeError(f"{arm['name']}: no complete checkpoint under {arm['ckpt_prefix']}")
            fallback = max(complete)
            logger.warning("%s: step %d never appeared, falling back to step %d",
                           arm["name"], want, fallback)
            return fallback
        logger.info("%s: waiting for step %d (have: %s)", arm["name"], want,
                    sorted(steps) or "none")
        time.sleep(120)


def _download_policy(s3, bucket: str, ckpt_prefix: str, step: int, dest: str) -> str:
    policy_prefix = f"{ckpt_prefix}/checkpoints/step_{step}/policy/"
    os.makedirs(dest, exist_ok=True)
    model_keys = [k for k in _list_keys(s3, bucket, policy_prefix)
                  if k.rsplit("/", 1)[-1].startswith("model_rank_")]
    if not model_keys:
        raise RuntimeError(f"no model_rank_*.pth under s3://{bucket}/{policy_prefix}")
    for key in sorted(model_keys):
        local = os.path.join(dest, key.rsplit("/", 1)[-1])
        if os.path.exists(local):
            continue
        logger.info("downloading %s", key)
        s3.download_file(bucket, key, local + ".part")
        os.replace(local + ".part", local)
    return dest


def _base_config_dir(work_dir: str) -> str:
    """A directory holding just the release config.json, for the converter's
    copy-non-weight-files step (our worker loads tokenizer/processor straight
    from the HF release, so nothing more is needed)."""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(work_dir, "base_config")
    os.makedirs(dest, exist_ok=True)
    if not os.path.exists(os.path.join(dest, "config.json")):
        src = hf_hub_download(BASE_MODEL, "config.json")
        shutil.copy2(src, os.path.join(dest, "config.json"))
    return dest


def _convert(python_bin: str, repo_root: str, policy_dir: str, base_cfg_dir: str,
             export_dir: str, slot: _ConvertSlot) -> None:
    if os.path.exists(os.path.join(export_dir, "model.safetensors.index.json")):
        logger.info("export already present at %s, skipping conversion", export_dir)
        return
    with slot:
        subprocess.run(
            [python_bin, os.path.join(repo_root, CONVERT_SCRIPT),
             "--cosmos-policy-ckpt", policy_dir,
             "--base-hf-ckpt", base_cfg_dir,
             "--output-dir", export_dir,
             "--overwrite"],
            check=True,
        )
    if not os.path.exists(os.path.join(export_dir, "model.safetensors.index.json")):
        raise RuntimeError(f"converter produced no index in {export_dir}")


# ── submission assembly ──
def _rank_rollouts(rollouts: list[str]) -> list[str]:
    """Best-first ordering for the benchmark's top1_score (score of rollout 1).

    No GT and no usable model logprobs exist at submission time, so rank by
    medoid self-consistency: each rollout's score is its mean bag-of-words
    cosine similarity to the other rollouts, and the one closest to the
    sample consensus goes first. Arm-agnostic, deterministic (stable sort),
    and order only affects top1 (topk/avgk are order-invariant).
    """
    from collections import Counter
    from math import sqrt

    bags = [Counter(r.lower().split()) for r in rollouts]
    norms = [sqrt(sum(v * v for v in b.values())) or 1.0 for b in bags]

    def _cos(i: int, j: int) -> float:
        bi, bj = bags[i], bags[j]
        if len(bj) < len(bi):
            bi, bj = bj, bi
        return sum(v * bj[k] for k, v in bi.items()) / (norms[i] * norms[j])

    n = len(rollouts)
    scores = [sum(_cos(i, j) for j in range(n) if j != i) / max(n - 1, 1) for i in range(n)]
    order = sorted(range(n), key=lambda i: -scores[i])
    return [rollouts[i] for i in order]


def _expected_keys(repo_root: str) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(os.path.join(repo_root, TEST_PARQUET_REPO_PATH))
    return [f"{clip_id}_{i}"
            for clip_id, row in df.iterrows()
            for i in range(len(json.loads(row["events"])))]


def _build_submission(jsonl_path: str, expected: list[str], group_size: int) -> tuple[dict, dict]:
    by_key: dict[str, list[str]] = {}
    n_err = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "rollouts" in rec:
                by_key[rec["submission_key"]] = _rank_rollouts(rec["rollouts"][:group_size])
            else:
                n_err += 1
    missing = [k for k in expected if k not in by_key]
    submission = {k: by_key.get(k, [""] * group_size) for k in expected}
    stats = {"answered": len(by_key), "expected": len(expected),
             "missing_or_error": len(missing), "missing_keys": missing, "jsonl_errors": n_err}
    return submission, stats


# ── main entrypoint ──
def track1_loop(training_fn_config: dict, experiment_tracker=None) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    arms = training_fn_config["arms"]
    if rank >= len(arms):
        logger.info("rank %d/%d: no arm assigned (only %d arms), exiting", rank, world_size, len(arms))
        return
    arm = arms[rank]

    bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    out_prefix = training_fn_config["out_s3_prefix"].rstrip("/")
    work_dir = training_fn_config.get("work_dir", "/mnt/work/tmp/track1")
    venv_dir = training_fn_config.get("venv_dir", "/mnt/work/tmp/alpamayo15_venv")
    group_size = int(training_fn_config.get("group_size", GROUP_SIZE_DEFAULT))
    max_events = training_fn_config.get("max_events")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(work_dir, exist_ok=True)
    logger.info("rank %d/%d -> arm %s", rank, world_size, arm["name"])

    keepalive = _GpuKeepalive()
    keepalive.start()

    import boto3

    s3 = boto3.client("s3")

    # 1. Test shards: rank 0 downloads once for the node, everyone else waits.
    shards_dir = os.path.join(work_dir, "test_shards")
    shards_marker = os.path.join(shards_dir, "DOWNLOAD_OK")
    if rank == 0:
        os.makedirs(shards_dir, exist_ok=True)
        if not os.path.exists(shards_marker):
            shard_keys = [k for k in _list_keys(s3, bucket, TEST_SHARDS_PREFIX) if k.endswith(".tar")]
            if not shard_keys:
                raise RuntimeError(f"no test shards under s3://{bucket}/{TEST_SHARDS_PREFIX}")
            for key in sorted(shard_keys):
                local = os.path.join(shards_dir, key.rsplit("/", 1)[-1])
                if os.path.exists(local):
                    continue
                logger.info("downloading %s", key)
                s3.download_file(bucket, key, local + ".part")
                os.replace(local + ".part", local)
            with open(shards_marker, "w") as f:
                f.write("ok")
    else:
        _wait_for_marker(shards_marker, 45 * 60, "rank0 shard download")

    # 2. Venv (idempotent + safe under concurrent ranks) + extra deps.
    from code_as_a_reward.ood_eval.bootstrap_venv import ensure_alpamayo15_venv

    python_bin = ensure_alpamayo15_venv(venv_dir, repo_root)
    # uv venvs ship without pip (smoke 09h6u9's failure) -- install extra deps
    # the way bootstrap_venv does, and let exactly one rank do it.
    deps_marker = os.path.join(venv_dir, "TRACK1_DEPS_OK")
    deps_lock = os.path.join(venv_dir, "TRACK1_DEPS_LOCK")
    if not os.path.exists(deps_marker):
        try:
            os.mkdir(deps_lock)
            am_installer = True
        except FileExistsError:
            am_installer = False
        if am_installer:
            uv_bin = os.path.expanduser("~/.local/bin/uv")
            env = dict(os.environ)
            env["UV_NO_CONFIG"] = "1"
            env["PATH"] = f"{os.path.dirname(uv_bin)}:{env.get('PATH', '')}"
            subprocess.run([uv_bin, "pip", "install", "--python", python_bin,
                            "scipy>=1.11", "safetensors>=0.4"], check=True, env=env)
            with open(deps_marker, "w") as f:
                f.write("ok")
        else:
            _wait_for_marker(deps_marker, 20 * 60, "track1 extra deps install")

    # 3. Resolve + download + convert this arm's checkpoint (base arm skips).
    export_dir = None
    provenance = {"arm": arm["name"], "base_model": BASE_MODEL, "ckpt": None}
    if arm.get("ckpt_prefix"):
        step = _resolve_step(s3, bucket, arm)
        provenance["ckpt"] = {"prefix": arm["ckpt_prefix"], "step": step}
        arm_dir = os.path.join(work_dir, "arms", arm["name"])
        policy_dir = os.path.join(arm_dir, f"policy_step_{step}")
        export_dir = os.path.join(arm_dir, f"export_step_{step}")
        if not os.path.exists(os.path.join(export_dir, "model.safetensors.index.json")):
            _download_policy(s3, bucket, arm["ckpt_prefix"], step, policy_dir)
            _convert(python_bin, repo_root, policy_dir, _base_config_dir(work_dir),
                     export_dir, _ConvertSlot(os.path.join(work_dir, "convert_slots")))
            shutil.rmtree(policy_dir, ignore_errors=True)  # ~20 GB of .pth no longer needed

    # 4. Run the GPU worker with S3-durable JSONL.
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_local = os.path.join(out_dir, f"{arm['name']}.jsonl")
    jsonl_key = f"{out_prefix}/rollouts_{arm['name']}.jsonl"
    _download_if_exists(s3, bucket, jsonl_key, jsonl_local)

    stop_sync = threading.Event()

    def _sync_loop() -> None:
        while not stop_sync.wait(S3_SYNC_INTERVAL_S):
            try:
                if os.path.exists(jsonl_local):
                    _put_file(s3, bucket, jsonl_key, jsonl_local)
            except Exception:
                logger.exception("periodic sync failed (will retry)")

    sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    sync_thread.start()

    cmd = [python_bin, "-m", "reasoning.track1_worker",
           "--arm", arm["name"],
           "--shards-dir", shards_dir,
           "--output-jsonl", jsonl_local,
           "--group-size", str(group_size)]
    if export_dir:
        cmd += ["--export-dir", export_dir]
    if max_events is not None:
        cmd += ["--max-events", str(max_events)]

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([repo_root, child_env.get("PYTHONPATH", "")])
    try:
        keepalive.stop()  # worker owns the GPU from here
        subprocess.run(cmd, check=True, env=child_env, cwd=repo_root)
    finally:
        stop_sync.set()
        sync_thread.join(timeout=30)
        if os.path.exists(jsonl_local):
            _put_file(s3, bucket, jsonl_key, jsonl_local)

    # 5. Assemble + upload submission.json for this arm.
    submission, stats = _build_submission(jsonl_local, _expected_keys(repo_root), group_size)
    if max_events is None and stats["missing_or_error"]:
        logger.error("%s: %d/%d keys missing or errored: %s", arm["name"],
                     stats["missing_or_error"], stats["expected"], stats["missing_keys"][:20])

    sub_local = os.path.join(out_dir, f"submission_{arm['name']}.json")
    with open(sub_local, "w") as f:
        json.dump(submission, f)
    _put_file(s3, bucket, f"{out_prefix}/submission_{arm['name']}.json", sub_local)

    provenance.update(stats={k: v for k, v in stats.items() if k != "missing_keys"},
                      group_size=group_size, max_events=max_events)
    prov_local = os.path.join(out_dir, f"provenance_{arm['name']}.json")
    with open(prov_local, "w") as f:
        json.dump(provenance, f, indent=2)
    _put_file(s3, bucket, f"{out_prefix}/provenance_{arm['name']}.json", prov_local)
    logger.info("%s DONE: s3://%s/%s/submission_%s.json (%s)",
                arm["name"], bucket, out_prefix, arm["name"], provenance["stats"])
