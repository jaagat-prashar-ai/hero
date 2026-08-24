"""Lilypad entrypoint: 3-arm open-loop faithfulness inference on one a100.8 node.

Ranks 0..len(arms)-1 each evaluate one arm (checkpoint) over the SAME seeded
500-clip subset of the held-out validation_1_scenario split; higher ranks exit.
Rank 0 mirrors the eval data slice (include_regex) from S3 first; every active
rank then downloads its own checkpoint, runs eval_faith_worker.py on its pinned
GPU, and uploads predictions.jsonl to results_s3_prefix/<arm-name>/.
"""
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from simlingo_lilypad.run import _s3_client, _wait_for_marker


def _download_and_extract_subset(bucket: str, prefix: str, include_regex: str, workdir: Path) -> None:
    # dir name must match the evalset_commentary.json path prefix exactly --
    # BaseDataset compares full path strings, not resolved files
    dest = workdir / "database" / "simlingo_v2_2025_01_10"
    marker = workdir / ".extract_done"
    if marker.exists():
        print(f"[faith] {marker} exists, skipping download/extract", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    s3 = _s3_client()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    pat = re.compile(include_regex)
    keep = [k for k in keys if pat.search(k)]
    if not keep:
        raise RuntimeError(f"include_regex {include_regex!r} matched 0 of {len(keys)} objects")
    print(f"[faith] extracting {len(keep)}/{len(keys)} objects matching {include_regex!r}", flush=True)
    for i, key in enumerate(keep):
        name = key.rsplit("/", 1)[-1]
        local = workdir / name
        print(f"[faith] ({i + 1}/{len(keep)}) {name}", flush=True)
        s3.download_file(bucket, key, str(local))
        if name.endswith(".tar.gz"):
            with tarfile.open(local, "r:gz") as tf:
                tf.extractall(dest)
            local.unlink()
        else:
            local.rename(dest / name)
    marker.touch()


def _download_checkpoint(bucket: str, arm: dict[str, Any], workdir: Path) -> Path:
    """Mirror the arm's checkpoint locally. kind=zero mirrors the whole ckpt dir
    (get_fp32_state_dict_from_zero_checkpoint needs latest + all shards);
    kind=pt downloads the single consolidated file."""
    s3 = _s3_client()
    local_root = workdir / "ckpts" / arm["name"]
    done = local_root / ".done"
    if done.exists():
        return local_root / arm["local_entry"]
    local_root.mkdir(parents=True, exist_ok=True)
    prefix = arm["ckpt_s3_prefix"].rstrip("/") + "/"
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    if not keys:
        raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
    for i, key in enumerate(keys):
        rel = key[len(prefix):]
        local = local_root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[faith:{arm['name']}] ckpt ({i + 1}/{len(keys)}) {rel}", flush=True)
        s3.download_file(bucket, key, str(local))
    done.touch()
    return local_root / arm["local_entry"]


def eval_faithfulness(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    cfg = training_fn_config
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    arms = cfg["arms"]
    workdir = Path(cfg.get("workdir", "/mnt/work/simlingo_faith"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(workdir / "hf_cache"))
    bucket = cfg["s3_bucket"]

    if rank == 0:
        _download_and_extract_subset(bucket, cfg["s3_prefix"].rstrip("/") + "/",
                                     cfg["include_regex"], workdir)
    else:
        _wait_for_marker(workdir, timeout_s=int(cfg.get("extract_timeout_s", 14400)))

    if rank >= len(arms):
        print(f"[faith] rank {rank} idle (only {len(arms)} arms)", flush=True)
        return
    arm = arms[rank]

    ckpt_path = _download_checkpoint(bucket, arm, workdir)
    base_cfg_local = workdir / f"base_config_rank{rank}.yaml"
    _s3_client().download_file(bucket, cfg["base_config_s3_key"], str(base_cfg_local))

    repo_root = Path(__file__).resolve().parents[1]
    sim_root = repo_root / "simlingo" / "simlingo"
    db_link = sim_root / "database"
    try:
        db_link.symlink_to(workdir / "database")
    except FileExistsError:
        pass

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{sim_root}:{repo_root}:{env.get('PYTHONPATH', '')}"
    # Ray exposes all colocated GPUs to every worker; pin one per rank
    parent_local_rank = int(os.environ.get("LOCAL_RANK", rank))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [g for g in visible.split(",") if g] or [str(i) for i in range(world_size)]
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids[parent_local_rank % len(gpu_ids)]

    out_local = workdir / "predictions" / arm["name"] / "predictions.jsonl"
    cmd = [
        sys.executable, str(repo_root / "simlingo_lilypad" / "eval_faith_worker.py"),
        "--config", str(base_cfg_local),
        "--checkpoint", str(ckpt_path),
        "--out", str(out_local),
        "--max-clips", str(cfg.get("max_clips", 500)),
        "--seed", str(cfg.get("subset_seed", 1234)),
        "--batch-size", str(cfg.get("batch_size", 4)),
    ]
    print(f"[faith:{arm['name']}] launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=sim_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"worker exited {result.returncode} for arm {arm['name']}")

    dest_key = f"{cfg['results_s3_prefix'].rstrip('/')}/{arm['name']}/predictions.jsonl"
    _s3_client().upload_file(str(out_local), bucket, dest_key)
    print(f"[faith:{arm['name']}] uploaded s3://{bucket}/{dest_key}", flush=True)
