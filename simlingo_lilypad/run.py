"""Lilypad entrypoint for SimLingo training runs.

Execution model: with cluster_resources.num_gpus=N, Lilypad's TorchTrainer
invokes smoke_train in N worker processes on the node, each pinned to one GPU
via CUDA_VISIBLE_DEVICES. Rank 0 mirrors the dataset from S3 onto the shared
node-local workdir; the other ranks wait on a marker file. Every rank then
execs simlingo_training/train.py as one member of a Lightning DDP/DeepSpeed
group (devices=1, num_nodes=N) rendezvousing on localhost.
"""
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import boto3

MASTER_PORT = "29617"  # distinct from Ray's own process-group port


def _s3_client():
    # AWS_ENDPOINT_URL_S3 / creds come from the workload runtime environment
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))


def _download_and_extract(bucket: str, prefix: str, workdir: Path, exclude_regex: str = None) -> None:
    dest = workdir / "database" / "simlingo"
    marker = workdir / ".extract_done"
    if marker.exists():
        print(f"[simlingo] {marker} exists, skipping download/extract", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()
    expected = int(os.environ.get("SIMLINGO_EXPECTED_OBJECTS", "0")) or None
    deadline = time.time() + 45 * 60
    while True:
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(o["Key"] for o in page.get("Contents", []))
        if expected is None or len(keys) >= expected:
            break
        if time.time() > deadline:
            raise RuntimeError(f"mirror incomplete: {len(keys)}/{expected} objects under s3://{bucket}/{prefix}")
        print(f"[simlingo] mirror still filling ({len(keys)}/{expected}), waiting 60s", flush=True)
        time.sleep(60)

    # applied AFTER the completeness check so SIMLINGO_EXPECTED_OBJECTS keeps
    # counting the whole mirror. The 2026-08-14 validation backfill (+341GB
    # compressed) pushed the extract-everything footprint past the ~85% disk
    # eviction threshold and killed all 8 full-data arms mid-extract on
    # 2026-08-16; excluding the backfill sensor chunks restores the exact
    # pre-backfill extract set that is proven to fit.
    if exclude_regex:
        pat = re.compile(exclude_regex)
        skipped = [k for k in keys if pat.search(k)]
        keys = [k for k in keys if not pat.search(k)]
        print(f"[simlingo] excluding {len(skipped)} objects matching {exclude_regex!r}", flush=True)

    tars = [k for k in keys if k.endswith(".tar.gz")]
    others = [k for k in keys if not k.endswith(".tar.gz")]
    print(f"[simlingo] mirroring {len(tars)} tarballs + {len(others)} other objects "
          f"from s3://{bucket}/{prefix}", flush=True)

    # tarballs contain top-level data/, dreamer/, commentary/, drivelm/ trees;
    # extracting them all into dest yields the sibling layout dataset_base expects.
    # Download -> extract -> delete keeps peak disk at extracted + 1 tarball.
    for i, key in enumerate(tars):
        name = key.rsplit("/", 1)[-1]
        local = workdir / name
        print(f"[simlingo] ({i + 1}/{len(tars)}) {name}", flush=True)
        s3.download_file(bucket, key, str(local))
        with tarfile.open(local, "r:gz") as tf:
            tf.extractall(dest)
        local.unlink()
    for key in others:
        s3.download_file(bucket, key, str(dest / key.rsplit("/", 1)[-1]))
    marker.touch()


def _wait_for_marker(workdir: Path, timeout_s: int = 3600) -> None:
    marker = workdir / ".extract_done"
    deadline = time.time() + timeout_s
    while not marker.exists():
        if time.time() > deadline:
            raise RuntimeError(f"timed out waiting for rank 0 to finish extracting ({marker})")
        time.sleep(20)


def _get_shared_run_name(workdir: Path, rank: int, timeout_s: int = 60) -> str:
    # config.py's wandb_name default is time.strftime(...) evaluated independently
    # per rank's own train.py process, and hydra.run.dir (-> the DeepSpeed
    # checkpoint dir) is derived from it -> without pinning this, ranks whose
    # launch is skewed by >=1s (near-guaranteed: ranks 1+ poll _wait_for_marker
    # every 20s) resolve DIFFERENT checkpoint directories and scatter shards.
    marker = workdir / ".run_name"
    if rank == 0:
        if not marker.exists():
            marker.write_text(time.strftime("%Y%m%d_%H%M%S"))
    else:
        deadline = time.time() + timeout_s
        while not marker.exists():
            if time.time() > deadline:
                raise RuntimeError(f"timed out waiting for rank 0 to write {marker}")
            time.sleep(1)
    return marker.read_text().strip()


def smoke_train(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    cfg = training_fn_config
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    workdir = Path(cfg.get("workdir", "/mnt/work/simlingo_smoke"))
    workdir.mkdir(parents=True, exist_ok=True)
    # keep the InternVL2 download off the small root disk
    os.environ.setdefault("HF_HOME", str(workdir / "hf_cache"))

    if rank == 0:
        _download_and_extract(cfg["s3_bucket"], cfg["s3_prefix"].rstrip("/") + "/", workdir,
                              exclude_regex=cfg.get("s3_exclude_regex"))
    else:
        # full-dataset extract (~650GB compressed) runs for hours; the 1h
        # default would kill ranks 1..7 before rank 0 finishes
        _wait_for_marker(workdir, timeout_s=int(cfg.get("extract_timeout_s", 3600)))

    # frozen driving checkpoint for the cycle learnability probe; the yaml's
    # hydra_overrides point train.py's `checkpoint=` at this local path
    ckpt_key = cfg.get("probe_ckpt_s3_key")
    if ckpt_key:
        ckpt_local = workdir / "probe_ckpt.pt"
        ckpt_marker = workdir / ".ckpt_done"
        if rank == 0:
            if not ckpt_marker.exists():
                print(f"[simlingo] downloading probe checkpoint s3://{cfg['s3_bucket']}/{ckpt_key}", flush=True)
                _s3_client().download_file(cfg["s3_bucket"], ckpt_key, str(ckpt_local))
                ckpt_marker.touch()
        else:
            deadline = time.time() + 1800
            while not ckpt_marker.exists():
                if time.time() > deadline:
                    raise RuntimeError(f"timed out waiting for rank 0 to download {ckpt_local}")
                time.sleep(10)

    repo_root = Path(__file__).resolve().parents[1]
    sim_root = repo_root / "simlingo" / "simlingo"
    # train.py resolves data_path relative to its cwd: database/simlingo -> node-local dir
    db_link = sim_root / "database"
    try:
        db_link.symlink_to(workdir / "database")
    except FileExistsError:
        pass

    # Each rank runs train.py as one "node" of a Lightning num_nodes=WORLD_SIZE
    # group (each Ray worker only sees its own GPU). Rendezvous on localhost at
    # a port distinct from Ray's; NODE_RANK gives Lightning the global rank and
    # RANK keeps rank_zero_only (e.g. the wandb logger) correct pre-init.
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{sim_root}:{env.get('PYTHONPATH', '')}"
    env.pop("WORLD_SIZE", None)
    env.pop("GROUP_RANK", None)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = MASTER_PORT
    env["NODE_RANK"] = str(rank)
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = "0"
    # Ray exposes ALL colocated GPUs to every worker; with LOCAL_RANK=0 each
    # subprocess would bind GPU 0 (8 ranks on one device -> NCCL 'invalid
    # usage'). Pin each subprocess to its own worker's GPU explicitly.
    parent_local_rank = int(os.environ.get("LOCAL_RANK", rank))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [g for g in visible.split(",") if g] or [str(i) for i in range(world_size)]
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids[parent_local_rank % len(gpu_ids)]

    run_name = _get_shared_run_name(workdir, rank)

    overrides = list(cfg.get("hydra_overrides", []))
    overrides += [f"gpus=1", f"num_nodes={world_size}", f"wandb_name={run_name}"]
    cmd = [sys.executable, "simlingo_training/train.py", *overrides]
    print(f"[simlingo] rank {rank}/{world_size} launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=sim_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode} on rank {rank}")
    print(f"[simlingo] rank {rank} training finished", flush=True)
