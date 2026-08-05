"""Lilypad entrypoint for SimLingo training runs.

Downloads the SimLingo dataset tarballs from the S3 mirror
(s3://research-datasets-chicago/hf-datasets/RenzKa/simlingo/), extracts them
into the node-local working dir in the layout dataset_base.py expects
(database/simlingo/{data,dreamer,commentary,drivelm}/simlingo/...), symlinks
that into the shipped repo, and execs simlingo_training/train.py with the
hydra overrides from the workload config.
"""
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import boto3


def _s3_client():
    # AWS_ENDPOINT_URL_S3 / creds come from the workload runtime environment
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))


def _download_and_extract(bucket: str, prefix: str, workdir: Path) -> None:
    dest = workdir / "database" / "simlingo"
    marker = workdir / ".extract_done"
    if marker.exists():
        print(f"[simlingo] {marker} exists, skipping download/extract", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
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


def smoke_train(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    cfg = training_fn_config
    workdir = Path(cfg.get("workdir", "/mnt/work/simlingo_smoke"))
    workdir.mkdir(parents=True, exist_ok=True)
    # keep the InternVL2 download off the small root disk
    os.environ.setdefault("HF_HOME", str(workdir / "hf_cache"))

    _download_and_extract(cfg["s3_bucket"], cfg["s3_prefix"].rstrip("/") + "/", workdir)

    repo_root = Path(__file__).resolve().parents[1]
    sim_root = repo_root / "simlingo" / "simlingo"
    # train.py resolves data_path relative to its cwd: database/simlingo -> node-local dir
    db_link = sim_root / "database"
    if not db_link.is_symlink() and not db_link.exists():
        db_link.symlink_to(workdir / "database")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{sim_root}:{env.get('PYTHONPATH', '')}"
    cmd = [sys.executable, "simlingo_training/train.py", *cfg.get("hydra_overrides", [])]
    print(f"[simlingo] launching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=sim_root, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}")
    print("[simlingo] training finished", flush=True)
