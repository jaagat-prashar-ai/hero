"""Lilypad entrypoint for SimLingo Bench2Drive evaluation.

Execution model: with cluster_resources.num_gpus=N, Lilypad invokes eval_b2d
in N worker processes on the node (RANK/WORLD_SIZE env, same as run.py).
Rank 0 downloads CARLA 0.9.15, the eval-code tarball (Bench2Drive/leaderboard/
scenario_runner/team_code are NOT shipped as code assets: their >20MB data
files would be silently dropped), and the consolidated checkpoint from S3 onto
the shared node-local workdir; other ranks wait on a marker. Each rank then
runs its static shard of the Bench2Drive routes (routes[rank::WORLD_SIZE])
sequentially: leaderboard_evaluator.py boots its own CARLA server per route
(-RenderOffScreen -graphicsadapter=<gpu>), results upload to S3 per route.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import boto3
import ujson
from botocore.config import Config

FAIL_STATUSES = (
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
)


def _s3_client():
    # OCI's S3-compat endpoint rejects s3transfer's chunked encoding
    # ("NotImplemented") -- same bug/fix as s3_checkpoint.py (BUGS.md 2026-07-01).
    # Callers must use put_object, never upload_file.
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


def _put_file(s3, bucket: str, key: str, path: Path) -> None:
    try:
        with open(path, "rb") as fh:
            s3.put_object(Bucket=bucket, Key=key, Body=fh.read())
    except Exception as e:  # an upload hiccup must not kill the rank's route loop
        print(f"[b2d-eval] WARN upload failed {key}: {e}", flush=True)


def _fetch_and_extract(s3, bucket: str, key: str, dest: Path, workdir: Path) -> None:
    local = workdir / key.rsplit("/", 1)[-1]
    print(f"[b2d-eval] fetching s3://{bucket}/{key}", flush=True)
    s3.download_file(bucket, key, str(local))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local, "r:gz") as tf:
        tf.extractall(dest)
    local.unlink()


def _rank0_setup(cfg: dict, workdir: Path, sim_root: Path) -> None:
    marker = workdir / ".setup_done"
    if marker.exists():
        print("[b2d-eval] setup marker exists, skipping", flush=True)
        return
    s3 = _s3_client()
    bucket = cfg["s3_bucket"]
    assets = cfg.get("assets_prefix", "simlingo-b2d-eval").rstrip("/")

    carla_root = workdir / "carla0915"
    if not (workdir / ".carla_done").exists():
        # AdditionalMaps shares CARLA's top-level layout; extracting both into
        # carla_root is what ImportAssets.sh would produce
        _fetch_and_extract(s3, bucket, f"{assets}/CARLA_0.9.15.tar.gz", carla_root, workdir)
        _fetch_and_extract(s3, bucket, f"{assets}/AdditionalMaps_0.9.15.tar.gz", carla_root, workdir)
        (workdir / ".carla_done").touch()

    # eval code lands inside the code-asset copy of simlingo/simlingo so all
    # relative imports/paths match the upstream repo layout
    _fetch_and_extract(s3, bucket, f"{assets}/eval_code.tar.gz", sim_root, workdir)

    session = cfg["checkpoint_session"]
    ckpt_prefix = cfg.get("checkpoint_prefix", "simlingo-checkpoints-consolidated").rstrip("/")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{ckpt_prefix}/{session}/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(ckpt_prefix) + 1:]
            local = workdir / "ckpts" / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"[b2d-eval] fetching {obj['Key']}", flush=True)
            s3.download_file(bucket, obj["Key"], str(local))
    marker.touch()


def _wait_for_marker(marker: Path, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while not marker.exists():
        if time.time() > deadline:
            raise RuntimeError(f"timed out waiting for rank 0 setup ({marker})")
        time.sleep(20)


def _route_done(result_file: Path) -> bool:
    """Ported from start_eval_simlingo.py's filter_completed: a route counts as
    done when its result JSON parses, progress is complete, and no record
    carries a failure status."""
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = ujson.load(f)
        progress = data["_checkpoint"]["progress"]
        if len(progress) < 2 or progress[0] < progress[1]:
            return False
        return all(r["status"] not in FAIL_STATUSES for r in data["_checkpoint"]["records"])
    except Exception:
        return False


def _kill_stray_carla(gpu: str) -> None:
    subprocess.run(
        "ps -ef | grep -- '-graphicsadapter=" + gpu + "' | grep -v grep | awk '{print $2}' | xargs -r kill -9",
        shell=True, check=False,
    )


def eval_b2d(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    cfg = training_fn_config
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    workdir = Path(cfg.get("workdir", "/mnt/work/simlingo_b2d_eval"))
    workdir.mkdir(parents=True, exist_ok=True)
    # InternVL2 processor/config downloads go to the big disk
    os.environ.setdefault("HF_HOME", str(workdir / "hf_cache"))

    repo_root = Path(__file__).resolve().parents[1]
    sim_root = repo_root / "simlingo" / "simlingo"

    if rank == 0:
        _rank0_setup(cfg, workdir, sim_root)
    else:
        _wait_for_marker(workdir / ".setup_done", timeout_s=int(cfg.get("setup_timeout_s", 5400)))
        # stagger model init so 8 ranks don't race the shared HF cache
        time.sleep(rank * 20)

    session = cfg["checkpoint_session"]
    epoch = cfg.get("checkpoint_epoch", "epoch=002")
    ckpt_path = workdir / "ckpts" / session / "checkpoints" / f"{epoch}.ckpt" / "pytorch_model.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint missing: {ckpt_path}")

    # GPU pinning, same scheme as run.py: Ray exposes all colocated GPUs to
    # every worker, so pin each rank to its own physical device
    parent_local_rank = int(os.environ.get("LOCAL_RANK", rank))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [g for g in visible.split(",") if g] or [str(i) for i in range(world_size)]
    phys_gpu = gpu_ids[parent_local_rank % len(gpu_ids)]

    env = dict(os.environ)
    env["CARLA_ROOT"] = str(workdir / "carla0915")
    env["SCENARIO_RUNNER_ROOT"] = str(sim_root / "Bench2Drive" / "scenario_runner")
    env["PYTHONPATH"] = ":".join([
        # pip `carla` provides the client lib but NOT the PythonAPI `agents.*`
        # package that scenario_runner imports (SLURM script exported this too)
        str(workdir / "carla0915" / "PythonAPI" / "carla"),
        str(sim_root),
        str(sim_root / "Bench2Drive" / "leaderboard"),
        str(sim_root / "Bench2Drive" / "scenario_runner"),
        str(sim_root / "team_code"),  # agent_simlingo does `import scenario_logger`
        env.get("PYTHONPATH", ""),
    ])
    env["CUDA_VISIBLE_DEVICES"] = phys_gpu

    route_dir = sim_root / "leaderboard" / "data" / "bench2drive_split"
    routes = sorted(route_dir.glob("*.xml"))
    if cfg.get("route_ids"):
        wanted = {str(r) for r in cfg["route_ids"]}
        routes = [r for r in routes if r.stem.split("_")[-1] in wanted]
    if cfg.get("max_routes"):
        routes = routes[: int(cfg["max_routes"])]
    my_routes = routes[rank::world_size]
    print(f"[b2d-eval] rank {rank}/{world_size} gpu {phys_gpu}: {len(my_routes)}/{len(routes)} routes", flush=True)

    eval_name = cfg["eval_name"]
    out_root = workdir / "results" / eval_name
    res_dir, out_dir, viz_root = out_root / "res", out_root / "out", out_root / "viz"
    for d in (res_dir, out_dir, viz_root):
        d.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()
    bucket = cfg["s3_bucket"]
    results_prefix = f"{cfg.get('results_prefix', 'simlingo-b2d-results').rstrip('/')}/{eval_name}"
    port = 10000 + int(phys_gpu) * 500
    tm_port = 30000 + int(phys_gpu) * 500

    n_ok = 0
    for route_xml in my_routes:
        route_id = route_xml.stem.split("_")[-1].zfill(3)
        result_file = res_dir / f"{route_id}_res.json"
        log_file = out_dir / f"{route_id}_out.log"

        for attempt in range(int(cfg.get("tries", 2))):
            if _route_done(result_file):
                break
            result_file.unlink(missing_ok=True)
            viz = viz_root / route_id
            shutil.rmtree(viz, ignore_errors=True)
            viz.mkdir(parents=True)
            env["SAVE_PATH"] = str(viz)
            cmd = [
                sys.executable, str(sim_root / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"),
                f"--routes={route_xml}", "--repetitions=1", "--track=SENSORS",
                f"--checkpoint={result_file}", "--timeout=600",
                f"--agent={sim_root / 'team_code' / 'agent_simlingo.py'}",
                f"--agent-config={ckpt_path}",
                f"--traffic-manager-seed={cfg.get('tm_seed', 1)}",
                f"--port={port}", f"--traffic-manager-port={tm_port}",
                f"--gpu-rank={phys_gpu}",
            ]
            print(f"[b2d-eval] rank {rank} route {route_id} attempt {attempt + 1}", flush=True)
            with open(log_file, "ab") as log:
                try:
                    subprocess.run(cmd, cwd=sim_root, env=env, stdout=log, stderr=log,
                                   timeout=int(cfg.get("route_timeout_s", 10800)))
                except subprocess.TimeoutExpired:
                    print(f"[b2d-eval] rank {rank} route {route_id} timed out", flush=True)
            _kill_stray_carla(phys_gpu)

        ok = _route_done(result_file)
        n_ok += ok
        if not ok and log_file.exists():
            tail = log_file.read_bytes()[-3000:].decode("utf-8", "replace")
            print(f"[b2d-eval] rank {rank} route {route_id} log tail:\n{tail}", flush=True)
        for local, sub in ((result_file, "res"), (log_file, "out")):
            if local.exists():
                _put_file(s3, bucket, f"{results_prefix}/{sub}/{local.name}", local)
        if cfg.get("upload_viz"):
            for p in sorted((viz_root / route_id).rglob("*")):
                if p.is_file():
                    _put_file(s3, bucket, f"{results_prefix}/viz/{route_id}/{p.relative_to(viz_root / route_id)}", p)
        print(f"[b2d-eval] rank {rank} route {route_id}: {'OK' if ok else 'FAILED'}", flush=True)

    print(f"[b2d-eval] rank {rank} finished: {n_ok}/{len(my_routes)} routes ok", flush=True)
    if n_ok < len(my_routes):
        raise RuntimeError(f"rank {rank}: {len(my_routes) - n_ok} routes failed after retries")
