# Lilypad entrypoint for the discrete-vs-diffusion sweep, fully self-contained
# on the cluster: samples 15 clips per event_cluster directly from S3 at job
# start (cluster_data.sample_and_resolve_clips, reusing
# masking.data.sample_clips's shard-scanning machinery -- no locally
# pre-built manifest or clip cache). Runs in Lilypad's BASE Python 3.10
# environment -- cannot import alpamayo_r1 directly (needs Python 3.12).
# Bootstraps an isolated Python 3.12 venv via `uv` (bootstrap_venv.py,
# matching this workstation's own ar1_venv setup), shards the resolved clip
# list by rank (same hashlib.md5(clip_id) % world_size convention
# masking/pref_pairs already use), writes this rank's shard to a small JSON
# file, then runs the actual per-clip work (perplexity/cluster_worker.py,
# including the actual file extraction for just this rank's clips) as a
# subprocess INSIDE that venv -- a clean process boundary since the parent
# (this function) and the child (cluster_worker.py) need different Python
# versions.
#
# Scan-once-broadcast: the S3 shard scan behind sample_and_resolve_clips is
# genuinely slow (confirmed directly -- didn't finish in 5 min locally,
# walks up to ~4200 shards with early-stop only once every cluster has
# enough candidates, and a couple of clusters are rare enough in what's
# actually uploaded to force a near-full scan). Doing this on all 8 ranks
# independently would multiply that cost 8x before any GPU inference even
# starts. Instead, only rank 0 runs the scan and writes the resolved list to
# a scratch S3 key (not /mnt/work -- confirmed in a prior project's cluster
# runs that this cluster's /mnt/work is not reliably shared/visible the same
# way across contexts); every other rank polls that same key until it
# appears, then reads it. The key is a deterministic function of
# (n_per_cluster, sample_seed) -- sampling is already deterministic given
# those, so reusing a stale key from an interrupted/requeued attempt with
# the SAME params is correct, not stale data.
#
# Scalar results are read straight off cluster_worker.py's stdout, which
# flows through to this pod's own log stream unmodified (capture_output is
# deliberately NOT set) -- so pref_pairs/fetch_from_logs.py's exact
# log-then-fetch retrieval pattern (content-filter on a fixed marker string)
# works for this project too, without needing its own separate fetch script
# yet.

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from collections import deque

import boto3
import botocore.exceptions

# Absolute dotted imports, matching masking/training/run.py's own convention
# (e.g. `from masking.masked_model import ...`) -- this module is loaded via
# the `perplexity.training.run` dotted training_fn path from the repo root,
# so perplexity's submodules resolve the same way masking's do.
from perplexity.cluster_data import sample_and_resolve_clips
from perplexity.training.bootstrap_venv import ensure_alpamayo_venv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLL_INTERVAL_S = 15
# How long ranks > 0 wait for rank 0's scan manifest. NOT 30 min: with
# n_per_cluster=15, the two rarest event clusters (~18-22 rows dataset-wide)
# can force the scan through most of the ~4200 shards -- ~2.5h at the
# ~0.5 shards/s observed on-cluster -- and a 30-min poll timeout would crash
# ranks 1-7 mid-scan and fail the whole sweep (the 1-GPU canaries never
# exercised this path). A requeued/repeat run reuses the cached manifest and
# returns immediately, so this ceiling only bites on the first sweep launch.
POLL_TIMEOUT_S = 4 * 3600


def _clip_rank(clip_id: str, world_size: int) -> int:
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % world_size


def _scan_manifest_key(n_per_cluster: int, sample_seed: int) -> str:
    # _v2: the t0_us/coc resolution semantics changed (parquet events column
    # instead of the WDS json blob -- see cluster_data.py), so manifests
    # written by the old code are wrong even when non-empty. Versioning the
    # key rather than deleting the old objects keeps this cache-safe without
    # any coordination: canary3's buggy 0-clip manifest still sits at the
    # unversioned key and must never be picked up by the reuse path below.
    return f"scratch/discrete_vs_diffusion/scan_manifest_v2_n{n_per_cluster}_seed{sample_seed}.json"


def _try_read_manifest(s3, bucket: str, key: str) -> list[dict] | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
            raise
        return None


def _wait_for_manifest(s3, bucket: str, key: str) -> list[dict]:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return json.loads(body)
        except botocore.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                raise
            time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"rank 0's scan result never appeared at s3://{bucket}/{key}")


def discrete_vs_diffusion_loop(training_fn_config: dict, experiment_tracker) -> None:
    n_per_cluster = training_fn_config.get("n_per_cluster", 15)
    sample_seed = training_fn_config.get("sample_seed", 42)
    s3_bucket = training_fn_config.get("s3_bucket", "research-datasets-chicago")
    venv_dir = training_fn_config.get("venv_dir", "/mnt/work/tmp/alpamayo_r1_venv")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    scan_key = _scan_manifest_key(n_per_cluster, sample_seed)
    s3 = boto3.client("s3")

    if rank == 0:
        # Reuse a manifest already written under the SAME (version,
        # n_per_cluster, seed) key before paying for a fresh scan. Sampling
        # is deterministic given those params, so a hit is exact, not stale.
        # This is what makes a preempted/requeued attempt cheap: canary3's
        # node loss forced a full RayJob restart that re-ran the ~20-min
        # shard scan from zero, and that idle CPU-only time is precisely
        # what got the workload killed by the idle-GPU reaper. An EMPTY
        # cached manifest is never reused -- 0 resolved clips means a prior
        # attempt's resolution failed (that's canary3's bug signature, and
        # with the bug fixed a legitimate scan finding literally nothing is
        # pathological enough to be worth re-checking), so rescan instead.
        all_entries = _try_read_manifest(s3, s3_bucket, scan_key)
        if all_entries:
            logger.info(
                "rank 0: reusing cached scan manifest s3://%s/%s (%d clips) -- skipping the scan",
                s3_bucket, scan_key, len(all_entries),
            )
        else:
            logger.info(
                "rank 0: running the S3 shard scan (%d clips/cluster, seed=%d) -- this is the "
                "slow step, expect several minutes to tens of minutes",
                n_per_cluster, sample_seed,
            )
            all_entries = sample_and_resolve_clips(
                n_per_cluster=n_per_cluster, seed=sample_seed, hf_token=os.environ.get("HF_TOKEN")
            )
            s3.put_object(
                Bucket=s3_bucket, Key=scan_key, Body=json.dumps(all_entries).encode("utf-8")
            )
            logger.info(
                "rank 0: resolved %d clips, wrote to s3://%s/%s for other ranks",
                len(all_entries), s3_bucket, scan_key,
            )
    else:
        logger.info(
            "rank %d: waiting for rank 0's scan result at s3://%s/%s ...",
            rank, s3_bucket, scan_key,
        )
        all_entries = _wait_for_manifest(s3, s3_bucket, scan_key)
        logger.info("rank %d: got %d resolved clips from rank 0", rank, len(all_entries))

    # Fail fast and loud on an empty resolution. canary3 resolved 0 clips
    # (see cluster_data.py's t0 fix) and then spent its remaining minutes
    # bootstrapping a venv it had nothing to feed -- the data bug was only
    # visible as a WARNING buried mid-log. 0 clips means the run cannot
    # produce anything: make it an EXPERIMENT_FAILED, not a quiet no-op.
    if not all_entries:
        raise RuntimeError(
            "resolved 0 clips across all event clusters -- check the t0_us/"
            "S3-availability filter (cluster_data.sample_and_resolve_clips)"
        )

    my_entries = [e for e in all_entries if _clip_rank(e["clip_id"], world_size) == rank]
    logger.info(
        "rank %d/%d: %d/%d clips assigned", rank, world_size, len(my_entries), len(all_entries)
    )

    perplexity_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(perplexity_dir)
    python_bin = ensure_alpamayo_venv(venv_dir, perplexity_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(my_entries, f)
        rank_manifest_path = f.name

    # cluster_worker.py itself uses perplexity/'s established flat-import
    # convention (matches dump_input_template.py/s3_clip_loader.py etc, run
    # with cwd=perplexity_dir), but cluster_data.py's own internal
    # `from masking.data.sample_clips import ...` needs the repo ROOT on
    # sys.path too -- cwd alone only puts perplexity_dir there.
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, perplexity_dir, child_env.get("PYTHONPATH", "")]
    )
    # Tee the worker's output: echo every line to this process's stdout
    # unmodified (so the DISCRETE_VS_DIFFUSION_CLIP_SUMMARY log-then-fetch
    # contract is unchanged) while keeping a rolling tail. canary7's
    # cluster_worker crash was invisible -- its final stderr lines (the
    # actual traceback) never reached OCI because the pod tore down before
    # the log shipper flushed them. Ray's own error propagation (the
    # exception raised here -> trial error.txt -> driver traceback) DID
    # ship reliably, so on failure the tail rides along in the exception
    # message.
    tail: deque[str] = deque(maxlen=80)
    proc = subprocess.Popen(
        [python_bin, "cluster_worker.py", "--manifest", rank_manifest_path, "--s3_bucket", s3_bucket],
        cwd=perplexity_dir,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    returncode = proc.wait()
    logger.info("rank %d: cluster_worker.py exited with code %d", rank, returncode)
    if returncode != 0:
        raise RuntimeError(
            f"cluster_worker.py failed on rank {rank} (exit {returncode}); "
            f"last {len(tail)} output lines:\n{''.join(tail)}"
        )
