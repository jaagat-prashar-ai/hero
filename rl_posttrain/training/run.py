# SPDX-License-Identifier: Apache-2.0
"""
run.py — Lilypad generic entrypoint for the alpamayo1_x_rl local-test RL
post-training run (Cosmos-RL + GRPO), vendored from NVlabs/alpamayo-recipes
at third_party/alpamayo-recipes.

This runs the recipe's own "Getting started" single-node local-test steps
end-to-end on one Lilypad a100.8 node (8 GPUs). workload_type: "generic"
runs entrypoint_fn directly on the Ray HEAD node/pod (Lilypad's
GenericWorkloadRunner just calls it synchronously, no GPU-aware scheduling)
-- and that head pod has NO GPU/driver passthrough in this cluster (confirmed
2026-07-16 on run alpamayo-rl-local-test-6kfkvp: `nvidia-smi` missing,
`libcuda.so.1` missing, vLLM/Triton both report zero GPUs). The real 8 GPUs
requested via cluster_resources go to a separate Ray WORKER pod instead. So
the actual work (venv bootstrap onward) is wrapped in a
`@ray.remote(num_gpus=...)` task and dispatched via `ray.get(...)` from the
plain entrypoint, so Ray schedules it onto that GPU worker pod. This does
NOT use Lilypad's per-rank training_fn replica convention (which would start
N independent copies, one per GPU) -- `cosmos-rl` itself fans out to all of
a single node's GPUs via its own controller/policy/rollout subprocess model,
so this needs to run as one process on one GPU-attached node, not N replicas.

  1. Bootstrap the recipe's isolated Python 3.12 uv venv (see
     bootstrap_venv.py) -- idempotent, persisted under workspace_dir so a
     preempted/requeued job doesn't redo the flash-attn build.
  2. Convert the released HF checkpoint into a training-ready checkpoint
     (scripts/convert_release_config_to_training.py) -- idempotent, skipped
     if the output dir already has a config.json.
  3. Download + curate a mini Physical AI (PAI) dataset subset
     (scripts/download_pai.py + scripts/curate_pai_samples.py) -- idempotent,
     skipped if the curated parquet already exists.
  4. Patch the recipe's local-test TOML (output_dir, model path, W&B
     naming) into a scratch copy -- never edits the vendored submodule.
  5. Launch `cosmos-rl --policy 1 --rollout 1 ...` as a streamed subprocess.

Full config reference (all keys optional, defaults shown):
    workspace_dir:      "/mnt/work/tmp/alpamayo_rl_job"  # persistent scratch root
    alpamayo_model:     "nvidia/Alpamayo-1.5-10B"        # or "nvidia/Alpamayo-R1-10B"
    pai_chunk_ids:      "3116"                           # download_pai.py --chunk-ids
    pai_num_samples:    16                                # clips curated for the mini subset
    pai_seed:           0
    dp_shard_size:      4       # [policy.parallelism].dp_shard_size (4 for 1 a100.8 node)
    train_epoch:         15      # [train].epoch
    wandb_project:      "alpamayo-rl"
    wandb_experiment:   "reasoning_vla_local_test"
    wandb_entity:       "research"
    reasoning:          false   # use the joint reasoning-motion reward variant
    reward_mode:        null    # null (derive from `reasoning`) | "motion" |
                                # "reasoning" | "llm_judge" (Anthropic-API
                                # trajectory-grounded judge; needs
                                # ANTHROPIC_API_KEY at submit time)
    num_reasoning_clips: 16     # reasoning/llm_judge modes: download_pai.py
                                # --num-reasoning-clips (dataset size)
    num_gpus:           8       # GPUs to request for the ray.remote GPU-node task
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import ray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "third_party" / "alpamayo-recipes"
RECIPE_DIR = RECIPE_ROOT / "recipes" / "alpamayo1_x_rl"

_DEFAULTS: dict[str, Any] = {
    "workspace_dir": "/mnt/work/tmp/alpamayo_rl_job",
    "alpamayo_model": "nvidia/Alpamayo-1.5-10B",
    "pai_chunk_ids": "3116",
    "pai_num_samples": 16,
    "pai_seed": 0,
    "dp_shard_size": 4,
    "train_epoch": 15,
    "wandb_project": "alpamayo-rl",
    "wandb_experiment": "reasoning_vla_local_test",
    "wandb_entity": "research",
    "reasoning": False,
    # Reward mode: null (derive from legacy `reasoning` bool: false->"motion",
    # true->"reasoning"), or explicitly one of "motion" | "reasoning" |
    # "llm_judge". "llm_judge" uses our own entry script + TOML template
    # (rl_posttrain/rewards/, rl_posttrain/toml/) that swap the recipe's
    # Lingo-Judge grader for the Anthropic-API trajectory-grounded judge --
    # requires ANTHROPIC_API_KEY in the environment at submit time.
    "reward_mode": None,
    # "llm_judge"/"reasoning" modes train on reasoning-bearing PAI clips
    # (download_pai.py --only-reasoning-chunks) instead of the pai_chunk_ids
    # motion subset; this sets --num-reasoning-clips.
    "num_reasoning_clips": 16,
    "num_gpus": 8,
}

_CAMERA_SUBPARTS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
_CALIBRATION_SUBPARTS = ["camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"]


def _run_streamed(cmd: list[str], **kwargs) -> None:
    """Run a subprocess with stdout/stderr flowing through to this pod's own
    log stream (not captured) so `lilypad workload logs` sees it live."""
    logger.info("run: %s", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _ensure_redis_server() -> None:
    """cosmos-rl's controller shells out to `redis-server` directly (return
    code 127 = command not found if missing) -- a system package, not
    something the recipe's own `uv sync` installs (the `redis` pip package
    it does pull in is just the Python client library). Confirmed missing on
    run alpamayo-rl-local-test-h13fcv (2026-07-16): `RuntimeError: Failed to
    start redis server ... with return code 127`. Install it once via apt if
    not already present (Lilypad's pod runs as root)."""
    import shutil

    if shutil.which("redis-server"):
        return
    logger.info("redis-server not found on PATH -- installing via apt")
    _run_streamed(["apt-get", "update"])
    _run_streamed(["apt-get", "install", "-y", "redis-server"])


def _convert_model(python_bin: str, model_dir: Path, alpamayo_model: str) -> None:
    if (model_dir / "config.json").exists():
        logger.info("model conversion: %s already has config.json, skipping", model_dir)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            python_bin,
            "scripts/convert_release_config_to_training.py",
            "--output-dir",
            str(model_dir),
            "--alpamayo-model",
            alpamayo_model,
        ],
        cwd=RECIPE_ROOT,
    )


def _download_pai(
    python_bin: str,
    pai_dir: Path,
    chunk_ids: str,
    num_samples: int,
) -> Path:
    mini_path = pai_dir / "clip_index_mini.parquet"
    if mini_path.exists():
        logger.info("PAI dataset: %s already exists, skipping download+curate", mini_path)
        return mini_path

    pai_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            python_bin,
            "scripts/download_pai.py",
            "--chunk-ids",
            chunk_ids,
            "--camera",
            *_CAMERA_SUBPARTS,
            "--calibration",
            *_CALIBRATION_SUBPARTS,
            "--labels",
            "egomotion",
            "--output-dir",
            str(pai_dir),
        ],
        cwd=RECIPE_ROOT,
    )
    _run_streamed(
        [
            python_bin,
            "scripts/curate_pai_samples.py",
            "--clip-index-path",
            str(pai_dir / "clip_index.parquet"),
            "--chunk",
            chunk_ids,
            "--num-samples",
            str(num_samples),
            "--output-path",
            str(mini_path),
        ],
        cwd=RECIPE_ROOT,
    )
    return mini_path


def _download_pai_reasoning(python_bin: str, pai_dir: Path, num_clips: int) -> Path:
    """Download the reasoning-bearing PAI subset used by the reasoning/
    llm_judge reward modes. Unlike _download_pai there is no separate curate
    step: `download_pai.py --only-reasoning-chunks --num-reasoning-clips N`
    both restricts the download to reasoning-annotated clips and writes the
    mini index (clip_index_reasoning_mini.parquet) the RL config reads --
    exact flag set from the recipe SKILL.md's "joint reward" section
    (`egomotion.offline`/`obstacle.offline` labels are required by the
    reasoning dataset's feature pipeline, not used by our reward directly).
    Idempotent: skipped when the mini index already exists (same convention
    as _download_pai)."""
    mini_path = pai_dir / "clip_index_reasoning_mini.parquet"
    if mini_path.exists():
        logger.info("PAI reasoning dataset: %s already exists, skipping download", mini_path)
        return mini_path

    pai_dir.mkdir(parents=True, exist_ok=True)
    _run_streamed(
        [
            python_bin,
            "scripts/download_pai.py",
            "--only-reasoning-chunks",
            "--num-reasoning-clips",
            str(num_clips),
            "--camera",
            *_CAMERA_SUBPARTS,
            "--calibration",
            *_CALIBRATION_SUBPARTS,
            "--labels",
            "egomotion",
            "egomotion.offline",
            "obstacle.offline",
            "--reasoning",
            "ood_reasoning.parquet",
            "--output-dir",
            str(pai_dir),
        ],
        cwd=RECIPE_ROOT,
    )
    return mini_path


def _patch_toml(
    template_path: Path,
    output_path: Path,
    *,
    output_dir: Path,
    model_dir: Path,
    dp_shard_size: int,
    epoch: int,
    wandb_project: str,
    wandb_experiment: str,
) -> None:
    import tomlkit

    doc = tomlkit.parse(template_path.read_text())
    doc["train"]["output_dir"] = str(output_dir)
    doc["train"]["epoch"] = epoch
    doc["policy"]["model_name_or_path"] = str(model_dir)
    doc["policy"]["parallelism"]["dp_shard_size"] = dp_shard_size
    doc["logging"]["project_name"] = wandb_project
    doc["logging"]["experiment_name"] = wandb_experiment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tomlkit.dumps(doc))


def _dump_latest_cosmos_logs(log_dir: Path, tail_lines: int = 200) -> None:
    """cosmos-rl writes controller.log/policy_<i>.log/rollout_<i>.log to disk
    under log_dir but only logs terse orchestration lines ("Process N exited
    with code 1") to its own stdout -- which is all Lilypad's job log
    captures. Dump the actual per-process tracebacks into our own log stream
    on failure so `lilypad workload logs` actually shows them."""
    run_dirs = sorted(
        (p for p in log_dir.glob("logs_*") if p.is_dir()), key=lambda p: p.stat().st_mtime
    )
    if not run_dirs:
        logger.warning("cosmos-rl failed but no per-process logs found under %s", log_dir)
        return
    latest = run_dirs[-1]
    for f in sorted(latest.glob("*.log")):
        text = f.read_text(errors="replace")
        tail = "\n".join(text.splitlines()[-tail_lines:])
        logger.error("===== tail of %s =====\n%s", f, tail)


def _resolve_reward_mode(cfg: dict[str, Any]) -> str:
    """Returns one of "motion" | "reasoning" | "llm_judge". The explicit
    reward_mode key wins; when unset (None), falls back to the legacy
    `reasoning` bool so pre-existing configs keep their exact behavior."""
    mode = cfg.get("reward_mode")
    if mode is None:
        return "reasoning" if bool(cfg["reasoning"]) else "motion"
    if mode not in ("motion", "reasoning", "llm_judge"):
        raise ValueError(f"reward_mode must be motion|reasoning|llm_judge, got {mode!r}")
    return mode


_SUMMARY_MARKERS = ("wandb:", "View run", "Run data is saved", "reward", "Reward", " step ", "Step ")


def _summarize_cosmos_logs(log_dir: Path) -> None:
    """On a successful run, print just the lines worth seeing (wandb run URL,
    reward/step progress) from the per-process logs rather than a full tail --
    confirmed missing entirely from our own captured stdout on run
    alpamayo-rl-local-test-b40s8k (2026-07-16), the first run to actually
    succeed: no reward numbers or wandb link ever showed up in
    `lilypad workload logs`, because policy_0.log/controller.log (where
    wandb.init() and per-step reward actually get logged) aren't tailed
    anywhere unless we do it ourselves."""
    run_dirs = sorted(
        (p for p in log_dir.glob("logs_*") if p.is_dir()), key=lambda p: p.stat().st_mtime
    )
    if not run_dirs:
        logger.warning("no per-process cosmos-rl logs found under %s", log_dir)
        return
    latest = run_dirs[-1]
    for f in sorted(latest.glob("*.log")):
        text = f.read_text(errors="replace")
        matches = [line for line in text.splitlines() if any(m in line for m in _SUMMARY_MARKERS)]
        if matches:
            logger.info("===== %s: reward/wandb lines =====\n%s", f, "\n".join(matches[-100:]))


# The cosmos-rl launcher logs this to its stdout when one of its
# controller/policy/rollout children dies, but then keeps waiting on the
# surviving processes forever instead of tearing the job down. Observed on
# canary alpamayo-rl-llm-judge-canary-xgo36t (2026-07-21): the rollout
# replica died 21 min in ("Process 1 failed with return code 1"), the policy
# GPUs then idled for 96 min until Lilypad's idle-GPU reaper killed the node
# (termination_reason IDLE_GPU), and the replica tracebacks were lost with
# the node-local /mnt/work. Watching for this marker lets us dump the
# per-process logs and fail fast while the files still exist.
_COSMOS_REPLICA_FAILURE_RE = re.compile(r"Process \d+ failed with return code \d+")


class _CosmosLogTailer(threading.Thread):
    """Periodically forwards NEW lines from cosmos-rl's per-process log files
    (controller.log/policy_<i>.log/rollout_<i>.log) into this pod's own log
    stream. Those files live on node-local /mnt/work and die with the node
    (confirmed by inspect-logs run alpamayo-rl-inspect-logs-774qca seeing an
    empty log dir), so shipping their lines to the captured stdout as they are
    written is the only way a replica traceback survives a hard node kill
    (e.g. the idle-GPU reaper). Capped per tick so a chatty vLLM startup
    can't flood OCI log ingestion; the cap drops the middle, not the tail,
    because tracebacks are what we're here for."""

    def __init__(self, log_dir: Path, interval_s: float = 30.0, max_lines_per_tick: int = 80):
        super().__init__(daemon=True, name="cosmos-log-tailer")
        self._log_dir = log_dir
        self._interval_s = interval_s
        self._max_lines = max_lines_per_tick
        self._offsets: dict[Path, int] = {}
        self._stop_event = threading.Event()
        self._poll_lock = threading.Lock()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._poll()

    def stop(self) -> None:
        """Signal the thread to stop, then do one final poll from the caller's
        thread so lines written right before a failure are always flushed."""
        self._stop_event.set()
        self._poll()

    def _poll(self) -> None:
        with self._poll_lock:
            for f in sorted(self._log_dir.glob("logs_*/*.log")):
                try:
                    size = f.stat().st_size
                    offset = self._offsets.get(f, 0)
                    if size <= offset:
                        continue
                    with open(f, "r", errors="replace") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                    self._offsets[f] = size
                except OSError:
                    continue
                lines = chunk.splitlines()
                if len(lines) > self._max_lines:
                    skipped = len(lines) - self._max_lines
                    lines = [f"... [{skipped} lines skipped] ..."] + lines[-self._max_lines :]
                logger.info("[tail %s]\n%s", f.name, "\n".join(lines))


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM (then SIGKILL) cosmos-rl's whole process group -- the launcher
    spawns its replicas via shell scripts, so killing just the launcher PID
    would orphan the GPU-holding children."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def _launch_cosmos_rl(
    python_bin: str,
    toml_path: Path,
    entry_script: Path,
    log_dir: Path,
    env: dict[str, str],
) -> None:
    cosmos_rl_bin = str(Path(python_bin).parent / "cosmos-rl")
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        cosmos_rl_bin,
        "--config",
        str(toml_path),
        "--policy",
        "1",
        "--rollout",
        "1",
        "--log-dir",
        str(log_dir),
        str(entry_script),
    ]
    logger.info("run: %s", " ".join(cmd))
    # Popen with captured+re-emitted stdout instead of _run_streamed: we must
    # SEE the launcher's output to catch the replica-failure marker (see
    # _COSMOS_REPLICA_FAILURE_RE) since the launcher does not exit on it.
    # start_new_session gives the launcher its own process group to kill.
    proc = subprocess.Popen(
        cmd,
        cwd=RECIPE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    tailer = _CosmosLogTailer(log_dir)
    tailer.start()
    failure_line: str | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if _COSMOS_REPLICA_FAILURE_RE.search(line):
                failure_line = line.strip()
                break
    except BaseException:
        # Don't leave a half-dead cosmos-rl holding GPUs if WE die (e.g.
        # Ray task cancellation) -- same idle-GPU-reaper trap as a replica
        # failure, just entered from the other side.
        tailer.stop()
        _terminate_process_group(proc)
        raise
    if failure_line is not None:
        logger.error(
            "cosmos-rl replica failure detected (%r) -- the launcher hangs "
            "instead of exiting here, so dumping per-process logs and "
            "terminating it ourselves",
            failure_line,
        )
        tailer.stop()
        _dump_latest_cosmos_logs(log_dir)
        _terminate_process_group(proc)
        raise RuntimeError(f"cosmos-rl replica failed: {failure_line}")
    returncode = proc.wait()
    tailer.stop()
    if returncode != 0:
        _dump_latest_cosmos_logs(log_dir)
        raise subprocess.CalledProcessError(returncode, cmd)
    _summarize_cosmos_logs(log_dir)


def _run_on_gpu_node(cfg: dict[str, Any]) -> None:
    """Runs on a Ray worker with real GPUs attached (see module docstring for why
    this can't just run inline in the plain `generic` entrypoint on the head)."""
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in the environment (gated PAI dataset + model)")

    workspace_dir = Path(cfg["workspace_dir"])
    venv_dir = workspace_dir / "venv"
    model_dir = workspace_dir / "alpamayo_model_converted_from_hf"
    pai_dir = workspace_dir / "PAI_mini"
    log_dir = workspace_dir / "logs"
    hf_home = workspace_dir / ".cache" / "huggingface"
    train_output_dir = workspace_dir / "outputs"
    # Scratch TOML name carries the reward mode so runs sharing the
    # persistent workspace_dir never clobber each other's patched config.
    toml_out = workspace_dir / "toml" / f"alpamayo_rvla_rl_{_resolve_reward_mode(cfg)}.local.toml"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    logger.info("rl_posttrain: workspace_dir=%s", workspace_dir)

    from rl_posttrain.training.bootstrap_venv import ensure_recipe_venv

    reward_mode = _resolve_reward_mode(cfg)

    # The LLM-judge reward calls the Anthropic API from inside the recipe
    # venv, which the vendored uv.lock knows nothing about -- installed as an
    # idempotent extra (see _ensure_extra_packages for why this must happen
    # even when the persistent venv already validates).
    extra_packages = ("anthropic",) if reward_mode == "llm_judge" else ()
    python_bin = ensure_recipe_venv(str(venv_dir), str(RECIPE_DIR), extra_packages=extra_packages)

    _convert_model(python_bin, model_dir, cfg["alpamayo_model"])

    # Dataset + TOML template + entry script per reward mode. "llm_judge"
    # trains on the same reasoning-bearing clips as "reasoning" (its reward
    # scores the decoded CoC, so rollout prompts must come from clips whose
    # data pipeline emits CoC sections) but swaps in our entry/TOML pair that
    # replaces the Lingo-Judge grader with the Anthropic-API judge.
    pai_reasoning_dir = workspace_dir / "PAI_Reasoning_mini"
    if reward_mode == "llm_judge":
        _download_pai_reasoning(python_bin, pai_reasoning_dir, int(cfg["num_reasoning_clips"]))
        template_path = REPO_ROOT / "rl_posttrain" / "toml" / "alpamayo_rvla_rl_llm_judge.toml"
        entry_script = REPO_ROOT / "rl_posttrain" / "rewards" / "llm_judge_entry.py"
    elif reward_mode == "reasoning":
        _download_pai_reasoning(python_bin, pai_reasoning_dir, int(cfg["num_reasoning_clips"]))
        template_path = RECIPE_DIR / "toml" / "alpamayo_rvla_rl_local_test_with_reasoning.toml"
        entry_script = (
            RECIPE_DIR / "models" / "reasoning_vla" / "alpamayo_cosmos_rl_post_training_reasoning_entry.py"
        )
    else:
        _download_pai(python_bin, pai_dir, str(cfg["pai_chunk_ids"]), int(cfg["pai_num_samples"]))
        template_path = RECIPE_DIR / "toml" / "alpamayo_rvla_rl_local_test.toml"
        entry_script = (
            RECIPE_DIR / "models" / "reasoning_vla" / "alpamayo_cosmos_rl_post_training_entry.py"
        )

    _patch_toml(
        template_path,
        toml_out,
        output_dir=train_output_dir,
        model_dir=model_dir,
        dp_shard_size=int(cfg["dp_shard_size"]),
        epoch=int(cfg["train_epoch"]),
        wandb_project=cfg["wandb_project"],
        wandb_experiment=cfg["wandb_experiment"],
    )

    _ensure_redis_server()

    venv_bin_dir = str(Path(python_bin).parent)
    subprocess_env = dict(os.environ)
    subprocess_env.update(
        {
            "ALPAMAYO_WORKSPACE": str(RECIPE_ROOT),
            "ALPAMAYO_MODEL_DIR": str(model_dir),
            "ALPAMAYO_PAI_LOCAL_DIR": str(pai_dir),
            "ALPAMAYO_LOG_DIR": str(log_dir),
            "HF_HOME": str(hf_home),
            "WANDB_ENTITY": cfg["wandb_entity"],
            # cosmos-rl's own launcher spawns launch_controller.sh/launch_replica.sh,
            # which invoke a bare `python` (not the venv's own bin/cosmos-rl absolute
            # path). Without prepending the venv's bin/ to PATH, that resolves to
            # whichever python is first on the inherited PATH -- confirmed via
            # controller.log on run alpamayo-rl-local-test-ze550o (2026-07-16):
            # `ModuleNotFoundError: No module named 'cosmos_rl'`, because the
            # controller ran under a python that isn't this venv's.
            "PATH": f"{venv_bin_dir}:{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(venv_dir),
        }
    )
    if hf_token:
        subprocess_env["HF_TOKEN"] = hf_token
    if wandb_key:
        subprocess_env["WANDB_API_KEY"] = wandb_key
    if reward_mode in ("reasoning", "llm_judge"):
        # Read by the reasoning/llm_judge entry scripts (mutually exclusive
        # with ALPAMAYO_PAI_LOCAL_DIR, which the motion entry reads).
        subprocess_env["ALPAMAYO_PAI_REASONING_LOCAL_DIR"] = str(pai_reasoning_dir)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if reward_mode == "llm_judge":
        if not anthropic_key:
            # Validated on the head node already; re-checked here because this
            # runs in a separate Ray task whose env is inherited independently.
            raise RuntimeError("ANTHROPIC_API_KEY is required for reward_mode=llm_judge")
        subprocess_env["ANTHROPIC_API_KEY"] = anthropic_key

    _launch_cosmos_rl(python_bin, toml_out, entry_script, log_dir, subprocess_env)

    logger.info("rl_posttrain: cosmos-rl run finished, checkpoints under %s", train_output_dir)


def rl_local_test_loop(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    """Lilypad-compatible generic entrypoint: rl_posttrain.training.run.rl_local_test_loop.

    Runs on the Ray head node (no GPUs here -- see module docstring). Does
    only cheap validation itself, then dispatches the actual work to a
    ray.remote task with num_gpus set so Ray schedules it onto the real
    GPU worker node.
    """
    cfg = {**_DEFAULTS, **training_fn_config}

    if not RECIPE_DIR.is_dir():
        raise RuntimeError(
            "Missing alpamayo-recipes vendor source at third_party/alpamayo-recipes -- run: "
            "git submodule update --init third_party/alpamayo-recipes"
        )
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required in the environment (gated PAI dataset + model)")
    if _resolve_reward_mode(cfg) == "llm_judge" and not os.environ.get("ANTHROPIC_API_KEY"):
        # Fail on the head node before any GPU worker (venv build, model
        # download) spends time -- the judge reward can't score a single
        # rollout without it.
        raise RuntimeError("ANTHROPIC_API_KEY is required for reward_mode=llm_judge")

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    remote_fn = ray.remote(_run_on_gpu_node).options(num_gpus=int(cfg["num_gpus"]))
    ray.get(remote_fn.remote(cfg))


def inspect_logs_loop(training_fn_config: dict[str, Any], experiment_tracker: Any = None) -> None:
    """Lilypad-compatible generic entrypoint: rl_posttrain.training.run.inspect_logs_loop.

    GPU-free companion to rl_local_test_loop: reads the per-process cosmos-rl
    logs already written to workspace_dir/logs by a prior run and prints their
    reward/wandb-link lines.

    CAVEAT (learned 2026-07-21 via alpamayo-rl-inspect-logs-774qca): /mnt/work
    is node-local, NOT a filesystem shared across workloads -- this entrypoint
    saw an empty log dir while chasing the llm-judge canary crash, and that
    canary itself re-bootstrapped its venv from scratch for the same reason.
    So this only works if it happens to land on the same node as the run it
    inspects. The reliable path is the live tailing/dumping that
    _launch_cosmos_rl now does in-run; keep this as a best-effort fallback.
    """
    cfg = {**_DEFAULTS, **training_fn_config}
    log_dir = Path(cfg["workspace_dir"]) / "logs"
    run_dirs = sorted((p for p in log_dir.glob("logs_*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        logger.warning("no per-process cosmos-rl logs found under %s", log_dir)
        return
    for run_dir in run_dirs:
        logger.info("===== run dir: %s =====", run_dir)
        for f in sorted(run_dir.glob("*.log")):
            text = f.read_text(errors="replace")
            lines = text.splitlines()
            logger.info(
                "----- %s (%d lines) -----\n%s", f, len(lines), "\n".join(lines[-300:])
            )
