#!/usr/bin/env python3
"""Launch hero Lilypad workloads via the Python SDK.

Reads a workload YAML, patches repo paths, applies optional overrides, and
submits with ``LaunchWorkload(WorkloadConfig(...))``.

Usage:
    python lilypad/launch.py masking/configs/cluster.yaml --dry-run
    python lilypad/launch.py build_wds/configs/cluster.yaml -n my-run \\
        -o workload_variant_config.entrypoint_fn_config.max_clips 10
    bash masking/configs/launch.sh masking --watch
    bash build_wds/configs/launch.sh build-wds --watch
"""
from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
ALPAMAYO1_5_PKG = REPO_ROOT / "third_party" / "alpamayo1.5" / "src" / "alpamayo1_5"

_LILYPAD_CRED_FILE = Path.home() / ".creds" / "lilypad.env"
_EXPORT_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
_OCI_CHECKSUM_ENV = (
    "AWS_REQUEST_CHECKSUM_CALCULATION",
    "AWS_RESPONSE_CHECKSUM_VALIDATION",
)

# Lilypad workers pre-install these; pinning them in pip overlay causes conflicts.
_FORBIDDEN_REQUIREMENT_PREFIXES = ("numpy==", "ray==", "wandb==")


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_lilypad_creds() -> None:
    """Load Lilypad creds from ~/.creds/lilypad.env if unset."""
    import os

    if not _LILYPAD_CRED_FILE.is_file():
        return

    wanted = {"HF_TOKEN", "WANDB_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
    missing = {k for k in wanted if not os.environ.get(k)}
    if not missing:
        return

    for line in _LILYPAD_CRED_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _EXPORT_RE.match(stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key in missing and not os.environ.get(key):
            os.environ[key] = value
            missing.discard(key)
        if not missing:
            return


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _patch_paths(cfg: dict[str, Any], config_path: pathlib.Path) -> None:
    """Resolve code_assets paths relative to the repo root."""
    try:
        code_assets = cfg["runtime_environment"]["code_assets"]
    except (KeyError, TypeError):
        return

    root = code_assets.get("root_directory", ".")
    root_path = pathlib.Path(str(root))
    if root_path.is_absolute():
        resolved_root = str(root_path.resolve())
    elif str(root) in (".", "./"):
        resolved_root = str(REPO_ROOT)
    else:
        resolved_root = str((config_path.parent / root_path).resolve())
    if resolved_root != str(root):
        print(f"  [patched] root_directory: {root} -> {resolved_root}")
    code_assets["root_directory"] = resolved_root

    req = code_assets.get("pip_requirements_path")
    if req:
        req_path = pathlib.Path(req)
        if not req_path.is_absolute():
            req_path = (REPO_ROOT / req_path).resolve()
        resolved_req = str(req_path)
        if resolved_req != req:
            print(f"  [patched] pip_requirements_path: {req} -> {resolved_req}")
        code_assets["pip_requirements_path"] = resolved_req


def _requirements_path(cfg: dict[str, Any]) -> pathlib.Path | None:
    code_assets = cfg.get("runtime_environment", {}).get("code_assets", {})
    if not isinstance(code_assets, dict):
        return None
    if code_assets.get("docker_image"):
        return None
    for key in ("pip_requirements_path", "uv_requirements_path"):
        raw = code_assets.get(key)
        if raw:
            path = pathlib.Path(str(raw))
            return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return None


def _is_build_wds_config(cfg: dict[str, Any]) -> bool:
    wvc = cfg.get("workload_variant_config", {})
    if not isinstance(wvc, dict):
        return False
    entrypoint = str(wvc.get("entrypoint_fn", ""))
    if entrypoint.startswith("build_wds."):
        return True
    req = _requirements_path(cfg)
    return req is not None and "build_wds" in req.parts


def _is_build_wds_requirements(req_path: pathlib.Path) -> bool:
    return "build_wds" in req_path.parts


def _validate_requirements_static(req_path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) from local requirements.txt policy checks."""
    errors: list[str] = []
    warnings: list[str] = []

    lines = req_path.read_text(encoding="utf-8").splitlines()
    has_torch_pin = False
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("torch==2.7.1"):
            has_torch_pin = True
        if any(stripped.startswith(prefix) for prefix in _FORBIDDEN_REQUIREMENT_PREFIXES):
            errors.append(
                f"{req_path.name}:{lineno}: {stripped!r} — do not pin numpy/ray/wandb "
                "(pre-installed on Lilypad workers; conflicts with lilypad-py)"
            )
        if stripped.startswith("torch==") and stripped != "torch==2.7.1":
            errors.append(
                f"{req_path.name}:{lineno}: {stripped!r} must be torch==2.7.1 "
                "(Lilypad pip overlay; base image torch 2.1.1 is too old for transformers>=4.57)"
            )
        elif stripped.startswith("torch") and not stripped.startswith("torch==2.7.1"):
            errors.append(
                f"{req_path.name}:{lineno}: pin torch as torch==2.7.1, not {stripped!r}"
            )
        if re.match(r"^physical[_-]?ai[_-]?av\b", stripped, re.IGNORECASE):
            if not _is_build_wds_requirements(req_path):
                errors.append(
                    f"{req_path.name}:{lineno}: physical_ai_av belongs in "
                    "build_wds/requirements.txt only (requires Python >=3.11)"
                )

    if not _is_build_wds_requirements(req_path) and not has_torch_pin:
        errors.append(
            f"{req_path.name} must pin torch==2.7.1 for Lilypad inference workloads"
        )

    return errors, warnings


def _validate_requirements_resolve(req_path: pathlib.Path) -> None:
    """Run Lilypad's uv pip compile check (same as submit-time validation)."""
    from lilypad.public.sdk_py import dependency_validator

    lilypad_version = importlib.metadata.version("lilypad-py")
    dependency_validator.build_workload_requirements_lockfile(req_path, lilypad_version)


def _validate_dependencies(
    cfg: dict[str, Any],
    *,
    skip_resolve: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate pip overlay requirements before submit."""
    errors: list[str] = []
    warnings: list[str] = []

    req_path = _requirements_path(cfg)
    if req_path is None:
        return errors, warnings

    if not req_path.is_file():
        errors.append(f"pip_requirements_path not found: {req_path}")
        return errors, warnings

    static_errors, static_warnings = _validate_requirements_static(req_path)
    errors.extend(static_errors)
    warnings.extend(static_warnings)

    build_wds = _is_build_wds_config(cfg)
    if build_wds:
        runtime = cfg.get("runtime_environment", {})
        const_env = runtime.get("constant_environment_variables", {}) if isinstance(runtime, dict) else {}
        if const_env.get("PIP_IGNORE_REQUIRES_PYTHON") != "1":
            warnings.append(
                "build_wds config should set runtime_environment.constant_environment_variables."
                "PIP_IGNORE_REQUIRES_PYTHON=1 (physical_ai_av requires Python >=3.11 on 3.10 workers)"
            )
        warnings.append(
            f"Skipping uv dependency resolve for {req_path.name} "
            "(physical_ai_av is installed with PIP_IGNORE_REQUIRES_PYTHON on cluster)"
        )
        return errors, warnings

    if skip_resolve:
        warnings.append("Skipping uv dependency resolve (--skip-dependency-validation)")
        return errors, warnings

    try:
        _validate_requirements_resolve(req_path)
        print(f"  [deps] uv resolve OK: {req_path.relative_to(REPO_ROOT)}")
    except ImportError:
        warnings.append(
            "lilypad dependency validator unavailable — install lilypad-py or use the "
            "lilypad-tools venv Python"
        )
    except Exception as exc:
        errors.append(
            f"Dependency resolution failed for {req_path.name} "
            f"(same check Lilypad runs at submit):\n{exc}"
        )

    return errors, warnings


def _validate_alpamayo_vendor() -> list[str]:
    errors: list[str] = []
    if not ALPAMAYO1_5_PKG.is_dir():
        errors.append(
            "Missing alpamayo1_5 vendor source at third_party/alpamayo1.5 — run: "
            "git submodule update --init third_party/alpamayo1.5"
        )
    return errors


def _validate_oci_checksum_env(cfg: dict[str, Any]) -> list[str]:
    """Warn when OCI S3 upload/download checksum env vars are missing."""
    warnings: list[str] = []
    if not _is_build_wds_config(cfg):
        return warnings

    runtime = cfg.get("runtime_environment", {})
    const_env = runtime.get("constant_environment_variables", {}) if isinstance(runtime, dict) else {}
    for key in _OCI_CHECKSUM_ENV:
        if const_env.get(key) != "when_required":
            warnings.append(
                f"build_wds config should set runtime_environment.constant_environment_variables."
                f"{key}=when_required (required for OCI S3 uploads; see ~/knowledge/gotchas/s3-oci-advanced.md)"
            )
    return warnings


def _validate_aws_required_env(cfg: dict[str, Any]) -> list[str]:
    """Warn when AWS keys are listed as submitter-required (Lilypad injects on cluster)."""
    warnings: list[str] = []
    req_env = cfg.get("runtime_environment", {}).get("required_environment_variables", [])
    if not isinstance(req_env, list):
        return warnings
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if var in req_env:
            warnings.append(
                f"Remove {var} from required_environment_variables — Lilypad injects S3 "
                "credentials on cluster workers; submitter creds come from ~/.creds/lilypad.env"
            )
    return warnings


def _entrypoint_fn_config(cfg: dict[str, Any]) -> dict[str, Any]:
    wvc = cfg.get("workload_variant_config", {})
    if not isinstance(wvc, dict):
        return {}
    fn_cfg = wvc.get("entrypoint_fn_config") or wvc.get("training_fn_config") or {}
    return fn_cfg if isinstance(fn_cfg, dict) else {}


def _validate_ffmpeg_av1(cfg: dict[str, Any]) -> list[str]:
    """Warn when build_wds AV1 transcoding is enabled but ffmpeg lacks an AV1 encoder."""
    warnings: list[str] = []
    if not _is_build_wds_config(cfg):
        return warnings

    fn_cfg = _entrypoint_fn_config(cfg)
    if str(fn_cfg.get("video_codec", "av1")) == "copy":
        return warnings

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        warnings.append(
            "ffmpeg not on PATH locally — cluster head must provide ffmpeg with "
            "libsvtav1 or libaom-av1 for video_codec=av1"
        )
        return warnings

    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        warnings.append(f"ffmpeg -encoders failed locally: {exc.stderr.strip()}")
        return warnings

    encoders = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith("."):
            encoders.add(parts[1])

    if "libsvtav1" in encoders:
        print("  [ffmpeg] libsvtav1 available locally")
    elif "libaom-av1" in encoders:
        print("  [ffmpeg] libaom-av1 available locally (libsvtav1 not found)")
    else:
        warnings.append(
            "ffmpeg found locally but no AV1 encoder (libsvtav1/libaom-av1) — "
            "cluster head must provide one for video_codec=av1"
        )
    return warnings


def _validate_hf_token(cfg: dict[str, Any], *, dry_run: bool) -> list[str]:
    import os

    wvc = cfg.get("workload_variant_config", {})
    fn_cfg = wvc.get("training_fn_config") or wvc.get("entrypoint_fn_config") or {}
    checkpoint = str(fn_cfg.get("checkpoint", ""))
    needs_hf = "/" in checkpoint or fn_cfg.get("hf_token") == ""
    if not needs_hf and not _is_build_wds_config(cfg):
        return []

    req_env = cfg.get("runtime_environment", {}).get("required_environment_variables", [])
    if isinstance(req_env, list) and "HF_TOKEN" in req_env and not os.environ.get("HF_TOKEN"):
        msg = (
            "HF_TOKEN is not set — required for HuggingFace model/dataset access. "
            "Add it to ~/.creds/lilypad.env before launching."
        )
        if dry_run:
            print(f"  [warn] {msg}")
            return []
        return [msg]
    return []


def _preflight(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    skip_dependency_validation: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    req_env = cfg.get("runtime_environment", {}).get("required_environment_variables", [])
    if isinstance(req_env, list):
        import os

        for var in req_env:
            if not os.environ.get(var):
                msg = f"{var} is not set in the environment"
                if dry_run:
                    print(f"  [warn] {msg}")
                else:
                    errors.append(msg)

    code_assets = cfg.get("runtime_environment", {}).get("code_assets", {})
    root = code_assets.get("root_directory", "")
    if root and not pathlib.Path(root).exists():
        errors.append(f"root_directory does not exist: {root}")

    req_path = code_assets.get("pip_requirements_path", "")
    if req_path and not pathlib.Path(req_path).is_file():
        errors.append(f"pip_requirements_path not found: {req_path}")

    dep_errors, dep_warnings = _validate_dependencies(
        cfg, skip_resolve=skip_dependency_validation,
    )
    errors.extend(dep_errors)
    warnings.extend(dep_warnings)

    if not _is_build_wds_config(cfg):
        errors.extend(_validate_alpamayo_vendor())

    errors.extend(_validate_hf_token(cfg, dry_run=dry_run))
    warnings.extend(_validate_oci_checksum_env(cfg))
    warnings.extend(_validate_aws_required_env(cfg))
    warnings.extend(_validate_ffmpeg_av1(cfg))

    wvc = cfg.get("workload_variant_config", {})
    if isinstance(wvc, dict):
        fn_cfg = wvc.get("entrypoint_fn_config") or wvc.get("training_fn_config") or {}
        if isinstance(fn_cfg, dict) and not fn_cfg.get("hf_token"):
            import os

            if fn_cfg.get("hf_token") == "" and not os.environ.get("HF_TOKEN"):
                # Only warn-level for empty hf_token — build_wds launch.sh injects it.
                pass

    return errors, warnings


def _summary(cfg: dict[str, Any], config_path: pathlib.Path) -> None:
    res = cfg.get("cluster_resources", {})
    wvc = cfg.get("workload_variant_config", {})
    fn_cfg = wvc.get("training_fn_config") or wvc.get("entrypoint_fn_config") or {}
    code = cfg.get("runtime_environment", {}).get("code_assets", {})
    fn_name = wvc.get("training_fn") or wvc.get("entrypoint_fn", "?")

    print(f"config:     {config_path.name}")
    print(f"run_name:   {cfg.get('name', '(unnamed)')}")
    print(f"entrypoint: {fn_name}")
    if res.get("num_gpus"):
        print(f"gpus:       {res.get('num_gpus')} x {res.get('gpu_machine_type', '?')}")
    if res.get("num_cpu_nodes"):
        print(f"cpu_nodes:  {res.get('num_cpu_nodes')}")
    print(f"region:     {res.get('allowed_regions', ['?'])[0]}")
    print(f"preempt:    {res.get('preemptible', '?')}")
    print(f"requeue:    {res.get('requeue_if_preempted', '?')}")
    if code.get("root_directory"):
        print(f"root_dir:   {code['root_directory']}")
    if fn_cfg.get("experiment"):
        print(f"experiment: {fn_cfg.get('experiment')}")
    if fn_cfg.get("rank") is not None:
        print(f"shard:      rank={fn_cfg.get('rank')} world_size={fn_cfg.get('world_size')}")


def _submit(cfg: dict[str, Any]) -> str:
    from lilypad.public.schemas.workload_config import WorkloadConfig
    from lilypad.public.sdk_py.lilypad_sdk import LaunchWorkload

    config = WorkloadConfig.model_validate(cfg)
    workload_id = LaunchWorkload(config)
    print(f"Workload launched: {workload_id}")
    return workload_id


def _watch(workload_id: str) -> None:
    subprocess.run(["lilypad", "watch", workload_id], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a hero Lilypad workload")
    parser.add_argument(
        "config",
        help="Workload YAML (e.g. masking/configs/cluster.yaml or build_wds/configs/cluster.yaml)",
    )
    parser.add_argument("-n", "--run-name", help="Override workload name")
    parser.add_argument(
        "-o",
        "--override",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        default=[],
        help="Dot-path override, e.g. workload_variant_config.training_fn_config.experiment b",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary only")
    parser.add_argument(
        "--skip-dependency-validation",
        action="store_true",
        help="Skip uv pip compile preflight (not recommended)",
    )
    parser.add_argument("--watch", action="store_true", help="Run lilypad watch after submit")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(config_path)

    _load_lilypad_creds()

    if args.run_name:
        cfg["name"] = args.run_name

    for key, value in args.override:
        _set_nested(cfg, key, _coerce_value(value))
        print(f"  [override] {key} = {_coerce_value(value)!r}")

    _patch_paths(cfg, config_path)

    errors, warnings = _preflight(
        cfg,
        dry_run=args.dry_run,
        skip_dependency_validation=args.skip_dependency_validation,
    )
    for warn in warnings:
        print(f"  [warn] {warn}")
    for err in errors:
        print(f"  [error] {err}", file=sys.stderr)

    _summary(cfg, config_path)

    if args.dry_run:
        if errors:
            print("\nDry run FAILED.", file=sys.stderr)
            sys.exit(1)
        try:
            from lilypad.public.schemas.workload_config import WorkloadConfig

            WorkloadConfig.model_validate(cfg)
            print("SDK validation: WorkloadConfig OK")
        except ImportError:
            print("SDK validation: skipped (lilypad not installed locally)")
        except Exception as exc:
            print(f"SDK validation FAILED: {exc}", file=sys.stderr)
            sys.exit(1)
        print("\nDry run OK — not submitting.")
        return

    if errors:
        sys.exit(1)

    print("Submitting...")
    workload_id = _submit(cfg)
    print(f"Monitor: lilypad watch {workload_id}")
    print(f"Logs:    oci-logs {workload_id}")

    if args.watch:
        _watch(workload_id)


if __name__ == "__main__":
    main()
