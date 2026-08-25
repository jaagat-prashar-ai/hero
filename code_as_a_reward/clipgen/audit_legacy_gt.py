# SPDX-License-Identifier: Apache-2.0
"""Quarantined own-GT audit for legacy ClipGen reward functions.

This tool never publishes or activates a legacy function. It evaluates each
legacy source against the current intact NVIDIA CoC/action pair and the exact
current GT-derived perturbation battery, then reports both the simple
POS_MIN/delta conditions and the stricter full gate result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen.run_prototype import _load_clip
from code_as_a_reward.clipgen.target_contract import (
    derive_target_contract,
    validate_gt_target,
)


def _load_manifests(directories: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for raw in directories:
        directory = Path(raw)
        manifest = json.loads((directory / "manifest.json").read_text())
        targets_path = directory / "targets.json"
        targets = (
            {str(row["clip_id"]): int(row["t0_us"]) for row in json.loads(targets_path.read_text())}
            if targets_path.exists()
            else {}
        )
        for original in manifest:
            entry = dict(original)
            clip_id = str(entry["clip_id"])
            if clip_id in entries:
                raise ValueError(f"duplicate manifest clip {clip_id}")
            if "t0_us" not in entry:
                if clip_id not in targets:
                    raise ValueError(f"clip {clip_id}: missing t0_us")
                entry["t0_us"] = targets[clip_id]
            for key in (
                "obstacle_parquet",
                "egomotion_parquet",
                "gt_coc",
                "overlay_jpeg",
                "waypoints_npy",
            ):
                if entry.get(key) and not Path(entry[key]).is_absolute():
                    entry[key] = str(directory / entry[key])
            entries[clip_id] = entry
    return entries


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def stage_s3_overlap(
    reward_fns_dir: str,
    staging_dir: str,
    *,
    bucket: str,
    prefix_format: str,
    n_shards: int,
    aws_profile: str | None,
    endpoint_url: str | None,
) -> list[str]:
    """Download metadata plus GT files only for legacy/function overlap."""

    import boto3

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3", endpoint_url=endpoint_url)
    source_ids = {path.stem for path in Path(reward_fns_dir).glob("*.py")}
    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)
    directories: list[str] = []
    downloads: list[tuple[str, Path]] = []
    path_keys = (
        "obstacle_parquet",
        "egomotion_parquet",
        "gt_coc",
        "overlay_jpeg",
        "waypoints_npy",
    )
    for shard in range(n_shards):
        prefix = prefix_format.format(shard=shard).strip("/")
        directory = root / f"shard_{shard}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read()
        )
        targets = json.loads(
            s3.get_object(Bucket=bucket, Key=f"{prefix}/targets.json")["Body"].read()
        )
        kept = [row for row in manifest if str(row["clip_id"]) in source_ids]
        kept_ids = {str(row["clip_id"]) for row in kept}
        kept_targets = [row for row in targets if str(row["clip_id"]) in kept_ids]
        (directory / "manifest.json").write_text(json.dumps(kept))
        (directory / "targets.json").write_text(json.dumps(kept_targets))
        directories.append(str(directory))
        for row in kept:
            for field in path_keys:
                value = row.get(field)
                if not value:
                    continue
                name = Path(value).name
                downloads.append((f"{prefix}/{name}", directory / name))

    def download(item: tuple[str, Path]) -> None:
        key, target = item
        s3.download_file(bucket, key, str(target))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, downloads))
    return directories


def audit(
    reward_fns_dir: str,
    manifest_dirs: list[str],
    out_json: str,
) -> dict[str, Any]:
    sources = {path.stem: path for path in Path(reward_fns_dir).glob("*.py")}
    manifests = _load_manifests(manifest_dirs)
    records: list[dict[str, Any]] = []

    for clip_id, path in sorted(sources.items()):
        entry = manifests.get(clip_id)
        if entry is None:
            records.append(
                {
                    "clip_id": clip_id,
                    "manifest_available": False,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            continue
        record: dict[str, Any] = {
            "clip_id": clip_id,
            "manifest_available": True,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "own_gt_positive_passed": False,
            "own_gt_delta_passed": False,
            "full_gate_passed": False,
        }
        try:
            clip = _load_clip(entry, None)
            contract = derive_target_contract(clip["gt_claims"], clip["gt_traj"])
            target_failures = validate_gt_target(contract, clip["gt_traj"])
            cases = gate_mod.build_perturbations(
                clip_id,
                clip["gt_claims"],
                np.asarray(clip["gt_waypoints"], dtype=np.float64),
                clip["hz"],
                tag="legacy_own_gt_audit",
                reward_spec=None,
            )
            result = gate_mod.run_gate(path.read_text(), cases)
            delta = float(result.pos_score - result.max_pert)
            positive_passed = bool(
                math.isfinite(result.pos_score) and result.pos_score >= gate_mod.POS_MIN
            )
            delta_passed = bool(
                positive_passed
                and math.isfinite(delta)
                and delta >= gate_mod.MIN_DROP - 1e-9
            )
            record.update(
                {
                    "valid_gt_target": not target_failures,
                    "gt_target_failures": target_failures,
                    "pos_score": result.pos_score,
                    "max_pert": result.max_pert,
                    "delta": delta,
                    "scores": result.scores,
                    "own_gt_positive_passed": positive_passed,
                    "own_gt_delta_passed": delta_passed,
                    "full_gate_passed": bool(result.passed),
                    "full_gate_failures": result.failures,
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    evaluable = [row for row in records if row["manifest_available"]]
    valid = [row for row in evaluable if row.get("valid_gt_target") is True]

    def counts(rows: list[dict[str, Any]]) -> dict[str, int | float]:
        positive = sum(bool(row.get("own_gt_positive_passed")) for row in rows)
        delta = sum(bool(row.get("own_gt_delta_passed")) for row in rows)
        full = sum(bool(row.get("full_gate_passed")) for row in rows)
        total = len(rows)
        return {
            "total": total,
            "own_gt_positive_passed": positive,
            "own_gt_positive_rate": _rate(positive, total),
            "own_gt_delta_passed": delta,
            "own_gt_delta_rate": _rate(delta, total),
            "full_gate_passed": full,
            "full_gate_rate": _rate(full, total),
        }

    report = {
        "schema_version": "clipgen.legacy-own-gt-audit.v1",
        "thresholds": {"pos_min": gate_mod.POS_MIN, "min_delta": gate_mod.MIN_DROP},
        "n_legacy_functions": len(sources),
        "n_manifest_clips": len(manifests),
        "n_evaluable_overlap": len(evaluable),
        "n_missing_manifest": len(sources) - len(evaluable),
        "all_evaluable": counts(evaluable),
        "valid_gt_only": counts(valid),
        "records": records,
    }
    Path(out_json).write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reward_fns_dir")
    parser.add_argument("out_json")
    parser.add_argument("manifest_dirs", nargs="*")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix-format")
    parser.add_argument("--s3-shards", type=int, default=25)
    parser.add_argument("--s3-staging-dir")
    parser.add_argument("--aws-profile")
    parser.add_argument("--endpoint-url")
    args = parser.parse_args()
    manifest_dirs = list(args.manifest_dirs)
    if args.s3_bucket or args.s3_prefix_format or args.s3_staging_dir:
        if not (args.s3_bucket and args.s3_prefix_format and args.s3_staging_dir):
            parser.error(
                "--s3-bucket, --s3-prefix-format, and --s3-staging-dir are required together"
            )
        if manifest_dirs:
            parser.error("provide local manifest_dirs or S3 staging options, not both")
        manifest_dirs = stage_s3_overlap(
            args.reward_fns_dir,
            args.s3_staging_dir,
            bucket=args.s3_bucket,
            prefix_format=args.s3_prefix_format,
            n_shards=args.s3_shards,
            aws_profile=args.aws_profile,
            endpoint_url=args.endpoint_url,
        )
    if not manifest_dirs:
        parser.error("no manifest directories provided")
    report = audit(args.reward_fns_dir, manifest_dirs, args.out_json)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
