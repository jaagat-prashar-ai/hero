# SPDX-License-Identifier: Apache-2.0
"""Audit whether a sealed ClipGen corpus can be restored from the S3 WDS mirror.

The audit is deliberately metadata-only: it lists published reward filenames and
range-reads tar headers at the offsets recorded in the checked-in OOD manifest.
It never downloads camera payloads and never touches Hugging Face.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any



TAR_BLOCK = 512
DEFAULT_BUCKET = "research-datasets-chicago"
CAMERAS = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
REQUIRED_SUFFIXES = {
    "json",
    "egomotion.parquet",
    *(f"{camera}.mp4" for camera in CAMERAS),
    *(f"{camera}.timestamps.parquet" for camera in CAMERAS),
}


def _s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"payload_signing_enabled": True},
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )


def _range_get(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    return s3.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes={start}-{start + length - 1}",
    )["Body"].read()


def _member_suffix(name: str, clip_id: str) -> str | None:
    prefix = f"{clip_id}."
    return name[len(prefix) :] if name.startswith(prefix) else None


def audit_clip_headers(bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    """Read only the selected clip's tar headers and report required members."""
    s3 = _s3_client()
    clip_id = str(row["clip_id"])
    key = str(row["shard_key"])
    pos = int(row["offset"])
    found: set[str] = set()
    first_member = True

    while True:
        header = _range_get(s3, bucket, key, pos, TAR_BLOCK)
        if len(header) < TAR_BLOCK or header == b"\x00" * TAR_BLOCK:
            break
        name = header[0:100].split(b"\x00", 1)[0].decode(errors="replace")
        size_field = header[124:136].split(b"\x00", 1)[0].strip()
        size = int(size_field, 8) if size_field else 0
        is_pax = "@PaxHeader" in name

        if not is_pax:
            suffix = _member_suffix(name, clip_id)
            if suffix is None:
                if not first_member:
                    break
            else:
                found.add(suffix)
                first_member = False

        pos += TAR_BLOCK + ((size + TAR_BLOCK - 1) // TAR_BLOCK) * TAR_BLOCK

    missing = sorted(REQUIRED_SUFFIXES - found)
    return {
        "clip_id": clip_id,
        "shard_key": key,
        "offset": int(row["offset"]),
        "complete": not missing,
        "missing_members": missing,
        "members_found": sorted(found),
    }


def _published_clip_ids(s3, bucket: str, prefixes: list[str]) -> set[str]:
    clip_ids: set[str] = set()
    for prefix in prefixes:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                key = str(obj["Key"])
                if "/reward_fns/" in key and key.endswith(".py"):
                    clip_ids.add(Path(key).stem)
    return clip_ids


def _prefixes_from_source_config(path: str | Path) -> list[str]:
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    entrypoint = config["workload_variant_config"]["entrypoint_fn_config"]
    raw = entrypoint["clipgen_run_s3_prefixes"]
    return [str(raw)] if isinstance(raw, str) else [str(item) for item in raw]


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    bucket = str(config.get("bucket", DEFAULT_BUCKET))
    source_config = str(
        config.get("source_config", "rl_posttrain/configs/faithfulness_compare_717.yaml")
    )
    manifest_path = Path(
        config.get("wds_manifest", "pref_pairs/configs/sample_clips_all.json")
    )
    output_key = str(
        config.get(
            "output_key",
            "alpamayo_rl/audits/faithfulness717_wds_coverage/report.json",
        )
    )
    max_workers = int(config.get("max_workers", 48))

    s3 = _s3_client()
    prefixes = _prefixes_from_source_config(source_config)
    published = _published_clip_ids(s3, bucket, prefixes)
    manifest_rows = json.loads(manifest_path.read_text())
    manifest_by_id = {str(row["clip_id"]): row for row in manifest_rows}

    mapped_ids = sorted(published & manifest_by_id.keys())
    missing_from_manifest = sorted(published - manifest_by_id.keys())
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(audit_clip_headers, bucket, manifest_by_id[clip_id]): clip_id
            for clip_id in mapped_ids
        }
        for index, future in enumerate(as_completed(futures), 1):
            clip_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "clip_id": clip_id,
                        "complete": False,
                        "missing_members": sorted(REQUIRED_SUFFIXES),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % 100 == 0 or index == len(futures):
                print(f"audited {index}/{len(futures)} mapped clips", flush=True)

    rows.sort(key=lambda row: row["clip_id"])
    complete_ids = sorted(row["clip_id"] for row in rows if row["complete"])
    incomplete = [row for row in rows if not row["complete"]]
    report = {
        "schema_version": "alpamayo.wds_coverage.v1",
        "bucket": bucket,
        "source_config": source_config,
        "wds_manifest": str(manifest_path),
        "published_reward_count": len(published),
        "manifest_clip_count": len(manifest_by_id),
        "mapped_reward_count": len(mapped_ids),
        "complete_reward_count": len(complete_ids),
        "missing_from_manifest_count": len(missing_from_manifest),
        "incomplete_member_count": len(incomplete),
        "complete_fraction_of_published": (
            len(complete_ids) / len(published) if published else 0.0
        ),
        "complete_clip_ids": complete_ids,
        "missing_from_manifest": missing_from_manifest,
        "incomplete": incomplete,
    }
    body = json.dumps(report, indent=2).encode()
    s3.put_object(Bucket=bucket, Key=output_key, Body=body)
    print(json.dumps({key: report[key] for key in (
        "published_reward_count",
        "manifest_clip_count",
        "mapped_reward_count",
        "complete_reward_count",
        "missing_from_manifest_count",
        "incomplete_member_count",
        "complete_fraction_of_published",
    )}, indent=2), flush=True)
    print(f"report=s3://{bucket}/{output_key}", flush=True)
    return report


def run_from_lilypad_config(
    training_fn_config: dict[str, Any], experiment_tracker=None
) -> None:
    del experiment_tracker
    run_audit(training_fn_config)


if __name__ == "__main__":
    run_audit({})
