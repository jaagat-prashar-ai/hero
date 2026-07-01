#!/usr/bin/env python3
"""Download shards/objects from the research-datasets-chicago OCI bucket to a local folder."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BUCKET = "research-datasets-chicago"
DEFAULT_DEST = Path.home() / "Desktop" / "Shards"
DEFAULT_PROFILE = "oci.chi"

_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def strip_s3_uri(uri: str) -> str:
    if not uri.startswith("s3://"):
        return uri
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    if bucket != BUCKET:
        raise SystemExit(f"error: expected bucket {BUCKET!r}, got {bucket!r} in {uri!r}")
    return key


def parse_shard_index(name: str) -> int | None:
    m = _INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def s3_uri(key: str) -> str:
    return f"s3://{BUCKET}/{key}"


def run_aws(args: list[str], profile: str) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", "--profile", profile] + args, capture_output=True, text=True)


def list_prefix(prefix: str, profile: str) -> list[tuple[str, int]]:
    proc = run_aws(["s3", "ls", s3_uri(prefix), "--recursive"], profile)
    if proc.returncode != 0:
        raise SystemExit(f"error: aws s3 ls failed for {prefix!r}:\n{proc.stderr.strip()}")
    objects: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        _date, _time, size, key = parts
        objects.append((key, int(size)))
    return objects


def stat_object(key: str, profile: str) -> tuple[str, int]:
    proc = run_aws(["s3", "ls", s3_uri(key)], profile)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(f"error: object not found: {key!r}\n{proc.stderr.strip()}")
    parts = proc.stdout.strip().split(maxsplit=3)
    size = int(parts[2]) if len(parts) >= 3 else 0
    return key, size


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def resolve_targets(args: argparse.Namespace) -> tuple[list[tuple[str, int]], bool]:
    """Return (list of (key, size)), and whether this was a full prefix mirror
    (used to pick the default flat/mirrored destination layout)."""
    if args.range:
        if not args.prefix:
            raise SystemExit("error: --range requires --prefix")
        if args.items:
            raise SystemExit("error: --range does not take positional items")
        start_idx = parse_shard_index(args.range[0])
        end_idx = parse_shard_index(args.range[1])
        if start_idx is None or end_idx is None:
            raise SystemExit("error: could not parse a numeric shard index from --range names")
        lo, hi = min(start_idx, end_idx), max(start_idx, end_idx)
        all_objects = list_prefix(args.prefix, args.profile)
        selected = [
            (key, size)
            for key, size in all_objects
            if (idx := parse_shard_index(key.rsplit("/", 1)[-1])) is not None and lo <= idx <= hi
        ]
        if not selected:
            raise SystemExit(f"error: no objects under {args.prefix!r} matched index range {lo}-{hi}")
        return sorted(selected), False

    if not args.items:
        raise SystemExit("error: provide at least one item to download, or use --range")

    is_prefix_mirror = len(args.items) == 1 and not args.prefix and args.items[0].rstrip().endswith("/")

    resolved: list[tuple[str, int]] = []
    for item in args.items:
        if item.startswith("s3://"):
            key = strip_s3_uri(item)
        elif args.prefix and "/" not in item:
            key = args.prefix.rstrip("/") + "/" + item
        else:
            key = item

        if key.endswith("/"):
            objs = list_prefix(key, args.profile)
            if not objs:
                raise SystemExit(f"error: no objects found under prefix {key!r}")
            resolved.extend(objs)
        else:
            resolved.append(stat_object(key, args.profile))

    return resolved, is_prefix_mirror


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download shards/objects from research-datasets-chicago to a local folder."
    )
    parser.add_argument(
        "items", nargs="*",
        help="S3 URI(s), bucket-relative key(s), prefix (ends in /), or shard filename(s) (with --prefix)",
    )
    parser.add_argument("--prefix", help="Bucket-relative prefix to combine with plain filenames or use with --range")
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"),
        help="Inclusive numeric shard range, e.g. shard_000_00000.tar shard_000_00009.tar",
    )
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help=f"Local destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=f"AWS CLI profile (default: {DEFAULT_PROFILE})")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without transferring")
    flat_group = parser.add_mutually_exclusive_group()
    flat_group.add_argument(
        "--flat", dest="flat", action="store_true", default=None,
        help="Save all files directly into --dest, ignoring S3 key structure",
    )
    flat_group.add_argument(
        "--no-flat", dest="flat", action="store_false",
        help="Mirror the S3 key structure under --dest",
    )
    args = parser.parse_args()

    if not args.items and not args.range:
        parser.error("provide at least one item to download, or use --range")

    targets, is_prefix_mirror = resolve_targets(args)
    flat = args.flat if args.flat is not None else not is_prefix_mirror

    total_size = sum(size for _, size in targets)
    print(f"{len(targets)} object(s), {human_size(total_size)} total")
    for key, size in targets:
        print(f"  {human_size(size):>10}  {key}")

    if args.dry_run:
        print("\nDry run — nothing downloaded.")
        return

    dest_root = Path(args.dest).expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)

    print()
    ok: list[Path] = []
    failed: list[str] = []
    for key, size in targets:
        if flat:
            local_path = dest_root / Path(key).name
        else:
            local_path = dest_root / key
            local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {key} -> {local_path} ({human_size(size)}) ...")
        proc = run_aws(["s3", "cp", s3_uri(key), str(local_path)], args.profile)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr.strip()}")
            failed.append(key)
        else:
            ok.append(local_path)

    print()
    print(f"Downloaded {len(ok)}/{len(targets)} object(s) to {dest_root}")
    for p in ok:
        print(f"  {p}")

    if failed:
        print(f"\n{len(failed)} object(s) FAILED:")
        for key in failed:
            print(f"  {key}")
        sys.exit(1)


if __name__ == "__main__":
    main()
