# SPDX-License-Identifier: Apache-2.0
"""
build_wds_worker.py — Lilypad training_fn entrypoint for distributed WDS sharding.

Each Lilypad replica reads its RANK / WORLD_SIZE from the environment, then
delegates to build_webdataset.main() to process its assigned slice of clips.

Lilypad cluster config key (build_wds/configs/cluster.yaml):
    entrypoint_fn: build_wds.data.build_wds_worker.build_wds_loop

Required training_fn_config keys:
    bucket          S3 bucket name
    prefix          S3 key prefix (default: "physicalai-av/wds")

HF_TOKEN must be set in the environment (required_environment_variables in
cluster.yaml) — it is never read from training_fn_config, so it can't leak
into the logged argv line.

Optional training_fn_config keys:
    workers         Local threads per node (default 8)
    clips_per_shard Clips packed per tar shard (default 50)
    splits          List of splits to build (default [train, val, test])
    max_clips       Cap clips per split — for smoke tests
    resume_file     Path to file of already-completed clip IDs
    skip_metadata_upload  If true, skip uploading metadata parquets (default false)
    skip_lidar      If true, skip downloading/writing LiDAR for this run (default false)
    video_codec     Camera MP4 encoding: copy or av1 (default av1)
    video_crf       AV1 CRF quality (default 32)
    video_preset    AV1 encoder preset (default 6)
"""

from __future__ import annotations

import logging
import os
import sys
import typing
from typing import Any

# Backport Python 3.11 additions used by physical_ai_av on Python 3.10 workers.
if sys.version_info < (3, 11):
    import enum

    # typing.Self
    if not hasattr(typing, "Self"):
        from typing_extensions import Self
        typing.Self = Self  # type: ignore[attr-defined]

    # enum.StrEnum
    if not hasattr(enum, "StrEnum"):
        class _StrEnum(str, enum.Enum):
            __str__ = str.__str__
        enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

# physical_ai_av uses scipy.spatial.transform.RigidTransform, which does not
# exist in any released scipy version. Provide a minimal stub so imports succeed.
# Calibration/egomotion features still serialize their rotation/translation data;
# they will work correctly once NVIDIA ships against a released scipy.
import scipy.spatial.transform as _sst
if not hasattr(_sst, "RigidTransform"):
    class _RigidTransform:
        def __init__(self, rotation=None, translation=None):
            self.rotation = rotation
            self.translation = translation
            self.pose = self

        @classmethod
        def identity(cls) -> "_RigidTransform":
            return cls()

        @classmethod
        def from_components(cls, rotation=None, translation=None) -> "_RigidTransform":
            return cls(rotation=rotation, translation=translation)

        def __repr__(self) -> str:
            return f"RigidTransform(rotation={self.rotation!r}, translation={self.translation!r})"

    _sst.RigidTransform = _RigidTransform  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def _build_argv(cfg: dict[str, Any], rank: int, world_size: int) -> list[str]:
    """Build the flags list for build_webdataset (no argv[0] program name)."""
    bucket          = cfg.get("bucket")
    prefix          = cfg.get("prefix", "physicalai-av/wds")
    workers         = int(cfg.get("workers", 8))
    clips_per_shard = int(cfg.get("clips_per_shard", 50))
    splits          = cfg.get("splits", ["train", "val", "test"])
    max_clips       = cfg.get("max_clips")
    resume_file     = cfg.get("resume_file")
    skip_meta       = bool(cfg.get("skip_metadata_upload", False))
    skip_lidar      = bool(cfg.get("skip_lidar", False))
    video_codec     = str(cfg.get("video_codec", "av1"))
    video_crf       = int(cfg.get("video_crf", 32))
    video_preset    = int(cfg.get("video_preset", 6))

    argv = [
        "--bucket",          bucket,
        "--prefix",          prefix,
        "--workers",         str(workers),
        "--clips_per_shard", str(clips_per_shard),
        "--splits",          *splits,
        "--rank",            str(rank),
        "--world_size",      str(world_size),
    ]

    if max_clips is not None:
        argv += ["--max_clips", str(max_clips)]
    if resume_file:
        argv += ["--resume_file", resume_file]
    if skip_meta:
        argv.append("--skip_metadata_upload")
    if skip_lidar:
        argv.append("--skip_lidar")
    argv += [
        "--video_codec", video_codec,
        "--video_crf", str(video_crf),
        "--video_preset", str(video_preset),
    ]

    return argv


def build_wds_loop(
    training_fn_config: dict[str, Any],
    experiment_tracker: Any = None,
) -> None:
    """Lilypad-compatible entrypoint for distributed WDS sharding."""
    cfg = training_fn_config

    # Rank is injected by Lilypad when num_replicas > 1; fall back to config.
    rank       = int(os.environ.get("RANK",       cfg.get("rank",       0)))
    world_size = int(os.environ.get("WORLD_SIZE", cfg.get("world_size", 1)))

    bucket = cfg.get("bucket")
    if not bucket:
        raise ValueError("training_fn_config.bucket is required")

    argv = _build_argv(cfg, rank, world_size)

    logger.info("build_wds_worker: rank=%d world_size=%d  argv=%s",
                rank, world_size, " ".join(argv))

    sys.argv = ["build_webdataset"] + argv
    from build_wds.data.build_webdataset import main
    main()
