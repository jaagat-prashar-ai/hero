# SPDX-License-Identifier: Apache-2.0
"""FFmpeg-based MP4 transcoding for build_wds camera streams."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

VideoCodec = Literal["copy", "av1"]

# Prefer SVT-AV1 (faster); fall back to libaom-av1 when libsvtav1 is unavailable.
_AV1_ENCODERS = ("libsvtav1", "libaom-av1")
_CODEC_ALIASES = frozenset({"av1", "av01", "libaom-av1", "libsvtav1"})


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH — required for video transcoding")
    return path


def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found on PATH — required for video transcoding")
    return path


def list_ffmpeg_encoders() -> list[str]:
    """Return encoder names reported by ``ffmpeg -encoders``."""
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    encoders: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith("."):
            encoders.append(parts[1])
    return encoders


def pick_av1_encoder(encoders: list[str] | None = None) -> str | None:
    available = set(encoders if encoders is not None else list_ffmpeg_encoders())
    for name in _AV1_ENCODERS:
        if name in available:
            return name
    return None


def check_ffmpeg_av1_available() -> tuple[bool, str | None]:
    """Return (ok, encoder_name) for preflight checks."""
    encoder = pick_av1_encoder()
    return encoder is not None, encoder


def ensure_ffmpeg_av1() -> str:
    """Return an AV1 encoder name, installing ffmpeg via apt on cluster if needed."""
    encoder = pick_av1_encoder()
    if encoder:
        return encoder

    apt = shutil.which("apt-get")
    if apt:
        logger.info("No AV1 encoder found — attempting apt-get install ffmpeg")
        subprocess.run([apt, "update", "-qq"], check=False)
        subprocess.run([apt, "install", "-y", "-qq", "ffmpeg"], check=False)
        encoder = pick_av1_encoder()
        if encoder:
            logger.info("ffmpeg AV1 encoder available after apt install: %s", encoder)
            return encoder

    raise RuntimeError(
        "No AV1 encoder available — install ffmpeg with libsvtav1 or libaom-av1 on the worker"
    )


def probe_codec(mp4_bytes: bytes) -> str | None:
    """Return the video stream codec name (e.g. h264, av1) or None."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                ffprobe_path(),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip().lower() or None
    except subprocess.CalledProcessError:
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def _av1_encode_args(encoder: str, crf: int, preset: int) -> list[str]:
    args = ["-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p", "-an"]
    if encoder == "libsvtav1":
        args.extend(["-preset", str(preset)])
    else:
        # libaom-av1: lower cpu-used = slower/better; map preset roughly.
        cpu_used = max(4, min(8, 12 - preset))
        args.extend(["-cpu-used", str(cpu_used)])
    return args


def transcode_mp4(
    mp4_bytes: bytes,
    *,
    codec: VideoCodec = "av1",
    crf: int = 32,
    preset: int = 6,
    camera_label: str = "",
) -> bytes:
    """Re-encode MP4 bytes to AV1, or return input unchanged when codec is copy."""
    if codec == "copy":
        return mp4_bytes

    src_codec = probe_codec(mp4_bytes)
    if src_codec in _CODEC_ALIASES:
        logger.info(
            "video %s: already %s — skipping transcode (%d MB)",
            camera_label or "?",
            src_codec,
            len(mp4_bytes) / 1e6,
        )
        return mp4_bytes

    encoder = pick_av1_encoder()
    if encoder is None:
        raise RuntimeError(
            "No AV1 encoder available — install ffmpeg with libsvtav1 or libaom-av1"
        )

    in_mb = len(mp4_bytes) / 1e6
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "in.mp4"
        out = Path(tmpdir) / "out.mp4"
        inp.write_bytes(mp4_bytes)

        cmd = [
            ffmpeg_path(), "-y",
            "-i", str(inp),
            *_av1_encode_args(encoder, crf, preset),
            str(out),
        ]
        _run_ffmpeg(cmd)
        out_bytes = out.read_bytes()

    out_mb = len(out_bytes) / 1e6
    ratio = out_mb / in_mb if in_mb > 0 else 0.0
    logger.info(
        "video %s: %s → AV1 (%s)  %.1f MB → %.1f MB (%.0f%%)",
        camera_label or "?",
        src_codec or "unknown",
        encoder,
        in_mb,
        out_mb,
        ratio * 100,
    )
    return out_bytes
