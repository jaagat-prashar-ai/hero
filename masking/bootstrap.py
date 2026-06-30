"""Ensure vendored alpamayo1_5 source is importable on Lilypad workers."""

from __future__ import annotations

import sys
from pathlib import Path

_ALPAMAYO_SRC = Path(__file__).resolve().parents[1] / "third_party" / "alpamayo1.5" / "src"


def ensure_alpamayo1_5() -> Path:
    """Add NVlabs/alpamayo1.5 to sys.path (not pip-installable on Lilypad Python 3.10)."""
    if not (_ALPAMAYO_SRC / "alpamayo1_5").is_dir():
        raise RuntimeError(
            f"alpamayo1_5 source missing at {_ALPAMAYO_SRC}. "
            "Run: git submodule update --init third_party/alpamayo1.5"
        )
    path = str(_ALPAMAYO_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    return _ALPAMAYO_SRC
