"""Ensure vendored alpamayo source trees are importable on Lilypad workers."""

from __future__ import annotations

import sys
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
_ALPAMAYO_SRC = _THIRD_PARTY / "alpamayo1.5" / "src"
_ALPAMAYO2_SRC = _THIRD_PARTY / "alpamayo2" / "src"


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


def ensure_alpamayo2() -> Path:
    """Add NVlabs/alpamayo2 to sys.path.

    Same trick as alpamayo1.5: the package pins requires-python ==3.12.* so it
    is not pip-installable on the 3.10 workers, but every module under
    src/alpamayo2_super parses under 3.10 (ast.parse sweep, 2026-08-11).
    """
    if not (_ALPAMAYO2_SRC / "alpamayo2_super").is_dir():
        raise RuntimeError(
            f"alpamayo2_super source missing at {_ALPAMAYO2_SRC}. "
            "Run: git submodule update --init third_party/alpamayo2"
        )
    path = str(_ALPAMAYO2_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    return _ALPAMAYO2_SRC
