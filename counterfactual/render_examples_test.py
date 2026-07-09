# SPDX-License-Identifier: Apache-2.0
"""render_examples_test.py — covers render_one's data-extraction/degenerate-
handling logic with synthetic fixtures (real matplotlib rendering, no model).

Regression coverage: an earlier version treated forced_cot as a plain string
(`alt_b.get("forced_cot", "")`), but it's actually stored as a dict
(`{"cot": [...], "meta_action": [...], "answer": [...]}`) throughout this
pipeline -- same shape report.py's is_degenerate already handles correctly.
That mismatch crashed with AttributeError on the very first real rerun."""

from __future__ import annotations

import tempfile
from pathlib import Path

from counterfactual.render_examples import render_one


def _scene_json(baseline_xy, alt_a_xy, alt_a_ade, alt_b_xy=None, alt_b_ade=None, forced_cot_text="a real alternative reasoning trace"):
    alt_b = None
    if alt_b_xy is not None:
        alt_b = {
            "token": "Change", "token_prob": 0.05, "traj_ade_m": alt_b_ade, "endpoint_shift_m": alt_b_ade * 2,
            "forced_cot": {"cot": [forced_cot_text], "meta_action": [""], "answer": [""]},
            "xy": alt_b_xy,
        }
    return {
        "single_token_swap_sweep": [{
            "step": 0, "sampled_token": "Keep", "sampled_prob": 0.9, "entropy": 0.5,
            "alternatives": [{
                "token": "Change", "token_prob": 0.05, "traj_ade_m": alt_a_ade, "endpoint_shift_m": alt_a_ade * 2,
                "forced_cot": {"cot": [""], "meta_action": [""], "answer": [""]}, "xy": alt_a_xy,
            }],
        }],
        "counterfactual_sweep": [{
            "step": 0, "sampled_token": "Keep", "sampled_prob": 0.9, "entropy": 0.5,
            "alternatives": [alt_b] if alt_b else [],
        }],
        "baseline_xy_a": baseline_xy,
        "baseline_xy_b": baseline_xy,
    }


_BASE_XY = [[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]]
_A_XY = [[0.0, 0.0], [1.0, 0.3], [2.0, 0.6]]
_B_XY = [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]


def test_render_one_handles_clean_option_b_without_crashing():
    scene = _scene_json(_BASE_XY, _A_XY, 0.3, alt_b_xy=_B_XY, alt_b_ade=0.6)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        assert render_one(scene, 0, "Change", out) is True
        assert out.exists() and out.stat().st_size > 0


def test_render_one_skips_degenerate_option_b_without_crashing():
    """The exact regression case: forced_cot is a dict with an empty cot
    string (degenerate generation) -- must not raise AttributeError."""
    scene = _scene_json(_BASE_XY, _A_XY, 0.3, alt_b_xy=_B_XY, alt_b_ade=20.0, forced_cot_text="")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        assert render_one(scene, 0, "Change", out) is True
        assert out.exists() and out.stat().st_size > 0


def test_render_one_handles_missing_option_b_entirely():
    scene = _scene_json(_BASE_XY, _A_XY, 0.3, alt_b_xy=None)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        assert render_one(scene, 0, "Change", out) is True


def test_render_one_returns_false_for_missing_example():
    scene = _scene_json(_BASE_XY, _A_XY, 0.3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        assert render_one(scene, 5, "NoSuchToken", out) is False
        assert not out.exists()
