# SPDX-License-Identifier: Apache-2.0
"""report_test.py — covers build_counterfactual_data's aggregation/dedup
logic and render_counterfactual_section's well-formedness, using synthetic
fixtures (no model, no real cluster data)."""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path

import pytest

from counterfactual.report import build_counterfactual_data, is_degenerate, render_counterfactual_section


def _alt(token: str, prob: float, ade_a: float, ade_b: float, forced_cot: str | None) -> dict:
    return {
        "token": token, "token_prob": prob,
        "d_curvature_mean": 0.0, "d_curvature_max": 0.0,
        "endpoint_shift_m": ade_a * 2, "traj_ade_m": ade_a,
        "forced_cot": {"cot": [forced_cot] if forced_cot is not None else [""], "meta_action": [""], "answer": [""]},
    }, ade_b


def _write_scene(tmp: Path, scene_id: str, cot: str, positions: list[dict]) -> None:
    """positions: list of {step, sampled_token, sampled_prob, entropy, alternatives: [(token, prob, ade_a, forced_cot_text)]}"""
    swap_a, swap_b = [], []
    for p in positions:
        alts_a, alts_b = [], []
        for token, prob, ade_a, ade_b, forced_cot in p["alternatives"]:
            alts_a.append({
                "token": token, "token_prob": prob, "d_curvature_mean": 0.0, "d_curvature_max": 0.0,
                "endpoint_shift_m": ade_a * 2, "traj_ade_m": ade_a,
                "forced_cot": {"cot": [""], "meta_action": [""], "answer": [""]},
            })
            alts_b.append({
                "token": token, "token_prob": prob, "d_curvature_mean": 0.0, "d_curvature_max": 0.0,
                "endpoint_shift_m": ade_b * 2, "traj_ade_m": ade_b,
                "forced_cot": {"cot": [forced_cot] if forced_cot is not None else [""], "meta_action": [""], "answer": [""]},
            })
        base = {"step": p["step"], "col": 100 + p["step"], "sampled_token": p["sampled_token"],
                "sampled_prob": p["sampled_prob"], "entropy": p["entropy"]}
        swap_a.append({**base, "alternatives": alts_a})
        swap_b.append({**base, "alternatives": alts_b})

    (tmp / f"{scene_id}.json").write_text(json.dumps({
        "token_alternative_map": {
            "cot": {"cot": [cot], "meta_action": [""], "answer": [""]},
            "summary": {"n_reasoning_tokens": len(positions)},
        },
        "single_token_swap_sweep": swap_a,
        "counterfactual_sweep": swap_b,
    }))


def test_is_degenerate_true_only_for_empty_forced_cot():
    assert is_degenerate({"forced_cot": {"cot": [""]}}) is True
    assert is_degenerate({"forced_cot": {"cot": ["real text"]}}) is False
    assert is_degenerate({"forced_cot": {"cot": ["  "]}}) is True  # whitespace-only counts as empty


def test_build_counterfactual_data_excludes_degenerate_from_stats():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_scene(tmp_path, "scene_a", "Keep distance to the lead vehicle", [
            {"step": 0, "sampled_token": "Keep", "sampled_prob": 0.9, "entropy": 0.5, "alternatives": [
                ("Ad", 0.05, 0.1, 0.2, "Adjust speed to the lead vehicle"),   # clean
                ("N", 0.02, 0.3, 20.0, None),                                 # degenerate (forced_cot=None -> "")
            ]},
        ])
        data = build_counterfactual_data(tmp_path)

    assert data["n_scenes"] == 1
    assert data["n_positions"] == 1
    assert data["n_alternatives"] == 2
    assert data["n_degenerate"] == 1
    # Only the clean (0.2) Option B value should count toward stats, not the degenerate 20.0.
    assert data["stats_b"]["max"] == pytest.approx(0.2)
    assert data["stats_b"]["mean"] == pytest.approx(0.2)
    # Option A stats are never filtered by degeneracy (it only applies to Option B).
    assert data["stats_a"]["max"] == pytest.approx(0.3)


def test_build_counterfactual_data_splits_option_a_by_step_too():
    """Regression test: an earlier version split Option B's ADE values by
    step (stats_b_step0/stats_b_other) but forgot to do the same for Option
    A, so the rendered report showed the SAME Option A number on the
    "Overall"/"step 0"/"later steps" rows. Both options must be split
    identically."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_scene(tmp_path, "scene_a", "cot", [
            {"step": 0, "sampled_token": "x", "sampled_prob": 0.9, "entropy": 0.1, "alternatives": [
                ("y", 0.05, 9.0, 9.5, "step0 alt cot"),  # Option A ade_a=9.0 at step 0
            ]},
            {"step": 1, "sampled_token": "z", "sampled_prob": 0.9, "entropy": 0.1, "alternatives": [
                ("w", 0.05, 1.0, 1.5, "step1 alt cot"),  # Option A ade_a=1.0 at step 1
            ]},
        ])
        data = build_counterfactual_data(tmp_path)

    assert data["stats_a_step0"]["median"] == pytest.approx(9.0)
    assert data["stats_a_other"]["median"] == pytest.approx(1.0)
    assert data["stats_a"]["median"] == pytest.approx(statistics.median([9.0, 1.0]))
    # The three must NOT all collapse to the same value.
    assert data["stats_a_step0"]["median"] != data["stats_a_other"]["median"]

    out = render_counterfactual_section(data)
    # Both the step-0 (9) and later-step (1) Option A medians must appear as
    # distinct rendered dumbbell values -- not the same overall-median
    # number (5, the median of [9, 1]) repeated on every row.
    assert 'swatch-a"></i>9<i' in out
    assert 'swatch-a"></i>1<i' in out


def test_build_counterfactual_data_sorts_scenes_by_max_clean_ade_descending():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_scene(tmp_path, "low_scene", "cot low", [
            {"step": 0, "sampled_token": "x", "sampled_prob": 0.9, "entropy": 0.1, "alternatives": [
                ("y", 0.05, 0.01, 0.02, "alt cot"),
            ]},
        ])
        _write_scene(tmp_path, "high_scene", "cot high", [
            {"step": 0, "sampled_token": "x", "sampled_prob": 0.9, "entropy": 0.1, "alternatives": [
                ("y", 0.05, 5.0, 6.0, "alt cot"),
            ]},
        ])
        data = build_counterfactual_data(tmp_path)

    assert [s["scene_id"] for s in data["scenes"]] == ["high_scene", "low_scene"]


def test_build_counterfactual_data_raises_on_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            build_counterfactual_data(Path(tmp))


def test_render_counterfactual_section_flags_degenerate_and_hides_its_ade_b():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_scene(tmp_path, "scene_a", "Keep distance", [
            {"step": 0, "sampled_token": "Keep", "sampled_prob": 0.9, "entropy": 0.5, "alternatives": [
                ("N", 0.02, 0.3, 20.0, None),  # degenerate
            ]},
        ])
        data = build_counterfactual_data(tmp_path)
        out = render_counterfactual_section(data)

    assert "generation incomplete" in out
    assert "scene_a" in out
    # Balanced <details>/<div> sanity: every details tag closes.
    assert out.count("<details") == out.count("</details>")
    # The degenerate alternative's Option B ade (20.0) must not appear as a
    # rendered number -- only the em-dash placeholder should show.
    assert "20</span>" not in out and "20.0" not in out
