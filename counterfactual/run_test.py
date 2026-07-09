# SPDX-License-Identifier: Apache-2.0
"""
run_test.py — covers only the pure JSON-flattening logic in run.py.

Uses plain types.SimpleNamespace stand-ins with the same field names as
counterfactual.py's AlternativeToken/CounterfactualResult dataclasses,
rather than importing those classes directly -- counterfactual.py imports
`transformers` at module level, which isn't installed in this sandbox (same
gap as rollout_harvester_test.py's alpamayo1_5 dependency). This is a
data-shape stand-in for a plain value container, not a fake model: real
verification of the actual model-touching code (_load_model,
counterfactual_sweep_loop) is exclusively the real Lilypad run, per this
project's standing "no fake model tests" preference.
"""

from __future__ import annotations

from types import SimpleNamespace

from counterfactual.run import _position_result_to_json


def test_position_result_to_json_flattens_dataclasses():
    position_result = {
        "step": 3,
        "col": 42,
        "sampled_token": " left",
        "sampled_prob": 0.62,
        "entropy": 0.71,
        "alternatives": [
            SimpleNamespace(
                forced_token=SimpleNamespace(token_id=99, text=" right", prob=0.21),
                forced_cot="... turn right at the intersection ...",
                d_curvature_mean=0.012,
                d_curvature_max=0.045,
                endpoint_shift_m=1.8,
                traj_ade_m=0.9,
            ),
        ],
    }
    out = _position_result_to_json(position_result)

    assert out["step"] == 3
    assert out["col"] == 42
    assert out["sampled_token"] == " left"
    assert out["sampled_prob"] == 0.62
    assert out["entropy"] == 0.71
    assert out["alternatives"] == [{
        "token": " right",
        "token_prob": 0.21,
        "d_curvature_mean": 0.012,
        "d_curvature_max": 0.045,
        "endpoint_shift_m": 1.8,
        "traj_ade_m": 0.9,
        "forced_cot": "... turn right at the intersection ...",
    }]


def test_position_result_to_json_handles_no_alternatives():
    out = _position_result_to_json({
        "step": 0, "col": 10, "sampled_token": "x", "sampled_prob": 1.0,
        "entropy": 0.0, "alternatives": [],
    })
    assert out["alternatives"] == []
