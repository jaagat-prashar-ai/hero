# SPDX-License-Identifier: Apache-2.0
"""Offline test for run_prototype.py's real-rollout gate loop (no network,
no GPU -- generate_reward_fn is monkeypatched to return canned sources, and
rollout_groups is a hand-built fixture standing in for a real Alpamayo
sample). Exercises the fix for the d50ad0 GT-overfit bug (see
run_prototype.py's module docstring): a candidate mirroring the real
failure mode (a conjunction requiring BOTH 'decelerate' AND 'yield'
commitments verbatim, like the real 333b20c5 reward_fns/) must be rejected
because no rollout in the group ever uses that exact phrasing, while a
genuinely scene-aware function (GOOD_FN, already proven against GT in
clipgen_test.py) passes against the same group's argmax.
"""

from __future__ import annotations

import json

import numpy as np

from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen import generate as generate_mod
from code_as_a_reward.clipgen import run_prototype
from code_as_a_reward.clipgen.clipgen_test import GOOD_FN, GT_COC, TESTDATA, _reactive_waypoints

CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
HZ = 10.0

# Mirrors the real 333b20c5 bug: a conjunction gated on BOTH 'decelerate' AND
# 'yield' commitments verbatim. Passed the old GT-only gate (GT_COC contains
# both words), but no real rollout below uses that exact phrasing, so its
# only reachable credit is the 0.1 mention-only component -- well below
# POS_MIN (0.7) on every real rollout's argmax.
OVERFIT_CONJUNCTION_FN = """\
def components(claims, traj):
    has_decelerate = any(c.maneuver == 'decelerate' for c in claims.commitments)
    has_yield = any(c.maneuver == 'yield' for c in claims.commitments)
    saw = any(c.entity in ('stopped_vehicle', 'vehicle_generic') for c in claims.perceptual)
    conjunction = 0.0
    if has_decelerate and has_yield:
        win = window(traj.speed_mps, traj.dt_s, 1.5, 4.5)
        if len(win) > 1 and win[0] > 0:
            drop = float(win[0] - win.min())
            conjunction = 0.8 if drop >= 4.0 else 0.0
    return {"saw_vehicle": 0.1 * saw, "yield_and_decelerate_executed": conjunction}


def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
"""


class _FakeOpenAIChat:
    """Stands in for generate.OpenAIChat so run()'s client construction
    never touches the network/credentials -- generate_reward_fn itself is
    monkeypatched below, so this client is never actually called."""

    def __init__(self, *a, **kw):
        pass


def _build_manifest(tmp_path) -> str:
    reactive = _reactive_waypoints()
    gt_coc_path = tmp_path / "gt.coc.txt"
    gt_coc_path.write_text(GT_COC)
    wp_path = tmp_path / "gt_wp.npy"
    np.save(wp_path, reactive)
    manifest = [
        {
            "clip_id": CLIP_ID,
            "obstacle_parquet": TESTDATA,
            "waypoints_npy": str(wp_path),
            "gt_coc": str(gt_coc_path),
            "hz": HZ,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(manifest_path)


def _rollout_group() -> dict:
    """A real rollout group never reproduces GT's exact wording -- rollout 0
    decelerates for the vehicle but never says "yield"; rollout 1 is a
    distinct, unrelated response (keep lane, no reaction)."""
    reactive = _reactive_waypoints()
    flat = gate_mod.flattened_waypoints(reactive)
    return {
        CLIP_ID: [
            {
                "rollout_id": 0,
                "coc_text": "There is a stopped vehicle ahead in my lane. I will decelerate because of it.",
                "waypoints": reactive.tolist(),
            },
            {
                "rollout_id": 1,
                "coc_text": "I will keep lane since the lane is clear ahead.",
                "waypoints": flat.tolist(),
            },
        ]
    }


def test_real_rollout_gate_rejects_overfit_then_passes_generalizing_fn(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_mod, "OpenAIChat", _FakeOpenAIChat)
    sources = iter([OVERFIT_CONJUNCTION_FN, GOOD_FN])

    def fake_generate_reward_fn(client, dossier, **kwargs):
        return generate_mod.GenerationResult(
            source=next(sources), transcript=[{"role": "assistant", "content": "fake"}], model="fake-model"
        )

    monkeypatch.setattr(run_prototype, "generate_reward_fn", fake_generate_reward_fn)

    manifest_path = _build_manifest(tmp_path)
    report = run_prototype.run(
        manifest_path, str(tmp_path / "out"), _rollout_group(), dry_run=False, backend="openai"
    )

    entry = report["clips"][CLIP_ID]
    assert len(entry["attempts"]) == 2, entry["attempts"]
    assert entry["attempts"][0]["passed"] is False, entry["attempts"][0]
    assert entry["attempts"][0]["pos_score"] < gate_mod.POS_MIN
    assert entry["attempts"][1]["passed"] is True, entry["attempts"][1]
    assert entry["passed"] is True
    assert (tmp_path / "out" / "reward_fns" / f"{CLIP_ID}.py").exists()


def test_missing_rollout_group_skips_clip_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_mod, "OpenAIChat", _FakeOpenAIChat)
    monkeypatch.setattr(
        run_prototype,
        "generate_reward_fn",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should never be called")),
    )
    manifest_path = _build_manifest(tmp_path)
    report = run_prototype.run(manifest_path, str(tmp_path / "out"), {}, dry_run=False, backend="openai")
    entry = report["clips"][CLIP_ID]
    assert not entry["passed"]
    assert "no sampled rollout group" in entry["error"]
    assert entry["attempts"] == []


def test_dry_run_needs_no_rollout_group(tmp_path):
    manifest_path = _build_manifest(tmp_path)
    report = run_prototype.run(manifest_path, str(tmp_path / "out"), {}, dry_run=True)
    entry = report["clips"][CLIP_ID]
    assert entry["n_rollouts"] == 0
    assert not entry["passed"]
    assert (tmp_path / "out" / f"{CLIP_ID}.dossier.txt").exists()
