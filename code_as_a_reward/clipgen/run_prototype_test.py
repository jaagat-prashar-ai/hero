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
            "t0_us": 0,
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
    rollouts = [
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
    return {
        CLIP_ID: {
            "schema_version": "clipgen.rollouts.v2",
            "clip_id": CLIP_ID,
            "t0_us": 0,
            "gt_waypoints": reactive.tolist(),
            # Rollout IDs are local to a stochastic forward-pass group, so
            # both independently sampled groups correctly use 0..N-1.
            "groups": {"generation": rollouts, "holdout": [dict(r) for r in rollouts]},
            "provenance": {
                "model": "fake",
                "generation_seed": 1,
                "holdout_seed": 2,
            },
        }
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
        manifest_path,
        str(tmp_path / "out"),
        _rollout_group(),
        dry_run=False,
        backend="openai",
        min_generation_rollouts=2,
        min_holdout_rollouts=2,
        holdout_top_k=1,
        cross_scene_negatives=0,
        holdout_min_unique_scores=2,
        holdout_max_saturation_fraction=1.0,
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
    report = run_prototype.run(
        manifest_path,
        str(tmp_path / "out"),
        {},
        dry_run=False,
        backend="openai",
        strict_rollouts=False,
    )
    entry = report["clips"][CLIP_ID]
    assert not entry["passed"]
    assert "no sampled rollout group" in entry["error"]
    assert entry["attempts"] == []


def test_dry_run_needs_no_rollout_group(tmp_path):
    manifest_path = _build_manifest(tmp_path)
    report = run_prototype.run(manifest_path, str(tmp_path / "out"), {}, dry_run=True)
    entry = report["clips"][CLIP_ID]
    assert entry["n_generation_rollouts"] == 0
    assert not entry["passed"]
    assert (tmp_path / "out" / f"{CLIP_ID}.dossier.txt").exists()


def test_offline_gt_builder_never_requires_or_records_policy_rollouts(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_mod, "OpenAIChat", _FakeOpenAIChat)

    def fake_generate_reward_fn(client, dossier, **kwargs):
        return generate_mod.GenerationResult(
            source=GOOD_FN,
            transcript=[{"role": "assistant", "content": "fake"}],
            model="fake-model",
        )

    monkeypatch.setattr(run_prototype, "generate_reward_fn", fake_generate_reward_fn)
    manifest_path = _build_manifest(tmp_path)
    report = run_prototype.run(
        manifest_path,
        str(tmp_path / "offline_out"),
        {},
        dry_run=False,
        backend="openai",
        offline_gt_only=True,
    )

    entry = report["clips"][CLIP_ID]
    assert report["pipeline_mode"] == "offline_gt_only"
    assert entry["n_generation_rollouts"] == 0
    assert entry["n_holdout_rollouts"] == 0
    assert entry["rollout_provenance"] is None
    assert entry["passed"] is True
    assert entry["attempts"][-1]["stage"] == "offline_gt_semantic_gate"
    assert entry["attempts"][-1]["pos_score"] >= gate_mod.POS_MIN
    assert (
        entry["attempts"][-1]["pos_score"] - entry["attempts"][-1]["max_pert"]
        >= gate_mod.MIN_DROP
    )
    assert report["acceptance"]["counts"]["published"] == 1
    assert (tmp_path / "offline_out" / "reward_fns" / f"{CLIP_ID}.py").exists()


def test_acceptance_summary_separates_sampling_from_reward_failures():
    report = {
        "clips": {
            "no-sample": {
                "passed": False,
                "target_contract": {"speed_profiles": ["decelerate"]},
                "attempts": [
                    {
                        "reward_spec": {"components": []},
                        "gt_gate_passed": True,
                        "eligible_rollout_ids": [],
                        "outcome": "NO_VALID_ROLLOUT",
                    }
                ],
            },
            "accepted": {
                "passed": True,
                "target_contract": {"speed_profiles": ["decelerate"]},
                "attempts": [
                    {
                        "stage": "generation_group",
                        "reward_spec": {"components": []},
                        "gt_gate_passed": True,
                        "eligible_rollout_ids": [2],
                        "passed": True,
                    }
                ],
                "holdout": {"passed": True},
                "cross_scene": {"passed": True},
            },
        }
    }
    summary = run_prototype.summarize_acceptance(report)
    assert summary["counts"]["total"] == 2
    assert summary["counts"]["no_valid_rollout"] == 1
    assert summary["counts"]["published"] == 1
    assert summary["rates"]["published"] == 0.5
    assert summary["rates"]["published_over_target_eligible"] == 1.0
    assert summary["action_families"]["decelerate"]["published"] == 1


def test_curve_calibration_candidates_expand_only_execution_curves():
    spec = {
        "schema_version": "clipgen.reward.v1",
        "scene_summary": "slow for a lead vehicle",
        "components": [
            {
                "name": "noticed_lead",
                "weight": 0.4,
                "claim": {
                    "kind": "perceptual",
                    "field": "entity",
                    "any_of": ["lead_vehicle"],
                },
                "trajectory": None,
            },
            {
                "name": "slow",
                "weight": 0.6,
                "claim": {
                    "kind": "commitment",
                    "field": "speed_profile",
                    "any_of": ["decelerate"],
                    "direction": "any",
                },
                "trajectory": {
                    "feature": "speed_drop",
                    "window_s": [0.0, 6.4],
                    "floor": 0.1,
                    "full": 2.4,
                    "power": 0.3,
                },
            },
        ],
    }
    candidates = list(run_prototype._curve_calibration_candidates(spec))
    assert len(candidates) == 20
    assert spec["components"][1]["trajectory"]["full"] == 2.4
    assert candidates[0]["components"][0] == spec["components"][0]
    assert candidates[0]["components"][1]["trajectory"]["full"] == 3.0
    assert candidates[-1]["components"][1]["trajectory"]["full"] == 6.0
