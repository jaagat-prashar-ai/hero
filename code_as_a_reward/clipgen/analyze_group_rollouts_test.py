# SPDX-License-Identifier: Apache-2.0
"""Offline tests for analyze_group_rollouts.py, against hand-built rollout
dumps -- no S3, no real training run needed (per the plan's "dry-run before
touching the training path" step). Reuses clipgen_test.py's GT_COC/GOOD_FN
fixtures so the "good" scene's pass/fail is pinned by the SAME reward
function that test_gate_passes_scene_aware_function already validates
against GT -- here it has to also work on rollouts it never saw.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from code_as_a_reward.clipgen import analyze_group_rollouts as agr
from code_as_a_reward.clipgen.clipgen_test import GOOD_FN, GT_COC, LENIENT_FN, _reactive_waypoints
from code_as_a_reward.clipgen.gate import flattened_waypoints

CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
HZ = 10.0


def _wp3(wp2: np.ndarray) -> list[list[float]]:
    return np.column_stack([wp2, np.zeros(len(wp2))]).tolist()


def _dump(clip_id: str, rollouts: list[dict]) -> dict:
    return {"scene_id": f"{clip_id}_0", "clip_id": clip_id, "hz": HZ, "rollouts": rollouts}


def test_score_scene_trusts_a_genuinely_good_argmax():
    """A rollout matching GT_COC's claims + a real reactive trajectory
    should win the group AND survive its own perturbation gate."""
    reactive = _reactive_waypoints()
    dump = _dump(
        CLIP_ID,
        [
            {"rollout_id": 0, "coc_text": GT_COC, "waypoints": _wp3(reactive), "reward": 0.4},
            {
                "rollout_id": 1,
                "coc_text": GT_COC,
                "waypoints": _wp3(reactive[::-1].copy()),
                "reward": 0.1,
            },
            {
                "rollout_id": 2,
                "coc_text": GT_COC,
                "waypoints": _wp3(flattened_waypoints(reactive)),
                "reward": 0.1,
            },
            {
                "rollout_id": 3,
                "coc_text": "I will proceed straight ahead, nothing notable in view.",
                "waypoints": _wp3(flattened_waypoints(reactive)),
                "reward": -0.2,
            },
        ],
    )
    record = agr.score_scene(dump, GOOD_FN)
    assert record["argmax_rollout_id"] == 0, record["rollouts"]
    assert record["argmax_gate"]["passed"], record["argmax_gate"]["failures"]
    assert record["argmax_gate"]["pos_score"] >= 0.7
    # Every rollout got a clipgen score and a components breakdown.
    for r in record["rollouts"]:
        assert np.isfinite(r["clipgen_score"])
        assert r["clipgen_components"] is not None
    # The record round-trips through JSON (what analyze() actually persists).
    json.dumps(record, default=str)


def test_group_validation_requires_rank_resolution_and_sensitive_winner():
    reactive = _reactive_waypoints()
    rollouts = [
        {"rollout_id": 0, "coc_text": GT_COC, "waypoints": _wp3(reactive)},
        {"rollout_id": 1, "coc_text": GT_COC, "waypoints": _wp3(reactive[::-1].copy())},
        {"rollout_id": 2, "coc_text": GT_COC, "waypoints": _wp3(flattened_waypoints(reactive))},
        {
            "rollout_id": 3,
            "coc_text": "I will proceed straight ahead, nothing notable in view.",
            "waypoints": _wp3(flattened_waypoints(reactive)),
        },
    ]
    result = agr.validate_rollout_group(
        CLIP_ID, f"{CLIP_ID}_holdout", HZ, rollouts, GOOD_FN, top_k=1
    )
    assert result.passed, result.failures
    assert result.unique_scores >= 3
    assert result.score_range >= 0.15

    flat = agr.validate_rollout_group(
        CLIP_ID,
        f"{CLIP_ID}_flat",
        HZ,
        rollouts,
        LENIENT_FN,
        top_k=1,
    )
    assert not flat.passed
    assert any("std" in failure or "distinct" in failure for failure in flat.failures)


def test_score_scene_catches_untrustworthy_argmax():
    """A reward function that can't discriminate (LENIENT_FN, flat 0.9)
    "wins" some rollout by tie-break, but that argmax must FAIL its own
    gate -- this is the concrete "best of a bad batch" case select-then-
    verify exists to catch."""
    reactive = _reactive_waypoints()
    dump = _dump(
        "fake-clip-lenient",
        [
            {"rollout_id": 0, "coc_text": GT_COC, "waypoints": _wp3(reactive), "reward": 0.9},
            {
                "rollout_id": 1,
                "coc_text": GT_COC,
                "waypoints": _wp3(reactive[::-1].copy()),
                "reward": 0.9,
            },
        ],
    )
    record = agr.score_scene(dump, LENIENT_FN)
    assert record["argmax_rollout_id"] is not None
    assert record["argmax_gate"] is not None
    assert not record["argmax_gate"]["passed"]
    assert any("must not be rewarded" in f for f in record["argmax_gate"]["failures"])


def test_score_scene_no_finite_rollout_leaves_gate_none():
    """A reward function that raises on every rollout must not crash the
    batch -- just no argmax/gate for that scene."""

    def _bad_source():
        return "def reward(claims, traj):\n    raise ValueError('boom')\n"

    dump = _dump(CLIP_ID, [{"rollout_id": 0, "coc_text": GT_COC, "waypoints": _wp3(_reactive_waypoints())}])
    record = agr.score_scene(dump, _bad_source())
    assert record["argmax_rollout_id"] is None
    assert record["argmax_gate"] is None
    assert not np.isfinite(record["rollouts"][0]["clipgen_score"])
    # The actual exception must be captured, not silently discarded -- a
    # "why did every rollout fail" question needs an answer, not just a fact.
    assert "boom" in record["rollouts"][0]["clipgen_error"]


def test_load_reward_source_local_missing_returns_none(tmp_path):
    assert agr.load_reward_source(str(tmp_path), "no-such-clip") is None


def test_load_reward_source_local_present(tmp_path):
    (tmp_path / "clip_a.py").write_text(GOOD_FN)
    assert agr.load_reward_source(str(tmp_path), "clip_a") == GOOD_FN


def test_iter_dump_files_local(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "b.json").write_text(json.dumps({"x": 2}))
    (tmp_path / "ignore.txt").write_text("not json")
    found = {json.loads(raw)["x"] for _, raw in agr.iter_dump_files(str(tmp_path))}
    assert found == {1, 2}


def test_build_heatmaps_writes_file(tmp_path):
    records = [
        {
            "clip_id": "clip_a",
            "argmax_gate": {"passed": True},
            "rollouts": [
                {"clipgen_score": 0.9, "reward": 0.2},
                {"clipgen_score": 0.1, "reward": -0.5},
            ],
        },
        {
            "clip_id": "clip_b",
            "argmax_gate": {"passed": False},
            "rollouts": [
                {"clipgen_score": 0.5, "reward": 0.0},
                {"clipgen_score": float("nan"), "reward": -0.1},
            ],
        },
    ]
    out = tmp_path / "heatmap.png"
    agr.build_heatmaps(records, out)
    assert out.exists() and out.stat().st_size > 0


def test_analyze_end_to_end_local_no_overlays(tmp_path):
    """The full analyze() path against local dirs, overlays disabled (no
    S3/warm-cache dependency) -- exercises dump discovery, reward-source
    loading, scoring, JSON persistence, and heatmap generation together."""
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    fns_dir = tmp_path / "reward_fns"
    fns_dir.mkdir()
    out_dir = tmp_path / "out"

    reactive = _reactive_waypoints()
    dump = _dump(
        CLIP_ID,
        [
            {"rollout_id": 0, "coc_text": GT_COC, "waypoints": _wp3(reactive), "reward": 0.4},
            {
                "rollout_id": 1,
                "coc_text": GT_COC,
                "waypoints": _wp3(reactive[::-1].copy()),
                "reward": 0.1,
            },
        ],
    )
    (dump_dir / "000001_f0d61901.json").write_text(json.dumps(dump))
    (fns_dir / f"{CLIP_ID}.py").write_text(GOOD_FN)
    # A dump with no cached reward fn must be skipped, not crash the run.
    (dump_dir / "000002_untouched.json").write_text(json.dumps(_dump("no-reward-fn-clip", [])))

    records = agr.analyze(str(dump_dir), str(fns_dir), str(out_dir), render_images=False)

    assert len(records) == 1
    assert records[0]["clip_id"] == CLIP_ID
    assert (out_dir / f"{records[0]['scene_id']}.json").exists()
    assert (out_dir / "heatmap.png").exists()
