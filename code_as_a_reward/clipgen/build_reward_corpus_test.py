# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

from code_as_a_reward.clipgen.build_reward_corpus import build_corpus


SPEC = {
    "schema_version": "clipgen.reward.v1",
    "scene_summary": "slow for a pedestrian",
    "components": [
        {
            "name": "noticed_pedestrian",
            "weight": 0.4,
            "claim": {
                "kind": "perceptual",
                "field": "entity",
                "any_of": ["pedestrian"],
            },
            "trajectory": None,
        },
        {
            "name": "slowing_execution",
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
                "full": 2.0,
                "power": 0.5,
            },
        },
    ],
}


def _write_artifact(run_dir, *, passed: bool, rollouts_used: bool = False):
    specs = run_dir / "reward_specs"
    specs.mkdir(parents=True)
    artifact = {
        "schema_version": "clipgen-v2",
        "pipeline_mode": "offline_gt_only",
        "clip_id": "clip-a",
        "spec": SPEC,
        "provenance": {"policy_rollouts_used": rollouts_used},
        "validation": {
            "gt_target": {"passed": True},
            "gt_semantic_gate": {
                "passed": passed,
                "pos_score": 0.9,
                "max_pert": 0.4,
            },
        },
    }
    (specs / "clip-a.json").write_text(json.dumps(artifact))


def test_build_corpus_accepts_gt_only_semantic_gate(tmp_path):
    run_dir, out_dir = tmp_path / "run", tmp_path / "corpus"
    _write_artifact(run_dir, passed=True)
    manifest = build_corpus([str(run_dir)], str(out_dir))
    assert manifest["schema_version"] == "clipgen.corpus.v2"
    assert manifest["n_clips"] == 1
    assert manifest["clips"][0]["pipeline_mode"] == "offline_gt_only"
    assert (out_dir / "clip-a.py").exists()


def test_build_corpus_rejects_failed_or_rollout_tainted_offline_artifact(tmp_path):
    for name, passed, rollouts_used in (
        ("failed", False, False),
        ("tainted", True, True),
    ):
        run_dir, out_dir = tmp_path / name, tmp_path / f"{name}-out"
        _write_artifact(run_dir, passed=passed, rollouts_used=rollouts_used)
        assert build_corpus([str(run_dir)], str(out_dir))["n_clips"] == 0
