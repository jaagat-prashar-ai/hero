# SPDX-License-Identifier: Apache-2.0
"""
build_clean_reasoning_actions_test.py — synthetic-scene tests for the
extraction logic (no GPU/model/network needed).
"""

from __future__ import annotations

from pref_pairs.build_clean_reasoning_actions import build_clean_entries


def test_build_clean_entries_drops_perturbations():
    scenes = [
        {
            "scene_id": "s1",
            "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
            "ground_truth_trace": "keep distance to the lead vehicle",
            "ground_truth_action": {"rollout_id": 0, "maneuver_class": "lane_keep", "waypoints": [[0, 0, 0]]},
            "perturbations": [{"perturbation_type": "negation_flip"}],
        },
    ]
    entries = build_clean_entries(scenes)
    assert len(entries) == 1
    entry = entries[0]
    assert "perturbations" not in entry
    assert entry["scene_id"] == "s1"
    assert entry["ground_truth_trace"] == "keep distance to the lead vehicle"
    assert entry["ground_truth_action"]["maneuver_class"] == "lane_keep"


def test_build_clean_entries_one_per_scene():
    scenes = [
        {"scene_id": "s1", "event_cluster": "A", "ground_truth_trace": "t1", "ground_truth_action": {}, "perturbations": []},
        {"scene_id": "s2", "event_cluster": "B", "ground_truth_trace": "t2", "ground_truth_action": {}, "perturbations": []},
    ]
    entries = build_clean_entries(scenes)
    assert [e["scene_id"] for e in entries] == ["s1", "s2"]
