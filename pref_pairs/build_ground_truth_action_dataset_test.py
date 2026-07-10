# SPDX-License-Identifier: Apache-2.0
"""
build_ground_truth_action_dataset_test.py — synthetic-row tests for the
rollout-selection, grouping, and join logic (no GPU/model/network): builds
rollout rows and perturbations.jsonl fixtures by hand so the join can be
checked against known expectations rather than just "it runs".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from pref_pairs.build_ground_truth_action_dataset import (
    build_dataset,
    fetch_rollout_rows,
    load_perturbations_by_scene,
    select_ground_truth_rollout,
)


def _rollout_row(scene_id: str, rollout_id: int, maneuver_class: str, coc_text: str, **overrides) -> dict:
    row = {
        "scene_id": scene_id,
        "rollout_id": rollout_id,
        "maneuver_class": maneuver_class,
        "coc_text": coc_text,
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "hz": 10.0,
        "native_accel_mps2": [0.1, 0.2],
        "native_curvature_per_m": [0.0, 0.0],
        "mean_acceleration_mps2": 0.15,
        "mean_deceleration_mps2": 0.0,
        "final_lateral_offset_m": 0.0,
        "total_heading_change_deg": 0.0,
    }
    row.update(overrides)
    return row


def _perturbation_row(scene_id: str, event_cluster: str, ground_truth_trace: str, perturbation_type: str) -> dict:
    return {
        "scene_id": scene_id,
        "event_cluster": event_cluster,
        "ground_truth_trace": ground_truth_trace,
        "trace_id": f"{scene_id}__{perturbation_type}",
        "perturbation_type": perturbation_type,
        "original_span": "foo",
        "perturbed_span": "bar",
        "perturbed_trace": ground_truth_trace.replace("foo", "bar"),
        "semantic_delta": "delta",
        "decision_impact": "impact",
        "plausibility_rationale": "rationale",
    }


def test_select_ground_truth_rollout_picks_alphabetically_first_class_then_lowest_rollout_id():
    # "lane_change_right" < "lane_keep" alphabetically, so lane_change_right's
    # rollouts must be preferred over lane_keep's even though lane_keep has
    # more rows and a lower rollout_id overall (rollout 1) -- this mirrors
    # scene_reasoning_report.render_scene_reasoning_markdown's groupby+sort.
    scene_df = pd.DataFrame([
        _rollout_row("s1", 1, "lane_keep", "keep text A"),
        _rollout_row("s1", 2, "lane_keep", "keep text B"),
        _rollout_row("s1", 13, "lane_change_right", "change text A"),
        _rollout_row("s1", 29, "lane_change_right", "change text B"),
    ])
    picked = select_ground_truth_rollout(scene_df)
    assert picked["maneuver_class"] == "lane_change_right"
    assert picked["rollout_id"] == 13
    assert picked["coc_text"] == "change text A"


def test_fetch_rollout_rows_dedupes_by_scene_and_rollout_id():
    # Simulates the known OCI dual-log-source double-ingest: each real log
    # line appears twice in the raw text.
    marker = "PREF_PAIRS_ROLLOUT_FULL"
    row = _rollout_row("s1", 0, "lane_keep", "text")
    line = f"{marker} {json.dumps(row)}"
    log_text = "\n".join([line, line])  # duplicated delivery

    df = fetch_rollout_rows("fake-workload", logs_fetcher=lambda _wid: log_text)
    assert len(df) == 1
    assert df.iloc[0]["scene_id"] == "s1"


def test_load_perturbations_by_scene_groups_rows():
    rows = [
        _perturbation_row("s1", "WORK_ZONES", "keep foo distance", "negation_flip"),
        _perturbation_row("s1", "WORK_ZONES", "keep foo distance", "spatial_error"),
        _perturbation_row("s2", "OTHER", "yield to foo", "causal_flip"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "perturbations.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        by_scene = load_perturbations_by_scene(path)

    assert set(by_scene) == {"s1", "s2"}
    assert len(by_scene["s1"]["perturbations"]) == 2
    assert len(by_scene["s2"]["perturbations"]) == 1
    assert by_scene["s1"]["ground_truth_trace"] == "keep foo distance"
    assert by_scene["s1"]["perturbations"][0]["perturbation_type"] == "negation_flip"


def test_build_dataset_joins_action_onto_matching_scene():
    rollout_df = pd.DataFrame([
        _rollout_row("s1", 13, "lane_change_right", "keep foo distance"),
        _rollout_row("s1", 1, "lane_keep", "other text"),
    ])
    perturbations_by_scene = {
        "s1": {
            "scene_id": "s1",
            "event_cluster": "WORK_ZONES",
            "ground_truth_trace": "keep foo distance",
            "perturbations": [{"perturbation_type": "negation_flip"}],
        },
    }

    dataset = build_dataset(rollout_df, perturbations_by_scene)

    assert len(dataset) == 1
    entry = dataset[0]
    assert entry["scene_id"] == "s1"
    assert entry["ground_truth_action"]["rollout_id"] == 13
    assert entry["ground_truth_action"]["maneuver_class"] == "lane_change_right"
    assert isinstance(entry["ground_truth_action"]["rollout_id"], int)  # not numpy.int64
    assert isinstance(entry["ground_truth_action"]["mean_acceleration_mps2"], float)
    assert entry["perturbations"] == perturbations_by_scene["s1"]["perturbations"]


def test_build_dataset_skips_scene_missing_from_rollout_logs():
    rollout_df = pd.DataFrame([_rollout_row("s1", 0, "lane_keep", "text")])
    perturbations_by_scene = {
        "s1": {"scene_id": "s1", "event_cluster": "X", "ground_truth_trace": "text", "perturbations": []},
        "s2_missing": {"scene_id": "s2_missing", "event_cluster": "X", "ground_truth_trace": "text", "perturbations": []},
    }

    dataset = build_dataset(rollout_df, perturbations_by_scene)

    assert [e["scene_id"] for e in dataset] == ["s1"]


def test_build_dataset_includes_scene_despite_coc_text_mismatch():
    # A mismatch is logged as a warning, not dropped -- the scene should
    # still appear in the output so a reviewer can see it flagged rather
    # than silently losing data.
    rollout_df = pd.DataFrame([_rollout_row("s1", 0, "lane_keep", "actual model text")])
    perturbations_by_scene = {
        "s1": {"scene_id": "s1", "event_cluster": "X", "ground_truth_trace": "different expected text", "perturbations": []},
    }

    dataset = build_dataset(rollout_df, perturbations_by_scene)

    assert len(dataset) == 1
    assert dataset[0]["ground_truth_action"]["rollout_id"] == 0
