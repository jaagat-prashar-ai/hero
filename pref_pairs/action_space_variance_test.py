# SPDX-License-Identifier: Apache-2.0
"""
action_space_variance_test.py — synthetic-row tests for the epsilon-
calibration aggregation math (no GPU/model needed): builds rollout rows by
hand with known values so the per-clip and per-cluster statistics can be
checked against hand-computed expectations, rather than just "it runs".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pref_pairs.action_space_variance import (
    SCALAR_FEATURES,
    _clip_id_and_t0,
    load_rollouts,
    per_clip_variance,
    per_cluster_range,
    write_report,
)


def _row(
    scene_id: str,
    rollout_id: int,
    event_cluster: str,
    native_accel_mps2,
    native_curvature_per_m,
    final_lateral_offset_m: float,
    **overrides,
) -> dict:
    """One synthetic JSONL row matching pref_pairs_loop's on-disk schema
    (raw rollout fields + trajectory_features scalar columns + event_cluster)."""
    row = {
        "scene_id": scene_id,
        "rollout_id": rollout_id,
        "event_cluster": event_cluster,
        "native_accel_mps2": native_accel_mps2,
        "native_curvature_per_m": native_curvature_per_m,
        "final_lateral_offset_m": final_lateral_offset_m,
        "initial_speed_mps": 5.0,
        "final_speed_mps": 5.0,
        "min_speed_mps": 5.0,
        "total_heading_change_deg": 0.0,
        "mean_acceleration_mps2": 0.0,
        "mean_deceleration_mps2": 0.0,
        "max_deceleration_mps2": 0.0,
    }
    row.update(overrides)
    return row


def _clip_rows(clip_id: str, t0_us: int, cluster: str, accels: list[list[float]], lateral_offsets: list[float]) -> list[dict]:
    """5 rollouts for one fixed scene (clip_id, t0_us): each rollout gets its
    own accel waypoint-vector and final_lateral_offset_m, everything else
    held constant -- exercises exactly the "same scene, K stochastic draws"
    shape this module analyzes."""
    scene_id = f"{clip_id}_{t0_us}"
    return [
        _row(scene_id, i, cluster, accel, [0.1] * len(accel), lateral)
        for i, (accel, lateral) in enumerate(zip(accels, lateral_offsets))
    ]


def test_clip_id_and_t0_splits_on_last_underscore():
    # clip_id is a UUID (hyphens only) so the LAST underscore is unambiguous.
    clip_id, t0_us = _clip_id_and_t0("0abe118e-aa79-41f6-a719-f2df8abaf1ea_1814777344")
    assert clip_id == "0abe118e-aa79-41f6-a719-f2df8abaf1ea"
    assert t0_us == "1814777344"


def _build_df() -> pd.DataFrame:
    # clip A: low-variance accel (waypoint 0 always 1.0, waypoint 1 always 2.0)
    # and low-variance lateral offset -> should end up with the SMALLER
    # per-cluster variance numbers.
    clip_a = _clip_rows(
        "aaaaaaaa-0000-0000-0000-000000000000", 1000,
        "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
        accels=[[1.0, 2.0]] * 5,
        lateral_offsets=[0.10, 0.11, 0.09, 0.10, 0.10],
    )
    # clip B: high-variance accel and lateral offset -> should end up with
    # the LARGER per-cluster variance numbers, same cluster as clip A.
    clip_b = _clip_rows(
        "bbbbbbbb-0000-0000-0000-000000000000", 2000,
        "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
        accels=[[1.0, 2.0], [3.0, 0.0], [-1.0, 4.0], [2.0, 2.0], [0.0, -2.0]],
        lateral_offsets=[0.0, 2.0, -2.0, 1.0, -1.0],
    )
    # clip C: different cluster entirely, and only 3/5 "rollouts" (incomplete).
    clip_c = _clip_rows(
        "cccccccc-0000-0000-0000-000000000000", 3000,
        "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
        accels=[[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        lateral_offsets=[0.0, 0.0, 0.0],
    )[:3]
    return pd.DataFrame(clip_a + clip_b + clip_c)


def test_per_clip_variance_matches_hand_computed_stats():
    df = _build_df()
    per_clip = per_clip_variance(df, expected_k=5).set_index("clip_id")

    clip_a = per_clip.loc["aaaaaaaa-0000-0000-0000-000000000000"]
    # Every rollout is IDENTICAL for clip A -> zero std/range everywhere.
    assert clip_a["n_rollouts"] == 5
    assert clip_a["complete"]
    assert clip_a["accel_std_mean_over_waypoints"] == 0.0
    assert clip_a["accel_range_max_over_waypoints"] == 0.0
    # lateral offsets [0.10, 0.11, 0.09, 0.10, 0.10] -> range = 0.11-0.09.
    assert np.isclose(clip_a["final_lateral_offset_m_range"], 0.02)

    clip_b = per_clip.loc["bbbbbbbb-0000-0000-0000-000000000000"]
    assert clip_b["accel_std_mean_over_waypoints"] > clip_a["accel_std_mean_over_waypoints"]
    # lateral offsets [0, 2, -2, 1, -1] -> range = 2 - (-2) = 4.
    assert np.isclose(clip_b["final_lateral_offset_m_range"], 4.0)

    clip_c = per_clip.loc["cccccccc-0000-0000-0000-000000000000"]
    assert clip_c["n_rollouts"] == 3
    assert not clip_c["complete"]  # expected_k=5 but only 3 rows present
    assert clip_c["event_cluster"] == "WORK_ZONES_TEMP_TRAFFIC_CONTROL"

    # Every SCALAR_FEATURES column produced *_mean/_std/_range fields.
    for feature in SCALAR_FEATURES:
        assert f"{feature}_mean" in per_clip.columns
        assert f"{feature}_std" in per_clip.columns
        assert f"{feature}_range" in per_clip.columns


def test_per_clip_variance_skips_none_native_action_without_crashing():
    scene_id = "dddddddd-0000-0000-0000-000000000000_4000"
    rows = [
        _row(scene_id, 0, "OTHER_LONGTAIL", [1.0, 1.0], [0.0, 0.0], 0.0),
        _row(scene_id, 1, "OTHER_LONGTAIL", None, None, 0.0),  # capture failed
        _row(scene_id, 2, "OTHER_LONGTAIL", [3.0, 3.0], [0.0, 0.0], 0.0),
    ]
    per_clip = per_clip_variance(pd.DataFrame(rows))
    row = per_clip.iloc[0]
    assert row["n_rollouts"] == 3  # all 3 rollouts counted...
    assert row["accel_n_available"] == 2  # ...but only 2 had native accel captured
    assert np.isclose(row["accel_per_waypoint_mean"][0], 2.0)  # mean(1.0, 3.0)


def test_per_cluster_range_pools_across_clips_and_orders_median_le_p90_le_max():
    df = _build_df()
    per_clip = per_clip_variance(df, expected_k=5)
    per_cluster = per_cluster_range(per_clip).set_index("event_cluster")

    pedestrian = per_cluster.loc["PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY"]
    assert pedestrian["n_clips"] == 2
    assert pedestrian["n_incomplete_clips"] == 0
    m = pedestrian["accel_std_mean_over_waypoints_median"]
    p90 = pedestrian["accel_std_mean_over_waypoints_p90"]
    mx = pedestrian["accel_std_mean_over_waypoints_max"]
    assert m <= p90 <= mx

    work_zones = per_cluster.loc["WORK_ZONES_TEMP_TRAFFIC_CONTROL"]
    assert work_zones["n_clips"] == 1
    assert work_zones["n_incomplete_clips"] == 1  # clip C only had 3/5 rollouts


def test_load_rollouts_reads_multiple_rank_shards():
    df = _build_df()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        half = len(df) // 2
        for rank, chunk in enumerate([df.iloc[:half], df.iloc[half:]]):
            path = tmp_path / f"pref_pairs_rollouts_rank{rank:02d}.jsonl"
            with open(path, "w") as fh:
                for _, row in chunk.iterrows():
                    fh.write(json.dumps(row.to_dict()) + "\n")

        loaded = load_rollouts(tmp_path)
        assert len(loaded) == len(df)


def test_write_report_produces_json_and_markdown_files():
    df = _build_df()
    per_clip = per_clip_variance(df, expected_k=5)
    per_cluster = per_cluster_range(per_clip)

    with tempfile.TemporaryDirectory() as tmp:
        json_path, md_path = write_report(per_clip, per_cluster, tmp)
        assert json_path.exists() and json_path.stat().st_size > 0
        assert md_path.exists() and md_path.stat().st_size > 0

        report = json.loads(json_path.read_text())
        assert {"per_clip", "per_cluster"} == set(report.keys())
        assert len(report["per_clip"]) == 3
        assert len(report["per_cluster"]) == 2
        clip_ids = {row["clip_id"] for row in report["per_clip"]}
        assert "aaaaaaaa-0000-0000-0000-000000000000" in clip_ids

        md_text = md_path.read_text()
        assert "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY" in md_text
        assert "aaaaaaaa-0000-0000-0000-000000000000" in md_text
