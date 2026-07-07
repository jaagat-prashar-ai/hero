# SPDX-License-Identifier: Apache-2.0
"""
fetch_from_logs_test.py — verifies the log-parsing + report-building
pipeline against synthetic raw log text (the exact shape `lilypad workload
logs` returns: "[timestamp] [pod] INFO:pref_pairs.training.run:<marker> <json>"
per line), WITHOUT shelling out to the real `lilypad` CLI -- tests inject a
fake logs_fetcher instead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pref_pairs.fetch_from_logs import (
    build_action_space_variance_report,
    build_scene_reasoning_reports,
    parse_marked_lines,
)
from pref_pairs.training.run import ROLLOUT_FULL_LOG_MARKER, SCENE_SUMMARY_LOG_MARKER


def _log_line(marker: str, payload: dict) -> str:
    """Mimics lilypad workload logs' actual line shape -- marker text is
    just a substring after the logger prefix, exactly what
    parse_marked_lines looks for."""
    return f"[2026-07-07 22:43:04 UTC] [rayjob-worker-abc] INFO:pref_pairs.training.run:{marker} {json.dumps(payload)}"


def test_parse_marked_lines_ignores_unrelated_lines_and_parses_matches():
    text = "\n".join([
        "[2026-07-07 22:38:04 UTC] [head] Collecting torch==2.7.1 (from ...)",
        _log_line(SCENE_SUMMARY_LOG_MARKER, {"scene_id": "a", "n_rollouts": 2}),
        "[2026-07-07 22:39:00 UTC] [worker] some unrelated line",
        _log_line(SCENE_SUMMARY_LOG_MARKER, {"scene_id": "b", "n_rollouts": 3}),
    ])
    rows = parse_marked_lines(text, SCENE_SUMMARY_LOG_MARKER)
    assert rows == [{"scene_id": "a", "n_rollouts": 2}, {"scene_id": "b", "n_rollouts": 3}]


def test_parse_marked_lines_skips_unparseable_payload_without_crashing():
    text = "\n".join([
        _log_line(SCENE_SUMMARY_LOG_MARKER, {"scene_id": "a"}),
        f"[t] [p] INFO:x:{SCENE_SUMMARY_LOG_MARKER} {{truncated json that never clo",
        _log_line(SCENE_SUMMARY_LOG_MARKER, {"scene_id": "b"}),
    ])
    rows = parse_marked_lines(text, SCENE_SUMMARY_LOG_MARKER)
    assert rows == [{"scene_id": "a"}, {"scene_id": "b"}]


_SCENE_SUMMARY_ROW = {
    "clip_id": "clip-a", "scene_id": "clip-a_1000", "t0_us": "1000",
    "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "n_rollouts": 2, "complete": True,
    "accel_n_available": 2, "accel_std_mean_over_waypoints": 0.1, "accel_range_max_over_waypoints": 0.2,
    "curvature_n_available": 2, "curvature_std_mean_over_waypoints": 0.01, "curvature_range_max_over_waypoints": 0.02,
    "initial_speed_mps_mean": 5.0, "initial_speed_mps_std": 0.0, "initial_speed_mps_range": 0.0,
    "final_speed_mps_mean": 0.0, "final_speed_mps_std": 0.0, "final_speed_mps_range": 0.0,
    "min_speed_mps_mean": 0.0, "min_speed_mps_std": 0.0, "min_speed_mps_range": 0.0,
    "final_lateral_offset_m_mean": 0.0, "final_lateral_offset_m_std": 0.0, "final_lateral_offset_m_range": 0.0,
    "total_heading_change_deg_mean": 0.0, "total_heading_change_deg_std": 0.0, "total_heading_change_deg_range": 0.0,
    "mean_acceleration_mps2_mean": -1.0, "mean_acceleration_mps2_std": 0.1, "mean_acceleration_mps2_range": 0.2,
    "mean_deceleration_mps2_mean": 1.0, "mean_deceleration_mps2_std": 0.1, "mean_deceleration_mps2_range": 0.2,
    "max_deceleration_mps2_mean": 2.0, "max_deceleration_mps2_std": 0.1, "max_deceleration_mps2_range": 0.2,
}


def test_build_action_space_variance_report_end_to_end_from_fake_logs():
    fake_text = "\n".join([
        _log_line(SCENE_SUMMARY_LOG_MARKER, _SCENE_SUMMARY_ROW),
        _log_line(SCENE_SUMMARY_LOG_MARKER, {**_SCENE_SUMMARY_ROW, "clip_id": "clip-b", "scene_id": "clip-b_2000"}),
    ])

    def fake_fetcher(workload_id, content_filter):
        assert workload_id == "wf-123"
        assert content_filter == SCENE_SUMMARY_LOG_MARKER
        return fake_text

    with tempfile.TemporaryDirectory() as tmp:
        per_clip_df, per_cluster_df = build_action_space_variance_report(
            "wf-123", tmp, logs_fetcher=fake_fetcher,
        )
        assert len(per_clip_df) == 2
        assert len(per_cluster_df) == 1  # both rows share one event_cluster
        assert (Path(tmp) / "action_space_variance_report.json").exists()
        assert (Path(tmp) / "action_space_variance_report.md").exists()


def test_build_scene_reasoning_reports_groups_by_scene_and_writes_files():
    rollout_row = {
        "scene_id": "clip-a_1000", "rollout_id": 0, "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
        "maneuver_class": "lane_keep", "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "coc_text": "Proceeding through the work zone at reduced speed.",
        "mean_acceleration_mps2": 0.0, "mean_deceleration_mps2": 0.0,
        "final_lateral_offset_m": 0.0, "total_heading_change_deg": 0.0,
    }
    fake_text = "\n".join([
        _log_line(ROLLOUT_FULL_LOG_MARKER, rollout_row),
        _log_line(ROLLOUT_FULL_LOG_MARKER, {**rollout_row, "rollout_id": 1}),
    ])

    def fake_fetcher(workload_id, content_filter):
        return fake_text

    with tempfile.TemporaryDirectory() as tmp:
        detailed_df = build_scene_reasoning_reports("wf-123", tmp, logs_fetcher=fake_fetcher)
        assert len(detailed_df) == 2
        assert (Path(tmp) / "clip-a_1000_actions.png").exists()
        assert (Path(tmp) / "clip-a_1000_reasoning.md").exists()
        md_text = (Path(tmp) / "clip-a_1000_reasoning.md").read_text()
        assert "Proceeding through the work zone at reduced speed." in md_text
