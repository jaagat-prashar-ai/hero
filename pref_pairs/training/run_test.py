# SPDX-License-Identifier: Apache-2.0
"""
run_test.py — unit tests for pref_pairs.training.run's pure helper
functions (distributed context, scene sharding, resume bookkeeping).

Deliberately does NOT test pref_pairs_loop() itself -- that function's
heavy imports (alpamayo1_5, masking.data.wds_dataset) are intentionally
lazy (see run.py's docstring), and this file only exercises what's
importable without them, same reasoning as rollout_harvester_test.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pref_pairs.training.run import (
    ROLLOUT_FULL_LOG_MARKER,
    SCENE_SUMMARY_LOG_MARKER,
    _build_detailed_log_lines,
    _build_scene_summary_log_line,
    _distributed_context,
    _load_done_scenes,
    _resolve_device,
    _results_path,
    _scene_owner,
)


def test_distributed_context_defaults_to_single_rank():
    rank, world_size, local_rank = _distributed_context({})
    assert (rank, world_size, local_rank) == (0, 1, 0)


def test_distributed_context_reads_env_vars(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    rank, world_size, local_rank = _distributed_context({})
    assert (rank, world_size, local_rank) == (3, 8, 1)


def test_scene_owner_is_deterministic_and_within_range():
    world_size = 8
    for scene_id in ["clip_a_100", "clip_b_200", "clip_c_300"]:
        owner_first = _scene_owner(scene_id, world_size)
        owner_second = _scene_owner(scene_id, world_size)
        assert owner_first == owner_second
        assert 0 <= owner_first < world_size


def test_scene_owner_distributes_across_ranks():
    # Not a statistical test -- just confirms distinct scenes don't all
    # collapse onto the same rank (which would defeat sharding entirely).
    world_size = 4
    owners = {_scene_owner(f"scene_{i}", world_size) for i in range(50)}
    assert len(owners) > 1


def test_results_path_single_rank_vs_multi_rank():
    outdir = Path("/tmp/pref_pairs_test_outdir")
    assert _results_path(outdir, rank=0, world_size=1) == outdir / "pref_pairs_rollouts.jsonl"
    assert (
        _results_path(outdir, rank=3, world_size=8)
        == outdir / "pref_pairs_rollouts_rank03.jsonl"
    )


def test_load_done_scenes_missing_file_returns_empty_set():
    assert _load_done_scenes(Path("/tmp/definitely_does_not_exist.jsonl")) == set()


def test_load_done_scenes_reads_scene_ids_and_skips_malformed_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.jsonl"
        path.write_text(
            json.dumps({"scene_id": "scene_a", "rollout_id": 0}) + "\n"
            + json.dumps({"scene_id": "scene_a", "rollout_id": 1}) + "\n"
            + "not valid json at all\n"
            + json.dumps({"scene_id": "scene_b", "rollout_id": 0}) + "\n"
        )
        done = _load_done_scenes(path)
        assert done == {"scene_a", "scene_b"}


def test_resolve_device_falls_back_to_cpu_without_cuda():
    # This sandbox has no CUDA-enabled torch build installed, so
    # torch.cuda.is_available() is False here -- exercises the fallback
    # branch directly rather than mocking it out.
    assert _resolve_device(local_rank=0) == "cpu"


def _scene_rows() -> list[dict]:
    """Two rollouts of one scene, in exactly pref_pairs_loop's on-disk row
    schema (record.to_json_dict() + features.to_row_dict() + maneuver_class
    + event_cluster) -- enough for action_space_variance.per_clip_variance
    to compute a real summary from."""
    base = {
        "scene_id": "clip-a_1000", "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
        "maneuver_class": "stop", "coc_text": "Slowing for a pedestrian.",
        "initial_speed_mps": 5.0, "final_speed_mps": 0.0, "min_speed_mps": 0.0,
        "final_lateral_offset_m": 0.0, "total_heading_change_deg": 0.0,
        "mean_acceleration_mps2": -1.0, "mean_deceleration_mps2": 1.0, "max_deceleration_mps2": 2.0,
    }
    return [
        {**base, "rollout_id": 0, "native_accel_mps2": [-1.0, -1.0], "native_curvature_per_m": [0.0, 0.0]},
        {**base, "rollout_id": 1, "native_accel_mps2": [-1.2, -0.8], "native_curvature_per_m": [0.1, -0.1]},
    ]


def test_build_scene_summary_log_line_has_marker_and_valid_json():
    line = _build_scene_summary_log_line(_scene_rows(), expected_k=2)
    assert line.startswith(SCENE_SUMMARY_LOG_MARKER + " ")

    payload = json.loads(line[len(SCENE_SUMMARY_LOG_MARKER) + 1:])
    assert payload["scene_id"] == "clip-a_1000"
    assert payload["event_cluster"] == "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY"
    assert payload["n_rollouts"] == 2
    assert payload["complete"] is True  # expected_k=2 matches len(scene_rows)
    # Bulky per-waypoint arrays must be stripped from the summary line.
    assert "accel_per_waypoint_mean" not in payload
    assert "accel_std_mean_over_waypoints" in payload  # scalar summary kept


def test_build_scene_summary_log_line_flags_incomplete_scene():
    line = _build_scene_summary_log_line(_scene_rows(), expected_k=100)
    payload = json.loads(line[len(SCENE_SUMMARY_LOG_MARKER) + 1:])
    assert payload["complete"] is False


def test_build_detailed_log_lines_preserve_full_rows_verbatim():
    rows = _scene_rows()
    lines = _build_detailed_log_lines(rows)
    assert len(lines) == 2
    for line, row in zip(lines, rows):
        assert line.startswith(ROLLOUT_FULL_LOG_MARKER + " ")
        payload = json.loads(line[len(ROLLOUT_FULL_LOG_MARKER) + 1:])
        assert payload == row  # full row round-trips exactly, nothing dropped
