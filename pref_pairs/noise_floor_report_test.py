# SPDX-License-Identifier: Apache-2.0
"""
noise_floor_report_test.py — covers the two pieces of real logic in
noise_floor_report.py (Markdown parsing + per-cluster stat aggregation); the
HTML renderer is checked only for well-formedness (balanced tags), since its
content is a direct pass-through of already-tested upstream data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pref_pairs.noise_floor_report import build_report_data, load_videos_b64, parse_reasoning_md, render_html


def test_parse_reasoning_md_ranks_quotes_by_frequency_per_class():
    text = """# Scene x -- reasoning vs. action across 4 rollouts

event_cluster: WORK_ZONES_TEMP_TRAFFIC_CONTROL

Class counts: lane_keep=3, stop=1

## lane_keep (3 rollouts)

### rollout 0
*mean_acceleration_mps2=0.1*

> Proceed through the work zone at reduced speed.

### rollout 1
*mean_acceleration_mps2=0.1*

> Proceed through the work zone at reduced speed.

### rollout 2
*mean_acceleration_mps2=0.1*

> Stay in lane behind the lead vehicle.

## stop (1 rollouts)

### rollout 3
*mean_acceleration_mps2=-0.5*

> Stop for the flagger.
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene_reasoning.md"
        path.write_text(text)
        class_counts, quotes_by_class = parse_reasoning_md(path)

    assert class_counts == {"lane_keep": 3, "stop": 1}
    lane_keep = quotes_by_class["lane_keep"]
    assert lane_keep["top"][0] == ("Proceed through the work zone at reduced speed.", 2)
    assert lane_keep["n_unique"] == 2  # "Proceed..." (x2) and "Stay in lane..." (x1)
    assert lane_keep["n_rollouts"] == 3
    assert quotes_by_class["stop"]["top"] == [("Stop for the flagger.", 1)]
    assert quotes_by_class["stop"]["n_unique"] == 1


_ROW_TEMPLATE = {
    "n_rollouts": 100, "complete": True,
    "accel_std_mean_over_waypoints": 0.3, "curvature_std_mean_over_waypoints": 0.01,
    "final_lateral_offset_m_std": 1.0, "total_heading_change_deg_std": 10.0,
}


def _write_fixture(tmp: Path, per_clip: list[dict]) -> None:
    report = {"per_clip": per_clip, "per_cluster": []}
    (tmp / "action_space_variance_report.json").write_text(json.dumps(report))
    (tmp / "scene_reasoning").mkdir()


def test_build_report_data_groups_by_cluster_and_clip_and_computes_stats():
    rows = [
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_100", "t0_us": "100",
         "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "accel_std_mean_over_waypoints": 0.2},
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_200", "t0_us": "200",
         "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "accel_std_mean_over_waypoints": 0.4},
        {**_ROW_TEMPLATE, "clip_id": "b", "scene_id": "b_50", "t0_us": "50",
         "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "accel_std_mean_over_waypoints": 0.6},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fixture(tmp_path, rows)
        data = build_report_data(tmp_path)

    wz = data["clusters"]["WORK_ZONES_TEMP_TRAFFIC_CONTROL"]
    assert wz["n_clips"] == 1  # both scenes belong to clip "a"
    assert wz["n_scenes"] == 2
    assert [s["t0_us"] for s in wz["clips"][0]["scenes"]] == ["100", "200"]  # sorted by t0
    assert wz["stats"]["accel_std"]["median"] == pytest.approx(0.3)  # median of [0.2, 0.4]

    ped = data["clusters"]["PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY"]
    assert ped["n_clips"] == 1
    assert ped["clips"][0]["scenes"][0]["reasoning"] is None  # no matching .md file written


def test_build_report_data_rejects_duplicate_scene_ids():
    rows = [
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_100", "t0_us": "100", "event_cluster": "OTHER_LONGTAIL"},
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_100", "t0_us": "100", "event_cluster": "OTHER_LONGTAIL"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fixture(tmp_path, rows)
        try:
            build_report_data(tmp_path)
            assert False, "expected ValueError for duplicate scene_ids"
        except ValueError:
            pass


def test_render_html_produces_balanced_details_tags():
    rows = [
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_100", "t0_us": "100", "event_cluster": "OTHER_LONGTAIL"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fixture(tmp_path, rows)
        data = build_report_data(tmp_path)
    out = render_html(data)
    assert out.count("<details") == out.count("</details>")
    assert "OTHER_LONGTAIL".title().replace("Or", "or") not in out  # sanity: label gets human-cased, not raw enum
    assert "Other Longtail" in out


def test_render_html_embeds_video_only_for_scenes_with_one():
    rows = [
        {**_ROW_TEMPLATE, "clip_id": "a", "scene_id": "a_100", "t0_us": "100", "event_cluster": "OTHER_LONGTAIL"},
        {**_ROW_TEMPLATE, "clip_id": "b", "scene_id": "b_200", "t0_us": "200", "event_cluster": "OTHER_LONGTAIL"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_fixture(tmp_path, rows)
        data = build_report_data(tmp_path)
    out = render_html(data, video_b64_by_scene={"a_100": "ZmFrZS1tcDQtYnl0ZXM="})
    assert out.count('<video class="scene-video"') == 1
    assert "data:video/mp4;base64,ZmFrZS1tcDQtYnl0ZXM=" in out


def test_load_videos_b64_reads_mp4s_keyed_by_scene_id():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "a_100.mp4").write_bytes(b"fake-mp4-bytes")
        (tmp_path / "not_a_video.txt").write_bytes(b"ignored")
        result = load_videos_b64(tmp_path)
    assert list(result.keys()) == ["a_100"]
    import base64
    assert base64.b64decode(result["a_100"]) == b"fake-mp4-bytes"


def test_load_videos_b64_returns_empty_for_missing_dir():
    assert load_videos_b64("/nonexistent/path") == {}
