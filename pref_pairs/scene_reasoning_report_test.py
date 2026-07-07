# SPDX-License-Identifier: Apache-2.0
"""
scene_reasoning_report_test.py — synthetic-row tests: verifies the Markdown
report preserves FULL CoT text verbatim (no truncation) and groups rollouts
by maneuver_class correctly, and that the PNG plot actually gets written.
No GPU/model dependency -- rows are hand-built dicts, same pattern as
action_space_variance_test.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from pref_pairs.scene_reasoning_report import (
    pick_scene_ids,
    render_scene_action_plot,
    render_scene_reasoning_markdown,
    write_scene_report,
)

_LONG_REASONING = (
    "The pedestrian ahead is stepping off the curb into our lane, and traffic "
    "in the adjacent lane prevents a lane change, so the safest action is to "
    "decelerate smoothly and come to a full stop before the crosswalk."
)


def _scene_df() -> pd.DataFrame:
    rows = [
        {
            "scene_id": "clip-a_1000", "rollout_id": 0, "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
            "maneuver_class": "stop", "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "coc_text": _LONG_REASONING,
            "mean_acceleration_mps2": -1.5, "mean_deceleration_mps2": 1.5,
            "final_lateral_offset_m": 0.0, "total_heading_change_deg": 0.0,
        },
        {
            "scene_id": "clip-a_1000", "rollout_id": 1, "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
            "maneuver_class": "stop", "waypoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "coc_text": "Line one.\nLine two of the same reasoning trace.",
            "mean_acceleration_mps2": -2.0, "mean_deceleration_mps2": 2.0,
            "final_lateral_offset_m": 0.0, "total_heading_change_deg": 0.0,
        },
        {
            "scene_id": "clip-a_1000", "rollout_id": 2, "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
            "maneuver_class": "lane_keep", "waypoints": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            "coc_text": "The pedestrian has already cleared the lane, proceed normally.",
            "mean_acceleration_mps2": 0.1, "mean_deceleration_mps2": 0.0,
            "final_lateral_offset_m": 0.0, "total_heading_change_deg": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_pick_scene_ids_is_deterministic_and_bounded():
    df = pd.concat([
        _scene_df(),
        _scene_df().assign(scene_id="clip-b_2000"),
        _scene_df().assign(scene_id="clip-c_3000"),
    ])
    picked = pick_scene_ids(df, n_scenes=2, seed=0)
    assert len(picked) == 2
    assert set(picked) <= {"clip-a_1000", "clip-b_2000", "clip-c_3000"}
    assert pick_scene_ids(df, n_scenes=2, seed=0) == picked  # same seed -> same picks


def test_render_scene_reasoning_markdown_preserves_full_coc_text_verbatim():
    df = _scene_df()
    with tempfile.TemporaryDirectory() as tmp:
        md_path = render_scene_reasoning_markdown(df, Path(tmp) / "out.md")
        text = md_path.read_text()

        # The FULL, untruncated reasoning trace must appear verbatim -- this
        # is the user's explicit ask ("full CoT! saved!"), not an excerpt.
        assert _LONG_REASONING in text
        assert "Line one." in text and "Line two of the same reasoning trace." in text
        assert "pedestrian has already cleared the lane" in text

        # Grouped under maneuver_class headers.
        assert "## stop (2 rollouts)" in text
        assert "## lane_keep (1 rollouts)" in text
        # groupby sorts alphabetically -- "lane_keep" before "stop".
        assert text.index("## lane_keep") < text.index("## stop")

        # Kinematic summary line present for each rollout.
        assert "mean_acceleration_mps2=-1.5" in text


def test_render_scene_action_plot_writes_a_nonempty_png():
    df = _scene_df()
    with tempfile.TemporaryDirectory() as tmp:
        png_path = render_scene_action_plot(df, Path(tmp) / "plot.png")
        assert png_path.exists()
        assert png_path.stat().st_size > 0


def test_write_scene_report_writes_both_files_named_by_scene_id():
    df = _scene_df()
    with tempfile.TemporaryDirectory() as tmp:
        png_path, md_path = write_scene_report(df, tmp)
        assert png_path.name == "clip-a_1000_actions.png"
        assert md_path.name == "clip-a_1000_reasoning.md"
        assert png_path.exists()
        assert md_path.exists()
