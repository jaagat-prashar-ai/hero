# SPDX-License-Identifier: Apache-2.0
"""
perturbation_generator_test.py -- covers only the pure parsing/formatting
helpers in perturbation_generator.py (extract_ground_truth_traces,
_extract_json_object, write_perturbations_jsonl). generate_perturbation /
generate_all_perturbations call the real Fable 5 API and are deliberately
NOT covered by a mocked-client test here -- see
feedback_no_fake_model_tests: this project treats faking a real model's
behavior as out of bounds even where mocking is normal practice. Those
functions are verified via an actual --max_scenes smoke-test run against the
live API before a full batch, not via pytest.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pref_pairs.perturbation_generator import (
    _extract_json_object,
    extract_ground_truth_traces,
    write_perturbations_jsonl,
)

_FIXED_REASONING_MD = """# Scene abc123_999 -- reasoning vs. action across 2 rollouts

event_cluster: WORK_ZONES_TEMP_TRAFFIC_CONTROL

Class counts: lane_change_left=2

## lane_change_left (2 rollouts)

### rollout 0
*mean_acceleration_mps2=0.123, mean_deceleration_mps2=0.0883, final_lateral_offset_m=4.34, total_heading_change_deg=1*

> Keep distance to the lead vehicle since it is directly ahead in our lane

### rollout 1
*mean_acceleration_mps2=0.0337, mean_deceleration_mps2=0.125, final_lateral_offset_m=4.06, total_heading_change_deg=2.42*

> Keep distance to the lead vehicle since it is directly ahead in our lane
"""

_MULTILINE_REASONING_MD = """# Scene multi_777 -- reasoning vs. action across 1 rollouts

event_cluster: PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY

Class counts: yield=1

## yield (1 rollouts)

### rollout 0
*mean_acceleration_mps2=0.0, mean_deceleration_mps2=0.2, final_lateral_offset_m=0.0, total_heading_change_deg=0*

> A pedestrian is crossing 8 meters ahead.
> Ego should yield until the crosswalk is clear.
"""

_DIVERGENT_REASONING_MD = """# Scene divergent_555 -- reasoning vs. action across 2 rollouts

event_cluster: OTHER_LONGTAIL

Class counts: proceed=2

## proceed (2 rollouts)

### rollout 0
*mean_acceleration_mps2=0.0, mean_deceleration_mps2=0.0, final_lateral_offset_m=0.0, total_heading_change_deg=0*

> First rollout's reasoning text.

### rollout 1
*mean_acceleration_mps2=0.0, mean_deceleration_mps2=0.0, final_lateral_offset_m=0.0, total_heading_change_deg=0*

> Second rollout's DIFFERENT reasoning text.
"""


def test_extract_ground_truth_traces_parses_scene_id_cluster_and_first_trace():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "abc123_999_reasoning.md").write_text(_FIXED_REASONING_MD)
        traces = extract_ground_truth_traces(tmp)
        assert len(traces) == 1
        assert traces[0]["scene_id"] == "abc123_999"
        assert traces[0]["event_cluster"] == "WORK_ZONES_TEMP_TRAFFIC_CONTROL"
        assert traces[0]["trace"] == "Keep distance to the lead vehicle since it is directly ahead in our lane"


def test_extract_ground_truth_traces_joins_multiline_blockquote():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "multi_777_reasoning.md").write_text(_MULTILINE_REASONING_MD)
        traces = extract_ground_truth_traces(tmp)
        assert len(traces) == 1
        assert traces[0]["trace"] == (
            "A pedestrian is crossing 8 meters ahead.\n"
            "Ego should yield until the crosswalk is clear."
        )


def test_extract_ground_truth_traces_uses_first_rollout_when_texts_diverge():
    # Fixed-reasoning mode should make every rollout's CoT identical; if a
    # file violates that (shouldn't happen, but shouldn't crash the scan
    # either), the first rollout's text is used as ground truth.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "divergent_555_reasoning.md").write_text(_DIVERGENT_REASONING_MD)
        traces = extract_ground_truth_traces(tmp)
        assert len(traces) == 1
        assert traces[0]["trace"] == "First rollout's reasoning text."


def test_extract_ground_truth_traces_scans_multiple_files_and_sorts_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "abc123_999_reasoning.md").write_text(_FIXED_REASONING_MD)
        (Path(tmp) / "multi_777_reasoning.md").write_text(_MULTILINE_REASONING_MD)
        traces = extract_ground_truth_traces(tmp)
        assert [t["scene_id"] for t in traces] == ["abc123_999", "multi_777"]


def test_extract_json_object_passes_through_plain_json():
    text = '{"a": 1}'
    assert _extract_json_object(text) == '{"a": 1}'


def test_extract_json_object_strips_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert json.loads(_extract_json_object(text)) == {"a": 1}


def test_extract_json_object_strips_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert json.loads(_extract_json_object(text)) == {"a": 1}


def test_write_perturbations_jsonl_writes_one_json_object_per_line():
    rows = [{"scene_id": "a", "perturbation_type": "causal_flip"}, {"scene_id": "b", "perturbation_type": "spatial_error"}]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = write_perturbations_jsonl(rows, Path(tmp) / "nested" / "perturbations.jsonl")
        lines = out_path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == rows[0]
        assert json.loads(lines[1]) == rows[1]
