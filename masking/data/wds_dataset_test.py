# SPDX-License-Identifier: Apache-2.0
"""
wds_dataset_test.py — covers _expand_clip_to_events' handling of the
"events": "None" ingestion artifact (see build_wds/data/build_webdataset.py's
matching fix): NVIDIA's own ood_reasoning.parquet has a small number of rows
where "events" is null, which build_webdataset.py's metadata writer
stringifies to the literal text "None". That's a legitimate "no sub-events
for this OOD tag" case and should be treated as an empty list quietly, while
a genuinely corrupt/unparseable events string should still warn as before.

Uses a minimal synthetic WDS sample (just enough columns for
_expand_clip_to_events to reach its events-parsing branch) rather than a real
clip -- egomotion/camera bytes are never touched because an empty sub_events
list short-circuits the per-event decode loop before either is read.
"""

from __future__ import annotations

import io
import json
import logging

import pandas as pd

from masking.data.wds_dataset import CAMERA_FEATURES, _expand_clip_to_events


def _minimal_sample(events_field) -> dict:
    # 1-second span is deliberately too short for _clamp_t0's history+future
    # window, so any real sub-event hits the safe "too short, skip" path
    # instead of reaching _interp_state (which needs x/y/z/quat columns this
    # minimal fixture doesn't have).
    ego_df = pd.DataFrame({"timestamp_us": [0, 1_000_000]})
    ego_buf = io.BytesIO()
    ego_df.to_parquet(ego_buf)
    sample = {
        "__key__": "clip-a",
        "json": json.dumps({
            "ood_events": [
                {"event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "feature": "camera_front_wide_120fov", "events": events_field}
            ],
        }),
        "egomotion.parquet": ego_buf.getvalue(),
    }
    for feat in CAMERA_FEATURES:
        sample[f"{feat}.mp4"] = b"placeholder"  # never decoded: sub_events is empty either way
    return sample


def test_stringified_none_events_yields_no_events_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="masking.data.wds_dataset"):
        items = _expand_clip_to_events(_minimal_sample("None"))
    assert items == []
    assert "could not parse stringified events field" not in caplog.text


def test_genuinely_corrupt_events_string_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="masking.data.wds_dataset"):
        items = _expand_clip_to_events(_minimal_sample("{not valid json"))
    assert items == []
    assert "could not parse stringified events field" in caplog.text


def test_valid_json_encoded_events_string_still_parses(caplog):
    # Not this bug's concern, but guards against the fix accidentally
    # breaking the normal (working) stringified-events path.
    events_json = json.dumps([{"event_start_timestamp": 0}])
    with caplog.at_level(logging.WARNING, logger="masking.data.wds_dataset"):
        items = _expand_clip_to_events(_minimal_sample(events_json))
    assert "could not parse stringified events field" not in caplog.text
    # This fixture's 1-second clip is too short for _clamp_t0's window, so the
    # event is legitimately skipped -- what matters here is that the JSON
    # string parsed cleanly, not the (empty) result of the window check.
    assert items == []
