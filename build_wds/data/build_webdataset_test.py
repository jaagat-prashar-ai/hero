# SPDX-License-Identifier: Apache-2.0
"""
build_webdataset_test.py — covers _serialize_ood_event_field's handling of a
null "events" value (see masking/data/wds_dataset_test.py for the matching
downstream fix).

build_webdataset.py itself requires Python >= 3.11 (physical_ai_av's use of
typing.Self), which this sandbox's default interpreter doesn't have --
physical_ai_av is stubbed out here the same way Task 1's rollout_harvester
tests stubbed alpamayo1_5, since the function under test never touches it.
All of this module's OTHER top-level imports (boto3, scipy, webdataset,
huggingface_hub) are real and available, so only physical_ai_av needs faking.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

if "physical_ai_av" not in sys.modules:
    _stub = types.ModuleType("physical_ai_av")
    _stub.PhysicalAIAVDatasetInterface = object
    sys.modules["physical_ai_av"] = _stub

from build_wds.data.build_webdataset import _serialize_ood_event_field  # noqa: E402


def test_null_events_field_serializes_to_empty_json_array_string():
    # None is what a missing "events" value looks like by the time it reaches
    # this function for a real clip (see masking/data/wds_dataset_test.py's
    # docstring for the confirmed real-world case).
    assert _serialize_ood_event_field("events", None) == "[]"


def test_nan_events_field_also_serializes_to_empty_json_array_string():
    # Defensive: a pandas nullable-float NaN reaching this function (rather
    # than a plain None) should be caught the same way, not fall through to
    # str(nan) == "nan".
    assert _serialize_ood_event_field("events", float("nan")) == "[]"


def test_real_events_json_string_passes_through_unchanged():
    events_json = '[{"event_start_timestamp": 1000}]'
    assert _serialize_ood_event_field("events", events_json) == events_json


def test_none_in_a_non_events_field_still_stringifies_as_before():
    # Only "events" gets the null special-case -- every other field keeps the
    # existing str(v) behavior (e.g. feature/event_cluster are never null in
    # practice, but this guards against over-broadening the special case).
    assert _serialize_ood_event_field("feature", None) == "None"


def test_numpy_array_field_still_uses_tolist():
    arr = np.array([1, 2, 3])
    assert _serialize_ood_event_field("some_array_field", arr) == [1, 2, 3]
