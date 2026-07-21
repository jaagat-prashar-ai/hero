# SPDX-License-Identifier: Apache-2.0
"""
obstacle_tracks_test.py — unit tests for obstacle_tracks.py, run against
the committed REAL obstacle.offline slice in testdata/ (clip f0d61901-...,
the same clip the parser tests' scene_reasoning fixture comes from) — not
synthetic boxes, so the tests pin the actual upstream schema, dtypes, and
frame convention this module's assumptions live or die by. Expected
numbers below (41 tracks, class counts, sample counts) were read off the
fixture when it was committed; if a fixture refresh changes them, that is
a REAL upstream-data change worth noticing, not a test to loosen quietly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from code_as_a_reward.obstacle_tracks import (
    OBSTACLE_LABEL_CLASSES,
    SceneObstacles,
    load_obstacle_tracks,
)

_CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
_FIXTURE_DIR = Path(__file__).parent / "testdata"
# The scene_id this clip appears as downstream is f"{_CLIP_ID}_12988806":
# rollout futures are 64 steps at 10Hz.
_T0_US = 12_988_806
_WINDOW_US = 6_400_000


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    return pd.read_parquet(_FIXTURE_DIR / f"obstacle_offline_{_CLIP_ID}.parquet")


@pytest.fixture(scope="module")
def scene(fixture_df) -> SceneObstacles:
    return SceneObstacles.from_dataframe(fixture_df, _CLIP_ID)


def test_fixture_parses_to_expected_tracks(scene):
    assert len(scene.tracks) == 41
    by_class = {}
    for t in scene.tracks:
        by_class[t.label_class] = by_class.get(t.label_class, 0) + 1
    assert by_class == {
        "automobile": 25, "person": 11, "other_vehicle": 2,
        "trailer": 1, "rider": 1, "protruding_object": 1,
    }


def test_tracks_are_time_ordered_with_parallel_arrays(scene):
    for t in scene.tracks:
        assert np.all(np.diff(t.timestamps_us) > 0), t.track_id
        assert t.centers_m.shape == (len(t.timestamps_us), 3)
        assert t.sizes_m.shape == (len(t.timestamps_us), 3)
        assert t.label_class in OBSTACLE_LABEL_CLASSES


def test_non_rig_frame_is_rejected(fixture_df):
    # The module's whole geometric interpretation rests on the rig-frame
    # assumption — a dataset revision switching frames must be a loud
    # error, not silently-wrong bearings.
    world = fixture_df.copy()
    world.loc[world.index[:5], "reference_frame"] = "world"
    with pytest.raises(ValueError, match="rig"):
        SceneObstacles.from_dataframe(world, _CLIP_ID)


def test_unknown_label_class_is_rejected(fixture_df):
    mutated = fixture_df.copy()
    mutated.loc[mutated.index[:5], "label_class"] = "cone"  # not in the upstream vocabulary
    with pytest.raises(ValueError, match="cone"):
        SceneObstacles.from_dataframe(mutated, _CLIP_ID)


def test_window_restricts_samples_and_empty_window_is_data(scene):
    track = max(scene.tracks, key=lambda t: len(t.timestamps_us))
    windowed = track.window(_T0_US, _T0_US + _WINDOW_US)
    assert 0 < len(windowed.timestamps_us) < len(track.timestamps_us)
    assert windowed.timestamps_us.min() >= _T0_US
    assert windowed.timestamps_us.max() <= _T0_US + _WINDOW_US
    # A window before the clip started is empty — valid data, not an error.
    empty = track.window(0, 1)
    assert len(empty.timestamps_us) == 0
    assert empty.min_ego_distance_m() == float("inf")
    assert np.isnan(empty.apparent_speed_mps())


def test_actors_present_filters_by_class_distance_and_samples(scene):
    all_near = scene.actors_present(_T0_US, _T0_US + _WINDOW_US, max_distance_m=50.0)
    assert len(all_near) == 18  # read off the real fixture at commit time
    people = scene.actors_present(
        _T0_US, _T0_US + _WINDOW_US, classes={"person"}, max_distance_m=50.0
    )
    assert people
    assert all(t.label_class == "person" for t in people)
    assert set(t.track_id for t in people) <= set(t.track_id for t in all_near)
    # Tightening the distance can only shrink the set.
    very_near = scene.actors_present(_T0_US, _T0_US + _WINDOW_US, max_distance_m=5.0)
    assert len(very_near) < len(all_near)
    # An absurd min_samples excludes everything.
    assert scene.actors_present(_T0_US, _T0_US + _WINDOW_US, min_samples=10_000) == []


def test_actors_present_rejects_unknown_class(scene):
    with pytest.raises(ValueError, match="pedestrian"):
        # 'pedestrian' is the PARSER's entity key, not the dataset's class
        # name ('person') — exactly the typo family this check exists for;
        # the verifier's entity->class mapping owns that translation.
        scene.actors_present(_T0_US, _T0_US + _WINDOW_US, classes={"pedestrian"})


def test_load_obstacle_tracks_cache_hit_needs_no_network(scene):
    loaded = load_obstacle_tracks(_CLIP_ID, cache_dir=_FIXTURE_DIR)
    assert len(loaded.tracks) == len(scene.tracks)
    assert loaded.clip_id == _CLIP_ID


def test_load_obstacle_tracks_cache_miss_raises_helpfully(tmp_path):
    # Only meaningful where physical_ai_av is NOT importable (the project's
    # 3.10 env — the environment this error message exists for). Under the
    # 3.11+ alpamayo env this would go to the network instead, so skip.
    try:
        import physical_ai_av  # noqa: F401
        pytest.skip("physical_ai_av importable; cache-miss would hit the network")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pre-populate the cache"):
        load_obstacle_tracks("00000000-not-a-real-clip", cache_dir=tmp_path)
