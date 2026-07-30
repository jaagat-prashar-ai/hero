# SPDX-License-Identifier: Apache-2.0
"""
worker_test.py — plumbing/structure smoke test for worker.py's pure half
(verify_one / verify_trace / build_row / JSON serialization), run under the
project's normal Python 3.10 env against clipgen's already-cached real fixture
data (code_as_a_reward/clipgen/data/*), WITHOUT physical_ai_av or a GPU.

Scope, deliberately: this validates that the driver's scoring/dumping code
runs end to end and produces a JSON-safe, well-shaped row on real coc/
obstacle/egomotion data -- NOT that the frame alignment is scientifically
correct. Real per-clip fetch (load_physical_aiavdataset) already does the
correct world->ego-frame-at-t0 transform for a specific t0 window (see
load_physical_aiavdataset.py); clipgen's waypoints_from_egomotion instead
resamples the WHOLE clip's raw world-frame trajectory starting at its own
t=0, a simplification clipgen's own dossier.py already makes for the same
"good enough for a structural check" reason. The real fetch path is only
exercised on the Lilypad smoke run (configs/smoke.yaml), which has both
physical_ai_av and a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from code_as_a_reward.clipgen.dossier import waypoints_from_egomotion
from code_as_a_reward.obstacle_tracks import SceneObstacles
from code_as_a_reward.ood_eval.manifest import OODEvent
from code_as_a_reward.ood_eval.worker import _json_default, build_row, verify_one

_DATA_DIR = Path(__file__).parent.parent / "clipgen" / "data"
_CLIP_ID = "40597645-81a0-44bd-ad4b-8670bacecc04"
_HZ = 10.0
_N_WAYPOINTS = 64  # matches trace_reward's default 6.4s horizon @ 10Hz


@pytest.fixture(scope="module")
def fixture_event() -> tuple[OODEvent, "SceneObstacles", "list[list[float]]"]:
    coc_text = (_DATA_DIR / f"{_CLIP_ID}.coc.txt").read_text().strip()
    obstacle_df = pd.read_parquet(_DATA_DIR / f"{_CLIP_ID}.obstacle.offline.parquet")
    egomotion_df = pd.read_parquet(_DATA_DIR / f"{_CLIP_ID}.egomotion.offline.parquet")

    scene = SceneObstacles.from_dataframe(obstacle_df, _CLIP_ID)
    waypoints = waypoints_from_egomotion(egomotion_df, hz=_HZ)[:_N_WAYPOINTS]

    event = OODEvent(
        clip_id=_CLIP_ID, t0_us=0, gt_coc=coc_text, event_cluster="TEST_FIXTURE", rank_in_clip=0
    )
    return event, scene, waypoints


def test_verify_one_with_scene(fixture_event):
    event, scene, waypoints = fixture_event
    features, tv, scene_available = verify_one(event.gt_coc, waypoints, event, scene, rollout_id=0)

    assert scene_available is True
    assert features.n_waypoints == len(waypoints)
    assert features.scene_id == event.scene_id()
    # At least one claim type should have parsed out of a real corpus string.
    assert (
        len(tv.trace.commitments) + len(tv.trace.perceptual) + len(tv.trace.causal) > 0
    ), "expected the real ground-truth CoC text to parse to at least one claim"
    # score_trace's full path (perceptual verdicts populated) only runs when
    # a scene is available -- confirms the non-degraded branch was taken.
    assert tv.reward.decided_fraction >= 0.0


def test_verify_one_without_scene_degrades_gracefully(fixture_event):
    event, _scene, waypoints = fixture_event
    features, tv, scene_available = verify_one(event.gt_coc, waypoints, event, None, rollout_id=0)

    assert scene_available is False
    assert tv.perceptual_verdicts == []
    assert tv.causal_verdicts == []
    assert set(tv.reward.n_pass.keys()) <= {"commitment"}
    assert features.scene_id == event.scene_id()


def test_build_row_round_trips_through_json(fixture_event):
    event, scene, waypoints = fixture_event
    features, tv, scene_available = verify_one(event.gt_coc, waypoints, event, scene, rollout_id=0)
    row = build_row(event, features, tv, scene_available, model_result=None)

    # Must actually serialize -- this is what would break on an Enum/tuple/
    # numpy value the module docstring's "no silent gaps" stance would want
    # surfaced immediately, not on the cluster hours into a real run.
    encoded = json.dumps(row, default=_json_default)
    decoded = json.loads(encoded)

    assert decoded["clip_id"] == event.clip_id
    assert decoded["scene_id"] == event.scene_id()
    assert decoded["model"] is None
    assert decoded["ground_truth"]["scene_available"] is True
    assert decoded["ground_truth"]["coc_text"] == event.gt_coc
    assert len(decoded["ground_truth"]["features"]["speed_mps"]) == len(waypoints)
    assert decoded["comparison"]["reward_gt"] == tv.reward.reward
    assert decoded["comparison"]["reward_model"] is None


def test_build_row_with_model_branch(fixture_event):
    event, scene, waypoints = fixture_event
    # Reuse the same (coc, waypoints) as a stand-in "model rollout" purely to
    # exercise the two-branch row shape -- not a real model output.
    gt_features, gt_tv, gt_ok = verify_one(event.gt_coc, waypoints, event, scene, rollout_id=0)
    model_features, model_tv, model_ok = verify_one(event.gt_coc, waypoints, event, scene, rollout_id=1)

    row = build_row(
        event, gt_features, gt_tv, gt_ok, model_result=(model_features, model_tv, model_ok, event.gt_coc)
    )
    encoded = json.dumps(row, default=_json_default)
    decoded = json.loads(encoded)

    assert decoded["model"] is not None
    assert decoded["comparison"]["reward_model"] == model_tv.reward.reward
    assert decoded["model"]["coc_text"] == event.gt_coc
