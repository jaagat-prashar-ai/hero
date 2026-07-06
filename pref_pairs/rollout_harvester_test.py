# SPDX-License-Identifier: Apache-2.0
"""
rollout_harvester_test.py — unit tests for pref_pairs.rollout_harvester.

Deliberately has NO GPU / real-model / network dependency: `_FakeModel` below
stands in for Alpamayo1_5, and `_build_tokenized_inputs` (the only method
that would call out to alpamayo1_5.helper.get_processor, which downloads a
processor config over the network) is monkeypatched out. That keeps these
tests exercising exactly the logic this file actually adds -- unpacking the
(B, ns, k, T, 3) trajectory tensor and the (B, ns, k) cot array into K
RolloutRecords, and their JSON serialization -- without needing the real
Alpamayo 1.5 weights.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pref_pairs.rollout_harvester import RolloutHarvester, _write_scene_records


class _FakeConfig:
    # Only attribute RolloutHarvester.harvest_scene actually reads off the config.
    _name_or_path = "fake-checkpoint"


class _FakeModel:
    """Stands in for Alpamayo1_5: returns fixed, easy-to-check tensors instead
    of running real VLM generation + diffusion sampling."""

    def __init__(self, num_waypoints: int) -> None:
        self.config = _FakeConfig()
        self.tokenizer = None  # unused; _build_tokenized_inputs is mocked out below
        self._num_waypoints = num_waypoints

    def sample_trajectories_from_data_with_vlm_rollout(self, data, **kwargs):
        k = kwargs["num_traj_samples"]
        t = self._num_waypoints
        # Distinct values per rollout (rollout i's waypoints are all == i)
        # so the test below can check each RolloutRecord got the RIGHT slice,
        # not just A slice, of the batched output tensor.
        pred_xyz = torch.stack([torch.full((t, 3), float(i)) for i in range(k)])
        pred_xyz = pred_xyz.unsqueeze(0).unsqueeze(0)  # -> (B=1, ns=1, k, T, 3)
        pred_rot = torch.zeros(1, 1, k, t, 3, 3)
        cot = np.array([f"reasoning-{i}" for i in range(k)]).reshape(1, 1, k)
        return pred_xyz, pred_rot, {"cot": cot}


def _make_harvester(num_waypoints: int = 5) -> RolloutHarvester:
    return RolloutHarvester(model=_FakeModel(num_waypoints), device="cpu")


def test_harvest_scene_unpacks_k_independent_rollouts():
    harvester = _make_harvester(num_waypoints=5)

    # Bypass the real chat-template / processor construction (network call) --
    # this test is only about what harvest_scene does with the model's output.
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(
            model_inputs={},
            scene_id="scene_a",
            k=3,
            seed=7,
            temperature=0.6,
            top_p=0.98,
            ground_truth_coc="nvidia verified reasoning",
        )

    assert len(records) == 3
    for i, record in enumerate(records):
        assert record.scene_id == "scene_a"
        assert record.rollout_id == i
        assert record.coc_text == f"reasoning-{i}"
        # Every waypoint in rollout i's trajectory should equal i (see _FakeModel).
        assert len(record.waypoints) == 5
        assert all(coord == float(i) for wp in record.waypoints for coord in wp)
        assert record.hz == 10.0
        assert record.sampling_params == {
            "seed": 7,
            "temperature": 0.6,
            "top_p": 0.98,
            "top_k": None,
            "k": 3,
        }
        assert record.model_version == "fake-checkpoint"
        assert record.ground_truth_coc == "nvidia verified reasoning"


def test_harvest_scene_without_ground_truth_coc():
    harvester = _make_harvester(num_waypoints=2)
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(model_inputs={}, scene_id="scene_b", k=1)
    assert records[0].ground_truth_coc is None


def test_write_scene_records_round_trips_through_json():
    harvester = _make_harvester(num_waypoints=4)
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(model_inputs={}, scene_id="scene_c", k=2, seed=1)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = _write_scene_records(records, Path(tmp), scene_id="scene_c")
        assert out_path == Path(tmp) / "scene_c.json"

        on_disk = json.loads(out_path.read_text())
        assert len(on_disk) == 2
        assert on_disk[0]["scene_id"] == "scene_c"
        assert on_disk[0]["rollout_id"] == 0
        assert on_disk[1]["rollout_id"] == 1
        assert on_disk[0]["waypoints"][0] == [0.0, 0.0, 0.0]
        assert on_disk[1]["waypoints"][0] == [1.0, 1.0, 1.0]
