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


class _FakeActionSpace:
    """Stands in for the real action_space -- specifically, its
    action_to_traj method, which is the one thing
    _sample_with_captured_action spies on to recover the native action
    tensor. accel_mean/std and curvature_mean/std are the identity
    transform (mean=0, std=1) so denormalized == raw, keeping the test's
    expected values simple."""

    def __init__(self) -> None:
        self.accel_mean = torch.tensor(0.0)
        self.accel_std = torch.tensor(1.0)
        self.curvature_mean = torch.tensor(0.0)
        self.curvature_std = torch.tensor(1.0)

    def action_to_traj(self, action, hist_xyz, hist_rot):
        # action: (k, T, 2) normalized [accel, kappa]. Reuse the "rollout i's
        # waypoints are all == i" pattern so the test can still check each
        # RolloutRecord got the RIGHT batch slice, same as before this
        # native-action-capture change existed.
        k, t, _ = action.shape
        pred_xyz = torch.stack([torch.full((t, 3), float(i)) for i in range(k)])
        pred_rot = torch.zeros(k, t, 3, 3)
        return pred_xyz, pred_rot


class _FakeModel:
    """Stands in for Alpamayo1_5: returns fixed, easy-to-check tensors instead
    of running real VLM generation + diffusion sampling."""

    def __init__(self, num_waypoints: int) -> None:
        self.config = _FakeConfig()
        self.tokenizer = None  # unused; _build_tokenized_inputs is mocked out below
        self.action_space = _FakeActionSpace()
        self._num_waypoints = num_waypoints

    def sample_trajectories_from_data_with_vlm_rollout(self, data, **kwargs):
        k = kwargs["num_traj_samples"]
        t = self._num_waypoints
        # Native action: rollout i gets accel=i, curvature=-i (distinct
        # per-rollout values, opposite signs, so a test can't mix the two
        # up by accident). Calling self.action_space.action_to_traj (rather
        # than building pred_xyz directly, as the pre-native-capture version
        # of this fake did) means _sample_with_captured_action's spy -- which
        # replaces this exact attribute -- actually gets exercised here, the
        # same way it would against the real upstream method.
        action = torch.stack(
            [
                torch.stack([torch.full((t,), float(i)), torch.full((t,), float(-i))], dim=-1)
                for i in range(k)
            ]
        )  # (k, T, 2)
        pred_xyz, pred_rot = self.action_space.action_to_traj(action, None, None)
        pred_xyz = pred_xyz.unsqueeze(0).unsqueeze(0)  # -> (B=1, ns=1, k, T, 3)
        pred_rot = pred_rot.unsqueeze(0).unsqueeze(0)
        cot = np.array([f"reasoning-{i}" for i in range(k)]).reshape(1, 1, k)
        meta_action = np.array([f"meta_action-{i}" for i in range(k)]).reshape(1, 1, k)
        answer = np.array([f"answer-{i}" for i in range(k)]).reshape(1, 1, k)
        return pred_xyz, pred_rot, {"cot": cot, "meta_action": meta_action, "answer": answer}


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
        assert record.meta_action_text == f"meta_action-{i}"
        assert record.answer_text == f"answer-{i}"
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
        # Native action capture (see rollout_harvester.py's "NATIVE ACTION
        # CAPTURE" docstring note): rollout i's native accel/curvature
        # should be exactly i / -i, per _FakeModel's construction, denormalized
        # through the identity-transform _FakeActionSpace.
        assert record.native_accel_mps2 == [float(i)] * 5
        assert record.native_curvature_per_m == [float(-i)] * 5


def test_harvest_scene_without_ground_truth_coc():
    harvester = _make_harvester(num_waypoints=2)
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(model_inputs={}, scene_id="scene_b", k=1)
    assert records[0].ground_truth_coc is None


def test_capture_failure_falls_back_to_none_gracefully():
    """If action_to_traj is never called (some hypothetical upstream code
    path that skips it), _sample_with_captured_action must not crash --
    capture just comes back empty, and RolloutRecord's native_* fields
    should be None rather than raising."""

    class _FakeModelNoCapture(_FakeModel):
        def sample_trajectories_from_data_with_vlm_rollout(self, data, **kwargs):
            k, t = kwargs["num_traj_samples"], self._num_waypoints
            # Deliberately does NOT call self.action_space.action_to_traj --
            # simulates a code path the spy never gets to observe.
            pred_xyz = torch.zeros(1, 1, k, t, 3)
            pred_rot = torch.zeros(1, 1, k, t, 3, 3)
            cot = np.array([f"reasoning-{i}" for i in range(k)]).reshape(1, 1, k)
            meta_action = np.array([f"meta_action-{i}" for i in range(k)]).reshape(1, 1, k)
            answer = np.array([f"answer-{i}" for i in range(k)]).reshape(1, 1, k)
            return pred_xyz, pred_rot, {"cot": cot, "meta_action": meta_action, "answer": answer}

    harvester = RolloutHarvester(model=_FakeModelNoCapture(num_waypoints=3), device="cpu")
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(model_inputs={}, scene_id="scene_no_capture", k=2)

    assert len(records) == 2
    assert all(r.native_accel_mps2 is None for r in records)
    assert all(r.native_curvature_per_m is None for r in records)


def test_captured_action_to_traj_is_always_restored_even_on_error():
    """A failure partway through sampling must not leave action_space
    permanently monkeypatched -- _sample_with_captured_action's try/finally
    is the thing guaranteeing this."""
    from pref_pairs.rollout_harvester import _sample_with_captured_action

    model = _FakeModel(num_waypoints=2)
    original = model.action_space.action_to_traj

    class _Boom(Exception):
        pass

    def _raise(*args, **kwargs):
        raise _Boom("simulated failure mid-sample")

    with mock.patch.object(
        model, "sample_trajectories_from_data_with_vlm_rollout", side_effect=_raise
    ):
        try:
            _sample_with_captured_action(model, {}, num_traj_samples=2)
        except _Boom:
            pass

    # NOTE: bound methods aren't identity-stable across separate attribute
    # accesses in Python (each access builds a fresh wrapper object), so
    # comparing with `is` here would fail even when restoration worked
    # correctly. `==` is the right check -- bound methods compare equal
    # when __self__ and __func__ match, which is exactly "same method,
    # same instance" i.e. genuinely restored.
    assert model.action_space.action_to_traj == original


def test_harvest_scene_chunks_large_k_into_max_batch_size_sub_batches():
    """k=5, max_batch_size=2 -> chunks of [2, 2, 1], each a SEPARATE call to
    sample_trajectories_from_data_with_vlm_rollout (see the seed assertion
    below), reassembled into 5 globally-renumbered RolloutRecords. This is
    the fix for k=100 CUDA-OOMing a single 80GB A100 in one batched call --
    see harvest_scene's docstring."""
    harvester = _make_harvester(num_waypoints=3)

    calls: list[int] = []
    original = harvester.model.sample_trajectories_from_data_with_vlm_rollout

    def _spy(data, **kwargs):
        calls.append(kwargs["num_traj_samples"])
        return original(data, **kwargs)

    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}), \
         mock.patch.object(
             harvester.model, "sample_trajectories_from_data_with_vlm_rollout", side_effect=_spy
         ):
        records = harvester.harvest_scene(
            model_inputs={}, scene_id="scene_chunked", k=5, seed=100, max_batch_size=2,
        )

    assert calls == [2, 2, 1]  # 3 separate calls, not 1 call of 5
    assert len(records) == 5
    assert [r.rollout_id for r in records] == [0, 1, 2, 3, 4]
    # sampling_params.k reflects the FULL requested k (5), not any chunk's
    # batch_k -- chunking is an internal memory-fitting detail, not part of
    # what was actually requested for this scene.
    assert all(r.sampling_params["k"] == 5 for r in records)
    # Each chunk got a distinct seed (100, 101, 102) -- reusing one seed per
    # chunk would reset the RNG and make every chunk's draws identical.
    assert all(r.sampling_params["seed"] in (100, 101, 102) for r in records)
    assert {records[0].sampling_params["seed"], records[2].sampling_params["seed"], records[4].sampling_params["seed"]} == {100, 101, 102}


def test_harvest_scene_max_batch_size_none_matches_original_single_call_behavior():
    """max_batch_size=None (the default) must behave EXACTLY like before
    this chunking feature existed: one call, sampling_params.k == k."""
    harvester = _make_harvester(num_waypoints=3)
    with mock.patch.object(RolloutHarvester, "_build_tokenized_inputs", return_value={}):
        records = harvester.harvest_scene(model_inputs={}, scene_id="scene_unchunked", k=4, seed=1)

    assert len(records) == 4
    assert all(r.sampling_params["seed"] == 1 for r in records)
    assert all(r.sampling_params["k"] == 4 for r in records)


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
