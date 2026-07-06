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
