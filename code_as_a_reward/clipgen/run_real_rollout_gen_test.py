# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from code_as_a_reward.clipgen.run_real_rollout_gen import merge_manifest_targets
from code_as_a_reward.clipgen.rollout_worker import _clip_seed, _valid_existing


def test_merge_manifest_targets_carries_authoritative_t0():
    merged = merge_manifest_targets(
        [{"clip_id": "a", "hz": 10.0}],
        [{"clip_id": "a", "t0_us": 123}],
    )
    assert merged == [{"clip_id": "a", "hz": 10.0, "t0_us": 123}]


def test_merge_manifest_targets_rejects_mismatch_and_duplicates():
    with pytest.raises(ValueError, match="sets differ"):
        merge_manifest_targets(
            [{"clip_id": "a"}],
            [{"clip_id": "b", "t0_us": 123}],
        )
    with pytest.raises(ValueError, match="duplicate"):
        merge_manifest_targets(
            [{"clip_id": "a"}, {"clip_id": "a"}],
            [{"clip_id": "a", "t0_us": 123}],
        )
    with pytest.raises(ValueError, match="manifest t0_us"):
        merge_manifest_targets(
            [{"clip_id": "a", "t0_us": 100}],
            [{"clip_id": "a", "t0_us": 123}],
        )


def test_resume_requires_exact_rollout_provenance(tmp_path):
    base_seed = 42
    generation_seed = _clip_seed(base_seed, "a")
    doc = {
        "schema_version": "clipgen.rollouts.v2",
        "clip_id": "a",
        "t0_us": 123,
        "gt_waypoints": [[0.0, 0.0], [1.0, 0.0]],
        "groups": {
            # IDs are scoped to their group; the sampler emits 0..N-1 for
            # both independent forward passes.
            "generation": [{"rollout_id": 0}],
            "holdout": [{"rollout_id": 0}],
        },
        "provenance": {
            "model": "checkpoint",
            "model_revision": "revision",
            "top_p": 0.98,
            "temperature": 1.0,
            "max_generation_length": 512,
            "base_seed": base_seed,
            "generation_seed": generation_seed,
            "holdout_seed": generation_seed + 1,
        },
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(doc))
    kwargs = {
        "clip_id": "a",
        "t0_us": 123,
        "generation_group_size": 1,
        "holdout_group_size": 1,
        "base_seed": base_seed,
        "model_checkpoint": "checkpoint",
        "model_revision": "revision",
        "top_p": 0.98,
        "temperature": 1.0,
        "max_generation_length": 512,
    }
    assert _valid_existing(str(path), **kwargs)
    doc["provenance"]["holdout_seed"] = generation_seed
    path.write_text(json.dumps(doc))
    assert not _valid_existing(str(path), **kwargs)
