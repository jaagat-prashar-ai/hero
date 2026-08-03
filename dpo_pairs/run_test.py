# SPDX-License-Identifier: Apache-2.0
"""
run_test.py — unit tests for dpo_pairs.run's pure helpers (sharding, resume
accounting, perturbation-corpus indexing) and dpo_pairs.fetch_from_logs'
parsing/dedup. No model, no GPU, no network — the model path is verified by
the real smoke_cluster.yaml run, per the repo's no-fake-model-tests rule.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dpo_pairs.fetch_from_logs import (
    MEASURE_LOG_MARKER,
    dedupe_rows,
    group_by_scene,
    parse_marked_lines,
)
from dpo_pairs.run import (
    _load_done_scenes,
    _results_path,
    _scene_owner,
    load_perturbations_by_scene,
)


class TestSceneOwner:
    def test_stable_and_in_range(self):
        for ws in (1, 2, 8):
            owner = _scene_owner("clip_123_456", ws)
            assert 0 <= owner < ws
            assert owner == _scene_owner("clip_123_456", ws)  # deterministic

    def test_world_size_one_owns_everything(self):
        assert _scene_owner("anything", 1) == 0


class TestResultsPath:
    def test_single_rank_has_no_suffix(self):
        p = _results_path(Path("/out"), rank=0, world_size=1)
        assert p.name == "dpo_measure_rows.jsonl"

    def test_multi_rank_is_rank_scoped(self):
        p = _results_path(Path("/out"), rank=3, world_size=8)
        assert p.name == "dpo_measure_rows_rank03.jsonl"


class TestLoadDoneScenes:
    def test_only_scene_done_rows_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text(
                json.dumps({"kind": "condition", "scene_id": "s1", "condition": "clean"}) + "\n"
                + json.dumps({"kind": "scene_done", "scene_id": "s1"}) + "\n"
                + json.dumps({"kind": "condition", "scene_id": "s2", "condition": "clean"}) + "\n"
                + "not json\n"
            )
            # s2 has condition rows but no scene_done marker: it was
            # interrupted mid-scene and must be re-run, not skipped.
            assert _load_done_scenes(path) == {"s1"}

    def test_missing_file_is_empty(self):
        assert _load_done_scenes(Path("/nonexistent/rows.jsonl")) == set()


class TestLoadPerturbationsByScene:
    def _write(self, rows) -> Path:
        td = tempfile.mkdtemp()
        path = Path(td) / "perturbations.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")
        return path

    def test_groups_by_scene(self):
        ok = {
            "scene_id": "s1", "perturbation_type": "causal_flip",
            "perturbed_trace": "p", "ground_truth_trace": "g",
        }
        path = self._write([ok, {**ok, "perturbation_type": "negation_flip"},
                            {**ok, "scene_id": "s2"}])
        by_scene = load_perturbations_by_scene(path)
        assert set(by_scene) == {"s1", "s2"}
        assert len(by_scene["s1"]) == 2

    def test_malformed_rows_skipped_not_fatal(self):
        ok = {
            "scene_id": "s1", "perturbation_type": "causal_flip",
            "perturbed_trace": "p", "ground_truth_trace": "g",
        }
        missing_field = {"scene_id": "s3", "perturbation_type": "spatial_error"}
        path = self._write([ok, missing_field, "{broken", ""])
        by_scene = load_perturbations_by_scene(path)
        assert set(by_scene) == {"s1"}


class TestFetchFromLogs:
    def test_parse_marked_lines_extracts_payloads(self):
        text = (
            f"2026-08-03 INFO {MEASURE_LOG_MARKER}" + json.dumps({"scene_id": "s1", "kind": "condition", "condition": "clean"}) + "\n"
            "unrelated line\n"
            f"{MEASURE_LOG_MARKER}not-json\n"
            f"prefix {MEASURE_LOG_MARKER}" + json.dumps({"scene_id": "s2", "kind": "condition", "condition": "control_rawids"}) + "\n"
        )
        rows = parse_marked_lines(text)
        assert [r["scene_id"] for r in rows] == ["s1", "s2"]

    def test_dedupe_is_per_scene_condition(self):
        r1 = {"scene_id": "s1", "kind": "condition", "condition": "clean"}
        r2 = {"scene_id": "s1", "kind": "condition", "condition": "perturbed__causal_flip"}
        r3 = {"scene_id": "s1", "kind": "diffusion_crosscheck", "condition": None}
        rows = dedupe_rows([r1, r1, r2, r3, r2])  # dual-log-source doubling
        assert rows == [r1, r2, r3]

    def test_group_by_scene(self):
        rows = [
            {"scene_id": "s1", "kind": "condition", "condition": "clean"},
            {"scene_id": "s2", "kind": "condition", "condition": "clean"},
            {"scene_id": "s1", "kind": "scene_done"},
            {"kind": "junk_without_scene"},
        ]
        grouped = group_by_scene(rows)
        assert set(grouped) == {"s1", "s2"}
        assert len(grouped["s1"]) == 2
