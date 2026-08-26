# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

from code_as_a_reward.clipgen.repair_queue import (
    _REPAIR_AUDIT_COLUMNS,
    development_feedback,
    load_records,
    make_batches,
    repair_audit_rows,
)


def _row(scene_id: str) -> dict:
    return {
        "schema_version": "clipgen.repair.v1",
        "clip_id": "clip-a",
        "scene_id": scene_id,
        "reward_source_sha256": "abc123",
        "failures": [f"failure-{scene_id}"],
        "top_gates": {},
    }


def test_repair_requires_repeated_groups_and_keeps_holdout_hidden():
    assert make_batches([_row("s1"), _row("s2")]) == []
    batches = make_batches([_row("s1"), _row("s2"), _row("s3")])
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.development) == 2
    assert len(batch.holdout) == 1
    feedback = development_feedback(batch)
    assert all(row["scene_id"] in feedback for row in batch.development)
    assert batch.holdout[0]["scene_id"] not in feedback


def test_repair_deduplicates_repeated_scene_records():
    rows = [_row("s1"), _row("s1"), _row("s2"), _row("s3")]
    batch = make_batches(rows)[0]
    assert len(batch.development) + len(batch.holdout) == 3


def test_load_records_accepts_production_record_directory(tmp_path):
    for index, scene in enumerate(("s1", "s2")):
        (tmp_path / f"{index}.json").write_text(json.dumps(_row(scene)))
    assert [row["scene_id"] for row in load_records(tmp_path)] == ["s1", "s2"]


def test_repair_audit_rows_expose_source_feedback_candidate_and_diff():
    rows = repair_audit_rows(
        [
            {
                "clip_id": "clip-a",
                "parent_sha256": "parent",
                "status": "accepted_proposal",
                "initial_reward_source": "def reward(): return 0.4",
                "attempts": [
                    {
                        "attempt": 1,
                        "candidate_sha256": "candidate",
                        "feedback_sent": "execution component ignored trajectory",
                        "candidate_source": "def reward(): return 0.8",
                        "source_diff": "-0.4\n+0.8",
                        "gt_gate": {"pos_score": 0.8, "max_pert": 0.2, "delta": 0.6},
                        "development_results": [{"passed": True}],
                        "failures": [],
                    }
                ],
            }
        ]
    )
    by_name = dict(zip(_REPAIR_AUDIT_COLUMNS, rows[0]))
    assert "return 0.4" in by_name["initial_reward_source"]
    assert "ignored trajectory" in by_name["feedback_sent_to_llm"]
    assert "return 0.8" in by_name["candidate_reward_source"]
    assert by_name["gt_delta"] == 0.6
