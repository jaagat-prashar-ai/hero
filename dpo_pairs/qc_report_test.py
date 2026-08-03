# SPDX-License-Identifier: Apache-2.0
"""qc_report_test.py — unit test for dpo_pairs.qc_report's pure data-prep
helper. Plot rendering is verified by running qc_report on the real pilot
output (eyeball step), not asserted on pixels here."""

from __future__ import annotations

from dpo_pairs.qc_report import collect_scatter_data


def _pair(semantic, cross_ade, ptype="causal_flip", cluster="C"):
    return {
        "event_cluster": cluster,
        "rejected": {"perturbation_type": ptype},
        "metrics": {"semantic_delta_cos": semantic, "cross_ade_m": cross_ade},
    }


class TestCollectScatterData:
    def test_collects_and_labels(self):
        data = collect_scatter_data([
            _pair(0.1, 1.5), _pair(0.3, 4.0, ptype="spatial_error", cluster="D"),
        ])
        assert data["semantic"] == [0.1, 0.3]
        assert data["cross_ade"] == [1.5, 4.0]
        assert data["ptype"] == ["causal_flip", "spatial_error"]
        assert data["cluster"] == ["C", "D"]

    def test_null_semantic_rows_dropped(self):
        data = collect_scatter_data([_pair(None, 2.0), _pair(0.2, 3.0)])
        assert len(data["semantic"]) == 1

    def test_missing_type_becomes_question_mark(self):
        p = _pair(0.2, 3.0)
        p["rejected"]["perturbation_type"] = None
        assert collect_scatter_data([p])["ptype"] == ["?"]
