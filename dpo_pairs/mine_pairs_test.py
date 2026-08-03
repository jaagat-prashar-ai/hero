# SPDX-License-Identifier: Apache-2.0
"""
mine_pairs_test.py — unit tests for dpo_pairs.mine_pairs' gating/ranking
math on synthetic trajectories with KNOWN separations
(pref_pairs.synthetic_trajectory_fixtures). No model, no GPU, no network.

Synthetic scene construction: clean = straight line + small gaussian jitter
(a tight sampling distribution), a "big effect" perturbation = lane_change
(meters away from clean), a "null effect" perturbation = the same straight
line with the same jitter magnitude (statistically indistinguishable from
clean). The gate must pass the first and kill the second.
"""

from __future__ import annotations

import numpy as np

from dpo_pairs.mine_pairs import (
    _DEFAULT_THRESHOLDS,
    ade,
    compute_scene_stats,
    evaluate_perturbation,
    index_conditions,
    kinematic_sanity,
    medoid_index,
    mine,
    pairwise_ades,
)
from pref_pairs.synthetic_trajectory_fixtures import lane_change, straight_line

_TC = {
    "cot_start": 1, "cot_end": 2, "traj_future_start": 3, "traj_future_end": 17,
    "traj_token_start_idx": 10, "traj_vocab_size": 4, "tokens_per_future_traj": 8,
}


def _samples(base_xyz: np.ndarray, n: int, jitter_m: float, seed: int) -> list[dict]:
    # Draw-to-draw variation is modeled as a rigid per-sample offset, NOT
    # per-waypoint white noise: white noise double-differentiates into
    # physically absurd accelerations (0.05 m jitter at 10 Hz -> tens of
    # m/s^2) and would trip the kinematic sanity gate on every sample.
    # Real AR-decoded trajectories vary smoothly between draws.
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        xyz = np.asarray(base_xyz, dtype=np.float64).copy()
        xyz[:, :2] += rng.normal(0.0, jitter_m, size=(1, 2))
        out.append({
            "sample_idx": i, "seed": i, "traj_token_ids": [0] * 8,
            "xyz": xyz.tolist(), "n_traj_tokens": 8, "hit_traj_future_end": True,
        })
    return out


def _condition_row(scene_id: str, condition: str, samples: list[dict],
                   perturbation_type: str | None = None) -> dict:
    return {
        "kind": "condition", "scene_id": scene_id, "condition": condition,
        "perturbation_type": perturbation_type,
        "trace_id": f"{scene_id}__{perturbation_type}" if perturbation_type else None,
        "event_cluster": "TEST_CLUSTER", "clip_id": "clip", "t0_us": 0,
        "coc_text": f"text for {condition}", "coc_token_ids": [5, 6],
        "samples": samples, "token_constants": _TC,
        "model_version": "test", "sampling_params": {}, "prompt_len": 4,
        "self_generated_coc": "template text",
    }


def _synthetic_scene(scene_id: str = "s1", n: int = 10, jitter: float = 0.05) -> list[dict]:
    clean_base = straight_line(speed_mps=10.0)
    return [
        _condition_row(scene_id, "control_rawids", _samples(clean_base, n, jitter, seed=0)),
        _condition_row(scene_id, "clean", _samples(clean_base, n, jitter, seed=1)),
        _condition_row(scene_id, "perturbed__spatial_error",
                       _samples(lane_change(amplitude_m=3.0), n, jitter, seed=2),
                       perturbation_type="spatial_error"),
        _condition_row(scene_id, "perturbed__negation_flip",
                       _samples(clean_base, n, jitter, seed=3),
                       perturbation_type="negation_flip"),
    ]


class TestTrajectoryMath:
    def test_ade_zero_for_identical(self):
        a = straight_line()
        assert ade(a, a) == 0.0

    def test_ade_known_offset(self):
        a = straight_line()
        b = a.copy()
        b[:, 1] += 2.0  # constant 2 m lateral offset
        assert abs(ade(a, b) - 2.0) < 1e-9

    def test_pairwise_count(self):
        xyzs = [straight_line() for _ in range(5)]
        assert len(pairwise_ades(xyzs)) == 10  # 5 choose 2

    def test_medoid_prefers_center(self):
        base = straight_line()
        outlier = base.copy()
        outlier[:, 1] += 10.0
        xyzs = [base, base.copy(), outlier]
        assert medoid_index(xyzs) in (0, 1)

    def test_kinematic_sanity(self):
        assert kinematic_sanity(straight_line(speed_mps=10.0), 10.0, 45.0, 12.0)
        teleport = straight_line()
        teleport[20, :2] += 50.0  # 50 m jump in one 0.1 s step
        assert not kinematic_sanity(teleport, 10.0, 45.0, 12.0)


class TestGating:
    def test_big_effect_qualifies_null_effect_dies(self):
        rows = _synthetic_scene()
        conditions = index_conditions(rows)
        stats = compute_scene_stats(conditions, _DEFAULT_THRESHOLDS)
        assert stats is not None

        big = evaluate_perturbation(
            conditions["perturbed__spatial_error"], stats,
            cluster_eps=stats["scene_eps"], thresholds=_DEFAULT_THRESHOLDS,
        )
        null = evaluate_perturbation(
            conditions["perturbed__negation_flip"], stats,
            cluster_eps=stats["scene_eps"], thresholds=_DEFAULT_THRESHOLDS,
        )
        assert big["qualifies"], big
        assert big["z_score"] > 3.0
        assert not null["qualifies"], null
        assert not null["gate_noise_floor"]

    def test_malformed_rejected_side_dies_on_sanity_gate(self):
        rows = _synthetic_scene()
        conditions = index_conditions(rows)
        pert = conditions["perturbed__spatial_error"]
        for s in pert["samples"]:
            s["hit_traj_future_end"] = False  # model never emitted the end marker
        stats = compute_scene_stats(conditions, _DEFAULT_THRESHOLDS)
        metrics = evaluate_perturbation(
            pert, stats, cluster_eps=stats["scene_eps"], thresholds=_DEFAULT_THRESHOLDS,
        )
        assert not metrics["qualifies"]
        assert not metrics["gate_kinematic_sanity"]
        assert metrics["gate_noise_floor"]  # effect is real, output is garbage

    def test_control_gate_kills_machinery_dominated_effect(self):
        rows = _synthetic_scene()
        conditions = index_conditions(rows)
        # Poison the control: pretend re-tokenization alone moves trajectories
        # by ~the same amount as the perturbation.
        shifted = lane_change(amplitude_m=3.5)
        conditions["control_rawids"]["samples"] = _samples(shifted, 10, 0.05, seed=9)
        stats = compute_scene_stats(conditions, _DEFAULT_THRESHOLDS)
        metrics = evaluate_perturbation(
            conditions["perturbed__spatial_error"], stats,
            cluster_eps=stats["scene_eps"], thresholds=_DEFAULT_THRESHOLDS,
        )
        assert not metrics["gate_control"]
        assert not metrics["qualifies"]


class TestMine:
    def test_end_to_end_pair_emission(self):
        by_scene = {"s1": _synthetic_scene("s1"), "s2": _synthetic_scene("s2")}
        pairs, report = mine(by_scene, semantic_delta=False)

        assert report["n_scenes_mined"] == 2
        assert report["n_perturbations_evaluated"] == 4
        assert len(pairs) == 2  # one qualifying perturbation per scene
        assert report["gate_deaths"]["noise_floor"] == 2  # the two null effects
        assert set(report["cluster_eps_m"]) == {"TEST_CLUSTER"}

        p = pairs[0]
        assert p["metrics"]["rank_within_scene"] == 1
        assert p["rejected"]["perturbation_type"] == "spatial_error"
        # Completion assembly invariant: [cot_start] + coc + [cot_end,
        # traj_future_start] + raw traj + [traj_future_end].
        comp = p["chosen"]["completion_token_ids"]
        assert comp[0] == _TC["cot_start"] and comp[-1] == _TC["traj_future_end"]
        assert comp[1:3] == [5, 6]  # coc ids
        assert comp[3:5] == [_TC["cot_end"], _TC["traj_future_start"]]
        assert all(t >= _TC["traj_token_start_idx"] for t in comp[5:-1])
        # ranked-points table includes NON-qualifying entries (the deliverable
        # covers every measured perturbation point, not just DPO pairs).
        assert len(report["maximal_perturbation_points"]["s1"]) == 2

    def test_incomplete_scene_skipped_with_accounting(self):
        rows = _synthetic_scene("s1")
        rows = [r for r in rows if r["condition"] != "clean"]
        pairs, report = mine({"s1": rows}, semantic_delta=False)
        assert pairs == []
        assert report["n_scenes_incomplete"] == 1
