# SPDX-License-Identifier: Apache-2.0
"""Tests for code_reward_entry's pure reward-shaping helpers.

Per the project's no-fake-model-tests preference, compute_reward_batch
(which needs the recipe venv's alpamayo1_x_rl + tokenizers) is NOT tested
here -- real verification is the canary cluster run. These tests pin down
the soft-gate blend's contract: far from the gates it reproduces the two
branch formulas exactly (same endpoints as the hard gate it replaced), and
across the gates it is continuous and monotone, which is the whole point of
the 2026-08-02 change (run n3sxdq's population median sat 0.02 from the
hard gate, so GRPO advantages encoded threshold luck).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rl_posttrain.rewards.code_reward_entry as cre  # noqa: E402
from rl_posttrain.rewards.code_reward_entry import (  # noqa: E402
    _graded_failure_reward,
    _soft_gate_blend,
)

ADE_T = 3.0
R_T = -0.4


def _blended(s: float, l2: float, pass_reward: float = -0.1) -> float:
    fail = _graded_failure_reward(
        l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
    )
    return _soft_gate_blend(
        pass_reward, fail, l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T
    )


class TestSoftGateBlend:
    def test_deep_pass_region_matches_pass_formula(self):
        # 5+ tau above the reasoning gate, l2 well under threshold.
        assert _blended(-0.05, 1.0, pass_reward=0.12) == pytest.approx(0.12, abs=1e-3)

    def test_deep_fail_region_matches_graded_failure(self):
        fail = _graded_failure_reward(
            2.0, -0.9, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
        )
        assert _blended(-0.9, 2.0, pass_reward=0.12) == pytest.approx(fail, abs=1e-3)

    def test_continuous_at_reasoning_gate(self):
        # The hard gate jumped ~0.37 across s = -0.4 at l2 = 2.0; the blend
        # must move gradually: no step bigger than the local slope allows.
        eps = 1e-4
        assert abs(_blended(R_T + eps, 2.0) - _blended(R_T - eps, 2.0)) < 0.01

    def test_continuous_at_ade_gate(self):
        eps = 1e-4
        assert abs(_blended(-0.2, ADE_T - eps) - _blended(-0.2, ADE_T + eps)) < 0.01

    def test_monotone_in_reasoning_score_near_gate(self):
        # Better reasoning must never score worse anywhere near the seam --
        # the property the hard gate violated for GRPO's within-group ranks.
        scores = [R_T + i * 0.01 for i in range(-20, 21)]
        rewards = [_blended(s, 2.0) for s in scores]
        assert all(b >= a - 1e-9 for a, b in zip(rewards, rewards[1:]))

    def test_gate_midpoint_is_halfway(self):
        # At exactly (s = R_T, l2 << ADE_T) the reasoning sigmoid is 0.5.
        fail = _graded_failure_reward(
            1.0, R_T, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
        )
        expected = 0.5 * (-0.1) + 0.5 * fail
        # l2 = 1.0 is >13 tau inside the ADE gate; its sigmoid is ~1.
        assert _blended(R_T, 1.0) == pytest.approx(expected, abs=1e-3)


class TestGradedFailureUnchanged:
    def test_no_cot_stays_flat_minus_one(self):
        assert (
            _graded_failure_reward(
                2.0, -0.2, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=False
            )
            == -1.0
        )

    def test_band_bounds(self):
        for s in (-1.0, -0.7, -0.41):
            for l2 in (1.0, 3.5, 10.0):
                r = _graded_failure_reward(
                    l2, s, ade_threshold=ADE_T, reasoning_threshold=R_T, cot_decoded=True
                )
                assert -1.0 <= r <= -0.5


class TestNeutralPrior:
    @pytest.fixture(autouse=True)
    def _reset_ema(self, monkeypatch):
        monkeypatch.setattr(cre, "_prior_ema", cre._PRIOR_INIT)

    def test_init_value(self):
        assert cre._neutral_prior() == pytest.approx(cre._PRIOR_INIT)

    def test_ema_tracks_observations(self):
        for _ in range(600):
            cre._observe_precision(0.7)
        assert cre._neutral_prior() == pytest.approx(0.7, abs=1e-3)

    def test_clamped_floor_and_ceiling(self):
        for _ in range(600):
            cre._observe_precision(0.0)
        assert cre._neutral_prior() == cre._PRIOR_MIN
        for _ in range(2000):
            cre._observe_precision(1.0)
        assert cre._neutral_prior() == cre._PRIOR_MAX

    def test_coverage_neutral_at_prior(self):
        # The property the fixed 0.8 violated: when a rollout's precision
        # equals the prior, its blended score must not depend on coverage.
        prior = cre._neutral_prior()
        r_effs = [df * prior + (1.0 - df) * prior for df in (0.0, 0.3, 0.7, 1.0)]
        assert max(r_effs) - min(r_effs) < 1e-12


class TestLrSchedulerPatch:
    @pytest.fixture()
    def fake_cosmos(self, monkeypatch):
        """Stub the two cosmos_rl modules the patch imports lazily."""
        import types

        built = []

        class FakeScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

            def get_last_lr(self):
                return [2e-6]

        def fake_build(optimizers, config, training_steps):
            sched = FakeScheduler()
            built.append((training_steps, sched))
            return sched

        optm = types.ModuleType("cosmos_rl.policy.trainer.optm")
        optm.build_lr_schedulers = fake_build
        logging_mod = types.ModuleType("cosmos_rl.utils.logging")
        logging_mod.logger = __import__("logging").getLogger("fake_cosmos")
        for name, mod in {
            "cosmos_rl": types.ModuleType("cosmos_rl"),
            "cosmos_rl.policy": types.ModuleType("cosmos_rl.policy"),
            "cosmos_rl.policy.trainer": types.ModuleType("cosmos_rl.policy.trainer"),
            "cosmos_rl.policy.trainer.optm": optm,
            "cosmos_rl.utils": types.ModuleType("cosmos_rl.utils"),
            "cosmos_rl.utils.logging": logging_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
        return built

    def _fake_trainer_cls(self):
        calls = []

        class FakeTrainer:
            optimizers = object()
            config = object()

            def step_training(self, *args, **kwargs):
                calls.append((args, kwargs))
                return {"ok": True}

        return FakeTrainer, calls

    def test_rebuilds_once_with_real_total(self, fake_cosmos):
        cls, calls = self._fake_trainer_cls()
        cre._patch_lr_scheduler_total_steps(cls)
        t = cls()
        # (rollouts, current_step, total_steps, ...) positional, like cosmos
        assert t.step_training(["r"], 1, 264, 0, None, True) == {"ok": True}
        t.step_training(["r"], 2, 264, 0, None, True)
        assert len(fake_cosmos) == 1  # guard: built exactly once
        assert fake_cosmos[0][0] == 264  # with the REAL total, not 1e6
        assert len(calls) == 2  # original always called

    def test_fast_forwards_on_mid_run_first_call(self, fake_cosmos):
        cls, _ = self._fake_trainer_cls()
        cre._patch_lr_scheduler_total_steps(cls)
        t = cls()
        t.step_training(["r"], 5, 264, 0, None, True)
        assert fake_cosmos[0][1].steps == 4  # current_step-1 catch-up steps

    def test_kwargs_call_style_supported(self, fake_cosmos):
        cls, _ = self._fake_trainer_cls()
        cre._patch_lr_scheduler_total_steps(cls)
        t = cls()
        t.step_training(["r"], current_step=1, total_steps=100)
        assert fake_cosmos[0][0] == 100

    def test_missing_total_steps_leaves_scheduler_alone(self, fake_cosmos):
        cls, calls = self._fake_trainer_cls()
        cre._patch_lr_scheduler_total_steps(cls)
        t = cls()
        t.step_training(["r"])  # no step info at all
        assert fake_cosmos == []  # no rebuild attempted
        assert len(calls) == 1  # original still ran


class TestObstacleManifest:
    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        cre._obstacle_manifest.cache_clear()
        cre._load_scene.cache_clear()
        yield
        cre._obstacle_manifest.cache_clear()
        cre._load_scene.cache_clear()

    def test_no_local_dir_means_no_manifest(self, monkeypatch):
        monkeypatch.delenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR", raising=False)
        assert cre._obstacle_manifest() is None

    def test_manifest_parsed(self, tmp_path, monkeypatch):
        d = tmp_path / "obstacles_by_clip"
        d.mkdir()
        (d / "_MANIFEST.txt").write_text("clip_a\nclip_b\n")
        monkeypatch.setenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR", str(tmp_path))
        assert cre._obstacle_manifest() == frozenset({"clip_a", "clip_b"})

    def test_load_scene_skips_known_absent_clip(self, tmp_path, monkeypatch):
        # A clip the manifest doesn't list must return None WITHOUT reaching
        # the avdi chunk-zip fallback (which would ImportError here -- the
        # recipe venv isn't installed in the test env, so reaching it fails
        # loudly rather than passing vacuously).
        d = tmp_path / "obstacles_by_clip"
        d.mkdir()
        (d / "_MANIFEST.txt").write_text("clip_a\n")
        monkeypatch.setenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR", str(tmp_path))
        assert cre._load_scene("clip_absent_upstream") is None
