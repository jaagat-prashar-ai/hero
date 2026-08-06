# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the clipgen prototype (no network, per repo convention).

Uses the real testdata obstacle clip plus synthetic trajectories: a
"reactive" trajectory that decelerates mid-clip (stands in for the expert)
and CoC text written against the scene, so gate behavior is exercised on
realistic structures end to end.
"""

import numpy as np
import pandas as pd
import pytest

from code_as_a_reward.clipgen import dossier as dossier_mod
from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen import sandbox
from code_as_a_reward.clipgen.generate import extract_code, render_gt_claims
from code_as_a_reward.coc_claim_parser import parse_coc_trace
from code_as_a_reward.obstacle_tracks import SceneObstacles

TESTDATA = "code_as_a_reward/testdata/obstacle_offline_f0d61901-cfa0-46a4-8992-ab9ea553fc35.parquet"
CLIP_ID = "f0d61901-cfa0-46a4-8992-ab9ea553fc35"
HZ = 10.0

GT_COC = (
    "There is a stopped vehicle ahead in my lane. I will decelerate and "
    "yield because of the stopped vehicle ahead."
)
def _scene() -> SceneObstacles:
    return SceneObstacles.from_dataframe(pd.read_parquet(TESTDATA), CLIP_ID)


def _reactive_waypoints(n: int = 60, hz: float = HZ) -> np.ndarray:
    """Drive straight at 8 m/s, shed ~5 m/s between t=2s and t=4s."""
    dt = 1.0 / hz
    speeds = np.full(n, 8.0)
    for i in range(n):
        t = i * dt
        if 2.0 <= t < 4.0:
            speeds[i] = 8.0 - 2.5 * (t - 2.0)
        elif t >= 4.0:
            speeds[i] = 3.0
    x = np.concatenate([[0.0], np.cumsum(speeds[:-1] * dt)])
    return np.stack([x, np.zeros(n)], axis=1)


GOOD_FN = """\
def reward(claims, traj):
    \"\"\"Decisive event: stopped vehicle ahead; expert sheds ~5 m/s by t=4s.

    Mention-only credit stays small and execution credit is gated on the
    reasoning being present, so every corrupted-rollout perturbation
    (reversed/flat trajectory, gutted claims) drops well below the intact
    pair.
    \"\"\"
    saw = any(c.entity in ("stopped_vehicle", "automobile") for c in claims.perceptual)
    committed = any(c.maneuver in ("yield", "decelerate", "stop") for c in claims.commitments)
    win = window(traj.speed_mps, traj.dt_s, 1.5, 4.5)
    executed = 0.0
    if len(win) > 1 and win[0] > 0:
        drop = float(win[0] - win.min())
        executed = min(drop / (0.6 * 5.0), 1.0)
    score = 0.14 * saw + 0.14 * committed
    if saw and committed:
        score += 0.72 * executed
    return min(score, 1.0)
"""

LENIENT_FN = "def reward(claims, traj):\n    return 0.9\n"


def test_dossier_renders_real_scene():
    traj = dossier_mod.features_from_waypoints(_reactive_waypoints(), HZ, CLIP_ID)
    text = dossier_mod.build_dossier(_scene(), traj, GT_COC)
    assert CLIP_ID in text
    assert "automobile" in text  # testdata clip is all automobiles
    assert "closest" in text and "m/s" in text
    assert GT_COC in text
    # ranked nearest-first and capped
    assert text.count("- track ") <= dossier_mod.MAX_TRACKS


def test_track_summaries_ranked_by_distance():
    summaries = dossier_mod.summarize_tracks(_scene())
    dists = [s.closest_approach_m for s in summaries]
    assert dists == sorted(dists)
    assert all(s.bearing_at_closest in ("ahead", "left", "right", "behind") for s in summaries)


def test_waypoints_from_egomotion_column_variants():
    ts = np.arange(0, 5_000_000, 100_000)  # us
    df = pd.DataFrame({"timestamp": ts, "x": ts / 1e6 * 8.0, "y": np.zeros(len(ts))})
    wp = dossier_mod.waypoints_from_egomotion(df, hz=10.0)
    assert wp.shape[1] == 2
    speeds = np.linalg.norm(np.diff(wp, axis=0), axis=1) * 10.0
    assert np.allclose(speeds, 8.0, atol=0.2)


def test_sandbox_rejects_imports_and_dunders():
    with pytest.raises(sandbox.RewardFnError, match="import"):
        sandbox.compile_reward_fn("import os\ndef reward(claims, traj):\n    return 1.0\n")
    with pytest.raises(sandbox.RewardFnError, match="dunder"):
        sandbox.compile_reward_fn(
            "def reward(claims, traj):\n    return claims.__class__\n"
        )
    with pytest.raises(sandbox.RewardFnError, match="reward"):
        sandbox.compile_reward_fn("def score(claims, traj):\n    return 1.0\n")


def test_sandbox_timeout_and_clamping():
    fn = sandbox.compile_reward_fn(
        "def reward(claims, traj):\n    x = 0\n    while True:\n        x += 1\n"
    )
    with pytest.raises(sandbox.RewardFnError, match="timed out"):
        sandbox.run_reward_fn(fn, None, None, timeout_s=0.2)
    fn = sandbox.compile_reward_fn("def reward(claims, traj):\n    return 7.5\n")
    assert sandbox.run_reward_fn(fn, None, None) == 1.0


def test_window_helper():
    vals = list(range(50))
    win = sandbox.window(vals, dt_s=0.1, t0=2.0, t1=4.0)
    assert win[0] == 20 and win[-1] in (40, 41)


def test_extract_code_takes_last_block():
    text = "draft:\n```python\nx = 1\n```\nfinal:\n```python\ndef reward(claims, traj):\n    return 0.0\n```\n"
    assert extract_code(text).startswith("def reward")


def test_render_gt_claims_shows_canonical_keys():
    trace = parse_coc_trace(GT_COC, scene_id=CLIP_ID)
    block = render_gt_claims(trace)
    assert GT_COC in block
    assert trace.commitments and trace.perceptual  # meaningful fixture parse
    for claim in trace.commitments:
        assert repr(claim.maneuver) in block
    for claim in trace.perceptual:
        assert repr(claim.entity) in block
    assert "Track 32" in block  # the do-not-match-dossier-vocab warning


def _gate_cases():
    gt_claims = parse_coc_trace(GT_COC, scene_id=CLIP_ID)
    return gate_mod.build_perturbations(CLIP_ID, gt_claims, _reactive_waypoints(), HZ)


def test_battery_contains_rollout_perturbations():
    names = {c.name for c in _gate_cases()}
    assert "positive:gt" in names
    assert {"perturb:reversed_traj", "perturb:no_reaction_traj", "perturb:gutted_claims"} <= names


def test_gate_passes_scene_aware_function():
    result = gate_mod.run_gate(GOOD_FN, _gate_cases())
    assert result.pos_score >= gate_mod.POS_MIN, result.scores
    assert result.max_pert <= result.pos_score - gate_mod.MIN_DROP, result.scores
    assert result.passed, result.failures


def test_gate_rejects_lenient_function_with_feedback():
    result = gate_mod.run_gate(LENIENT_FN, _gate_cases())
    assert not result.passed
    assert any("must not be rewarded" in f for f in result.failures)
    assert "scored" in result.feedback()


def test_gate_rejects_over_budget_function():
    # Components sum past 1.0: the clamp would hide corruption drops, so the
    # gate must reject on the raw pre-clamp value even though every clamped
    # score looks plausible.
    over_budget = GOOD_FN.replace(
        "score = 0.14 * saw + 0.14 * committed", "score = 0.6 * saw + 0.6 * committed"
    ).replace("return min(score, 1.0)", "return score")
    result = gate_mod.run_gate(over_budget, _gate_cases())
    assert not result.passed
    assert any("before the [0,1] clamp" in f for f in result.failures)
    assert any("over budget by" in f for f in result.failures)


def test_gate_rejects_raising_function():
    result = gate_mod.run_gate(
        "def reward(claims, traj):\n    return 1.0 / 0.0\n", _gate_cases()
    )
    assert not result.passed
    assert any("raised instead" in f for f in result.failures)
