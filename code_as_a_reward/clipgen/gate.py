# SPDX-License-Identifier: Apache-2.0
"""The verification gate: empirical accept/reject for generated functions.

A candidate is accepted only if it separates trajectories it should reward
from trajectories it shouldn't (VLM-CaR's expert-vs-random check, adapted):

- POSITIVE: the clip's GT reasoning (parsed to claims) + GT trajectory
  must score >= POS_MIN.
- NEGATIVES (each must score low; the p95 across all negatives <= NEG_P95_MAX):
  * GT claims paired with other clips' trajectories
  * GT claims paired with the GT trajectory time-reversed
  * GT claims paired with a flattened (no-reaction, constant-speed) trajectory
  * other clips' claims paired with the GT trajectory

Negatives transform WAYPOINTS and re-extract features, so derived fields
(events, min speed) stay consistent rather than being hand-edited.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from code_as_a_reward.clipgen.sandbox import RewardFnError, compile_reward_fn, run_reward_fn
from pref_pairs.trajectory_features import TrajectoryFeatures, extract_features

POS_MIN = 0.7
NEG_P95_MAX = 0.3


@dataclasses.dataclass
class GateCase:
    name: str
    claims: object  # ParsedCoCTrace
    traj: TrajectoryFeatures
    kind: str  # "positive" | "negative"


@dataclasses.dataclass
class GateResult:
    passed: bool
    pos_score: float
    neg_p95: float
    scores: dict[str, float]
    failures: list[str]  # feedback lines for the regeneration prompt

    def feedback(self) -> str:
        return "\n".join(self.failures) if self.failures else "all checks passed"


def _refeature(waypoints: np.ndarray, hz: float, scene_id: str, tag: str) -> TrajectoryFeatures:
    return extract_features(waypoints, hz=hz, scene_id=f"{scene_id}:{tag}", rollout_id=0)


def flattened_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """A no-reaction counterfactual: continue at the initial velocity."""
    w = np.asarray(waypoints, dtype=np.float64)
    v = w[1] - w[0] if len(w) > 1 else np.zeros(2)
    steps = np.arange(len(w), dtype=np.float64)[:, None]
    return w[0] + steps * v


def _too_similar(a: np.ndarray, b: np.ndarray, min_dev_m: float = 2.0) -> bool:
    """A transformed trajectory that barely deviates from the GT is not a
    valid counterfactual (stationary clips: reversing or flattening a
    parked ego reproduces the positive) -- such negatives are skipped."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return True
    return float(np.max(np.linalg.norm(a[:n] - b[:n], axis=1))) < min_dev_m


def build_cases(
    clip_id: str,
    gt_claims,
    gt_waypoints: np.ndarray,
    hz: float,
    others: list[tuple[object, np.ndarray]],
) -> list[GateCase]:
    """Assemble the positive and the negative battery for one clip.

    `others` carries (claims, waypoints) from the OTHER prototype clips --
    both directions of mismatch are exercised.
    """
    gt_traj = _refeature(gt_waypoints, hz, clip_id, "gt")
    cases = [GateCase("positive:gt", gt_claims, gt_traj, "positive")]
    reversed_wp = np.asarray(gt_waypoints)[::-1].copy()
    if not _too_similar(gt_waypoints, reversed_wp):
        cases.append(
            GateCase(
                "negative:reversed_traj",
                gt_claims,
                _refeature(reversed_wp, hz, clip_id, "rev"),
                "negative",
            )
        )
    flat_wp = flattened_waypoints(gt_waypoints)
    if not _too_similar(gt_waypoints, flat_wp):
        cases.append(
            GateCase(
                "negative:no_reaction_traj",
                gt_claims,
                _refeature(flat_wp, hz, clip_id, "flat"),
                "negative",
            )
        )
    for i, (other_claims, other_wp) in enumerate(others):
        cases.append(
            GateCase(
                f"negative:other_traj_{i}",
                gt_claims,
                _refeature(other_wp, hz, clip_id, f"other{i}"),
                "negative",
            )
        )
        cases.append(GateCase(f"negative:other_claims_{i}", other_claims, gt_traj, "negative"))
    return cases


def run_gate(
    source: str,
    cases: list[GateCase],
    pos_min: float = POS_MIN,
    neg_p95_max: float = NEG_P95_MAX,
) -> GateResult:
    """Score every case; an exception on any case is an automatic failure."""
    fn = compile_reward_fn(source)
    scores: dict[str, float] = {}
    failures: list[str] = []
    for case in cases:
        try:
            scores[case.name] = run_reward_fn(fn, case.claims, case.traj)
        except RewardFnError as e:
            scores[case.name] = float("nan")
            failures.append(f"{case.name}: raised instead of scoring ({e})")

    pos_scores = [
        s for c, s in zip(cases, scores.values()) if c.kind == "positive" and np.isfinite(s)
    ]
    neg_scores = [
        s for c, s in zip(cases, scores.values()) if c.kind == "negative" and np.isfinite(s)
    ]
    pos_score = float(min(pos_scores)) if pos_scores else float("nan")
    neg_p95 = float(np.percentile(neg_scores, 95)) if neg_scores else float("nan")

    if not np.isfinite(pos_score) or pos_score < pos_min:
        failures.append(
            f"positive case scored {pos_score:.2f}, needs >= {pos_min} -- the ground-truth"
            " reasoning+trajectory pair must be rewarded"
        )
    for case in cases:
        s = scores[case.name]
        if case.kind == "negative" and np.isfinite(s) and s > neg_p95_max:
            failures.append(
                f"{case.name} scored {s:.2f}, wanted <= {neg_p95_max} -- this pairing is"
                " wrong-by-construction and must not be rewarded"
            )
    passed = bool(
        np.isfinite(pos_score)
        and pos_score >= pos_min
        and np.isfinite(neg_p95)
        and neg_p95 <= neg_p95_max
        and not any("raised instead" in f for f in failures)
    )
    return GateResult(passed=passed, pos_score=pos_score, neg_p95=neg_p95, scores=scores, failures=failures)
