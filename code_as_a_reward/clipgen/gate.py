# SPDX-License-Identifier: Apache-2.0
"""The verification gate: empirical accept/reject for generated functions.

A candidate must separate a rollout from corrupted variants of that SAME
rollout (VLM-CaR's expert-vs-random check, moved onto the rollout itself):

- POSITIVE: the rollout's claims + trajectory must score >= POS_MIN.
- PERTURBATIONS (each must score at least MIN_DROP below the positive):
  * same claims, trajectory time-reversed
  * same claims, flattened (no-reaction, constant-velocity) trajectory
  * claims gutted (no perceptual/commitment/causal content), same trajectory
  * commitments removed (perception-only reasoning), same trajectory

At generation time the GT pair stands in for the rollout. At selection time
(score a rollout group, take the argmax), the same battery verifies the
winner: a function insensitive to corruption of the very rollout it ranked
highest is rejected. The old cross-clip negatives (other clips' claims or
trajectories) are gone -- with semantically similar clips in the pool they
produced unwinnable cases (identical canonical claims), and rollout-derived
perturbations are guaranteed contrastive by construction.

Perturbations transform WAYPOINTS and re-extract features, so derived
fields (events, min speed) stay consistent rather than being hand-edited.
"""

# implement the counterfactuals here. 
# deeper look at counterfactual VLA. 
# When the trajectory doesn't pass, the ADE (l2_dist < ade_threshold=3.0) or the reasoning gate (reasoning_score > reasoning_threshold=-0.4) fails, 
# rl_posttrain/rewards/aggregated_reward_llm_judge.py:108 (_graded_failure_reward) computes the reward instead of the continuous mixing formula: 

# No CoC decoded at all -> flat -1.0 (matches the vendored behaviir; there's nothing to grade)
# CoC present but a gate failed -> reward is graded within [-1.0, -0.5] based on how close the rollout came to passing:
    # l2_closeness = min(1.0, ade_threshold / l2_dist) - 1.0 right at the threshold, decaying hyperbolically as the trajectory error grows. 
    # reasoning_closeness - maps the judge's reasoning score linearly from [-1, reasoning_threshold] onto [0, 1]
    # Final: -1.0 + 0.5 * 0.5 * (l2_closeness + reasoning_closeness)

# The point (per the module's docstring) is to avoid the vendored variant's flat -1.0 for every gate failure: since GRPO normalizes advantages within a rollout group, an 
# all-fail group with identical -1.0 rewards has zero variance and contributes no gradient. This graded band keeps failed rollouts ordered by "how close to passing" while staying strictly below every passing rollout's reward (whose floor is ~'-0.2' with current weights), so failing groups still push 
# the policy toward the gate boundary instead of training nothing. 

 
from __future__ import annotations

import dataclasses

import numpy as np

from code_as_a_reward.clipgen.sandbox import RewardFnError, compile_reward_fn, run_reward_fn
from pref_pairs.trajectory_features import TrajectoryFeatures, extract_features

POS_MIN = 0.7
# A corrupted rollout must score at least this far below the intact one.
# At the minimum passing positive (0.7) this reproduces the old absolute
# ceiling (perturbations <= 0.3); for stronger positives it scales.
MIN_DROP = 0.4


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
    max_pert: float  # highest score any perturbation achieved
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


def build_perturbations(
    scene_id: str,
    claims,
    waypoints: np.ndarray,
    hz: float,
    tag: str = "gt",
) -> list[GateCase]:
    """The positive plus corrupted variants of the SAME rollout.

    At generation time the GT (claims, waypoints) pair stands in for the
    rollout (tag="gt"); at selection time call this on the argmax rollout
    of a group to verify the function is sensitive around its own winner.
    """
    traj = _refeature(waypoints, hz, scene_id, tag)
    cases = [GateCase(f"positive:{tag}", claims, traj, "positive")]
    reversed_wp = np.asarray(waypoints)[::-1].copy()
    if not _too_similar(waypoints, reversed_wp):
        cases.append(
            GateCase(
                "perturb:reversed_traj",
                claims,
                _refeature(reversed_wp, hz, scene_id, "rev"),
                "negative",
            )
        )
    flat_wp = flattened_waypoints(waypoints)
    if not _too_similar(waypoints, flat_wp):
        cases.append(
            GateCase(
                "perturb:no_reaction_traj",
                claims,
                _refeature(flat_wp, hz, scene_id, "flat"),
                "negative",
            )
        )
    gutted = dataclasses.replace(claims, perceptual=[], commitments=[], causal=[])
    cases.append(GateCase("perturb:gutted_claims", gutted, traj, "negative"))
    if claims.commitments:
        no_commit = dataclasses.replace(claims, commitments=[], causal=[])
        cases.append(GateCase("perturb:no_commitments", no_commit, traj, "negative"))
    return cases


def _traj_facts(traj) -> str:
    """Measured facts a generated function's checks run against -- shown in
    feedback so the generator can see WHY a check fails (e.g. an exact
    monotonicity test failing on the noisy GT itself)."""
    speed = np.asarray(traj.speed_mps, dtype=np.float64)
    lat = np.asarray(traj.lateral_offset_m, dtype=np.float64)
    t_min = float(np.argmin(speed)) * traj.dt_s if len(speed) else 0.0
    rising = int(np.sum(np.diff(speed) > 0)) if len(speed) > 1 else 0
    return (
        f"speed {traj.initial_speed_mps:.1f}->{traj.final_speed_mps:.1f} m/s"
        f" (min {traj.min_speed_mps:.1f} at t={t_min:.1f}s,"
        f" drop {traj.initial_speed_mps - traj.min_speed_mps:.1f});"
        f" {rising}/{max(len(speed) - 1, 1)} speed steps INCREASE (noise);"
        f" lateral final {traj.final_lateral_offset_m:+.2f} m,"
        f" max |{float(np.max(np.abs(lat))) if len(lat) else 0.0:.2f}| m;"
        f" stop_event={traj.stop_event}"
    )


def run_gate(
    source: str,
    cases: list[GateCase],
    pos_min: float = POS_MIN,
    min_drop: float = MIN_DROP,
) -> GateResult:
    """Score every case; the positive must clear pos_min and every
    perturbation must score at least min_drop below it. An exception on
    any case is an automatic failure."""
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
    pert_scores = [
        s for c, s in zip(cases, scores.values()) if c.kind == "negative" and np.isfinite(s)
    ]
    pos_score = float(min(pos_scores)) if pos_scores else float("nan")
    max_pert = float(max(pert_scores)) if pert_scores else float("nan")

    if not np.isfinite(pos_score) or pos_score < pos_min:
        failures.append(
            f"positive case scored {pos_score:.2f}, needs >= {pos_min} -- the intact"
            " reasoning+trajectory pair must be rewarded"
        )
    ceiling = pos_score - min_drop
    if np.isfinite(ceiling):
        for case in cases:
            s = scores[case.name]
            if case.kind == "negative" and np.isfinite(s) and s > ceiling:
                failures.append(
                    f"{case.name} scored {s:.2f}, needs <= {ceiling:.2f} (positive"
                    f" {pos_score:.2f} minus drop {min_drop}) -- a corrupted variant of"
                    " the rollout must not be rewarded"
                )
    passed = not failures
    if failures:
        # Show the measured numbers behind the positive and every case named
        # in a failure line -- without them the generator cannot see why a
        # check misfires (g9349h: monotonicity tests failing on the noisy GT
        # left retries blind and near-identical).
        cited = {c.name for c in cases if c.kind == "positive"} | {
            f.split(" ", 1)[0].rstrip(":") for f in failures
        }
        facts = [f"  {c.name}: {_traj_facts(c.traj)}" for c in cases if c.name in cited]
        failures.append("measured trajectory facts per case:\n" + "\n".join(facts))
    return GateResult(passed=passed, pos_score=pos_score, max_pert=max_pert, scores=scores, failures=failures)
