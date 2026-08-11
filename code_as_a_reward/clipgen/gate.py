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
  * commitments' direction/maneuver flipped to a different-but-valid
    canonical value (e.g. "nudge left" -> "nudge right"), same trajectory --
    covers claim<->trajectory CORRECTNESS, which gutted_claims (presence
    only) does not: a function that only checks a claim is there, never
    that it matches what the trajectory actually did, passes gutted_claims
    but should fail this

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

from code_as_a_reward.clipgen.sandbox import (
    RewardFnError,
    compile_reward_module,
    run_components_fn,
    run_reward_fn,
)
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
    # Per-case component breakdown from the function's own components()
    # decomposition, when it defines one: {case_name: {component: value}}.
    components: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)

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


_DIRECTION_FLIP = {"left": "right", "right": "left"}
# Longitudinal claims are flipped by speed_profile (accelerate<->decelerate
# -- a real contradiction), not by maneuver key: stop/yield/wait/decelerate
# all share speed_profile="decelerate" and are legitimate synonyms for "the
# ego is slowing down," so swapping among THEM is not a corruption a good
# function should be expected to catch (confirmed against this module's own
# GOOD_FN test fixture, whose `committed` check treats them as
# interchangeable on purpose).
_SPEED_PROFILE_FLIP = {"accelerate": "decelerate", "decelerate": "accelerate"}
# Lateral claims without a direction are flipped between "committed lateral
# movement" and "committed to staying in lane" -- the other real
# contradiction available when there's no left/right to flip.
_LATERAL_MOVE_MANEUVERS = {"lane_change", "nudge", "merge", "turn", "enter", "exit"}


def _corrupt_identity(claims):
    """Flip each commitment's direction (or, lacking one, its speed-profile
    or lateral-movement identity) to a different-but-valid canonical value --
    same claim SHAPE, wrong identity. `gutted_claims` already tests claim
    PRESENCE; this tests claim<->trajectory CORRECTNESS, which the
    generation prompt calls "the sharpest reversal discriminator" but no
    prior case actually exercised. Returns None if there is nothing to flip
    (no commitments, or every commitment is a maneuver with no clear
    opposite, e.g. create_gap/proceed)."""
    if not claims.commitments:
        return None
    new_commitments = []
    changed = False
    for c in claims.commitments:
        direction = _DIRECTION_FLIP.get(c.direction) if c.direction else None
        if direction is not None:
            new_commitments.append(dataclasses.replace(c, direction=direction))
            changed = True
            continue
        flipped_profile = _SPEED_PROFILE_FLIP.get(c.speed_profile)
        if flipped_profile is not None:
            new_commitments.append(
                dataclasses.replace(c, maneuver=flipped_profile, speed_profile=flipped_profile)
            )
            changed = True
            continue
        if c.maneuver in _LATERAL_MOVE_MANEUVERS:
            new_commitments.append(dataclasses.replace(c, maneuver="keep_lane"))
            changed = True
            continue
        if c.maneuver == "keep_lane":
            new_commitments.append(dataclasses.replace(c, maneuver="lane_change"))
            changed = True
            continue
        new_commitments.append(c)
    return dataclasses.replace(claims, commitments=new_commitments) if changed else None


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
    corrupted = _corrupt_identity(claims)
    if corrupted is not None:
        cases.append(GateCase("perturb:corrupted_identity", corrupted, traj, "negative"))
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
        f" total turn {traj.total_heading_change_deg:+.0f} deg;"
        f" lateral final {traj.final_lateral_offset_m:+.2f} m,"
        f" max |{float(np.max(np.abs(lat))) if len(lat) else 0.0:.2f}| m"
        " (curvature-contaminated when turn is large);"
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
    fn, components_fn = compile_reward_module(source)
    scores: dict[str, float] = {}
    raws: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for case in cases:
        try:
            raw = run_reward_fn(fn, case.claims, case.traj, raw=True)
            raws[case.name] = raw
            scores[case.name] = min(1.0, max(0.0, raw))
        except RewardFnError as e:
            scores[case.name] = float("nan")
            failures.append(f"{case.name}: raised instead of scoring ({e})")
        if components_fn is not None:
            try:
                components[case.name] = run_components_fn(
                    components_fn, case.claims, case.traj
                )
            except RewardFnError as e:
                failures.append(f"{case.name}: components() failed ({e})")

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
    # Over-budget detection is MECHANICAL, not heuristic (8xvbos: 8/15 clips
    # saturated despite the prompt's sum-to-1.0 rule): any case returning
    # >1.0 before the clamp proves component maxima exceed the budget, which
    # both hides corruption drops behind the clamp and destroys ranking
    # resolution at selection time (multiple rollouts clamp to the same 1.0).
    # Unconditional reject, with the exact overshoot named so the retry can
    # rebudget instead of guessing.
    over = {n: r for n, r in raws.items() if r > 1.0 + 1e-9}
    # The components() decomposition closes the self-clamping hole: a
    # function ending in min(score, 1.0) hides an over-budget sum from the
    # raw probe, but its own component values cannot (8xvbos saturated 8/15
    # clips this way). Also verify the decomposition is truthful -- feedback
    # built on components that do not reconstruct the score misleads retries.
    for name, comp in components.items():
        total = float(sum(comp.values()))
        if total > 1.0 + 1e-9:
            over.setdefault(name, total)
        raw = raws.get(name)
        if raw is not None and abs(min(1.0, max(0.0, total)) - min(1.0, max(0.0, raw))) > 0.02:
            failures.append(
                f"{name}: components() sums to {total:.2f} but reward() returned"
                f" {raw:.2f} -- reward must return exactly the clamped sum of"
                " components so the breakdown is trustworthy"
            )
    if over:
        worst = max(over.items(), key=lambda kv: kv[1])
        failures.append(
            f"{len(over)} case(s) returned MORE than 1.0 before the [0,1] clamp"
            f" (worst: {worst[0]} returned {worst[1]:.2f}, over budget by"
            f" {worst[1] - 1.0:.2f}). Your component maxima sum past 1.0, so the"
            " clamp absorbs exactly the credit a corruption is supposed to lose."
            " Rebudget so all component maxima sum to EXACTLY 1.0."
        )
    passed = not failures
    # Saturation signature (xc7vt9): component weights summing past 1.0 make
    # the [0,1] clamp award claims-carrying corruptions the same 1.0 as the
    # positive -- the trajectory checks are dead code behind the clamp. Name
    # the arithmetic in feedback; score deltas alone read as a logic bug.
    if pos_score == 1.0 and max_pert == 1.0:
        failures.append(
            "positive and a corruption BOTH scored exactly 1.0: your component"
            " maxima likely sum past 1.0, so the [0,1] clamp saturates and your"
            " trajectory checks cannot lower any claims-carrying case."
            " Rebudget so all components sum to exactly 1.0."
        )
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
        if components:
            # The observability fix retries were missing (pduuqq/8xvbos:
            # scores identical across attempts because feedback never said
            # WHICH component carried a corrupted case).
            rows = [
                f"  {c.name}: "
                + ", ".join(f"{k}={v:.2f}" for k, v in components[c.name].items())
                + f" (sum {sum(components[c.name].values()):.2f})"
                for c in cases
                if c.name in cited and c.name in components
            ]
            failures.append(
                "per-component breakdown -- a corrupted case scoring near the"
                " positive keeps its credit in the components whose values did"
                " not drop below the positive's:\n" + "\n".join(rows)
            )
            # Name the exact culprit component(s), don't just show the table
            # and hope the retry notices (2026-08-10 corpus352 run: 58% of
            # clips reaching the positive bar still failed on perturbation
            # margin alone, because a component kept most of its credit
            # under a corruption and nothing ever said so directly).
            pos_case_name = next((c.name for c in cases if c.kind == "positive"), None)
            pos_components = components.get(pos_case_name, {}) if pos_case_name else {}
            culprit_lines = []
            for case in cases:
                if case.kind != "negative" or case.name not in components:
                    continue
                s = scores.get(case.name)
                if not (np.isfinite(s) and np.isfinite(ceiling) and s > ceiling):
                    continue
                case_components = components[case.name]
                culprits = [
                    f"{comp_name} (kept {case_components.get(comp_name, 0.0):.2f} of {pos_val:.2f})"
                    for comp_name, pos_val in pos_components.items()
                    if pos_val > 0.0 and case_components.get(comp_name, 0.0) >= 0.7 * pos_val
                ]
                if culprits:
                    culprit_lines.append(f"  {case.name}: {', '.join(culprits)}")
            if culprit_lines:
                failures.append(
                    "NAMED CULPRIT COMPONENTS -- these specific components kept most or"
                    " all of their credit under a corruption that should have removed"
                    " it. Each one is not actually conditioned on BOTH the claim and the"
                    " trajectory agreeing -- add that condition, or cut its weight:\n"
                    + "\n".join(culprit_lines)
                )
    return GateResult(
        passed=passed,
        pos_score=pos_score,
        max_pert=max_pert,
        scores=scores,
        failures=failures,
        components=components,
    )
