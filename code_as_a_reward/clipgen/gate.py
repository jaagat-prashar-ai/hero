# SPDX-License-Identifier: Apache-2.0
"""The verification gate: empirical accept/reject for generated functions.

A candidate must separate a rollout from corrupted variants of that SAME
rollout (VLM-CaR's expert-vs-random check, moved onto the rollout itself):

- POSITIVE: the rollout's claims + trajectory must score >= POS_MIN.
- PERTURBATIONS (each must score at least MIN_DROP below the positive):
  * same claims, action-specific contradictory/no-reaction trajectory
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
    min_drop: float = MIN_DROP


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
# Lateral claims without a direction are flipped between "committed lateral
# movement" and "committed to staying in lane" -- the other real
# contradiction available when there's no left/right to flip.
_LATERAL_MOVE_MANEUVERS = {
    "lane_change", "nudge", "merge", "turn", "enter", "exit", "overtake"
}


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
    profiles = {c.speed_profile for c in claims.commitments}
    # A trace often expresses one longitudinal decision with two parser
    # aliases (for example, "decelerate to maintain distance"). Corrupt the
    # whole family together; flipping only `decelerate` while leaving
    # `maintain` made a correct any-of rubric score the negative unchanged.
    if "accelerate" in profiles:
        opposite_profile = "decelerate"
    elif "decelerate" in profiles:
        opposite_profile = "accelerate"
    elif profiles & {"maintain", "adapt"}:
        # A stable/adaptive-speed trace must not retain credit when its
        # reasoning instead promises acceleration.
        opposite_profile = "accelerate"
    else:
        opposite_profile = None

    new_commitments = []
    changed = False
    for c in claims.commitments:
        # Semantic axis first: an incidental "left/right" word attached to
        # a stop or slowdown must not turn a longitudinal contradiction into
        # an irrelevant direction flip. This now matches the generator
        # contract that longitudinal rewards ignore .direction.
        if opposite_profile is not None and c.speed_profile in {
            "accelerate", "decelerate", "maintain", "adapt"
        }:
            replacement = {"speed_profile": opposite_profile}
            if c.maneuver not in _LATERAL_MOVE_MANEUVERS:
                replacement["maneuver"] = opposite_profile
            c = dataclasses.replace(c, **replacement)
            changed = True
        direction = _DIRECTION_FLIP.get(c.direction) if c.direction else None
        if c.maneuver in _LATERAL_MOVE_MANEUVERS and direction is not None:
            new_commitments.append(dataclasses.replace(c, direction=direction))
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
        if c.maneuver in {"keep_distance", "proceed"}:
            new_commitments.append(
                dataclasses.replace(c, maneuver="stop", speed_profile="decelerate")
            )
            changed = True
            continue
        if c.maneuver == "reverse":
            new_commitments.append(dataclasses.replace(c, maneuver="proceed"))
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


def forced_departure_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Make stopped/creeping motion depart decisively along initial heading."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) < 2:
        return w
    direction = w[min(5, len(w) - 1), :2] - w[0, :2]
    norm = float(np.linalg.norm(direction))
    if norm < 0.1:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    ramp = np.linspace(0.0, 8.0, len(w))[:, None]
    w[:, :2] = w[0, :2] + ramp * direction
    return w


def forced_stop_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Make an accelerating/proceeding trajectory settle to a stop early."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) < 2:
        return w
    stop_i = max(1, min(len(w) - 1, len(w) // 4))
    w[stop_i:] = w[stop_i]
    return w


def mirrored_lateral_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Reflect lateral motion and force a meaningful opposite-side offset."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) < 2:
        return w
    rel_y = w[:, 1] - w[0, 1]
    mirrored = -rel_y
    if float(np.max(np.abs(rel_y))) < 2.0:
        sign = -1.0 if float(rel_y[-1]) >= 0.0 else 1.0
        mirrored = mirrored + sign * np.linspace(0.0, 3.0, len(w))
    w[:, 1] = w[0, 1] + mirrored
    return w


def forced_lane_departure_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Leave the GT-relative path corridor smoothly by roughly one lane."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) < 2:
        return w
    u = np.linspace(0.0, 1.0, len(w))
    smooth = u * u * (3.0 - 2.0 * u)
    sign = -1.0 if float(w[-1, 1] - w[0, 1]) >= 0.0 else 1.0
    w[:, 1] += sign * 4.0 * smooth
    return w


def oscillatory_speed_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Preserve path direction while making speed conspicuously non-cautious."""
    w = np.asarray(waypoints, dtype=np.float64)
    if len(w) < 3:
        return w.copy()
    delta = np.diff(w[:, :2], axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    unit = np.divide(delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 1e-8)
    factors = np.where((np.arange(len(delta)) // 4) % 2 == 0, 0.05, 2.50)
    out = np.empty_like(w)
    out[0] = w[0]
    out[1:, :2] = w[0, :2] + np.cumsum(unit * (lengths * factors)[:, None], axis=0)
    if w.shape[1] > 2:
        out[1:, 2:] = w[1:, 2:]
    return out


def surged_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """An unsafe over-progress counterfactual along the same coarse path."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) > 1:
        w[:, :2] = w[0, :2] + 2.5 * (w[:, :2] - w[0, :2])
    return w


def forced_forward_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Replace a reversing action with decisive straight-ahead progress."""
    w = np.asarray(waypoints, dtype=np.float64).copy()
    if len(w) > 1:
        w[:, 0] = w[0, 0] + np.linspace(0.0, 10.0, len(w))
        w[:, 1] = w[0, 1]
    return w


def build_perturbations(
    scene_id: str,
    claims,
    waypoints: np.ndarray,
    hz: float,
    tag: str = "gt",
    reward_spec: dict | None = None,
) -> list[GateCase]:
    """The positive plus corrupted variants of the SAME rollout.

    At generation time the GT (claims, waypoints) pair stands in for the
    rollout (tag="gt"); at selection time call this on the argmax rollout
    of a group to verify the function is sensitive around its own winner.
    """
    traj = _refeature(waypoints, hz, scene_id, tag)
    cases = [GateCase(f"positive:{tag}", claims, traj, "positive")]
    # Time reversal is not a guaranteed negative: a trajectory that slows in
    # the middle and then recovers can still slow when reversed (17/17 final
    # GT-gate failures in the fresh canary had this label error).  Use only
    # action-specific corruptions whose motion contradicts the scored intent.
    flat_wp = flattened_waypoints(waypoints)
    spec_features = {
        component.get("trajectory", {}).get("feature")
        for component in (reward_spec or {}).get("components", [])
        if isinstance(component.get("trajectory"), dict)
    }
    generic_no_reaction_is_negative = reward_spec is None or bool(
        spec_features
        & {
            "speed_drop",
            "speed_gain",
            "speed_reduction_fraction",
            "stationary_quality",
            "stop_dwell_fraction",
            "late_stationary_quality",
        }
    )
    if generic_no_reaction_is_negative and not _too_similar(waypoints, flat_wp):
        cases.append(
            GateCase(
                "perturb:no_reaction_traj",
                claims,
                _refeature(flat_wp, hz, scene_id, "flat"),
                "negative",
            )
        )
    if reward_spec is None:
        profiles = {c.speed_profile for c in claims.commitments}
        lateral_commitment = any(
            c.maneuver in _LATERAL_MOVE_MANEUVERS for c in claims.commitments
        )
    else:
        spec_components = reward_spec.get("components") or []
        profiles = {
            value
            for component in spec_components
            if isinstance(component.get("claim"), dict)
            and component["claim"].get("kind") == "commitment"
            and component["claim"].get("field") == "speed_profile"
            for value in component["claim"].get("any_of", [])
        }
        lateral_commitment = any(
            isinstance(component.get("claim"), dict)
            and component["claim"].get("kind") == "commitment"
            and component["claim"].get("field") == "maneuver"
            and set(component["claim"].get("any_of", [])) & _LATERAL_MOVE_MANEUVERS
            for component in spec_components
        )
    # Do not let low-motion scenes skip trajectory validation altogether.
    # Add a semantic opposite tailored to the claimed action whenever the
    # generic reverse/flatten transformations are insufficient.
    if "decelerate" in profiles and _too_similar(waypoints, flat_wp):
        depart_wp = forced_departure_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:accelerate_or_depart",
                claims,
                _refeature(depart_wp, hz, scene_id, "depart"),
                "negative",
            )
        )
    if "accelerate" in profiles:
        stop_wp = forced_stop_waypoints(waypoints)
        if not _too_similar(waypoints, stop_wp, min_dev_m=0.5):
            cases.append(
                GateCase(
                    "perturb:forced_stop",
                    claims,
                    _refeature(stop_wp, hz, scene_id, "stop"),
                    "negative",
                )
            )
    if "path_corridor_quality" in spec_features:
        depart_wp = forced_lane_departure_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:lane_departure_traj",
                claims,
                _refeature(depart_wp, hz, scene_id, "lanedepart"),
                "negative",
            )
        )
    elif lateral_commitment:
        mirror_wp = mirrored_lateral_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:opposite_lateral_traj",
                claims,
                _refeature(mirror_wp, hz, scene_id, "latmirror"),
                "negative",
            )
        )
    if "speed_stability_quality" in spec_features:
        oscillatory_wp = oscillatory_speed_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:unstable_speed_traj",
                claims,
                _refeature(oscillatory_wp, hz, scene_id, "unstable"),
                "negative",
            )
        )
    if "cautious_progress_quality" in spec_features:
        stop_wp = forced_stop_waypoints(waypoints)
        if _too_similar(waypoints, stop_wp, min_dev_m=0.5):
            stop_wp = np.repeat(np.asarray(waypoints, dtype=np.float64)[:1], len(waypoints), axis=0)
        cases.append(
            GateCase(
                "perturb:no_progress_traj",
                claims,
                _refeature(stop_wp, hz, scene_id, "noprogress"),
                "negative",
            )
        )
        surge_wp = surged_waypoints(waypoints)
        if _too_similar(waypoints, surge_wp, min_dev_m=0.5):
            surge_wp = forced_departure_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:unsafe_surge_traj",
                claims,
                _refeature(surge_wp, hz, scene_id, "surge"),
                "negative",
            )
        )
    if "heading_corridor_quality" in spec_features:
        forward_wp = forced_forward_waypoints(waypoints)
        cases.append(
            GateCase(
                "perturb:forward_instead_of_reverse_traj",
                claims,
                _refeature(forward_wp, hz, scene_id, "forward"),
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
    if np.isfinite(pos_score):
        for case in cases:
            s = scores[case.name]
            required_drop = case.min_drop if case.min_drop is not None else min_drop
            ceiling = pos_score - required_drop
            if case.kind == "negative" and np.isfinite(s) and s > ceiling:
                failures.append(
                    f"{case.name} scored {s:.2f}, needs <= {ceiling:.2f} (positive"
                    f" {pos_score:.2f} minus drop {required_drop}) -- a corrupted variant of"
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
        negative_components = {k: v for k, v in comp.items() if v < -1e-9}
        if negative_components:
            failures.append(
                f"{name}: components must be non-negative, got {negative_components}"
            )
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
                case_ceiling = pos_score - (
                    case.min_drop if case.min_drop is not None else min_drop
                )
                if not (
                    np.isfinite(s) and np.isfinite(case_ceiling) and s > case_ceiling
                ):
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
