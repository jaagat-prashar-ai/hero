# SPDX-License-Identifier: Apache-2.0
"""GT-derived, reward-independent rollout eligibility for ClipGen.

The generated reward must not get to define which sampled rollout counts as
its own positive.  This module derives a small target contract directly from
the NVIDIA GT CoC and GT trajectory, then uses it to distinguish two cases:

* a plausible target-following rollout that the candidate reward under-scores;
* a group containing no plausible positive, which must be resampled/skipped.

The contract is deliberately coarser than the generated rubric.  It recognizes
action families and tolerant kinematics; it does not reproduce the GT wording
or demand the exact GT trajectory.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np

from code_as_a_reward.clipgen.reward_spec import (
    validate_reward_spec,
    validate_reward_spec_semantics,
)


_DIRECTIONAL_LATERAL_MOVES = frozenset(
    {"lane_change", "nudge", "merge", "turn", "enter", "exit", "overtake"}
)
_LATERAL_MOVES = _DIRECTIONAL_LATERAL_MOVES | {"keep_lane"}
_BEHAVIOR_MANEUVERS = frozenset({"keep_lane", "keep_distance", "proceed", "reverse"})
_VEHICLE_FAMILY = frozenset(
    {"lead_vehicle", "stopped_vehicle", "cross_traffic", "vehicle_generic", "oncoming_vehicle"}
)


@dataclasses.dataclass(frozen=True)
class TargetContract:
    entities: frozenset[str]
    speed_profiles: frozenset[str]
    lateral_maneuvers: frozenset[str]
    lateral_directions: frozenset[str]
    requires_stop: bool
    gt_speed_drop_mps: float
    gt_speed_gain_mps: float
    gt_heading_change_deg: float
    gt_lateral_change_m: float
    behavior_maneuvers: frozenset[str] = frozenset()
    gt_reference_speed_mps: tuple[float, ...] = ()
    gt_reference_lateral_m: tuple[float, ...] = ()
    gt_reference_heading_deg: tuple[float, ...] = ()
    gt_reference_speed_scalar_mps: float = 0.0
    gt_speed_p90_deviation_mps: float = 0.0
    gt_progress_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": sorted(self.entities),
            "speed_profiles": sorted(self.speed_profiles),
            "lateral_maneuvers": sorted(self.lateral_maneuvers),
            "lateral_directions": sorted(self.lateral_directions),
            "requires_stop": self.requires_stop,
            "gt_speed_drop_mps": self.gt_speed_drop_mps,
            "gt_speed_gain_mps": self.gt_speed_gain_mps,
            "gt_heading_change_deg": self.gt_heading_change_deg,
            "gt_lateral_change_m": self.gt_lateral_change_m,
            "behavior_maneuvers": sorted(self.behavior_maneuvers),
            "gt_reference_speed_mps": list(self.gt_reference_speed_mps),
            "gt_reference_lateral_m": list(self.gt_reference_lateral_m),
            "gt_reference_heading_deg": list(self.gt_reference_heading_deg),
            "gt_reference_speed_scalar_mps": self.gt_reference_speed_scalar_mps,
            "gt_speed_p90_deviation_mps": self.gt_speed_p90_deviation_mps,
            "gt_progress_m": self.gt_progress_m,
        }


def target_contract_from_dict(payload: dict[str, Any]) -> TargetContract:
    """Restore the immutable contract carried in live repair evidence."""

    values = dict(payload)
    for key in (
        "entities",
        "speed_profiles",
        "lateral_maneuvers",
        "lateral_directions",
        "behavior_maneuvers",
    ):
        values[key] = frozenset(values.get(key) or [])
    for key in (
        "gt_reference_speed_mps",
        "gt_reference_lateral_m",
        "gt_reference_heading_deg",
    ):
        values[key] = tuple(values.get(key) or [])
    return TargetContract(**values)


@dataclasses.dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    failures: tuple[str, ...]


def _speed_extrema(traj: Any) -> tuple[float, float]:
    speed = np.asarray(traj.speed_mps, dtype=np.float64)
    if len(speed) < 2:
        return 0.0, 0.0
    early_n = max(1, min(len(speed) // 4, int(round(0.5 / traj.dt_s))))
    initial = float(np.mean(speed[:early_n]))
    later = speed[early_n:]
    if not len(later):
        return 0.0, 0.0
    late_n = max(1, min(len(speed) // 4, int(round(0.5 / traj.dt_s))))
    return max(0.0, initial - float(np.min(later))), max(
        0.0, float(np.mean(speed[-late_n:])) - initial
    )


def _coarse_series(values: Any, n: int = 9) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return tuple(0.0 for _ in range(n))
    sampled = np.interp(
        np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(arr)), arr
    )
    return tuple(float(value) for value in sampled)


def derive_target_contract(gt_claims: Any, gt_traj: Any) -> TargetContract:
    speed_profiles = frozenset(
        c.speed_profile for c in gt_claims.commitments if c.speed_profile is not None
    )
    lateral = frozenset(
        c.maneuver for c in gt_claims.commitments if c.maneuver in _LATERAL_MOVES
    )
    behaviors = frozenset(
        c.maneuver for c in gt_claims.commitments if c.maneuver in _BEHAVIOR_MANEUVERS
    )
    directions = frozenset(
        c.direction
        for c in gt_claims.commitments
        if c.maneuver in _DIRECTIONAL_LATERAL_MOVES and c.direction in {"left", "right"}
    )
    requires_stop = any(c.maneuver in {"stop", "wait"} for c in gt_claims.commitments)
    drop, gain = _speed_extrema(gt_traj)
    speed = np.asarray(gt_traj.speed_mps, dtype=np.float64)
    speed_scalar = float(np.median(speed)) if len(speed) else 0.0
    return TargetContract(
        entities=frozenset(c.entity for c in gt_claims.perceptual),
        speed_profiles=speed_profiles,
        lateral_maneuvers=lateral,
        lateral_directions=directions,
        requires_stop=requires_stop,
        gt_speed_drop_mps=drop,
        gt_speed_gain_mps=gain,
        gt_heading_change_deg=float(gt_traj.total_heading_change_deg),
        gt_lateral_change_m=float(gt_traj.final_lateral_offset_m),
        behavior_maneuvers=behaviors,
        gt_reference_speed_mps=_coarse_series(speed),
        gt_reference_lateral_m=_coarse_series(gt_traj.lateral_offset_m),
        gt_reference_heading_deg=_coarse_series(gt_traj.heading_deg),
        gt_reference_speed_scalar_mps=speed_scalar,
        gt_speed_p90_deviation_mps=(
            float(np.percentile(np.abs(speed - speed_scalar), 90)) if len(speed) else 0.0
        ),
        gt_progress_m=float(np.sum(speed) * gt_traj.dt_s) if len(speed) else 0.0,
    )


def _entities_overlap(target: frozenset[str], observed: set[str]) -> bool:
    if target & observed:
        return True
    return bool(target & _VEHICLE_FAMILY and observed & _VEHICLE_FAMILY)


def classify_rollout(contract: TargetContract, claims: Any, traj: Any) -> EligibilityResult:
    """Coarse, independent target check used before candidate argmax gating."""

    failures: list[str] = []
    observed_profiles = {c.speed_profile for c in claims.commitments if c.speed_profile}
    observed_lateral = {c.maneuver for c in claims.commitments if c.maneuver in _LATERAL_MOVES}
    observed_behaviors = {
        c.maneuver for c in claims.commitments if c.maneuver in _BEHAVIOR_MANEUVERS
    }
    observed_directions = {
        c.direction
        for c in claims.commitments
        if c.maneuver in _LATERAL_MOVES and c.direction in {"left", "right"}
    }
    observed_entities = {c.entity for c in claims.perceptual}

    if contract.speed_profiles and not contract.speed_profiles & observed_profiles:
        failures.append(
            "missing target speed-profile commitment " + "/".join(sorted(contract.speed_profiles))
        )
    if contract.lateral_maneuvers and not contract.lateral_maneuvers & observed_lateral:
        failures.append(
            "missing target lateral commitment " + "/".join(sorted(contract.lateral_maneuvers))
        )
    behavior_alias = (
        "proceed" in contract.behavior_maneuvers and "adapt" in observed_profiles
    ) or (
        "keep_distance" in contract.behavior_maneuvers and "maintain" in observed_profiles
    )
    if (
        contract.behavior_maneuvers
        and not contract.behavior_maneuvers & observed_behaviors
        and not behavior_alias
    ):
        failures.append(
            "missing target behavior commitment "
            + "/".join(sorted(contract.behavior_maneuvers))
        )
    if contract.lateral_directions and not contract.lateral_directions & observed_directions:
        failures.append(
            "missing target lateral direction " + "/".join(sorted(contract.lateral_directions))
        )
    if not contract.lateral_maneuvers and observed_lateral:
        failures.append("adds an unsupported lateral maneuver: " + "/".join(sorted(observed_lateral)))
    if contract.entities and not _entities_overlap(contract.entities, observed_entities):
        failures.append("does not mention a GT-relevant entity family")

    drop, gain = _speed_extrema(traj)
    speed = np.asarray(traj.speed_mps, dtype=np.float64)
    if contract.requires_stop:
        late_n = max(1, min(len(speed), int(round(1.0 / traj.dt_s))))
        late_p90 = float(np.percentile(speed[-late_n:], 90)) if len(speed) else float("inf")
        stopped_fraction = float(np.mean(speed <= 0.5)) if len(speed) else 0.0
        if late_p90 > 0.75 or stopped_fraction < 0.50:
            failures.append(
                f"does not sustain the stop (late p90 {late_p90:.2f} m/s, "
                f"stopped fraction {stopped_fraction:.2f})"
            )
    elif "decelerate" in contract.speed_profiles:
        required = max(0.25, 0.35 * contract.gt_speed_drop_mps)
        if drop < required:
            failures.append(f"speed drop {drop:.2f} m/s is below tolerant target {required:.2f}")
    if "accelerate" in contract.speed_profiles:
        required = max(0.25, 0.35 * contract.gt_speed_gain_mps)
        if gain < required:
            failures.append(f"speed gain {gain:.2f} m/s is below tolerant target {required:.2f}")

    if contract.lateral_directions:
        direction = next(iter(contract.lateral_directions))
        # Alpamayo keyframe ego coordinates use positive heading/lateral for
        # left motion and negative heading/lateral for right motion.
        sign = 1.0 if direction == "left" else -1.0
        signed_heading = float(traj.total_heading_change_deg) * sign
        signed_lateral = float(traj.final_lateral_offset_m) * sign
        gt_heading = abs(contract.gt_heading_change_deg)
        gt_lateral = abs(contract.gt_lateral_change_m)
        heading_ok = signed_heading >= max(3.0, 0.25 * gt_heading)
        lateral_ok = signed_lateral >= max(0.35, 0.25 * gt_lateral)
        if not (heading_ok or lateral_ok):
            failures.append("trajectory does not execute the target lateral direction")

    if "maintain" in contract.speed_profiles and not (
        contract.speed_profiles & {"accelerate", "decelerate"}
    ):
        tolerance = max(2.0, 2.0 * contract.gt_speed_p90_deviation_mps)
        observed = float(np.median(speed)) if len(speed) else 0.0
        if abs(observed - contract.gt_reference_speed_scalar_mps) > tolerance:
            failures.append("trajectory lies outside the target stable-speed envelope")

    return EligibilityResult(eligible=not failures, failures=tuple(failures))


def validate_gt_target(contract: TargetContract, gt_traj: Any) -> list[str]:
    """Reject contradictory/unverifiable GT inputs before asking an LLM.

    A generated reward cannot simultaneously treat the NVIDIA CoC as the
    semantic target and the NVIDIA action as the empirical positive when
    those two sources disagree. Such clips are data-quality quarantines,
    not reward-generation failures.
    """

    failures: list[str] = []
    if "accelerate" in contract.speed_profiles and contract.gt_speed_gain_mps < 0.25:
        failures.append(
            "GT CoC commits to acceleration but the expert action has no sustained "
            f"late speed gain ({contract.gt_speed_gain_mps:.2f} m/s)"
        )
    if (
        "decelerate" in contract.speed_profiles
        and not contract.requires_stop
        and contract.gt_speed_drop_mps < 0.25
    ):
        failures.append(
            "GT CoC commits to deceleration but the expert action has no measurable "
            f"speed drop ({contract.gt_speed_drop_mps:.2f} m/s)"
        )
    if contract.requires_stop:
        speed = np.asarray(gt_traj.speed_mps, dtype=np.float64)
        late_n = max(1, min(len(speed), int(round(1.0 / gt_traj.dt_s))))
        late_p90 = float(np.percentile(speed[-late_n:], 90)) if len(speed) else float("inf")
        stopped_fraction = float(np.mean(speed <= 0.5)) if len(speed) else 0.0
        if late_p90 > 0.75 or stopped_fraction < 0.50:
            failures.append(
                "GT CoC requires a stop but the expert action does not sustain one "
                f"(late p90 {late_p90:.2f} m/s, stopped fraction {stopped_fraction:.2f})"
            )
    if contract.lateral_directions:
        direction = next(iter(contract.lateral_directions))
        sign = 1.0 if direction == "left" else -1.0
        heading_ok = sign * contract.gt_heading_change_deg >= 3.0
        lateral_ok = sign * contract.gt_lateral_change_m >= 0.35
        if not (heading_ok or lateral_ok):
            failures.append(
                f"GT CoC commits {direction} but the expert action does not execute that direction"
            )
    if "proceed" in contract.behavior_maneuvers and contract.gt_progress_m < 1.0:
        failures.append(
            "GT CoC commits to proceeding but the expert action makes less than 1 m progress"
        )
    discriminative = (
        bool(contract.speed_profiles & {"accelerate", "decelerate", "maintain", "adapt"})
        or contract.requires_stop
        or bool(contract.lateral_maneuvers)
        or bool(contract.behavior_maneuvers)
    )
    if not discriminative:
        failures.append(
            "GT target has no currently verifiable discriminative action family "
            "(speed envelope, stop, lane/path, following, or progress behavior)"
        )
    return failures


def calibrate_spec_against_target(
    spec: dict[str, Any], contract: TargetContract
) -> dict[str, Any]:
    """Mechanically calibrate an LLM's semantic spec to the GT contract.

    The model is useful for choosing a compact scene rubric, but it is a poor
    numeric constraint solver: fresh-canary retries repeatedly missed the
    exact weight and 1.15--1.25x threshold bands.  Those quantities are
    derived data, so the compiler owns them.  This keeps generation focused
    on semantics and makes every emitted curve reproducible.
    """

    # Validate only canonical LLM-owned semantics here. All numeric fields
    # and the primary execution component are replaced below from sealed GT
    # evidence, so malformed generated tolerances/anchors must not consume an
    # LLM retry or reduce corpus coverage.
    out = validate_reward_spec_semantics(copy.deepcopy(spec))
    components = out["components"]
    perception = [c for c in components if c["claim"]["kind"] == "perceptual"]

    # Keep the LLM's scene interpretation, but compile action semantics into
    # one primary conjunction.  Splitting the 0.60 action budget across
    # several weak/dead axes made it mathematically impossible for a single
    # action corruption to achieve MIN_DROP=0.40.
    if contract.entities:
        perception = [
            c
            for c in perception
            if _entities_overlap(contract.entities, frozenset(c["claim"].get("any_of", [])))
        ]
        if not perception:
            perception = [
                {
                    "name": "target_scene_context",
                    "weight": 0.40,
                    "claim": {
                        "kind": "perceptual",
                        "field": "entity",
                        "any_of": sorted(contract.entities),
                    },
                    "trajectory": None,
                }
            ]
        old = sum(float(c["weight"]) for c in perception)
        for c in perception:
            c["weight"] = 0.40 * float(c["weight"]) / old
        perception[-1]["weight"] = 0.40 - sum(float(c["weight"]) for c in perception[:-1])
        action_weight = 0.60
    else:
        perception = []
        action_weight = 1.0

    direction = next(iter(contract.lateral_directions), None)
    behavior = contract.behavior_maneuvers
    profiles = contract.speed_profiles
    if contract.requires_stop:
        field, values, feature = "speed_profile", sorted(profiles or {"decelerate"}), "late_stationary_quality"
    elif direction is not None and contract.lateral_maneuvers:
        field, values, feature = "maneuver", sorted(contract.lateral_maneuvers), "path_corridor_quality"
    elif "decelerate" in profiles:
        # The execution feature defines the commitment that must be stated.
        # Do not let an accompanying "maintain"/keep-distance claim collect
        # credit for an unstated deceleration merely because both appeared in
        # the GT annotation.
        field, values, feature = "speed_profile", ["decelerate"], "speed_drop"
    elif "accelerate" in profiles:
        # Likewise, speed_gain requires an acceleration commitment.  The old
        # union accepted "maintain" here, which gave full acceleration credit
        # to reasoning that only said "keep distance".
        field, values, feature = "speed_profile", ["accelerate"], "speed_gain"
    elif "keep_lane" in behavior:
        field, values, feature = "maneuver", ["keep_lane"], "path_corridor_quality"
    elif "keep_distance" in behavior:
        field, values, feature = "maneuver", ["keep_distance"], "cautious_progress_quality"
    elif "adapt" in profiles:
        field, values, feature = "speed_profile", sorted(profiles), "cautious_progress_quality"
    elif "maintain" in profiles:
        field, values, feature = "speed_profile", sorted(profiles), "speed_stability_quality"
    elif "proceed" in behavior:
        field, values, feature = "maneuver", ["proceed"], "cautious_progress_quality"
    elif "reverse" in behavior:
        field, values, feature = "maneuver", ["reverse"], "heading_corridor_quality"
    else:
        # validate_gt_target rejects this case before generation; retaining a
        # defensive error here makes direct callers fail loudly.
        raise ValueError("target contract has no compilable primary behavior")

    used_names = {c["name"] for c in perception}
    name = "primary_executed_behavior"
    suffix = 2
    while name in used_names:
        name = f"primary_executed_behavior_{suffix}"
        suffix += 1
    claim = {
        "kind": "commitment",
        "field": field,
        "any_of": values,
        "direction": direction if field == "maneuver" and direction else "any",
    }
    rule: dict[str, Any] = {
        "feature": feature,
        "window_s": [0.0, 6.4],
        "floor": 0.0,
        "full": 1.05,
        "power": 0.30,
    }
    if feature == "late_stationary_quality":
        rule.update(window_s=[3.0, 6.4], reference_speed_mps=1.0)
    elif feature in {"speed_drop", "speed_gain"}:
        magnitude = contract.gt_speed_drop_mps if feature == "speed_drop" else contract.gt_speed_gain_mps
        rule.update(floor=0.05 * magnitude, full=1.20 * magnitude)
    elif feature == "path_corridor_quality":
        rule.update(
            reference_lateral_m=list(contract.gt_reference_lateral_m),
            corridor_half_width_m=1.75,
        )
    elif feature == "heading_corridor_quality":
        rule.update(
            reference_heading_deg=list(contract.gt_reference_heading_deg),
            heading_tolerance_deg=45.0,
        )
    elif feature == "speed_stability_quality":
        rule.update(
            reference_speed_mps=contract.gt_reference_speed_scalar_mps,
            speed_tolerance_mps=max(
                1.5,
                0.20 * contract.gt_reference_speed_scalar_mps,
                2.0 * contract.gt_speed_p90_deviation_mps,
            ),
        )
    elif feature == "cautious_progress_quality":
        profile = np.asarray(contract.gt_reference_speed_mps, dtype=np.float64)
        scalar = float(np.median(profile)) if len(profile) else 0.0
        variation = float(np.percentile(np.abs(profile - scalar), 90)) if len(profile) else 0.0
        rule.update(
            reference_speed_profile_mps=list(contract.gt_reference_speed_mps),
            speed_tolerance_mps=max(2.0, 2.0 * variation),
            reference_progress_m=contract.gt_progress_m,
            progress_tolerance_m=max(6.0, 0.35 * contract.gt_progress_m),
        )

    out["components"] = [
        *perception,
        {"name": name, "weight": action_weight, "claim": claim, "trajectory": rule},
    ]
    return validate_reward_spec(out)


def validate_spec_against_target(spec: dict[str, Any], contract: TargetContract) -> list[str]:
    """Reject semantically mis-shaped specs before empirical gates run."""

    components = spec.get("components") or []
    commitment_components = [
        c for c in components if isinstance(c.get("claim"), dict) and c["claim"].get("kind") == "commitment"
    ]
    perception_components = [
        c for c in components if isinstance(c.get("claim"), dict) and c["claim"].get("kind") == "perceptual"
    ]
    claimed_profiles = {
        value
        for component in commitment_components
        if component["claim"].get("field") == "speed_profile"
        for value in component["claim"].get("any_of", [])
    }
    claimed_maneuvers = {
        value
        for component in commitment_components
        if component["claim"].get("field") == "maneuver"
        for value in component["claim"].get("any_of", [])
    }
    claimed_lateral_maneuvers = claimed_maneuvers & _LATERAL_MOVES
    features = {
        component.get("trajectory", {}).get("feature")
        for component in commitment_components
        if isinstance(component.get("trajectory"), dict)
    }
    failures: list[str] = []
    scored_entities = {
        value
        for component in perception_components
        if component["claim"].get("field") == "entity"
        for value in component["claim"].get("any_of", [])
    }
    relevant_entity_weight = sum(
        float(component.get("weight", 0.0))
        for component in perception_components
        if _entities_overlap(
            contract.entities,
            frozenset(component["claim"].get("any_of", [])),
        )
    )
    if contract.entities and not _entities_overlap(contract.entities, scored_entities):
        failures.append("spec does not score any GT-relevant entity family")
    elif contract.entities and relevant_entity_weight < 0.40 - 1e-9:
        failures.append(
            "GT-relevant entity mention needs at least 0.40 total weight so a wrong-scene "
            "reasoning trace drops by the required 0.40 while still needing execution "
            "credit to clear POS_MIN"
        )
    # Require the profile that corresponds to the selected execution axis,
    # not every verb that happened to coexist in the GT sentence.  Requiring
    # the union forced calibration to let a maintain-only claim earn a
    # speed_gain component (and vice versa), defeating reasoning/action
    # alignment.  keep_distance remains a valid exact substitute for a
    # maintain-only contract.
    if contract.requires_stop or "decelerate" in contract.speed_profiles:
        required_profiles = frozenset({"decelerate"})
    elif "accelerate" in contract.speed_profiles:
        required_profiles = frozenset({"accelerate"})
    elif "adapt" in contract.speed_profiles:
        required_profiles = frozenset({"adapt"})
    elif "maintain" in contract.speed_profiles:
        required_profiles = frozenset({"maintain"})
    else:
        required_profiles = frozenset()
    missing_profiles = (
        frozenset()
        if required_profiles == {"maintain"}
        and "keep_distance" in contract.behavior_maneuvers
        and "keep_distance" in claimed_maneuvers
        else required_profiles - claimed_profiles
    )
    if missing_profiles:
        failures.append(
            "spec does not score the GT-decisive speed-profile wording; "
            f"missing {sorted(missing_profiles)}"
        )
    if contract.lateral_maneuvers and not contract.lateral_maneuvers & claimed_lateral_maneuvers:
        failures.append("spec does not score the GT lateral maneuver family")
    if not contract.lateral_maneuvers and claimed_lateral_maneuvers:
        failures.append(
            "spec adds a lateral scoring axis that is absent from the GT target; "
            "unrelated corruptions would preserve the primary score"
        )
    behavior_alias = (
        "proceed" in contract.behavior_maneuvers and "adapt" in claimed_profiles
    ) or (
        "keep_distance" in contract.behavior_maneuvers and "maintain" in claimed_profiles
    )
    if (
        contract.behavior_maneuvers
        and not contract.behavior_maneuvers & claimed_maneuvers
        and not behavior_alias
    ):
        # maintain/adapt speed profiles are valid realizations for the
        # corresponding longitudinal behavior; keep_lane/proceed are not.
        unresolved = contract.behavior_maneuvers - {"keep_distance"}
        if unresolved:
            failures.append("spec does not score the GT path/progress behavior family")
    if contract.requires_stop and not features & {"stop_dwell_fraction", "late_stationary_quality"}:
        failures.append(
            "GT requires a sustained stop; speed_drop alone is insufficient. Use "
            "stop_dwell_fraction or late_stationary_quality."
        )
    if contract.requires_stop:
        stop_rules = [
            component.get("trajectory")
            for component in commitment_components
            if isinstance(component.get("trajectory"), dict)
            and component["trajectory"].get("feature")
            in {"stop_dwell_fraction", "late_stationary_quality"}
        ]
        for rule in stop_rules:
            reference = float(rule.get("reference_speed_mps", 0.0))
            floor = float(rule.get("floor", 0.0))
            full = float(rule.get("full", 0.0))
            if not 0.25 <= reference <= 1.0:
                failures.append(
                    "sustained-stop reference_speed_mps must be in [0.25, 1.0] so "
                    "near-stationary quality is tolerant rather than numerically brittle"
                )
            if floor > 0.25 or full < 1.05:
                failures.append(
                    "normalized sustained-stop features require floor <= 0.25 and full >= "
                    "1.05; keeping full just above the theoretical 1.0 maximum prevents "
                    "many correct rollouts from tying at a saturated score"
                )
            if (
                rule.get("feature") == "stop_dwell_fraction"
                and isinstance(rule.get("window_s"), list)
                and float(rule["window_s"][0]) < 3.0 - 1e-9
            ):
                failures.append(
                    "stop_dwell_fraction window must start at or after 3.0s so a "
                    "time-reversed stop cannot retain early dwell credit"
                )
    if "accelerate" in contract.speed_profiles and "speed_gain" not in features:
        failures.append("GT acceleration requires a speed_gain execution component")
    if (
        "decelerate" in contract.speed_profiles
        and not contract.requires_stop
        and not features & {"speed_drop", "speed_reduction_fraction"}
    ):
        failures.append("GT deceleration requires speed_drop or speed_reduction_fraction")
    if contract.lateral_maneuvers and "path_corridor_quality" not in features:
        failures.append("GT lane/path behavior requires path_corridor_quality")
    if "reverse" in contract.behavior_maneuvers and "heading_corridor_quality" not in features:
        failures.append("GT reverse behavior requires heading_corridor_quality")
    if (
        "maintain" in contract.speed_profiles
        and not contract.speed_profiles & {"accelerate", "decelerate"}
        and "keep_distance" not in contract.behavior_maneuvers
        and "speed_stability_quality" not in features
    ):
        failures.append("GT maintain-speed behavior requires speed_stability_quality")
    if (
        not contract.speed_profiles & {"accelerate", "decelerate"}
        and ("adapt" in contract.speed_profiles or contract.behavior_maneuvers & {"keep_distance", "proceed"})
        and "cautious_progress_quality" not in features
    ):
        failures.append("GT cautious/progress behavior requires cautious_progress_quality")

    # Keep the expert comfortably positive without making it the saturation
    # point. A `full` value below the measured GT magnitude caused unseen
    # rollout groups to collapse to repeated 1.0 scores and zero GRPO
    # resolution. The lower bound is mechanical so prompt drift cannot
    # reintroduce that failure mode.
    gt_magnitudes = {
        "speed_drop": contract.gt_speed_drop_mps,
        "speed_gain": contract.gt_speed_gain_mps,
        "heading_left": abs(contract.gt_heading_change_deg),
        "heading_right": abs(contract.gt_heading_change_deg),
        "lateral_left": abs(contract.gt_lateral_change_m),
        "lateral_right": abs(contract.gt_lateral_change_m),
    }
    for component in commitment_components:
        rule = component.get("trajectory")
        if not isinstance(rule, dict):
            continue
        magnitude = float(gt_magnitudes.get(rule.get("feature"), 0.0))
        if magnitude <= 1e-9 or rule.get("full") is None:
            continue
        full = float(rule["full"])
        floor = float(rule.get("floor", 0.0))
        if full < 1.15 * magnitude - 1e-9:
            failures.append(
                f"{rule.get('feature')} full must be at least 1.15x measured GT "
                f"magnitude ({magnitude:.3f}) to preserve rollout score resolution"
            )
        if full > 1.25 * magnitude + 1e-9:
            failures.append(
                f"{rule.get('feature')} full must be at most 1.25x measured GT "
                f"magnitude ({magnitude:.3f}) so competent execution contributes "
                "the required 0.40 score"
            )
        if floor > 0.10 * magnitude + 1e-9:
            failures.append(
                f"{rule.get('feature')} floor must be at most 0.10x measured GT "
                f"magnitude ({magnitude:.3f}) so plausible positives retain tolerance"
            )
    return failures
