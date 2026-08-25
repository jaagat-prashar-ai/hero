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

from code_as_a_reward.clipgen.reward_spec import validate_reward_spec


_LATERAL_MOVES = frozenset({"lane_change", "nudge", "merge", "turn", "enter", "exit"})
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
        }


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


def derive_target_contract(gt_claims: Any, gt_traj: Any) -> TargetContract:
    speed_profiles = frozenset(
        c.speed_profile for c in gt_claims.commitments if c.speed_profile is not None
    )
    lateral = frozenset(
        c.maneuver for c in gt_claims.commitments if c.maneuver in _LATERAL_MOVES
    )
    directions = frozenset(
        c.direction
        for c in gt_claims.commitments
        if c.maneuver in _LATERAL_MOVES and c.direction in {"left", "right"}
    )
    requires_stop = any(c.maneuver in {"stop", "wait"} for c in gt_claims.commitments)
    drop, gain = _speed_extrema(gt_traj)
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
    discriminative = (
        bool(contract.speed_profiles & {"accelerate", "decelerate"})
        or contract.requires_stop
        or bool(contract.lateral_maneuvers)
    )
    if not discriminative:
        failures.append(
            "GT target has no currently verifiable discriminative action family "
            "(accelerate, decelerate/stop, or lateral maneuver)"
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

    out = validate_reward_spec(copy.deepcopy(spec))
    components = out["components"]
    perception = [c for c in components if c["claim"]["kind"] == "perceptual"]
    commitments = [c for c in components if c["claim"]["kind"] == "commitment"]

    if contract.entities:
        relevant = [
            c
            for c in perception
            if _entities_overlap(
                contract.entities, frozenset(c["claim"].get("any_of", []))
            )
        ]
        if not relevant:
            used = {c["name"] for c in components}
            name = "target_scene_context"
            suffix = 2
            while name in used:
                name = f"target_scene_context_{suffix}"
                suffix += 1
            relevant = [
                {
                    "name": name,
                    "weight": 0.40,
                    "claim": {
                        "kind": "perceptual",
                        "field": "entity",
                        "any_of": sorted(contract.entities),
                    },
                    "trajectory": None,
                }
            ]
        perception = relevant
        perception_budget = 0.40
        commitment_budget = 0.60
    else:
        # With no GT-verifiable scene entity, mention-only credit cannot be
        # independently anchored.  Spend the whole budget on executed intent.
        perception = []
        perception_budget = 0.0
        commitment_budget = 1.0

    def rebudget(items: list[dict[str, Any]], budget: float) -> None:
        if not items:
            return
        old = sum(float(item["weight"]) for item in items)
        if old <= 0.0:
            each = budget / len(items)
            for item in items:
                item["weight"] = each
        else:
            for item in items:
                item["weight"] = budget * float(item["weight"]) / old
        # Make the sum exact after floating-point proportional allocation.
        items[-1]["weight"] = budget - sum(float(i["weight"]) for i in items[:-1])

    rebudget(perception, perception_budget)
    rebudget(commitments, commitment_budget)

    direction = next(iter(contract.lateral_directions), None)
    lateral_feature = None
    if direction is not None:
        if abs(contract.gt_heading_change_deg) >= 3.0:
            lateral_feature = f"heading_{direction}"
        elif abs(contract.gt_lateral_change_m) >= 0.35:
            lateral_feature = f"lateral_{direction}"

    longitudinal = [
        c for c in commitments if c["claim"].get("field") == "speed_profile"
    ]
    lateral = [c for c in commitments if c["claim"].get("field") == "maneuver"]
    for component in longitudinal:
        if contract.speed_profiles:
            component["claim"]["any_of"] = sorted(contract.speed_profiles)
        component["claim"]["direction"] = "any"
    for component in lateral:
        if contract.lateral_maneuvers:
            component["claim"]["any_of"] = sorted(contract.lateral_maneuvers)
        if direction is not None:
            component["claim"]["direction"] = direction
            if lateral_feature is not None:
                component["trajectory"]["feature"] = lateral_feature

    # Guarantee the required longitudinal execution family even if the model
    # chose a valid-but-wrong feature.  Stop targets retain a second speed-drop
    # component when proposed, but always get a sustained late-stop component.
    if longitudinal:
        if contract.requires_stop:
            if not any(
                c["trajectory"]["feature"]
                in {"stop_dwell_fraction", "late_stationary_quality"}
                for c in longitudinal
            ):
                longitudinal[-1]["trajectory"]["feature"] = "late_stationary_quality"
        elif "accelerate" in contract.speed_profiles and "decelerate" not in contract.speed_profiles:
            for component in longitudinal:
                component["trajectory"]["feature"] = "speed_gain"
        elif "decelerate" in contract.speed_profiles:
            for component in longitudinal:
                component["trajectory"]["feature"] = "speed_drop"

    magnitudes = {
        "speed_drop": contract.gt_speed_drop_mps,
        "speed_gain": contract.gt_speed_gain_mps,
        "heading_left": abs(contract.gt_heading_change_deg),
        "heading_right": abs(contract.gt_heading_change_deg),
        "lateral_left": abs(contract.gt_lateral_change_m),
        "lateral_right": abs(contract.gt_lateral_change_m),
    }
    for component in commitments:
        rule = component["trajectory"]
        feature = rule["feature"]
        if feature in {"stop_dwell_fraction", "late_stationary_quality"}:
            rule.update(
                {
                    "window_s": [3.0, 6.4],
                    "floor": 0.0,
                    "full": 1.05,
                    "reference_speed_mps": 1.0,
                    "power": 0.30,
                }
            )
            continue
        magnitude = float(magnitudes.get(feature, 0.0))
        if magnitude > 1e-9:
            rule.update(
                {
                    "window_s": [0.0, 6.4],
                    "floor": 0.05 * magnitude,
                    "full": 1.20 * magnitude,
                    "power": 0.30,
                }
            )
            rule.pop("reference_speed_mps", None)

    out["components"] = [*perception, *commitments]
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
    missing_profiles = contract.speed_profiles - claimed_profiles
    if missing_profiles:
        failures.append(
            "spec does not score every GT-supported speed-profile wording; "
            f"missing {sorted(missing_profiles)}"
        )
    if contract.lateral_maneuvers and not contract.lateral_maneuvers & claimed_maneuvers:
        failures.append("spec does not score the GT lateral maneuver family")
    if not contract.lateral_maneuvers and claimed_maneuvers:
        failures.append(
            "spec adds a lateral scoring axis that is absent from the GT target; "
            "unrelated corruptions would preserve the primary score"
        )
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
