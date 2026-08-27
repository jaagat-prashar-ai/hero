# SPDX-License-Identifier: Apache-2.0
"""Validated ClipGen reward DSL and deterministic evaluator/compiler.

The language model chooses semantic components. The compiler owns weights,
GT-relative execution features, and calibrated numbers; it also writes all
executable Python. Every accepted final spec has non-negative weights summing
exactly to one, bounded scene-reasoning credit, at least one
commitment/trajectory conjunction, and monotonic graded trajectory curves.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from code_as_a_reward.coc_claim_parser import (
    ENTITY_PATTERNS,
    MANEUVER_PATTERNS,
)

SCHEMA_VERSION = "clipgen.reward.v1"
MAX_PERCEPTION_WEIGHT = 0.40
MAX_HORIZON_S = 6.5

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_ENTITIES = frozenset(key for key, _ in ENTITY_PATTERNS)
_MANEUVERS = frozenset(key for key, *_ in MANEUVER_PATTERNS)
_SPEED_PROFILES = frozenset({"accelerate", "decelerate", "maintain", "adapt"})
_LATERAL_MANEUVERS = frozenset(
    {"lane_change", "nudge", "merge", "turn", "enter", "exit", "overtake", "keep_lane"}
)
_MEASURABLE_MANEUVERS = _LATERAL_MANEUVERS | {"keep_distance", "proceed", "reverse"}
_FEATURES = frozenset(
    {
        "speed_drop",
        "speed_gain",
        "speed_reduction_fraction",
        "heading_left",
        "heading_right",
        "lateral_left",
        "lateral_right",
        "stationary_quality",
        "stop_dwell_fraction",
        "late_stationary_quality",
        # Broad GT-relative contracts for behaviors that are meaningful but
        # are not an acceleration, stop, or lane-change magnitude.  These
        # compare coarse envelopes, never exact waypoint traces.
        "speed_stability_quality",
        "cautious_progress_quality",
        "path_corridor_quality",
        "heading_corridor_quality",
    }
)


class RewardSpecError(ValueError):
    """The LLM-proposed reward specification violates the DSL contract."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RewardSpecError(f"{label} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardSpecError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise RewardSpecError(f"{label} must be finite, got {value!r}")
    return number


def _validate_claim(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RewardSpecError(f"{label} must be an object")
    kind = raw.get("kind")
    field = raw.get("field")
    values = raw.get("any_of")
    if kind not in {"perceptual", "commitment"}:
        raise RewardSpecError(f"{label}.kind must be perceptual or commitment")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise RewardSpecError(f"{label}.any_of must be a non-empty string list")
    values = sorted(set(values))
    out: dict[str, Any] = {"kind": kind, "field": field, "any_of": values}
    if kind == "perceptual":
        if field != "entity":
            raise RewardSpecError(f"{label}: perceptual claims may only match entity")
        unknown = set(values) - _ENTITIES
        if unknown:
            raise RewardSpecError(f"{label}: unknown perceptual entities {sorted(unknown)}")
    else:
        if field == "speed_profile":
            unknown = set(values) - _SPEED_PROFILES
        elif field == "maneuver":
            unknown = set(values) - _MANEUVERS
        else:
            raise RewardSpecError(
                f"{label}: commitment field must be speed_profile or maneuver"
            )
        if unknown:
            raise RewardSpecError(f"{label}: unknown commitment values {sorted(unknown)}")
        direction = raw.get("direction", "any")
        if direction not in {"any", "left", "right"}:
            raise RewardSpecError(f"{label}.direction must be any, left, or right")
        out["direction"] = direction
        if field == "maneuver" and not set(values) <= _MEASURABLE_MANEUVERS:
            raise RewardSpecError(
                f"{label}: exact maneuver matching is restricted to measurable path, "
                "following, and progress behaviors; use speed_profile otherwise"
            )
    return out


def _validate_trajectory(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RewardSpecError(f"{label} must be an object")
    feature = raw.get("feature")
    if feature not in _FEATURES:
        raise RewardSpecError(f"{label}.feature must be one of {sorted(_FEATURES)}")
    window = raw.get("window_s")
    if (
        not isinstance(window, list)
        or len(window) != 2
    ):
        raise RewardSpecError(f"{label}.window_s must be [start_s, end_s]")
    start = _finite_number(window[0], f"{label}.window_s[0]")
    end = _finite_number(window[1], f"{label}.window_s[1]")
    if start < 0.0 or end <= start or end > MAX_HORIZON_S:
        raise RewardSpecError(
            f"{label}.window_s must satisfy 0 <= start < end <= {MAX_HORIZON_S}"
        )
    floor = _finite_number(raw.get("floor"), f"{label}.floor")
    full = _finite_number(raw.get("full"), f"{label}.full")
    if floor < 0.0 or full <= floor:
        raise RewardSpecError(f"{label} requires 0 <= floor < full")
    out = {
        "feature": feature,
        "window_s": [start, end],
        "floor": floor,
        "full": full,
    }
    power = _finite_number(raw.get("power", 1.0), f"{label}.power")
    if not 0.25 <= power <= 2.0:
        raise RewardSpecError(f"{label}.power must be in [0.25, 2.0]")
    out["power"] = power
    if feature in {"stationary_quality", "stop_dwell_fraction", "late_stationary_quality"}:
        reference = _finite_number(raw.get("reference_speed_mps"), f"{label}.reference_speed_mps")
        if reference <= 0.0:
            raise RewardSpecError(f"{label}.reference_speed_mps must be > 0")
        out["reference_speed_mps"] = reference
    elif feature == "speed_stability_quality":
        reference = _finite_number(raw.get("reference_speed_mps"), f"{label}.reference_speed_mps")
        tolerance = _finite_number(raw.get("speed_tolerance_mps"), f"{label}.speed_tolerance_mps")
        if reference < 0.0 or tolerance <= 0.0:
            raise RewardSpecError(
                f"{label} requires reference_speed_mps >= 0 and speed_tolerance_mps > 0"
            )
        out.update(reference_speed_mps=reference, speed_tolerance_mps=tolerance)
    elif feature == "cautious_progress_quality":
        profile = raw.get("reference_speed_profile_mps")
        if not isinstance(profile, list) or len(profile) < 3:
            raise RewardSpecError(
                f"{label}.reference_speed_profile_mps must contain at least 3 samples"
            )
        normalized_profile = [
            _finite_number(value, f"{label}.reference_speed_profile_mps[{i}]")
            for i, value in enumerate(profile)
        ]
        if min(normalized_profile) < 0.0:
            raise RewardSpecError(f"{label}.reference_speed_profile_mps must be non-negative")
        speed_tolerance = _finite_number(
            raw.get("speed_tolerance_mps"), f"{label}.speed_tolerance_mps"
        )
        reference_progress = _finite_number(
            raw.get("reference_progress_m"), f"{label}.reference_progress_m"
        )
        progress_tolerance = _finite_number(
            raw.get("progress_tolerance_m"), f"{label}.progress_tolerance_m"
        )
        if speed_tolerance <= 0.0 or reference_progress < 0.0 or progress_tolerance <= 0.0:
            raise RewardSpecError(
                f"{label} requires positive speed/progress tolerances and non-negative reference progress"
            )
        out.update(
            reference_speed_profile_mps=normalized_profile,
            speed_tolerance_mps=speed_tolerance,
            reference_progress_m=reference_progress,
            progress_tolerance_m=progress_tolerance,
        )
    elif feature in {"path_corridor_quality", "heading_corridor_quality"}:
        reference_key = (
            "reference_lateral_m"
            if feature == "path_corridor_quality"
            else "reference_heading_deg"
        )
        tolerance_key = (
            "corridor_half_width_m"
            if feature == "path_corridor_quality"
            else "heading_tolerance_deg"
        )
        reference = raw.get(reference_key)
        if not isinstance(reference, list) or len(reference) < 3:
            raise RewardSpecError(
                f"{label}.{reference_key} must contain at least 3 coarse anchors"
            )
        normalized_reference = [
            _finite_number(value, f"{label}.{reference_key}[{i}]")
            for i, value in enumerate(reference)
        ]
        corridor = _finite_number(
            raw.get(tolerance_key), f"{label}.{tolerance_key}"
        )
        if corridor <= 0.0:
            raise RewardSpecError(f"{label}.{tolerance_key} must be > 0")
        out[reference_key] = normalized_reference
        out[tolerance_key] = corridor
    return out


def validate_reward_spec(raw: Any) -> dict[str, Any]:
    """Validate and normalize a JSON-like reward specification."""
    if not isinstance(raw, dict):
        raise RewardSpecError("reward spec must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RewardSpecError(f"schema_version must be {SCHEMA_VERSION!r}")
    summary = raw.get("scene_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RewardSpecError("scene_summary must be a non-empty string")
    components = raw.get("components")
    if not isinstance(components, list) or not components:
        raise RewardSpecError("components must be a non-empty list")

    normalized_components: list[dict[str, Any]] = []
    names: set[str] = set()
    perception_weight = 0.0
    commitment_count = 0
    for index, component in enumerate(components):
        label = f"components[{index}]"
        if not isinstance(component, dict):
            raise RewardSpecError(f"{label} must be an object")
        name = component.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise RewardSpecError(f"{label}.name must match {_NAME_RE.pattern}")
        if name in names:
            raise RewardSpecError(f"duplicate component name {name!r}")
        names.add(name)
        weight = _finite_number(component.get("weight"), f"{label}.weight")
        if not 0.0 < weight <= 1.0:
            raise RewardSpecError(f"{label}.weight must be in (0, 1]")
        claim = _validate_claim(component.get("claim"), f"{label}.claim")
        trajectory = component.get("trajectory")
        if claim["kind"] == "perceptual":
            if trajectory is not None:
                raise RewardSpecError(
                    f"{label}: perception credit must be mention-only, not trajectory-gated"
                )
            perception_weight += weight
            normalized_trajectory = None
        else:
            if trajectory is None:
                raise RewardSpecError(
                    f"{label}: commitment credit must be conjoined with trajectory execution"
                )
            normalized_trajectory = _validate_trajectory(trajectory, f"{label}.trajectory")
            commitment_count += 1
        normalized_components.append(
            {
                "name": name,
                "weight": weight,
                "claim": claim,
                "trajectory": normalized_trajectory,
            }
        )

    total = sum(c["weight"] for c in normalized_components)
    if abs(total - 1.0) > 1e-6:
        raise RewardSpecError(f"component weights must sum exactly to 1.0, got {total:.8f}")
    if perception_weight > MAX_PERCEPTION_WEIGHT + 1e-9:
        raise RewardSpecError(
            f"perception-only weight must be <= {MAX_PERCEPTION_WEIGHT}, got {perception_weight}"
        )
    if not commitment_count:
        raise RewardSpecError("at least one commitment/trajectory component is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_summary": summary.strip(),
        "components": normalized_components,
    }


def validate_reward_spec_semantics(raw: Any) -> dict[str, Any]:
    """Validate only the LLM-owned semantic shell before GT calibration.

    Numeric weights, trajectory references, tolerances, and curve parameters
    are derived from the sealed GT target immediately afterward. Rejecting an
    otherwise useful semantic proposal because the model emitted ``null`` for
    a tolerance or too few reference anchors wastes repair attempts on values
    the compiler will overwrite. Canonical claim vocabulary and component
    identity remain strict; these are genuinely model-owned semantics.

    The returned shell is intentionally a syntactically valid placeholder so
    existing compiler plumbing can carry it. It must be passed through
    ``calibrate_spec_against_target`` before publication or scoring.
    """

    if not isinstance(raw, dict):
        raise RewardSpecError("reward spec must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RewardSpecError(f"schema_version must be {SCHEMA_VERSION!r}")
    summary = raw.get("scene_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RewardSpecError("scene_summary must be a non-empty string")
    components = raw.get("components")
    if not isinstance(components, list) or not components:
        raise RewardSpecError("components must be a non-empty list")

    normalized_components: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, component in enumerate(components):
        label = f"components[{index}]"
        if not isinstance(component, dict):
            raise RewardSpecError(f"{label} must be an object")
        name = component.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise RewardSpecError(f"{label}.name must match {_NAME_RE.pattern}")
        if name in names:
            raise RewardSpecError(f"duplicate component name {name!r}")
        names.add(name)
        claim = _validate_claim(component.get("claim"), f"{label}.claim")
        try:
            weight = _finite_number(component.get("weight"), f"{label}.weight")
        except RewardSpecError:
            weight = 1.0
        if weight <= 0.0:
            weight = 1.0
        trajectory = None
        if claim["kind"] == "commitment":
            # Placeholder only. The target compiler replaces the entire
            # commitment/trajectory component, including this feature.
            trajectory = {
                "feature": "speed_gain",
                "window_s": [0.0, 6.4],
                "floor": 0.0,
                "full": 1.0,
                "power": 1.0,
            }
        normalized_components.append(
            {
                "name": name,
                "weight": weight,
                "claim": claim,
                "trajectory": trajectory,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_summary": summary.strip(),
        "components": normalized_components,
    }


def reward_spec_digest(spec: dict[str, Any]) -> str:
    normalized = validate_reward_spec(spec)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _claim_matches(rule: dict[str, Any], claims: Any) -> bool:
    claim_list = claims.perceptual if rule["kind"] == "perceptual" else claims.commitments
    wanted = set(rule["any_of"])
    for claim in claim_list:
        if getattr(claim, rule["field"], None) not in wanted:
            continue
        direction = rule.get("direction", "any")
        # Direction is required only for genuinely lateral families. The DSL
        # never attaches it to longitudinal speed-profile predicates.
        if direction != "any" and getattr(claim, "direction", None) != direction:
            continue
        return True
    return False


def _series_window(values: Any, dt_s: float, start: float, end: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    i0 = max(0, int(math.floor(start / dt_s)))
    i1 = min(len(arr), int(math.ceil(end / dt_s)) + 1)
    return arr[i0:i1]


def _trajectory_measure(rule: dict[str, Any], traj: Any) -> float:
    start, end = rule["window_s"]
    feature = rule["feature"]
    speed = _series_window(traj.speed_mps, traj.dt_s, start, end)
    heading = _series_window(traj.heading_deg, traj.dt_s, start, end)
    lateral = _series_window(traj.lateral_offset_m, traj.dt_s, start, end)
    if feature == "path_corridor_quality":
        if len(lateral) < 2:
            return 0.0
        reference = np.asarray(rule["reference_lateral_m"], dtype=np.float64)
        candidate_x = np.linspace(0.0, 1.0, len(lateral))
        reference_x = np.linspace(0.0, 1.0, len(reference))
        expected = np.interp(candidate_x, reference_x, reference)
        p90_error = float(np.percentile(np.abs(lateral - expected), 90))
        return max(0.0, 1.0 - p90_error / rule["corridor_half_width_m"])
    if feature == "heading_corridor_quality":
        if len(heading) < 2:
            return 0.0
        reference = np.asarray(rule["reference_heading_deg"], dtype=np.float64)
        expected = np.interp(
            np.linspace(0.0, 1.0, len(heading)),
            np.linspace(0.0, 1.0, len(reference)),
            np.rad2deg(np.unwrap(np.deg2rad(reference))),
        )
        actual = np.rad2deg(np.unwrap(np.deg2rad(heading)))
        error = (actual - expected + 180.0) % 360.0 - 180.0
        p90_error = float(np.percentile(np.abs(error), 90))
        return max(0.0, 1.0 - p90_error / rule["heading_tolerance_deg"])
    if feature.startswith("speed_") or feature in {
        "stationary_quality",
        "stop_dwell_fraction",
        "late_stationary_quality",
        "cautious_progress_quality",
    }:
        if len(speed) < 2:
            return 0.0
        early_n = max(1, min(len(speed) // 4, int(round(0.5 / traj.dt_s))))
        initial = float(np.mean(speed[:early_n]))
        if feature == "speed_drop":
            return max(0.0, initial - float(np.min(speed[early_n:]))) if len(speed) > early_n else 0.0
        if feature == "speed_gain":
            # Endpoint-directed gain, not initial-to-maximum. A decelerating
            # trajectory reversed in time has the same extrema and used to
            # receive identical acceleration credit. Comparing late and
            # early windows makes the feature causal and reversal-sensitive.
            late_n = max(1, min(len(speed) // 4, int(round(0.5 / traj.dt_s))))
            return max(0.0, float(np.mean(speed[-late_n:])) - initial)
        if feature == "speed_reduction_fraction":
            if initial <= 0.25 or len(speed) <= early_n:
                return 0.0
            return max(0.0, min(1.0, (initial - float(np.min(speed[early_n:]))) / initial))
        if feature == "speed_stability_quality":
            deviation = float(np.percentile(np.abs(speed - rule["reference_speed_mps"]), 90))
            return max(0.0, 1.0 - deviation / rule["speed_tolerance_mps"])
        if feature == "cautious_progress_quality":
            reference = np.asarray(rule["reference_speed_profile_mps"], dtype=np.float64)
            expected = np.interp(
                np.linspace(0.0, 1.0, len(speed)),
                np.linspace(0.0, 1.0, len(reference)),
                reference,
            )
            speed_error = float(np.percentile(np.abs(speed - expected), 90))
            speed_quality = max(0.0, 1.0 - speed_error / rule["speed_tolerance_mps"])
            progress = float(np.sum(speed) * traj.dt_s)
            progress_quality = max(
                0.0,
                1.0
                - abs(progress - rule["reference_progress_m"])
                / rule["progress_tolerance_m"],
            )
            return min(speed_quality, progress_quality)
        reference = rule["reference_speed_mps"]
        if feature == "stop_dwell_fraction":
            return float(np.mean(speed <= reference))
        if feature == "late_stationary_quality":
            late_n = max(1, min(len(speed), int(round(1.0 / traj.dt_s))))
            late_p90 = float(np.percentile(speed[-late_n:], 90))
            return max(0.0, 1.0 - late_p90 / reference)
        return max(0.0, 1.0 - float(np.max(speed)) / reference)
    if feature.startswith("heading_"):
        if len(heading) < 2:
            return 0.0
        signed = float(np.rad2deg(np.unwrap(np.deg2rad(heading))[-1] - np.unwrap(np.deg2rad(heading))[0]))
        # Alpamayo keyframes encode left as positive and right as negative.
        return max(0.0, signed if feature == "heading_left" else -signed)
    if len(lateral) < 2:
        return 0.0
    edge_n = max(1, min(len(lateral) // 4, int(round(0.5 / traj.dt_s))))
    signed = float(np.mean(lateral[-edge_n:]) - np.mean(lateral[:edge_n]))
    return max(0.0, signed if feature == "lateral_left" else -signed)


def evaluate_reward_spec_components(spec: dict[str, Any], claims: Any, traj: Any) -> dict[str, float]:
    """Evaluate named non-negative component contributions."""
    normalized = validate_reward_spec(spec)
    out: dict[str, float] = {}
    for component in normalized["components"]:
        if not _claim_matches(component["claim"], claims):
            out[component["name"]] = 0.0
            continue
        trajectory = component["trajectory"]
        grade = 1.0
        if trajectory is not None:
            measure = _trajectory_measure(trajectory, traj)
            floor, full = trajectory["floor"], trajectory["full"]
            linear = max(0.0, min(1.0, (measure - floor) / (full - floor)))
            grade = linear ** trajectory.get("power", 1.0)
        out[component["name"]] = component["weight"] * grade
    return out


def compile_reward_spec_to_source(spec: dict[str, Any]) -> str:
    """Compile a validated spec to deterministic sandbox-compatible source."""
    normalized = validate_reward_spec(copy.deepcopy(spec))
    literal = repr(normalized)
    digest = reward_spec_digest(normalized)
    return (
        f'"""Deterministically compiled ClipGen reward spec {digest}."""\n'
        f"REWARD_SPEC = {literal}\n\n"
        "def components(claims, traj):\n"
        "    return evaluate_reward_spec_components(REWARD_SPEC, claims, traj)\n\n"
        "def reward(claims, traj):\n"
        "    return sum(components(claims, traj).values())\n"
    )
