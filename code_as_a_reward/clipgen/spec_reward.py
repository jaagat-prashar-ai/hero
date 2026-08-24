# SPDX-License-Identifier: Apache-2.0
"""Spec -> reward-function compiler.

The 2026-08-24 dead-on-GT audit (BUGS.md, scratchpad gatecal/) showed 67%
of failed clips carry functions that score <0.7 on the ground-truth pair
itself, and the causes are almost entirely arithmetic the generator LLM
does in its head: component budgets that cannot reach 0.7 (30%), invented
time windows / over-tight thresholds (44%), and checks keyed to claims the
briefing never contained (26%). This module removes the arithmetic from
the LLM entirely: the model emits a small JSON SPEC (its actual competence
-- scene judgment), and this compiler emits the Python source, making
whole failure classes impossible by construction:

- weights are normalized so component maxima sum to EXACTLY 1.0;
- the primary event alone can reach the 0.7 positive bar (sparse-rollout
  reachability is a compiler guarantee, not a prompt plea);
- trajectory factors are time-directional (reversal-sensitive), floored
  at a fraction of the scene magnitude, graded up to it, and DECAY past
  ~1.5x of it -- fixing the monotone "more braking is always better"
  grading that made passing functions rank anti-correlated with ADE;
- claim gates are family-level idioms (speed_profile / lateral set with
  an excluded direction), never exact keys;
- no time windows exist at all: quantities are computed from fixed early/
  late fractions of whatever series the rollout actually has.

The output source passes sandbox.compile_reward_module unchanged and is
gate-verified exactly like free-form functions -- the empirical gate stays
the arbiter of the spec's scene judgment.
"""

from __future__ import annotations

import json

VALID_FAMILIES = ("decelerate", "accelerate", "lateral", "keep_lane", "maintain")
VALID_QUANTITIES = ("speed_drop", "speed_gain", "lateral_shift", "turn", "speed_steady")
# family -> quantities that can verify it
FAMILY_QUANTITIES = {
    "decelerate": ("speed_drop",),
    "accelerate": ("speed_gain",),
    "lateral": ("lateral_shift", "turn"),
    "keep_lane": ("lateral_shift",),  # verified as staying near center
    # car-following / steady-progress scenes: rollouts claim keep_distance/
    # adapt_speed/proceed (the shard0 spec prototype died on exactly these --
    # 12/12 rollouts claiming keep_distance with no matching family to pick).
    # Verified as speed staying within the scene's band (speed_steady) or
    # small lateral drift (lateral_shift, inverted).
    "maintain": ("speed_steady", "lateral_shift"),
}
VALID_ENTITIES = frozenset({
    "pedestrian", "cyclist", "workers", "emergency_vehicle", "lead_vehicle",
    "stopped_vehicle", "cutin_vehicle", "vehicle_generic", "cross_traffic",
    "oncoming_traffic", "construction_cones", "barricades", "work_zone",
    "signal", "crosswalk", "intersection", "roundabout", "gate",
    "ramp_or_freeway", "curve", "shoulder_or_median", "lane", "speed_hump",
    "speed_limit_sign", "weather_or_surface", "animal",
})
MENTION_SHARE = 0.15  # of each event's weight, when it lists entities


class SpecError(ValueError):
    """Invalid spec; the message is written to be fed back to the LLM."""


def validate_spec(spec: dict) -> list[dict]:
    events = spec.get("events")
    if not isinstance(events, list) or not (1 <= len(events) <= 3):
        raise SpecError("spec.events must be a list of 1-3 events")
    out = []
    for i, ev in enumerate(events):
        fam = ev.get("family")
        if fam not in VALID_FAMILIES:
            raise SpecError(f"events[{i}].family {fam!r} not in {VALID_FAMILIES}")
        q = ev.get("quantity")
        if q not in FAMILY_QUANTITIES[fam]:
            raise SpecError(
                f"events[{i}].quantity {q!r} cannot verify family {fam!r};"
                f" allowed: {FAMILY_QUANTITIES[fam]}"
            )
        mag = ev.get("magnitude")
        if not isinstance(mag, (int, float)) or mag <= 0:
            raise SpecError(f"events[{i}].magnitude must be a positive number (scene units)")
        # clamp to trajectory noise floors -- shard0 v3 lost clips to bands
        # like 0.02 m that no real (or even GT) trajectory sits inside
        mag = max(float(mag), {"speed_drop": 2.0, "speed_gain": 2.0,
                               "lateral_shift": 0.8, "turn": 8.0,
                               "speed_steady": 1.5}[q])
        ents = ev.get("entities") or []
        bad = [e for e in ents if e not in VALID_ENTITIES]
        if bad:
            raise SpecError(f"events[{i}].entities contains non-canonical keys {bad}")
        d = ev.get("direction")
        if d not in (None, "left", "right"):
            raise SpecError(f"events[{i}].direction must be left/right/null")
        xd = ev.get("excluded_direction")
        if xd not in (None, "left", "right"):
            raise SpecError(f"events[{i}].excluded_direction must be left/right/null")
        w = ev.get("weight")
        if not isinstance(w, (int, float)) or w <= 0:
            raise SpecError(f"events[{i}].weight must be > 0")
        name = str(ev.get("name") or f"event{i}")
        name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)[:30]
        out.append({
            "name": name or f"event{i}", "family": fam, "quantity": q,
            "magnitude": float(mag), "entities": list(ents),
            "direction": d, "excluded_direction": xd, "weight": float(w),
        })
    total = sum(ev["weight"] for ev in out)
    for ev in out:
        ev["weight"] /= total  # maxima sum to exactly 1.0, by construction
    # Reachability is a COMPILER GUARANTEE, not a request: the primary
    # event's conjunction alone must clear the 0.7 positive bar (the
    # shard0 v2 prototype lost 16 clips to two-event specs whose primary
    # normalized to 0.6 -- a sparse rollout could never reach 0.7). The
    # primary is therefore floored at 0.75 (secondaries rescaled) and,
    # in compile_spec, carries NO mention split.
    out.sort(key=lambda e: -e["weight"])
    if out[0]["weight"] < 0.75:
        deficit = 0.75 - out[0]["weight"]
        rest = sum(e["weight"] for e in out[1:])
        for e in out[1:]:
            e["weight"] *= (rest - deficit) / rest
        out[0]["weight"] = 0.75
    return out


_QUANTITY_CODE = {
    # time-directional on purpose: early minus later-min (or symmetric),
    # so a time-reversed trajectory does NOT preserve the value.
    "speed_drop": (
        "    sp = list(traj.speed_mps)\n"
        "    n = len(sp)\n"
        "    early = sum(sp[: max(1, n // 4)]) / max(1, n // 4)\n"
        "    later = min(sp[n // 4 :]) if n > 4 else min(sp) if sp else 0.0\n"
        "    x = early - later\n"
    ),
    "speed_gain": (
        "    sp = list(traj.speed_mps)\n"
        "    n = len(sp)\n"
        "    early = sum(sp[: max(1, n // 4)]) / max(1, n // 4)\n"
        "    later = max(sp[n // 4 :]) if n > 4 else max(sp) if sp else 0.0\n"
        "    x = later - early\n"
    ),
    # signed toward the event's direction (+left in the t0 frame); with no
    # direction, magnitude of the final offset.
    "lateral_shift": (
        "    lat = traj.final_lateral_offset_m\n"
        "    x = {SIGN}\n"
    ),
    "turn": (
        "    h = traj.total_heading_change_deg\n"
        "    x = {SIGN_H}\n"
    ),
    # x = worst speed excursion from the early mean; small x = steady.
    "speed_steady": (
        "    sp = list(traj.speed_mps)\n"
        "    n = len(sp)\n"
        "    early = sum(sp[: max(1, n // 4)]) / max(1, n // 4)\n"
        "    x = max(abs(s - early) for s in sp) if sp else 0.0\n"
    ),
}

_GATE_CODE = {
    # car-following rollouts phrase mild slowing as keep_distance/adapt_speed
    # (shard0: 12/12 rollouts claiming keep_distance on a lead-vehicle-slowing
    # scene) -- both are faithful claims for a decelerating trajectory; the
    # identity corruption still separates (it flips decelerate->accelerate,
    # and gutted/no_commitments strip everything).
    "decelerate": (
        "any(c.speed_profile in ('decelerate', 'maintain', 'adapt')"
        " for c in claims.commitments)"
    ),
    "accelerate": "any(c.speed_profile == 'accelerate' for c in claims.commitments)",
    "lateral": (
        "any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter',"
        " 'exit') and c.direction != {XD!r} for c in claims.commitments)"
    ),
    "keep_lane": "any(c.maneuver == 'keep_lane' for c in claims.commitments)",
    "maintain": (
        "any(c.maneuver in ('keep_distance', 'adapt_speed', 'proceed',"
        " 'keep_lane') or c.speed_profile in ('maintain', 'adapt')"
        " for c in claims.commitments)"
    ),
}


def _graded(mag: float, invert: bool = False) -> str:
    """Graded factor, peaked at the scene magnitude:
    0 below 0.4*M, linear up to 1.0 at 0.9*M, plateau to 1.5*M, then decay
    toward 0.4 by 3*M (overshoot is worth less than a matched maneuver,
    never rewarded more). keep_lane inverts: full credit for staying under
    0.4*M of lateral drift, decaying to 0 by 2*M."""
    m = float(mag)
    if invert:
        # full credit anywhere inside the allowed band M; decay to 0 by 3M
        return (
            f"    g = 1.0 if x <= {m:.4f} else max(0.0, 1.0 - (x - {m:.4f}) / {2.0 * m:.4f})\n"
        )
    return (
        f"    if x < {0.4 * m:.4f}:\n"
        f"        g = 0.0\n"
        f"    elif x < {0.9 * m:.4f}:\n"
        f"        g = (x - {0.4 * m:.4f}) / {0.5 * m:.4f}\n"
        f"    elif x <= {1.5 * m:.4f}:\n"
        f"        g = 1.0\n"
        f"    else:\n"
        f"        g = max(0.4, 1.0 - (x - {1.5 * m:.4f}) / {1.5 * m:.4f})\n"
    )


def compile_spec(spec: dict) -> str:
    """Spec dict -> reward/components source (sandbox contract)."""
    events = validate_spec(spec)
    lines = [
        '"""Compiled from a generator spec (spec_reward.py). Events: '
        + ", ".join(f"{e['name']}({e['family']}/{e['quantity']}@{e['magnitude']:g})"
                    for e in events)
        + '"""',
        "def components(claims, traj):",
        "    comps = {}",
    ]
    for idx, ev in enumerate(events):
        w = ev["weight"]
        # primary event (idx 0, largest weight): pure conjunction at full
        # weight so it alone reaches >= 0.75 -- mention credit only on
        # secondary events.
        has_mention = bool(ev["entities"]) and idx > 0
        w_conj = w * (1.0 - MENTION_SHARE) if has_mention else w
        qcode = _QUANTITY_CODE[ev["quantity"]]
        if ev["quantity"] == "lateral_shift":
            if ev["family"] in ("keep_lane", "maintain"):
                qcode = qcode.replace("{SIGN}", "abs(lat)")
            elif ev["direction"] == "left":
                qcode = qcode.replace("{SIGN}", "lat")
            elif ev["direction"] == "right":
                qcode = qcode.replace("{SIGN}", "-lat")
            else:
                qcode = qcode.replace("{SIGN}", "abs(lat)")
        if ev["quantity"] == "turn":
            if ev["direction"] == "left":
                qcode = qcode.replace("{SIGN_H}", "h")
            elif ev["direction"] == "right":
                qcode = qcode.replace("{SIGN_H}", "-h")
            else:
                qcode = qcode.replace("{SIGN_H}", "abs(h)")
        gate = _GATE_CODE[ev["family"]]
        if ev["family"] == "lateral":
            gate = gate.replace("{XD!r}", repr(ev["excluded_direction"]))
        lines.append(qcode.rstrip())
        lines.append(_graded(ev["magnitude"], invert=(ev["family"] in ("keep_lane", "maintain"))).rstrip())
        lines.append(f"    gate = {gate}")
        lines.append(f"    comps[{ev['name'] + '_conj'!r}] = {w_conj:.4f} * g if gate else 0.0")
        if has_mention:
            ent = tuple(ev["entities"])
            lines.append(
                f"    comps[{ev['name'] + '_mention'!r}] = "
                f"{w * MENTION_SHARE:.4f} if any(p.entity in {ent!r} "
                f"for p in claims.perceptual) else 0.0"
            )
    lines += [
        "    return comps",
        "",
        "def reward(claims, traj):",
        "    return min(1.0, max(0.0, sum(components(claims, traj).values())))",
        "",
    ]
    return "\n".join(lines)


def parse_spec_reply(text: str) -> dict:
    """Extract the last JSON object from an LLM reply."""
    start = text.rfind("{")
    # walk back to the OUTERMOST { of the last JSON blob
    depth, i = 0, len(text) - 1
    last_close = text.rfind("}")
    if last_close == -1:
        raise SpecError("no JSON object in reply")
    depth = 0
    for i in range(last_close, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    else:
        raise SpecError("unbalanced JSON braces in reply")
    try:
        return json.loads(text[start : last_close + 1])
    except json.JSONDecodeError as e:
        raise SpecError(f"invalid JSON spec: {e}") from e
