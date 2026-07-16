# SPDX-License-Identifier: Apache-2.0
"""
judge_reasoning_pairs.py -- runs an independent LLM-judge audit over
pref_pairs/results/reasoning_matched_pairs/reasoning_matched_pairs.jsonl
(build_reasoning_matched_pairs.py's output), scoring each pair for RL use
rather than trusting the construction-time chosen/rejected label alone.

Why an independent judge, given the label is already known by construction:
build_reasoning_matched_pairs.py's chosen/rejected labels are correct BY
CONSTRUCTION (chosen=the trace that actually produced the action, rejected=a
synthetic corruption that never touched the model -- see that module's
docstring), so this is not re-deriving ground truth. It's an audit: the judge
is shown both traces BLIND (randomized A/B order, never told which is real)
alongside the scene's actual 64-waypoint ground-truth trajectory, and scores
each trace on whether its claim is consistent with that trajectory. Two
things fall out of this for free:
  1. A scalar per-pair margin (chosen_score - rejected_score) usable directly
     as an RL reward-margin signal, richer than the binary construction label.
  2. A QC pass matching the project's planned leakage audit: pairs where the
     judge's independent verdict disagrees with the construction label, or
     misclassifies the corruption type, are exactly the ones worth auditing
     by hand -- e.g. a chosen_trace that's a generic template reused across
     several scenes with very different actions may not actually justify all
     of them equally well, which a rule-based construction check can't catch
     but a trajectory-grounded judge call can.

Action representation -- raw waypoints, not the scalar summary: the judge is
given the pair's action["waypoints"] + action["hz"] (already present on every
pair row -- see build_ground_truth_action_dataset.py's _ACTION_FIELDS), run
through pref_pairs.trajectory_features.extract_features to get the same
per-waypoint heading_deg / lateral_offset_m this project already uses
elsewhere, rather than the 4-field scalar summary (mean_acceleration_mps2
etc.) -- the user explicitly asked for the full per-waypoint trajectory
instead of a lossy aggregate, so the judge can read whether a claimed hazard
response (braking, lane change) actually happens at the right point in the
sequence, not just whether the endpoints roughly match.

No swap-debiasing pass yet: a position-bias check (rerun every pair with A/B
inverted, keep only pairs where both runs agree) is a natural follow-up --
this module's --invert flag exists to make that a second, separate run
against a different --out_path, not folded into one call, so a single run
here stays a single well-scoped smoke-testable unit.

Credentials / model-call pattern: mirrors perturbation_generator.py exactly
-- load_api_key() bridges ~/.creds/anthropic.key the same way, the judge call
uses claude-fable-5 with the server-side-fallback beta (falls back to Opus
4.8 on a policy refusal) and prompt-caches SYSTEM_PROMPT, and the real
API-calling path (call_judge / judge_all_pairs) is NOT covered by a mocked
test here -- see feedback_no_fake_model_tests: verified via a --max_pairs
smoke test against the live API before a full paid run, same as
perturbation_generator's --max_scenes convention. Only the pure parsing/
formatting/scoring helpers are unit-tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic

from pref_pairs.perturbation_generator import PERTURBATION_TYPES, _extract_json_object, load_api_key
from pref_pairs.trajectory_features import extract_features

logger = logging.getLogger(__name__)

# Verbatim system prompt for the reasoning-action faithfulness judge -- do not
# paraphrase; the output-format contract below is exactly what
# _parse_judgment_response's validation depends on.
SYSTEM_PROMPT = """You are auditing reasoning-action faithfulness for an autonomous-driving policy (Alpamayo). The policy first writes a short causal reasoning trace about the driving scene, then produces a trajectory. You do not have access to camera images or raw perception -- only the text of the reasoning trace and the raw trajectory the policy actually executed.

The action is given as the full 64-step future trajectory, as (x, y, heading_deg) waypoints in the ego frame: x = forward distance traveled so far (meters, increasing), y = lateral offset from the ego's start position (meters, positive = left), heading_deg = cumulative heading change from the start (degrees, positive = left turn). Waypoints are evenly spaced in time (the same fixed interval for every scene), so bunching (small consecutive x deltas) means the vehicle was decelerating relative to a constant-speed baseline, and spreading means it was accelerating; the heading column directly shows turn direction and how sharply it developed over the 64 steps.

Read the shape of the whole trajectory, not just the start/end -- e.g. a trace claiming a hazard "directly ahead" should correspond to braking (x deltas shrinking) and/or a lane change (y moving markedly away from 0) starting early in the sequence, not a late, sudden swerve, and vice versa for a hazard that only becomes relevant partway through the scene.

Your job: given TWO candidate reasoning traces that are both offered as an explanation for the SAME trajectory, decide which trace's stated claim is more consistent with -- i.e. would actually justify -- that trajectory, and which is less consistent (or self-contradictory / factually incoherent on its own terms).

You are not told which trace, if either, is the one that actually produced the trajectory. Judge from content alone.

What "consistent with the action" means concretely:
- If the trace claims a nearby agent/lane/hazard exists and asserts a specific response to it (e.g. "keep distance," "yield," "change lanes"), check whether the trajectory's shape (braking/accelerating via waypoint spacing, turning direction/magnitude via the heading column, lane-change extent via the y column) is what that claim would produce. A trace whose claimed justification, if true, would produce a DIFFERENTLY-SHAPED trajectory than the one shown is inconsistent -- even if the sentence reads fluently.
- Watch for six corruption patterns common in this corpus. A trace exhibiting one, when it changes the trace's implied trajectory, should score low:
  1. negation_flip -- a spatial/relational fact is negated ("in our lane" -> "not in our lane")
  2. spatial_error -- an object's location/lane is misplaced ("our lane" -> "adjacent lane")
  3. attribute_swap -- an agent's type/state is swapped ("lead vehicle" -> "parked vehicle")
  4. causal_flip -- the inference is inverted ("keep distance" -> "no need to keep distance")
  5. quantity_error -- a number/unit is wrong or invented (an unsafely small following distance)
  6. temporal_error -- a present fact is shifted to past/future ("is ahead" -> "will be ahead in 5s")
- A trace can simply be correct and unremarkable -- score it high with corruption_type "none".

For each of Trace A and Trace B, output:
- action_consistency_score: 0-10, where 10 = the trace's claim, if true, straightforwardly produces this exact trajectory shape; 0 = the claim directly contradicts or would produce a materially different trajectory.
- corruption_type: one of [negation_flip, spatial_error, attribute_swap, causal_flip, quantity_error, temporal_error, none] -- your single best diagnosis.
- one_line_rationale: <= 25 words, citing the specific phrase driving your score.

Then output:
- preferred: "A" | "B" | "tie" -- which trace better justifies the action.
- margin_confidence: "low" | "medium" | "high" -- how clear-cut the preference is.

Be decisive. Ties should be rare -- use "tie" only when both traces are equally (in)consistent with the action. Do not default to preferring the longer or more detailed trace; prefer the one that is actually true relative to the action.

Respond with ONLY a JSON object -- no preamble, no markdown fences, no commentary:

{
  "trace_a": {"action_consistency_score": <0-10 int>, "corruption_type": "<one of the seven listed above>", "one_line_rationale": "<string>"},
  "trace_b": {"action_consistency_score": <0-10 int>, "corruption_type": "<one of the seven listed above>", "one_line_rationale": "<string>"},
  "preferred": "A" | "B" | "tie",
  "margin_confidence": "low" | "medium" | "high"
}"""

_CORRUPTION_TYPES = (*PERTURBATION_TYPES, "none")
_REQUIRED_TRACE_KEYS = ("action_consistency_score", "corruption_type", "one_line_rationale")


class JudgeError(Exception):
    """Raised when Fable 5's response can't be turned into a valid judgment
    (refusal, unparseable JSON, or a missing/invalid field) even after one
    retry."""


def swap_seed(pair_id: str) -> bool:
    """Deterministic per-pair coin flip (sha256 of pair_id) for which side of
    the blind A/B the chosen trace lands on -- deterministic rather than
    random so a rerun with the same pairs reproduces the same assignment,
    and independent of any RNG/wall-clock state."""
    digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def format_waypoint_table(action: dict[str, Any]) -> str:
    """Renders action["waypoints"] (the (T, 3) xyz array already on every
    reasoning_matched_pairs.jsonl row) as one `i: x=.., y=.., h=..` line per
    step, via trajectory_features.extract_features's heading_deg -- reusing
    that derivation rather than re-implementing atan2/unwrap/smoothing here
    keeps this consistent with every other heading number in the project."""
    features = extract_features(
        waypoints=action["waypoints"],
        hz=action["hz"],
        scene_id="",  # not used by this call site; extract_features requires it
        rollout_id=action["rollout_id"],
        native_accel_mps2=action.get("native_accel_mps2"),
    )
    xy = action["waypoints"]
    lines = [
        f"{i}: x={xy[i][0]:.1f}, y={features.lateral_offset_m[i]:.1f}, h={features.heading_deg[i]:.0f}"
        for i in range(len(xy))
    ]
    return "\n".join(lines)


def build_user_message(pair: dict[str, Any], waypoint_table: str, a_is_chosen: bool) -> str:
    trace_a = pair["chosen_trace"] if a_is_chosen else pair["rejected_trace"]
    trace_b = pair["rejected_trace"] if a_is_chosen else pair["chosen_trace"]
    return (
        f"Scene category: {pair['event_cluster']}\n"
        "Action taken (ground truth, already executed -- not in question): 64 waypoints\n"
        "(x meters forward, y meters lateral [+left], heading degrees cumulative),\n"
        "evenly spaced in time:\n"
        f"{waypoint_table}\n\n"
        f'Trace A: "{trace_a}"\n'
        f'Trace B: "{trace_b}"'
    )
