# SPDX-License-Identifier: Apache-2.0
"""
llm_judge.py -- single-trace LLM-judge scorer used as the GRPO reasoning
reward: given ONE chain-of-causation (CoC) trace and the trajectory the SAME
rollout actually produced, ask Claude Fable 5 how consistent the stated
reasoning is with that trajectory (0-10).

Relationship to pref_pairs/judge_reasoning_pairs.py: that module is the
validated pairwise (blind A/B) judge that produced
pref_pairs/results/judged_pairs/judged_pairs.jsonl -- 717 pairs, 84.6%
agreement with construction labels. This module is its single-trace sibling
for online RL: GRPO scores each rollout independently, so there is no second
trace to compare against, and the trajectory shown to the judge is the
rollout's OWN decoded prediction rather than the dataset's ground-truth
action. Everything that could be shared IS shared (imported, not copied):

  - format_waypoint_table (via pref_pairs.judge_reasoning_pairs) so the
    x/y/heading table the judge reads here is derived by exactly the same
    trajectory_features.extract_features path as in the judged-pairs dataset
    -- if the two ever disagreed, GRPO's reward scale would silently drift
    from the offline calibration numbers (chosen median 7, rejected median 1).
  - load_api_key / _extract_json_object (via pref_pairs.perturbation_generator)
    for the ~/.creds/anthropic.key bridging and fence-stripping conventions.

Where this runs: inside the recipe's Python 3.12 uv venv on the Lilypad GPU
worker, called from cosmos-rl's reward path (multi-threaded -- the stock
LingoJudgeGrader carries a forward lock for its GPU model; we need no lock
because each call is a stateless HTTPS request, and thread-concurrency is
exactly what hides the API latency behind other rollouts' scoring).

Failure policy: transient API errors are retried with exponential backoff;
refusals / unparseable responses get fresh-call retries (same convention as
the pairwise judge). After all retries are exhausted we RAISE rather than
return a placeholder score -- a visible crash is recoverable (the workload
requeues / the run is investigated), whereas silently feeding a made-up
reward into GRPO corrupts training in a way nobody would notice until the
reward curves look wrong.

Per the project's no-fake-model-tests preference, the API-calling path
(judge_trace) has NO mocked unit test; only the pure helpers
(_build_user_message, _parse_single_judgment, normalize_score) are
unit-tested. Real verification is the canary cluster run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Repo root = parents[2] of this file (rl_posttrain/rewards/llm_judge.py).
# cosmos-rl may import/execute reward code in worker processes whose
# sys.path was not prepared by our entry script, so make the pref_pairs
# imports below self-sufficient rather than relying on the caller.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pref_pairs.judge_reasoning_pairs import format_waypoint_table  # noqa: E402
from pref_pairs.perturbation_generator import _extract_json_object, load_api_key  # noqa: E402

# All trajectories in this project are fixed 10 Hz sequences (TIME_STEP_S
# convention shared by wds_dataset.py / load_physical_aiavdataset.py); the
# decoded RL rollout trajectories follow the same convention.
TRAJECTORY_HZ = 10.0

# Single-trace variant of judge_reasoning_pairs.SYSTEM_PROMPT. The waypoint
# semantics paragraphs are kept verbatim from the pairwise prompt (they
# define the exact table format format_waypoint_table renders); the pairwise
# A/B instructions are replaced by an absolute 0-10 rubric, and the
# corruption-type taxonomy is dropped -- online rollouts are free-form model
# text, not taxonomy-perturbed corpus entries, so a forced diagnosis label
# would be noise. Do not paraphrase: _parse_single_judgment validates the
# exact output contract below.
SYSTEM_PROMPT = """You are auditing reasoning-action faithfulness for an autonomous-driving policy (Alpamayo). The policy first writes a short causal reasoning trace about the driving scene, then produces a trajectory. You do not have access to camera images or raw perception -- only the text of the reasoning trace and the raw trajectory the policy produced.

The action is given as the full 64-step future trajectory, as (x, y, heading_deg) waypoints in the ego frame: x = forward distance traveled so far (meters, increasing), y = lateral offset from the ego's start position (meters, positive = left), heading_deg = cumulative heading change from the start (degrees, positive = left turn). Waypoints are evenly spaced in time (the same fixed interval for every scene), so bunching (small consecutive x deltas) means the vehicle was decelerating relative to a constant-speed baseline, and spreading means it was accelerating; the heading column directly shows turn direction and how sharply it developed over the 64 steps.

Read the shape of the whole trajectory, not just the start/end -- e.g. a trace claiming a hazard "directly ahead" should correspond to braking (x deltas shrinking) and/or a lane change (y moving markedly away from 0) starting early in the sequence, not a late, sudden swerve, and vice versa for a hazard that only becomes relevant partway through the scene.

Your job: given ONE reasoning trace offered as the explanation for this trajectory, score how consistent the trace's stated claims are with -- i.e. how well they would actually justify -- that trajectory.

What "consistent with the action" means concretely:
- If the trace claims a nearby agent/lane/hazard exists and asserts a specific response to it (e.g. "keep distance," "yield," "change lanes"), check whether the trajectory's shape (braking/accelerating via waypoint spacing, turning direction/magnitude via the heading column, lane-change extent via the y column) is what that claim would produce. A trace whose claimed justification, if true, would produce a DIFFERENTLY-SHAPED trajectory than the one shown is inconsistent -- even if the sentence reads fluently.
- A trace that is internally self-contradictory, or asserts a response it visibly does not take (says "slow down" while the waypoints spread), scores low.
- Do not reward length or detail: a short trace whose single claim matches the trajectory outranks a long trace with one contradicted claim.
- An empty, truncated, or non-reasoning trace (boilerplate, repetition, no causal content) scores 0-2.

Output:
- action_consistency_score: 0-10, where 10 = the trace's claims, if true, straightforwardly produce this exact trajectory shape; 0 = the claims directly contradict or would produce a materially different trajectory.
- one_line_rationale: <= 25 words, citing the specific phrase driving your score.

Respond with ONLY a JSON object -- no preamble, no markdown fences, no commentary:

{
  "action_consistency_score": <0-10 int>,
  "one_line_rationale": "<string>"
}"""


class JudgeRewardError(Exception):
    """Raised when a valid judgment can't be obtained for a rollout even
    after all retries (persistent API failure, refusal, or invalid JSON).
    Deliberately fatal -- see module docstring's failure policy."""


def _build_user_message(trace: str, waypoint_table: str) -> str:
    """User turn for one rollout. Mirrors judge_reasoning_pairs.build_user_message
    minus the A/B framing; the trajectory here is the rollout's own predicted
    path (faithfulness of reasoning to own action), not a ground-truth action."""
    return (
        "Trajectory produced by the policy: 64 waypoints\n"
        "(x meters forward, y meters lateral [+left], heading degrees cumulative),\n"
        "evenly spaced in time:\n"
        f"{waypoint_table}\n\n"
        f'Reasoning trace: "{trace}"'
    )


def _parse_single_judgment(text: str) -> dict[str, Any]:
    """Parses + validates one single-trace judgment. Pure (no network) so it's
    directly unit-testable, unlike judge_trace itself -- same split as the
    pairwise judge's _parse_judgment_response."""
    parsed = json.loads(_extract_json_object(text))
    if "action_consistency_score" not in parsed or "one_line_rationale" not in parsed:
        raise JudgeRewardError(f"missing required keys in judgment: {sorted(parsed)}")
    score = parsed["action_consistency_score"]
    # bool is an int subclass; a judge answering `true` must not score as 1.
    if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 10):
        raise JudgeRewardError(f"action_consistency_score {score!r} not an int in [0, 10]")
    return parsed


def normalize_score(score: int) -> float:
    """Maps the judge's 0-10 integer onto the recipe's reasoning-score scale.

    The vendored aggregated_reward_with_reasoning expects the grader to
    produce a raw quality in [0, 1] and then uses `raw - 1.0` (i.e. a value
    in [-1, 0], where 0 is best) with reasoning_threshold = -0.4. Mapping
    score/10 - 1.0 preserves that contract exactly, so the recipe's
    threshold -0.4 corresponds to judge score 6 -- a sensible acceptance bar
    given the judged-pairs calibration (chosen traces median 7, corrupted
    traces median 1)."""
    return score / 10.0 - 1.0


def waypoint_table_from_xyz(waypoints_xyz: Any, hz: float = TRAJECTORY_HZ) -> str:
    """Renders a (T, 3) xyz array (list or numpy) via the SAME
    format_waypoint_table used to build the judged-pairs dataset, by wrapping
    it in the minimal action-dict shape that function expects."""
    waypoints = waypoints_xyz.tolist() if hasattr(waypoints_xyz, "tolist") else list(waypoints_xyz)
    return format_waypoint_table({"waypoints": waypoints, "hz": hz, "rollout_id": 0})


# One client per process, lazily built: cosmos-rl scores rewards from
# multiple threads, and anthropic.Anthropic is documented thread-safe, so a
# shared client (with its connection pool) is both correct and cheaper than
# per-call construction.
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                import anthropic

                # On the cluster the key arrives as ANTHROPIC_API_KEY via the
                # workload env; load_api_key only falls back to
                # ~/.creds/anthropic.key for local/workstation use.
                load_api_key()
                _CLIENT = anthropic.Anthropic()
    return _CLIENT


# Transient-API retry schedule (seconds). Deliberately generous: a GRPO step
# blocks on its rollouts' rewards, so waiting out a rate-limit spike is far
# cheaper than crashing an 8-GPU node. Non-transient failures (auth, bad
# request) raise immediately -- retrying those only delays the inevitable.
_BACKOFF_S = (2, 5, 15, 30, 60)


def judge_trace(
    trace: str,
    waypoints_xyz: Any,
    hz: float = TRAJECTORY_HZ,
    model: str = "claude-fable-5",
) -> int:
    """Scores one CoC trace against the trajectory it produced. Returns the
    raw 0-10 integer (callers wanting the recipe scale apply normalize_score).

    Two nested retry layers, mirroring the pairwise judge's conventions:
      - transport layer: transient API errors (rate limit / overload /
        timeout) retry on _BACKOFF_S; the SDK's own built-in retries sit
        below this as a first line of defense.
      - content layer: a refusal or invalid/unparseable judgment gets ONE
        fresh call (not a repair turn), then raises JudgeRewardError.
    """
    import anthropic

    client = _get_client()
    waypoint_table = waypoint_table_from_xyz(waypoints_xyz, hz)
    user_message = _build_user_message(trace, waypoint_table)

    content_retries_left = 1
    attempt = 0
    while True:
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=512,
                # Same server-side-fallback pattern as the pairwise judge and
                # perturbation_generator: a Fable 5 policy refusal retries on
                # Opus 4.8 within the same API call.
                betas=["server-side-fallback-2026-06-01"],
                fallbacks=[{"model": "claude-opus-4-8"}],
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_message}],
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt >= len(_BACKOFF_S):
                raise JudgeRewardError(f"transient API failure persisted after {attempt} backoff retries: {e}") from e
            delay = _BACKOFF_S[attempt]
            attempt += 1
            logger.warning("judge_trace: transient API error (%s), retry %d in %ds", type(e).__name__, attempt, delay)
            time.sleep(delay)
            continue

        if response.stop_reason == "refusal":
            if content_retries_left > 0:
                content_retries_left -= 1
                logger.warning("judge_trace: refused even with fallback, retrying once")
                continue
            raise JudgeRewardError("judge refused after retry")

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return _parse_single_judgment(text)["action_consistency_score"]
        except (json.JSONDecodeError, JudgeRewardError) as e:
            if content_retries_left > 0:
                content_retries_left -= 1
                logger.warning("judge_trace: invalid judgment (%s), retrying once", e)
                continue
            raise JudgeRewardError(f"invalid judgment after retry: {e}") from e
