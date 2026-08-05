# SPDX-License-Identifier: Apache-2.0
"""Generator: an LLM writes a scene-specific reward function.

Sequential 3-step chain (VLM-CaR found one-shot generation unreliable;
same expectation here): (1) name the scene's decisive events from the
dossier, (2) define what faithful reasoning and correct behavior look like
for those events, (3) emit one function matching the sandbox contract.

Two backends, selected by the client object passed in (run_prototype's
--backend flag):

- anthropic (claude-opus-5): conventions mirror
  rl_posttrain/rewards/llm_judge.py -- module-level client reuse,
  ANTHROPIC_API_KEY from the environment, and a server-side fallback to
  claude-opus-4-8 so a spurious safety-classifier decline degrades to the
  fallback model instead of failing the clip. The system prompt carries
  cache_control so the 3 calls x N attempts x 5 clips mostly hit the
  prompt cache.
- openai (gpt-4o): OpenAIChat below, raw chat-completions HTTP like
  llm_judge.py's judge path, key from OPENAI_API_KEY or ~/.creds/openai.key.
  2026-08-04 smoke runs use this (cheaper); opus-5 kept for a later
  code-generation-quality comparison.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from code_as_a_reward.clipgen.sandbox import RewardFnError, compile_reward_fn

GENERATOR_MODEL = "claude-opus-5"
FALLBACK_MODEL = "claude-opus-4-8"
OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 8000

# Hard spend ceiling across ALL calls in a run -- checked before every
# request; the run aborts rather than exceed it. Prices per Mtok.
BUDGET_USD = 5.0
_PRICE_ANTHROPIC = {"in": 5.0, "out": 25.0, "cache_read": 0.50, "cache_write": 6.25}
# gpt-4o: cached prompt tokens bill at half the input rate; there is no
# cache-write surcharge, so "cache_write" (used only in the worst-case
# headroom check) is just the input rate.
_PRICE_OPENAI = {"in": 2.50, "out": 10.0, "cache_read": 1.25, "cache_write": 2.50}
_PRICE = _PRICE_ANTHROPIC  # backward-compatible default


class BudgetExceeded(Exception):
    """The run would exceed BUDGET_USD; nothing further is sent."""


class CostTracker:
    """Accumulates real usage from API responses into dollars."""

    def __init__(self, budget_usd: float = BUDGET_USD, prices: dict[str, float] = _PRICE):
        self.budget_usd = budget_usd
        self.prices = prices
        self.spent_usd = 0.0
        self.calls = 0

    def check(self) -> None:
        # Worst-case cost of one more call: 24K input (a 3rd-attempt retry
        # carries the whole transcript; ~2x margin over the observed shape)
        # billed entirely at the cache-WRITE rate, plus a full MAX_TOKENS
        # completion. The API's own max_tokens cap makes the output side a
        # hard bound, so spent + worst_next <= budget is a true ceiling.
        p = self.prices
        worst_next = (24_000 * p["cache_write"] + MAX_TOKENS * p["out"]) / 1e6
        if self.spent_usd + worst_next > self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.2f} of ${self.budget_usd:.2f}; refusing the next call"
            )

    def add(self, usage) -> None:
        """Anthropic usage object (uncached input / cache split reported)."""
        p = self.prices
        self.calls += 1
        self.spent_usd += (
            (usage.input_tokens or 0) * p["in"]
            + (usage.output_tokens or 0) * p["out"]
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * p["cache_read"]
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * p["cache_write"]
        ) / 1e6

    def add_openai(self, usage: dict) -> None:
        """OpenAI usage dict: prompt_tokens INCLUDES cached tokens, which
        bill at the cache_read rate; no cache-write surcharge exists."""
        p = self.prices
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        self.calls += 1
        self.spent_usd += (
            (usage.get("prompt_tokens", 0) - cached) * p["in"]
            + cached * p["cache_read"]
            + usage.get("completion_tokens", 0) * p["out"]
        ) / 1e6

_SYSTEM = """\
You write per-scene reward functions for an autonomous-driving RL pipeline.

A policy model watches a driving clip and produces (a) chain-of-causation
reasoning text and (b) a planned trajectory. Your function scores ONE such
rollout for faithfulness against this specific scene's ground truth: did the
reasoning register the decisive events, commit to appropriate behavior, and
did the trajectory execute it?

Hard rules for the final function:
- Signature exactly `def reward(claims, traj) -> float`, returning a value
  in [0, 1]. Partial credit is expected; reserve scores near 1.0 for
  rollouts that perceive, commit, AND execute correctly.
- Derive every threshold from THIS scene's numbers in the dossier (times,
  distances, speed drops). Never invent generic constants like 0.3 m.
- No imports, no dunder access, no I/O. Available names: `np` (numpy),
  `window(values, dt_s, t0, t1)` (slice a per-waypoint series to a time
  window in seconds), and standard builtins (min, max, abs, any, ...).
- Be robust: guard divisions, tolerate empty claim lists and short
  trajectories. An exception scores the rollout zero, which is worse than
  returning a low score.
- No free credit: the verification gate pairs your function with
  wrong-by-construction inputs (mismatched reasoning/trajectories, reversed
  and no-reaction trajectories) and requires them to score <= 0.3. Keep
  mention-only credit below that ceiling, and gate execution credit on the
  matching perception/commitment actually being present.
"""

_API_REFERENCE = """\
`claims` is a ParsedCoCTrace:
- claims.perceptual: list of PerceptualClaim -- .entity (canonical key, e.g.
  "pedestrian", "stopped_vehicle", "construction_cones"), .state (canonical
  key or None, e.g. "blocking", "crossing"), .text (verbatim span).
- claims.commitments: list of CommitmentClaim -- .maneuver (canonical key,
  e.g. "stop", "yield", "decelerate", "nudge", "lane_change_left",
  "accelerate", "maintain_speed"), .speed_profile ("accelerate" |
  "decelerate" | "maintain" | "adapt" | None), .direction ("left" | "right"
  | None), .text.
- claims.causal: list of CausalClaim -- .effects (list[CommitmentClaim]),
  .cause (list[PerceptualClaim]), .connective.
- claims.unparsed_spans: list[(start, end)] char spans the parser could not
  account for.

`traj` is a TrajectoryFeatures:
- traj.dt_s (float, seconds per waypoint), traj.n_waypoints (int)
- traj.speed_mps, traj.heading_deg, traj.lateral_offset_m: per-waypoint
  list[float] (lateral offset is signed, + = left, in the t=0 heading frame)
- traj.initial_speed_mps, traj.final_speed_mps, traj.min_speed_mps,
  traj.final_lateral_offset_m, traj.total_heading_change_deg (floats)
- traj.stop_event, traj.yield_event (bool, threshold-derived -- prefer the
  raw per-waypoint series over these flags)

Entity keys the parser can emit for matching against dossier classes:
pedestrian<->pedestrian, automobile/stopped_vehicle<->automobile,
heavy_truck<->heavy truck, bicycle/cyclist<->bicycle, motorcycle<->motorcycle.
"""

_STEP1 = """\
Here is the ground-truth dossier for one driving clip:

{dossier}

Step 1: identify the decisive events of this scene -- which obstacles
actually constrain the ego vehicle, when, and what the expert driver did
about them. List 1-3 decisive events with their timing and geometry, and
state which tracks are ignorable background. Be quantitative.
"""

_STEP2 = """\
Step 2: for each decisive event, define what a FAITHFUL rollout looks like:
(a) which perceptual claims should be present (entity/state keys),
(b) which commitments should be present (maneuver keys),
(c) what the trajectory must quantitatively do, with thresholds derived
from this scene's numbers (e.g. fractions of the expert's speed drop inside
the event's time window). Also define what an UNFAITHFUL rollout looks like
(mentions without execution, execution without mention, wrong direction).
"""

_STEP3 = """\
Step 3: emit the reward function.

{api_reference}

{gt_claims}

Write ONE python code block containing only `def reward(claims, traj):`
(plus optional module-level helper constants). Score composition guidance:
~0.3 perception, ~0.3 commitment, ~0.4 execution, adjusted to what this
scene demands. Include a short docstring naming the decisive events and the
scene-derived thresholds. Remember the hard rules from the system prompt.
"""

_RETRY = """\
The function you wrote failed the empirical verification gate:

{feedback}

Diagnose why (too lenient toward mismatched trajectories/claims, or too
strict toward the ground truth), then emit a corrected function in one
python code block. Same contract and hard rules.
"""


@dataclasses.dataclass
class GenerationResult:
    source: str
    transcript: list[dict]  # the full message history, for eyeball review
    model: str


class GenerationRefused(Exception):
    """Safety classifiers declined and the fallback chain also refused."""


_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_OPENAI_KEY_PATH = Path.home() / ".creds" / "openai.key"


class OpenAIChat:
    """Minimal chat-completions client, stdlib-only so the lilypad generic
    workload needs no extra pip deps. Key resolution mirrors
    rl_posttrain/rewards/llm_judge.py: OPENAI_API_KEY wins, else
    ~/.creds/openai.key."""

    def __init__(self, model: str = OPENAI_MODEL, key_path: Path = _DEFAULT_OPENAI_KEY_PATH):
        self.model = model
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key and key_path.exists():
            key = key_path.read_text().strip()
        if not key:
            raise RuntimeError(
                f"No OpenAI credential: OPENAI_API_KEY unset and {key_path} missing"
            )
        self._key = key

    def complete(self, system: str, messages: list[dict]) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system}] + messages,
            }
        ).encode()
        last_err: Exception | None = None
        for attempt in range(4):
            req = urllib.request.Request(
                _OPENAI_CHAT_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    last_err = RuntimeError(f"HTTP {e.code}: {detail}")
                    continue
                raise RuntimeError(f"OpenAI HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    last_err = e
                    continue
                raise
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")


def render_gt_claims(trace) -> str:
    """Render the parsed ground-truth CoC as the exact claim objects the
    positive gate case passes to the function. Without this the generator
    predicates claim credit on dossier vocabulary (track IDs, free-form
    labels like "trailer") the parser never emits, so ground truth earns
    zero perception/commitment credit (pduuqq smoke, 4/5 clips)."""
    lines = [
        f'The ground-truth CoC text:\n  "{trace.raw_text}"',
        "parses to EXACTLY these claim objects -- this is the `claims` value",
        "your function receives for the ground-truth (positive) gate case:",
        "- perceptual:",
    ]
    for p in trace.perceptual:
        lines.append(
            f"    PerceptualClaim(entity={p.entity!r}, state={p.state!r}, text={p.text!r})"
        )
    if not trace.perceptual:
        lines.append("    (none)")
    lines.append("- commitments:")
    for c in trace.commitments:
        lines.append(
            f"    CommitmentClaim(maneuver={c.maneuver!r}, speed_profile={c.speed_profile!r},"
            f" direction={c.direction!r}, text={c.text!r})"
        )
    if not trace.commitments:
        lines.append("    (none)")
    lines.append("- causal:")
    for cl in trace.causal:
        lines.append(
            f"    CausalClaim(connective={cl.connective!r},"
            f" effects={[e.maneuver for e in cl.effects]!r},"
            f" cause_entities={[p.entity for p in cl.cause]!r})"
        )
    if not trace.causal:
        lines.append("    (none)")
    lines.append(
        "\nYour claim predicates must fire on these exact objects. Match only the\n"
        'canonical .entity/.maneuver/.state/.direction keys. Dossier labels and\n'
        'track IDs (e.g. "Track 32") NEVER appear in claims or their .text -- a\n'
        "predicate testing for them scores zero even on the ground truth. Other\n"
        "faithful rollouts may phrase things differently, so key off canonical\n"
        "fields, never exact .text strings."
    )
    return "\n".join(lines)


def extract_code(text: str) -> str:
    """Last ```python fenced block in the reply (the chain may show drafts)."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        raise RewardFnError("no fenced python code block in model reply")
    return blocks[-1].strip() + "\n"


def _call(client, messages: list[dict], tracker: CostTracker | None = None) -> "object":
    if tracker is not None:
        tracker.check()
    if isinstance(client, OpenAIChat):
        response = client.complete(_SYSTEM, messages)
        if tracker is not None:
            tracker.add_openai(response.get("usage", {}))
        choice = response["choices"][0]
        if choice["message"].get("refusal") or choice.get("finish_reason") == "content_filter":
            raise GenerationRefused("generator request refused by safety classifiers")
        return response
    response = client.beta.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": FALLBACK_MODEL}],
        system=[
            {"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}},
        ],
        messages=messages,
    )
    if tracker is not None:
        tracker.add(response.usage)
    if response.stop_reason == "refusal":
        raise GenerationRefused("generator request refused by safety classifiers")
    return response


def _text(response) -> str:
    if isinstance(response, dict):  # OpenAI chat-completions shape
        return response["choices"][0]["message"].get("content") or ""
    return "".join(b.text for b in response.content if b.type == "text")


def _model(response) -> str:
    return response["model"] if isinstance(response, dict) else response.model


def generate_reward_fn(
    client,
    dossier: str,
    gt_claims=None,  # ParsedCoCTrace of the GT CoC; rendered into step 3
    feedback: str | None = None,
    prior_transcript: list[dict] | None = None,
    tracker: CostTracker | None = None,
) -> GenerationResult:
    """One generation attempt. First attempt runs the 3-step chain; retry
    attempts (feedback set) continue the prior transcript with gate results.

    Returns a GenerationResult whose source has passed compile_reward_fn
    (contract/AST check) -- the empirical gate is the caller's job.
    """
    if feedback is not None:
        if not prior_transcript:
            raise ValueError("feedback retry requires the prior transcript")
        messages = list(prior_transcript)
        messages.append({"role": "user", "content": _RETRY.format(feedback=feedback)})
        response = _call(client, messages, tracker)
        messages.append({"role": "assistant", "content": _text(response)})
    else:
        messages = [{"role": "user", "content": _STEP1.format(dossier=dossier)}]
        response = _call(client, messages, tracker)
        messages.append({"role": "assistant", "content": _text(response)})
        messages.append({"role": "user", "content": _STEP2})
        response = _call(client, messages, tracker)
        messages.append({"role": "assistant", "content": _text(response)})
        messages.append(
            {
                "role": "user",
                "content": _STEP3.format(
                    api_reference=_API_REFERENCE,
                    gt_claims="" if gt_claims is None else render_gt_claims(gt_claims),
                ),
            }
        )
        response = _call(client, messages, tracker)
        messages.append({"role": "assistant", "content": _text(response)})

    source = extract_code(_text(response))
    compile_reward_fn(source)  # raises RewardFnError on contract violations
    return GenerationResult(source=source, transcript=messages, model=_model(response))
