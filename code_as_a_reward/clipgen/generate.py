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

import base64
import dataclasses
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from code_as_a_reward.clipgen.sandbox import RewardFnError, compile_reward_module

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
reasoning text and (b) a planned trajectory. You're shown three things about
the ground truth for this scene: the camera frame at t0 with the expert's
actual future trajectory drawn on it, the expert's own reasoning (what it
says it noticed and committed to), and the expert's actual measured actions
(the dossier's trajectory numbers). From those three things, infer a
practical rubric for "was this rollout an ACCEPTABLE, faithful response to
this scene" -- not "does it exactly reproduce GT." A real rollout from the
policy will differ from GT in its exact phrasing, its exact numbers, and
even its exact timing; build reasonably flexible bounds and tolerances into
your checks from the start, rather than GT's literal values as hard
requirements.

How you score is entirely up to you. Reason about what faithfulness means
for THIS scene and design whatever logic, heuristics, and intermediate
quantities capture it best. Perception, commitment, and execution checks
are natural ingredients, but their structure, weighting, and combination
are yours to invent per scene -- no fixed rubric is prescribed. Scores are
used to RANK candidate rollouts, so correct ordering matters more than
calibrated absolute values.

Your function will be empirically verified, not just trusted: after GRPO
samples a group of rollouts, the highest-scoring one (the argmax) gets
checked against corrupted variants of ITSELF -- the same reasoning with a
time-reversed or no-reaction trajectory, the same trajectory with the
reasoning gutted, with its commitments removed, or with a commitment
flipped to a different maneuver. A good function scores the intact argmax
meaningfully higher (roughly >= 0.7) than every one of its own corruptions
(each dropping by roughly >= 0.4). You don't need to solve this as an exact
formula up front -- keep one design principle in mind, and let the
empirical feedback on a failed attempt guide the rest: only checks that
require BOTH a commitment claim being present AND the trajectory actually
executing it survive every corruption. Checks based on the claim alone, or
the trajectory alone, tend to keep their credit under some corruption, so
they should carry less weight than checks requiring both. Checks a
corruption preserves -- aggregate statistics like min/max speed,
order-insensitive quantities, "some deviation happened somewhere" -- hand
the corruption the same credit as the real thing; anchor checks in the
maneuver's temporal shape instead (what happens, WHEN, in what order, in
which direction). Timing is the sharpest reversal discriminator: a
time-reversed trajectory keeps the same speeds but events happen at
mirrored times.

This conjunction principle only works for COMMITMENT claims
(claims.commitments) -- claims about an intended ACTION. A trajectory is a
record of physical motion, so it can directly confirm or refute an action
claim: if the reasoning says "I will stop," the trajectory either shows a
stop or it doesn't. PERCEPTUAL claims (claims.perceptual) are claims about
what was noticed, not done -- there is nothing in a trajectory that can
verify what the model perceived, no matter which trajectory quantity you
pick. Gating a perceptual claim behind a trajectory condition (e.g. "award
credit if a pedestrian is mentioned AND the car's speed dropped below X")
does not add real verification -- the two things aren't actually related,
so that condition can pass or fail for reasons that have nothing to do
with whether the perception was accurate, and on scenes where the
trajectory doesn't change under a given corruption, a check like this can
end up doing nothing at all while looking like a safeguard. Give
perceptual claims mention-only credit (present in claims.perceptual or
not) and keep that credit small; save real conjunction credit -- and the
larger component weights -- for commitments, the only claim type a
trajectory can actually verify.

The same logic cuts the other way: a component built ONLY from a
trajectory quantity, with NO claims.commitments check at all, is equally
unverified -- two of the five corruptions (no_commitments, gutted_claims)
strip the reasoning entirely and leave the trajectory untouched, so a
trajectory-only component keeps its FULL credit under both, no matter how
sharp its threshold is (this is not a maybe -- it is certain, by
construction of those two corruptions). If you can name a
claims.commitments maneuver key the trajectory is executing, require it in
the same check: `any(c.maneuver == '<key>' for c in claims.commitments)
and <trajectory condition>`. If the behavior doesn't correspond to any
maneuver claim the model would plausibly state (e.g. "kept the lane" as
opposed to "yielded" or "decelerated"), it isn't something a trajectory
alone can prove was a FAITHFUL response to reasoning -- give it near-zero
weight, the same way you would an unpaired perceptual claim, rather than a
large "execution" score.

If your first attempt doesn't clear the empirical check, you'll see exactly
which case failed, by how much, and the real measured numbers behind it --
use that concrete feedback to adjust your checks, rather than trying to get
every threshold and weight perfect before ever seeing real data.

Hard rules for the final function:
- Signature exactly `def reward(claims, traj) -> float`, returning a value
  in [0, 1].
- ALSO define `def components(claims, traj) -> dict` returning the named
  component contributions (e.g. {"saw_pedestrian": 0.1, "stop_executed":
  0.5, ...}); `reward` must return exactly min(1.0, max(0.0,
  sum(components(claims, traj).values()))). The gate calls components() on
  every case, rejects the function if the pre-clamp sum exceeds 1.0 or
  disagrees with reward(), and shows you the per-case breakdown when a case
  fails -- use that to reallocate credit rather than trying to hit an exact
  budget from the start.
- Use this scene's dossier numbers (times, distances, speed drops) as a
  reference point for your thresholds, with reasonable margin -- not as
  exact values a real rollout must reproduce.
- No imports, no dunder access, no I/O. Available names: `np` (numpy),
  `window(values, dt_s, t0, t1)` (slice a per-waypoint series to a time
  window in seconds), and standard builtins (min, max, abs, any, ...).
- Be robust: guard divisions, tolerate empty claim lists and short
  trajectories. An exception scores the rollout zero, which is worse than
  returning a low score.
- Real trajectories are NOISY: speed and lateral-offset series jitter from
  waypoint to waypoint, including the ground truth's. Never require
  monotonicity, exact equality (`==`/`!=` against a float), or
  all()-over-raw-series conditions -- they fail on the ground truth itself.
  Compare windowed aggregates (means, extrema over a time window) against
  thresholds with tolerance.
"""

_API_REFERENCE = """\
`claims` is a ParsedCoCTrace:
- claims.perceptual: list of PerceptualClaim -- .entity (canonical key, e.g.
  "pedestrian", "stopped_vehicle", "construction_cones"), .state (canonical
  key or None, e.g. "blocking", "crossing"), .text (verbatim span).
- claims.commitments: list of CommitmentClaim -- .maneuver (canonical key,
  e.g. "stop", "yield", "decelerate", "nudge", "lane_change", "keep_lane",
  "accelerate", "proceed" -- direction is a SEPARATE field, never part of
  the maneuver key itself: "lane_change_left" is not canonical, use
  .maneuver == "lane_change" and .direction == "left"), .speed_profile ("accelerate" |
  "decelerate" | "maintain" | "adapt" | None), .direction ("left" | "right"
  | None), .text.
- claims.causal: list of CausalClaim -- .effects (list[CommitmentClaim]),
  .cause (list[PerceptualClaim]), .connective.
- claims.unparsed_spans: list[(start, end)] char spans the parser could not
  account for.

`traj` is a TrajectoryFeatures:
- traj.dt_s (float, seconds per waypoint), traj.n_waypoints (int)
- traj.speed_mps, traj.heading_deg, traj.lateral_offset_m: per-waypoint
  list[float] (lateral offset is signed, + = left, in the t=0 heading frame.
  CAUTION: the frame is frozen at t=0, so on a curving road lateral offset
  accumulates the road's geometry and does NOT measure in-lane position --
  heed the dossier's total-heading-change/curvature warning before using it)
- traj.initial_speed_mps, traj.final_speed_mps, traj.min_speed_mps,
  traj.final_lateral_offset_m, traj.total_heading_change_deg (floats)
- traj.stop_event, traj.yield_event (bool, threshold-derived -- prefer the
  raw per-waypoint series over these flags)

Entity keys the parser can emit for matching against dossier classes:
pedestrian<->pedestrian, automobile/stopped_vehicle<->automobile,
heavy_truck<->heavy truck, bicycle/cyclist<->bicycle, motorcycle<->motorcycle.

`window()` returns a numpy ARRAY -- never use it in a boolean context
(`if arr:` raises ValueError); use len(arr) > 0, arr.any(), or arr.all().
"""

_STEP1 = """\
Here is the ground-truth dossier for one driving clip:

{dossier}

Step 1: identify the decisive events of this scene -- which obstacles
actually constrain the ego vehicle, when, and what the expert driver did
about them. List 1-3 decisive events with their timing and geometry, and
state which tracks are ignorable background. Be quantitative.
"""

_STEP1_IMAGE_NOTE = """\
Attached is the ego's front-wide camera frame at t0 with the expert's
future trajectory projected onto it as an orange polyline (dots every
0.4 s; the overlay is short or absent when the ego barely moves, e.g. a
creep toward a flagger). Use the image to ground your scene
understanding -- what the decisive obstacles actually ARE, where they sit
in the ego's view, and how the expert's drawn path responds to them --
and let that shape the reward logic you design. The dossier's numbers
remain authoritative for exact geometry and timing.

"""

_STEP2 = """\
Step 2: for each decisive event, define what an ACCEPTABLE rollout looks
like -- not an exact GT match:
(a) which perceptual claims should reasonably be present (entity/state
keys),
(b) which commitments should reasonably be present (maneuver keys),
(c) what the trajectory should approximately do, with flexible thresholds
inspired by (not copied from) this scene's numbers -- a real rollout will
differ from GT in its exact values and timing.
Also define what an UNFAITHFUL rollout looks like (mentions without
execution, execution without mention, wrong direction) -- this is what your
function actually needs to separate.
"""

_STEP3 = """\
Step 3: emit the reward function.

{api_reference}

{gt_claims}

{gt_traj_facts}

Write ONE python code block containing `def components(claims, traj):`
(the named component contributions) and `def reward(claims, traj):`
(exactly the clamped sum of components -- typically
`return min(1.0, max(0.0, sum(components(claims, traj).values())))`).
Compose the score however best fits this scene -- you decide the component
structure and weighting. Include a short docstring naming the decisive
events and the scene-derived thresholds. Remember the hard rules from the
system prompt.
"""

_RETRY = """\
The function you wrote failed the empirical verification gate:

{feedback}

Before writing any code, answer these in one or two sentences each:
1. For each failing case: WHICH check in your function let it through
   (or blocked the positive)?
2. WHAT measured fact from the facts block above separates that case from
   the positive (e.g. the TIME at which minimum speed occurs, not its
   magnitude)?
3. Which of your components require BOTH a COMMITMENT claim AND matching
   trajectory execution (real conjunctions -- this only works for
   commitments, never perceptual claims, since a trajectory cannot verify
   what was perceived), versus claim-only or trajectory-only? If a failing
   corruption kept most of its score, check whether the surviving
   component is a PERCEPTUAL claim gated behind a trajectory condition --
   that trajectory condition is not doing real verification and may not
   even be affecting the score (the trajectory may be identical between
   the positive and this corruption). Give perceptual claims mention-only
   credit instead, and shift the freed weight onto a real commitment
   conjunction. The mirror-image bug is just as common: a component that
   checks ONLY traj (no claims.commitments reference anywhere in it) will
   survive no_commitments and gutted_claims with IDENTICAL credit every
   time, regardless of threshold -- if a failing corruption's surviving
   component has no claims.commitments check in its code at all, that is
   the bug. Per the system prompt: if you can name a claims.commitments
   maneuver key the trajectory is executing, gate on it. Only use
   near-zero weight if you can state which maneuver key you considered
   and why it doesn't plausibly apply here -- shrinking without that
   justification is not an acceptable fix.
4. Any component that scored 0.00 on the POSITIVE case (see the breakdown)
   is mis-keyed (a claim not in the GT parse) or mis-timed (a window or
   threshold the real data never satisfies -- recheck against the measured
   facts). Fix or REMOVE it; dead components cap the positive below the bar
   no matter what else you do.
Then rewrite the offending check AROUND that separating fact. Do NOT just
nudge thresholds -- if a case scored identically to the positive, your
current checks cannot see the difference and one of them must be replaced.
Emit the corrected function in one python code block. Same contract and
hard rules.
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
                # Constraint-following codegen, not creative writing: at the
                # default temperature 1.0, yw4umq-era runs repeatedly "knew"
                # a numeric gate constraint yet sampled code that ignored it.
                "temperature": 0.2,
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
        # Label fields with their REAL attribute names -- 8sys0r lost an
        # attempt to `cc.cause_entities`, copied verbatim from the old
        # invented label here (.cause is the actual attribute).
        lines.append(
            f"    CausalClaim(connective={cl.connective!r},"
            f" effects={[e.maneuver for e in cl.effects]!r} (.maneuver of each),"
            f" cause={[p.entity for p in cl.cause]!r} (.entity of each))"
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


def render_gt_traj_facts(facts: str) -> str:
    """Frame the measured (horizon-truncated, when a rollout group exists)
    GT trajectory numbers as a pre-flight check. 8750ne collapsed 11/15
    positives because execution predicates were anchored at invented time
    windows (a yield window at t=9-20s when the GT's stop is at t=0.9s) --
    the model only saw the measured facts in retry feedback, after the
    budget was already spent on dead checks. Since the 2026-08-09
    real-rollout redesign the actual gate reference is a REAL sampled
    rollout's own argmax, not GT (run_prototype.py) -- this wording used to
    claim otherwise; fixed after udqm59's 333b20c5 analysis, where a window
    anchored to GT's own event past the real rollout's 6.4s horizon
    returned an empty slice on all 12 sampled rollouts."""
    return (
        "These are the ground-truth trajectory's numbers, restricted to the"
        " window a real rollout can actually cover (see the dossier's"
        " ROLLOUT HORIZON section) -- measured:\n  " + facts + "\n"
        "The function you write will actually be gate-verified against a"
        " REAL sampled policy rollout's own claims and trajectory, not this"
        " GT pair -- its exact numbers, timing, and even its claimed"
        " maneuvers may differ from GT. Use these numbers as a realistic"
        " dry run of your thresholds, not as the literal case you'll be"
        " scored against. A component whose check cannot fire on THIS data"
        " (wrong time window, wrong maneuver direction, a threshold never"
        " crossed) contributes 0.0 here and will likely contribute 0.0 on"
        " the real rollout too. Likewise, a component keyed to a claim that"
        " is NOT in the GT parse above is dead weight. Never anchor a time"
        " window past what these facts cover -- no real rollout will have"
        " data beyond that mark."
    )


def build_step1_message(dossier: str, overlay_jpeg: bytes | None, api: str) -> dict:
    """The opening user turn; multimodal when a scene overlay is supplied.

    api selects the content-part schema: "openai" (chat-completions
    image_url data URI) or "anthropic" (base64 image source block). The
    image rides in the SAME turn as the step-1 text so scene understanding
    is a factor from the first reasoning step onward, and the transcript
    carries it through every later call in the chain."""
    text = _STEP1.format(dossier=dossier)
    if overlay_jpeg is None:
        return {"role": "user", "content": text}
    b64 = base64.b64encode(overlay_jpeg).decode()
    text = _STEP1_IMAGE_NOTE + text
    if api == "openai":
        image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    else:
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }
    return {"role": "user", "content": [image, {"type": "text", "text": text}]}


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
    overlay_jpeg: bytes | None = None,  # scene frame + projected waypoints
    gt_traj_facts: str | None = None,  # gate._traj_facts of the GT trajectory
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
        api = "openai" if isinstance(client, OpenAIChat) else "anthropic"
        messages = [build_step1_message(dossier, overlay_jpeg, api)]
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
                    gt_traj_facts=""
                    if gt_traj_facts is None
                    else render_gt_traj_facts(gt_traj_facts),
                ),
            }
        )
        response = _call(client, messages, tracker)
        messages.append({"role": "assistant", "content": _text(response)})

    source = extract_code(_text(response))
    try:
        # Raises RewardFnError on contract violations, including a missing
        # components() -- run_prototype's invalid-reply path feeds that
        # message back as a retry, so the model self-corrects instead of
        # the clip dying. Attach the transcript built above so that retry
        # can happen in-conversation even when THIS is the first attempt
        # (no earlier successful transcript exists yet) -- without it,
        # run_prototype had no transcript to retry against and silently
        # discarded the error, letting the same mistake repeat for the
        # whole attempt budget (confirmed: ba207fc6 repeated an identical
        # non-canonical maneuver value on attempts 1-4).
        compile_reward_module(source, require_components=True)
    except RewardFnError as e:
        e.transcript = messages
        raise
    return GenerationResult(source=source, transcript=messages, model=_model(response))
