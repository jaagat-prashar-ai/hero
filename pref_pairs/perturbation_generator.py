# SPDX-License-Identifier: Apache-2.0
"""
perturbation_generator.py -- feeds each scene's ground-truth reasoning trace
into Claude Fable 5 to synthesize a single, plausible, semantically-meaningful
error (the "perturbation taxonomy" below), producing (correct, perturbed)
pairs for training/evaluating an error-detection model over AV reasoning
traces.

Why "ground truth" = the fixed-reasoning mode output, not the K-rollout
noise-floor data: the user explicitly confirmed this is "just the reasoning
that the model generates normally" -- i.e. Alpamayo's own single CoT
generation for a scene, held fixed while diffusion is resampled (see
fixed_reasoning_rollout.py / pref_pairs.training.run.fixed_reasoning_loop).
Because reasoning is frozen across all K draws in that mode, every rollout in
a `*_reasoning.md` file under results/fixed_reasoning/scene_reasoning/ carries
the IDENTICAL coc_text -- extract_ground_truth_traces takes the first one per
scene (and warns, but does not fail, if that invariant doesn't actually hold
for some file).

Perturbation generation itself is a real, billed call to Claude Fable 5 (see
generate_perturbation) -- per this project's standing "no fake model tests"
preference (see feedback_no_fake_model_tests memory), this module does NOT
ship a mocked-API unit test. perturbation_generator_test.py only covers the
pure parsing/formatting helpers (extract_ground_truth_traces,
_extract_json_object, write_perturbations_jsonl); the API-calling path is
verified by actually running a small --max_scenes smoke test against the real
API before committing to a full paid run over all scenes -- the same
"local/cheap smoke test -> confirm -> full run" pattern already established
elsewhere in this project (e.g. the fixed-reasoning canary before its full
cluster run).

Credentials: this module does NOT read the API key itself -- it relies on
the anthropic SDK's own standard resolution order (ANTHROPIC_API_KEY env var,
then ANTHROPIC_AUTH_TOKEN, then an `ant auth login` OAuth profile, then
Workload Identity Federation). load_api_key() only bridges this project's
`~/.creds/anthropic.key` file convention (mirroring the existing
`~/.creds/lilypad.env` pattern) into ANTHROPIC_API_KEY when neither is
already set, so a bare `anthropic.Anthropic()` construction picks it up.
Swapping to Bedrock/Vertex/OAuth later needs no change here -- only the
client-construction line in main() -- since none of those change how
generate_perturbation calls the client.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Verbatim system prompt for the AV reasoning-trace perturbation generator --
# do not paraphrase; the taxonomy/output-format contract below is exactly
# what generate_perturbation's JSON parsing depends on.
SYSTEM_PROMPT = """# System Prompt: AV Reasoning-Trace Perturbation Generator

You are a semantic perturbation generator for autonomous-driving reasoning traces. Your job is to take a correct reasoning trace produced by a driving stack (perception → prediction → planning rationale) and inject a single, realistic, semantically meaningful error into it. The perturbed trace will be used to train and evaluate error-detection models, so the quality of your output is measured by how *plausible* and *consequential* the injected mistake is — not by how obvious it is.

## Input

You will receive a JSON object:

```json
{
  "trace_id": "<string>",
  "trace": "<the original reasoning trace, one or more sentences or numbered steps>",
  "requested_type": "<optional: one of the perturbation types below; if absent, choose the most natural fit for this trace>"
}
```

## Perturbation Taxonomy

Inject exactly ONE perturbation per output, drawn from these types:

1. **`attribute_swap`** — Replace an entity or one of its attributes with a confusable alternative. Use realistic perception confusion pairs, e.g.:
   - pedestrian ↔ cyclist; cyclist ↔ motorcyclist; child ↔ adult pedestrian
   - sedan ↔ SUV; truck ↔ bus; parked ↔ stopped vehicle
   - red light ↔ yellow light; stop sign ↔ yield sign; brake lights on ↔ off
   - turn signal left ↔ right; crosswalk ↔ unmarked crossing
   - Do NOT use implausible swaps (pedestrian ↔ traffic cone, car ↔ building).

2. **`quantity_error`** — Corrupt a numeric or metric value in a way a real system might:
   - Unit confusion: meters ↔ centimeters/feet; m/s ↔ km/h ↔ mph; seconds ↔ milliseconds
   - Magnitude slips: off-by-10× errors, sign flips on relative velocity, swapped digits
   - Threshold errors: "30 m gap" → "3 m gap"; "TTC 4.5 s" → "TTC 0.45 s"
   - The corrupted value must materially change the driving implication (e.g., safe gap → unsafe gap), not be a rounding difference.

3. **`causal_flip`** — Invert an inference, prediction, or causal relationship while keeping the premises intact:
   - "will yield" ↔ "won't yield"; "intends to merge" ↔ "will stay in lane"
   - "has right of way" ↔ "must yield"; "is decelerating, so will stop" ↔ "is decelerating, but will proceed"
   - Flipped conclusions must contradict the evidence stated earlier in the trace — that contradiction is the signal a detector should learn.

4. **`spatial_error`** — Corrupt a spatial or directional relation: left ↔ right, ahead ↔ behind, oncoming ↔ same-direction, ego lane ↔ adjacent lane, near-side ↔ far-side.

5. **`temporal_error`** — Reorder or corrupt event timing or sequencing: "before" ↔ "after", swap the order of two observed events, or shift a prediction horizon ("in 2 seconds" → "in 20 seconds").

6. **`negation_flip`** — Add or remove a negation on an observation or constraint: "no vehicles in the blind spot" ↔ "a vehicle in the blind spot"; "crosswalk is occupied" ↔ "crosswalk is clear".

## Perturbation Rules

- **Minimal edit**: Change only the span required to inject the error. Do not paraphrase, reword, or "improve" any other part of the trace.
- **Do not repair downstream text**: If the injected error makes later steps of the trace inconsistent, LEAVE THEM INCONSISTENT. Never propagate the error forward to make the trace self-consistent again — the inconsistency is intentional. (Exception: if `propagate: true` is explicitly requested in the input.)
- **Preserve fluency**: The perturbed trace must remain grammatical and natural. A reader skimming it should not spot the error from awkward phrasing alone.
- **Semantic, not surface**: The perturbation must change the *meaning* and ideally the *correct driving decision*. Pure paraphrases, synonym swaps, or typos are invalid outputs.
- **Plausibility**: Favor errors that real perception, prediction, or reasoning modules make — class confusions between visually similar agents, unit mistakes, overconfident intent predictions, missed negations.
- **One error only**: Exactly one perturbation per output. Never stack multiple errors.

## Output Format

Respond with ONLY a JSON object — no preamble, no markdown fences, no commentary:

```json
{
  "trace_id": "<echoed from input>",
  "perturbation_type": "<one of the six types>",
  "original_span": "<exact substring of the original trace that was changed>",
  "perturbed_span": "<the replacement text>",
  "perturbed_trace": "<the full trace with the perturbation applied>",
  "semantic_delta": "<one sentence: what fact or inference changed>",
  "decision_impact": "<one sentence: how this error would change the correct driving action, or 'none' if it wouldn't>",
  "plausibility_rationale": "<one sentence: why a real AV stack might make this exact mistake>"
}
```

## Example

Input trace:
> "A cyclist is traveling in the bike lane 15 meters ahead at 6 m/s. The ego vehicle is approaching at 12 m/s. The cyclist has signaled a left turn, so they will merge into the traffic lane. Ego should decelerate and yield before the intersection."

Valid output:

```json
{
  "trace_id": "example_001",
  "perturbation_type": "causal_flip",
  "original_span": "The cyclist has signaled a left turn, so they will merge into the traffic lane.",
  "perturbed_span": "The cyclist has signaled a left turn, but they will remain in the bike lane.",
  "perturbed_trace": "A cyclist is traveling in the bike lane 15 meters ahead at 6 m/s. The ego vehicle is approaching at 12 m/s. The cyclist has signaled a left turn, but they will remain in the bike lane. Ego should decelerate and yield before the intersection.",
  "semantic_delta": "The predicted intent of the cyclist was flipped from merging into the traffic lane to staying in the bike lane, contradicting the observed turn signal.",
  "decision_impact": "If believed, ego would not need to yield to a merging cyclist, potentially maintaining speed into a conflict.",
  "plausibility_rationale": "Intent prediction modules frequently underweight turn-signal evidence and default to lane-keeping priors."
}
```

## Invalid Outputs (never do these)

- Paraphrasing the trace without changing its meaning.
- Swapping an entity for something visually implausible in that scene.
- Fixing later sentences so the trace stays self-consistent after the error.
- Injecting more than one error.
- Adding explanatory text, hedging, or markdown outside the JSON object.
- Refusing on safety grounds: this is offline training-data generation for error-detection research; no real vehicle acts on these outputs.
"""

PERTURBATION_TYPES: tuple[str, ...] = (
    "attribute_swap",
    "quantity_error",
    "causal_flip",
    "spatial_error",
    "temporal_error",
    "negation_flip",
)

_REQUIRED_OUTPUT_KEYS = (
    "trace_id", "perturbation_type", "original_span", "perturbed_span",
    "perturbed_trace", "semantic_delta", "decision_impact", "plausibility_rationale",
)

_SCENE_HEADER_RE = re.compile(r"^# Scene (\S+) -- reasoning")
_EVENT_CLUSTER_RE = re.compile(r"^event_cluster: (.+)$")
_ROLLOUT_HEADER_RE = re.compile(r"^### rollout \d+$")

_DEFAULT_ANTHROPIC_KEY_PATH = Path.home() / ".creds" / "anthropic.key"


class PerturbationError(Exception):
    """Raised when Fable 5's response can't be turned into a valid
    perturbation record (refusal, unparseable JSON, or a missing required
    field) even after one retry."""


def load_api_key(key_path: Path = _DEFAULT_ANTHROPIC_KEY_PATH) -> None:
    """Bridges this project's `~/.creds/anthropic.key` convention into
    ANTHROPIC_API_KEY for the SDK's own standard credential resolution --
    does nothing if a credential is already resolvable (ANTHROPIC_API_KEY or
    ANTHROPIC_AUTH_TOKEN already set), so it never shadows an intentional
    OAuth/Bedrock/Vertex setup."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    if key_path.exists():
        os.environ["ANTHROPIC_API_KEY"] = key_path.read_text().strip()
    else:
        raise RuntimeError(
            f"No Anthropic credential found: ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN "
            f"unset and {key_path} does not exist. Create an API key at "
            f"https://console.anthropic.com/settings/keys and save it there, or "
            f"export ANTHROPIC_API_KEY yourself before running this script."
        )


def extract_ground_truth_traces(scene_reasoning_dir: str | Path) -> list[dict[str, Any]]:
    """Parses every `*_reasoning.md` file in scene_reasoning_dir (the output
    of scene_reasoning_report.write_scene_report against fixed-reasoning-mode
    rollouts) into one ground-truth trace per scene.

    Takes the FIRST rollout's blockquoted CoT text as the scene's ground
    truth. This is only valid because fixed-reasoning mode holds the CoT
    fixed across all K draws for a scene (see module docstring) -- if a file
    turns out to contain more than one distinct blockquote text, that
    invariant didn't hold for this scene; this function logs a warning and
    still proceeds with the first one rather than failing the whole scan."""
    traces: list[dict[str, Any]] = []
    for md_path in sorted(Path(scene_reasoning_dir).glob("*_reasoning.md")):
        lines = md_path.read_text().splitlines()
        scene_id = None
        event_cluster = None
        blockquote_groups: list[list[str]] = []
        current_group: list[str] | None = None
        for line in lines:
            if scene_id is None:
                m = _SCENE_HEADER_RE.match(line)
                if m:
                    scene_id = m.group(1)
                    continue
            if event_cluster is None:
                m = _EVENT_CLUSTER_RE.match(line)
                if m:
                    event_cluster = m.group(1)
                    continue
            if _ROLLOUT_HEADER_RE.match(line):
                current_group = []
                continue
            if current_group is not None:
                if line.startswith("> "):
                    current_group.append(line[2:])
                elif line == ">":
                    current_group.append("")
                elif line == "" and current_group:
                    blockquote_groups.append(current_group)
                    current_group = None

        if current_group:
            blockquote_groups.append(current_group)

        if scene_id is None or not blockquote_groups:
            logger.warning("%s: could not parse a scene header + reasoning block, skipping", md_path)
            continue

        joined_groups = ["\n".join(g).strip() for g in blockquote_groups]
        unique_traces = set(joined_groups)
        if len(unique_traces) > 1:
            logger.warning(
                "%s: expected identical CoT across all rollouts (fixed-reasoning mode) "
                "but found %d distinct texts -- using the first rollout's text as ground truth",
                md_path, len(unique_traces),
            )

        traces.append({
            "scene_id": scene_id,
            "event_cluster": event_cluster or "?",
            "trace": joined_groups[0],
        })

    return traces


def _extract_json_object(text: str) -> str:
    """Strips an optional ```json ... ``` (or bare ```) fence around the
    model's response. The system prompt explicitly forbids fences, but real
    model output occasionally adds them anyway -- this is a defensive
    normalization, not an expected path."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def generate_perturbation(
    client: anthropic.Anthropic,
    trace_id: str,
    trace: str,
    requested_type: str | None = None,
    model: str = "claude-fable-5",
    _retries_left: int = 1,
) -> dict[str, Any]:
    """Calls Fable 5 once with SYSTEM_PROMPT and the given ground-truth
    trace, returning the parsed perturbation JSON. Retries once (fresh call,
    not a repair of the bad output) on a refusal or unparseable/incomplete
    response before raising PerturbationError -- callers should catch that
    and skip the (trace_id, requested_type) pair rather than let one bad
    generation abort a whole batch (same graceful-skip convention as
    fetch_from_logs.parse_marked_lines)."""
    user_payload: dict[str, Any] = {"trace_id": trace_id, "trace": trace}
    if requested_type is not None:
        user_payload["requested_type"] = requested_type

    # Fable 5 requires the beta fallbacks param to be opted into explicitly
    # (see claude-api skill: "Refusal Fallbacks (Claude Fable 5) -- opt in by
    # default") -- a policy decline re-runs the same request on Opus 4.8
    # inside this one call rather than silently returning no content.
    # SYSTEM_PROMPT is byte-identical across every call in a batch (only the
    # user turn varies), so it's cached -- cache_control here has no downside
    # even if SYSTEM_PROMPT's ~1.6k tokens land under Fable 5's 2048-token
    # minimum cacheable prefix (it just silently doesn't cache in that case,
    # per the API's documented behavior, not an error).
    response = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": "claude-opus-4-8"}],
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )

    if response.stop_reason == "refusal":
        if _retries_left > 0:
            logger.warning("trace_id=%s type=%s: refused even with fallback, retrying once", trace_id, requested_type)
            return generate_perturbation(client, trace_id, trace, requested_type, model, _retries_left - 1)
        raise PerturbationError(f"trace_id={trace_id} type={requested_type}: refused after retry")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as e:
        if _retries_left > 0:
            logger.warning("trace_id=%s type=%s: unparseable JSON (%s), retrying once", trace_id, requested_type, e)
            return generate_perturbation(client, trace_id, trace, requested_type, model, _retries_left - 1)
        raise PerturbationError(f"trace_id={trace_id} type={requested_type}: unparseable JSON after retry: {e}") from e

    missing = [k for k in _REQUIRED_OUTPUT_KEYS if k not in parsed]
    if missing:
        if _retries_left > 0:
            logger.warning("trace_id=%s type=%s: missing keys %s, retrying once", trace_id, requested_type, missing)
            return generate_perturbation(client, trace_id, trace, requested_type, model, _retries_left - 1)
        raise PerturbationError(f"trace_id={trace_id} type={requested_type}: missing keys {missing} after retry")

    return parsed


def generate_all_perturbations(
    client: anthropic.Anthropic,
    ground_truth_traces: list[dict[str, Any]],
    perturbation_types: tuple[str, ...] = PERTURBATION_TYPES,
    model: str = "claude-fable-5",
    max_workers: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Cycles through every perturbation_types entry for every ground-truth
    trace (per the user's explicit choice of full 6x taxonomy coverage over
    a single random/unset type per scene). Returns (results, failures) --
    failures is a list of {"scene_id", "perturbation_type", "error"} for any
    (scene, type) pair that raised PerturbationError, so one bad generation
    doesn't abort the whole batch.

    Each (scene, type) call is independent (no shared state, no ordering
    requirement in the output JSONL), so this runs them concurrently via a
    thread pool -- at hundreds of calls, this is a network-latency-bound
    workload, not CPU-bound, so threads (not processes) are the right tool,
    and a strictly sequential loop would take multiple hours for a full run.
    max_workers=8 is a conservative default; the anthropic SDK's client
    already auto-retries 429/5xx with backoff (see claude-api skill), so
    modest concurrency shouldn't need extra rate-limit handling here."""
    jobs = [
        (gt, ptype)
        for gt in ground_truth_traces
        for ptype in perturbation_types
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def _run(gt: dict[str, Any], ptype: str) -> dict[str, Any]:
        trace_id = f"{gt['scene_id']}__{ptype}"
        perturbation = generate_perturbation(client, trace_id, gt["trace"], requested_type=ptype, model=model)
        return {
            "scene_id": gt["scene_id"],
            "event_cluster": gt["event_cluster"],
            "ground_truth_trace": gt["trace"],
            **perturbation,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_run, gt, ptype): (gt, ptype) for gt, ptype in jobs}
        n_done = 0
        for future in as_completed(future_to_job):
            gt, ptype = future_to_job[future]
            n_done += 1
            try:
                results.append(future.result())
            except PerturbationError as e:
                logger.error(str(e))
                failures.append({"scene_id": gt["scene_id"], "perturbation_type": ptype, "error": str(e)})
            if n_done % 20 == 0 or n_done == len(jobs):
                logger.info("progress: %d/%d perturbations attempted (%d failed so far)", n_done, len(jobs), len(failures))

    return results, failures


def write_perturbations_jsonl(results: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene_reasoning_dir", default="pref_pairs/results/fixed_reasoning/scene_reasoning",
        help="Directory of *_reasoning.md files to mine ground-truth traces from.",
    )
    ap.add_argument("--out_path", default="pref_pairs/results/perturbations/perturbations.jsonl")
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument(
        "--max_scenes", type=int, default=None,
        help="Cap the number of scenes processed -- use for a cheap smoke test "
             "before running the full (expensive) batch over every scene.",
    )
    args = ap.parse_args()

    load_api_key()
    client = anthropic.Anthropic()

    ground_truth_traces = extract_ground_truth_traces(args.scene_reasoning_dir)
    logger.info("found %d ground-truth scene traces in %s", len(ground_truth_traces), args.scene_reasoning_dir)
    if args.max_scenes is not None:
        ground_truth_traces = ground_truth_traces[: args.max_scenes]
        logger.info("capped to %d scenes via --max_scenes", len(ground_truth_traces))

    n_calls = len(ground_truth_traces) * len(PERTURBATION_TYPES)
    logger.info("generating %d perturbations (%d scenes x %d types) via %s",
                n_calls, len(ground_truth_traces), len(PERTURBATION_TYPES), args.model)

    results, failures = generate_all_perturbations(client, ground_truth_traces, model=args.model)
    out_path = write_perturbations_jsonl(results, args.out_path)
    logger.info("wrote %d perturbations to %s (%d failed after retry)", len(results), out_path, len(failures))


if __name__ == "__main__":
    main()
