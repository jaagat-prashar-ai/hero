# SPDX-License-Identifier: Apache-2.0
"""
perturbation_generator_v2.py — claim-targeted, style-preserving CoC
perturbations. Successor to pref_pairs/perturbation_generator.py (v1), built
to fix the failure its own postmortem documented
(pref_pairs/results/faithfulness_embedding_grader/SUMMARY.md): v1
perturbations leave a TEXT FINGERPRINT — a constant embedding-space
direction with no trajectory input classifies chosen-vs-perturbed at 95.8%,
so any pair corpus built from them is text-only-solvable. v1's own scratch
note (perturbation_generator.py:348-351) points the same way: edit
VERIFIABLE QUANTITIES while preserving style.

What v2 changes:

  1. CLAIM TARGETING. Each trace is pre-parsed with
     code_as_a_reward.coc_claim_parser.parse_coc_trace into typed claims
     with char spans; every perturbation targets exactly ONE claim, whose
     span/type/details are handed to the model. The taxonomy is anchored to
     what the code-reward verifiers can check (commitment_verifier /
     perceptual_verifier), i.e. every edit changes a verifiable quantity:

       commitment_direction_flip   CommitmentClaim.direction left <-> right
       commitment_maneuver_swap    swap within the MANEUVER_PATTERNS lexicon,
                                   respecting axis (stop<->proceed,
                                   accelerate<->decelerate, ...)
       perceptual_state_flip       STATE_PATTERNS antonym (blocking<->clear,
                                   red<->green, occupied<->empty, ...)
       perceptual_entity_swap      confusable in-lexicon entity
                                   (pedestrian<->cyclist, ...)
       causal_cause_substitution   replace the cause clause's perceptual
                                   content with a different plausible
                                   in-lexicon cause
       quantity_edit               corrupt a numeric value (distance/speed/
                                   time) so the driving implication changes

  2. STYLE CONTRACT + MECHANICAL VALIDATION. The prompt requires: only the
     target span changes, replacement uses vocabulary already common in this
     corpus, word count within ±2, no hedging/qualifier insertions (v1's
     fingerprint source). validate_perturbation() then checks MECHANICALLY
     that the trace outside the edited span is byte-identical and the word
     count rule holds — a violation is retried once fresh, then skipped
     (graceful-skip, same as v1).

  3. FINGERPRINT GATE. fingerprint_check() re-runs the constant
     text-direction baseline (the exact attack that broke v1, from the
     grader's baselines.json methodology) on a finished corpus: mean
     embedding-difference direction fit on a train split, sign-of-projection
     classification on holdout. The corpus is DPO-eligible only if holdout
     accuracy <= 0.60 (near-chance); run it via --fingerprint_check.

API conventions are v1's verbatim (imported, not copied): ~/.creds/
anthropic.key bridging, beta server-side fallbacks to opus-4-8, cached
system block, ThreadPoolExecutor, retry-on-refusal. Per this project's
standing rules there is NO mocked-API test, and main() is a DRY RUN unless
--confirm is passed: it prints the call count + cost estimate and exits, so
a paid run can't start without the cost having been stated and approved.
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

from code_as_a_reward.coc_claim_parser import ParsedCoCTrace, parse_coc_trace
from pref_pairs.perturbation_generator import (
    PerturbationError,
    _extract_json_object,
    extract_ground_truth_traces,
    load_api_key,
    write_perturbations_jsonl,
)

logger = logging.getLogger(__name__)

TAXONOMY_VERSION = "v2"

PERTURBATION_TYPES_V2: tuple[str, ...] = (
    "commitment_direction_flip",
    "commitment_maneuver_swap",
    "perceptual_state_flip",
    "perceptual_entity_swap",
    "causal_cause_substitution",
    "quantity_edit",
)

# Numeric quantities with a driving-relevant unit — the quantity_edit targets.
_QUANTITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m/s|km/h|mph|m\b|meters?\b|s\b|seconds?\b|deg(?:rees)?\b|%)",
    re.IGNORECASE,
)

# Verbatim system prompt — the output-format contract below is exactly what
# generate_perturbation_v2's JSON parsing and mechanical validation depend on.
SYSTEM_PROMPT_V2 = """# System Prompt: AV Reasoning-Trace Perturbation Generator (v2, claim-targeted)

You are a semantic perturbation generator for autonomous-driving chain-of-causation (CoC) reasoning traces. You will be given a correct trace AND one specific target claim inside it (with its exact character span and parsed details). Your job is to inject exactly ONE semantically meaningful error INTO THAT CLAIM ONLY, so the edited trace implies a materially different driving action.

These traces come from one driving model and share a tight, telegraphic style ("Nudge left due to cones blocking the lane, then accelerate."). Your edit must be indistinguishable in style from the surrounding corpus — a previous version of this generator failed because its edits carried a detectable textual fingerprint.

## Input

```json
{
  "trace_id": "<string>",
  "trace": "<the original trace>",
  "requested_type": "<one of the six types below>",
  "target": {
    "claim_type": "<commitment|perceptual|causal|quantity>",
    "claim_text": "<exact substring of trace being targeted>",
    "span": [<start>, <end>],
    "details": { ... parsed claim fields: maneuver/axis/direction, entity/state, connective, ... }
  }
}
```

## Perturbation Types

1. **`commitment_direction_flip`** — Flip the target commitment's direction word: left <-> right. Change nothing else.
2. **`commitment_maneuver_swap`** — Replace the target maneuver verb with a DIFFERENT maneuver of materially different consequence, using only verbs this corpus already uses (stop, yield, wait, proceed, accelerate, decelerate, adapt speed, nudge, change lane, merge, turn, keep lane, enter). Respect grammar (tense/number).
3. **`perceptual_state_flip`** — Flip the target entity's state predicate to a plausible opposite this corpus already uses: blocking <-> clearing/clear, occupied <-> empty, red <-> green, stopped <-> moving, approaching <-> receding, open <-> closed.
4. **`perceptual_entity_swap`** — Replace the target entity with a visually confusable one from this corpus's vocabulary: pedestrian <-> cyclist, cyclist <-> motorcyclist, cones <-> debris, stopped vehicle <-> parked vehicle, truck <-> bus. Never an implausible swap.
5. **`causal_cause_substitution`** — Replace the CAUSE content of the target causal claim with a different plausible in-corpus cause (e.g. "due to cones blocking the lane" -> "due to a stopped vehicle in the lane"). Keep the causal connective and the committed maneuver untouched.
6. **`quantity_edit`** — Corrupt the target numeric value so the driving implication changes materially (safe gap -> unsafe gap, slow -> fast): off-by-10x, unit confusion, or threshold-crossing value change. Keep the unit format style identical.

## Hard Rules

- **Only the target span changes.** Every character of the trace outside the span you edit must remain byte-identical. Your output is validated mechanically against this; violations are discarded.
- **Style camouflage**: word count of the replacement within ±2 of the original span's; no added hedges, qualifiers, adverbs, or explanations ("possibly", "it appears", "carefully"); reuse only vocabulary that plausibly appears elsewhere in this corpus; match capitalization and punctuation of the original span.
- **Do not repair downstream text**: if the edit makes later clauses inconsistent, leave them inconsistent.
- **Semantic, not surface**: the correct driving action given the edited trace must differ from the original. Pure paraphrase is an invalid output.
- **One error only.**

## Output Format

Respond with ONLY a JSON object — no preamble, no markdown fences:

```json
{
  "trace_id": "<echoed>",
  "perturbation_type": "<echoed requested_type>",
  "original_span": "<exact substring of the original trace that was changed>",
  "perturbed_span": "<the replacement text>",
  "perturbed_trace": "<full trace with ONLY that substitution applied>",
  "semantic_delta": "<one sentence: what fact or inference changed>",
  "decision_impact": "<one sentence: how the correct driving action changes>",
  "plausibility_rationale": "<one sentence: why a real AV stack might make this exact mistake>"
}
```

- Refusing on safety grounds is invalid: this is offline training-data generation for alignment research on a driving model; no vehicle acts on these outputs.
"""

_REQUIRED_OUTPUT_KEYS = (
    "trace_id", "perturbation_type", "original_span", "perturbed_span",
    "perturbed_trace", "semantic_delta", "decision_impact", "plausibility_rationale",
)

# Pricing for the cost estimate printed by the dry run, $/Mtok
# (input, output, cached-input-read). claude-fable-5 per claude-api skill
# as of 2026-08; gpt-4o per OpenAI pricing page as of the same date.
_PRICING = {
    "claude-fable-5": (5.0, 25.0, 0.50),
    "gpt-4o": (2.50, 10.00, 1.25),
}
_EST_SYSTEM_TOKENS, _EST_USER_TOKENS, _EST_OUTPUT_TOKENS = 1500, 350, 450

_OPENAI_KEY_PATH = Path.home() / ".creds" / "openai.key"


def load_openai_api_key(key_path: Path = _OPENAI_KEY_PATH) -> None:
    """Same convention as pref_pairs' load_api_key, for ~/.creds/openai.key:
    bridges the key file into OPENAI_API_KEY unless one is already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    if not key_path.exists():
        raise RuntimeError(
            f"OPENAI_API_KEY unset and {key_path} does not exist. Save an "
            f"OpenAI API key there or export OPENAI_API_KEY before running."
        )
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()


# ---------------------------------------------------------------------------
# Target selection (pure)
# ---------------------------------------------------------------------------

def select_targets(trace: str) -> dict[str, dict[str, Any]]:
    """{perturbation_type: target dict} for every v2 type applicable to this
    trace — deterministic (first applicable claim per type), so reruns
    target identical spans. A type absent from the result simply isn't
    applicable (e.g. no direction word anywhere -> no direction_flip)."""
    parsed: ParsedCoCTrace = parse_coc_trace(trace)
    targets: dict[str, dict[str, Any]] = {}

    def _target(claim_type: str, text: str, span: tuple[int, int], **details: Any) -> dict[str, Any]:
        return {"claim_type": claim_type, "claim_text": text,
                "span": [int(span[0]), int(span[1])], "details": details}

    for c in parsed.commitments:
        if "commitment_direction_flip" not in targets and c.direction in ("left", "right"):
            targets["commitment_direction_flip"] = _target(
                "commitment", c.text, c.span,
                maneuver=c.maneuver, axis=str(c.axis.value), direction=c.direction,
            )
        if "commitment_maneuver_swap" not in targets:
            targets["commitment_maneuver_swap"] = _target(
                "commitment", c.text, c.span,
                maneuver=c.maneuver, axis=str(c.axis.value),
                speed_profile=c.speed_profile, direction=c.direction,
            )

    for p in parsed.perceptual:
        if "perceptual_state_flip" not in targets and p.state is not None:
            targets["perceptual_state_flip"] = _target(
                "perceptual", p.text, p.span, entity=p.entity, state=p.state,
            )
        if "perceptual_entity_swap" not in targets:
            targets["perceptual_entity_swap"] = _target(
                "perceptual", p.text, p.span, entity=p.entity, state=p.state,
            )

    for cc in parsed.causal:
        if cc.cause and "causal_cause_substitution" not in targets:
            targets["causal_cause_substitution"] = _target(
                "causal", cc.text, cc.span,
                connective=cc.connective,
                cause_entities=[c.entity for c in cc.cause],
                effect_maneuvers=[e.maneuver for e in cc.effects],
            )
            break

    m = _QUANTITY_RE.search(trace)
    if m:
        targets["quantity_edit"] = _target("quantity", m.group(0), m.span())

    return targets


# ---------------------------------------------------------------------------
# Mechanical validation (pure)
# ---------------------------------------------------------------------------

def validate_perturbation(original_trace: str, result: dict[str, Any]) -> str | None:
    """None if the model's output honors the hard rules, else a short reason
    string (used in logs and to decide the retry). Checks, in order:
      1. original_span occurs in the original trace;
      2. perturbed_trace == original with EXACTLY that one substitution
         (tried at every occurrence of original_span, in case it repeats);
      3. spans differ (a no-op paraphrase is invalid);
      4. word count within ±2 (the style-camouflage rule)."""
    orig_span = result.get("original_span") or ""
    pert_span = result.get("perturbed_span") or ""
    pert_trace = result.get("perturbed_trace") or ""

    if not orig_span or orig_span not in original_trace:
        return "original_span not found in trace"
    if orig_span == pert_span:
        return "identity edit (original_span == perturbed_span)"

    ok_somewhere = False
    start = 0
    while True:
        i = original_trace.find(orig_span, start)
        if i == -1:
            break
        candidate = original_trace[:i] + pert_span + original_trace[i + len(orig_span):]
        if candidate == pert_trace:
            ok_somewhere = True
            break
        start = i + 1
    if not ok_somewhere:
        return "perturbed_trace differs from original outside the edited span"

    wc_delta = abs(len(pert_span.split()) - len(orig_span.split()))
    if wc_delta > 2:
        return f"word-count delta {wc_delta} > 2"
    return None


# ---------------------------------------------------------------------------
# Generation (real, billed API calls — no mocked tests, smoke-then-scale)
# ---------------------------------------------------------------------------

def _call_backend(client: Any, model: str, user_payload: dict[str, Any]) -> tuple[str, bool]:
    """One model call; returns (response_text, refused). Dispatches on the
    client's type so the validation/retry pipeline stays backend-agnostic."""
    if client.__class__.__module__.startswith("openai"):
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            # JSON mode guarantees parseable output (SYSTEM_PROMPT_V2 already
            # demands a bare JSON object, which JSON mode requires be mentioned).
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason == "content_filter"

    response = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": "claude-opus-4-8"}],
        system=[{"type": "text", "text": SYSTEM_PROMPT_V2, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text, response.stop_reason == "refusal"


def generate_perturbation_v2(
    client: Any,
    trace_id: str,
    trace: str,
    requested_type: str,
    target: dict[str, Any],
    model: str = "claude-fable-5",
    _retries_left: int = 1,
) -> dict[str, Any]:
    """One claim-targeted perturbation. Retries once (fresh call) on refusal,
    unparseable JSON, missing keys, or MECHANICAL VALIDATION failure — the
    last being the v2 addition — then raises PerturbationError for the
    caller to skip (never abort the batch)."""
    user_payload = {
        "trace_id": trace_id, "trace": trace,
        "requested_type": requested_type, "target": target,
    }

    def _retry(reason: str) -> dict[str, Any]:
        if _retries_left > 0:
            logger.warning("trace_id=%s: %s, retrying once", trace_id, reason)
            return generate_perturbation_v2(
                client, trace_id, trace, requested_type, target, model, _retries_left - 1,
            )
        raise PerturbationError(f"trace_id={trace_id}: {reason} after retry")

    text, refused = _call_backend(client, model, user_payload)
    if refused:
        return _retry("refused")

    try:
        parsed = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as e:
        return _retry(f"unparseable JSON ({e})")

    missing = [k for k in _REQUIRED_OUTPUT_KEYS if k not in parsed]
    if missing:
        return _retry(f"missing keys {missing}")

    violation = validate_perturbation(trace, parsed)
    if violation is not None:
        return _retry(f"mechanical validation failed: {violation}")

    return parsed


def generate_all_perturbations_v2(
    client: Any,
    ground_truth_traces: list[dict[str, Any]],
    model: str = "claude-fable-5",
    max_workers: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """One call per (scene, applicable type). Types the claim parser finds no
    target for are skipped silently per scene (counted in the log line) —
    unlike v1's fixed 6x fan-out, the job list is corpus-shaped."""
    jobs: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for gt in ground_truth_traces:
        targets = select_targets(gt["trace"])
        for ptype, target in targets.items():
            jobs.append((gt, ptype, target))
    logger.info("%d jobs across %d scenes (avg %.1f applicable types/scene)",
                len(jobs), len(ground_truth_traces),
                len(jobs) / max(len(ground_truth_traces), 1))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def _run(gt: dict[str, Any], ptype: str, target: dict[str, Any]) -> dict[str, Any]:
        trace_id = f"{gt['scene_id']}__{ptype}"
        perturbation = generate_perturbation_v2(
            client, trace_id, gt["trace"], ptype, target, model=model,
        )
        return {
            "scene_id": gt["scene_id"],
            "event_cluster": gt["event_cluster"],
            "ground_truth_trace": gt["trace"],
            "target_claim_type": target["claim_type"],
            "target_claim_span": target["span"],
            "taxonomy_version": TAXONOMY_VERSION,
            "generator_model": model,
            **perturbation,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_run, gt, ptype, target): (gt, ptype)
                         for gt, ptype, target in jobs}
        n_done = 0
        for future in as_completed(future_to_job):
            gt, ptype = future_to_job[future]
            n_done += 1
            try:
                results.append(future.result())
            except PerturbationError as e:
                logger.error(str(e))
                failures.append({"scene_id": gt["scene_id"], "perturbation_type": ptype,
                                 "error": str(e)})
            if n_done % 20 == 0 or n_done == len(jobs):
                logger.info("progress: %d/%d (%d failed)", n_done, len(jobs), len(failures))

    return results, failures


# ---------------------------------------------------------------------------
# Fingerprint gate
# ---------------------------------------------------------------------------

def constant_direction_accuracy(
    clean_texts: list[str], perturbed_texts: list[str],
    holdout_fraction: float = 0.5, seed: int = 0,
) -> float:
    """The exact attack that broke v1 (grader baselines.json,
    constant_mean_diff_direction): fit the mean (clean - perturbed) embedding
    direction on a train split, classify holdout pairs by projection sign.
    Returns holdout accuracy — 0.5 is chance; v1 scored 0.958. The v2 corpus
    is DPO-eligible only at <= 0.60."""
    import numpy as np

    from pref_pairs.faithfulness_embedding_grader import embed_traces

    assert len(clean_texts) == len(perturbed_texts)
    clean = embed_traces(clean_texts).numpy()
    pert = embed_traces(perturbed_texts).numpy()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clean))
    n_train = max(1, int(len(idx) * (1 - holdout_fraction)))
    train, hold = idx[:n_train], idx[n_train:]

    direction = (clean[train] - pert[train]).mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0 or len(hold) == 0:
        return 0.5
    direction /= norm
    # A pair is "classified correctly" if the clean side projects higher.
    correct = ((clean[hold] - pert[hold]) @ direction > 0).mean()
    return float(correct)


def fingerprint_check(perturbations_path: str | Path, threshold: float = 0.60) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(perturbations_path) if line.strip()]
    acc = constant_direction_accuracy(
        [r["ground_truth_trace"] for r in rows],
        [r["perturbed_trace"] for r in rows],
    )
    verdict = {
        "n_pairs": len(rows),
        "constant_direction_holdout_accuracy": acc,
        "threshold": threshold,
        "passes": acc <= threshold,
        "v1_reference_accuracy": 0.958,
    }
    logger.info("fingerprint check: %.3f (threshold %.2f) -> %s",
                acc, threshold, "PASS" if verdict["passes"] else "FAIL")
    return verdict


# ---------------------------------------------------------------------------
# CLI — dry run by default; --confirm required for the paid batch
# ---------------------------------------------------------------------------

def estimate_cost(n_calls: int, model: str = "claude-fable-5") -> float:
    """USD estimate. System prompt is cache-read after the first call; user
    turn + output are per-call. Unknown models fall back to fable pricing."""
    if n_calls == 0:
        return 0.0
    price_in, price_out, price_cache_read = _PRICING.get(model, _PRICING["claude-fable-5"])
    first = _EST_SYSTEM_TOKENS / 1e6 * price_in
    rest = (n_calls - 1) * _EST_SYSTEM_TOKENS / 1e6 * price_cache_read
    user = n_calls * _EST_USER_TOKENS / 1e6 * price_in
    out = n_calls * _EST_OUTPUT_TOKENS / 1e6 * price_out
    return first + rest + user + out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene_reasoning_dir",
                    default="pref_pairs/results/fixed_reasoning/scene_reasoning")
    ap.add_argument("--out_path", default="dpo_pairs/results/perturbations_v2/perturbations.jsonl")
    ap.add_argument("--backend", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--model", default=None,
                    help="default: claude-fable-5 (anthropic) / gpt-4o (openai)")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--confirm", action="store_true",
                    help="Actually run the paid batch. Without this flag the script only "
                         "prints the job count + cost estimate (cost-approval discipline).")
    ap.add_argument("--fingerprint_check", metavar="JSONL",
                    help="Run the constant-direction fingerprint gate on an existing "
                         "perturbations JSONL and exit (no API cost).")
    args = ap.parse_args()

    if args.fingerprint_check:
        verdict = fingerprint_check(args.fingerprint_check)
        print(json.dumps(verdict, indent=2))
        return

    ground_truth_traces = extract_ground_truth_traces(args.scene_reasoning_dir)
    if args.max_scenes is not None:
        ground_truth_traces = ground_truth_traces[: args.max_scenes]

    model = args.model or ("gpt-4o" if args.backend == "openai" else "claude-fable-5")
    n_jobs = sum(len(select_targets(gt["trace"])) for gt in ground_truth_traces)
    cost = estimate_cost(n_jobs, model)
    logger.info("%d scenes -> %d claim-targeted jobs, estimated cost $%.2f on %s",
                len(ground_truth_traces), n_jobs, cost, model)

    if not args.confirm:
        logger.info("DRY RUN (no --confirm): no API calls made. Re-run with --confirm "
                    "after the cost above is approved.")
        return

    if args.backend == "openai":
        load_openai_api_key()
        from openai import OpenAI

        client = OpenAI()
    else:
        load_api_key()
        import anthropic

        client = anthropic.Anthropic()
    results, failures = generate_all_perturbations_v2(client, ground_truth_traces, model=model)
    out_path = write_perturbations_jsonl(results, args.out_path)
    logger.info("wrote %d perturbations to %s (%d failed after retry)",
                len(results), out_path, len(failures))


if __name__ == "__main__":
    main()
