# SPDX-License-Identifier: Apache-2.0
"""Restricted execution for generated reward functions.

Generated code runs inside the reward path eventually, so the contract is
enforced here, once, for both the gate and any future training-time use:
no imports, no dunder access, no non-canonical claim-attribute predicates
or dossier-only literals, a whitelisted builtin set, a wall-clock timeout,
and a return value clamped to [0, 1]. A bad function can waste a clip; it
must never take down a run.
"""

from __future__ import annotations

import ast
import re
import threading

import numpy as np

from code_as_a_reward.coc_claim_parser import ENTITY_PATTERNS, MANEUVER_PATTERNS, STATE_PATTERNS

class RewardFnError(Exception):
    """Any way a generated function can be invalid or misbehave."""


# Canonical keys the parser can actually emit onto claim attributes -- a
# generated predicate comparing against anything else (e.g. `claim.entity ==
# "trailer"`, a dossier noun the parser never produces) silently scores 0 on
# every real claim, GT included (BUGS.md 2026-08-04, pduuqq analysis).
_CANONICAL_KEYS: dict[str, frozenset[str]] = {
    "entity": frozenset(key for key, _ in ENTITY_PATTERNS),
    "maneuver": frozenset(key for key, *_ in MANEUVER_PATTERNS),
    "state": frozenset(key for key, _ in STATE_PATTERNS),
    "direction": frozenset({"left", "right"}),
    "speed_profile": frozenset({"accelerate", "decelerate", "maintain", "adapt"}),
}

# "Track 32"-style dossier track ids are obstacle_tracks.py bookkeeping, never
# present in a rollout's own CoC text -- a predicate hardcoding one (e.g.
# `"Track 32" in claim.text`) can only ever fire on the clip it was written
# against, defeating the point of a reward function meant to score any
# rollout for that scene (same pduuqq failure mode).
_TRACK_LITERAL_RE = re.compile(r"\btrack\s+\d+\b", re.I)


def _string_const(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _check_canonical_vocab(tree: ast.AST) -> None:
    """Reject predicates that can't generalize beyond the clip they were written for.

    Scoped to actual COMPARISONS only (a claim attribute vs. a string
    literal) -- never to every string in the source. A docstring or comment
    that merely *mentions* a track id (narration, harmless) must not be
    confused with code that *compares against* one (the real bug: a
    predicate like `"Track 32" in claim.text` that can only ever match the
    one clip it was written against, since no rollout's own CoC text will
    ever contain a dossier track id)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for left, right in zip(operands, operands[1:]):
            for attr_node, other in ((left, right), (right, left)):
                if not isinstance(attr_node, ast.Attribute):
                    continue
                # `attr in ('a', 'b')` membership: lint every element, not
                # just bare string constants -- the taught idioms are
                # tuple-membership predicates, and a typo'd key inside a
                # set must fail as loudly as one in an == comparison.
                elements = (
                    other.elts
                    if isinstance(other, (ast.Tuple, ast.List, ast.Set))
                    else [other]
                )
                for element in elements:
                    literal = _string_const(element)
                    if literal is None:
                        continue
                    if _TRACK_LITERAL_RE.search(literal):
                        raise RewardFnError(
                            f"comparing .{attr_node.attr} against dossier-only literal {literal!r}; "
                            "no rollout's own CoC text will ever contain a track id"
                        )
                    canonical = _CANONICAL_KEYS.get(attr_node.attr)
                    if canonical is not None and literal not in canonical:
                        raise RewardFnError(
                            f"non-canonical .{attr_node.attr} value {literal!r}; "
                            f"the parser only ever emits: {sorted(canonical)}"
                        )


def _check_no_float_equality(tree: ast.AST) -> None:
    """Reject exact float-literal equality/inequality checks (`x == 0.0`,
    `x != 4.2`). Real trajectory-derived quantities (speed, offsets, windowed
    extrema) are noisy floats that never hit an exact literal -- this
    silently zeros a component on every real rollout, GT included
    (02fd6a8f's `speed_window.min() == 0.0`: dead across all 12 real
    rollouts in the udqm59 smoke despite every one of them physically
    stopping to within 0.05 m/s). Int/bool/string equality (counts,
    canonical keys) is unaffected -- only a float-typed literal operand
    trips this. The system prompt already asks for this in prose (see
    generate.py's "never require... exact equality"); that alone did not
    stop gpt-4o writing it twice on the same clip, so it's enforced here."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            continue
        for operand in (node.left, *node.comparators):
            if isinstance(operand, ast.Constant) and isinstance(operand.value, float):
                raise RewardFnError(
                    f"exact equality against float literal {operand.value!r} -- real "
                    "trajectory data is noisy and will never hit an exact value; use "
                    "`>=`/`<=` with generous one-sided margin instead (a two-sided "
                    "`abs(x - target) <= eps` band only when the scene genuinely needs one)"
                )


_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None", "ValueError", "ZeroDivisionError", "Exception",
    )
    if (isinstance(__builtins__, dict) and name in __builtins__)
    or (not isinstance(__builtins__, dict) and hasattr(__builtins__, name))
}


def window(values, dt_s: float, t0: float, t1: float) -> np.ndarray:
    """Slice a per-waypoint series to the [t0, t1] second window.

    Provided to generated code so scene timing ("between t=2.0s and t=4.2s")
    maps onto TrajectoryFeatures' fixed-rate lists without index arithmetic.
    """
    arr = np.asarray(values, dtype=np.float64)
    i0 = max(0, int(t0 / dt_s))
    i1 = min(len(arr), int(np.ceil(t1 / dt_s)) + 1)
    return arr[i0:i1]


def _check_ast(source: str) -> None:
    """Reject imports, dunder access, and non-canonical claim predicates before anything executes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RewardFnError("generated code must not import anything")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RewardFnError(f"dunder attribute access forbidden: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RewardFnError(f"dunder name forbidden: {node.id}")
    _check_canonical_vocab(tree)
    _check_no_float_equality(tree)


def compile_reward_module(source: str, require_components: bool = False):
    """Compile generated source; return (reward_fn, components_fn_or_None).

    `components(claims, traj) -> dict[str, float]` is the function's own
    score decomposition (reward == clamp(sum(components.values()))). The
    gate uses it for over-budget detection that survives internal clamping
    and for per-component retry feedback. require_components=True enforces
    its presence (generation-time contract); the gate itself stays lenient
    so it can still verify legacy/selection-time functions.

    Raises RewardFnError on syntax errors, banned constructs, or a missing
    `reward` definition.
    """
    try:
        _check_ast(source)
    except SyntaxError as e:
        raise RewardFnError(f"syntax error: {e}") from e
    namespace = {"__builtins__": _SAFE_BUILTINS, "np": np, "window": window}
    try:
        exec(compile(source, "<generated_reward>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        raise RewardFnError(f"execution of module body failed: {e}") from e
    fn = namespace.get("reward")
    if not callable(fn):
        raise RewardFnError("source must define a callable named `reward`")
    components = namespace.get("components")
    if not callable(components):
        if require_components:
            raise RewardFnError(
                "source must ALSO define `def components(claims, traj) -> dict`"
                " returning the named component contributions whose clamped sum"
                " is exactly what `reward` returns"
            )
        components = None
    return fn, components


def compile_reward_fn(source: str):
    """Compile generated source into a callable `reward(claims, traj)`."""
    return compile_reward_module(source)[0]


class _Timeout:
    """Thread-based wall-clock guard -- works from any thread, unlike the
    prior signal.SIGALRM approach, which only works on the interpreter's
    MAIN thread and raised "ValueError: signal only works in main thread of
    the main interpreter" on every single call once this ran under Ray
    Train's TorchTrainer (which executes training_fn on a worker thread,
    not main) -- confirmed against a real cluster run, every rollout in
    every group failing identically (see BUGS.md).

    A raw daemon thread, NOT concurrent.futures.ThreadPoolExecutor: an
    executor's workers are non-daemon, so a genuinely hung function (e.g.
    the deliberate infinite loop in clipgen_test.py's timeout test) leaks a
    thread that blocks the WHOLE PROCESS from exiting -- confirmed directly
    (the test suite hung indefinitely with the executor version). A daemon
    thread leaks the same way but never blocks interpreter shutdown."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def run(self, fn, *args):
        box: dict = {}

        def target():
            try:
                box["value"] = fn(*args)
            except BaseException as e:  # noqa: BLE001 -- re-raised on the caller's thread below
                box["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.seconds)
        if thread.is_alive():
            raise RewardFnError("reward function timed out")
        if "error" in box:
            raise box["error"]
        return box["value"]


def run_reward_fn(fn, claims, traj, timeout_s: float = 2.0, raw: bool = False) -> float:
    """Execute a compiled reward function; return its score clamped to [0, 1].

    Raises RewardFnError on exception, timeout, or a non-finite/non-numeric
    return -- the caller decides whether that means gate failure (prototype)
    or neutral abstain (training path).

    raw=True returns the pre-clamp value instead: the gate uses it to detect
    over-budget component sums (a positive returning >1.0 means the [0,1]
    clamp is absorbing weight, so corruptions that keep the over-allocated
    components saturate to the same score as the positive -- 8xvbos, 8/15
    clips).
    """
    try:
        value = _Timeout(timeout_s).run(fn, claims, traj)
    except RewardFnError:
        raise
    except Exception as e:
        raise RewardFnError(f"reward function raised: {type(e).__name__}: {e}") from e
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise RewardFnError(f"reward returned non-numeric {value!r}") from e
    if not np.isfinite(value):
        raise RewardFnError(f"reward returned non-finite {value!r}")
    return value if raw else min(1.0, max(0.0, value))


def run_components_fn(fn, claims, traj, timeout_s: float = 2.0) -> dict[str, float]:
    """Execute a compiled components function; return {name: float}.

    Same guard rails as run_reward_fn; raises RewardFnError on exception,
    timeout, non-dict return, or any non-finite/non-numeric value.
    """
    try:
        value = _Timeout(timeout_s).run(fn, claims, traj)
    except RewardFnError:
        raise
    except Exception as e:
        raise RewardFnError(f"components function raised: {type(e).__name__}: {e}") from e
    if not isinstance(value, dict):
        raise RewardFnError(f"components returned non-dict {type(value).__name__}")
    out: dict[str, float] = {}
    for k, v in value.items():
        try:
            v = float(v)
        except (TypeError, ValueError) as e:
            raise RewardFnError(f"components[{k!r}] is non-numeric {v!r}") from e
        if not np.isfinite(v):
            raise RewardFnError(f"components[{k!r}] is non-finite {v!r}")
        out[str(k)] = v
    return out
