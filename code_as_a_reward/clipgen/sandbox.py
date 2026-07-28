# SPDX-License-Identifier: Apache-2.0
"""Restricted execution for generated reward functions.

Generated code runs inside the reward path eventually, so the contract is
enforced here, once, for both the gate and any future training-time use:
no imports, no dunder access, a whitelisted builtin set, a wall-clock
timeout, and a return value clamped to [0, 1]. A bad function can waste a
clip; it must never take down a run.
"""

from __future__ import annotations

import ast
import signal

import numpy as np

class RewardFnError(Exception):
    """Any way a generated function can be invalid or misbehave."""


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
    """Reject imports and dunder access before anything executes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RewardFnError("generated code must not import anything")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RewardFnError(f"dunder attribute access forbidden: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RewardFnError(f"dunder name forbidden: {node.id}")


def compile_reward_fn(source: str):
    """Compile generated source into a callable `reward(claims, traj)`.

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
    return fn


class _Timeout:
    """SIGALRM-based wall-clock guard (POSIX main thread; fine for the
    prototype harness -- the training integration would run out-of-process)."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._raise)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise RewardFnError("reward function timed out")


def run_reward_fn(fn, claims, traj, timeout_s: float = 2.0) -> float:
    """Execute a compiled reward function; return its score clamped to [0, 1].

    Raises RewardFnError on exception, timeout, or a non-finite/non-numeric
    return -- the caller decides whether that means gate failure (prototype)
    or neutral abstain (training path).
    """
    try:
        with _Timeout(timeout_s):
            value = fn(claims, traj)
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
    return min(1.0, max(0.0, value))
