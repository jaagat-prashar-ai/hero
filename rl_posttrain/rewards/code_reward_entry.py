# SPDX-License-Identifier: Apache-2.0
"""
code_reward_entry.py -- the code-as-a-reward GRPO integration, in ONE file:
both the cosmos-rl entry script (run.py launches this for reward_mode="code")
and the reward implementation it wires in.

Structure, top to bottom:

  1. REWARD HALF -- compute_reward / compute_reward_batch with the exact
     signature + (reward, reward_dict) contract of the vendored recipe's
     aggregated_reward_with_reasoning, derived from our validated
     aggregated_reward_llm_judge with exactly ONE component swapped: the
     reasoning score comes from code_as_a_reward's deterministic claim
     verification (parse the rollout's CoC into typed claims -> verify
     commitments against the trajectory that SAME rollout produced and
     perceptual claims against the clip's obstacle.offline tracks ->
     precision-over-decided-claims scalar in [0, 1]) instead of an
     Anthropic-API judge. No network, no API key, no thread pool -- the
     whole score is local CPU work, so the NCCL-starvation problems that
     shaped the judge reward (watchdog timeout, group batching for latency
     hiding) don't apply; group_reward_calculation stays enabled in the
     shared TOML and is simply cheap here.

     Everything around the reasoning component is kept IDENTICAL to
     aggregated_reward_llm_judge: trajectory decode + ADE, comfort, the
     mixing formula and its TOML weight keys under
     [custom.alpamayo.reward], the ade_threshold = 3.0 /
     reasoning_threshold = -0.4 gates, and the graded failure band -- so
     a code-reward run differs from the running llm-judge experiments in
     the reasoning signal ONLY. (Both entries deliberately mirror the
     vendored reasoning term, which was inverted -- full weight at
     barely-passing, zero at perfect; fixed 2026-07-30, see BUGS.md.)

     Score mapping: TraceReward.reward r in [0, 1] -> reasoning_score
     r - 1.0 in [-1, 0], the same scale the judge's normalize_score and
     the vendored grader's `sigmoid - 1.0` land on. Before the coverage
     blend below, reasoning_threshold = -0.4 meant "at least 60% of
     decided claims verified"; under the blend the bar is
     coverage-dependent (r > 0.8 - 0.2/decided_fraction).

     Abstention: r is None when NOTHING in the trace was decided (all
     claims hit missing ground truth -- Phase 0's ~27% unverifiable
     claim mass; the canonical shape is "keep a safe distance from the
     lead vehicle", where keep_distance and the 'lead' attribute both
     abstain). Punishing that would grade the model on OUR data gaps,
     so precision is coverage-blended toward a neutral prior:
     r_eff = decided_fraction * r + (1 - decided_fraction) * 0.8, and a
     fully undecided trace lands exactly on the neutral -0.2 (minus the
     unparsed penalty). The blend, not a fixed fallback, is what keeps
     the ordering honest -- a fixed neutral passed the gate while a
     half-verified trace failed it, making "say only unverifiable
     things" strictly beat "be half right". `code_decided_fraction` and
     `code_undecided_cnt` are logged so W&B shows how much of the reward
     rides on the prior.

  2. ENTRY HALF -- mirrors llm_judge_entry.py (env contract, vLLM
     registration, ModelSpec components, hydra overrides), plus one piece
     the judge never needed: _SceneTaggedDataset, an AlpamayoCosmosDataset
     subclass whose get_reference_answer also stamps the sample's
     scene_id (f"{clip_id}_{t0_us}", the exact naming every
     code_as_a_reward artifact carries -- see perceptual_verifier.
     split_scene_id) and the future-waypoint rate into the reference
     dict. The vendored reference only ships trajectory tensors, and the
     verifiers must know WHICH clip/window to check perceptual claims
     against. The vendored launcher hardcodes its dataset class, so
     main() replicates launch_alpamayo_model's few lines with the
     subclass swapped in -- the submodule itself stays untouched, per
     rl_posttrain convention.

Obstacle data at reward time: reward_mode="code" downloads the same PAI
reasoning subset as llm_judge, whose --labels already include
obstacle.offline (run.py). _load_scene reads a clip's tracks from the
per-clip parquets run.py pre-extracts into obstacles_by_clip/ at setup
(falling back to the recipe's dataset interface over the chunk zips when
the extraction is absent; alp_state is initialized in every cosmos-rl
process because each replica executes this entry), LRU-cached per clip. Degradation is deliberate and LOUD, never
silent: unknown label classes / class-inconsistent tracks are dropped with
a logged warning (the entity mapping only maps to known classes, so
dropping unknowns cannot flip any computable verdict), and if a clip's
scene cannot be loaded at all the trace is scored on commitment claims
only, with code_scene_available=0.0 logged per rollout so a dead
perceptual pipeline is visible in W&B instead of masquerading as training
signal.
"""

# ruff: noqa: E402

from __future__ import annotations

import functools
import logging
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# Reward code executes inside cosmos-rl worker processes whose sys.path is
# the recipe venv's -- our repo modules aren't installed there, so resolve
# them relative to this file (same pattern as aggregated_reward_llm_judge).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# cosmos_rl's logger when available (workers), stdlib logging otherwise so
# the module stays importable -- and unit-testable -- in the project venv.
try:
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stall diagnosis (run a1npli, 2026-07-27).
#
# That run spent ~16h of its 16h53m parked in ~1h stalls -- 19 events, 35
# optimizer steps total -- and still exited 0, because run.py raises
# COSMOS_NCCL_TIMEOUT_MS / COSMOS_ROLLOUT_CMD_WAIT_TIMEOUT to 1h for this
# reward mode, so each stall is absorbed instead of aborting. The stall was
# attributed to _load_scene's cold obstacle.offline parse and "fixed" twice
# (COSMOS_* bump, then e901539's per-clip pre-extraction), but that parse
# measures ~0.5s end-to-end against the real dataset: a 53.9 MB chunk zip
# opens in 0.003s, one member reads in 0.079s, and from_dataframe takes
# 0.01-0.40s. So the hang is real but still UNNAMED, and no amount of reading
# will name it -- three attributions have already been wrong.
#
# _StackSampler dumps every thread's stack on a fixed interval. A stall of any
# length therefore leaves the SAME frame repeated across consecutive dumps,
# which points at a file and line instead of inviting a fourth guess. It runs
# in every replica process (main() starts it), covering the rollout and
# dataloader paths too -- the hang need not be in the reward at all.
#
# Output goes through cosmos-rl's logger into this replica's log file on
# node-local /mnt/work, which run.py's _CosmosLogTailer ships to captured
# stdout; that forwarding only reaches OCI now that run.py configures its
# logger. Cost is one wakeup per interval, so it is safe to leave enabled.
class _StackSampler(threading.Thread):
    def __init__(self, interval_s: float = 300.0):
        super().__init__(daemon=True, name="code-reward-stack-sampler")
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                names = {t.ident: t.name for t in threading.enumerate()}
                parts = [
                    f"--- thread {names.get(tid, '?')} ({tid}) ---\n"
                    + "".join(traceback.format_stack(frame))
                    for tid, frame in sys._current_frames().items()
                ]
                # WARNING, not INFO: cosmos-rl's own level is configurable and
                # a sample that gets filtered out is worthless.
                logger.warning("[code_reward] stack sample:\n%s", "\n".join(parts))
            except Exception:
                logger.exception("[code_reward] stack sampler failed")

    def stop(self) -> None:
        self._stop_event.set()


def _timed(fn):
    """Logs wall time for fn. Applied UNDER functools.lru_cache so only cache
    MISSES are timed -- the cold path is the one under suspicion."""

    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        _t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            logger.warning(
                "[code_reward] %s%r took %.2fs (cold)", fn.__name__, args, time.perf_counter() - _t0
            )

    return _wrap


# ---------------------------------------------------------------------------
# Reward half. _get_reward_cfg / _graded_failure_reward are verbatim from
# aggregated_reward_llm_judge (see its docstrings for the full rationale,
# incl. why graded failure beats the vendored flat -1.0 under GRPO's
# within-group advantage normalization).
# ---------------------------------------------------------------------------

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "reasoning_weight",
]


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward].
    Verbatim from the vendored aggregated_reward_with_reasoning."""
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "Reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_REWARD_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_REWARD_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing key(s) in [custom.alpamayo.reward]: {missing}")

    return {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}


_FAILURE_BAND_WIDTH = 0.5


def _graded_failure_reward(
    l2_dist: float,
    reasoning_score: float,
    *,
    ade_threshold: float,
    reasoning_threshold: float,
    cot_decoded: bool,
) -> float:
    """Verbatim from aggregated_reward_llm_judge: gate-failing rollouts land
    in [-1.0, -0.5], ordered by how close each gated quantity came, so
    all-fail GRPO groups still carry advantage variance; a rollout with no
    decoded CoC stays at the flat -1.0."""
    if not cot_decoded:
        return -1.0
    l2_closeness = min(1.0, ade_threshold / l2_dist) if l2_dist > 0 else 1.0
    reasoning_closeness = min(
        1.0, max(0.0, (reasoning_score + 1.0) / (reasoning_threshold + 1.0))
    )
    return -1.0 + _FAILURE_BAND_WIDTH * 0.5 * (l2_closeness + reasoning_closeness)


# Neutral anchor of the coverage blend in _score_cot (there, 1.0 + this =
# the neutral prior r=0.8 that decided_fraction interpolates against), and
# exactly where a fully-undecided trace lands: -0.2 is the midpoint of the
# gate-passing band (-0.4, 0], because our missing ground truth must not
# read as model unfaithfulness.
_UNDECIDED_REASONING_SCORE = -0.2


@functools.lru_cache(maxsize=256)
@_timed
def _load_scene(clip_id: str):
    """One clip's SceneObstacles from the locally-downloaded PAI labels, or
    None when unavailable (callers then score commitments only).

    Reads through the recipe's own dataset interface -- the training
    dataloader's PAIDataset already holds a PhysicalAIAVDatasetLocalInterface
    over ALPAMAYO_PAI_REASONING_LOCAL_DIR, so obstacle.offline resolves to
    an on-disk chunk file, never HF. LRU-cached: a run touches each clip
    ~n_generation x epochs times, and the parse is the expensive part.

    Tolerated-and-logged (NOT raised): label classes outside
    OBSTACLE_LABEL_CLASSES and tracks with inconsistent classes are dropped
    -- the fixture-derived vocabulary may be incomplete against the full
    dataset, ENTITY_TO_CLASSES only maps to known classes so dropped
    unknowns can't flip any computable verdict, and killing a multi-hour
    GPU run over one unmapped truck subclass is the wrong trade. Everything
    else (missing feature, unexpected shape) degrades to None with a
    logged exception; the per-rollout code_scene_available metric keeps the
    degradation visible."""
    import pandas as pd

    from code_as_a_reward.obstacle_tracks import OBSTACLE_LABEL_CLASSES, SceneObstacles

    try:
        raw = None
        local_dir = os.environ.get("ALPAMAYO_PAI_REASONING_LOCAL_DIR")
        if local_dir:
            per_clip = Path(local_dir) / "obstacles_by_clip" / f"{clip_id}.obstacle.offline.parquet"
            if per_clip.exists():
                raw = pd.read_parquet(per_clip)
        if raw is None:
            # Fallback: recipe interface over the chunk zips (slow first
            # touch per chunk -- kept so a missing extraction degrades
            # instead of dying; run.py's _extract_obstacles_by_clip is what
            # populates obstacles_by_clip/ at setup).
            import alpamayo1_x_rl.state as alp_state

            avdi = alp_state.get_dataloaders()["train"].dataset.avdi
            raw = avdi.get_clip_feature(clip_id, "obstacle.offline")
        # get_clip_feature returns a DataFrame for parquet-chunk features and
        # a {filename: DataFrame} dict for zip-chunk features it has no
        # dedicated reader for -- accept both layouts.
        if isinstance(raw, dict):
            raw = next(
                (v for k, v in raw.items() if "obstacle" in str(k) and isinstance(v, pd.DataFrame)),
                None,
            )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            logger.error(
                f"[code_reward] clip {clip_id}: no obstacle.offline data "
                f"(got {type(raw).__name__}) -- scoring commitments only"
            )
            return None
        df = raw.reset_index(drop=True)

        unknown = set(df["label_class"].unique()) - set(OBSTACLE_LABEL_CLASSES)
        if unknown:
            n_before = len(df)
            df = df[~df["label_class"].isin(unknown)]
            logger.warning(
                f"[code_reward] clip {clip_id}: dropped {n_before - len(df)} rows with "
                f"unknown label classes {sorted(unknown)} (vocabulary candidates for "
                "OBSTACLE_LABEL_CLASSES / ENTITY_TO_CLASSES)"
            )
        class_counts = df.groupby("track_id")["label_class"].nunique()
        bad_tracks = class_counts[class_counts > 1].index
        if len(bad_tracks) > 0:
            df = df[~df["track_id"].isin(bad_tracks)]
            logger.warning(
                f"[code_reward] clip {clip_id}: dropped {len(bad_tracks)} tracks with "
                "inconsistent label classes"
            )
        if df.empty:
            logger.error(f"[code_reward] clip {clip_id}: no usable obstacle rows after filtering")
            return None
        return SceneObstacles.from_dataframe(df, clip_id)
    except Exception:
        logger.exception(
            f"[code_reward] clip {clip_id}: obstacle scene load failed -- scoring commitments only"
        )
        return None


def _score_cot(
    pred_cot: str,
    pred_xyz_np,  # (T, 3) float ndarray, ego frame at t0 -- decode_rollout_trajectory's output
    scene_id: str,
    rollout_id: int,
    hz: float,
    scene,  # SceneObstacles | None
) -> tuple[float, dict[str, float]]:
    """Deterministic reasoning score for one rollout's decoded CoC, in
    [-1, 0], plus the audit metrics logged into its reward_dict.

    Full path (scene available): parse_coc_trace -> extract_features over
    the rollout's own predicted waypoints -> score_trace, whose TraceReward
    is precision over DECIDED claims with abstains excluded and an
    unparsed-text penalty (see trace_reward.py for the semantics).
    Degraded path (scene is None): commitment claims verified against the
    trajectory only; perceptual/causal claims are untestable without the
    scene, which is exactly an all-abstain on that half."""
    from code_as_a_reward.coc_claim_parser import parse_coc_trace
    from code_as_a_reward.commitment_verifier import Verdict, verify_trace_commitments
    from code_as_a_reward.trace_reward import RewardConfig, score_trace
    from pref_pairs.trajectory_features import extract_features

    trace = parse_coc_trace(pred_cot, scene_id=scene_id, rollout_id=rollout_id)
    features = extract_features(pred_xyz_np, hz=hz, scene_id=scene_id, rollout_id=rollout_id)
    horizon_us = int(round(len(pred_xyz_np) / hz * 1e6))

    if scene is not None:
        tr = score_trace(trace, features, scene, horizon_us=horizon_us).reward
        r = tr.reward
        decided_fraction = tr.decided_fraction
        n_fail = float(sum(tr.n_fail.values()))
        atomic_precision = tr.atomic_precision
        unparsed_fraction = tr.unparsed_char_fraction
    else:
        verdicts = verify_trace_commitments(trace, features)
        n_pass = sum(v.verdict is Verdict.PASS for v in verdicts)
        n_fail = float(sum(v.verdict is Verdict.FAIL for v in verdicts))
        decided = n_pass + int(n_fail)
        r = (n_pass / decided) if decided else None
        decided_fraction = (decided / len(verdicts)) if verdicts else 0.0
        atomic_precision = r
        unparsed_chars = sum(e - s for s, e in trace.unparsed_spans)
        unparsed_fraction = (unparsed_chars / len(trace.raw_text)) if trace.raw_text else 0.0

    # Coverage-scaled credit: precision r is blended toward the neutral prior
    # by decided_fraction, so what a trace can gain or lose from verification
    # is proportional to how much of it was verifiable. This removes the
    # all-abstain cliff: previously a fully-undecided trace took a fixed -0.2
    # and PASSED the reasoning gate while a trace with half its claims
    # verified took -0.5 and FAILED it, so "say only unverifiable things"
    # (keep_distance + no-ground-truth entities, the most natural
    # lead-vehicle phrasing there is) strictly beat "be half right". A fully
    # undecided trace keeps exactly the old -0.2 -- now minus the unparsed
    # penalty score_trace could not apply (r=None short-circuits it there),
    # so unparseable text cannot ride the neutral prior either.
    neutral_r = 1.0 + _UNDECIDED_REASONING_SCORE
    if r is None:
        r_eff = max(0.0, neutral_r - RewardConfig().unparsed_penalty * unparsed_fraction)
    else:
        r_eff = decided_fraction * r + (1.0 - decided_fraction) * neutral_r
    reasoning_score = r_eff - 1.0

    aux = {
        "code_reward_raw": float(r) if r is not None else math.nan,
        "code_atomic_precision": float(atomic_precision)
        if atomic_precision is not None
        else math.nan,
        "code_decided_fraction": float(decided_fraction),
        "code_n_fail": n_fail,
        "code_scene_available": 0.0 if scene is None else 1.0,
        "code_undecided_cnt": 0.0 if r is not None else 1.0,
        "code_no_cot_cnt": 0.0,
    }
    return reasoning_score, aux


def compute_reward_batch(
    to_be_evaluated_list: list[str],
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[list[float], list[dict[str, float]]]:
    """Scores one prompt's whole rollout group -- the shape cosmos-rl uses
    with [train.train_policy].group_reward_calculation on (kept on in the
    shared TOML; here it's merely cheap, not load-bearing). Geometry decode
    and claim verification are both local, so the loop stays serial.

    Per-component semantics match aggregated_reward_llm_judge exactly
    except the reasoning score's source (module docstring, item 1)."""
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

    from code_as_a_reward.perceptual_verifier import split_scene_id

    w = _get_reward_cfg(config)
    gt_fut_xyz = reference["ego_future_xyz"]
    scene_id = reference.get("scene_id")
    # Fail loud, not neutral: without a scene_id every trace would silently
    # take the undecided fallback and the "code" reward would train on ADE
    # + a constant -- worse than crashing.
    assert isinstance(scene_id, str) and scene_id, (
        "reference carries no scene_id -- reward_mode='code' must run through this "
        "entry's _SceneTaggedDataset (vendored AlpamayoCosmosDataset does not stamp it)"
    )
    hz = float(reference.get("future_hz") or 10.0)
    clip_id, _t0_us = split_scene_id(scene_id)
    scene = _load_scene(clip_id)

    ade_threshold = 3.0
    reasoning_threshold = -0.4

    rewards: list[float] = []
    reward_dicts: list[dict[str, float]] = []
    for rollout_id, to_be_evaluated in enumerate(to_be_evaluated_list):
        predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
            to_be_evaluated,
            reference["ego_history_xyz"],
            reference["ego_history_rot"],
            tokenizer=tokenizer,
            traj_tokenizer=traj_tokenizer,
            model_config=model_config,
        )

        l2_dist = float(calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0]))

        comfort_dict_t = compute_comfort(
            predicted_fut_xyz[:, None, None, ...],
            predicted_fut_rot[:, None, None, ...],
        )
        comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t)) - 1.0

        pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
        pred_cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)
        pred_xyz_np = predicted_fut_xyz[0].detach().float().cpu().numpy()

        if pred_cot_decoded:
            reasoning_score, aux = _score_cot(
                pred_cot, pred_xyz_np, scene_id, rollout_id, hz, scene
            )
        else:
            # No decoded CoC: nothing to verify; flat -1.0 below (the
            # vendored missing-CoC penalty), same as the judge reward. The
            # aux still carries the FULL key set (NaN for the two precision
            # metrics, replaced with the group mean after the loop): the
            # vendored aggregator 0-fills missing keys into step means.
            reasoning_score, aux = -1.0, {
                "code_reward_raw": math.nan,
                "code_atomic_precision": math.nan,
                "code_decided_fraction": 0.0,
                "code_n_fail": 0.0,
                "code_scene_available": 0.0 if scene is None else 1.0,
                "code_undecided_cnt": 1.0,
                "code_no_cot_cnt": 1.0,
            }

        if pred_cot_decoded and reasoning_score > reasoning_threshold and l2_dist < ade_threshold:
            # DELIBERATE deviation from the vendored formula (and from every
            # run before 2026-07-30): the vendored reasoning term was
            # `w * (reasoning_score / reasoning_threshold)`, which DECREASES
            # in reasoning quality -- perfect reasoning (score 0) earned 0
            # while barely-passing (score -> -0.4) earned the full weight, so
            # inside the passing band the gradient pushed reasoning DOWN
            # toward the gate. `1 - score/threshold` is the same quantity
            # mirrored: 0 at the gate, full weight at perfect. The l2 and
            # comfort terms already pointed the right way; reasoning was the
            # only inverted component. See BUGS.md 2026-07-30 (2nd entry).
            final_reward = (
                -w["traj_l2_weight"] * (l2_dist / ade_threshold)
                + w["comfort_weight"] * comfort_score
                + w["reasoning_weight"] * (1.0 - reasoning_score / reasoning_threshold)
            )
        else:
            final_reward = _graded_failure_reward(
                l2_dist,
                reasoning_score,
                ade_threshold=ade_threshold,
                reasoning_threshold=reasoning_threshold,
                cot_decoded=pred_cot_decoded,
            )

        logger.debug(
            f"[code_reward] scene={scene_id} rollout={rollout_id} l2={l2_dist:.3f} "
            f"reasoning={reasoning_score:.3f} cot_decoded={pred_cot_decoded} "
            f"final={final_reward:.4f} aux={aux}"
        )
        rewards.append(float(final_reward))
        reward_dicts.append(
            {
                "traj_L2": l2_dist,
                "comfort_reward": comfort_score,
                "reasoning_score": float(reasoning_score),
                "reward": float(final_reward),
                **aux,
            }
        )

    # cosmos-rl's aggregate_report_data (utils/util.py) reduces these dicts
    # into per-step W&B metrics with np.mean(data.get(k, 0)): a single NaN
    # poisons the step's mean, and a missing key silently counts as 0 (which
    # for the two precision metrics means "everything failed"). So undecided
    # rollouts carry the group's decided-only mean instead of the NaN
    # sentinel -- neutral w.r.t. the step mean -- falling back to the r-scale
    # neutral prior when the whole group decided nothing. The _cnt counters
    # above (summed per step by the aggregator's suffix convention) keep the
    # fill-in rate visible in W&B.
    neutral_r = 1.0 + _UNDECIDED_REASONING_SCORE
    for key in ("code_reward_raw", "code_atomic_precision"):
        vals = [d[key] for d in reward_dicts if not math.isnan(d[key])]
        fill = (sum(vals) / len(vals)) if vals else neutral_r
        for d in reward_dicts:
            if math.isnan(d[key]):
                d[key] = fill

    return rewards, reward_dicts


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Single-rollout shape of the vendored contract; thin wrapper over
    compute_reward_batch with a batch of one."""
    rewards, reward_dicts = compute_reward_batch(
        [to_be_evaluated],
        reference,
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        config=config,
        model_config=model_config,
    )
    return rewards[0], reward_dicts[0]


# ---------------------------------------------------------------------------
# Entry half. Mirrors llm_judge_entry.py; executed (as __main__) by every
# cosmos-rl replica process, which is also what makes alp_state available
# wherever the reward fn later runs.
# ---------------------------------------------------------------------------


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    """Same shape as llm_judge_entry's reward fn; imports this module back by
    its canonical package name so the compute code resolves identically in
    processes that only ever saw this file as __main__."""
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    import alpamayo1_x_rl.state as alp_state
    from rl_posttrain.rewards.code_reward_entry import compute_reward, compute_reward_batch

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    fn = compute_reward_batch if isinstance(to_be_evaluated, list) else compute_reward
    return fn(
        to_be_evaluated,
        reference,
        tokenizer=alp_state.get_tokenizer(),
        traj_tokenizer=alp_state.get_traj_tokenizer(),
        config=config,
        model_config=alp_state.get_ckpt_cfg(),
    )


def _read_ckpt_path_from_toml() -> str:
    """[policy].model_name_or_path from the Cosmos TOML -- tiny copy of the
    vendored launcher's private helper rather than an import of it, so this
    file's only vendored dependencies are public entry points.

    The TOML path comes from `--config <path>` in sys.argv: cosmos-rl's
    controller/replica shells invoke the entry as
    `python entry.py --port ... --config /tmp/<patched>.toml` and do NOT
    export COSMOS_CONFIG into that process (canary
    alpamayo-rl-code-reward-b2wwha, 2026-07-23, died on exactly that
    KeyError -- COSMOS_CONFIG is run.py's head-node convention, kept here
    only as a fallback)."""
    import tomllib

    toml_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            toml_path = sys.argv[i + 1]
            break
    toml_path = toml_path or os.environ.get("COSMOS_CONFIG")
    if not toml_path:
        raise RuntimeError(
            "No --config <path> in sys.argv and no COSMOS_CONFIG env var -- "
            "the Cosmos orchestrator should pass --config automatically."
        )
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["policy"]["model_name_or_path"]


def main() -> None:
    os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
    os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

    # Names the a1npli stall (see _StackSampler). 0 disables.
    _sample_s = float(os.getenv("CODE_REWARD_STACK_SAMPLE_S", "300"))
    if _sample_s > 0:
        _StackSampler(interval_s=_sample_s).start()
        logger.warning("[code_reward] stack sampler armed, every %.0fs", _sample_s)

    # ── The a1npli stall, root-caused (canary eivn91, 2026-07-29) ────────────
    # Stack samples caught both sides of it: the rollout parked in
    # post_rollout_completion -> make_request_with_retry -> time.sleep(jitter)
    # (client.py:570, network_util.py:135) while all 4 policy ranks spun in
    # grouped_send -> nccl_group_end -> time.sleep(0.001) (rl_worker.py:410,
    # pynccl.py:228) waiting on the weight-sync collective for a rollout that
    # could not report. One failed localhost HTTP POST idles all 5 GPUs.
    #
    # The ~51 min quantum was never a computation -- it is cosmos-rl's stock
    # retry ladder running to exhaustion. CosmosHttpRetryConfig defaults are
    # max_retries=60, retries_per_delay=5, initial_delay=1s, backoff x2 capped
    # at 60s, and each failed attempt sleeps (1+random())*delay, so the
    # expected total is 3172s = 52.9 min (jitter range 35.2-70.5 min). Run
    # a1npli's 14 slow steps measured 50.0-54.2 min, dead centre, in 1x/2x/3x
    # multiples -- i.e. the ladder exhausting once, twice or three times.
    # train/iteration_time stayed flat at 22s throughout because the stall is
    # entirely OUTSIDE the training step.
    #
    # A ~53 min blocking retry is defensible for a background job riding out a
    # network blip; it is indefensible on the critical path of a synchronous
    # distributed step. Capping it does two things at once: the stall drops to
    # ~1.9 min, AND post_rollout_completion's own `except` (client.py:578)
    # finally runs, logging the underlying exception at ERROR -- the one fact
    # still missing, since network_util logs per-attempt failures at DEBUG
    # where run.py's tailer truncation was dropping them.
    #
    # Read at ApiClient construction time (client.py:86), NOT at import, so
    # patching here -- before launch_worker builds the client -- takes effect.
    # 20 keeps a real ~2 min retry budget so a transient blip still recovers.
    try:
        from cosmos_rl.utils import constant as _cosmos_constant

        _prev = _cosmos_constant.COSMOS_HTTP_RETRY_CONFIG.max_retries
        _cap = int(os.getenv("CODE_REWARD_HTTP_MAX_RETRIES", "20"))
        _cosmos_constant.COSMOS_HTTP_RETRY_CONFIG.max_retries = _cap
        logger.warning(
            "[code_reward] COSMOS_HTTP_RETRY_CONFIG.max_retries %d -> %d "
            "(caps the rollout-report stall at ~1.9 min instead of ~52.9 min)",
            _prev,
            _cap,
        )
    except Exception:
        logger.exception("[code_reward] could not cap COSMOS_HTTP_RETRY_CONFIG.max_retries")

    pai_reasoning_local_dir = os.getenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR")
    if not pai_reasoning_local_dir:
        raise RuntimeError(
            "Missing required env var ALPAMAYO_PAI_REASONING_LOCAL_DIR "
            "(expected PAI reasoning dataset root, e.g. /path/to/PAI_Reasoning_mini)."
        )
    # Deliberately NO Anthropic-credential check: unlike llm_judge, this
    # reward never leaves the node.

    # vLLM registration (verbatim from the vendored reasoning entry).
    try:
        from vllm import ModelRegistry as vllm_model_registry

        from alpamayo1_x_rl.models.reasoning_vla.vllm_wrapper import ReasoningVLAModelForVLLM

        vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
    except Exception as e:
        logger.warning(f"Failed to register ReasoningVLA model with vLLM: {e}")

    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from cosmos_rl.policy.model.base import ModelRegistry

    import alpamayo1_x_rl.state as alp_state
    from alpamayo1_x_rl.base_dataset import AlpamayoCosmosDataset
    from alpamayo1_x_rl.models.reasoning_vla.cosmos_wrapper import ReasoningVLACosmos
    from alpamayo1_x_rl.models.reasoning_vla.data_packer import RVLADataPacker
    from alpamayo1_x_rl.models.reasoning_vla.rollout import ReasoningVLAVllmRollout  # noqa: F401 (Cosmos registry)
    from alpamayo1_x_rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401 (Cosmos registry)
    from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper

    from rl_posttrain.rewards.scene_reference_dataset import SceneReferenceDataset

    class _SceneTaggedDataset(SceneReferenceDataset):
        """Vendored dataset + two reference keys the code reward needs --
        scene_id (f"{clip_id}_{t0_us}") and future_hz -- plus, via the
        SceneReferenceDataset base, the judge's scene payload
        (scene_frame_jpeg / scene_cam_intr / scene_cam_extr), which the code
        reward uses only for W&B overlay visualization. Unlike the judge
        (where a missing frame must fail the run -- it would silently change
        the experiment), the overlay here is cosmetic, so scene-payload
        failures degrade to the plain reference instead of raising.

        scene_id/future_hz are derived the same way the underlying
        PAIDataset's __getitem__ derives them, from the same dataset
        attributes, so the tag always names the window the sample was
        actually built from."""

        def get_reference_answer(self, idx: int) -> dict[str, Any]:
            try:
                ref = super().get_reference_answer(idx)
            except Exception:
                logger.exception(
                    f"[code_reward] scene payload failed for idx={idx}; "
                    "continuing without W&B overlay keys"
                )
                ref = AlpamayoCosmosDataset.get_reference_answer(self, idx)
            if not ref:
                return ref
            ds = self.dataset
            try:
                clip_id = str(ds.clip_ids[idx])
                t0_us = int(
                    ds.DEFAULT_T0_US
                    if ds.use_default_keyframe
                    else ds.avdi.get_clip_key_frame(clip_id)
                )
                ref["scene_id"] = f"{clip_id}_{t0_us}"
                ref["future_hz"] = 1.0 / float(ds.time_step)
            except Exception:
                # Leave scene_id unset and let the reward's assert fail the
                # run with a message naming this dataset, rather than
                # half-scoring with a wrong window.
                logger.exception(f"[code_reward] failed to stamp scene_id for idx={idx}")
            return ref

    # The remainder replicates the vendored launch_alpamayo_model verbatim,
    # with _SceneTaggedDataset in place of its hardcoded AlpamayoCosmosDataset
    # (the ModelSpec/launcher pair offers no dataset hook, and the submodule
    # is never edited).
    ckpt_path = _read_ckpt_path_from_toml()
    alp_state.init_once(
        ckpt_path,
        hydra_config_path="hydra_configs",
        hydra_config_name="alpamayo1_5_rvla_rl_pai",
        overrides=[
            f"data.train.dataset.local_dir={pai_reasoning_local_dir}",
            "data.train.dataset.clip_index_metadata=clip_index_reasoning_mini.parquet",
            "data.train.dataset.features_metadata=features.csv",
            "data.train.dataset.use_default_keyframe=False",
            "data.train.dataset.reasoning_metadata=reasoning/ood_reasoning.parquet",
        ],
    )

    ModelRegistry.register_model(
        ReasoningVLACosmos,
        ReasoningVLAWeightMapper,
        data_packer_cls=RVLADataPacker,
    )

    launch_worker(
        dataset=lambda config: _SceneTaggedDataset(split="train"),
        data_packer=RVLADataPacker(),
        reward_fns=[_reasoning_vla_reward_fn],
        val_dataset=lambda config: _SceneTaggedDataset(split="val"),
        val_data_packer=RVLADataPacker(),
        val_reward_fns=[_reasoning_vla_reward_fn],
    )


if __name__ == "__main__":
    main()
