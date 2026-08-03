# SPDX-License-Identifier: Apache-2.0
"""
mine_pairs.py — Stage 3 of the dpo_pairs pipeline: turn dpo_pairs/run.py's
measurement rows into gated, ranked DPO preference pairs.

Offline and numpy-only on purpose (no torch, no model): reads the per-scene
JSON files written by run.py's S3 channel or dpo_pairs/fetch_from_logs.py
(identical shape), applies the noise-floor gate, and emits
dpo_pairs.jsonl + mining_report.json.

The statistics, in the order they're computed:

  scene_eps      p90 of the PAIRWISE ADE distribution among the clean
                 condition's N samples — the per-scene AR-pathway sampling
                 noise floor. Recomputed here from this run's own samples;
                 the diffusion-pathway floor in pref_pairs' variance report
                 does NOT transfer to the AR head (different sampler).
  cluster_eps    p90 over member scenes' scene_eps per event_cluster —
                 catches scenes whose own N draws happened to be tight.
  control_ade    median over sample_idx-paired ADE(clean_i, control_rawids_i)
                 — the re-tokenization machinery error (clean forces TEXT,
                 control forces the template's raw ids; see
                 ar_forced_rollout's TOKENIZATION SYMMETRY note).
  cross_ade      per perturbation: median over ALL (i, j) of
                 ADE(perturbed_j, clean_i) — full cross, robust to single
                 outlier draws. Chunk-level common random numbers do NOT
                 make samples pairwise comparable across conditions (AR
                 sampling diverges token-by-token), so the gate is
                 distributional, never per-sample-paired.

  GATE (all must hold for a pair to qualify):
    1. cross_ade > max(scene_eps, cluster_eps)        — beats sampling noise
    2. cross_ade > k_control * control_ade            — beats machinery error
    3. rejected side is well-formed + kinematically sane — a perturbation
       that makes the model emit garbage clears gates 1-2 trivially but
       teaches DPO nothing about reasoning→action coupling (plan risk R3)

  z-score        (cross_ade − mean(clean pairwise ADE)) / std(...) —
                 standardized effect size; rank_within_scene /
                 rank_within_cluster over qualifying pairs. The top spans
                 per scene, ranked by z, ARE the "maximal semantic
                 perturbation points" deliverable.

Pair construction: chosen = the clean sample nearest the clean medoid,
rejected = the perturbed sample nearest the perturbed medoid — the
representative of each induced mode, not a noise outlier. Completion token
ids are assembled with token_math.assemble_completion_ids from the row's own
token_constants, so this module never loads a tokenizer.

Semantic delta (MiniLM cosine distance between clean and perturbed text) is
computed when sentence-transformers is importable, else nulled with a
warning — it is QC metadata, not part of the gate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dpo_pairs.token_math import assemble_completion_ids

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "eps_percentile": 90.0,   # scene_eps / cluster_eps percentile
    "k_control": 3.0,         # cross_ade must exceed k * control_ade
    "min_wellformed_frac": 0.9,  # fraction of rejected-condition samples that hit traj_future_end
    "max_speed_mps": 45.0,    # kinematic sanity on the REJECTED medoid sample
    "max_accel_mps2": 12.0,   # (finite-difference at hz; generous physical bounds,
    "hz": 10.0,               #  NOT comfort bounds — rejected SHOULD drive badly, just not impossibly)
}


# ---------------------------------------------------------------------------
# Trajectory math (xy-only, same convention as masking/training/run.py's ade_m)
# ---------------------------------------------------------------------------

def ade(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-waypoint L2 in xy over the common horizon."""
    t = min(len(a), len(b))
    return float(np.linalg.norm(np.asarray(a)[:t, :2] - np.asarray(b)[:t, :2], axis=-1).mean())


def endpoint_delta(a: np.ndarray, b: np.ndarray) -> float:
    t = min(len(a), len(b))
    return float(np.linalg.norm(np.asarray(a)[t - 1, :2] - np.asarray(b)[t - 1, :2]))


def pairwise_ades(xyzs: list[np.ndarray]) -> np.ndarray:
    """All N*(N-1)/2 pairwise ADEs among one condition's samples."""
    out = [ade(xyzs[i], xyzs[j]) for i in range(len(xyzs)) for j in range(i + 1, len(xyzs))]
    return np.asarray(out, dtype=np.float64)


def cross_ades(xyzs_a: list[np.ndarray], xyzs_b: list[np.ndarray]) -> np.ndarray:
    """All |A|x|B| cross ADEs between two conditions' samples."""
    return np.asarray([ade(a, b) for a in xyzs_a for b in xyzs_b], dtype=np.float64)


def medoid_index(xyzs: list[np.ndarray]) -> int:
    """Index of the sample minimizing total ADE to all others — the
    representative of the condition's induced mode, not a noise outlier."""
    if len(xyzs) == 1:
        return 0
    costs = [sum(ade(xyzs[i], xyzs[j]) for j in range(len(xyzs)) if j != i)
             for i in range(len(xyzs))]
    return int(np.argmin(costs))


def kinematic_sanity(xyz: np.ndarray, hz: float, max_speed_mps: float, max_accel_mps2: float) -> bool:
    """Physically-possible check on ONE trajectory (finite differences).
    Deliberately generous bounds: the rejected side is SUPPOSED to drive
    badly — this only rejects physically impossible output (teleporting
    waypoints from degenerate token sequences)."""
    xy = np.asarray(xyz, dtype=np.float64)[:, :2]
    if len(xy) < 3:
        return False
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * hz
    a = np.abs(np.diff(v)) * hz
    return bool(v.max() <= max_speed_mps and a.max() <= max_accel_mps2)


def final_lateral_offset(xyz: np.ndarray) -> float:
    """y of the final waypoint — ego frame is x-forward / y-left (ISO 8855)
    anchored at t0, so this is the trajectory's terminal lateral offset."""
    return float(np.asarray(xyz)[-1, 1])


# ---------------------------------------------------------------------------
# Scene loading / condition indexing
# ---------------------------------------------------------------------------

def load_scene_rows(measure_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """{scene_id: rows} from a directory of per-scene JSON files (run.py's S3
    channel or fetch_from_logs.py output — identical shape)."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(measure_dir).glob("*.json")):
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("%s: unparseable, skipped", path)
            continue
        if rows:
            by_scene[path.stem] = rows
    return by_scene


def index_conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{condition: row} for the scene's kind=condition rows; duplicate
    conditions (requeue re-runs) keep the first occurrence."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("kind") != "condition":
            continue
        out.setdefault(row["condition"], row)
    return out


def _sample_xyzs(row: dict[str, Any]) -> list[np.ndarray]:
    return [np.asarray(s["xyz"], dtype=np.float64) for s in row["samples"]]


def _wellformed_frac(row: dict[str, Any]) -> float:
    samples = row["samples"]
    if not samples:
        return 0.0
    expected = row.get("token_constants", {}).get("tokens_per_future_traj")
    ok = 0
    for s in samples:
        hit = s.get("hit_traj_future_end", False)
        full = expected is None or s.get("n_traj_tokens") == expected
        ok += int(hit and full)
    return ok / len(samples)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def compute_scene_stats(conditions: dict[str, dict[str, Any]],
                        thresholds: dict[str, float]) -> dict[str, Any] | None:
    """Per-scene floor statistics. None if the scene lacks the clean or
    control condition (can happen for scenes in-flight at fetch time)."""
    clean = conditions.get("clean")
    control = conditions.get("control_rawids")
    if clean is None or control is None:
        return None
    clean_xyzs = _sample_xyzs(clean)
    control_xyzs = _sample_xyzs(control)
    if len(clean_xyzs) < 2:
        return None

    clean_pairwise = pairwise_ades(clean_xyzs)
    paired = [ade(a, b) for a, b in zip(clean_xyzs, control_xyzs)]
    return {
        "scene_eps": float(np.percentile(clean_pairwise, thresholds["eps_percentile"])),
        "clean_pairwise_mean": float(clean_pairwise.mean()),
        "clean_pairwise_std": float(clean_pairwise.std()),
        "control_ade": float(np.median(paired)),
        "clean_wellformed_frac": _wellformed_frac(clean),
        "clean_xyzs": clean_xyzs,
        "clean_medoid_idx": medoid_index(clean_xyzs),
    }


def evaluate_perturbation(
    pert_row: dict[str, Any],
    scene_stats: dict[str, Any],
    cluster_eps: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Gate + effect metrics for one perturbed condition against its scene's
    clean condition. Returns a metrics dict with `qualifies` and the
    per-gate booleans (kept separately so the mining report can say WHY
    pairs died, not just how many)."""
    pert_xyzs = _sample_xyzs(pert_row)
    clean_xyzs = scene_stats["clean_xyzs"]

    cross = cross_ades(pert_xyzs, clean_xyzs)
    cross_ade = float(np.median(cross))
    scene_eps = scene_stats["scene_eps"]
    control_ade = scene_stats["control_ade"]

    pert_medoid = medoid_index(pert_xyzs)
    wellformed = _wellformed_frac(pert_row)
    sane = kinematic_sanity(
        pert_xyzs[pert_medoid], thresholds["hz"],
        thresholds["max_speed_mps"], thresholds["max_accel_mps2"],
    )

    std = scene_stats["clean_pairwise_std"]
    z = float((cross_ade - scene_stats["clean_pairwise_mean"]) / std) if std > 0 else None

    gate_noise = cross_ade > max(scene_eps, cluster_eps)
    gate_control = cross_ade > thresholds["k_control"] * control_ade
    gate_sane = wellformed >= thresholds["min_wellformed_frac"] and sane

    clean_medoid_xyz = clean_xyzs[scene_stats["clean_medoid_idx"]]
    pert_medoid_xyz = pert_xyzs[pert_medoid]
    return {
        "cross_ade_m": cross_ade,
        "scene_eps_m": scene_eps,
        "cluster_eps_m": cluster_eps,
        "control_ade_m": control_ade,
        "effect_ratio": cross_ade / max(scene_eps, cluster_eps) if max(scene_eps, cluster_eps) > 0 else None,
        "z_score": z,
        "endpoint_delta_m": endpoint_delta(pert_medoid_xyz, clean_medoid_xyz),
        "lateral_offset_delta_m": abs(
            final_lateral_offset(pert_medoid_xyz) - final_lateral_offset(clean_medoid_xyz)
        ),
        "rejected_wellformed_frac": wellformed,
        "gate_noise_floor": gate_noise,
        "gate_control": gate_control,
        "gate_kinematic_sanity": gate_sane,
        "qualifies": gate_noise and gate_control and gate_sane,
        "pert_medoid_idx": pert_medoid,
    }


def _completion(row: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    tc = row["token_constants"]
    completion_ids = assemble_completion_ids(
        row["coc_token_ids"], sample["traj_token_ids"], tc, tc["traj_token_start_idx"],
    )
    return {
        "coc_text": row["coc_text"],
        "coc_token_ids": row["coc_token_ids"],
        "traj_token_ids": sample["traj_token_ids"],  # offset-normalized [0, traj_vocab_size)
        "completion_token_ids": completion_ids,
        "xyz": sample["xyz"],
        "seed": sample["seed"],
        "sample_idx": sample["sample_idx"],
    }


def _semantic_deltas(pairs: list[dict[str, Any]]) -> None:
    """Cosine distance between clean and perturbed CoC embeddings, in place.
    QC metadata only (never part of the gate) — nulled with a warning when
    sentence-transformers isn't installed in this environment."""
    try:
        from pref_pairs.faithfulness_embedding_grader import embed_traces
    except Exception:
        logger.warning("sentence-transformers unavailable — semantic_delta_cos nulled")
        for p in pairs:
            p["metrics"]["semantic_delta_cos"] = None
        return
    texts: list[str] = []
    for p in pairs:
        texts.append(p["chosen"]["coc_text"])
        texts.append(p["rejected"]["coc_text"])
    emb = embed_traces(texts).numpy()
    for i, p in enumerate(pairs):
        a, b = emb[2 * i], emb[2 * i + 1]
        p["metrics"]["semantic_delta_cos"] = float(1.0 - np.dot(a, b))


def mine(
    by_scene: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float] | None = None,
    semantic_delta: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The full mining pass: (qualifying pairs sorted by z desc, report).
    The report carries every floor, per-gate death count, and the per-scene
    ranked perturbation-point table — including non-qualifying entries, so
    'maximal semantic perturbation points' is reportable even for scenes
    that produced no DPO pair."""
    thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Pass 1: per-scene stats, then per-cluster floors.
    scene_stats: dict[str, dict[str, Any]] = {}
    scene_conditions: dict[str, dict[str, dict[str, Any]]] = {}
    n_incomplete = 0
    for scene_id, rows in by_scene.items():
        conditions = index_conditions(rows)
        stats = compute_scene_stats(conditions, thresholds)
        if stats is None:
            n_incomplete += 1
            continue
        scene_stats[scene_id] = stats
        scene_conditions[scene_id] = conditions

    cluster_scene_eps: dict[str, list[float]] = {}
    scene_cluster: dict[str, str] = {}
    for scene_id, conditions in scene_conditions.items():
        cluster = conditions["clean"].get("event_cluster") or "?"
        scene_cluster[scene_id] = cluster
        cluster_scene_eps.setdefault(cluster, []).append(scene_stats[scene_id]["scene_eps"])
    cluster_eps = {
        c: float(np.percentile(v, thresholds["eps_percentile"]))
        for c, v in cluster_scene_eps.items()
    }

    # Pass 2: evaluate every perturbed condition.
    pairs: list[dict[str, Any]] = []
    ranked_points: dict[str, list[dict[str, Any]]] = {}
    gate_deaths = {"noise_floor": 0, "control": 0, "kinematic_sanity": 0}
    n_evaluated = 0
    for scene_id, conditions in scene_conditions.items():
        stats = scene_stats[scene_id]
        cluster = scene_cluster[scene_id]
        clean_row = conditions["clean"]
        scene_points: list[dict[str, Any]] = []
        for condition, row in conditions.items():
            if not condition.startswith("perturbed__"):
                continue
            n_evaluated += 1
            metrics = evaluate_perturbation(row, stats, cluster_eps[cluster], thresholds)
            scene_points.append({
                "perturbation_type": row.get("perturbation_type"),
                "trace_id": row.get("trace_id"),
                "cross_ade_m": metrics["cross_ade_m"],
                "z_score": metrics["z_score"],
                "qualifies": metrics["qualifies"],
            })
            if not metrics["qualifies"]:
                for gate, key in (("noise_floor", "gate_noise_floor"),
                                  ("control", "gate_control"),
                                  ("kinematic_sanity", "gate_kinematic_sanity")):
                    if not metrics[key]:
                        gate_deaths[gate] += 1
                continue

            chosen_sample = clean_row["samples"][stats["clean_medoid_idx"]]
            rejected_sample = row["samples"][metrics.pop("pert_medoid_idx")]
            pairs.append({
                "pair_id": row.get("trace_id") or f"{scene_id}__{row.get('perturbation_type')}",
                "scene_id": scene_id,
                "event_cluster": cluster,
                "clip_id": clean_row.get("clip_id"),
                "t0_us": clean_row.get("t0_us"),
                "prompt": {
                    "manifest_path": clean_row.get("manifest_path"),
                    "bucket": clean_row.get("bucket"),
                    "checkpoint": clean_row.get("model_version"),
                    "prompt_len_tokens": clean_row.get("prompt_len"),
                    "prompt_ids_sha256": clean_row.get("prompt_ids_sha256"),
                    "prompt_construction": (
                        "helper.create_message(frames, camera_indices, nav_text=None) + "
                        "apply_chat_template(continue_final_message=True) + fuse_traj_tokens"
                    ),
                },
                "chosen": _completion(clean_row, chosen_sample),
                "rejected": {
                    **_completion(row, rejected_sample),
                    "perturbation_type": row.get("perturbation_type"),
                },
                "metrics": metrics,
                "provenance": {
                    "model_version": clean_row.get("model_version"),
                    "sampling_params": clean_row.get("sampling_params"),
                    "token_constants": clean_row.get("token_constants"),
                    "self_generated_coc": clean_row.get("self_generated_coc"),
                },
            })
        scene_points.sort(key=lambda p: p["cross_ade_m"], reverse=True)
        ranked_points[scene_id] = scene_points

    # Ranks over qualifying pairs (z desc; ties by cross_ade).
    pairs.sort(key=lambda p: (-(p["metrics"]["z_score"] or 0.0), -p["metrics"]["cross_ade_m"]))
    by_scene_rank: dict[str, int] = {}
    by_cluster_rank: dict[str, int] = {}
    for p in pairs:
        s, c = p["scene_id"], p["event_cluster"]
        by_scene_rank[s] = by_scene_rank.get(s, 0) + 1
        by_cluster_rank[c] = by_cluster_rank.get(c, 0) + 1
        p["metrics"]["rank_within_scene"] = by_scene_rank[s]
        p["metrics"]["rank_within_cluster"] = by_cluster_rank[c]

    if semantic_delta and pairs:
        _semantic_deltas(pairs)

    report = {
        "thresholds": thresholds,
        "n_scenes_loaded": len(by_scene),
        "n_scenes_incomplete": n_incomplete,
        "n_scenes_mined": len(scene_conditions),
        "n_perturbations_evaluated": n_evaluated,
        "n_pairs": len(pairs),
        "qualify_rate": len(pairs) / n_evaluated if n_evaluated else None,
        "gate_deaths": gate_deaths,
        "cluster_eps_m": cluster_eps,
        "scene_eps_m": {s: st["scene_eps"] for s, st in scene_stats.items()},
        "control_ade_m": {s: st["control_ade"] for s, st in scene_stats.items()},
        "maximal_perturbation_points": ranked_points,
    }
    return pairs, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure_dir", default="dpo_pairs/results/measure",
                    help="Directory of per-scene measurement JSONs (S3 pull or fetch_from_logs).")
    ap.add_argument("--out_path", default="dpo_pairs/results/dpo_pairs/dpo_pairs.jsonl")
    ap.add_argument("--report_path", default="dpo_pairs/results/dpo_pairs/mining_report.json")
    ap.add_argument("--k_control", type=float, default=_DEFAULT_THRESHOLDS["k_control"])
    ap.add_argument("--eps_percentile", type=float, default=_DEFAULT_THRESHOLDS["eps_percentile"])
    ap.add_argument("--min_wellformed_frac", type=float,
                    default=_DEFAULT_THRESHOLDS["min_wellformed_frac"])
    ap.add_argument("--no_semantic_delta", action="store_true")
    args = ap.parse_args()

    by_scene = load_scene_rows(args.measure_dir)
    logger.info("loaded %d scenes from %s", len(by_scene), args.measure_dir)
    pairs, report = mine(
        by_scene,
        thresholds={
            "k_control": args.k_control,
            "eps_percentile": args.eps_percentile,
            "min_wellformed_frac": args.min_wellformed_frac,
        },
        semantic_delta=not args.no_semantic_delta,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair) + "\n")
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("%d pairs -> %s (qualify rate %.1f%%; gate deaths %s)",
                len(pairs), out_path,
                100.0 * (report["qualify_rate"] or 0.0), report["gate_deaths"])


if __name__ == "__main__":
    main()
