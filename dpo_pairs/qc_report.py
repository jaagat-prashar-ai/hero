# SPDX-License-Identifier: Apache-2.0
"""
qc_report.py — Stage-3 QC for the dpo_pairs pipeline: the plots a human must
eyeball before trusting mined pairs enough to train on them (plan §Verification).

  1. semantic_vs_trajectory.png — semantic delta (MiniLM cosine distance,
     from mine_pairs' metrics) vs cross_ade, colored by perturbation type,
     with each cluster's eps as a horizontal line. The load-bearing question:
     is there a positive relationship at all? (If trajectory response is
     uncorrelated with semantic magnitude, the model isn't reading the
     reasoning the way the DPO hypothesis assumes.)
  2. ade_histograms.png — per-cluster clean-clean pairwise ADE (the noise
     floor) vs clean-perturbed cross ADE, cluster_eps marked. The gate is
     only meaningful if these distributions visibly separate for real effects.
  3. pair_trajectories/ — plan-view (x/y) trajectory-fan plots for the top-K
     and bottom-K qualifying pairs: clean fan (all N samples) vs perturbed
     fan, medoids bolded. "Plausible wrong driving, not degenerate garbage"
     is a human judgment — this is where it happens. (Camera-frame overlays
     à la pref_pairs/render_trajectory_overlay.py need WDS frame fetches and
     can be added when needed; the plan view answers the go/no-go question.)

Inputs are mine_pairs.py's outputs plus the measurement dir (for the sample
fans). Matplotlib only, headless (Agg).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_pairs(pairs_path: str | Path) -> list[dict[str, Any]]:
    with open(pairs_path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collect_scatter_data(pairs: list[dict[str, Any]]) -> dict[str, list]:
    """(semantic_delta, cross_ade, type, cluster) tuples for qualifying pairs
    that have a semantic delta. Pure — unit-testable."""
    out: dict[str, list] = {"semantic": [], "cross_ade": [], "ptype": [], "cluster": []}
    for p in pairs:
        m = p["metrics"]
        if m.get("semantic_delta_cos") is None:
            continue
        out["semantic"].append(m["semantic_delta_cos"])
        out["cross_ade"].append(m["cross_ade_m"])
        out["ptype"].append(p["rejected"].get("perturbation_type") or "?")
        out["cluster"].append(p.get("event_cluster") or "?")
    return out


def _scatter_plot(pairs: list[dict[str, Any]], report: dict[str, Any], out_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = collect_scatter_data(pairs)
    if not data["semantic"]:
        logger.warning("no pairs with semantic deltas — scatter skipped "
                       "(rerun mine_pairs without --no_semantic_delta)")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    ptypes = sorted(set(data["ptype"]))
    for ptype in ptypes:
        idx = [i for i, t in enumerate(data["ptype"]) if t == ptype]
        ax.scatter([data["semantic"][i] for i in idx],
                   [data["cross_ade"][i] for i in idx],
                   label=ptype, alpha=0.7, s=24)
    for cluster, eps in sorted(report.get("cluster_eps_m", {}).items()):
        ax.axhline(eps, color="gray", linestyle="--", linewidth=0.8)
        ax.annotate(f"eps {cluster[:24]}", (ax.get_xlim()[0], eps),
                    fontsize=6, color="gray", va="bottom")
    ax.set_xlabel("semantic delta (MiniLM cosine distance, clean vs perturbed CoC)")
    ax.set_ylabel("cross ADE vs clean condition [m]")
    ax.set_title(f"semantic vs trajectory delta — {len(data['semantic'])} qualifying pairs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _ade_histograms(measure_dir: Path, report: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dpo_pairs.mine_pairs import (
        _sample_xyzs,
        cross_ades,
        index_conditions,
        load_scene_rows,
        pairwise_ades,
    )

    by_cluster: dict[str, dict[str, list[float]]] = {}
    for scene_id, rows in load_scene_rows(measure_dir).items():
        conditions = index_conditions(rows)
        clean = conditions.get("clean")
        if clean is None:
            continue
        cluster = clean.get("event_cluster") or "?"
        bucket = by_cluster.setdefault(cluster, {"noise": [], "effect": []})
        clean_xyzs = _sample_xyzs(clean)
        if len(clean_xyzs) >= 2:
            bucket["noise"].extend(pairwise_ades(clean_xyzs).tolist())
        for condition, row in conditions.items():
            if condition.startswith("perturbed__"):
                bucket["effect"].extend(cross_ades(_sample_xyzs(row), clean_xyzs).tolist())

    clusters = sorted(by_cluster)
    if not clusters:
        logger.warning("no scenes in %s — histograms skipped", measure_dir)
        return
    fig, axes = plt.subplots(len(clusters), 1, figsize=(8, 2.4 * len(clusters)), squeeze=False)
    for ax, cluster in zip(axes[:, 0], clusters):
        d = by_cluster[cluster]
        bins = np.linspace(0, max(max(d["noise"], default=1), max(d["effect"], default=1)), 60)
        ax.hist(d["noise"], bins=bins, alpha=0.6, label="clean-clean (noise floor)", density=True)
        ax.hist(d["effect"], bins=bins, alpha=0.6, label="clean-perturbed (effect)", density=True)
        eps = report.get("cluster_eps_m", {}).get(cluster)
        if eps is not None:
            ax.axvline(eps, color="k", linestyle="--", linewidth=1, label=f"cluster_eps={eps:.2f} m")
        ax.set_title(cluster, fontsize=8)
        ax.set_xlabel("ADE [m]", fontsize=7)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pair_trajectory_plots(
    pairs: list[dict[str, Any]], measure_dir: Path, out_dir: Path, top_k: int,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dpo_pairs.mine_pairs import index_conditions, load_scene_rows

    by_scene = load_scene_rows(measure_dir)
    # top-K strongest and bottom-K weakest qualifying pairs — the weakest are
    # the ones nearest the gate, where "did this really clear noise?" is
    # most in doubt.
    ordered = sorted(pairs, key=lambda p: p["metrics"]["cross_ade_m"], reverse=True)
    selected = ordered[:top_k] + (ordered[-top_k:] if len(ordered) > top_k else [])

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in selected:
        rows = by_scene.get(p["scene_id"])
        if rows is None:
            continue
        conditions = index_conditions(rows)
        clean = conditions.get("clean")
        pert = conditions.get(f"perturbed__{p['rejected']['perturbation_type']}")
        if clean is None or pert is None:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        for row, color, label in ((clean, "tab:blue", "clean"), (pert, "tab:red", "perturbed")):
            for i, s in enumerate(row["samples"]):
                xyz = np.asarray(s["xyz"])
                ax.plot(xyz[:, 1], xyz[:, 0], color=color, alpha=0.25, linewidth=0.8,
                        label=label if i == 0 else None)
        for side, color in (("chosen", "tab:blue"), ("rejected", "tab:red")):
            xyz = np.asarray(p[side]["xyz"])
            ax.plot(xyz[:, 1], xyz[:, 0], color=color, linewidth=2.2)
        ax.invert_xaxis()  # y is left-positive (ISO 8855): plot left on the left
        ax.set_xlabel("lateral y [m] (left +)")
        ax.set_ylabel("forward x [m]")
        m = p["metrics"]
        ax.set_title(
            f"{p['pair_id']}\ncross_ade={m['cross_ade_m']:.2f} m  "
            f"eps={max(m['scene_eps_m'], m['cluster_eps_m']):.2f} m  z={m['z_score']:.1f}",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(out_dir / f"{p['pair_id']}.png", dpi=150)
        plt.close(fig)
        n += 1
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs_path", default="dpo_pairs/results/dpo_pairs/dpo_pairs.jsonl")
    ap.add_argument("--report_path", default="dpo_pairs/results/dpo_pairs/mining_report.json")
    ap.add_argument("--measure_dir", default="dpo_pairs/results/measure")
    ap.add_argument("--out_dir", default="dpo_pairs/results/qc")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    pairs = load_pairs(args.pairs_path)
    report = json.loads(Path(args.report_path).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made_scatter = _scatter_plot(pairs, report, out_dir / "semantic_vs_trajectory.png")
    _ade_histograms(Path(args.measure_dir), report, out_dir / "ade_histograms.png")
    n_traj = _pair_trajectory_plots(pairs, Path(args.measure_dir),
                                    out_dir / "pair_trajectories", args.top_k)
    logger.info("QC written to %s (scatter=%s, %d pair trajectory plots)",
                out_dir, made_scatter, n_traj)


if __name__ == "__main__":
    main()
