# SPDX-License-Identifier: Apache-2.0
"""
maneuver_report.py — auditing report for classify_maneuvers.py's output:
class distribution, threshold sensitivity, and a top-down PNG spot-check.

Three independent pieces (each usable on its own, composed by main() below):
  1. class_distribution_report -- overall + per-scene maneuver_class counts,
     flagging scenes where all K rollouts share one class vs. scenes with
     high class diversity (both are useful signals: a scene where every
     rollout gets the same class either means the model is very confident/
     low-diversity there -- worth checking against Task 1's diversity
     concern for an already-RL-post-trained model -- or a scene with many
     classes represented is a good candidate for trajectory-matched pair
     mining, since that needs >=2 distinct classes' worth of same-class
     clusters to pair within).
  2. threshold_sensitivity_report -- re-classifies the whole dataset with
     each threshold perturbed +/-20% (one at a time, holding the rest at
     baseline) and reports what fraction of labels flip. A threshold with a
     high flip rate is fragile: small, defensible disagreements about where
     exactly "a turn" starts would meaningfully change the mined dataset.
  3. render_scene_spot_checks -- top-down matplotlib plots of a sample of
     scenes' K rollouts, colored by assigned maneuver_class, for eyeballing
     -- especially useful for catching a left/right sign error (see
     classify_maneuvers.py's "Sign convention" docstring note).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module only ever saves PNGs, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pref_pairs.classify_maneuvers import (
    MANEUVER_CLASSES,
    FeatureConfig,
    ManeuverConfig,
    classify_directory,
)

logger = logging.getLogger(__name__)

# One consistent color per class across every plot in this module -- AND in
# scene_reasoning_report.py, which imports this directly rather than
# building its own map, so a color means the same maneuver across every
# plot either module produces, not just within one.
CLASS_COLORS = dict(zip(MANEUVER_CLASSES, plt.cm.tab10.colors[: len(MANEUVER_CLASSES)]))

# (section, key) pairs perturbed independently by threshold_sensitivity_report.
# Deliberately spans BOTH configs sharing maneuver_thresholds.yaml -- stop/
# yield live in FeatureConfig, turn/lane_change/proceed_accelerate in
# ManeuverConfig -- since a fragile threshold in either one matters equally.
_PERTURBABLE_THRESHOLDS = [
    ("stop", "speed_mps"),
    ("stop", "duration_s"),
    ("stop", "recovery_speed_mps"),
    ("turn", "heading_change_deg"),
    ("lane_change", "lateral_offset_m"),
    ("lane_change", "heading_change_deg"),
    ("yield", "drop_fraction"),
    ("yield", "min_speed_mps"),
    ("yield", "recovery_fraction"),
    ("proceed_accelerate", "mean_accel_mps2"),
]


def class_distribution_report(labels_df: pd.DataFrame) -> dict[str, Any]:
    """Overall + per-scene maneuver_class distribution."""
    overall_counts = labels_df["maneuver_class"].value_counts().to_dict()
    overall_fractions = (labels_df["maneuver_class"].value_counts(normalize=True)).to_dict()

    per_scene: dict[str, dict[str, int]] = {}
    uniform_scenes: list[str] = []  # every rollout in the scene got the SAME class
    diversity_by_scene: dict[str, int] = {}  # scene_id -> number of distinct classes

    for scene_id, group in labels_df.groupby("scene_id"):
        counts = group["maneuver_class"].value_counts().to_dict()
        per_scene[scene_id] = counts
        n_distinct = len(counts)
        diversity_by_scene[scene_id] = n_distinct
        if n_distinct == 1:
            uniform_scenes.append(scene_id)

    most_diverse = sorted(diversity_by_scene.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "n_rollouts": len(labels_df),
        "n_scenes": labels_df["scene_id"].nunique(),
        "overall_counts": overall_counts,
        "overall_fractions": overall_fractions,
        "per_scene_counts": per_scene,
        "uniform_scenes": uniform_scenes,  # all K rollouts share one class
        "most_diverse_scenes": most_diverse[:10],  # (scene_id, n_distinct_classes)
    }


def _perturb_raw_config(raw_config: dict[str, Any], section: str, key: str, factor: float) -> dict:
    """Return a deep copy of raw_config with one threshold scaled by `factor`."""
    perturbed = copy.deepcopy(raw_config)
    perturbed[section][key] = perturbed[section][key] * factor
    return perturbed


def threshold_sensitivity_report(
    rollouts_dir: str | Path,
    raw_config: dict[str, Any],
    baseline_labels: pd.DataFrame,
    perturb_frac: float = 0.20,
) -> pd.DataFrame:
    """Re-classify the whole dataset with each threshold perturbed
    +/-perturb_frac, one at a time, and report the fraction of rollouts
    whose maneuver_class flips relative to `baseline_labels`.

    A high flip rate on a given threshold/direction means that rule's
    boundary is fragile -- a defensible +/-20% disagreement about where a
    class starts would meaningfully change the mined dataset.
    """
    baseline = baseline_labels.set_index(["scene_id", "rollout_id"])["maneuver_class"]
    rows: list[dict[str, Any]] = []

    for section, key in _PERTURBABLE_THRESHOLDS:
        for direction, factor in (("+20%", 1.0 + perturb_frac), ("-20%", 1.0 - perturb_frac)):
            perturbed_raw = _perturb_raw_config(raw_config, section, key, factor)
            feature_config = FeatureConfig.from_dict(perturbed_raw)
            maneuver_config = ManeuverConfig.from_dict(perturbed_raw)

            perturbed_labels, _ = classify_directory(rollouts_dir, feature_config, maneuver_config)
            perturbed = perturbed_labels.set_index(["scene_id", "rollout_id"])["maneuver_class"]

            aligned_baseline, aligned_perturbed = baseline.align(perturbed)
            flipped = (aligned_baseline != aligned_perturbed).mean()

            rows.append(
                {
                    "threshold": f"{section}.{key}",
                    "direction": direction,
                    "pct_flipped": float(flipped) * 100.0,
                }
            )

    sensitivity_df = pd.DataFrame(rows)
    fragile = sensitivity_df.sort_values("pct_flipped", ascending=False).head(5)
    logger.info("Most threshold-sensitive rules (top 5):\n%s", fragile.to_string(index=False))
    return sensitivity_df


def render_scene_spot_checks(
    rollouts_dir: str | Path,
    labels_df: pd.DataFrame,
    out_dir: str | Path,
    n_scenes: int = 10,
    seed: int = 0,
) -> list[Path]:
    """Top-down (x, y) plot of every rollout in a sample of scenes, colored
    by maneuver_class, for manual eyeball verification -- see
    classify_maneuvers.py's "Sign convention" docstring note on WHY this
    matters (catching a left/right sign flip)."""
    scene_ids = sorted(labels_df["scene_id"].unique())
    rng = np.random.default_rng(seed)
    sample = rng.choice(scene_ids, size=min(n_scenes, len(scene_ids)), replace=False)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for scene_id in sample:
        scene_path = Path(rollouts_dir) / f"{scene_id}.json"
        rollout_records = json.loads(scene_path.read_text())
        scene_labels = labels_df[labels_df["scene_id"] == scene_id].set_index("rollout_id")

        fig, ax = plt.subplots(figsize=(6, 6))
        for record in rollout_records:
            xy = np.asarray(record["waypoints"])[:, :2]
            maneuver_class = scene_labels.loc[record["rollout_id"], "maneuver_class"]
            ax.plot(
                xy[:, 0], xy[:, 1],
                color=CLASS_COLORS.get(maneuver_class, "gray"),
                alpha=0.7, linewidth=1.5,
            )
        ax.scatter([0], [0], marker="*", color="black", s=80, zorder=5, label="t=0 (ego)")
        ax.set_xlabel("x (m, forward)")
        ax.set_ylabel("y (m, left)")
        ax.set_title(f"scene {scene_id}")
        ax.axis("equal")
        handles = [
            plt.Line2D([0], [0], color=color, label=cls)
            for cls, color in CLASS_COLORS.items()
            if cls in scene_labels["maneuver_class"].values
        ]
        ax.legend(handles=handles, loc="best", fontsize=8)

        out_path = out_dir_path / f"{scene_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    logger.info("Wrote %d scene spot-check PNGs to %s", len(written), out_dir_path)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts_dir", required=True)
    ap.add_argument("--config", default="pref_pairs/configs/maneuver_thresholds.yaml")
    ap.add_argument("--out_dir", default="pref_pairs/results")
    ap.add_argument("--n_spot_check_scenes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as fh:
        raw_config = yaml.safe_load(fh)
    feature_config = FeatureConfig.from_dict(raw_config)
    maneuver_config = ManeuverConfig.from_dict(raw_config)

    labels_df, _ = classify_directory(args.rollouts_dir, feature_config, maneuver_config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distribution = class_distribution_report(labels_df)
    (out_dir / "maneuver_class_distribution.json").write_text(json.dumps(distribution, indent=2))
    logger.info(
        "%d rollouts / %d scenes; %d scenes are single-class; overall: %s",
        distribution["n_rollouts"], distribution["n_scenes"],
        len(distribution["uniform_scenes"]), distribution["overall_counts"],
    )

    sensitivity_df = threshold_sensitivity_report(args.rollouts_dir, raw_config, labels_df)
    sensitivity_df.to_csv(out_dir / "maneuver_threshold_sensitivity.csv", index=False)

    render_scene_spot_checks(
        args.rollouts_dir, labels_df, out_dir / "maneuver_spot_check_plots",
        n_scenes=args.n_spot_check_scenes, seed=args.seed,
    )


if __name__ == "__main__":
    main()
