# SPDX-License-Identifier: Apache-2.0
"""
scene_reasoning_report.py — qualitative "how does the model's reasoning vary
across repeated rollouts of the SAME scene, and does it line up with the
action it actually took?" report. For one scene's K rollouts (e.g. the K=100
same-scene draws pref_pairs_loop writes for epsilon-calibration), renders:

  * a top-down (x, y) trajectory plot, one line per rollout, colored by
    maneuver_class (pref_pairs.maneuver_report.CLASS_COLORS) -- the cheapest
    already-computed proxy for "which rollouts took a similar action",
  * a companion Markdown file listing every rollout's FULL CoT reasoning
    trace (coc_text, untruncated) grouped under its maneuver_class, plus a
    one-line kinematic summary (mean accel, mean curvature, final lateral
    offset, total heading change) -- so a reviewer can scan whether
    same-class rollouts reason similarly and different-class rollouts
    reason differently too (or don't, which is itself an unfaithful-
    reasoning signal worth flagging for the later claim-verification tasks).

Why maneuver_class instead of a text-clustering algorithm: this environment
has no NLP/embedding library (no sklearn, no sentence-transformers), and
maneuver_class already discretizes the ACTION side into semantically
meaningful buckets (stop/turn/lane_change/yield/proceed/lane_keep) via
pref_pairs.classify_maneuvers's rule cascade -- reusing it directly compares
"reasoning conditioned on action" without inventing a separate, unvalidated
reasoning-clustering method. A proper text/embedding clustering of the CoT
itself belongs to the project brief's later claim-extraction/consistency-
checking tasks, not reinvented here as a side effect of a plotting script.

Input: the same pref_pairs_rollouts*.jsonl schema action_space_variance.py
reads (one row per rollout, from pref_pairs.training.run.pref_pairs_loop) --
reuses that module's load_rollouts rather than re-implementing JSONL loading.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module only ever saves files, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pref_pairs.action_space_variance import load_rollouts
from pref_pairs.maneuver_report import CLASS_COLORS

logger = logging.getLogger(__name__)

# Kinematic columns already computed per-rollout by trajectory_features.py
# (see pref_pairs.training.run.pref_pairs_loop) -- surfaced as the one-line
# action summary next to each rollout's full reasoning trace.
_SUMMARY_COLUMNS = [
    "mean_acceleration_mps2", "mean_deceleration_mps2",
    "final_lateral_offset_m", "total_heading_change_deg",
]


def pick_scene_ids(df: pd.DataFrame, n_scenes: int, seed: int) -> list[str]:
    """Sample n_scenes distinct scene_ids from the loaded rollouts (same
    sample-without-replacement pattern as maneuver_report.render_scene_spot_checks,
    so repeated calls with the same seed pick the same scenes)."""
    scene_ids = sorted(df["scene_id"].unique())
    rng = np.random.default_rng(seed)
    return list(rng.choice(scene_ids, size=min(n_scenes, len(scene_ids)), replace=False))


def render_scene_action_plot(scene_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Top-down (x, y) plot of every rollout's waypoints, colored by
    maneuver_class -- the "cluster the actions" view. One line per rollout;
    legend lists only the classes actually present in this scene."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in scene_df.iterrows():
        xy = np.asarray(row["waypoints"])[:, :2]
        ax.plot(
            xy[:, 0], xy[:, 1],
            color=CLASS_COLORS.get(row["maneuver_class"], "gray"),
            alpha=0.5, linewidth=1.2,
        )
    ax.scatter([0], [0], marker="*", color="black", s=80, zorder=5, label="t=0 (ego)")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    scene_id = scene_df["scene_id"].iloc[0]
    ax.set_title(f"scene {scene_id} -- {len(scene_df)} rollouts")
    ax.axis("equal")
    present_classes = scene_df["maneuver_class"].unique()
    handles = [
        plt.Line2D([0], [0], color=color, label=cls)
        for cls, color in CLASS_COLORS.items()
        if cls in present_classes
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_scene_reasoning_markdown(scene_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Full CoT reasoning trace for every rollout, grouped under its
    maneuver_class -- lets a reviewer scan whether rollouts that took the
    SAME action reasoned similarly, and whether rollouts that took
    DIFFERENT actions reasoned differently too."""
    scene_id = scene_df["scene_id"].iloc[0]
    event_cluster = scene_df["event_cluster"].iloc[0] if "event_cluster" in scene_df.columns else "?"
    lines = [
        f"# Scene {scene_id} -- reasoning vs. action across {len(scene_df)} rollouts",
        "",
        f"event_cluster: {event_cluster}",
        "",
        "Class counts: "
        + ", ".join(f"{cls}={n}" for cls, n in scene_df["maneuver_class"].value_counts().items()),
        "",
    ]

    for maneuver_class, group in scene_df.groupby("maneuver_class"):
        lines.append(f"## {maneuver_class} ({len(group)} rollouts)")
        lines.append("")
        for _, row in group.sort_values("rollout_id").iterrows():
            summary = ", ".join(
                f"{col}={row[col]:.3g}" for col in _SUMMARY_COLUMNS if col in row and pd.notna(row[col])
            )
            lines.append(f"### rollout {row['rollout_id']}")
            lines.append(f"*{summary}*")
            lines.append("")
            # Full CoT text, verbatim -- blockquoted so multi-line reasoning
            # renders as one visually distinct block per rollout.
            coc_text = str(row.get("coc_text", ""))
            lines.extend(f"> {line}" for line in coc_text.splitlines() or [""])
            lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def write_scene_report(scene_df: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    scene_id = scene_df["scene_id"].iloc[0]
    out_dir_path = Path(out_dir)
    png_path = render_scene_action_plot(scene_df, out_dir_path / f"{scene_id}_actions.png")
    md_path = render_scene_reasoning_markdown(scene_df, out_dir_path / f"{scene_id}_reasoning.md")
    return png_path, md_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir", required=True,
        help="Directory containing pref_pairs_rollouts*.jsonl (pref_pairs_loop's output).",
    )
    ap.add_argument("--out_dir", default="pref_pairs/results/scene_reasoning")
    ap.add_argument("--scene_id", default=None, help="Render only this scene (default: sample --n_scenes).")
    ap.add_argument("--n_scenes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_rollouts(args.results_dir)
    scene_ids = [args.scene_id] if args.scene_id else pick_scene_ids(df, args.n_scenes, args.seed)

    for scene_id in scene_ids:
        scene_df = df[df["scene_id"] == scene_id]
        png_path, md_path = write_scene_report(scene_df, args.out_dir)
        logger.info("scene %s: wrote %s and %s", scene_id, png_path, md_path)


if __name__ == "__main__":
    main()
