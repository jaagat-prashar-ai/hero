# SPDX-License-Identifier: Apache-2.0
"""
action_space_variance.py — Task 3 ("epsilon-calibration / same_action") of
the pref-pairs faithfulness project: how much does Alpamayo 1.5's own action
output vary across repeated stochastic rollouts of the *identical* scene?
That per-scenario-type variance is the empirical noise floor pair-mining
needs before it can call two DIFFERENT scenes' rollouts "the same action" or
"different actions".

Input: the JSONL output of `pref_pairs.training.run.pref_pairs_loop` (one
row per rollout) run with a manifest of clips sampled 1-per-cluster-slot via
`masking.data.sample_clips --clusters ...` and a LARGE `k` (e.g. 100) so
each scene's rollouts really are "same clip, repeated draws" rather than a
diverse sample across many different scenes. This module does not care how
`k` was set -- it just measures whatever cross-rollout spread is present in
each scene_id's rows.

Two levels of aggregation, mirroring maneuver_report.py's per-scene /
overall split:
  * per_clip_variance -- one row per scene_id (== one clip's fixed scene
    here): cross-rollout mean/std/range of the model's native per-waypoint
    (accel, curvature) action, plus cross-rollout mean/std/range of the
    scalar kinematic features trajectory_features.py already computes.
  * per_cluster_range -- pools per_clip_variance's std/range values across
    all clips sharing an event_cluster, reporting the median/p90/max of that
    distribution. p90 (not the raw max, which is one clip's worst case, and
    not the median, which is "typical" rather than "safe") is what we
    recommend as epsilon for a scenario type: a same-action threshold set at
    p90 of the observed same-scene noise says "at least 90% of same-scene
    rollout pairs for this scenario type would NOT be flagged as different
    actions by pure sampling noise alone."
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cross-rollout spread is computed for each of these scalar features (already
# present on every JSONL row -- see trajectory_features.TrajectoryFeatures.to_row_dict)
# in addition to the native per-waypoint accel/curvature arrays handled separately below.
SCALAR_FEATURES = [
    "initial_speed_mps",
    "final_speed_mps",
    "min_speed_mps",
    "final_lateral_offset_m",
    "total_heading_change_deg",
    "mean_acceleration_mps2",
    "mean_deceleration_mps2",
    "max_deceleration_mps2",
]

# (native_accel_mps2, native_curvature_per_m) -> short prefix used in output field names.
_NATIVE_ACTION_FIELDS = [("native_accel_mps2", "accel"), ("native_curvature_per_m", "curvature")]


def load_rollouts(results_dir: str | Path) -> pd.DataFrame:
    """Read every pref_pairs_rollouts*.jsonl shard under results_dir (Lilypad
    writes one file per rank -- see training/run.py's `_results_path`) into
    one DataFrame. Also accepts a single non-sharded
    pref_pairs_rollouts.jsonl (world_size<=1 / local runs)."""
    paths = sorted(glob.glob(str(Path(results_dir) / "pref_pairs_rollouts*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No pref_pairs_rollouts*.jsonl files found under {results_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    logger.info("Loaded %d rollout rows from %d file(s) under %s", len(rows), len(paths), results_dir)
    return pd.DataFrame(rows)


def _clip_id_and_t0(scene_id: str) -> tuple[str, str]:
    """scene_id is f"{clip_id}_{t0_us}" (rollout_harvester.harvest_dataset's
    convention). clip_id itself is a UUID (hyphens, no underscores), so
    splitting on the LAST underscore reliably separates it from t0_us."""
    clip_id, _, t0_us = scene_id.rpartition("_")
    return clip_id, t0_us


def _pooled_native_action(group: pd.DataFrame, field: str) -> np.ndarray | None:
    """Stack one scene's rollouts' native_accel_mps2 (or curvature) lists
    into a (n_available, T) array. Rows where capture failed (None -- see
    rollout_harvester.py's "NATIVE ACTION CAPTURE" docstring note) are
    skipped, not zero-filled, so they don't silently pull the stats toward
    zero. Rows whose length T disagrees with the majority (shouldn't happen
    for a genuinely-fixed scene, but a resumed/edited manifest could mix
    scenes) are also dropped and logged, rather than crashing the whole
    report over one bad row.
    """
    values = [np.asarray(v, dtype=np.float64) for v in group[field] if v is not None]
    if not values:
        return None

    lengths = {v.shape[0] for v in values}
    if len(lengths) > 1:
        majority_len = max(lengths, key=lambda l: sum(1 for v in values if v.shape[0] == l))
        n_dropped = sum(1 for v in values if v.shape[0] != majority_len)
        logger.warning(
            "scene %s: %d/%d rollouts have a %s length mismatch (keeping the "
            "majority length %d) -- unexpected for a fixed scene",
            group["scene_id"].iloc[0], n_dropped, len(values), field, majority_len,
        )
        values = [v for v in values if v.shape[0] == majority_len]

    return np.stack(values, axis=0) if values else None


def per_clip_variance(df: pd.DataFrame, expected_k: int | None = None) -> pd.DataFrame:
    """One row per scene_id: cross-rollout variance/range of the native
    action (accel, curvature) and of the scalar kinematic features.

    `expected_k`, if given (e.g. the config's `k`), flags scenes where fewer
    rollouts were actually written (a preempted/incomplete Lilypad shard)
    rather than silently treating a partial scene as a full one.
    """
    rows: list[dict[str, Any]] = []

    for scene_id, group in df.groupby("scene_id"):
        clip_id, t0_us = _clip_id_and_t0(scene_id)
        clusters = group["event_cluster"].unique()
        if len(clusters) > 1:
            logger.warning(
                "scene %s: rollouts disagree on event_cluster %s -- using the first",
                scene_id, list(clusters),
            )

        row: dict[str, Any] = {
            "clip_id": clip_id,
            "scene_id": scene_id,
            "t0_us": t0_us,
            "event_cluster": clusters[0],
            "n_rollouts": len(group),
            "complete": expected_k is None or len(group) >= expected_k,
        }

        for field, prefix in _NATIVE_ACTION_FIELDS:
            stacked = _pooled_native_action(group, field)
            row[f"{prefix}_n_available"] = 0 if stacked is None else stacked.shape[0]
            if stacked is None:
                row[f"{prefix}_per_waypoint_mean"] = None
                row[f"{prefix}_per_waypoint_std"] = None
                row[f"{prefix}_per_waypoint_min"] = None
                row[f"{prefix}_per_waypoint_max"] = None
                row[f"{prefix}_std_mean_over_waypoints"] = None
                row[f"{prefix}_range_max_over_waypoints"] = None
                continue

            per_wp_mean = stacked.mean(axis=0)
            per_wp_std = stacked.std(axis=0)
            per_wp_min = stacked.min(axis=0)
            per_wp_max = stacked.max(axis=0)

            row[f"{prefix}_per_waypoint_mean"] = per_wp_mean.tolist()
            row[f"{prefix}_per_waypoint_std"] = per_wp_std.tolist()
            row[f"{prefix}_per_waypoint_min"] = per_wp_min.tolist()
            row[f"{prefix}_per_waypoint_max"] = per_wp_max.tolist()
            # Single-number summaries: the "typical" spread (mean std across
            # waypoints) and the "worst-case" spread (the widest single
            # waypoint's max-min range) -- both are pooled again one level up
            # in per_cluster_range.
            row[f"{prefix}_std_mean_over_waypoints"] = float(per_wp_std.mean())
            row[f"{prefix}_range_max_over_waypoints"] = float((per_wp_max - per_wp_min).max())

        for feature in SCALAR_FEATURES:
            values = group[feature].to_numpy(dtype=np.float64)
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_std"] = float(values.std())
            row[f"{feature}_range"] = float(values.max() - values.min())

        rows.append(row)

    return pd.DataFrame(rows)


# (per_clip column, human label) pairs pooled into the per-cluster "final range".
_CLUSTER_SUMMARY_METRICS = [
    ("accel_std_mean_over_waypoints", "native accel std (m/s^2)"),
    ("accel_range_max_over_waypoints", "native accel worst-waypoint range (m/s^2)"),
    ("curvature_std_mean_over_waypoints", "native curvature std (1/m)"),
    ("curvature_range_max_over_waypoints", "native curvature worst-waypoint range (1/m)"),
] + [
    (f"{feature}_std", f"{feature} std") for feature in SCALAR_FEATURES
] + [
    (f"{feature}_range", f"{feature} range") for feature in SCALAR_FEATURES
]


def per_cluster_range(per_clip_df: pd.DataFrame) -> pd.DataFrame:
    """Pool per-clip variance/range metrics across all clips in an
    event_cluster. median/p90/max of that pooled distribution IS the "final
    range" deliverable -- p90 is the recommended epsilon (see module
    docstring), median and max are reported alongside it so the choice is
    inspectable rather than a hidden constant.
    """
    rows: list[dict[str, Any]] = []
    for cluster, group in per_clip_df.groupby("event_cluster"):
        row: dict[str, Any] = {
            "event_cluster": cluster,
            "n_clips": len(group),
            "n_incomplete_clips": int((~group["complete"]).sum()),
        }
        for column, label in _CLUSTER_SUMMARY_METRICS:
            values = group[column].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                row[f"{column}_median"] = None
                row[f"{column}_p90"] = None
                row[f"{column}_max"] = None
                continue
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_p90"] = float(np.percentile(values, 90))
            row[f"{column}_max"] = float(values.max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("event_cluster").reset_index(drop=True)


def _render_markdown(per_clip_df: pd.DataFrame, per_cluster_df: pd.DataFrame) -> str:
    """Human-readable companion to the JSON report: a per-cluster summary
    table (the calibrated epsilon per scenario type) followed by a per-clip
    appendix (exact clip_ids + their key stats)."""
    lines = ["# Action-space variance report (epsilon-calibration / same_action)", ""]

    lines.append("## Per-scenario-type summary (recommended epsilon = p90 column)")
    lines.append("")
    header = ["event_cluster", "n_clips", "n_incomplete"]
    headline_columns = [
        "accel_std_mean_over_waypoints", "curvature_std_mean_over_waypoints",
        "final_lateral_offset_m_std", "total_heading_change_deg_std",
    ]
    for column, _ in [(c, l) for c, l in _CLUSTER_SUMMARY_METRICS if c in headline_columns]:
        header += [f"{column}_median", f"{column}_p90"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, row in per_cluster_df.iterrows():
        cells = [row["event_cluster"], str(row["n_clips"]), str(row["n_incomplete_clips"])]
        for column, _ in [(c, l) for c, l in _CLUSTER_SUMMARY_METRICS if c in headline_columns]:
            for stat in ("median", "p90"):
                val = row[f"{column}_{stat}"]
                cells.append(f"{val:.4g}" if val is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "(Full per-metric median/p90/max for every pooled variance/range "
        "column is in the companion .json file's `per_cluster` section.)"
    )
    lines.append("")

    lines.append("## Per-clip appendix")
    lines.append("")
    appendix_cols = [
        "clip_id", "event_cluster", "n_rollouts", "complete",
        "accel_std_mean_over_waypoints", "curvature_std_mean_over_waypoints",
        "final_lateral_offset_m_std", "total_heading_change_deg_std",
    ]
    lines.append("| " + " | ".join(appendix_cols) + " |")
    lines.append("|" + "---|" * len(appendix_cols))
    for _, row in per_clip_df.sort_values(["event_cluster", "clip_id"]).iterrows():
        cells = []
        for col in appendix_cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def write_report(
    per_clip_df: pd.DataFrame, per_cluster_df: pd.DataFrame, out_dir: str | Path
) -> tuple[Path, Path]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    json_path = out_dir_path / "action_space_variance_report.json"
    json_path.write_text(
        json.dumps(
            {
                "per_clip": per_clip_df.to_dict(orient="records"),
                "per_cluster": per_cluster_df.to_dict(orient="records"),
            },
            indent=2,
        )
    )

    md_path = out_dir_path / "action_space_variance_report.md"
    md_path.write_text(_render_markdown(per_clip_df, per_cluster_df))

    logger.info("Wrote %s and %s", json_path, md_path)
    return json_path, md_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir", required=True,
        help="Directory containing pref_pairs_rollouts*.jsonl (pref_pairs_loop's output).",
    )
    ap.add_argument("--out_dir", default="pref_pairs/results")
    ap.add_argument(
        "--expected_k", type=int, default=None,
        help="k used when harvesting (e.g. 100) -- flags scenes with fewer rollouts than this.",
    )
    args = ap.parse_args()

    df = load_rollouts(args.results_dir)
    per_clip_df = per_clip_variance(df, expected_k=args.expected_k)
    per_cluster_df = per_cluster_range(per_clip_df)

    write_report(per_clip_df, per_cluster_df, args.out_dir)
    logger.info(
        "%d clips across %d clusters. Per-cluster summary:\n%s",
        len(per_clip_df), len(per_cluster_df),
        per_cluster_df[["event_cluster", "n_clips", "n_incomplete_clips"]].to_string(index=False),
    )


if __name__ == "__main__":
    main()
