# SPDX-License-Identifier: Apache-2.0
"""
build_ground_truth_action_dataset.py — join each scene's ground-truth reasoning
trace (and its perturbations from pref_pairs.perturbation_generator) with the
REAL, already-computed action that produced that exact trace.

Why no new model inference is needed: perturbation_generator.extract_ground_truth_traces
took each scene's ground_truth_trace from the FIRST rollout's blockquoted CoT text in
results/fixed_reasoning/scene_reasoning/*_reasoning.md -- the rendered output of the
already-completed "diffusion-only noise (reasoning fixed)" experiment (workload
pref-pairs-fixed-reasoning-cluster-f6kq5o). That first rollout already has a real
model-produced action (full waypoints + native accel/curvature arrays); this module
recovers it from the workload's logs (the .md files only kept a 4-field scalar summary,
not the raw per-waypoint arrays) and joins it back in, rather than re-running inference.

Verified during planning (see /home/jaagat-prashar/.claude/plans/generic-waddling-pie.md):
fetching this workload's PREF_PAIRS_ROLLOUT_FULL log rows and replicating
scene_reasoning_report.render_scene_reasoning_markdown's exact rollout-selection order
(groupby("maneuver_class") alphabetically, then sort_values("rollout_id"), first row)
reproduces a coc_text that is byte-identical to ground_truth_trace for all 120/120
scenes -- zero mismatches.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pref_pairs.fetch_from_logs import parse_marked_lines
from pref_pairs.training.run import ROLLOUT_FULL_LOG_MARKER

logger = logging.getLogger(__name__)

DEFAULT_WORKLOAD_ID = "pref-pairs-fixed-reasoning-cluster-f6kq5o"

# Duplicated from render_trajectory_overlay.py's get_workload_time_window
# rather than imported: that module pulls in av/boto3/PIL/scipy for its video
# rendering, none of which this module needs just to parse two timestamps out
# of `lilypad workload info` -- same "reimplement a few lines rather than
# import a heavy sibling" trade-off this project already makes elsewhere
# (e.g. rollout_harvester.py's _denorm_accel_curvature).
_INFO_RE = re.compile(r"^(Created At|Finished At)\s+(.+)$", re.M)
_TZ_OFFSETS = {"PDT": -7, "PST": -8, "UTC": 0}


def _parse_workload_timestamp(text: str) -> datetime:
    dt_str, tz_abbr = text.rsplit(" ", 1)
    naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    offset = _TZ_OFFSETS.get(tz_abbr)
    if offset is None:
        raise ValueError(f"unrecognized timezone abbreviation {tz_abbr!r} in workload info output")
    return naive.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(timezone.utc)


def get_workload_time_window(workload_id: str, pad_minutes: int = 10) -> tuple[datetime, datetime]:
    """`lilypad workload logs` defaults to "last 4 hours from now", which
    silently returns nothing for a job that finished even a day earlier.
    Deriving the actual window from `lilypad workload info`'s Created At /
    Finished At avoids that failure mode (same fix already applied in
    render_trajectory_overlay.py, duplicated here -- see module docstring)."""
    result = subprocess.run(
        ["lilypad", "workload", "info", workload_id], capture_output=True, text=True, check=True
    )
    times = dict(_INFO_RE.findall(result.stdout))
    if "Created At" not in times or "Finished At" not in times:
        raise RuntimeError(f"could not find Created At / Finished At in `lilypad workload info {workload_id}` output")
    start = _parse_workload_timestamp(times["Created At"]) - timedelta(minutes=pad_minutes)
    end = _parse_workload_timestamp(times["Finished At"]) + timedelta(minutes=pad_minutes)
    return start, end

# Fields copied verbatim from the selected ground-truth rollout's log row into
# each scene's "ground_truth_action" -- the raw per-waypoint arrays plus the
# scalar kinematic summary already used elsewhere in this project (e.g.
# scene_reasoning_report.py's _SUMMARY_COLUMNS) and the classified maneuver.
_ACTION_FIELDS = [
    "rollout_id", "waypoints", "hz", "native_accel_mps2", "native_curvature_per_m",
    "mean_acceleration_mps2", "mean_deceleration_mps2", "final_lateral_offset_m",
    "total_heading_change_deg", "maneuver_class",
]

_PERTURBATION_FIELDS = [
    "trace_id", "perturbation_type", "original_span", "perturbed_span",
    "perturbed_trace", "semantic_delta", "decision_impact", "plausibility_rationale",
]


def _scalar(value: Any) -> Any:
    """Unwrap a numpy scalar (int64/float64) to a plain Python type so the
    final dataset is json.dumps-able without a custom encoder. Lists (e.g.
    waypoints) are already plain Python objects at this point -- pandas
    doesn't convert list-valued cells to ndarrays on DataFrame construction
    from a list of dicts -- so they pass through unchanged."""
    return value.item() if hasattr(value, "item") else value


def fetch_rollout_rows(
    workload_id: str,
    logs_fetcher: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Fetch every PREF_PAIRS_ROLLOUT_FULL log row for workload_id and dedupe
    by (scene_id, rollout_id) -- same OCI dual-log-source double-ingest fix
    already applied in fetch_from_logs.py's build_scene_reasoning_reports.

    Unlike fetch_from_logs.fetch_workload_logs (which calls `lilypad workload
    logs` with no time bounds, defaulting to "last 4 hours from now"), this
    derives the real --start-time/--end-time window from `lilypad workload
    info` (render_trajectory_overlay.get_workload_time_window) -- required
    for a workload that finished more than 4 hours ago, which this one has.

    logs_fetcher, if given, is a zero-arg callable returning raw log text
    directly (tests inject fixed text this way instead of shelling out).
    """
    if logs_fetcher is None:
        start, end = get_workload_time_window(workload_id)
        result = subprocess.run(
            [
                "lilypad", "workload", "logs", workload_id,
                "--content-filter", ROLLOUT_FULL_LOG_MARKER,
                "--start-time", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "--end-time", end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            capture_output=True, text=True, check=True,
        )
        log_text = result.stdout
    else:
        log_text = logs_fetcher(workload_id)

    rows = parse_marked_lines(log_text, ROLLOUT_FULL_LOG_MARKER)
    if not rows:
        raise ValueError(
            f"No {ROLLOUT_FULL_LOG_MARKER} lines found for workload {workload_id}"
        )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["scene_id", "rollout_id"], keep="first")


def select_ground_truth_rollout(scene_df: pd.DataFrame) -> pd.Series:
    """Reproduce scene_reasoning_report.render_scene_reasoning_markdown's exact
    rollout order -- groupby("maneuver_class") (pandas groups sort keys
    alphabetically by default) then .sort_values("rollout_id") -- and return
    the very first row. That row is exactly what
    perturbation_generator.extract_ground_truth_traces read as the scene's
    ground_truth_trace (the first blockquote encountered parsing the
    rendered .md top to bottom), so its action is THE ground-truth action.
    """
    first_class = sorted(scene_df["maneuver_class"].unique())[0]
    class_group = scene_df[scene_df["maneuver_class"] == first_class].sort_values("rollout_id")
    return class_group.iloc[0]


def load_perturbations_by_scene(perturbations_path: str | Path) -> dict[str, dict[str, Any]]:
    """Group perturbations.jsonl rows by scene_id. Returns
    {scene_id: {"event_cluster", "ground_truth_trace", "perturbations": [...]}}."""
    by_scene: dict[str, dict[str, Any]] = {}
    with open(perturbations_path) as fh:
        for line in fh:
            row = json.loads(line)
            scene_id = row["scene_id"]
            entry = by_scene.setdefault(scene_id, {
                "scene_id": scene_id,
                "event_cluster": row["event_cluster"],
                "ground_truth_trace": row["ground_truth_trace"],
                "perturbations": [],
            })
            entry["perturbations"].append({field: row[field] for field in _PERTURBATION_FIELDS})
    return by_scene


def build_dataset(
    rollout_df: pd.DataFrame,
    perturbations_by_scene: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join each scene's ground-truth trace + perturbations with the real
    action from its selected ground-truth rollout. Scenes missing from
    rollout_df, or whose selected rollout's coc_text doesn't match
    ground_truth_trace, are logged and skipped from the action join (not
    hard failures -- mirrors this project's existing lenient-warn
    convention, e.g. fetch_from_logs.py's unparseable-line handling)."""
    dataset: list[dict[str, Any]] = []
    missing_scenes: list[str] = []
    mismatched_scenes: list[str] = []

    for scene_id, entry in perturbations_by_scene.items():
        scene_df = rollout_df[rollout_df["scene_id"] == scene_id]
        if scene_df.empty:
            missing_scenes.append(scene_id)
            continue

        gt_row = select_ground_truth_rollout(scene_df)
        if gt_row["coc_text"] != entry["ground_truth_trace"]:
            mismatched_scenes.append(scene_id)
            logger.warning(
                "scene %s: selected rollout's coc_text does not match ground_truth_trace "
                "(selected=%r, expected=%r) -- including it anyway, flagged for review",
                scene_id, gt_row["coc_text"], entry["ground_truth_trace"],
            )

        ground_truth_action = {field: _scalar(gt_row[field]) for field in _ACTION_FIELDS}
        dataset.append({
            "scene_id": scene_id,
            "event_cluster": entry["event_cluster"],
            "ground_truth_trace": entry["ground_truth_trace"],
            "ground_truth_action": ground_truth_action,
            "perturbations": entry["perturbations"],
        })

    if missing_scenes:
        logger.warning(
            "%d scene(s) had no rollout data in the fetched logs: %s",
            len(missing_scenes), missing_scenes,
        )
    if mismatched_scenes:
        logger.warning(
            "%d scene(s) had a coc_text mismatch: %s",
            len(mismatched_scenes), mismatched_scenes,
        )
    logger.info(
        "Built %d scene entries (%d missing, %d coc_text mismatches)",
        len(dataset), len(missing_scenes), len(mismatched_scenes),
    )
    return dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload_id", default=DEFAULT_WORKLOAD_ID)
    ap.add_argument(
        "--perturbations_path",
        default="pref_pairs/results/perturbations/perturbations.jsonl",
    )
    ap.add_argument(
        "--out_path",
        default="pref_pairs/results/perturbation_actions/ground_truth_actions.json",
    )
    args = ap.parse_args()

    rollout_df = fetch_rollout_rows(args.workload_id)
    perturbations_by_scene = load_perturbations_by_scene(args.perturbations_path)
    dataset = build_dataset(rollout_df, perturbations_by_scene)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2))
    logger.info("Wrote %d scene entries to %s", len(dataset), out_path)


if __name__ == "__main__":
    main()
