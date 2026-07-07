# SPDX-License-Identifier: Apache-2.0
"""
fetch_from_logs.py — reconstruct action_space_variance / scene_reasoning_report
inputs from a Lilypad workload's LOG STREAM instead of its outdir file.

Why this exists: pref_pairs.training.run.pref_pairs_loop writes its full
results to a local `outdir` path on whatever machine the job actually runs
on. When that path isn't reachable from wherever you'd read results
afterward -- confirmed to happen for this project: the job runs on a
us-chicago-1 Lilypad cluster, and this workstation's own identically-named
/mnt/work mount is a DIFFERENT, unrelated filesystem -- outdir is simply
unreachable, full stop. There is no retry or config fix for that; it's two
different disks that happen to share a path.

pref_pairs_loop's log_scene_summary / log_detailed_scenes config flags (see
its module docstring's "RESULT RETRIEVAL" section) are the workaround: they
ALSO emit the same data through `logger.info(...)`, which IS retrievable via
`lilypad workload logs <id> --content-filter <marker>` from anywhere with
Lilypad CLI access, regardless of storage topology. This module greps that
log output for the two markers and feeds the parsed rows straight into the
EXISTING report builders (action_space_variance.per_cluster_range /
write_report, scene_reasoning_report.write_scene_report) -- no new report
logic here, just an alternate data source for the same ones.

Known limitation: `lilypad workload logs` is itself marked EXPERIMENTAL by
the CLI, and very long single log lines could in principle be truncated by
some layer of the logging pipeline (untested at the full job's scale) --
_parse_marked_lines skips (and counts) any line that fails to parse as JSON
rather than crashing the whole fetch, so a handful of truncated lines lose
just that one scene/rollout, not the entire report.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pref_pairs.action_space_variance import per_cluster_range, write_report
from pref_pairs.scene_reasoning_report import write_scene_report
from pref_pairs.training.run import ROLLOUT_FULL_LOG_MARKER, SCENE_SUMMARY_LOG_MARKER

logger = logging.getLogger(__name__)


def fetch_workload_logs(workload_id: str, content_filter: str) -> str:
    """Thin subprocess wrapper around `lilypad workload logs` -- kept
    separate from the parsing logic below so tests can supply fixed log
    text directly instead of shelling out to the real CLI."""
    result = subprocess.run(
        ["lilypad", "workload", "logs", workload_id, "--content-filter", content_filter],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_marked_lines(log_text: str, marker: str) -> list[dict[str, Any]]:
    """Extract every JSON payload following `marker` in log_text, one per
    line. Lines without the marker are ignored; lines WITH the marker whose
    payload fails to parse are skipped and logged, not fatal (see module
    docstring's "Known limitation" note)."""
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for line in log_text.splitlines():
        idx = line.find(marker)
        if idx == -1:
            continue
        payload = line[idx + len(marker):].strip()
        try:
            rows.append(json.loads(payload))
        except json.JSONDecodeError:
            n_skipped += 1
    if n_skipped:
        logger.warning("parse_marked_lines(%s): skipped %d unparseable line(s)", marker, n_skipped)
    return rows


def build_action_space_variance_report(
    workload_id: str, out_dir: str | Path,
    logs_fetcher=fetch_workload_logs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every SCENE_SUMMARY_LOG_MARKER line for workload_id and feed it
    into the existing per_cluster_range / write_report pipeline -- one row
    per scene, already computed on the worker (see
    run.py._build_scene_summary_log_line), so no per-clip recomputation is
    needed here."""
    log_text = logs_fetcher(workload_id, SCENE_SUMMARY_LOG_MARKER)
    summaries = parse_marked_lines(log_text, SCENE_SUMMARY_LOG_MARKER)
    if not summaries:
        raise ValueError(
            f"No {SCENE_SUMMARY_LOG_MARKER} lines found for workload {workload_id} -- "
            "was log_scene_summary=true set in this run's config?"
        )
    per_clip_df = pd.DataFrame(summaries)
    per_cluster_df = per_cluster_range(per_clip_df)
    write_report(per_clip_df, per_cluster_df, out_dir)
    return per_clip_df, per_cluster_df


def build_scene_reasoning_reports(
    workload_id: str, out_dir: str | Path,
    logs_fetcher=fetch_workload_logs,
) -> pd.DataFrame:
    """Fetch every ROLLOUT_FULL_LOG_MARKER line for workload_id, group by
    scene_id, and render one scene_reasoning_report per scene present
    (bounded by that run's log_detailed_scenes cap, not by manifest size)."""
    log_text = logs_fetcher(workload_id, ROLLOUT_FULL_LOG_MARKER)
    rows = parse_marked_lines(log_text, ROLLOUT_FULL_LOG_MARKER)
    if not rows:
        raise ValueError(
            f"No {ROLLOUT_FULL_LOG_MARKER} lines found for workload {workload_id} -- "
            "was log_detailed_scenes > 0 set in this run's config?"
        )
    detailed_df = pd.DataFrame(rows)
    for scene_id, scene_df in detailed_df.groupby("scene_id"):
        png_path, md_path = write_scene_report(scene_df, out_dir)
        logger.info("scene %s: wrote %s and %s", scene_id, png_path, md_path)
    return detailed_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload_id", required=True)
    ap.add_argument("--out_dir", default="pref_pairs/results")
    ap.add_argument("--scene_reasoning_out_dir", default="pref_pairs/results/scene_reasoning")
    ap.add_argument(
        "--skip", choices=["summary", "detailed"], action="append", default=[],
        help="Skip one of the two fetches (e.g. if that run didn't enable it).",
    )
    args = ap.parse_args()

    if "summary" not in args.skip:
        per_clip_df, per_cluster_df = build_action_space_variance_report(args.workload_id, args.out_dir)
        logger.info("%d clips across %d clusters (see %s)", len(per_clip_df), len(per_cluster_df), args.out_dir)

    if "detailed" not in args.skip:
        detailed_df = build_scene_reasoning_reports(args.workload_id, args.scene_reasoning_out_dir)
        logger.info(
            "%d rollouts across %d detailed scenes (see %s)",
            len(detailed_df), detailed_df["scene_id"].nunique(), args.scene_reasoning_out_dir,
        )


if __name__ == "__main__":
    main()
