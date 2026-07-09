# SPDX-License-Identifier: Apache-2.0
"""
fetch_from_logs.py — reconstruct counterfactual.py's per-scene results from a
Lilypad workload's LOG STREAM, mirroring pref_pairs/fetch_from_logs.py's
rationale exactly: run.py's outdir is a plain local path on whichever machine
the job actually ran on and is not reliably reachable from the submitting
workstation afterward, so the three log markers
(COUNTERFACTUAL_TOKEN_MAP / COUNTERFACTUAL_SWAP_A / COUNTERFACTUAL_SWAP_B)
are the real retrieval path.

Same confirmed OCI Logging Analytics dual-log-source duplication as
pref_pairs/fetch_from_logs.py -- `lilypad workload logs` queries both 'Ray
Application Logs' and 'Kubernetes Container Generic Logs' and returns every
matching row with no cross-source dedup, so a pod watched by both collectors
delivers each real log line twice. Deduped here on scene_id per marker (each
scene produces exactly one line per marker, unlike pref_pairs' per-rollout
markers).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_MAP_LOG_MARKER = "COUNTERFACTUAL_TOKEN_MAP "
SWAP_A_LOG_MARKER = "COUNTERFACTUAL_SWAP_A "
SWAP_B_LOG_MARKER = "COUNTERFACTUAL_SWAP_B "


def get_workload_time_window(workload_id: str, pad_minutes: int = 10):
    """Same as pref_pairs.render_trajectory_overlay.get_workload_time_window
    -- reimplemented here rather than imported, to keep this module runnable
    without a pref_pairs import (mirrors that module's own independence
    choice). `lilypad workload logs` defaults to "last 4 hours from now",
    which silently returns nothing for a job that ran even a day earlier."""
    import re
    from datetime import datetime, timedelta, timezone

    result = subprocess.run(
        ["lilypad", "workload", "info", workload_id], capture_output=True, text=True, check=True,
    )
    info_re = re.compile(r"^(Created At|Finished At)\s+(.+)$", re.M)
    times = dict(info_re.findall(result.stdout))
    if "Created At" not in times or "Finished At" not in times:
        raise RuntimeError(f"could not find Created At / Finished At in `lilypad workload info {workload_id}` output")

    tz_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+)$")

    def parse(ts: str) -> datetime:
        m = tz_re.match(ts.strip())
        if not m:
            raise ValueError(f"unrecognized timestamp format: {ts!r}")
        naive = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        offsets = {"PST": -8, "PDT": -7, "UTC": 0}
        if m.group(2) not in offsets:
            raise ValueError(f"unrecognized timezone abbreviation {m.group(2)!r}")
        return naive.replace(tzinfo=timezone(timedelta(hours=offsets[m.group(2)]))).astimezone(timezone.utc)

    start = parse(times["Created At"]) - timedelta(minutes=pad_minutes)
    end = parse(times["Finished At"]) + timedelta(minutes=pad_minutes)
    return start, end


def fetch_workload_logs(workload_id: str, content_filter: str) -> str:
    """Thin subprocess wrapper, time-windowed via get_workload_time_window --
    kept separate from parsing so tests can supply fixed log text directly."""
    start, end = get_workload_time_window(workload_id)
    result = subprocess.run(
        [
            "lilypad", "workload", "logs", workload_id, "--content-filter", content_filter,
            "--start-time", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--end-time", end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_marked_lines(log_text: str, marker: str) -> list[dict[str, Any]]:
    """Extract every JSON payload following `marker` in log_text, one per
    line. Same "skip and count, don't crash on one bad line" behavior as
    pref_pairs.fetch_from_logs.parse_marked_lines."""
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


def fetch_and_dedupe(workload_id: str, marker: str, logs_fetcher=fetch_workload_logs) -> list[dict[str, Any]]:
    """Fetch + parse one marker's lines, deduped by scene_id (see module
    docstring's dual-log-source note)."""
    log_text = logs_fetcher(workload_id, marker)
    rows = parse_marked_lines(log_text, marker)
    seen: set[str] = set()
    deduped = []
    for row in rows:
        sid = row.get("scene_id")
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(row)
    return deduped


def build_scene_reports(workload_id: str, out_dir: str | Path, logs_fetcher=fetch_workload_logs) -> int:
    """Fetch all three markers, join by scene_id, and write one JSON per
    scene to out_dir/{scene_id}.json -- mirrors the shape run.py's own
    (unreachable) per-scene outdir write already uses, so downstream tooling
    doesn't need to know results came from the log stream instead."""
    token_maps = {r["scene_id"]: r for r in fetch_and_dedupe(workload_id, TOKEN_MAP_LOG_MARKER, logs_fetcher)}
    swap_a = {r["scene_id"]: r for r in fetch_and_dedupe(workload_id, SWAP_A_LOG_MARKER, logs_fetcher)}
    swap_b = {r["scene_id"]: r for r in fetch_and_dedupe(workload_id, SWAP_B_LOG_MARKER, logs_fetcher)}

    if not token_maps:
        raise ValueError(
            f"No {TOKEN_MAP_LOG_MARKER.strip()} lines found for workload {workload_id}"
        )

    all_scene_ids = set(token_maps) | set(swap_a) | set(swap_b)
    incomplete = [
        sid for sid in all_scene_ids
        if sid not in token_maps or sid not in swap_a or sid not in swap_b
    ]
    if incomplete:
        logger.warning(
            "%d/%d scenes missing at least one of the 3 markers (likely still "
            "in-flight when logs were fetched, or a requeue re-ran a scene "
            "partway): %s",
            len(incomplete), len(all_scene_ids), sorted(incomplete)[:5],
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for sid in sorted(all_scene_ids):
        if sid in incomplete:
            continue
        (out_dir / f"{sid}.json").write_text(json.dumps({
            "token_alternative_map": {
                "cot": token_maps[sid]["cot"],
                "n_reasoning_tokens": token_maps[sid]["n_reasoning_tokens"],
                "mean_entropy": token_maps[sid]["mean_entropy"],
                "positions": token_maps[sid]["positions"],
            },
            "single_token_swap_sweep": swap_a[sid]["positions"],
            "counterfactual_sweep": swap_b[sid]["positions"],
            "baseline_xy_a": swap_a[sid].get("baseline_xy"),
            "baseline_xy_b": swap_b[sid].get("baseline_xy"),
        }, indent=2))
        n_written += 1
    return n_written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload_id", required=True)
    ap.add_argument("--out_dir", default="counterfactual/results")
    args = ap.parse_args()

    n = build_scene_reports(args.workload_id, args.out_dir)
    logger.info("%d complete scene report(s) written to %s", n, args.out_dir)


if __name__ == "__main__":
    main()
