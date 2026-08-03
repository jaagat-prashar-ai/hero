# SPDX-License-Identifier: Apache-2.0
"""
fetch_from_logs.py — reconstruct dpo_pairs/run.py's measurement rows from a
Lilypad workload's LOG STREAM, mirroring counterfactual/fetch_from_logs.py
(and pref_pairs') rationale exactly: run.py's outdir is a plain local path on
whichever machine the job ran on and is not reliably reachable afterward, so
the DPO_MEASURE log lines are a real retrieval path (alongside the per-scene
S3 JSONs when results_s3_prefix was set — S3 is preferred when available,
this script is the fallback and the cross-check).

Unlike counterfactual's one-line-per-marker-per-scene shape, run.py emits ONE
DPO_MEASURE line per (scene, condition) row — dedup key is
(scene_id, kind, condition) rather than scene_id alone (same OCI
dual-log-source duplication: every real line can arrive twice).

Output: one JSON per scene in out_dir/{scene_id}.json holding the scene's
row list — byte-compatible with what run.py put_object's to S3, so
mine_pairs.py reads either source identically.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEASURE_LOG_MARKER = "DPO_MEASURE "


def get_workload_time_window(workload_id: str, pad_minutes: int = 10):
    """Same as counterfactual.fetch_from_logs.get_workload_time_window —
    reimplemented for the same runnable-standalone reason. `lilypad workload
    logs` defaults to "last 4 hours from now", which silently returns nothing
    for a job that ran even a day earlier."""
    import re
    from datetime import datetime, timedelta, timezone

    result = subprocess.run(
        ["lilypad", "workload", "info", workload_id], capture_output=True, text=True, check=True,
    )
    info_re = re.compile(r"^(Created At|Finished At)\s+(.+)$", re.M)
    times = dict(info_re.findall(result.stdout))
    if "Created At" not in times or "Finished At" not in times:
        raise RuntimeError(
            f"could not find Created At / Finished At in `lilypad workload info {workload_id}` output"
        )

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


def parse_marked_lines(log_text: str, marker: str = MEASURE_LOG_MARKER) -> list[dict[str, Any]]:
    """Extract every JSON payload following `marker`, one per line — skip and
    count bad lines, never crash on one (same convention as the two sibling
    fetch scripts)."""
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
        logger.warning("parse_marked_lines: skipped %d unparseable line(s)", n_skipped)
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup on (scene_id, kind, condition) — each real (scene, condition)
    row is emitted exactly once by run.py, but the OCI dual log source can
    deliver it twice, and a preemption requeue can re-run a scene that never
    reached its scene_done marker (later duplicate wins is NOT wanted there:
    first-seen wins keeps the row whose scene eventually completed, since
    completed scenes are fetched first in log order... in practice the rows
    are identical byte-for-byte for a re-run with the same seeds, so
    first-seen is simply deterministic)."""
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("scene_id"), row.get("kind"), row.get("condition"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def group_by_scene(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sid = row.get("scene_id")
        if sid:
            by_scene.setdefault(sid, []).append(row)
    return by_scene


def build_scene_files(
    workload_id: str, out_dir: str | Path, logs_fetcher=fetch_workload_logs,
) -> int:
    """Fetch, dedupe, group, and write one {scene_id}.json per scene —
    byte-compatible with run.py's per-scene S3 uploads. Scenes with no
    `clean` condition row are written anyway but logged: mine_pairs.py
    requires clean + control_rawids + at least one perturbed row and will
    skip them with its own accounting."""
    rows = dedupe_rows(parse_marked_lines(logs_fetcher(workload_id, MEASURE_LOG_MARKER.strip())))
    if not rows:
        raise ValueError(f"No {MEASURE_LOG_MARKER.strip()} lines found for workload {workload_id}")

    by_scene = group_by_scene(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_no_clean = 0
    for sid, scene_rows in sorted(by_scene.items()):
        if not any(r.get("condition") == "clean" for r in scene_rows):
            n_no_clean += 1
        (out_dir / f"{sid}.json").write_text(json.dumps(scene_rows))
    if n_no_clean:
        logger.warning("%d/%d scenes have no `clean` row (in-flight at fetch time?)",
                       n_no_clean, len(by_scene))
    return len(by_scene)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload_id", required=True)
    ap.add_argument("--out_dir", default="dpo_pairs/results/measure")
    args = ap.parse_args()

    n = build_scene_files(args.workload_id, args.out_dir)
    logger.info("%d scene file(s) written to %s", n, args.out_dir)


if __name__ == "__main__":
    main()
