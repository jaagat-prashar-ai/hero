"""Bundle real batch_experiment_{a,b,c}.jsonl results (and video preview
manifest, if present) into a data.js file the dashboard can load via a plain
<script src="data.js"> tag -- this works when index.html is opened directly
via file:// (double-click), unlike fetch(), which browsers block for local
files -- so the dashboard shows real data with no server needed.

Drops delta_xy_per_waypoint arrays (not used by any chart -- traj_xy/ade_m/
endpoint_m already cover what the dashboard renders) to keep the bundle small.

Usage:
    python3 masking/dashboard/generate_data_js.py \
        --a batch_experiment_a.jsonl --b batch_experiment_b.jsonl --c batch_experiment_c.jsonl \
        --videos masking/dashboard/videos/manifest.json \
        --out masking/dashboard/data.js
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_trimmed(path: str, kind: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            r.pop("delta_xy_per_waypoint", None)
            if kind == "b":
                for w in r.get("per_word_salience_top20", []):
                    w.pop("delta_xy_per_waypoint", None)
            if kind == "c":
                for entry in r.get("prefix_sweep", []) + r.get("suffix_sweep", []):
                    entry.pop("delta_xy_per_waypoint", None)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--videos", default=None, help="Path to videos/manifest.json, if previews exist")
    ap.add_argument("--out", default="masking/dashboard/data.js")
    args = ap.parse_args()

    data = {
        "a": _load_trimmed(args.a, "a"),
        "b": _load_trimmed(args.b, "b"),
        "c": _load_trimmed(args.c, "c"),
        "videos": [],
    }
    if args.videos and Path(args.videos).exists():
        data["videos"] = json.loads(Path(args.videos).read_text())

    out = Path(args.out)
    out.write_text("window.REAL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.0f} KB) -- a={len(data['a'])} b={len(data['b'])} c={len(data['c'])} "
          f"videos={len(data['videos'])} rows")


if __name__ == "__main__":
    main()
