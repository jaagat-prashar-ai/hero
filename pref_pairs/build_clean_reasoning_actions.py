# SPDX-License-Identifier: Apache-2.0
"""
build_clean_reasoning_actions.py — standalone (reasoning, action) file, one
entry per scene, with no perturbation data attached.

This is a pure extraction from pref_pairs/results/perturbation_actions/
ground_truth_actions.json -- NOT a new model run. That file already joins
each scene's ground-truth reasoning (extracted from the fixed-reasoning
noise-floor workload pref-pairs-fixed-reasoning-cluster-f6kq5o) with the
SAME rollout's real action (waypoints, native accel/curvature, scalar
features) -- see build_ground_truth_action_dataset.py's module docstring.
This module just drops the "perturbations" nesting so a reader only
interested in (reasoning, action) pairs -- not the perturbation-generator
work -- has one clean, self-contained file scoped to exactly that.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_clean_entries(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": scene["scene_id"],
            "event_cluster": scene["event_cluster"],
            "ground_truth_trace": scene["ground_truth_trace"],
            "ground_truth_action": scene["ground_truth_action"],
        }
        for scene in scenes
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in_path",
        default="pref_pairs/results/perturbation_actions/ground_truth_actions.json",
    )
    ap.add_argument(
        "--out_path",
        default="pref_pairs/results/clean_reasoning_actions/clean_reasoning_actions.json",
    )
    args = ap.parse_args()

    with open(args.in_path) as fh:
        scenes = json.load(fh)
    entries = build_clean_entries(scenes)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    logger.info("Wrote %d clean (reasoning, action) entries to %s", len(entries), out_path)


if __name__ == "__main__":
    main()
