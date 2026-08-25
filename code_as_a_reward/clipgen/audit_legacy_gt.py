# SPDX-License-Identifier: Apache-2.0
"""Quarantined own-GT audit for legacy ClipGen reward functions.

This tool never publishes or activates a legacy function. It evaluates each
legacy source against the current intact NVIDIA CoC/action pair and the exact
current GT-derived perturbation battery, then reports both the simple
POS_MIN/delta conditions and the stricter full gate result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from code_as_a_reward.clipgen import gate as gate_mod
from code_as_a_reward.clipgen.run_prototype import _load_clip
from code_as_a_reward.clipgen.target_contract import (
    derive_target_contract,
    validate_gt_target,
)


def _load_manifests(directories: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for raw in directories:
        directory = Path(raw)
        manifest = json.loads((directory / "manifest.json").read_text())
        targets_path = directory / "targets.json"
        targets = (
            {str(row["clip_id"]): int(row["t0_us"]) for row in json.loads(targets_path.read_text())}
            if targets_path.exists()
            else {}
        )
        for original in manifest:
            entry = dict(original)
            clip_id = str(entry["clip_id"])
            if clip_id in entries:
                raise ValueError(f"duplicate manifest clip {clip_id}")
            if "t0_us" not in entry:
                if clip_id not in targets:
                    raise ValueError(f"clip {clip_id}: missing t0_us")
                entry["t0_us"] = targets[clip_id]
            for key in (
                "obstacle_parquet",
                "egomotion_parquet",
                "gt_coc",
                "overlay_jpeg",
                "waypoints_npy",
            ):
                if entry.get(key) and not Path(entry[key]).is_absolute():
                    entry[key] = str(directory / entry[key])
            entries[clip_id] = entry
    return entries


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit(
    reward_fns_dir: str,
    manifest_dirs: list[str],
    out_json: str,
) -> dict[str, Any]:
    sources = {path.stem: path for path in Path(reward_fns_dir).glob("*.py")}
    manifests = _load_manifests(manifest_dirs)
    records: list[dict[str, Any]] = []

    for clip_id, path in sorted(sources.items()):
        entry = manifests.get(clip_id)
        if entry is None:
            records.append(
                {
                    "clip_id": clip_id,
                    "manifest_available": False,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            continue
        record: dict[str, Any] = {
            "clip_id": clip_id,
            "manifest_available": True,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "own_gt_positive_passed": False,
            "own_gt_delta_passed": False,
            "full_gate_passed": False,
        }
        try:
            clip = _load_clip(entry, None)
            contract = derive_target_contract(clip["gt_claims"], clip["gt_traj"])
            target_failures = validate_gt_target(contract, clip["gt_traj"])
            cases = gate_mod.build_perturbations(
                clip_id,
                clip["gt_claims"],
                np.asarray(clip["gt_waypoints"], dtype=np.float64),
                clip["hz"],
                tag="legacy_own_gt_audit",
                reward_spec=None,
            )
            result = gate_mod.run_gate(path.read_text(), cases)
            delta = float(result.pos_score - result.max_pert)
            positive_passed = bool(
                math.isfinite(result.pos_score) and result.pos_score >= gate_mod.POS_MIN
            )
            delta_passed = bool(
                positive_passed
                and math.isfinite(delta)
                and delta >= gate_mod.MIN_DROP - 1e-9
            )
            record.update(
                {
                    "valid_gt_target": not target_failures,
                    "gt_target_failures": target_failures,
                    "pos_score": result.pos_score,
                    "max_pert": result.max_pert,
                    "delta": delta,
                    "scores": result.scores,
                    "own_gt_positive_passed": positive_passed,
                    "own_gt_delta_passed": delta_passed,
                    "full_gate_passed": bool(result.passed),
                    "full_gate_failures": result.failures,
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    evaluable = [row for row in records if row["manifest_available"]]
    valid = [row for row in evaluable if row.get("valid_gt_target") is True]

    def counts(rows: list[dict[str, Any]]) -> dict[str, int | float]:
        positive = sum(bool(row.get("own_gt_positive_passed")) for row in rows)
        delta = sum(bool(row.get("own_gt_delta_passed")) for row in rows)
        full = sum(bool(row.get("full_gate_passed")) for row in rows)
        total = len(rows)
        return {
            "total": total,
            "own_gt_positive_passed": positive,
            "own_gt_positive_rate": _rate(positive, total),
            "own_gt_delta_passed": delta,
            "own_gt_delta_rate": _rate(delta, total),
            "full_gate_passed": full,
            "full_gate_rate": _rate(full, total),
        }

    report = {
        "schema_version": "clipgen.legacy-own-gt-audit.v1",
        "thresholds": {"pos_min": gate_mod.POS_MIN, "min_delta": gate_mod.MIN_DROP},
        "n_legacy_functions": len(sources),
        "n_manifest_clips": len(manifests),
        "n_evaluable_overlap": len(evaluable),
        "n_missing_manifest": len(sources) - len(evaluable),
        "all_evaluable": counts(evaluable),
        "valid_gt_only": counts(valid),
        "records": records,
    }
    Path(out_json).write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reward_fns_dir")
    parser.add_argument("out_json")
    parser.add_argument("manifest_dirs", nargs="+")
    args = parser.parse_args()
    report = audit(args.reward_fns_dir, args.manifest_dirs, args.out_json)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
