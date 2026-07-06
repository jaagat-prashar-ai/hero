# SPDX-License-Identifier: Apache-2.0
"""
classify_maneuvers.py — rule-based maneuver-class labeling for Alpamayo
rollouts. Consumes trajectory_features.py's per-rollout kinematic features
and assigns exactly one maneuver_class per rollout via a priority-ordered
rule cascade (first match wins). No learning here, by design (see task
Non-goals) -- every threshold lives in configs/maneuver_thresholds.yaml.

Sign convention (please verify against real data before trusting this):
  We assume the standard robotics ego-frame convention -- x forward, y LEFT,
  z up (ISO 8855-style) -- so POSITIVE lateral offset and POSITIVE heading
  change (CCW) both mean LEFT. Neither this codebase's action-space code
  nor its docstrings state the sign of curvature/heading explicitly, so
  this is an assumption, not a verified fact. The top-down PNG spot-check
  this module's report produces (matplotlib, trajectories colored by
  label) is exactly the mechanism to catch a left/right sign flip -- check
  a handful of turn_left/turn_right-labeled plots by eye before trusting
  the direction labels for anything downstream.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pref_pairs.trajectory_features import FeatureConfig, TrajectoryFeatures, extract_features

logger = logging.getLogger(__name__)

MANEUVER_CLASSES = (
    "lane_keep",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
    "stop",
    "yield",
    "proceed/accelerate",
)


@dataclasses.dataclass
class ManeuverConfig:
    """The rule-cascade thresholds -- the subset of maneuver_thresholds.yaml
    that trajectory_features.py does NOT already own (that module owns
    smoothing/initial_heading/stop/yield; see its FeatureConfig)."""

    turn_heading_change_deg: float = 45.0
    lane_change_lateral_offset_m: float = 2.5
    lane_change_heading_change_deg: float = 45.0
    proceed_accel_mps2: float = 0.5
    ambiguous_margin_fraction: float = 0.10

    # Kept here too (duplicated from FeatureConfig) only so `_boundary_margins`
    # can report a margin against the stop/yield thresholds without importing
    # FeatureConfig's full state -- the authoritative values for actually
    # detecting stop_event/yield_event live in trajectory_features.FeatureConfig.
    stop_speed_mps: float = 0.5
    yield_drop_fraction: float = 0.30

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManeuverConfig":
        return cls(
            turn_heading_change_deg=d.get("turn", {}).get("heading_change_deg", 45.0),
            lane_change_lateral_offset_m=d.get("lane_change", {}).get("lateral_offset_m", 2.5),
            lane_change_heading_change_deg=d.get("lane_change", {}).get(
                "heading_change_deg", 45.0
            ),
            proceed_accel_mps2=d.get("proceed_accelerate", {}).get("mean_accel_mps2", 0.5),
            ambiguous_margin_fraction=d.get("ambiguous_margin_fraction", 0.10),
            stop_speed_mps=d.get("stop", {}).get("speed_mps", 0.5),
            yield_drop_fraction=d.get("yield", {}).get("drop_fraction", 0.30),
        )


def load_configs(path: str | Path) -> tuple[FeatureConfig, ManeuverConfig]:
    """Load both configs from the one shared YAML file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return FeatureConfig.from_dict(raw), ManeuverConfig.from_dict(raw)


@dataclasses.dataclass
class ClassificationResult:
    maneuver_class: str
    # name -> signed relative margin to that rule's boundary, e.g.
    # (value - threshold) / threshold. Computed for EVERY rule regardless of
    # which one fired, so the ambiguous-case sidecar can flag a rollout that
    # is nearly a turn even though it defaulted to lane_keep.
    boundary_margins: dict[str, float]

    @property
    def min_abs_margin(self) -> float:
        return min(abs(m) for m in self.boundary_margins.values())


def classify(features: TrajectoryFeatures, config: ManeuverConfig) -> ClassificationResult:
    """Apply the priority-ordered rule cascade. First match wins; lane_keep
    is the unconditional default, so every rollout gets exactly one label
    (no NaNs/exceptions possible -- every branch below returns a string)."""
    margins = _boundary_margins(features, config)

    if features.stop_event:
        return ClassificationResult("stop", margins)

    heading = features.total_heading_change_deg
    if abs(heading) > config.turn_heading_change_deg:
        return ClassificationResult("turn_left" if heading > 0 else "turn_right", margins)

    lateral = features.final_lateral_offset_m
    if (
        abs(lateral) > config.lane_change_lateral_offset_m
        and abs(heading) < config.lane_change_heading_change_deg
    ):
        direction = "left" if lateral > 0 else "right"
        return ClassificationResult(f"lane_change_{direction}", margins)

    if features.yield_event:
        return ClassificationResult("yield", margins)

    if features.mean_acceleration_mps2 > config.proceed_accel_mps2:
        return ClassificationResult("proceed/accelerate", margins)

    return ClassificationResult("lane_keep", margins)


def _boundary_margins(features: TrajectoryFeatures, config: ManeuverConfig) -> dict[str, float]:
    """Signed relative margin of this rollout's features to each rule's
    threshold: (value - threshold) / threshold. Positive means "past" the
    threshold in the direction that rule cares about; the ambiguous-case
    sidecar flags rollouts where ANY of these is within
    ambiguous_margin_fraction of 0, regardless of which rule actually fired.

    stop/yield use the rollout's min_speed_mps / drop-fraction as a proxy for
    "how close to the stop/yield boundary" -- a simplification, since those
    two rules are actually defined by a compound condition (speed AND
    duration for stop; drop AND recovery for yield) that doesn't reduce to a
    single scalar distance. Good enough for the auditing purpose this sidecar
    serves (flagging "look at this one by eye"), not meant to be exact.
    """

    def rel_margin(value: float, threshold: float) -> float:
        return (value - threshold) / threshold if threshold != 0 else 0.0

    return {
        "stop_min_speed": rel_margin(config.stop_speed_mps, max(features.min_speed_mps, 1e-6)),
        "turn_heading": rel_margin(abs(features.total_heading_change_deg),
                                    config.turn_heading_change_deg),
        "lane_change_lateral": rel_margin(abs(features.final_lateral_offset_m),
                                           config.lane_change_lateral_offset_m),
        "lane_change_heading": rel_margin(config.lane_change_heading_change_deg,
                                           max(abs(features.total_heading_change_deg), 1e-6)),
        "proceed_accel": rel_margin(features.mean_acceleration_mps2, config.proceed_accel_mps2),
    }


def classify_directory(
    rollouts_dir: str | Path,
    feature_config: FeatureConfig,
    maneuver_config: ManeuverConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Classify every rollout in every {scene_id}.json file under
    rollouts_dir (Task 1's RolloutHarvester output format).

    Returns (labels_df, ambiguous_cases) -- ambiguous_cases is a list of
    dicts ready to write as the sidecar audit file.
    """
    rows: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    scene_files = sorted(Path(rollouts_dir).glob("*.json"))
    if not scene_files:
        logger.warning("classify_directory: no scene JSON files found in %s", rollouts_dir)

    for scene_path in scene_files:
        rollout_records = json.loads(scene_path.read_text())
        for record in rollout_records:
            features = extract_features(
                record["waypoints"],
                hz=record["hz"],
                scene_id=record["scene_id"],
                rollout_id=record["rollout_id"],
                config=feature_config,
            )
            result = classify(features, maneuver_config)

            row = features.to_row_dict()
            row["maneuver_class"] = result.maneuver_class
            rows.append(row)

            if result.min_abs_margin <= maneuver_config.ambiguous_margin_fraction:
                ambiguous.append(
                    {
                        "scene_id": features.scene_id,
                        "rollout_id": features.rollout_id,
                        "maneuver_class": result.maneuver_class,
                        "boundary_margins": result.boundary_margins,
                        "min_abs_margin": result.min_abs_margin,
                    }
                )

    labels_df = pd.DataFrame(rows)
    # Every rollout must get exactly one label -- fail loudly here rather
    # than silently shipping a partially-labeled table downstream.
    assert not labels_df.empty, "no rollouts found to classify"
    assert labels_df["maneuver_class"].notna().all(), "every rollout must get a non-null label"
    return labels_df, ambiguous


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts_dir", required=True, help="Directory of {scene_id}.json files.")
    ap.add_argument(
        "--config",
        default="pref_pairs/configs/maneuver_thresholds.yaml",
        help="Shared thresholds YAML (see FeatureConfig/ManeuverConfig).",
    )
    ap.add_argument("--out_parquet", default="pref_pairs/results/maneuver_labels.parquet")
    ap.add_argument(
        "--out_ambiguous",
        default="pref_pairs/results/maneuver_labels_ambiguous.jsonl",
        help="Sidecar file: rollouts within ambiguous_margin_fraction of a rule boundary.",
    )
    args = ap.parse_args()

    feature_config, maneuver_config = load_configs(args.config)
    labels_df, ambiguous = classify_directory(args.rollouts_dir, feature_config, maneuver_config)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %d labeled rollouts to %s", len(labels_df), out_parquet)

    out_ambiguous = Path(args.out_ambiguous)
    out_ambiguous.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ambiguous, "w") as fh:
        for case in ambiguous:
            fh.write(json.dumps(case) + "\n")
    logger.info(
        "Flagged %d/%d rollouts as ambiguous (within %.0f%% of a rule boundary) -> %s",
        len(ambiguous), len(labels_df), maneuver_config.ambiguous_margin_fraction * 100,
        out_ambiguous,
    )

    logger.info("Class distribution:\n%s", labels_df["maneuver_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
