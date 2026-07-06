# SPDX-License-Identifier: Apache-2.0
"""
maneuver_report_test.py — integration test: writes synthetic scene JSON
files in Task 1's on-disk RolloutHarvester format to a temp directory, then
exercises the full classify_directory -> report pipeline end to end. This
also doubles as a smoke test of the on-disk format contract between
rollout_harvester.py and classify_maneuvers.py, which we can't otherwise
test against a real GPU rollout in this environment.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pref_pairs.classify_maneuvers import FeatureConfig, ManeuverConfig, classify_directory
from pref_pairs.maneuver_report import (
    class_distribution_report,
    render_scene_spot_checks,
    threshold_sensitivity_report,
)
from pref_pairs.synthetic_trajectory_fixtures import (
    HZ,
    accelerate,
    lane_change,
    ramp_to_stop,
    straight_line,
    turn,
    yield_dip,
)

RAW_CONFIG = {
    "smoothing": {"window": 3},
    "initial_heading": {"window": 3},
    "stop": {"speed_mps": 0.5, "duration_s": 1.0, "recovery_speed_mps": 2.0},
    "turn": {"heading_change_deg": 45.0},
    "lane_change": {"lateral_offset_m": 2.5, "heading_change_deg": 45.0},
    "yield": {"drop_fraction": 0.30, "min_speed_mps": 2.0, "recovery_fraction": 0.50},
    "proceed_accelerate": {"mean_accel_mps2": 0.5},
    "ambiguous_margin_fraction": 0.10,
}


def _write_scene(out_dir: Path, scene_id: str, waypoints_by_rollout: dict[int, "np.ndarray"]) -> None:
    records = [
        {
            "scene_id": scene_id,
            "rollout_id": rollout_id,
            "coc_text": f"synthetic rollout {rollout_id}",
            "waypoints": wp.tolist(),
            "hz": HZ,
            "sampling_params": {"seed": 0, "temperature": 0.6, "top_p": 0.98, "top_k": None, "k": len(waypoints_by_rollout)},
            "model_version": "test-fixture",
            "ground_truth_coc": None,
        }
        for rollout_id, wp in waypoints_by_rollout.items()
    ]
    (out_dir / f"{scene_id}.json").write_text(json.dumps(records))


def _build_synthetic_rollouts_dir(tmp_dir: Path) -> Path:
    """Two scenes: one where every rollout is the SAME maneuver (uniform),
    one where rollouts span several different classes (diverse) -- so
    class_distribution_report's uniform/diverse detection has both cases to
    find."""
    rollouts_dir = tmp_dir / "rollouts"
    rollouts_dir.mkdir()

    _write_scene(
        rollouts_dir, "scene_uniform",
        {i: straight_line() for i in range(5)},  # all 5 rollouts are lane_keep
    )
    _write_scene(
        rollouts_dir, "scene_diverse",
        {
            0: straight_line(),
            1: ramp_to_stop(),
            2: lane_change(),
            3: turn(),
            4: yield_dip(),
            5: accelerate(),
        },
    )
    return rollouts_dir


def test_class_distribution_report_flags_uniform_and_diverse_scenes():
    with tempfile.TemporaryDirectory() as tmp:
        rollouts_dir = _build_synthetic_rollouts_dir(Path(tmp))
        labels_df, _ = classify_directory(
            rollouts_dir, FeatureConfig.from_dict(RAW_CONFIG), ManeuverConfig.from_dict(RAW_CONFIG)
        )
        report = class_distribution_report(labels_df)

        assert report["n_scenes"] == 2
        assert report["n_rollouts"] == 5 + 6
        assert "scene_uniform" in report["uniform_scenes"]
        assert "scene_diverse" not in report["uniform_scenes"]
        # scene_diverse should be the most diverse (6 distinct classes >
        # scene_uniform's 1).
        most_diverse_ids = [scene_id for scene_id, _ in report["most_diverse_scenes"]]
        assert most_diverse_ids[0] == "scene_diverse"


def test_threshold_sensitivity_report_has_one_row_per_threshold_per_direction():
    with tempfile.TemporaryDirectory() as tmp:
        rollouts_dir = _build_synthetic_rollouts_dir(Path(tmp))
        labels_df, _ = classify_directory(
            rollouts_dir, FeatureConfig.from_dict(RAW_CONFIG), ManeuverConfig.from_dict(RAW_CONFIG)
        )
        sensitivity_df = threshold_sensitivity_report(rollouts_dir, RAW_CONFIG, labels_df)

        # 10 perturbable thresholds x 2 directions (+20%/-20%).
        assert len(sensitivity_df) == 20
        assert set(sensitivity_df.columns) == {"threshold", "direction", "pct_flipped"}
        assert sensitivity_df["pct_flipped"].between(0, 100).all()


def test_render_scene_spot_checks_writes_one_png_per_sampled_scene():
    with tempfile.TemporaryDirectory() as tmp:
        rollouts_dir = _build_synthetic_rollouts_dir(Path(tmp))
        labels_df, _ = classify_directory(
            rollouts_dir, FeatureConfig.from_dict(RAW_CONFIG), ManeuverConfig.from_dict(RAW_CONFIG)
        )
        out_dir = Path(tmp) / "plots"
        written = render_scene_spot_checks(rollouts_dir, labels_df, out_dir, n_scenes=10, seed=0)

        # Only 2 scenes exist even though n_scenes=10 was requested.
        assert len(written) == 2
        for path in written:
            assert path.exists()
            assert path.stat().st_size > 0
