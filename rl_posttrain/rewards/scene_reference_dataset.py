# SPDX-License-Identifier: Apache-2.0
"""
scene_reference_dataset.py -- AlpamayoCosmosDataset subclass whose
get_reference_answer additionally ships the judge's scene payload
(scene_frame_jpeg / scene_cam_intr / scene_cam_extr, built by
scene_overlay.build_scene_reference) alongside the vendored trajectory keys.

Why a subclass with a COPIED body instead of super() + extras: the vendored
get_reference_answer performs the expensive self.dataset[idx] fetch (per-clip
video decode); calling super() and then fetching again to reach image_frames
would double that cost for every reward evaluation. Same
copy-with-one-block-changed convention as aggregated_reward_llm_judge.py --
the vendored submodule is never edited, and the supported extension point is
our own entry script (llm_judge_entry.py), which passes this class to
cosmos-rl's launch_worker instead of the vendored AlpamayoCosmosDataset.

The scene keys are REQUIRED downstream (the reward fails loud without them),
so this class raises rather than returning a scene-less reference when
calibration or frames are missing from the sample -- with include_extr_intr
set in the recipe's hydra config (alpamayo1_5_rvla_rl_pai.yaml does), that
only happens on a genuinely broken dataset.

Kept in its own importable module (not defined inside the entry script):
cosmos-rl may pickle dataset objects into worker processes, and classes
defined in a __main__-executed entry file don't survive unpickling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alpamayo1_x_rl.base_dataset import AlpamayoCosmosDataset  # noqa: E402
from cosmos_rl.utils.logging import logger  # noqa: E402  # pyright: ignore[reportMissingImports]

from rl_posttrain.rewards.scene_overlay import build_scene_reference  # noqa: E402


class SceneReferenceDataset(AlpamayoCosmosDataset):
    """AlpamayoCosmosDataset + the LLM judge's scene payload in the reference."""

    def get_reference_answer(self, idx: int) -> dict[str, Any]:
        """Vendored body (alpamayo1_x_rl/base_dataset.py) verbatim, plus the
        scene keys from the same already-fetched sample."""
        try:
            sample = self.dataset[idx]
        except Exception as e:
            logger.error(f"[SceneReferenceDataset] Error getting reference answer: {e}")
            return {}
        if not isinstance(sample, dict) or "ego_future_xyz" not in sample:
            return {}
        reference = {
            "ego_future_xyz": sample["ego_future_xyz"],
            "ego_future_rot": sample["ego_future_rot"],
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
            "egomotion_road_boundaries": sample.get("egomotion_road_boundaries", None),
            "egomotion_lanelines": sample.get("egomotion_lanelines", None),
            "ego_lwh": sample.get("ego_lwh", None),
            "ego_length_offset": sample.get("ego_length_offset", None),
            "obstacle_bbox_history": sample.get("obstacle_bbox_history", None),
            "obstacle_bbox_future": sample.get("obstacle_bbox_future", None),
            "cot": sample.get("cot", ""),
        }
        from rl_posttrain.rewards.heldout_validation import sample_identity

        clip_id, t0_us, future_hz = sample_identity(self.dataset, idx)
        reference["scene_id"] = f"{clip_id}_{t0_us}"
        reference["future_hz"] = future_hz
        reference.update(build_scene_reference(sample))
        return reference
