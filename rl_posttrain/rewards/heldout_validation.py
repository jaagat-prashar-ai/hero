# SPDX-License-Identifier: Apache-2.0
"""Deterministic clip-level train/validation split for matched reward ablations."""

from __future__ import annotations

import copy
import functools
import hashlib
import os
from typing import Any


def _validation_rollout_count(prompt_count: int, n_generation: int) -> int:
    """Return the completion count emitted by a Cosmos validation pass."""
    if prompt_count < 0:
        raise ValueError(f"prompt_count must be nonnegative, got {prompt_count}")
    if n_generation < 1:
        raise ValueError(f"n_generation must be positive, got {n_generation}")
    return prompt_count * n_generation


def install_cosmos_validation_compat() -> None:
    """Fix validation completion accounting in the pinned Cosmos revision.

    Cosmos stores ``val_datasize`` as the number of prompts, while validation
    reports contain one rollout for every generated completion.  With
    ``validation.n_generation > 1`` its controller therefore waits forever:
    for example, 119 prompts produce 714 reported rollouts, which can never
    equal 119.  Keep the richer multi-generation validation and make the
    controller's expected cardinality use the same unit as its reports.

    The pinned controller also advances tqdm by the cumulative result count
    on every report.  The status wrapper converts that one update to a delta
    without changing validation results or completion behavior.
    """
    from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
    from cosmos_rl.dispatcher.status import PolicyStatusManager

    activate = ControllerDataFetcher.validation_activate_dataloader
    if not getattr(activate, "_hero_validation_cardinality_patch", False):

        @functools.wraps(activate)
        def activate_with_completion_count(self: Any, validation_step: int):
            base_size = getattr(self, "_hero_validation_prompt_count", None)
            if base_size is None:
                base_size = int(self.val_datasize or 0)
                self._hero_validation_prompt_count = base_size
            validation = getattr(self.config, "validation", None)
            n_generation = int(getattr(validation, "n_generation", 1))
            self.val_datasize = _validation_rollout_count(base_size, n_generation)
            return activate(self, validation_step)

        activate_with_completion_count._hero_validation_cardinality_patch = True
        ControllerDataFetcher.validation_activate_dataloader = (
            activate_with_completion_count
        )

    report = PolicyStatusManager.validation_report_validation_results
    if not getattr(report, "_hero_validation_progress_patch", False):

        @functools.wraps(report)
        def report_with_delta_progress(
            self: Any,
            validation_step: int,
            validation_results: Any,
            rollout_status_manager: Any,
        ):
            previous = sum(
                len(group)
                for group in self.val_report_data.get(validation_step, [])
            )
            progress = self.data_fetcher.activated_val_tqdm
            original_update = None
            if progress is not None:
                original_update = progress.update

                def update_delta(cumulative: int = 1) -> Any:
                    return original_update(max(0, int(cumulative) - previous))

                progress.update = update_delta
            try:
                return report(
                    self,
                    validation_step,
                    validation_results,
                    rollout_status_manager,
                )
            finally:
                if (
                    progress is not None
                    and original_update is not None
                    and self.data_fetcher.activated_val_tqdm is progress
                ):
                    progress.update = original_update

        report_with_delta_progress._hero_validation_progress_patch = True
        PolicyStatusManager.validation_report_validation_results = (
            report_with_delta_progress
        )

def _base_index(dataset: Any, idx: int) -> tuple[Any, int]:
    """Unwrap nested Subset-like views and preserve the base PAI index.

    Ray can load the recipe and this module through different import paths,
    producing two class objects for ``torch.utils.data.Subset``.  A strict
    ``isinstance`` check then misses a perfectly ordinary Subset.  Its public
    ``dataset``/``indices`` interface is the stable contract we need here.
    """
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        idx = int(dataset.indices[idx])
        dataset = dataset.dataset
    return dataset, idx


def sample_identity(dataset: Any, idx: int) -> tuple[str, int, float]:
    """Return (clip_id, t0_us, future_hz) through an optional Subset wrapper."""
    base, base_idx = _base_index(dataset, idx)
    clip_id = str(base.clip_ids[base_idx])
    t0_us = int(
        base.DEFAULT_T0_US
        if base.use_default_keyframe
        else base.avdi.get_clip_key_frame(clip_id)
    )
    return clip_id, t0_us, 1.0 / float(base.time_step)


def _is_validation_clip(clip_id: str, *, fraction: float, seed: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < fraction


def install_heldout_validation_split(dataloaders: Any) -> tuple[int, int]:
    """Replace the train loader with disjoint deterministic train/val views.

    The split is keyed by clip id rather than row order, so all reward arms use
    exactly the same clips even when dataloader ordering changes.  It is only
    activated when ``FAITHFULNESS_VAL_FRACTION`` is positive.
    """
    from torch.utils.data import Subset

    fraction = float(os.getenv("FAITHFULNESS_VAL_FRACTION", "0"))
    if fraction <= 0:
        return len(dataloaders["train"].dataset), 0
    if not 0 < fraction < 1:
        raise ValueError(f"FAITHFULNESS_VAL_FRACTION must be in (0, 1), got {fraction}")

    seed = os.getenv("FAITHFULNESS_VAL_SEED", "20260825")
    loader = dataloaders["train"]
    base = loader.dataset
    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx in range(len(base)):
        clip_id, _t0_us, _hz = sample_identity(base, idx)
        (val_indices if _is_validation_clip(clip_id, fraction=fraction, seed=seed) else train_indices).append(idx)
    if not train_indices or not val_indices:
        raise RuntimeError(
            f"deterministic split produced train={len(train_indices)} val={len(val_indices)}"
        )

    train_loader = copy.copy(loader)
    val_loader = copy.copy(loader)
    object.__setattr__(train_loader, "dataset", Subset(base, train_indices))
    object.__setattr__(val_loader, "dataset", Subset(base, val_indices))
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader
    return len(train_indices), len(val_indices)
