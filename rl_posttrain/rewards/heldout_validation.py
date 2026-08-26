# SPDX-License-Identifier: Apache-2.0
"""Deterministic clip-level train/validation split for matched reward ablations."""

from __future__ import annotations

import copy
import hashlib
import os
from typing import Any

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
