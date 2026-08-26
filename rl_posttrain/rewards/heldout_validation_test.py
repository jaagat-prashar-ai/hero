# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rl_posttrain.rewards.heldout_validation import _base_index


class _Base:
    pass


class _ForeignSubset:
    """Subset interface without inheriting this process's torch Subset class."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices


def test_base_index_unwraps_foreign_and_nested_subset_interfaces():
    base = _Base()
    inner = _ForeignSubset(base, [8, 3, 5])
    outer = _ForeignSubset(inner, [2, 0])

    resolved, idx = _base_index(outer, 0)

    assert resolved is base
    assert idx == 5
