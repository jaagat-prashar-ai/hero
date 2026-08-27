# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import types

from rl_posttrain.rewards.heldout_validation import (
    _base_index,
    _validation_rollout_count,
    install_cosmos_validation_compat,
)


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


def test_validation_rollout_count_uses_completion_cardinality():
    assert _validation_rollout_count(119, 6) == 714
    assert _validation_rollout_count(119, 1) == 119


def test_cosmos_validation_compat_scales_once_and_is_idempotent(monkeypatch):
    class _Progress:
        def __init__(self):
            self.updates = []

        def update(self, value=1):
            self.updates.append(value)

    class _Fetcher:
        def validation_activate_dataloader(self, validation_step):
            self.activated_step = validation_step

    class _Status:
        def validation_report_validation_results(
            self, validation_step, validation_results, rollout_status_manager
        ):
            self.val_report_data.setdefault(validation_step, []).extend(
                validation_results
            )
            cumulative = sum(
                len(group) for group in self.val_report_data[validation_step]
            )
            self.data_fetcher.activated_val_tqdm.update(cumulative)

    data_module = types.ModuleType("cosmos_rl.dispatcher.data.data_fetcher")
    data_module.ControllerDataFetcher = _Fetcher
    status_module = types.ModuleType("cosmos_rl.dispatcher.status")
    status_module.PolicyStatusManager = _Status
    monkeypatch.setitem(
        sys.modules, "cosmos_rl.dispatcher.data.data_fetcher", data_module
    )
    monkeypatch.setitem(sys.modules, "cosmos_rl.dispatcher.status", status_module)

    install_cosmos_validation_compat()
    install_cosmos_validation_compat()

    fetcher = _Fetcher()
    fetcher.val_datasize = 119
    fetcher.config = types.SimpleNamespace(
        validation=types.SimpleNamespace(n_generation=6)
    )
    fetcher.activated_val_tqdm = _Progress()
    fetcher.validation_activate_dataloader(25)
    fetcher.validation_activate_dataloader(50)
    assert fetcher.val_datasize == 714

    status = _Status()
    status.val_report_data = {}
    status.data_fetcher = fetcher
    status.validation_report_validation_results(25, [[1] * 6] * 64, None)
    status.validation_report_validation_results(25, [[1] * 6] * 55, None)
    assert fetcher.activated_val_tqdm.updates == [384, 330]
