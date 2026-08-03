# SPDX-License-Identifier: Apache-2.0
"""
perturbation_generator_v2_test.py — unit tests for the PURE helpers of
dpo_pairs.perturbation_generator_v2: claim-target selection, mechanical
validation, and the cost estimate. The API-calling path is deliberately NOT
mocked (project rule) — it is verified by a real --max_scenes 2 smoke run
before any full paid batch.
"""

from __future__ import annotations

from dpo_pairs.perturbation_generator_v2 import (
    PERTURBATION_TYPES_V2,
    estimate_cost,
    select_targets,
    validate_perturbation,
)

_TRACE = (
    "Nudge left due to construction cones blocking the right side of our lane, "
    "then accelerate to 15 m/s."
)


class TestSelectTargets:
    def test_rich_trace_gets_most_types(self):
        targets = select_targets(_TRACE)
        assert set(targets) <= set(PERTURBATION_TYPES_V2)
        # This trace has: a directional commitment (nudge left), maneuvers,
        # a stateful perceptual claim (cones blocking), a causal link
        # (due to), and a quantity (15 m/s).
        for expected in ("commitment_direction_flip", "commitment_maneuver_swap",
                         "perceptual_state_flip", "perceptual_entity_swap",
                         "causal_cause_substitution", "quantity_edit"):
            assert expected in targets, expected

    def test_targets_carry_spans_into_trace(self):
        targets = select_targets(_TRACE)
        for ptype, target in targets.items():
            s, e = target["span"]
            assert _TRACE[s:e] == target["claim_text"] or target["claim_type"] == "quantity", ptype

    def test_quantity_span_matches_regex(self):
        target = select_targets(_TRACE)["quantity_edit"]
        s, e = target["span"]
        assert _TRACE[s:e] == "15 m/s" == target["claim_text"]

    def test_bare_trace_gets_no_inapplicable_types(self):
        targets = select_targets("Proceed.")
        assert "commitment_direction_flip" not in targets  # no direction word
        assert "quantity_edit" not in targets              # no number
        assert "causal_cause_substitution" not in targets  # no connective

    def test_deterministic(self):
        assert select_targets(_TRACE) == select_targets(_TRACE)


class TestValidatePerturbation:
    def _result(self, **kw):
        base = {
            "original_span": "left",
            "perturbed_span": "right",
            "perturbed_trace": _TRACE.replace("Nudge left", "Nudge right", 1),
        }
        base.update(kw)
        return base

    def test_clean_substitution_passes(self):
        assert validate_perturbation(_TRACE, self._result()) is None

    def test_span_not_in_trace(self):
        r = self._result(original_span="oncoming bus")
        assert "not found" in validate_perturbation(_TRACE, r)

    def test_identity_edit_rejected(self):
        r = self._result(perturbed_span="left",
                         perturbed_trace=_TRACE)
        assert "identity" in validate_perturbation(_TRACE, r)

    def test_out_of_span_edit_rejected(self):
        # Model flipped the direction AND "helpfully" repaired downstream text.
        bad = _TRACE.replace("Nudge left", "Nudge right", 1).replace("accelerate", "decelerate")
        r = self._result(perturbed_trace=bad)
        assert "outside the edited span" in validate_perturbation(_TRACE, r)

    def test_wordy_replacement_rejected(self):
        r = self._result(
            perturbed_span="right, though it appears possibly unclear whether that is safe",
            perturbed_trace=_TRACE.replace(
                "left", "right, though it appears possibly unclear whether that is safe", 1),
        )
        assert "word-count" in validate_perturbation(_TRACE, r)

    def test_repeated_span_any_occurrence_ok(self):
        trace = "Slow down. Slow down again."
        r = {
            "original_span": "Slow down",
            "perturbed_span": "Speed up",
            "perturbed_trace": "Slow down. Speed up again.",  # second occurrence edited
        }
        assert validate_perturbation(trace, r) is None


class TestEstimateCost:
    def test_zero_calls_zero_cost(self):
        assert estimate_cost(0) == 0.0

    def test_pilot_scale_is_a_few_dollars(self):
        # ~720 calls (120 scenes x ~6 applicable types) must land in the
        # announced "order of a few dollars" envelope, not tens.
        cost = estimate_cost(720)
        assert 1.0 < cost < 15.0, cost

    def test_monotone_in_calls(self):
        assert estimate_cost(100) < estimate_cost(700)
