# SPDX-License-Identifier: Apache-2.0
"""
rule_based_generator_test.py — unit tests for dpo_mechanical's mechanical
substitution rules. Pure functions only, no model/API, per the repo's
no-mocked-model-or-API-tests rule (there's nothing to mock here — this
generator makes no external calls at all).
"""

from __future__ import annotations

from dpo_pairs.perturbation_generator_v2 import select_targets, validate_perturbation

from dpo_mechanical.rule_based_generator import (
    SUPPORTED_TYPES,
    generate_all_perturbations_rule_based,
    generate_perturbation_rule_based,
)

_TRACE = (
    "Nudge left due to construction cones blocking the right side of our lane, "
    "then accelerate to 15 m/s."
)


class TestDirectionFlip:
    def test_flips_the_commitment_direction_only(self):
        target = select_targets(_TRACE)["commitment_direction_flip"]
        result = generate_perturbation_rule_based(_TRACE, "commitment_direction_flip", target)
        assert result["original_span"] == "left"
        assert result["perturbed_span"] == "right"
        # The OTHER "right" (perceptual clause) must be left untouched.
        assert result["perturbed_trace"] == _TRACE.replace("Nudge left", "Nudge right", 1)
        assert validate_perturbation(_TRACE, result) is None

    def test_preserves_capitalization(self):
        trace = "Turn Left due to a stopped vehicle ahead, then proceed."
        target = select_targets(trace)["commitment_direction_flip"]
        result = generate_perturbation_rule_based(trace, "commitment_direction_flip", target)
        assert result["perturbed_span"] == "Right"


class TestStateFlip:
    def test_flips_blocking_to_clearing(self):
        target = select_targets(_TRACE)["perceptual_state_flip"]
        result = generate_perturbation_rule_based(_TRACE, "perceptual_state_flip", target)
        assert result["original_span"] == "blocking"
        assert result["perturbed_span"] == "clearing"
        assert validate_perturbation(_TRACE, result) is None

    def test_no_table_entry_skips_gracefully(self):
        # "stopped" is a real STATE_PATTERNS key with no STATE_FLIP entry.
        trace = "Wait due to a stopped vehicle ahead, then proceed."
        target = select_targets(trace)["perceptual_state_flip"]
        assert target["details"]["state"] == "stopped"
        assert generate_perturbation_rule_based(trace, "perceptual_state_flip", target) is None

    def test_bare_adjective_inflection_skipped_not_misgrammared(self):
        # "clear" matches the "clearing" key's pattern (which also covers
        # clear/cleared/clears/clearing) but isn't the gerund STATE_FLIP
        # assumes — swapping it in would read as "the lane is blocking".
        trace = "Keep lane since the lane is clear ahead."
        target = select_targets(trace)["perceptual_state_flip"]
        assert target["details"]["state"] == "clearing"
        assert generate_perturbation_rule_based(trace, "perceptual_state_flip", target) is None


class TestEntitySwap:
    def test_swaps_pedestrian_for_cyclist_preserving_plural(self):
        trace = "Yield due to pedestrians crossing ahead, then proceed."
        target = select_targets(trace)["perceptual_entity_swap"]
        result = generate_perturbation_rule_based(trace, "perceptual_entity_swap", target)
        assert result["original_span"] == "pedestrians"
        assert result["perturbed_span"] == "cyclists"
        assert validate_perturbation(trace, result) is None

    def test_no_table_entry_skips_gracefully(self):
        # "work_zone" is a real ENTITY_PATTERNS key with no ENTITY_SWAP entry.
        trace = "Decelerate due to a work zone ahead, then proceed."
        target = select_targets(trace)["perceptual_entity_swap"]
        assert target["details"]["entity"] == "work_zone"
        assert generate_perturbation_rule_based(trace, "perceptual_entity_swap", target) is None

    def test_scooter_rider_stranding_skipped(self):
        # The cyclist pattern's alternation matches just "scooter", stranding
        # "rider" -- swapping would produce "the pedestrian rider ahead".
        trace = "Nudge left due to the scooter rider ahead in our lane."
        target = select_targets(trace)["perceptual_entity_swap"]
        assert target["details"]["entity"] == "cyclist"
        assert generate_perturbation_rule_based(trace, "perceptual_entity_swap", target) is None

    def test_bare_scooter_without_rider_still_swaps(self):
        trace = "Keep distance to the scooter ahead since it is directly in our lane."
        target = select_targets(trace)["perceptual_entity_swap"]
        result = generate_perturbation_rule_based(trace, "perceptual_entity_swap", target)
        assert result["original_span"] == "scooter"
        assert result["perturbed_span"] == "pedestrian"


class TestQuantityEdit:
    def test_multiplies_value_by_ten_keeping_unit(self):
        target = select_targets(_TRACE)["quantity_edit"]
        result = generate_perturbation_rule_based(_TRACE, "quantity_edit", target)
        assert result["original_span"] == "15 m/s"
        assert result["perturbed_span"] == "150 m/s"
        assert validate_perturbation(_TRACE, result) is None

    def test_preserves_decimal_places_and_no_space(self):
        trace = "Decelerate due to 3.5% grade ahead, then proceed."
        target = select_targets(trace)["quantity_edit"]
        result = generate_perturbation_rule_based(trace, "quantity_edit", target)
        assert result["original_span"] == "3.5%"
        assert result["perturbed_span"] == "35.0%"


class TestUnsupportedTypes:
    def test_maneuver_swap_and_causal_substitution_not_attempted(self):
        assert "commitment_maneuver_swap" not in SUPPORTED_TYPES
        assert "causal_cause_substitution" not in SUPPORTED_TYPES
        target = select_targets(_TRACE)["commitment_maneuver_swap"]
        assert generate_perturbation_rule_based(_TRACE, "commitment_maneuver_swap", target) is None


class TestGenerateAll:
    def test_end_to_end_on_two_scenes(self):
        gts = [
            {"scene_id": "sceneA", "event_cluster": "WORK_ZONES_TEMP_TRAFFIC_CONTROL",
             "trace": _TRACE},
            {"scene_id": "sceneB", "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
             "trace": "Yield due to pedestrians crossing ahead, then proceed."},
        ]
        rows, n_skipped = generate_all_perturbations_rule_based(gts)
        assert n_skipped >= 0
        assert all(r["perturbation_type"] in SUPPORTED_TYPES for r in rows)
        assert all(r["taxonomy_version"] == "rule_v1" for r in rows)
        scene_a_rows = [r for r in rows if r["scene_id"] == "sceneA"]
        assert {r["perturbation_type"] for r in scene_a_rows} == {
            "commitment_direction_flip", "perceptual_state_flip", "quantity_edit",
        }
        for r in rows:
            assert validate_perturbation(r["ground_truth_trace"], r) is None
