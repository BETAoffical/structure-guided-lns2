from __future__ import annotations

import copy
import unittest
from unittest.mock import ANY, patch

from experiments.closed_loop_confirmation import (
    _generate_fixed_structshell_runtime_candidates,
)
from experiments.neighborhood_candidates import candidate_id
from lns2_selector.runtime.fixed_structshell import (
    validate_fixed_structshell_augmentation,
)
from lns2_selector.runtime.structshell_component16 import (
    STRUCTSHELL_COMPONENT16_POOL_ID,
    STRUCTSHELL_COMPONENT16_RUNTIME_ID,
    generate_structshell_component16_runtime_candidates,
    structshell_component16_ablation_gate,
    structshell_component16_augmentation,
    validate_structshell_component16_augmentation,
)
from lns2_selector.runtime.structshell_dual16 import (
    generate_structshell_dual16_runtime_candidates,
    structshell_dual16_augmentation,
)


COMPONENT_FAMILY = "structpool-conflict-component:16"
HOTSPOT_FAMILY = "structpool-spatiotemporal-hotspot:16"


def _candidate(agents: list[int], families: list[str]) -> dict:
    normalized = sorted(set(map(int, agents)))
    return {
        "candidate_id": candidate_id(normalized),
        "agents": normalized,
        "actual_size": len(normalized),
        "selection_families": list(families),
        "selection_rank_by_family": {family: 0 for family in families},
        "proposal_count_by_family": {family: 1 for family in families},
        "proposal_seeds": [],
        "seed_agents": [],
        "structpool_family_groups": list(families),
    }


class StructShellComponent16Test(unittest.TestCase):
    def test_exact_registered_contract_and_all_state_gate(self) -> None:
        expected = {
            "enabled": True,
            "pool_id": STRUCTSHELL_COMPONENT16_POOL_ID,
            "runtime_id": STRUCTSHELL_COMPONENT16_RUNTIME_ID,
            "source_pool_id": "stride-structshell-dual16-v1",
            "source_runtime_id": "stride-structshell-dual16-runtime-v3",
            "full_union_required": False,
            "full_union_audit_preserved": True,
            "runtime_filter_id": "component_family_fixed16_v1",
            "source_mode": "structshell_component16",
            "nominal_size": 16,
            "runtime_structural_family_sizes": {
                "conflict_component": [16],
                "spatiotemporal_hotspot": [16],
            },
            "retained_selection_family": COMPONENT_FAMILY,
            "maximum_generated_candidates": 2,
            "maximum_added_candidates": 1,
            "maximum_total_candidates": 64,
            "static_grid_cache": True,
            "activation_gate": {"gate_id": "component16_ablation_all_states"},
        }
        config = structshell_component16_augmentation()
        self.assertEqual(config, expected)
        self.assertEqual(validate_structshell_component16_augmentation(config), expected)
        self.assertEqual(validate_fixed_structshell_augmentation(config), expected)

        state = {
            "agents": [{"id": 0}, {"id": 1}],
            "conflict_edges": [[0, 1]],
            "num_of_colliding_pairs": 1,
            "done": False,
            "feasible": False,
        }
        gate = structshell_component16_ablation_gate(state, config)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["gate_id"], "component16_ablation_all_states")
        self.assertEqual(gate["reason"], "component16_ablation_all_states")

        changed = copy.deepcopy(config)
        changed["maximum_added_candidates"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported Component16"):
            validate_structshell_component16_augmentation(changed)

    @patch(
        "lns2_selector.runtime.structshell_component16."
        "generate_structpool_candidate_subset"
    )
    def test_distinct_component_and_hotspot_retains_only_component(
        self, subset_mock
    ) -> None:
        base = [_candidate([0, 1], ["v2"]), _candidate([2, 3], ["v2"])]
        frozen_base = copy.deepcopy(base)
        component = _candidate([4, 5], [COMPONENT_FAMILY])
        hotspot = _candidate([6, 7], [HOTSPOT_FAMILY])
        subset_mock.return_value = [component, hotspot]

        result = generate_structshell_component16_runtime_candidates(
            {"agents": [{"id": index} for index in range(8)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=base,
            config=structshell_component16_augmentation(),
        )

        subset_mock.assert_called_once_with(
            ANY,
            ANY,
            family_sizes={
                "conflict_component": (16,),
                "spatiotemporal_hotspot": (16,),
            },
        )
        self.assertEqual(base, frozen_base)
        self.assertEqual(result.challengers, [component])
        self.assertEqual(result.structural_candidate_count, 1)
        self.assertNotIn(
            hotspot["candidate_id"],
            {row["candidate_id"] for row in result.candidates},
        )

    @patch(
        "lns2_selector.runtime.structshell_component16."
        "generate_structpool_candidate_subset"
    )
    def test_component_hotspot_exact_alias_is_retained_once(self, subset_mock) -> None:
        base = [_candidate([0, 1], ["v2"])]
        alias = _candidate([2, 3], [COMPONENT_FAMILY, HOTSPOT_FAMILY])
        subset_mock.return_value = [alias]

        result = generate_structshell_component16_runtime_candidates(
            {"agents": [{"id": index} for index in range(4)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=base,
            config=structshell_component16_augmentation(),
        )

        self.assertEqual(result.challengers, [alias])
        self.assertEqual(result.structural_candidate_count, 1)
        self.assertEqual(len(result.candidates), 2)

    @patch(
        "lns2_selector.runtime.structshell_component16."
        "generate_structpool_candidate_subset"
    )
    def test_component_v2_exact_alias_keeps_frozen_base_row(self, subset_mock) -> None:
        base = [_candidate([0, 1], ["v2"]), _candidate([2, 3], ["v2"])]
        frozen_base = copy.deepcopy(base)
        component = _candidate([2, 3], [COMPONENT_FAMILY])
        subset_mock.return_value = [component]

        result = generate_structshell_component16_runtime_candidates(
            {"agents": [{"id": index} for index in range(4)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=base,
            config=structshell_component16_augmentation(),
        )

        tied_id = component["candidate_id"]
        tied = next(row for row in result.candidates if row["candidate_id"] == tied_id)
        self.assertEqual(tied, frozen_base[1])
        self.assertEqual(result.challengers, [])
        self.assertEqual(result.structural_candidate_count, 1)
        self.assertEqual(result.exact_duplicate_count, 1)
        self.assertEqual(
            result.provenance_by_candidate_id[tied_id],
            ("structshell_equal_four_size", "v2_base"),
        )
        self.assertEqual(base, frozen_base)

    @patch(
        "experiments.closed_loop_confirmation."
        "generate_structshell_component16_runtime_candidates"
    )
    def test_closed_loop_dispatches_component16_runtime(self, generator_mock) -> None:
        expected = object()
        generator_mock.return_value = expected
        state = {"state": "sentinel"}
        analysis = object()
        base = [_candidate([0, 1], ["v2"])]
        config = structshell_component16_augmentation()

        actual = _generate_fixed_structshell_runtime_candidates(
            state,
            analysis,
            v2_candidates=base,
            config=config,
        )

        self.assertIs(actual, expected)
        generator_mock.assert_called_once_with(
            state,
            analysis,
            v2_candidates=base,
            config=config,
        )

    @patch(
        "lns2_selector.runtime.structshell_dual16."
        "generate_structpool_candidate_subset"
    )
    def test_active_dual16_contract_and_two_arm_output_are_unchanged(
        self, subset_mock
    ) -> None:
        component = _candidate([2, 3], [COMPONENT_FAMILY])
        hotspot = _candidate([4, 5], [HOTSPOT_FAMILY])
        subset_mock.return_value = [component, hotspot]
        config = structshell_dual16_augmentation()

        result = generate_structshell_dual16_runtime_candidates(
            {"agents": [{"id": index} for index in range(6)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=[_candidate([0, 1], ["v2"])],
            config=config,
        )

        self.assertEqual(config["maximum_added_candidates"], 2)
        self.assertEqual(config["maximum_total_candidates"], 65)
        self.assertCountEqual(result.challengers, [component, hotspot])
        self.assertEqual(result.structural_candidate_count, 2)


if __name__ == "__main__":
    unittest.main()
