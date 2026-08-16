from __future__ import annotations

import copy
import itertools
import unittest
from unittest.mock import ANY, patch

from experiments.neighborhood_candidates import candidate_id
from lns2_selector.runtime.hybridstructpool import (
    hybridstructpool_runtime_augmentation,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    overall_rollback_routed_hybridstructpool_augmentation,
    rollback_aware_routed_hybridstructpool_augmentation,
    routed_hybridstructpool_augmentation,
    validate_any_hybridstructpool_augmentation,
)
from lns2_selector.runtime.structshell_single_family import (
    STRUCTSHELL_SINGLE_FAMILY_POOL_ID,
    STRUCTSHELL_SINGLE_FAMILY_PROFILES,
    generate_structshell_single_family_runtime_candidates,
    structshell_single_family_augmentation,
    structshell_single_family_ablation_gate,
    validate_structshell_single_family_augmentation,
)


def _candidate(agents: list[int], family: str) -> dict:
    normalized = sorted(set(map(int, agents)))
    return {
        "candidate_id": candidate_id(normalized),
        "agents": normalized,
        "actual_size": len(normalized),
        "selection_families": [family],
        "selection_rank_by_family": {family: 0},
        "proposal_count_by_family": {family: 1},
        "proposal_seeds": [],
        "seed_agents": [],
        "structpool_family_groups": (
            [] if family == "v2" else [family]
        ),
    }


class StructShellSingleFamilyTest(unittest.TestCase):
    def test_profiles_are_single_size_exact_contracts(self) -> None:
        expected_variants = {
            "bottleneck": "bottleneck_crossing",
            "conflict_component": "conflict_component",
            "hotspot": "spatiotemporal_hotspot",
            "path_overlap": "path_overlap",
        }
        for profile in STRUCTSHELL_SINGLE_FAMILY_PROFILES:
            config = structshell_single_family_augmentation(profile)
            self.assertEqual(config["pool_id"], STRUCTSHELL_SINGLE_FAMILY_POOL_ID)
            self.assertEqual(config["nominal_size"], 16)
            self.assertEqual(
                config["runtime_structural_family_sizes"],
                {expected_variants[profile]: [16]},
            )
            self.assertEqual(config["maximum_added_candidates"], 1)
            self.assertNotIn("exact_rollback_guard", config)
            self.assertEqual(
                validate_structshell_single_family_augmentation(config), config
            )
            self.assertEqual(validate_any_hybridstructpool_augmentation(config), config)
        size_32 = structshell_single_family_augmentation("path_overlap", 32)
        self.assertEqual(
            size_32["runtime_structural_family_sizes"], {"path_overlap": [32]}
        )

    def test_contract_rejects_profile_size_and_field_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported single-family"):
            structshell_single_family_augmentation("topology_boundary")
        with self.assertRaisesRegex(ValueError, "8/16/24/32"):
            structshell_single_family_augmentation("hotspot", 12)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            structshell_single_family_augmentation("hotspot", 16.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            structshell_single_family_augmentation("hotspot", True)  # type: ignore[arg-type]
        changed = structshell_single_family_augmentation("hotspot")
        changed["maximum_added_candidates"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported single-family"):
            validate_structshell_single_family_augmentation(changed)

    def test_new_factory_does_not_mutate_any_existing_factory_output(self) -> None:
        existing = {
            "full": hybridstructpool_runtime_augmentation(),
            "routed": [
                routed_hybridstructpool_augmentation(mode)
                for mode in ("structshell_only", "causal_only", "routed_structshell")
            ],
            "candidate_guard": rollback_aware_routed_hybridstructpool_augmentation(),
            "state_guard": overall_rollback_routed_hybridstructpool_augmentation(),
        }
        frozen = copy.deepcopy(existing)
        for profile in STRUCTSHELL_SINGLE_FAMILY_PROFILES:
            structshell_single_family_augmentation(profile)
        self.assertEqual(existing, frozen)
        self.assertEqual(hybridstructpool_runtime_augmentation(), frozen["full"])
        self.assertEqual(
            [
                routed_hybridstructpool_augmentation(mode)
                for mode in ("structshell_only", "causal_only", "routed_structshell")
            ],
            frozen["routed"],
        )
        self.assertEqual(
            rollback_aware_routed_hybridstructpool_augmentation(),
            frozen["candidate_guard"],
        )
        self.assertEqual(
            overall_rollback_routed_hybridstructpool_augmentation(),
            frozen["state_guard"],
        )

    def test_ablation_gate_passes_every_nonterminal_conflict_structure(self) -> None:
        minimal = {
            "agents": [{"id": 0}, {"id": 1}],
            "conflict_edges": [[0, 1]],
            "num_of_colliding_pairs": 1,
            "done": False,
            "feasible": False,
        }
        minimal_result = structshell_single_family_ablation_gate(
            minimal, structshell_single_family_augmentation("bottleneck")
        )
        self.assertTrue(minimal_result["passed"])

        sparse_edges = [
            list(edge) for edge in list(itertools.combinations(range(8), 2))[:16]
        ]
        sparse = {
            "agents": [{"id": index} for index in range(96)],
            "conflict_edges": sparse_edges,
            "num_of_colliding_pairs": len(sparse_edges),
        }
        sparse_result = structshell_single_family_ablation_gate(
            sparse, structshell_single_family_augmentation("bottleneck")
        )
        self.assertTrue(sparse_result["passed"])
        self.assertEqual(
            sparse_result["reason"], "single_family_ablation_all_states"
        )

        dense_edges = [[index, index + 16] for index in range(16)]
        dense = {
            "agents": [{"id": index} for index in range(96)],
            "conflict_edges": dense_edges,
            "num_of_colliding_pairs": len(dense_edges),
        }
        dense_result = structshell_single_family_ablation_gate(
            dense, structshell_single_family_augmentation("bottleneck")
        )
        self.assertTrue(dense_result["passed"])
        self.assertEqual(dense_result["active_conflict_agent_count"], 32)

        terminal = {
            "agents": [{"id": index} for index in range(8)],
            "conflict_edges": [],
            "num_of_colliding_pairs": 0,
            "done": True,
            "feasible": True,
        }
        terminal_result = structshell_single_family_ablation_gate(
            terminal, structshell_single_family_augmentation("bottleneck")
        )
        self.assertFalse(terminal_result["passed"])
        self.assertEqual(terminal_result["reason"], "terminal_state")

        invalid = copy.deepcopy(sparse)
        invalid["agents"].append({"id": 0})
        with self.assertRaisesRegex(ValueError, "duplicate agent ids"):
            structshell_single_family_ablation_gate(
                invalid, structshell_single_family_augmentation("bottleneck")
            )

    @patch(
        "lns2_selector.runtime.structshell_single_family."
        "generate_structpool_candidate_subset"
    )
    def test_generator_uses_only_the_registered_cell_and_keeps_v2_exact(
        self, subset_mock
    ) -> None:
        base = [_candidate([0, 1], "v2"), _candidate([2, 3], "v2")]
        frozen_base = copy.deepcopy(base)
        structural = _candidate([4, 5, 6], "spatiotemporal_hotspot")
        subset_mock.return_value = [structural]
        config = structshell_single_family_augmentation("hotspot")

        result = generate_structshell_single_family_runtime_candidates(
            {"agents": [{"id": index} for index in range(8)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=base,
            v2_anchors=[base[0]],
            config=config,
        )

        subset_mock.assert_called_once_with(
            ANY,
            ANY,
            family_sizes={"spatiotemporal_hotspot": (16,)},
        )
        self.assertEqual(base, frozen_base)
        by_id = {row["candidate_id"]: row for row in result.candidates}
        for row in frozen_base:
            self.assertEqual(by_id[row["candidate_id"]], row)
        self.assertEqual(result.challengers, [structural])
        self.assertEqual(result.structural_candidate_count, 1)
        self.assertEqual(result.causal_candidate_count, 0)
        self.assertEqual(result.causal_attempts, [])
        self.assertEqual(result.causal_generation_seconds, 0.0)

    @patch(
        "lns2_selector.runtime.structshell_single_family."
        "generate_structpool_candidate_subset"
    )
    def test_generator_rejects_more_than_one_structural_candidate(
        self, subset_mock
    ) -> None:
        base = [_candidate([0, 1], "v2")]
        subset_mock.return_value = [
            _candidate([2, 3], "conflict_component"),
            _candidate([4, 5], "conflict_component"),
        ]
        with self.assertRaisesRegex(RuntimeError, "more than one candidate"):
            generate_structshell_single_family_runtime_candidates(
                {"agents": [{"id": index} for index in range(8)]},
                object(),  # type: ignore[arg-type]
                v2_candidates=base,
                v2_anchors=base,
                config=structshell_single_family_augmentation(
                    "conflict_component"
                ),
            )


if __name__ == "__main__":
    unittest.main()
