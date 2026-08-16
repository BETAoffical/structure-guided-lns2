from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

from experiments.neighborhood_candidates import candidate_id
from lns2_selector.runtime.causalclosurepool import CausalClosurePoolResult
from lns2_selector.runtime.hybridstructpool import (
    RUNTIME_STRUCTURAL_FAMILY_SIZES,
    STRUCTURAL_SIZES,
    generate_hybridstructpool_candidates,
    generate_hybridstructpool_runtime_candidates,
    hybridstructpool_high_stress_gate,
    hybridstructpool_runtime_augmentation,
    merge_hybridstructpool_candidates,
    reduce_hybridstructpool_challengers,
    validate_hybridstructpool_augmentation,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    generate_routed_hybridstructpool_runtime_candidates,
    overall_rollback_routed_hybridstructpool_augmentation,
    rollback_aware_routed_hybridstructpool_augmentation,
    routed_hybridstructpool_augmentation,
    routed_hybridstructpool_high_stress_gate,
    validate_any_hybridstructpool_augmentation,
    validate_overall_rollback_routed_hybridstructpool_augmentation,
    validate_rollback_aware_routed_hybridstructpool_augmentation,
    validate_routed_hybridstructpool_augmentation,
)


def _candidate(agents: list[int], kind: str) -> dict:
    normalized = sorted(set(agents))
    return {
        "candidate_id": candidate_id(normalized),
        "agents": normalized,
        "candidate_kind": kind,
        "actual_size": len(normalized),
        "proposal_audit": {
            "global_event_incident_coverage": 0.5,
            "global_pair_internal_coverage": 0.25,
            "conflict_component_reach": 0.5,
        },
        "structpool_family_groups": (
            [kind] if kind not in {"base", "causalclosure"} else []
        ),
    }


class HybridStructPoolTest(unittest.TestCase):
    def test_runtime_contract_preserves_the_complete_structural_grid(self) -> None:
        config = hybridstructpool_runtime_augmentation()
        self.assertTrue(config["full_union_required"])
        self.assertTrue(config["full_union_audit_preserved"])
        self.assertEqual(config["runtime_filter_id"], "none")
        self.assertEqual(config["structural_sizes"], [8, 16, 24, 32])
        self.assertEqual(
            config["runtime_structural_family_sizes"],
            {
                family: list(sizes)
                for family, sizes in RUNTIME_STRUCTURAL_FAMILY_SIZES.items()
            },
        )
        self.assertEqual(validate_hybridstructpool_augmentation(config), config)
        config["maximum_total_candidates"] = 12
        with self.assertRaisesRegex(ValueError, "unsupported HybridStructPool"):
            validate_hybridstructpool_augmentation(config)

    def test_runtime_gate_matches_registered_high_stress_identity(self) -> None:
        state = {
            "agents": [{"id": index} for index in range(96)],
            "conflict_edges": [[index, index + 1] for index in range(16)],
            "num_of_colliding_pairs": 16,
        }
        result = hybridstructpool_high_stress_gate(
            state, hybridstructpool_runtime_augmentation()
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "high_stress_state")

    def test_routed_contract_preserves_sources_without_mutating_v8(self) -> None:
        legacy = hybridstructpool_runtime_augmentation()
        self.assertEqual(validate_any_hybridstructpool_augmentation(legacy), legacy)
        for mode in ("structshell_only", "causal_only", "routed_structshell"):
            config = routed_hybridstructpool_augmentation(mode)
            self.assertEqual(config["source_mode"], mode)
            self.assertEqual(
                validate_routed_hybridstructpool_augmentation(config), config
            )
            self.assertEqual(
                validate_any_hybridstructpool_augmentation(config), config
            )
        self.assertEqual(
            hybridstructpool_runtime_augmentation(), legacy
        )

    def test_rollback_aware_contract_is_separately_versioned_and_frozen(self) -> None:
        config = rollback_aware_routed_hybridstructpool_augmentation()
        self.assertEqual(config["pool_id"], "stride-hybridstructpool-routed-v2")
        self.assertEqual(config["source_mode"], "structshell_only")
        self.assertEqual(config["exact_rollback_guard"]["exact_rollback_limit"], 3)
        self.assertEqual(
            validate_rollback_aware_routed_hybridstructpool_augmentation(config),
            config,
        )
        self.assertEqual(validate_any_hybridstructpool_augmentation(config), config)
        config["exact_rollback_guard"]["exact_rollback_limit"] = 2
        with self.assertRaisesRegex(ValueError, "rollback-aware"):
            validate_any_hybridstructpool_augmentation(config)

    def test_state_bounded_contract_is_separately_versioned_and_strict(self) -> None:
        legacy = rollback_aware_routed_hybridstructpool_augmentation()
        config = overall_rollback_routed_hybridstructpool_augmentation()
        self.assertEqual(config["pool_id"], "stride-hybridstructpool-routed-v3")
        self.assertEqual(config["source_mode"], "routed_structshell")
        self.assertEqual(
            config["activation_gate"]["gate_id"],
            "stride-highstress-conflict-structure-v1",
        )
        guard = config["exact_rollback_guard"]
        self.assertEqual(guard["exact_rollback_limit"], 3)
        self.assertEqual(guard["fallback"], "fresh_v2_only")
        self.assertEqual(guard["repair_state_cache"], "pre_budget_only")
        self.assertEqual(
            validate_overall_rollback_routed_hybridstructpool_augmentation(config),
            config,
        )
        self.assertEqual(validate_any_hybridstructpool_augmentation(config), config)
        self.assertEqual(
            rollback_aware_routed_hybridstructpool_augmentation(), legacy
        )
        config["exact_rollback_guard"]["exact_rollback_limit"] = 2
        with self.assertRaisesRegex(ValueError, "state-bounded rollback"):
            validate_any_hybridstructpool_augmentation(config)

    def test_routed_gate_removes_total_agent_shortcut(self) -> None:
        sparse_edges = [
            list(edge)
            for edge in list(itertools.combinations(range(8), 2))[:16]
        ]
        sparse = {
            "agents": [{"id": index} for index in range(96)],
            "conflict_edges": sparse_edges,
            "num_of_colliding_pairs": len(sparse_edges),
        }
        legacy = routed_hybridstructpool_high_stress_gate(
            sparse, routed_hybridstructpool_augmentation("structshell_only")
        )
        strict = routed_hybridstructpool_high_stress_gate(
            sparse, routed_hybridstructpool_augmentation("routed_structshell")
        )
        self.assertTrue(legacy["passed"])
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["reason"], "stress_indicator_below_threshold")

        dense_edges = [[index, index + 16] for index in range(16)]
        dense = {
            "agents": [{"id": index} for index in range(96)],
            "conflict_edges": dense_edges,
            "num_of_colliding_pairs": len(dense_edges),
        }
        dense_result = routed_hybridstructpool_high_stress_gate(
            dense, routed_hybridstructpool_augmentation("routed_structshell")
        )
        self.assertTrue(dense_result["passed"])
        self.assertEqual(dense_result["active_conflict_agent_count"], 32)

    def test_exact_sets_merge_with_v2_authoritative_and_full_provenance(self) -> None:
        base = [_candidate([1, 2], "base"), _candidate([3, 4], "base")]
        structural = [_candidate([1, 2], "structural"), _candidate([5, 6], "structural")]
        causal = [_candidate([5, 6], "causalclosure"), _candidate([7, 8], "causalclosure")]
        frozen_base = [dict(row) for row in base]

        result = merge_hybridstructpool_candidates(base, structural, causal)

        self.assertEqual(base, frozen_base)
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual(result.exact_duplicate_count, 2)
        first_id = candidate_id([1, 2])
        self.assertEqual(
            result.provenance_by_candidate_id[first_id],
            ("structshell_equal_four_size", "v2_base"),
        )
        self.assertEqual(
            next(row for row in result.candidates if row["candidate_id"] == first_id)[
                "candidate_kind"
            ],
            "base",
        )
        self.assertNotIn(first_id, {row["candidate_id"] for row in result.challengers})
        result.candidates[0]["agents"].append(999)
        self.assertEqual(base, frozen_base)
        self.assertTrue(
            all(999 not in row["agents"] for row in result.challengers)
        )

    def test_duplicate_within_one_source_is_rejected(self) -> None:
        row = _candidate([1, 2], "base")
        with self.assertRaisesRegex(ValueError, "duplicate sets"):
            merge_hybridstructpool_candidates([row, dict(row)], [], [])

    def test_candidate_identity_is_checked(self) -> None:
        row = _candidate([1, 2], "base")
        row["candidate_id"] = "neighborhood-invalid"
        with self.assertRaisesRegex(ValueError, "identity"):
            merge_hybridstructpool_candidates([row], [], [])

    @patch("lns2_selector.runtime.hybridstructpool.generate_causalclosure_candidates")
    @patch("lns2_selector.runtime.hybridstructpool.generate_structpool_candidate_grid")
    def test_generator_exposes_full_grid_without_budget_reduction(
        self, grid_mock, causal_mock
    ) -> None:
        base = [_candidate([1, 2], "base")]
        structural = [_candidate([3, 4], "structural")]
        causal = [_candidate([5, 6], "causalclosure")]
        grid_mock.return_value = structural
        causal_mock.return_value = CausalClosurePoolResult(
            candidates=causal,
            attempts=[{"decision": "selected"}],
            raw_candidate_count=1,
            pareto_front_count=1,
            oversized_family_count=0,
            family_attempt_count=1,
            oversized_core_count=0,
            core_attempt_count=1,
        )

        result = generate_hybridstructpool_candidates(
            {"agents": [{"id": 1}, {"id": 2}]},
            object(),
            v2_candidates=base,
            v2_anchors=base,
        )

        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.causal_attempts, [{"decision": "selected"}])
        self.assertEqual(
            tuple(grid_mock.call_args.kwargs["neighborhood_sizes"]), STRUCTURAL_SIZES
        )

    def test_generator_rejects_asymmetric_size_contract(self) -> None:
        base = [_candidate([1, 2], "base")]
        with self.assertRaisesRegex(ValueError, "symmetric sizes"):
            generate_hybridstructpool_candidates(
                {"agents": [{"id": 1}, {"id": 2}]},
                object(),
                v2_candidates=base,
                v2_anchors=base,
                structural_sizes=(8, 16, 24),
            )

    @patch("lns2_selector.runtime.hybridstructpool.generate_causalclosure_candidates")
    @patch("lns2_selector.runtime.hybridstructpool.generate_structpool_candidate_subset")
    def test_runtime_generator_uses_the_registered_complete_structural_grid(
        self, subset_mock, causal_mock
    ) -> None:
        base = [_candidate([1, 2], "base")]
        subset_mock.return_value = [_candidate([3, 4], "structural")]
        causal_mock.return_value = CausalClosurePoolResult(
            candidates=[_candidate([5, 6], "causalclosure")],
            attempts=[],
            raw_candidate_count=1,
            pareto_front_count=1,
            oversized_family_count=0,
            family_attempt_count=1,
            oversized_core_count=0,
            core_attempt_count=1,
        )

        result = generate_hybridstructpool_runtime_candidates(
            {"agents": [{"id": 1}, {"id": 2}]},
            object(),
            v2_candidates=base,
            v2_anchors=base,
            structural_family_sizes=RUNTIME_STRUCTURAL_FAMILY_SIZES,
        )

        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(
            subset_mock.call_args.kwargs["family_sizes"],
            RUNTIME_STRUCTURAL_FAMILY_SIZES,
        )
        with self.assertRaisesRegex(ValueError, "runtime structural mask"):
            generate_hybridstructpool_runtime_candidates(
                {"agents": [{"id": 1}, {"id": 2}]},
                object(),
                v2_candidates=base,
                v2_anchors=base,
                structural_family_sizes={"conflict_component": (24,)},
            )

    @patch(
        "lns2_selector.runtime.hybridstructpool_routed.generate_causalclosure_candidates"
    )
    @patch(
        "lns2_selector.runtime.hybridstructpool_routed.generate_structpool_candidate_subset"
    )
    def test_routed_generators_change_only_enabled_source(
        self, subset_mock, causal_mock
    ) -> None:
        base = [_candidate([1, 2], "base")]
        structural = [_candidate([3, 4], "structural")]
        causal = [_candidate([5, 6], "causalclosure")]
        subset_mock.return_value = structural
        causal_mock.return_value = CausalClosurePoolResult(
            candidates=causal,
            attempts=[{"decision": "selected"}],
            raw_candidate_count=1,
            pareto_front_count=1,
            oversized_family_count=0,
            family_attempt_count=1,
            oversized_core_count=0,
            core_attempt_count=1,
        )

        struct_result = generate_routed_hybridstructpool_runtime_candidates(
            {"agents": [{"id": 1}, {"id": 2}]},
            object(),
            v2_candidates=base,
            v2_anchors=base,
            config=routed_hybridstructpool_augmentation("structshell_only"),
        )
        self.assertEqual(
            {row["candidate_id"] for row in struct_result.candidates},
            {row["candidate_id"] for row in base + structural},
        )
        subset_mock.assert_called_once()
        causal_mock.assert_not_called()

        subset_mock.reset_mock()
        causal_result = generate_routed_hybridstructpool_runtime_candidates(
            {"agents": [{"id": 1}, {"id": 2}]},
            object(),
            v2_candidates=base,
            v2_anchors=base,
            config=routed_hybridstructpool_augmentation("causal_only"),
        )
        self.assertEqual(
            {row["candidate_id"] for row in causal_result.candidates},
            {row["candidate_id"] for row in base + causal},
        )
        subset_mock.assert_not_called()
        causal_mock.assert_called_once()

    def test_budget_reducer_preserves_v2_and_round_robins_semantic_groups(self) -> None:
        base = [_candidate([0, 1], "base")]
        structural = []
        groups = (
            "topology_boundary",
            "conflict_component",
            "spatiotemporal_hotspot",
            "bottleneck_crossing",
            "path_overlap",
        )
        for index, group in enumerate(groups):
            structural.append(_candidate([10 + index, 20 + index], group))
        causal = [_candidate([30, 31], "causalclosure")]
        result = merge_hybridstructpool_candidates(base, structural, causal)

        selected = reduce_hybridstructpool_challengers(
            result, maximum_challengers=6
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(len(result.candidates), 7)
        self.assertEqual(
            {row["candidate_id"] for row in selected},
            {row["candidate_id"] for row in structural + causal},
        )

    def test_budget_reducer_rejects_unregistered_budget(self) -> None:
        result = merge_hybridstructpool_candidates(
            [_candidate([0, 1], "base")], [_candidate([2, 3], "path_overlap")], []
        )
        with self.assertRaisesRegex(ValueError, "budgets"):
            reduce_hybridstructpool_challengers(result, maximum_challengers=7)


if __name__ == "__main__":
    unittest.main()
