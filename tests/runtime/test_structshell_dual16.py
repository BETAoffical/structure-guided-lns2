from __future__ import annotations

import copy
import unittest
from unittest.mock import ANY, patch

from experiments.closed_loop_confirmation import (
    _generate_fixed_structshell_runtime_candidates,
)
from experiments.neighborhood_candidates import candidate_id
from lns2_selector.runtime.hybridstructpool_routed import (
    validate_any_hybridstructpool_augmentation,
)
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    STRUCTSHELL_DUAL16_RUNTIME_ID,
    generate_structshell_dual16_runtime_candidates,
    structshell_dual16_ablation_gate,
    structshell_dual16_augmentation,
    validate_structshell_dual16_augmentation,
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
        "structpool_family_groups": [] if family == "v2" else [family],
    }


class StructShellDual16Test(unittest.TestCase):
    def test_exact_registered_contract(self) -> None:
        expected = {
            "enabled": True,
            "pool_id": STRUCTSHELL_DUAL16_POOL_ID,
            "runtime_id": STRUCTSHELL_DUAL16_RUNTIME_ID,
            "full_union_required": False,
            "full_union_audit_preserved": True,
            "runtime_filter_id": "dual_family_fixed16_v1",
            "source_mode": "structshell_dual16",
            "nominal_size": 16,
            "runtime_structural_family_sizes": {
                "conflict_component": [16],
                "spatiotemporal_hotspot": [16],
            },
            "maximum_added_candidates": 2,
            "maximum_total_candidates": 65,
            "static_grid_cache": True,
            "activation_gate": {"gate_id": "dual16_ablation_all_states"},
        }
        config = structshell_dual16_augmentation()
        self.assertEqual(config, expected)
        self.assertEqual(validate_structshell_dual16_augmentation(config), expected)
        self.assertEqual(validate_any_hybridstructpool_augmentation(config), expected)
        changed = copy.deepcopy(config)
        changed["maximum_added_candidates"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported Dual16"):
            validate_structshell_dual16_augmentation(changed)

    def test_gate_passes_nonterminal_conflicted_state(self) -> None:
        state = {
            "agents": [{"id": 0}, {"id": 1}],
            "conflict_edges": [[0, 1]],
            "num_of_colliding_pairs": 1,
            "done": False,
            "feasible": False,
        }
        result = structshell_dual16_ablation_gate(
            state, structshell_dual16_augmentation()
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["gate_id"], "dual16_ablation_all_states")
        self.assertEqual(result["reason"], "dual16_ablation_all_states")

    @patch(
        "lns2_selector.runtime.structshell_dual16."
        "generate_structpool_candidate_subset"
    )
    def test_two_fixed16_cells_keep_the_full_v2_pool(self, subset_mock) -> None:
        base = [_candidate([0, 1], "v2"), _candidate([2, 3], "v2")]
        frozen_base = copy.deepcopy(base)
        component = _candidate([4, 5], "conflict_component")
        hotspot = _candidate([6, 7], "spatiotemporal_hotspot")
        subset_mock.return_value = [component, hotspot]

        result = generate_structshell_dual16_runtime_candidates(
            {"agents": [{"id": index} for index in range(8)]},
            object(),  # type: ignore[arg-type]
            v2_candidates=base,
            v2_anchors=[base[0]],
            config=structshell_dual16_augmentation(),
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
        by_id = {row["candidate_id"]: row for row in result.candidates}
        for row in frozen_base:
            self.assertEqual(by_id[row["candidate_id"]], row)
        self.assertCountEqual(result.challengers, [component, hotspot])
        self.assertEqual(result.structural_candidate_count, 2)
        self.assertEqual(result.causal_candidate_count, 0)
        self.assertEqual(result.causal_generation_seconds, 0.0)

    @patch(
        "lns2_selector.runtime.structshell_dual16."
        "generate_structpool_candidate_subset"
    )
    def test_generator_enforces_two_challenger_cap(self, subset_mock) -> None:
        base = [_candidate([0, 1], "v2")]
        subset_mock.return_value = [
            _candidate([2, 3], "conflict_component"),
            _candidate([4, 5], "spatiotemporal_hotspot"),
            _candidate([6, 7], "conflict_component"),
        ]
        with self.assertRaisesRegex(RuntimeError, "more than two candidates"):
            generate_structshell_dual16_runtime_candidates(
                {"agents": [{"id": index} for index in range(8)]},
                object(),  # type: ignore[arg-type]
                v2_candidates=base,
                v2_anchors=base,
                config=structshell_dual16_augmentation(),
            )

    @patch(
        "experiments.closed_loop_confirmation."
        "generate_structshell_dual16_runtime_candidates"
    )
    def test_closed_loop_dispatches_dual16_runtime(self, generator_mock) -> None:
        expected = object()
        generator_mock.return_value = expected
        state = {"state": "sentinel"}
        analysis = object()
        base = [_candidate([0, 1], "v2")]
        config = structshell_dual16_augmentation()

        actual = _generate_fixed_structshell_runtime_candidates(
            state,
            analysis,
            v2_candidates=base,
            v2_anchors=[base[0]],
            config=config,
        )

        self.assertIs(actual, expected)
        generator_mock.assert_called_once_with(
            state,
            analysis,
            v2_candidates=base,
            v2_anchors=[base[0]],
            config=config,
        )


if __name__ == "__main__":
    unittest.main()
