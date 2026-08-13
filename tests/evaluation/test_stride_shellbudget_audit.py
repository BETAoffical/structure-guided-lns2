from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_shellbudget_audit import (
    BUDGETS,
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    _candidate_view,
    reduce_size_family_balanced_maximin,
)


def _candidate(index: int, size: int, family: str) -> dict:
    return {
        "candidate_id": f"candidate-{index:02d}",
        "agents": tuple(range(index * 10, index * 10 + size)),
        "nominal_size": size,
        "families": (family,),
        "family_size_slots": ((family, size),),
        "structural_coverage": 0.5,
    }


class ShellBudgetAuditTest(unittest.TestCase):
    def test_registration_freezes_ranker_free_fallback(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                root
                / "configs"
                / "stride_shellbudget_audit_v1_registration.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["schema"], CONFIG_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(tuple(config["candidate_contract"]["candidate_budgets"]), BUDGETS)
        self.assertEqual(
            config["decision"]["if_no_budget_passes"],
            "retain_full_exact_deduplicated_equal_four_size_pool",
        )
        self.assertFalse(config["claim_boundary"]["ranker_used"])
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])
        self.assertFalse(
            config["claim_boundary"]["long_tail_avoidance_claim_allowed"]
        )

    def test_reducer_balances_all_four_sizes(self) -> None:
        families = ("a", "b", "c", "d")
        candidates = [
            _candidate(index, size, families[index % len(families)])
            for index, size in enumerate((8, 8, 16, 16, 24, 24, 32, 32))
        ]
        selected = reduce_size_family_balanced_maximin(candidates, 6)
        counts = {
            size: sum(row["nominal_size"] == size for row in selected)
            for size in (8, 16, 24, 32)
        }
        self.assertEqual(len(selected), 6)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(count > 0 for count in counts.values()))

    def test_reducer_is_deterministic_and_ignores_outcome_fields(self) -> None:
        candidates = [
            _candidate(index, size, f"family-{index % 3}")
            for index, size in enumerate((8, 16, 24, 32) * 3)
        ]
        expected = [
            row["candidate_id"]
            for row in reduce_size_family_balanced_maximin(candidates, 8)
        ]
        enriched = [
            {
                **row,
                "seed_mean": 1000.0 - index,
                "no_progress_rate": float(index % 2),
                "classification": "beneficial" if index % 2 else "adverse",
            }
            for index, row in enumerate(reversed(candidates))
        ]
        observed = [
            row["candidate_id"]
            for row in reduce_size_family_balanced_maximin(enriched, 8)
        ]
        self.assertEqual(observed, expected)

    def test_reducer_keeps_complete_pool_when_under_budget(self) -> None:
        candidates = [
            _candidate(index, size, f"family-{index}")
            for index, size in enumerate((8, 16, 24, 32))
        ]
        selected = reduce_size_family_balanced_maximin(candidates, 6)
        self.assertEqual(
            [row["candidate_id"] for row in selected],
            sorted(row["candidate_id"] for row in candidates),
        )

    def test_reducer_stops_before_exhausted_size_breaks_balance(self) -> None:
        sizes = (8,) * 6 + (16,) * 3 + (24,) * 3 + (32,) * 2
        candidates = [
            _candidate(index, size, f"family-{index % 4}")
            for index, size in enumerate(sizes)
        ]
        selected = reduce_size_family_balanced_maximin(candidates, 12)
        counts = {
            size: sum(row["nominal_size"] == size for row in selected)
            for size in (8, 16, 24, 32)
        }
        self.assertLess(len(selected), 12)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_candidate_view_requires_one_registered_nominal_size(self) -> None:
        row = {
            "candidate_id": "candidate",
            "agents": list(range(8)),
            "actual_size": 8,
            "selection_families": [
                "structpool-path-overlap:8",
                "structpool-conflict-component:16",
            ],
            "features": {
                "realized.incident_event_coverage": 0.5,
                "realized.incident_conflict_coverage": 0.5,
                "realized.internal_conflict_coverage": 0.5,
                "realized.component_coverage_max": 0.5,
            },
        }
        with self.assertRaisesRegex(ValueError, "spans nominal sizes"):
            _candidate_view(row)


if __name__ == "__main__":
    unittest.main()
