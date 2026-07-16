from __future__ import annotations

import copy
import unittest

from experiments.contextual_repair_order_audit import (
    POLICIES,
    _ordered_features,
    _permuted_rows,
    _summarize,
)


def state() -> dict:
    return {
        "num_of_colliding_pairs": 3,
        "conflict_edges": [[10, 20], [20, 30], [30, 40]],
        "agents": [
            {"id": 10, "conflict_degree": 1, "delay": 4, "path": [0, 1], "path_cost": 1},
            {"id": 20, "conflict_degree": 3, "delay": 1, "path": [2, 1, 0], "path_cost": 2},
            {"id": 30, "conflict_degree": 2, "delay": 2, "path": [3], "path_cost": 0},
            {"id": 40, "conflict_degree": 1, "delay": 0, "path": [4, 3], "path_cost": 1},
        ],
    }


def row(decision: str, policy: str, value: float, map_id: str = "map-a") -> dict:
    return {
        "state_id": decision.split("::")[0],
        "decision_id": decision,
        "candidate_id": decision.split("::")[1],
        "map_id": map_id,
        "policy": policy,
        "features": {"context": float(hash(decision) % 7), f"rule.{policy}": 1.0},
        "target": {
            "normalized_h4_auc": value,
            "conflict_auc": value * 10,
            "feasible_rate": 1.0,
            "final_conflicts": value,
        },
    }


class ContextualRepairOrderAuditTests(unittest.TestCase):
    def test_order_features_support_non_contiguous_agents_and_depend_on_rule(self) -> None:
        degree = _ordered_features(state(), [10, 20, 30], "conflict_degree_descending")
        delay = _ordered_features(state(), [10, 20, 30], "delay_descending")
        self.assertEqual(degree["order.rule.conflict_degree_descending"], 1.0)
        self.assertEqual(delay["order.rule.delay_descending"], 1.0)
        self.assertNotEqual(
            degree["order.conflict_degree.first"], delay["order.conflict_degree.first"]
        )
        self.assertIn("order.internal_edge_distance.mean", degree)

    def test_context_permutation_preserves_targets_and_rule_membership(self) -> None:
        rows = [
            row(decision, policy, float(index))
            for index, decision in enumerate(("state-a::candidate-a", "state-b::candidate-b"))
            for policy in POLICIES
        ]
        shuffled = _permuted_rows(rows, __import__("random").Random(1))
        original_targets = {
            (value["decision_id"], value["policy"]): value["target"] for value in rows
        }
        self.assertEqual(
            original_targets,
            {(value["decision_id"], value["policy"]): value["target"] for value in shuffled},
        )
        self.assertEqual(len(shuffled), len(rows))

    def test_summary_uses_map_as_bootstrap_unit_and_detects_collapse(self) -> None:
        config = {
            "bootstrap_samples": 20,
            "bootstrap_seed": 3,
            "thresholds": {"near_oracle_normalized_regret": 0.05},
        }
        records = []
        for index in range(4):
            records.append(
                {
                    "map_id": f"map-{index // 2}",
                    "model_policy": POLICIES[0],
                    "model": {"normalized_h4_auc": 0.8, "feasible_rate": 1, "final_conflicts": 1},
                    "fixed": {"normalized_h4_auc": 1.0, "feasible_rate": 1, "final_conflicts": 1},
                    "uniform": {"normalized_h4_auc": 1.1},
                    "oracle": {"normalized_h4_auc": 0.8},
                }
            )
        summary = _summarize(records, config)
        self.assertEqual(summary["map_count"], 2)
        self.assertEqual(summary["maps_no_worse"], 2)
        self.assertEqual(summary["maximum_policy_share"], 1.0)
        self.assertAlmostEqual(summary["model_vs_fixed_auc_improvement"], 0.2)
        self.assertAlmostEqual(summary["bootstrap"]["mean"], 0.2)

    def test_order_features_do_not_read_outcomes(self) -> None:
        baseline = _ordered_features(state(), [10, 20, 30], POLICIES[0])
        changed = copy.deepcopy(state())
        changed["runtime"] = 999
        changed["low_level"] = {"generated": 999}
        self.assertEqual(baseline, _ordered_features(changed, [10, 20, 30], POLICIES[0]))


if __name__ == "__main__":
    unittest.main()
