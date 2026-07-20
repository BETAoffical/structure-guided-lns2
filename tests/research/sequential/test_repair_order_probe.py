from __future__ import annotations

import copy
import unittest

from research.studies.sequential.repair_order_probe import (
    analyze_trials,
    order_conditions,
    repair_order_for_policy,
    select_probe_candidates,
    select_probe_states,
    solution_fingerprint,
)


def candidate(candidate_id: str, size: int) -> dict:
    agents = list(range(size))
    return {
        "candidate_id": candidate_id,
        "agents": agents,
        "actual_size": size,
        "selection_families": [f"target:{size}"],
        "proposal_count_by_family": {f"target:{size}": 1},
        "proposal_seeds": [100 + size],
        "seed_agents": [0],
    }


def state_row(map_number: int, state_number: int) -> dict:
    candidates = [
        candidate(f"candidate-{map_number}-{state_number}-{index}", size)
        for index, size in enumerate((4, 8, 16, 4, 8, 16, 4))
    ]
    return {
        "state_id": f"state-{map_number}-{state_number}",
        "split": "policy_train",
        "map_id": f"map-{map_number:02d}",
        "task_id": f"task-{map_number}-{state_number}",
        "layout_mode": ("regular_beltway", "compartmentalized", "dead_end_aisles")[map_number % 3],
        "task_variant": "balanced_80" if state_number % 2 else "bottleneck_100",
        "agent_count": 100,
        "solver_seed": state_number % 3 + 1,
        "stage": ("early", "middle", "late")[state_number % 3],
        "conflict_severity": ("low", "medium", "high")[state_number % 3],
        "source_selected_candidate_id": candidates[0]["candidate_id"],
        "candidates": candidates,
    }


def config() -> dict:
    return {
        "source": {"required_split": "policy_train"},
        "selection": {
            "map_count": 12,
            "states_per_map": 2,
            "total_states": 24,
            "candidates_per_state": 6,
            "required_sizes": [4, 8, 16],
            "state_balance_fields": ["stage", "conflict_severity", "task_variant", "solver_seed"],
        },
        "order_conditions": {
            "random_crn_trials": 8,
            "deterministic_repeats": 2,
            "deterministic_policies": [
                "agent_id_ascending",
                "conflict_degree_descending",
                "delay_descending",
                "path_length_descending",
            ],
        },
        "bootstrap_samples": 20,
        "bootstrap_seed": 7,
        "thresholds": {
            "minimum_crn_spearman": 0.5,
            "minimum_crn_pareto_jaccard": 0.5,
            "minimum_crn_best_jaccard": 0.5,
            "minimum_solution_divergence_fraction": 0.5,
            "minimum_c1_conflict_divergence_fraction": 0.3,
            "minimum_oracle_auc_improvement": 0.05,
            "minimum_positive_opportunity_fraction": 0.6,
            "bootstrap_lower_bound": 0.0,
            "fixed_rule_normalized_regret_tolerance": 0.05,
            "fixed_rule_dominance_share": 0.8,
        },
    }


def observation() -> dict:
    return {
        "sum_of_costs": 10,
        "num_of_colliding_pairs": 1,
        "conflict_edges": [[0, 1]],
        "runtime": 1.0,
        "low_level": {"expanded": 1, "generated": 2, "reopened": 0, "runs": 1},
        "agents": [
            {"id": 0, "path": [0, 1], "path_cost": 1, "delay": 3, "conflict_degree": 1},
            {"id": 1, "path": [1, 0, 1], "path_cost": 2, "delay": 1, "conflict_degree": 2},
            {"id": 2, "path": [2], "path_cost": 0, "delay": 0, "conflict_degree": 0},
        ],
    }


def outcome(condition: dict, candidate_number: int, *, better: bool) -> dict:
    policy = condition.get("policy")
    random_condition = condition["kind"] == "random_crn"
    auc = 10.0 + candidate_number
    conflict = 2 + candidate_number
    solution = "random"
    order = [0, 1]
    if not random_condition:
        offsets = {
            "agent_id_ascending": 0.0,
            "conflict_degree_descending": 1.0,
            "delay_descending": 2.0,
            "path_length_descending": 3.0,
        }
        auc = 5.0 + candidate_number + offsets[str(policy)]
        conflict = int(offsets[str(policy)])
        solution = f"solution-{policy}"
    return {
        "condition": condition,
        "initial_random_seed": 9000 + int(condition.get("trial_index") or 0),
        "initial_actual_repair_order": order,
        "raw_conflict_trajectory": [4, conflict, 0],
        "h1": {
            "feasible": False,
            "final_conflicts": conflict,
            "conflict_auc": 3.0,
            "solution_fingerprint": solution,
        },
        "h4": {
            "feasible": True,
            "final_conflicts": 0,
            "conflict_auc": auc,
            "solution_fingerprint": f"final-{solution}",
        },
    }


class RepairOrderProbeTests(unittest.TestCase):
    def test_state_and_candidate_selection_are_deterministic_and_balanced(self) -> None:
        rows = [state_row(map_number, index) for map_number in range(12) for index in range(8)]
        selected = select_probe_states(rows, config())
        reverse = select_probe_states(list(reversed(rows)), config())
        self.assertEqual([row["state_id"] for row in selected], [row["state_id"] for row in reverse])
        self.assertEqual(len(selected), 24)
        counts = {}
        for row in selected:
            counts[row["map_id"]] = counts.get(row["map_id"], 0) + 1
            chosen = select_probe_candidates(row, config())
            self.assertEqual(len(chosen), 6)
            self.assertIn(row["source_selected_candidate_id"], {value["candidate_id"] for value in chosen})
            self.assertTrue({4, 8, 16}.issubset({value["actual_size"] for value in chosen}))
        self.assertEqual(set(counts.values()), {2})

    def test_order_policies_and_solution_fingerprint(self) -> None:
        state = observation()
        self.assertEqual(repair_order_for_policy(state, [0, 1, 2], "agent_id_ascending"), [0, 1, 2])
        self.assertEqual(repair_order_for_policy(state, [0, 1, 2], "conflict_degree_descending"), [1, 0, 2])
        self.assertEqual(repair_order_for_policy(state, [0, 1, 2], "delay_descending"), [0, 1, 2])
        self.assertEqual(repair_order_for_policy(state, [0, 1, 2], "path_length_descending"), [1, 0, 2])
        changed = copy.deepcopy(state)
        changed["runtime"] = 99.0
        changed["low_level"]["generated"] = 999
        self.assertEqual(solution_fingerprint(state), solution_fingerprint(changed))
        changed["agents"][0]["path"] = [0, 2]
        self.assertNotEqual(solution_fingerprint(state), solution_fingerprint(changed))

    def test_registered_condition_count(self) -> None:
        values = order_conditions(config())
        self.assertEqual(len(values), 16)
        self.assertEqual(sum(row["kind"] == "random_crn" for row in values), 8)
        self.assertEqual(sum(row["kind"] == "deterministic" for row in values), 8)

    def test_analysis_routes_to_fixed_rule_when_it_dominates(self) -> None:
        selected = [state_row(0, 0)]
        selected[0]["probe_candidates"] = select_probe_candidates(selected[0], config())
        rows = []
        for candidate_number, selected_candidate in enumerate(selected[0]["probe_candidates"]):
            for condition in order_conditions(config()):
                rows.append(
                    {
                        "state_id": selected[0]["state_id"],
                        "candidate_id": selected_candidate["candidate_id"],
                        "condition_id": condition["condition_id"],
                        "status": "ok",
                        "complete": True,
                        "outcome": outcome(condition, candidate_number, better=True),
                    }
                )
        report = analyze_trials(selected, rows, config(), formal=False)
        self.assertTrue(report["gates"]["integrity"])
        self.assertTrue(report["gates"]["crn_stability"])
        self.assertTrue(report["gates"]["repair_order_material"])
        self.assertEqual(report["decision"], "adopt_fixed_repair_order")
        self.assertEqual(report["metrics"]["fixed_rules"]["best_policy"], "agent_id_ascending")


if __name__ == "__main__":
    unittest.main()
