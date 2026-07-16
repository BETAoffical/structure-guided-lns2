from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.repair_collection import state_fingerprint
from experiments.sequential_credit_audit import (
    _spearman_effectiveness,
    aggregate_trials,
    analyze_index,
    best_ids,
    build_dry_run,
    conflict_severity,
    execute_horizon4_trial,
    pareto_ids,
    select_audit_states,
)


def make_state(conflicts: int, *, feasible: bool = False, iteration: int = 0) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": feasible,
        "done": feasible,
        "iteration": iteration,
        "rows": 2,
        "cols": 2,
        "sum_of_costs": 4,
        "num_of_colliding_pairs": conflicts,
        "runtime": 0.0,
        "low_level": {
            "expanded": iteration,
            "generated": iteration * 2,
            "reopened": 0,
            "runs": iteration,
        },
        "obstacles": [False, False, False, False],
        "conflict_edges": [[0, 1]] if conflicts else [],
        "agents": [
            {
                "id": 0,
                "path": [0, 1],
                "delay": 1,
                "conflict_degree": int(bool(conflicts)),
            },
            {
                "id": 1,
                "path": [1, 0],
                "delay": 0,
                "conflict_degree": int(bool(conflicts)),
            },
        ],
    }


def make_candidate(candidate_id: str, agents: list[int] | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "agents": agents or [0, 1],
        "actual_size": len(agents or [0, 1]),
        "selection_families": ["target:4"],
        "proposal_count_by_family": {"target:4": 1},
        "proposal_seeds": [100],
        "seed_agents": [0],
    }


def make_config() -> dict:
    return {
        "schema_version": 1,
        "source": {"required_split": "policy_train"},
        "selection": {
            "map_count": 12,
            "states_per_map": 8,
            "total_states": 96,
            "stratification": [
                "stage",
                "conflict_severity",
                "task_variant",
                "solver_seed",
            ],
            "conflict_density_thresholds": [0.001, 0.01],
        },
        "evaluation_trials": 4,
        "horizon": 4,
        "workers": 4,
        "trial_process_timeout_seconds": 180.0,
        "bootstrap_samples": 20,
        "bootstrap_seed": 7,
        "thresholds": {
            "minimum_split_half_spearman": -1.0,
            "minimum_split_half_pareto_jaccard": 0.0,
            "minimum_split_half_best_jaccard": 0.0,
            "maximum_h1_h4_pareto_jaccard": 1.0,
            "minimum_changed_best_fraction": 0.0,
            "minimum_oracle_auc_improvement": -1.0,
            "minimum_positive_opportunity_fraction": 0.0,
            "minimum_maps_non_worse": 0,
            "bootstrap_lower_bound": -1.0,
        },
    }


def make_source(map_number: int, state_number: int, *, split: str = "policy_train") -> dict:
    state = make_state((state_number % 5) + 1)
    candidate = make_candidate(f"candidate-{map_number}-{state_number}")
    return {
        "state_id": f"state-{map_number}-{state_number}",
        "episode_id": f"episode-{map_number}-{state_number}",
        "split": split,
        "map_id": f"map-{map_number:02d}",
        "task_id": f"task-{map_number}-{state_number}",
        "layout_mode": "regular_beltway",
        "task_variant": "balanced_80" if state_number % 2 else "bottleneck_100",
        "agent_count": 80,
        "solver_seed": state_number % 3 + 1,
        "stage": ("early", "middle", "late")[state_number % 3],
        "decision_index": state_number,
        "prefix_actions": [],
        "state": state,
        "state_fingerprint": state_fingerprint(state),
        "source_selected_candidate_id": candidate["candidate_id"],
        "candidates": [candidate],
    }


class SequenceEnvironment:
    def __init__(self, conflict_sequence: list[int]) -> None:
        self.conflict_sequence = conflict_sequence
        self.index = 0
        self.state = make_state(conflict_sequence[0])

    def reset(self, seed: int) -> dict:
        self.index = 0
        self.state = make_state(self.conflict_sequence[0])
        return copy.deepcopy(self.state)

    def step(self, action: dict) -> dict:
        self.index += 1
        conflicts = self.conflict_sequence[min(self.index, len(self.conflict_sequence) - 1)]
        self.state = make_state(conflicts, feasible=conflicts == 0, iteration=self.index)
        return {
            "observation": copy.deepcopy(self.state),
            "metrics": {
                "action_valid": True,
                "neighborhood": sorted(action["agents"]),
                "step_runtime": 0.01,
            },
            "terminated": conflicts == 0,
            "truncated": False,
        }


class SequentialCreditAuditTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_excludes_validation(self) -> None:
        rows = [make_source(map_number, state_number) for map_number in range(12) for state_number in range(9)]
        rows.append(make_source(99, 0, split="policy_validation"))
        config = make_config()
        first = select_audit_states(rows, config)
        second = select_audit_states(list(reversed(rows)), config)
        self.assertEqual([row["state_id"] for row in first], [row["state_id"] for row in second])
        self.assertEqual(len(first), 96)
        counts = {}
        for row in first:
            counts[row["map_id"]] = counts.get(row["map_id"], 0) + 1
            self.assertEqual(row["split"], "policy_train")
        self.assertEqual(set(counts.values()), {8})

    def test_conflict_severity_uses_normalized_density(self) -> None:
        row = make_source(0, 0)
        row["state"]["num_of_colliding_pairs"] = 1
        self.assertEqual(conflict_severity(row, [0.001, 0.01]), "low")
        row["state"]["num_of_colliding_pairs"] = 100
        self.assertEqual(conflict_severity(row, [0.001, 0.01]), "high")

    def test_dry_run_counts_trials_and_repair_bounds(self) -> None:
        config = make_config()
        selected = [make_source(0, 0), make_source(0, 1)]
        selected[0]["candidates"] *= 2
        report = build_dry_run(
            config, selected, smoke_states=None, observed_repair_seconds_min=0.01
        )
        self.assertEqual(report["state_count"], 2)
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(report["trial_jobs"], 12)
        self.assertEqual(report["repair_count_lower_bound"], 12)
        self.assertEqual(report["repair_count_upper_bound"], 48)
        self.assertAlmostEqual(
            report["serial_native_repair_seconds_empirical_lower_bound"], 0.12
        )

    def test_early_feasible_trial_pads_horizon_with_zero(self) -> None:
        environment = SequenceEnvironment([3, 0])
        source = make_source(0, 0)
        source["state"] = make_state(3)
        source["state_fingerprint"] = state_fingerprint(source["state"])
        with patch("experiments.sequential_credit_audit.analyze_static_grid", return_value=None):
            result = execute_horizon4_trial(
                environment,
                source,
                source["candidates"][0],
                0,
                SimpleNamespace(),
                {},
            )
        self.assertEqual(result["raw_conflict_trajectory"], [3, 0])
        self.assertEqual(result["padded_conflict_trajectory"], [3, 0, 0, 0, 0])
        self.assertEqual(result["h4"]["conflict_auc"], 1.5)
        self.assertTrue(result["h1"]["feasible"])

    def test_frozen_continuation_runs_after_initial_candidate(self) -> None:
        environment = SequenceEnvironment([4, 3, 0])
        source = make_source(0, 0)
        source["state"] = make_state(4)
        source["state_fingerprint"] = state_fingerprint(source["state"])
        continuation = make_candidate("continuation")
        bundle = SimpleNamespace(
            models={"realized_dynamic": object()},
            ranges={"realized_dynamic": {}},
        )
        with (
            patch("experiments.sequential_credit_audit.analyze_static_grid", return_value=None),
            patch(
                "experiments.sequential_credit_audit.generate_online_candidates",
                return_value=([continuation], {"proposal_count": 1, "candidate_count": 1, "proposal_seconds": 0.0}),
            ),
            patch("experiments.sequential_credit_audit.online_candidate_rows", return_value=[{"features": {"realized_dynamic": {}}, "candidate_key": "continuation"}]),
            patch("experiments.sequential_credit_audit.score_online_candidates", return_value=(0, [1.0], 1.0)),
            patch("experiments.sequential_credit_audit.feature_range_diagnostic", return_value={"outside_fraction": 0.0}),
            patch("experiments.sequential_credit_audit.repair_random_seed", return_value=123),
        ):
            result = execute_horizon4_trial(environment, source, source["candidates"][0], 0, bundle, {})
        self.assertEqual(result["raw_conflict_trajectory"], [4, 3, 0])
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][1]["controller"]["policy"], "frozen_v1_realized_dynamic")

    def test_pareto_and_best_sets_ignore_candidate_order(self) -> None:
        candidates = [
            {"candidate_id": "a", "h4": {"feasible_rate": 1.0, "final_conflicts": 0.0, "conflict_auc": 2.0}},
            {"candidate_id": "b", "h4": {"feasible_rate": 1.0, "final_conflicts": 0.0, "conflict_auc": 1.0}},
            {"candidate_id": "c", "h4": {"feasible_rate": 0.5, "final_conflicts": 1.0, "conflict_auc": 0.5}},
        ]
        self.assertEqual(pareto_ids(candidates, "h4"), pareto_ids(list(reversed(candidates)), "h4"))
        self.assertEqual(best_ids(candidates, "h4"), {"b"})

    def test_split_half_spearman_uses_candidate_identity(self) -> None:
        first = [
            {"candidate_id": "a", "feasible_rate": 1.0, "final_conflicts": 0.0, "conflict_auc": 1.0},
            {"candidate_id": "b", "feasible_rate": 0.0, "final_conflicts": 2.0, "conflict_auc": 2.0},
        ]
        self.assertAlmostEqual(_spearman_effectiveness(first, list(reversed(first))), 1.0)

    def test_four_trial_aggregation_and_analysis(self) -> None:
        source = make_source(0, 0)
        second = make_candidate("second")
        source["candidates"].append(second)
        rows = []
        for candidate in source["candidates"]:
            for trial_index in range(4):
                best = candidate["candidate_id"] == source["source_selected_candidate_id"]
                final = 0 if best else 2
                rows.append(
                    {
                        "state_id": source["state_id"],
                        "candidate_id": candidate["candidate_id"],
                        "trial_index": trial_index,
                        "complete": True,
                        "outcome": {
                            "h1": {"feasible": best, "final_conflicts": final, "conflict_auc": float(final)},
                            "h4": {"feasible": best, "final_conflicts": final, "conflict_auc": float(final)},
                            "low_level": {"generated": 10},
                        },
                    }
                )
        source["conflict_severity"] = "low"
        index = aggregate_trials([source], rows, 4)
        self.assertEqual(index[0]["candidate_count"], 2)
        self.assertEqual(index[0]["h4_best_ids"], [source["source_selected_candidate_id"]])
        report = analyze_index(index, make_config(), formal=False)
        self.assertEqual(report["metrics"]["trial_count"], 8)
        self.assertEqual(report["metrics"]["integrity_error_count"], 0)

    def test_missing_trial_is_rejected(self) -> None:
        source = make_source(0, 0)
        source["conflict_severity"] = "low"
        with self.assertRaisesRegex(ValueError, "incomplete trials"):
            aggregate_trials([source], [], 4)


if __name__ == "__main__":
    unittest.main()
