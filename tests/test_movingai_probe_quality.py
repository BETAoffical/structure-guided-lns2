from __future__ import annotations

import unittest

from experiments.movingai_probe_quality import (
    _best_effectiveness_keys,
    _density_alignment_statistic,
    _exact_context_signals,
    _mean_pairwise_jaccard,
    _unique_multiset_permutations,
    audit_probe_records,
)


def _state(map_id: str, episode: int) -> dict:
    return {
        "state_id": f"{map_id}-episode-{episode}-decision-0",
        "state_fingerprint": f"fingerprint-{map_id}",
        "episode_id": f"{map_id}-episode-{episode}",
        "decision_index": 0,
        "state": {
            "context": {
                "map_id": map_id,
                "task_id": f"{map_id}-task",
                "agent_count": 10,
                "scenario_type": "movingai_random_1",
            }
        },
    }


def _outcome(state: dict, heuristic: str, trial: int, conflicts: int) -> dict:
    action = {
        "seed_agent": 0,
        "heuristic": heuristic,
        "neighborhood_size": 4,
    }
    neighborhood = [0, 1, 2, 3] if heuristic == "target" else [0, 4, 5, 6]
    return {
        "state_id": state["state_id"],
        "state_fingerprint": state["state_fingerprint"],
        "episode_id": state["episode_id"],
        "trial_index": trial,
        "trial_seed": episode_seed(state["episode_id"], heuristic, trial),
        "candidate_action": action,
        "conflict_trajectory": [4, conflicts],
        "steps": [
            {"step": 0, "conflicts": 4},
            {"step": 1, "metrics": {"neighborhood": neighborhood}},
        ],
        "horizon_outcomes": [
            {
                "horizon": 1,
                "available": True,
                "solved": conflicts == 0,
                "conflicts_after": conflicts,
                "conflict_auc": (4 + conflicts) / 2.0,
                "branch_runtime": 0.01,
                "low_level_delta": {"generated": 10 + conflicts},
            }
        ],
    }


def episode_seed(episode: str, heuristic: str, trial: int) -> int:
    return sum(ord(value) for value in f"{episode}-{heuristic}-{trial}")


def _settings() -> dict:
    return {
        "bootstrap_samples": 20,
        "winner_bootstrap_samples": 20,
        "maximum_exact_assignments": 1000,
        "monte_carlo_permutations": 100,
        "minimum_unique_states": 2,
        "minimum_repairable_tasks": 2,
        "minimum_repair_label_maps": 2,
        "minimum_density_pair_maps": 0,
        "minimum_scenarios_per_map": 1,
        "minimum_layout_replicates_per_family": 1,
        "minimum_trials_per_candidate": 4,
        "minimum_mean_statewise_action_effect": 0.0,
        "minimum_trial_split_spearman": -1.0,
        "minimum_neighborhood_jaccard": 0.0,
        "maximum_exact_context_p_value": 1.0,
    }


class MovingAIProbeQualityTests(unittest.TestCase):
    def test_best_effectiveness_prefers_feasibility_before_mean_conflicts(self) -> None:
        candidates = [
            {
                "candidate_key": "sometimes-solves",
                "solved_rate": 0.5,
                "conflicts_after": 5.0,
            },
            {
                "candidate_key": "never-solves",
                "solved_rate": 0.0,
                "conflicts_after": 4.0,
            },
        ]
        self.assertEqual(
            _best_effectiveness_keys(candidates), {"sometimes-solves"}
        )

    def test_exact_map_permutations_preserve_label_multiplicity(self) -> None:
        permutations = list(
            _unique_multiset_permutations(["a", "a", "b", "b", "c", "c", "d"])
        )
        self.assertEqual(len(permutations), 630)
        self.assertEqual(len(set(permutations)), 630)

    def test_directional_density_statistic_detects_aligned_changes(self) -> None:
        families = ["target:4", "random:4"]
        units = []
        counts = []
        for map_id in ("a", "b", "c"):
            units.extend(
                [
                    {
                        "map_id": map_id,
                        "agent_count": 10,
                        "support": {"target:4": 0.0, "random:4": 1.0},
                    },
                    {
                        "map_id": map_id,
                        "agent_count": 20,
                        "support": {"target:4": 1.0, "random:4": 0.0},
                    },
                ]
            )
            counts.extend([10, 20])
        observed = _density_alignment_statistic(units, counts, families)
        reversed_one = list(counts)
        reversed_one[0], reversed_one[1] = reversed_one[1], reversed_one[0]
        self.assertGreater(observed, _density_alignment_statistic(units, reversed_one, families))
        report = _exact_context_signals(units, families)
        self.assertEqual(
            report["density_directional_alignment"]["assignment_count"], 8
        )
        self.assertEqual(report["density_directional_alignment"]["method"], "exact")

    def test_pairwise_neighborhood_jaccard_reports_stochasticity(self) -> None:
        self.assertEqual(_mean_pairwise_jaccard([[1, 2], [1, 2]]), 1.0)
        self.assertAlmostEqual(
            _mean_pairwise_jaccard([[1, 2], [2, 3]]), 1.0 / 3.0
        )

    def test_audit_pools_duplicate_solver_states_and_detects_auc_redundancy(self) -> None:
        states = []
        outcomes = []
        qualification = []
        for map_id in ("map-a", "map-b"):
            qualification.append(
                {
                    "status": "ok",
                    "repairable": True,
                    "map_id": map_id,
                    "task_id": f"{map_id}-task",
                }
            )
            for episode in (0, 1):
                state = _state(map_id, episode)
                states.append(state)
                for trial in (0, 1):
                    target = 0 if map_id == "map-a" else 3
                    collision = 3 if map_id == "map-a" else 0
                    outcomes.append(_outcome(state, "target", trial, target))
                    outcomes.append(_outcome(state, "collision", trial, collision))
        report = audit_probe_records(
            qualification, states, outcomes, _settings()
        )
        self.assertEqual(report["coverage"]["raw_state_rows"], 4)
        self.assertEqual(report["coverage"]["unique_state_fingerprints"], 2)
        self.assertEqual(report["coverage"]["minimum_pooled_trials_per_candidate"], 4)
        self.assertEqual(
            report["label_quality"]["horizon1_auc_affine_identity_max_error"], 0.0
        )
        self.assertFalse(
            report["label_quality"]["horizon1_auc_is_independent_objective"]
        )


if __name__ == "__main__":
    unittest.main()
