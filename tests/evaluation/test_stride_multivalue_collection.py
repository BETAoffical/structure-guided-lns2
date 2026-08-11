from __future__ import annotations

import unittest

from experiments.stride_multivalue_collection import (
    _direction_against_anchor,
    _stability_group,
    history_before_decision,
    select_pilot_occurrences,
)


def _state(edges: list[tuple[int, int]], revision: int) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "rows": 1,
        "cols": 7,
        "sum_of_costs": 6 + revision,
        "obstacles": [0] * 7,
        "agents": [
            {"id": agent, "path": [agent, agent + 1]}
            for agent in range(6)
        ],
        "conflict_edges": [list(edge) for edge in edges],
        "num_of_colliding_pairs": len(edges),
        "feasible": not edges,
        "version": revision,
    }


def _transition(agents: list[int], candidate: str, success: bool = True) -> dict:
    return {
        "action": {"agents": agents},
        "metrics": {"neighborhood": agents, "replan_success": success},
        "controller": {"selected_candidate_id": candidate},
    }


class MultiValueCollectionTest(unittest.TestCase):
    def test_anchor_direction_uses_future_value_order(self) -> None:
        def summary(auc: float) -> dict:
            return {
                "horizons": {
                    str(horizon): {
                        "feasible_probability": 0.5,
                        "right_censored_fraction": 0.0,
                        "mean_normalized_conflict_auc": auc,
                        "mean_original_edge_residual_rate": 0.25,
                        "mean_new_conflict_edge_rate": 0.1,
                    }
                    for horizon in (1, 8, 32, 128)
                }
            }

        self.assertEqual(
            _direction_against_anchor(
                "challenger", summary(0.2), "anchor", summary(0.3)
            ),
            1,
        )
        self.assertEqual(
            _direction_against_anchor(
                "challenger", summary(0.4), "anchor", summary(0.3)
            ),
            -1,
        )

    def test_stability_group_requires_every_registered_gate(self) -> None:
        states = [
            {
                "teachers": [
                    {
                        "top3_overlap": 1.0,
                        "paired_rank_correlation": 0.8,
                        "cross_seed_normalized_regret": 0.01,
                    },
                    {
                        "top3_overlap": 1.0,
                        "paired_rank_correlation": 0.7,
                        "cross_seed_normalized_regret": 0.0,
                    },
                ]
            }
        ]
        directions = [{"directions_agree": True} for _ in range(4)]
        gates = {
            "seed_half_top3_overlap_minimum": 0.8,
            "paired_rank_correlation_minimum": 0.6,
            "cross_teacher_direction_agreement_minimum": 0.7,
            "cross_seed_normalized_regret_maximum": 0.02,
        }
        self.assertTrue(_stability_group(states, directions, gates)["passed"])
        directions[0]["directions_agree"] = False
        directions[1]["directions_agree"] = False
        self.assertFalse(_stability_group(states, directions, gates)["passed"])

    def test_history_uses_only_strict_prefix_and_keeps_repetition(self) -> None:
        states = [
            _state([(0, 1)], 0),
            _state([(0, 1), (2, 3)], 1),
            _state([(0, 1)], 2),
            _state([(4, 5)], 3),
        ]
        transitions = [
            _transition([0, 1], "a"),
            _transition([0, 1], "a"),
            _transition([4, 5], "future"),
        ]
        history = history_before_decision(
            states, transitions, decision_index=2
        )
        self.assertEqual(history.recent_candidate_ids, ("a", "a"))
        self.assertEqual(history.recent_neighborhood_exact_repeat, (False, True))
        self.assertEqual(history.persistent_conflict_edges, ((0, 1),))
        self.assertEqual(history.disappeared_conflict_edges, ((2, 3),))
        self.assertNotIn("future", history.recent_candidate_ids)

    def test_pilot_selection_is_map_balanced_and_deterministic(self) -> None:
        rows = []
        for map_id in ("map-a", "map-b"):
            for task in range(3):
                for decision in range(3):
                    rows.append(
                        {
                            "map_id": map_id,
                            "task_id": f"{map_id}-task-{task}",
                            "case_id": f"{map_id}-{task}-{decision}",
                            "decision_index": decision,
                        }
                    )
        first = select_pilot_occurrences(rows, per_map=4)
        second = select_pilot_occurrences(list(reversed(rows)), per_map=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(
            {map_id: sum(row["map_id"] == map_id for row in first)
             for map_id in ("map-a", "map-b")},
            {"map-a": 4, "map-b": 4},
        )


if __name__ == "__main__":
    unittest.main()
