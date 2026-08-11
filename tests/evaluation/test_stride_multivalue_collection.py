from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.stride_multivalue_collection import (
    ROLLOUT_STATE_SCHEMA,
    _collect_rollout_state,
    _direction_against_anchor,
    _episode_job_id,
    _recovery_window_decision,
    _single_rollout,
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
    def test_productive_timeout_does_not_consume_failure_budget(self) -> None:
        decision = _recovery_window_decision(
            status="timeout", previous_count=10, completed_count=15
        )
        self.assertEqual(decision["classification"], "productive_window")
        self.assertEqual(decision["new_episode_checkpoint_count"], 5)
        self.assertEqual(decision["failure_budget_increment"], 0)
        stalled = _recovery_window_decision(
            status="timeout", previous_count=15, completed_count=15
        )
        self.assertEqual(stalled["classification"], "no_progress_window")
        self.assertEqual(stalled["failure_budget_increment"], 1)
        errored = _recovery_window_decision(
            status="error", previous_count=15, completed_count=16
        )
        self.assertEqual(errored["classification"], "execution_error")
        self.assertEqual(errored["failure_budget_increment"], 1)

    def test_compatible_run_fingerprint_resumes_completed_state(self) -> None:
        with TemporaryDirectory() as temporary:
            state = {
                "state_occurrence_id": "occurrence",
                "candidates": [{"candidate_id": "candidate"}],
            }
            episode_id = _episode_job_id("v2-full", "candidate", 0)
            output = Path(temporary) / "state.json"
            output.write_text(
                json.dumps(
                    {
                        "schema": ROLLOUT_STATE_SCHEMA,
                        "run_fingerprint": "legacy",
                        "complete": True,
                        "completed_episode_job_ids": [episode_id],
                        "episodes": [],
                    }
                ),
                encoding="utf-8",
            )
            result = _collect_rollout_state(
                {
                    "state_record": state,
                    "output_path": str(output),
                    "run_fingerprint": "new",
                    "compatible_run_fingerprints": ["legacy"],
                    "teachers": ["v2-full"],
                    "trial_indices": [0],
                }
            )
            self.assertEqual(result["status"], "resumed")
            self.assertEqual(result["episode_count"], 1)

    def test_single_rollout_reuses_qualification_before_policy(self) -> None:
        with TemporaryDirectory() as temporary:
            job = {
                "root": temporary,
                "parent": {},
                "dataset": str(Path(temporary) / "dataset"),
                "runtime": str(Path(temporary) / "runtime.json"),
                "qualification": str(Path(temporary) / "qualification"),
                "cohort_job_keys": [("task", 3), ("other", 5)],
                "state_root": str(Path(temporary) / "state"),
            }
            state = {
                "task_id": "task",
                "solver_seed": 3,
                "state_occurrence_id": "occurrence",
            }
            candidate = {
                "candidate_id": "candidate",
                "candidate_position": 0,
                "multivalue_role": "paretopool_challenger",
                "agents": [1, 2],
            }
            manifest = {
                "task_id": "task",
                "solver_seed": 3,
                "status": "ok",
                "trace_sha256": "f" * 64,
                "summary": {
                    "stop_reason": "success",
                    "controller_totals": {"forced_first_action_count": 1},
                },
            }
            with (
                mock.patch(
                    "experiments.stride_multivalue_collection._teacher_kwargs",
                    return_value=("realized_dynamic", {}),
                ),
                mock.patch(
                    "experiments.stride_multivalue_collection._episode_override",
                    return_value={"forced": True},
                ),
                mock.patch(
                    "experiments.stride_multivalue_collection.run_closed_loop_collection"
                ) as run,
                mock.patch(
                    "experiments.stride_multivalue_collection._read_jsonl",
                    return_value=[manifest],
                ),
            ):
                result = _single_rollout(
                    job, state, candidate, teacher="v2-full", trial_index=0
                )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(run.call_count, 2)
            first = run.call_args_list[0].kwargs
            second = run.call_args_list[1].kwargs
            self.assertEqual(first["phase"], "qualify")
            self.assertEqual(first["job_keys"], {("task", 3), ("other", 5)})
            self.assertIn("qualification_source", first)
            self.assertEqual(second["phase"], "realized_dynamic")
            self.assertEqual(second["job_keys"], {("task", 3)})
            self.assertNotIn("qualification_source", second)
            self.assertTrue(second["resume"])

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
