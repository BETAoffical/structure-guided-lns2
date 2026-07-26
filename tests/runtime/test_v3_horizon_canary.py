from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.v3_horizon_canary import (
    canary_eligible_state_ids,
    replay_historical_rollout,
    select_canary_rollouts,
    select_canary_state_ids,
)


class HorizonCanaryTests(unittest.TestCase):
    def test_state_selection_round_robins_layout_agent_cells(self) -> None:
        decisions = [
            {
                "state_id": f"{layout}-{agents}-{index}",
                "layout_mode": layout,
                "agent_count": agents,
            }
            for layout in ("a", "b")
            for agents in (100, 600)
            for index in range(2)
        ]
        selected = select_canary_state_ids(decisions, 4)
        self.assertEqual(
            set(selected),
            {"a-100-0", "a-600-0", "b-100-0", "b-600-0"},
        )

    def test_state_selection_excludes_states_without_full_size_coverage(self) -> None:
        decisions = [
            {"state_id": "eligible", "layout_mode": "a", "agent_count": 100},
            {"state_id": "ineligible", "layout_mode": "a", "agent_count": 100},
        ]
        selected = select_canary_state_ids(
            decisions, 1, eligible_state_ids={"eligible"}
        )
        self.assertEqual(selected, ["eligible"])

    def test_eligible_states_require_adaptive_and_all_three_sizes(self) -> None:
        rows = []
        for state_id, sizes in (("complete", (4, 8, 16)), ("partial", (4, 8))):
            rows.append(
                {
                    "state_id": state_id,
                    "candidate_id": "official_adaptive",
                    "trial_index": 0,
                    "actual_size": 0,
                }
            )
            rows.extend(
                {
                    "state_id": state_id,
                    "candidate_id": f"size-{size}",
                    "trial_index": 0,
                    "actual_size": size,
                }
                for size in sizes
            )
        self.assertEqual(canary_eligible_state_ids(rows), {"complete"})

    def test_rollout_selection_keeps_trial_zero_adaptive_and_each_size(self) -> None:
        rows = []
        for trial in (0, 1):
            rows.append(
                {
                    "state_id": "state",
                    "candidate_id": "official_adaptive",
                    "trial_index": trial,
                    "actual_size": 0,
                }
            )
            for size in (4, 8, 16):
                rows.append(
                    {
                        "state_id": "state",
                        "candidate_id": f"size-{size}",
                        "trial_index": trial,
                        "actual_size": size,
                    }
                )
        selected, errors = select_canary_rollouts(rows, ["state"])
        self.assertEqual(errors, [])
        self.assertEqual(len(selected), 4)
        self.assertEqual({int(row["trial_index"]) for row in selected}, {0})

    def test_fixed_action_replay_matches_repair_semantics(self) -> None:
        initial = {
            "num_of_colliding_pairs": 2,
            "feasible": False,
            "done": False,
        }
        transition = {
            "observation": {
                "num_of_colliding_pairs": 0,
                "feasible": True,
                "done": True,
            },
            "metrics": {
                "step_applied": True,
                "replan_success": True,
                "requested_random_seed": 7,
                "requested_pp_random_seed": 7,
                "applied_pp_random_seed": 7,
                "repair_order": [1, 2],
            },
            "terminated": True,
            "truncated": False,
        }

        class Environment:
            def step(self, _action):
                return transition

        historical = {
            "state_id": "state",
            "candidate_id": "candidate",
            "route": "model",
            "actual_size": 4,
            "trial_index": 0,
            "executed_steps": 1,
            "initial_repair_fingerprint": "repair-2",
            "final_repair_fingerprint": "repair-0",
            "conflict_trajectory": [2, 0],
            "steps": [
                {
                    "action": {
                        "mode": "explicit_neighborhood",
                        "agents": [1, 2],
                        "random_seed": 7,
                        "pp_random_seed": 7,
                    },
                    "before_repair_fingerprint": "repair-2",
                    "after_repair_fingerprint": "repair-0",
                    "conflicts_before": 2,
                    "conflicts_after": 0,
                    "repair_outcome": "feasible",
                }
            ],
            "h3": {"feasible": True},
        }

        def repair_fingerprint(state):
            return f"repair-{state['num_of_colliding_pairs']}"

        with (
            patch(
                "experiments.v3_horizon_canary._source_replay_job",
                return_value=({}, {}),
            ),
            patch(
                "experiments.v3_horizon_canary.replay_prefix",
                return_value=(Environment(), initial),
            ),
            patch(
                "experiments.v3_horizon_canary.repair_structure_fingerprint",
                side_effect=repair_fingerprint,
            ),
        ):
            row = replay_historical_rollout(
                {"prefix_actions": []}, historical
            )
        self.assertTrue(row["passed"])
        self.assertEqual(row["mismatch_count"], 0)
        self.assertEqual(row["replayed_steps"], 1)

    def test_legacy_seed_contract_does_not_require_explicit_pp_seed(self) -> None:
        initial = {
            "num_of_colliding_pairs": 2,
            "feasible": False,
            "done": False,
        }
        transition = {
            "observation": {
                "num_of_colliding_pairs": 1,
                "feasible": False,
                "done": False,
            },
            "metrics": {
                "step_applied": True,
                "replan_success": True,
                "requested_random_seed": 7,
                "requested_pp_random_seed": -1,
                "applied_pp_random_seed": -1,
                "repair_order": [1, 2],
            },
            "terminated": False,
            "truncated": False,
        }

        class Environment:
            def step(self, _action):
                return transition

        historical = {
            "state_id": "state",
            "candidate_id": "candidate",
            "route": "model",
            "actual_size": 4,
            "trial_index": 0,
            "executed_steps": 1,
            "initial_repair_fingerprint": "repair-2",
            "final_repair_fingerprint": "repair-1",
            "conflict_trajectory": [2, 1],
            "steps": [
                {
                    "action": {
                        "mode": "explicit_neighborhood",
                        "agents": [1, 2],
                        "random_seed": 7,
                    },
                    "before_repair_fingerprint": "repair-2",
                    "after_repair_fingerprint": "repair-1",
                    "conflicts_before": 2,
                    "conflicts_after": 1,
                    "repair_outcome": "conflict_reduced",
                }
            ],
            "h3": {"feasible": False},
        }

        def repair_fingerprint(state):
            return f"repair-{state['num_of_colliding_pairs']}"

        with (
            patch(
                "experiments.v3_horizon_canary._source_replay_job",
                return_value=({}, {}),
            ),
            patch(
                "experiments.v3_horizon_canary.replay_prefix",
                return_value=(Environment(), initial),
            ),
            patch(
                "experiments.v3_horizon_canary.repair_structure_fingerprint",
                side_effect=repair_fingerprint,
            ),
        ):
            row = replay_historical_rollout({"prefix_actions": []}, historical)
        self.assertTrue(row["passed"])
        self.assertEqual(
            row["steps"][0]["seed_contract"], "legacy-random-seed-coupled"
        )


if __name__ == "__main__":
    unittest.main()
