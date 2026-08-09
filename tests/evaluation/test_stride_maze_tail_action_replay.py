from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_maze_tail_action_replay import (
    robust_action_order,
    validate_preflight_config,
    validate_replay_config,
)


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_CONFIG = ROOT / "configs" / "stride_maze_tail_action_replay_preflight_v1.json"


def _rule() -> dict:
    return {
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "require_winner_no_progress_rate_not_worse": True,
        "first_fixed_half": list(range(8)),
        "second_fixed_half": list(range(8, 16)),
        "tie_epsilon": 1e-12,
    }


def _replay_config() -> dict:
    return {
        "schema": "lns2.stride.maze_tail_action_replay_config.v1",
        "scientific_status": "preregistered_first_divergence_sixteen_seed_paired_action_replay",
        "experiment_id": "stride-maze-tail-action-replay-v1",
        "trial_indices": list(range(16)),
        "strictly_paired_pp_seeds_within_state_and_index": True,
        "candidate_roles": ["v2_action", "challenger_action"],
        "primary_outcome": "current_step_conflict_reduction_normalized_by_before_conflicts",
        "runtime_used_in_label": False,
        "future_trajectory_read": False,
        "outcome_based_state_filtering": False,
        "robust_order_rule": _rule(),
        "analysis_gates": {
            "minimum_robustly_ordered_fraction": 0.70,
            "minimum_robustly_ordered_tail_count": 8,
            "minimum_robust_tail_map_count": 2,
            "minimum_robust_tail_task_count": 4,
            "minimum_robust_tail_solver_seed_count": 2,
            "minimum_robust_tail_per_challenger": 2,
        },
        "inputs": {
            "preflight_report": {"path": "build/a.json", "sha256": "a" * 64},
            "state_selection": {"path": "build/b.jsonl", "sha256": "b" * 64},
        },
        "expected_state_count": 24,
        "workers": 1,
        "state_timeout_seconds": 1800.0,
    }


class MazeTailActionReplayTest(unittest.TestCase):
    def test_preflight_config_is_frozen(self) -> None:
        config = json.loads(PREFLIGHT_CONFIG.read_text(encoding="utf-8"))
        validate_preflight_config(config)
        changed = copy.deepcopy(config)
        changed["preflight_gates"]["minimum_divergence_comparison_count"] = 1
        with self.assertRaises(ValueError):
            validate_preflight_config(changed)

    def test_replay_config_contract_is_frozen(self) -> None:
        config = _replay_config()
        validate_replay_config(config)
        changed = copy.deepcopy(config)
        changed["future_trajectory_read"] = True
        with self.assertRaises(ValueError):
            validate_replay_config(changed)

    def test_robust_challenger_requires_both_halves(self) -> None:
        result = robust_action_order(
            [0.0] * 16,
            [0.04] * 16,
            [True] * 16,
            [True] * 16,
            _rule(),
        )
        self.assertEqual(result["robust_winner"], "challenger_action")
        unstable = robust_action_order(
            [0.0] * 16,
            [0.08] * 8 + [-0.01] * 8,
            [True] * 16,
            [True] * 16,
            _rule(),
        )
        self.assertEqual(unstable["robust_winner"], "uncertain")

    def test_robust_v2_and_no_progress_guard(self) -> None:
        result = robust_action_order(
            [0.05] * 16,
            [0.0] * 16,
            [True] * 16,
            [False] * 16,
            _rule(),
        )
        self.assertEqual(result["robust_winner"], "v2_action")
        guarded = robust_action_order(
            [0.0] * 16,
            [0.04] * 16,
            [True] * 16,
            [False] * 16,
            _rule(),
        )
        self.assertEqual(guarded["robust_winner"], "uncertain")


if __name__ == "__main__":
    unittest.main()
