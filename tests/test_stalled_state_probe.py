from __future__ import annotations

import unittest

from experiments.stalled_state_probe import (
    choose_probe_branches,
    find_terminal_stall,
    paired_probe_seed,
    summarize_probe_trials,
)
from experiments.stall_guard import repair_structure_fingerprint


class StalledStateProbeTests(unittest.TestCase):
    def test_repair_structure_ignores_attempt_counters_and_timeout_flag(self) -> None:
        base = {
            "initialized": True,
            "initial_solution_complete": True,
            "feasible": False,
            "done": False,
            "iteration": 1,
            "rows": 2,
            "cols": 2,
            "sum_of_costs": 4,
            "num_of_colliding_pairs": 1,
            "low_level": {"expanded": 10},
            "runtime": 1.0,
            "obstacles": [],
            "conflict_edges": [[0, 1]],
            "agents": [{"id": 0, "path": [[0, 0], [0, 1]]}],
        }
        after = dict(base)
        after.update(
            {
                "done": True,
                "iteration": 2,
                "low_level": {"expanded": 20},
                "runtime": 2.0,
            }
        )
        self.assertEqual(
            repair_structure_fingerprint(base), repair_structure_fingerprint(after)
        )

    def test_terminal_stall_selects_first_failed_unchanged_decision(self) -> None:
        decisions = [
            {
                "decision_index": 0,
                "before_fingerprint": "a",
                "after_fingerprint": "b",
                "before_conflicts": 2,
                "actual_metrics": {"replan_success": True},
            },
            *[
                {
                    "decision_index": index,
                    "before_fingerprint": "b",
                    "after_fingerprint": "b",
                    "before_conflicts": 1,
                    "actual_metrics": {"replan_success": False, "conflicts_after": 1},
                }
                for index in range(1, 5)
            ],
        ]
        stall = find_terminal_stall(decisions)
        self.assertEqual(stall["start_decision_index"], 1)
        self.assertEqual(stall["length"], 4)

    def test_terminal_stall_rejects_short_or_successful_tail(self) -> None:
        with self.assertRaises(ValueError):
            find_terminal_stall(
                [
                    {
                        "decision_index": 0,
                        "before_fingerprint": "a",
                        "after_fingerprint": "a",
                        "before_conflicts": 1,
                        "actual_metrics": {
                            "replan_success": True,
                            "conflicts_after": 1,
                        },
                    }
                ]
            )

    def test_branch_selection_uses_full_ranking_and_deduplicates(self) -> None:
        candidates = [
            {"candidate_id": "c16", "actual_size": 16, "agents": list(range(16))},
            {"candidate_id": "c8", "actual_size": 8, "agents": list(range(8))},
            {"candidate_id": "c4", "actual_size": 4, "agents": list(range(4))},
        ]
        branches, aliases = choose_probe_branches(candidates, [3.0, 2.0, 1.0])
        self.assertEqual(aliases["rank1"], "c16")
        self.assertEqual(aliases["rank2"], "c8")
        self.assertEqual(aliases["size_le_8"], "c8")
        self.assertEqual(aliases["size_le_4"], "c4")
        self.assertEqual(len(branches), 4)

    def test_paired_seeds_are_stable_and_unique(self) -> None:
        first = [paired_probe_seed("state", index) for index in range(8)]
        second = [paired_probe_seed("state", index) for index in range(8)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_gate_accepts_success_rate_or_positive_escape(self) -> None:
        aliases = {
            "rank1": "rank1",
            "rank2": "rank2",
            "size_le_8": "size8",
            "size_le_4": "size4",
            "official_adaptive": "official",
        }
        trials = []
        for key in set(aliases.values()):
            for trial in range(8):
                success = key == "size8" and trial < 2
                conflict_delta = 1 if key == "official" and trial < 2 else 0
                trials.append(
                    {
                        "branch_key": key,
                        "replan_success": success,
                        "conflict_delta": conflict_delta,
                        "conflicts_after": 10 - conflict_delta,
                        "pp_replan_seconds": 1.0,
                        "total_decision_seconds": 1.1,
                    }
                )
        _summaries, gate = summarize_probe_trials(trials, aliases)
        self.assertTrue(gate["passed"])
        self.assertEqual(
            set(gate["supported_alternatives"]), {"size_le_8", "official_adaptive"}
        )


if __name__ == "__main__":
    unittest.main()
