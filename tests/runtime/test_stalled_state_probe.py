from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stalled_state_probe import (
    _checkpoint_valid,
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
                    "before_repair_fingerprint": "repair-b",
                    "after_repair_fingerprint": "repair-b",
                    "repair_state_changed": False,
                    "before_conflicts": 1,
                    "actual_metrics": {"replan_success": False, "conflicts_after": 1},
                }
                for index in range(1, 5)
            ],
        ]
        stall = find_terminal_stall(decisions)
        self.assertEqual(stall["start_decision_index"], 1)
        self.assertEqual(stall["length"], 4)

    def test_terminal_stall_rejects_equal_conflicts_when_structure_changed(self) -> None:
        decisions = [
            {
                "decision_index": index,
                "before_fingerprint": f"full-{index}",
                "after_fingerprint": f"full-{index + 1}",
                "before_repair_fingerprint": f"repair-{index}",
                "after_repair_fingerprint": f"repair-{index + 1}",
                "repair_state_changed": True,
                "before_conflicts": 1,
                "actual_metrics": {
                    "replan_success": False,
                    "conflicts_after": 1,
                },
            }
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "no terminal unchanged-state"):
            find_terminal_stall(decisions)

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

    def test_all_candidate_probe_covers_every_unique_candidate(self) -> None:
        candidates = [
            {"candidate_id": "c4", "actual_size": 4, "agents": [0, 1]},
            {"candidate_id": "c8", "actual_size": 8, "agents": [2, 3]},
            {"candidate_id": "c16", "actual_size": 16, "agents": [4, 5]},
        ]
        branches, aliases = choose_probe_branches(
            candidates, [0.2, 0.9, 0.5], all_candidates=True
        )
        self.assertEqual(len(branches), 4)
        by_alias = {
            alias: next(
                row for row in branches if row["branch_key"] == branch_key
            )["candidate"]["candidate_id"]
            for alias, branch_key in aliases.items()
            if alias != "official_adaptive"
        }
        self.assertEqual(by_alias["rank1"], "c8")
        self.assertEqual(by_alias["rank_2"], "c16")
        self.assertEqual(by_alias["rank_3"], "c4")

    def test_paired_seeds_are_stable_and_unique(self) -> None:
        first = [paired_probe_seed("state", index) for index in range(8)]
        second = [paired_probe_seed("state", index) for index in range(8)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_one_step_summary_never_claims_an_oracle_gate(self) -> None:
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
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "oracle_audit_required")
        self.assertEqual(gate["supported_alternatives"], [])

    def test_completed_v1_checkpoint_is_rejected_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            payload = {
                "schema": "lns2.stalled_state_probe.v1",
                "schema_version": 1,
                "complete": True,
            }
            original = json.dumps(payload)
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint is invalid"):
                _checkpoint_valid(
                    path,
                    run_fingerprint="run",
                    branch={
                        "branch_key": "candidate",
                        "aliases": ["rank1"],
                        "mode": "explicit_neighborhood",
                        "candidate": {
                            "candidate_id": "candidate",
                            "agents": [1, 2],
                            "actual_size": 2,
                        },
                        "rank": 1,
                        "score": 1.0,
                    },
                    trial_index=0,
                    before_fingerprint="before",
                    before_repair_fingerprint="repair-before",
                )
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
