from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.v3_pilot import _coverage_report, decision_stage, select_v3_pilot_states
from experiments.v3_training import _gate_checks, _outcome_name, _rows


def source_rows(split: str, quota: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    serial = 0
    for layout in ("maze", "room", "random"):
        for agents in (400, 600):
            cell: list[tuple[bool, float]] = []
            cell.extend((True, 10.0) for _ in range(quota))
            cell.extend((False, 100.0 - index) for index in range(4 * quota))
            for no_progress, repair_seconds in cell:
                map_id = f"{split}-{layout}-{agents}-{serial % max(2, quota)}"
                rows.append(
                    {
                        "split": split,
                        "map_id": map_id,
                        "layout_mode": layout,
                        "agent_count": agents,
                        "task_id": f"task-{split}-{serial}",
                        "solver_seed": 1,
                        "episode_id": f"episode-{split}-{serial}",
                        "decision_index": serial % 30,
                        "decision_stage": decision_stage(serial % 30),
                        "before_repair_fingerprint": f"fingerprint-{split}-{serial}",
                        "actual_metrics": {"replan_success": not no_progress},
                        "repair_state_changed": not no_progress,
                        "source_repair_seconds": repair_seconds,
                        "source_controller_seconds": 0.1,
                    }
                )
                serial += 1
    return rows


class V3PilotSelectionTests(unittest.TestCase):
    def test_balanced_selection_has_registered_144_and_36_states(self) -> None:
        train = source_rows("policy_train", 8)
        diagnostic = source_rows("policy_validation", 2)

        def decisions(_root: Path, split: str) -> list[dict[str, object]]:
            return train if split == "policy_train" else diagnostic

        with patch("experiments.v3_pilot._source_decisions", side_effect=decisions):
            selected, report = select_v3_pilot_states(Path("unused"))
        self.assertEqual(len(selected), 180)
        self.assertEqual(
            report["selected_state_count_by_split"],
            {"policy_train": 144, "policy_validation": 36},
        )
        self.assertFalse(report["train_diagnostic_map_overlap"])
        self.assertEqual(len({row["before_repair_fingerprint"] for row in selected}), 180)

    def test_coverage_rejects_missing_nominal_size(self) -> None:
        decisions = [{"state_id": "state"}]
        features = [
            {
                "state_id": "state",
                "candidate_id": f"c{size}",
                "route": "model",
                "selection_families": [f"target:{size}"],
                "base_selected": size == 4,
            }
            for size in (4, 8)
        ] + [
            {
                "state_id": "state",
                "candidate_id": "official_adaptive",
                "route": "official_adaptive",
                "selection_families": [],
                "base_selected": False,
            }
        ]
        trials = [
            {"state_id": "state", "candidate_id": row["candidate_id"]}
            for row in features
            for _ in range(2)
        ]
        report = _coverage_report(decisions, features, trials, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(any("represented sizes" in error for error in report["errors"]))


class V3TrainingRowsTests(unittest.TestCase):
    def test_conflict_reduction_retention_is_diagnostic_not_a_hard_gate(self) -> None:
        checks, diagnostics = _gate_checks(
            {
                "effective_rate_delta": 0.0,
                "no_progress_rate_delta": 0.0,
                "conflict_reduction_ratio": 0.67,
                "efficiency_ratio": 1.14,
            },
            1.30,
            {
                "cell_count": 6,
                "noninferior_cell_count": 5,
                "worst_efficiency_ratio": 0.95,
            },
        )
        self.assertTrue(all(checks.values()))
        self.assertFalse(
            diagnostics["mean_conflict_reduction_retention_at_least_98pct"]
        )

    def test_underfilled_nominal_candidate_is_not_discarded(self) -> None:
        features = [
            {
                "split": "policy_train",
                "state_id": "state",
                "candidate_id": "underfilled-8",
                "map_id": "map",
                "layout_mode": "maze",
                "agent_count": 400,
                "route": "model",
                "actual_size": 7,
                "base_selected": True,
                "main_score": 1.0,
                "features": {"realized_dynamic": {"x": 2.0}},
            }
        ]
        trials = [
            {
                "split": "policy_train",
                "state_id": "state",
                "candidate_id": "underfilled-8",
                "status": "ok",
                "complete": True,
                "outcome": {
                    "repair_outcome": "accepted_noop",
                    "conflicts_before": 3,
                    "conflicts_after": 3,
                    "repair_seconds": 0.5,
                },
            }
        ]
        rows = _rows(features, trials, "policy_train", ("x",))
        self.assertEqual(rows[0]["actual_size"], 7)
        self.assertEqual(rows[0]["no_progress"], 1)

    def test_outcome_labels_distinguish_accepted_noop(self) -> None:
        self.assertEqual(
            _outcome_name(
                {
                    "conflicts_before": 5,
                    "conflicts_after": 5,
                    "hard_failure": False,
                    "repair_state_changed": False,
                }
            ),
            "accepted_noop",
        )


if __name__ == "__main__":
    unittest.main()
