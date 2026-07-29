from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

from experiments.stall_training_extension import (
    _derived_config,
    _sequence_row,
    _source_protocol_path,
    _validate_source_protocol,
    unfinished_training_manifests,
)


def _manifest(task: str, *, success: bool, stop_reason: str) -> dict:
    return {
        "split": "policy_train",
        "policy": "realized_dynamic",
        "status": "ok",
        "task_id": task,
        "solver_seed": 0,
        "summary": {"success": success, "stop_reason": stop_reason},
    }


class StallTrainingExtensionTests(unittest.TestCase):
    def test_extension_scales_decision_and_time_budgets_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "output"
            source.write_text(
                """{
  "max_decisions": 30,
  "metric_iteration_budget": 30,
  "wall_time_budget_seconds": 120.0,
  "episode_process_timeout_seconds": 180.0,
  "environment": {
    "max_repair_iterations": 30,
    "time_limit": 120.0
  }
}""",
                encoding="utf-8",
            )

            derived = _derived_config(source, output, 120)
            configuration = json.loads(derived.read_text(encoding="utf-8"))

            self.assertEqual(configuration["max_decisions"], 120)
            self.assertEqual(configuration["metric_iteration_budget"], 120)
            self.assertEqual(configuration["wall_time_budget_seconds"], 480.0)
            self.assertEqual(
                configuration["episode_process_timeout_seconds"], 720.0
            )
            self.assertEqual(
                configuration["environment"]["max_repair_iterations"], 120
            )
            self.assertEqual(configuration["environment"]["time_limit"], 480.0)

    def test_selects_only_unsuccessful_repair_limit_episodes(self) -> None:
        selected = unfinished_training_manifests(
            [
                _manifest("solved", success=True, stop_reason="success"),
                _manifest("unfinished", success=False, stop_reason="repair_limit"),
            ]
        )
        self.assertEqual([row["task_id"] for row in selected], ["unfinished"])

    def test_rejects_non_training_source(self) -> None:
        row = _manifest("task", success=False, stop_reason="repair_limit")
        row["split"] = "policy_validation"
        with self.assertRaisesRegex(ValueError, "policy_train only"):
            unfinished_training_manifests([row])

    def test_rejects_unexpected_unsuccessful_stop_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "repair limit"):
            unfinished_training_manifests(
                [_manifest("task", success=False, stop_reason="external_timeout")]
            )

    def test_finds_protocol_for_a_previous_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection = root / "collection"
            protocol = root / "protocol" / "policy_train_extension.json"
            collection.mkdir()
            protocol.parent.mkdir()
            protocol.write_text("{}", encoding="utf-8")
            self.assertEqual(_source_protocol_path(collection), protocol)

    def test_accepts_an_explicit_source_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "source.json"
            protocol.write_text("{}", encoding="utf-8")
            self.assertEqual(
                _source_protocol_path(root / "collection", protocol),
                protocol.resolve(),
            )

    def test_rejects_source_protocol_that_does_not_match_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_source_protocol(
                {"solver_seeds": [1, 2], "split": "policy_train"},
                {"solver_seeds": [0], "split": "policy_train"},
            )

    def test_same_repair_fingerprint_is_not_a_new_independent_state(self) -> None:
        trigger = {
            "decision_index": 6,
            "before_fingerprint": "full",
            "before_repair_fingerprint": "repair",
            "selected_candidate_id": "candidate",
            "candidate_pool": [{"candidate_id": "candidate"}],
            "prefix_actions": [],
        }
        history = [
            {
                "decision_index": 6,
                "attempt_key": "attempt",
                "actual_neighborhood_key": "neighborhood",
                "selected_candidate_id": "candidate",
                "repair_outcome": "hard_failure",
            }
        ]
        row = _sequence_row(
            {
                "qualification_class": "confirmed_long_stall",
                "resolution": "no_change_in_complete_future_window",
                "trigger": trigger,
                "history": history,
                "threshold": 3,
                "run_length": 7,
                "observed_after_trigger": 4,
                "recovery_delay_decisions": None,
                "distinct_attempt_count": 7,
                "distinct_neighborhood_count": 5,
            },
            {
                "map_id": "map",
                "task_id": "task",
                "solver_seed": 0,
                "episode_id": "episode",
            },
            source_transition_count=6,
            source_sequence_classes={"repair": "unresolved_stall"},
            continued_root=Path("collection"),
        )
        self.assertTrue(row["existing_sequence_in_source"])
        self.assertFalse(row["new_independent_state"])
        self.assertTrue(row["classification_changed_from_source"])


if __name__ == "__main__":
    unittest.main()
