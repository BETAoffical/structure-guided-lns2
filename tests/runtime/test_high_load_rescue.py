from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from experiments.high_load_rescue import (
    collect_high_load_rescue_data,
    is_no_progress_decision,
    paired_rescue_seed,
    size12_pilot_gate,
)
from experiments.repair_aware import repair_aware_order
from scripts.run_high_load_rescue_pipeline import _pilot_tasks, run_pipeline


class HighLoadFailureSelectionTests(unittest.TestCase):
    def test_pipeline_requires_resume_for_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pass --resume"):
                run_pipeline(
                    mode="pilot",
                    output=output,
                    resume=False,
                    workers=1,
                )

    def test_pilot_uses_all_dense_task_variants(self) -> None:
        rows = [
            {
                "layout_mode": "regular_beltway",
                "map_id": "map-0",
                "task_id": f"task-{index}",
                "agent_count": 400 if index % 2 == 0 else 600,
            }
            for index in range(4)
        ]
        with mock.patch(
            "scripts.run_high_load_rescue_pipeline._read_jsonl",
            return_value=rows,
        ):
            self.assertEqual(
                _pilot_tasks(Path("unused"), "policy_train"),
                [f"task-{index}" for index in range(4)],
            )

    def test_collection_rejects_an_empty_size_set_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted unique positive"):
            collect_high_load_rescue_data(
                source_roots={"policy_train": ".", "policy_validation": "."},
                output="unused",
                controller_bundle="unused",
                maximum_states={"policy_train": 1, "policy_validation": 1},
                neighborhood_sizes=(),
            )

    def test_hard_failure_and_structural_noop_are_no_progress(self) -> None:
        self.assertTrue(
            is_no_progress_decision(
                {"actual_metrics": {"replan_success": False}, "repair_state_changed": False}
            )
        )
        self.assertTrue(
            is_no_progress_decision(
                {"actual_metrics": {"replan_success": True}, "repair_state_changed": False}
            )
        )
        self.assertFalse(
            is_no_progress_decision(
                {"actual_metrics": {"replan_success": True}, "repair_state_changed": True}
            )
        )

    def test_paired_seed_depends_on_state_and_trial_not_arm(self) -> None:
        self.assertEqual(paired_rescue_seed("state", 2), paired_rescue_seed("state", 2))
        self.assertNotEqual(paired_rescue_seed("state", 1), paired_rescue_seed("state", 2))


class Size12PilotGateTests(unittest.TestCase):
    def test_gate_requires_sixty_states_and_three_size12_winners(self) -> None:
        features = []
        trials = []
        for state_index in range(60):
            state_id = f"s{state_index}"
            for size in (4, 8, 12, 16):
                candidate_id = f"{state_id}-{size}"
                features.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "actual_size": size,
                        "route": "model",
                    }
                )
                reduction = 4 if size == 12 and state_index < 3 else 1
                seconds = 1.0 if size == 12 else 2.0
                trials.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "outcome": {
                            "conflict_reduction": reduction,
                            "repair_seconds": seconds,
                        },
                    }
                )
        report = size12_pilot_gate(features, trials)
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["size12_selected_state_count"], 3)


class WallClockOrderingTests(unittest.TestCase):
    def test_efficiency_is_primary_runtime_order(self) -> None:
        candidates = [{"candidate_id": "slow"}, {"candidate_id": "fast"}]
        predictions = {
            "progress_probability": [0.9, 0.8],
            "conflict_reduction": [10.0, 8.0],
            "repair_seconds": [10.0, 2.0],
            "hard_failure_probability": [0.0, 0.0],
            "efficiency": [0.9, 3.2],
        }
        self.assertEqual(
            repair_aware_order(candidates, predictions, [2.0, 1.0]), [1, 0]
        )


if __name__ == "__main__":
    unittest.main()
