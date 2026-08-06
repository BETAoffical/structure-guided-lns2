from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_da2_recovery_qualification import (
    recovery_task_summaries,
    validate_da2_recovery_qualification_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_da2_recovery_primary_qualification_design.json"
)


class RobustActionDA2RecoveryQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_primary_qualification_is_valid(self) -> None:
        validate_da2_recovery_qualification_design(
            self.config, project_root=ROOT
        )
        self.assertTrue(
            self.config["candidate_ladder_errors_or_timeouts_allowed"]
        )
        self.assertTrue(
            self.config["selected_tasks_must_have_both_solver_seeds_ok"]
        )

    def test_candidate_repair_outcomes_remain_forbidden(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["forbidden_selection_inputs"].remove("candidate_runtime")
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_da2_recovery_qualification_design(
                changed, project_root=ROOT
            )

    def test_failed_candidate_does_not_remove_complete_paired_tasks(self) -> None:
        manifest = [
            {
                "task_id": f"map__derived_uniform_random__task_seed_0337__agents_{agents:04d}",
                "map_id": "map",
                "layout_mode": "dao_high_topology",
                "agent_count": agents,
            }
            for agents in (100, 200, 300)
        ]
        results = []
        for source in manifest:
            for seed in (1, 2):
                row = {
                    "task_id": source["task_id"],
                    "map_id": "map",
                    "agent_count": source["agent_count"],
                    "solver_seed": seed,
                }
                if source["agent_count"] == 300 and seed == 1:
                    row.update({"status": "error", "error": "empty path"})
                else:
                    row.update(
                        {
                            "status": "ok",
                            "initial_complete": True,
                            "initial_conflicts": source["agent_count"] // 10,
                            "state_fingerprint": f"state-{source['agent_count']}-{seed}",
                        }
                    )
                results.append(row)

        summaries, errors, forbidden, failed = recovery_task_summaries(
            manifest, results, [1, 2]
        )

        self.assertEqual(len(summaries), 2)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual(len(failed), 1)


if __name__ == "__main__":
    unittest.main()
