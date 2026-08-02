from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.stride_stage4r_quick import (
    CONTROLLERS,
    TTF_CLOCK_SCHEMA,
    analyze_stage4r_quick,
    stage4r_quick_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_stage4r_quick.json"


class StrideStage4RQuickTest(unittest.TestCase):
    def test_schedule_is_complete_and_rotates_first_position(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        schedule = stage4r_quick_schedule(config)
        self.assertEqual(len(schedule), 33)
        key_counts = Counter(
            (row["task_id"], row["solver_seed"]) for row in schedule
        )
        self.assertEqual(set(key_counts.values()), {3})
        first_counts = Counter(
            row["controller"]
            for row in schedule
            if row["within_key_position"] == 0
        )
        self.assertEqual(sorted(first_counts.values()), [3, 4, 4])

    def test_analysis_uses_capped_ttf_with_success_constraint(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        task_ids = config["stride_stage4r_quick"]["registered_task_ids"]
        controller_ttf = {
            "v2-full": 10.0,
            "stride-control-v1": 9.0,
            "stride-quality-v1": 8.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_schedule.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            for controller in CONTROLLERS:
                target = root / "controllers" / controller
                target.mkdir(parents=True)
                rows = []
                for task_id in task_ids:
                    rows.append(
                        {
                            "status": "ok",
                            "task_id": task_id,
                            "solver_seed": 1,
                            "summary": {
                                "success": True,
                                "initial_fingerprint": f"initial-{task_id}",
                                "initial_conflicts": 5,
                                "wall_time_to_feasible": controller_ttf[controller],
                                "capped_wall_time_to_feasible": controller_ttf[controller],
                                "repair_iterations": 2,
                                "normalized_fixed_budget_conflict_auc": 0.1,
                                "normalized_wall_clock_conflict_auc": 0.1,
                                "repair_wall_seconds": 1.0,
                                "controller_totals": {
                                    "pp_replan_seconds": 0.8,
                                    "controller_seconds_before_repair": 0.2,
                                    "proposal_seconds": 0.05,
                                    "feature_seconds": 0.1,
                                    "inference_seconds": 0.01,
                                    "state_export_seconds": 0.01,
                                },
                                "invalid_action_count": 0,
                                "fingerprint_mismatch_count": 0,
                                "pruner_fallback_count": 0,
                                "ttf_clock_schema": TTF_CLOCK_SCHEMA,
                            },
                        }
                    )
                (target / "realized_dynamic_manifest.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
            report = analyze_stage4r_quick(CONFIG, root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["primary_winner"], "stride-quality-v1")
        self.assertEqual(
            report["next_decision"],
            "quality_candidate_warrants_larger_development_quick",
        )
        self.assertTrue(
            report["comparisons_vs_v2_full"]["stride-quality-v1"][
                "success_noninferior"
            ]
        )


if __name__ == "__main__":
    unittest.main()
