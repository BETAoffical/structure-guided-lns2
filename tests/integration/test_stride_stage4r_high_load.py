from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.stride_stage4r_high_load import (
    CONTROLLERS,
    analyze_high_load_diagnostic,
    high_load_schedule,
)
from experiments.stride_stage4r_quick import TTF_CLOCK_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_stage4r_high_load_diagnostic.json"


class StrideStage4RHighLoadTest(unittest.TestCase):
    def test_schedule_is_complete_and_rotates_first_controller(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        schedule = high_load_schedule(config)
        self.assertEqual(len(schedule), 32)
        key_counts = Counter(
            (row["cohort_id"], row["task_id"], row["solver_seed"])
            for row in schedule
        )
        self.assertEqual(set(key_counts.values()), {2})
        first_counts = Counter(
            row["controller"]
            for row in schedule
            if row["within_key_position"] == 0
        )
        self.assertEqual(dict(first_counts), {"v2-full": 8, "stride-quality-v1": 8})

    def test_analysis_reports_paired_ttf_and_feature_shift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = high_load_schedule(config)
            (root / "execution_schedule.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in schedule),
                encoding="utf-8",
            )
            for cohort in config["cohorts"]:
                for controller in CONTROLLERS:
                    target = root / "cohorts" / cohort["id"] / "controllers" / controller
                    target.mkdir(parents=True)
                    rows = []
                    for task_id in cohort["tasks"]:
                        for seed in config["solver_seeds"]:
                            ttf = 10.0 if controller == "v2-full" else 8.0
                            rows.append(
                                {
                                    "status": "ok",
                                    "task_id": task_id,
                                    "solver_seed": seed,
                                    "summary": {
                                        "success": True,
                                        "initial_fingerprint": f"{task_id}-{seed}",
                                        "initial_conflicts": 10,
                                        "wall_time_to_feasible": ttf,
                                        "capped_wall_time_to_feasible": ttf,
                                        "repair_iterations": 5,
                                        "conflict_trajectory": [10, 10, 4, 0],
                                        "normalized_wall_clock_conflict_auc": 0.1,
                                        "controller_totals": {
                                            "model_decision_count": 5,
                                            "pp_replan_seconds": 1.0,
                                            "controller_seconds_before_repair": 0.5,
                                            "pruner_fallback_count": 0,
                                            "pruner_ood_fallback_count": 0,
                                            "selected_feature_diagnostic_count": 5,
                                            "selected_feature_outside_fraction_sum": (
                                                1.0 if controller == "stride-quality-v1" else 0.0
                                            ),
                                        },
                                        "invalid_action_count": 0,
                                        "fingerprint_mismatch_count": 0,
                                        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
                                    },
                                }
                            )
                    (target / "realized_dynamic_manifest.jsonl").write_text(
                        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8",
                    )
            report = analyze_high_load_diagnostic(CONFIG, root)

        self.assertTrue(report["passed"])
        self.assertAlmostEqual(
            report["comparison_vs_v2_full"]["mean_capped_ttf_relative_improvement"],
            0.2,
        )
        self.assertTrue(
            report["diagnosis"]["selected_feature_distribution_shift"]
        )
        self.assertEqual(
            report["controller_summaries"]["stride-quality-v1"][
                "mean_selected_feature_outside_fraction"
            ],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
