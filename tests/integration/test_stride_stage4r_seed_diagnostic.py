from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.stride_stage4r_seed_diagnostic import (
    CONTROLLERS,
    REGISTERED_SOLVER_SEEDS,
    TTF_CLOCK_SCHEMA,
    analyze_seed_diagnostic,
    seed_diagnostic_schedule,
    validate_seed_diagnostic_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_stage4r_seed_diagnostic.json"


def _summary(
    task_id: str,
    solver_seed: int,
    *,
    capped_ttf: float,
    repair_iterations: int,
) -> dict[str, object]:
    trajectory = [5]
    for iteration in range(repair_iterations):
        trajectory.append(max(0, 5 - iteration - 1))
    trajectory[-1] = 0
    return {
        "success": True,
        "initial_fingerprint": f"initial-{task_id}-{solver_seed}",
        "initial_conflicts": 5,
        "wall_time_to_feasible": capped_ttf,
        "capped_wall_time_to_feasible": capped_ttf,
        "repair_iterations": repair_iterations,
        "conflict_trajectory": trajectory,
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
            "pruner_fallback_count": 0,
        },
        "invalid_action_count": 0,
        "fingerprint_mismatch_count": 0,
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
    }


class StrideStage4RSeedDiagnosticTest(unittest.TestCase):
    def test_schedule_is_complete_and_balances_first_position(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        schedule = seed_diagnostic_schedule(config)
        self.assertEqual(len(schedule), 48)
        key_counts = Counter(
            (row["task_id"], row["solver_seed"]) for row in schedule
        )
        self.assertEqual(set(key_counts.values()), {3})
        first_counts = Counter(
            row["controller"]
            for row in schedule
            if row["within_key_position"] == 0
        )
        self.assertEqual(sorted(first_counts.values()), [5, 5, 6])

    def test_registration_rejects_changed_scientific_contract(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_seed_diagnostic_config(config)
        for mutation in ("formal", "seed", "budget", "task"):
            changed = copy.deepcopy(config)
            protocol = changed["stride_stage4r_seed_diagnostic"]
            if mutation == "formal":
                protocol["formal_speed_claim"] = True
            elif mutation == "seed":
                protocol["solver_seeds"] = [1, 2, 3]
            elif mutation == "budget":
                protocol["wall_time_budget_seconds"] = 300.0
            else:
                protocol["registered_task_ids"] = protocol["registered_task_ids"][:3]
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    validate_seed_diagnostic_config(changed)

    def test_analysis_identifies_cross_seed_tail_weakness(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        protocol = config["stride_stage4r_seed_diagnostic"]
        problem_tasks = set(protocol["problem_task_ids"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_schedule.jsonl").write_text("{}\n", encoding="utf-8")
            for controller in CONTROLLERS:
                target = root / "controllers" / controller
                target.mkdir(parents=True)
                rows = []
                for task_id in protocol["registered_task_ids"]:
                    for solver_seed in REGISTERED_SOLVER_SEEDS:
                        capped_ttf = 10.0
                        iterations = 2
                        if controller == "stride-control-v1":
                            capped_ttf = 11.0
                            iterations = 3
                        elif controller == "stride-quality-v1":
                            if task_id in problem_tasks and solver_seed <= 3:
                                capped_ttf = 12.0
                                iterations = 3
                            elif task_id in problem_tasks:
                                capped_ttf = 9.0
                            else:
                                capped_ttf = 8.0
                        rows.append(
                            {
                                "status": "ok",
                                "task_id": task_id,
                                "solver_seed": solver_seed,
                                "summary": _summary(
                                    task_id,
                                    solver_seed,
                                    capped_ttf=capped_ttf,
                                    repair_iterations=iterations,
                                ),
                            }
                        )
                (target / "realized_dynamic_manifest.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )

            report = analyze_seed_diagnostic(CONFIG, root)

        self.assertTrue(report["passed"])
        self.assertIsNone(report["formal_primary_winner"])
        self.assertTrue(report["diagnosis"]["intrinsic_tail_weakness"])
        self.assertFalse(report["diagnosis"]["seed_sensitive"])
        self.assertEqual(
            report["next_decision"],
            "revise_one_step_label_for_residual_state_hardness",
        )
        for task_id in problem_tasks:
            evidence = report["diagnosis"]["problem_task_evidence"][task_id]
            self.assertEqual(evidence["slower_or_more_rounds_count"], 3)

    def test_analysis_preserves_errors_without_a_runtime_claim(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        protocol = config["stride_stage4r_seed_diagnostic"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_schedule.jsonl").write_text("{}\n", encoding="utf-8")
            for controller in CONTROLLERS:
                target = root / "controllers" / controller
                target.mkdir(parents=True)
                rows = [
                    {
                        "status": "error",
                        "task_id": task_id,
                        "solver_seed": solver_seed,
                        "summary": None,
                        "error_kind": "synthetic_error",
                    }
                    for task_id in protocol["registered_task_ids"]
                    for solver_seed in REGISTERED_SOLVER_SEEDS
                ]
                (target / "realized_dynamic_manifest.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )

            report = analyze_seed_diagnostic(CONFIG, root)

        self.assertFalse(report["passed"])
        self.assertIsNone(report["formal_primary_winner"])
        self.assertIsNone(report["observed_mean_capped_ttf_best_controller"])
        self.assertEqual(
            report["next_decision"],
            "repair_execution_or_analysis_before_runtime_conclusion",
        )


if __name__ == "__main__":
    unittest.main()
