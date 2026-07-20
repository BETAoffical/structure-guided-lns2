from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.engineering.legacy_tradeoff.lns2_tradeoff import (
    CONTROLLER_ORDER,
    generate_timeout_sensitivity_artifacts,
    timeout_job_keys,
)


class TimeoutSensitivityTests(unittest.TestCase):
    @staticmethod
    def _collections(
        root: Path,
        *,
        budget: float,
        timeout_controller: str | None,
        new_success_controller: str | None = None,
    ) -> dict[str, Path]:
        roots: dict[str, Path] = {}
        modes = {
            "official_adaptive": "v1-full",
            "v1-full": "v1-full",
            "v2-full": "v2-full",
            "v2-balanced": "v2-balanced",
        }
        auc = {
            "official_adaptive": 80.0,
            "v1-full": 90.0,
            "v2-full": 70.0,
            "v2-balanced": 75.0,
        }
        for controller in CONTROLLER_ORDER:
            collection = root / controller
            collection.mkdir(parents=True)
            roots[controller] = collection
            run = {
                "controller": modes[controller],
                "dataset_fingerprint": "dataset-a",
                "run_fingerprint": f"{budget:g}-{controller}",
                "trace_format": "delta-gzip-v2",
                "feature_backend": "native",
                "controller_bundle": (
                    {"main_ranker_semantic_fingerprint": "model-a"}
                    if controller in {"v2-full", "v2-balanced"}
                    else None
                ),
                "balanced_config": (
                    {
                        "configuration_fingerprint": "balanced-a",
                        "source": {"selection_unit": "complete_episode"},
                    }
                    if controller == "v2-balanced"
                    else None
                ),
                "configuration": {
                    "severity_thresholds": {"low_max": 0.001, "medium_max": 0.01},
                    "wall_time_budget_seconds": budget,
                    "episode_process_timeout_seconds": budget + 60.0,
                    "environment": {"time_limit": budget},
                    "max_decisions": 100,
                    "metric_iteration_budget": 100,
                },
            }
            (collection / "run_config.json").write_text(
                json.dumps(run), encoding="utf-8"
            )
            trace = collection / "episode.jsonl.gz"
            with gzip.open(trace, "wt", encoding="utf-8") as stream:
                stream.write("{}\n")
            policy = (
                "official_adaptive"
                if controller == "official_adaptive"
                else "realized_dynamic"
            )
            repairs = 50 if budget == 300.0 else 100
            success = controller == new_success_controller
            balanced = controller == "v2-balanced"
            full = controller in {"v1-full", "v2-full"}
            summary = {
                "repairable": True,
                "success": success,
                "external_timeout": controller == timeout_controller,
                "initial_fingerprint": "initial-a",
                "initial_conflicts": 10,
                "final_conflicts": 0 if success else int(auc[controller] / 10),
                "repair_iterations": repairs,
                "fixed_budget_conflict_auc": auc[controller],
                "capped_wall_time_to_feasible": 20.0 if success else budget,
                "repair_wall_seconds": 10.0,
                "final_sum_of_costs": 100,
                "final_low_level": {
                    "expanded": 100,
                    "generated": 200,
                    "reopened": 1,
                },
                "invalid_action_count": 0,
                "fingerprint_mismatch_count": 0,
                "model_decision_count": repairs if balanced else 0,
                "official_decision_count": 0,
                "model_route_fraction": 1.0 if balanced else 0.0,
                "route_switch_count": 0,
                "controller_totals": {
                    "learned_decisions": repairs if full else 0,
                    "controller_seconds_before_repair": 1.0,
                },
            }
            manifest = {
                "episode_id": f"episode-{controller}",
                "policy": policy,
                "layout_mode": "maze",
                "map_id": "map-a",
                "task_id": "task-a",
                "agent_count": 100,
                "solver_seed": 1,
                "status": "ok",
                "summary": summary,
                "trace_file": trace.name,
                "trace_bytes": trace.stat().st_size,
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
            (collection / f"{policy}_manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
        return roots

    def test_timeout_cohort_runs_all_four_and_does_not_change_primary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = self._collections(
                root / "primary", budget=300.0, timeout_controller="v1-full"
            )
            sensitivity = self._collections(
                root / "sensitivity",
                budget=600.0,
                timeout_controller=None,
                new_success_controller="v2-full",
            )
            self.assertEqual(timeout_job_keys(primary), {("task-a", 1)})

            report = generate_timeout_sensitivity_artifacts(
                primary,
                sensitivity,
                root / "report",
                sensitivity_budget_seconds=600.0,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_task_seed_count"], 1)
            self.assertEqual(report["completed_episode_count"], 4)
            self.assertEqual(report["new_success_count"], 1)
            self.assertTrue(report["main_promotion_metrics_unchanged"])
            self.assertEqual(
                report["conclusion"], "primary_conclusion_is_budget_sensitive"
            )
            self.assertTrue((root / "report" / "timeout_sensitivity_episodes.csv").is_file())
            self.assertTrue((root / "report" / "timeout_sensitivity_report.md").is_file())

    def test_no_primary_timeout_needs_no_extra_collections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = self._collections(
                root / "primary", budget=300.0, timeout_controller=None
            )
            report = generate_timeout_sensitivity_artifacts(
                primary,
                None,
                root / "report",
                sensitivity_budget_seconds=600.0,
            )
            self.assertEqual(report["selected_task_seed_count"], 0)
            self.assertEqual(report["completed_episode_count"], 0)
            self.assertEqual(report["conclusion"], "no_timeout_sensitivity_needed")


if __name__ == "__main__":
    unittest.main()
