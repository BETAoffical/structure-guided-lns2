from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.balanced_wall_clock import (
    analyze_scheduled,
    conflict_stratum,
    select_balanced_cohort,
)


class BalancedWallClockTests(unittest.TestCase):
    def test_conflict_strata_are_fixed_and_exclude_zero_and_extreme(self) -> None:
        self.assertIsNone(conflict_stratum(0))
        self.assertEqual(conflict_stratum(1), "low")
        self.assertEqual(conflict_stratum(10), "low")
        self.assertEqual(conflict_stratum(11), "medium")
        self.assertEqual(conflict_stratum(100), "medium")
        self.assertEqual(conflict_stratum(101), "high")
        self.assertEqual(conflict_stratum(500), "high")
        self.assertIsNone(conflict_stratum(501))

    def test_selector_is_blind_and_balances_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset" / "balanced_wall_clock"
            qualification = root / "qualification"
            rows = []
            results = []
            bounds = {"low": 5, "medium": 50, "high": 200}
            for stratum, conflicts in bounds.items():
                for source in ("generated", "movingai"):
                    for index in range(6):
                        task_id = f"{stratum}-{source}-{index}"
                        map_id = f"{stratum}-{source}-map-{index // 2}"
                        rows.append(
                            {
                                "split": "balanced_wall_clock",
                                "task_id": task_id,
                                "map_id": map_id,
                                "layout_mode": source,
                                "source_group": source,
                                "agent_count": 100 if index % 2 == 0 else 400,
                            }
                        )
                        results.append(
                            {
                                "status": "ok",
                                "initial_complete": True,
                                "task_id": task_id,
                                "map_id": map_id,
                                "layout_mode": source,
                                "agent_count": 100 if index % 2 == 0 else 400,
                                "solver_seed": 1,
                                "initial_conflicts": conflicts,
                                "state_fingerprint": task_id,
                            }
                        )
            # Pad qualification to its preregistered 216 reset size with excluded zeros.
            for index in range(180):
                task = rows[index % len(rows)]
                results.append(
                    {
                        "status": "ok",
                        "initial_complete": True,
                        "task_id": task["task_id"],
                        "map_id": task["map_id"],
                        "layout_mode": task["layout_mode"],
                        "agent_count": task["agent_count"],
                        "solver_seed": 10 + index,
                        "initial_conflicts": 0,
                        "state_fingerprint": f"zero-{index}",
                    }
                )
            dataset.mkdir(parents=True)
            qualification.mkdir(parents=True)
            (dataset / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (qualification / "qualification_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
            )
            report = select_balanced_cohort(root / "dataset", qualification, root / "out")
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_count"], 36)
            schedule = json.loads((root / "out" / "execution_schedule.json").read_text())
            orders = [tuple(row["controller_order"]) for row in schedule["entries"]]
            self.assertEqual(len(set(orders)), 6)
            self.assertTrue(all(orders.count(order) == 6 for order in set(orders)))

    def test_analysis_uses_paired_map_and_stratum_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_rows = []
            for index in range(12):
                schedule_rows.append(
                    {
                        "task_id": f"task-{index}",
                        "solver_seed": 1,
                        "map_id": f"map-{index}",
                        "source_group": "movingai" if index % 2 else "generated",
                        "agent_band": "small" if index % 2 else "large",
                        "conflict_stratum": ("low", "medium", "high")[index % 3],
                        "schedule_group": index % 6,
                        "controller_order": [
                            "official_adaptive",
                            "v2-full",
                            "mixed-full-v2",
                        ],
                    }
                )
            cohort = root / "cohort"
            cohort.mkdir()
            (cohort / "execution_schedule.json").write_text(
                json.dumps({"entries": schedule_rows}), encoding="utf-8"
            )
            collection = root / "collection"
            for group in range(6):
                selected = [row for row in schedule_rows if row["schedule_group"] == group]
                for controller in ("official_adaptive", "v2-full", "mixed-full-v2"):
                    phase = (
                        "official_adaptive"
                        if controller == "official_adaptive"
                        else "realized_dynamic"
                    )
                    path = collection / f"order_{group}" / controller
                    path.mkdir(parents=True)
                    rows = []
                    for item in selected:
                        ttf = 120.0 if controller == "official_adaptive" else 100.0
                        if controller == "mixed-full-v2":
                            ttf = 90.0
                        rows.append(
                            {
                                **{key: item[key] for key in ("task_id", "solver_seed", "map_id")},
                                "status": "ok",
                                "summary": {
                                    "success": True,
                                    "initial_fingerprint": f"fp-{item['task_id']}",
                                    "initial_conflicts": 10,
                                    "capped_wall_time_to_feasible": ttf,
                                    "fixed_budget_conflict_auc": ttf,
                                    "normalized_fixed_budget_conflict_auc": ttf / 1000.0,
                                    "repair_iterations": 2,
                                    "final_low_level": {"generated": 10, "expanded": 5},
                                    "invalid_action_count": 0,
                                    "fingerprint_mismatch_count": 0,
                                    "controller_totals": {},
                                },
                            }
                        )
                    (path / f"{phase}_manifest.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                    )
            report = analyze_scheduled(collection, cohort, root / "report")
            self.assertEqual(report["decision"], "mixed_full_candidate")
            self.assertAlmostEqual(
                report["comparisons"]["mixed_vs_v2_ttf_improvement"], 0.1
            )
            self.assertTrue(all(report["promotion_gate"].values()))
            self.assertEqual(
                report["comparisons"]["mixed_vs_v2_map_bootstrap"]["map_count"], 12
            )


if __name__ == "__main__":
    unittest.main()
