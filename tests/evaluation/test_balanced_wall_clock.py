from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.balanced_wall_clock import (
    analyze_scheduled,
    audit_balanced_cohort_difficulty,
    build_replacement_dataset,
    collect_scheduled,
    conflict_stratum,
    initial_pp_load_stratum,
    materialize_compute_load_candidate_pool,
    select_balanced_cohort,
    select_compute_load_balanced_cohort,
)
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import summarize_initial_state_complexity


class BalancedWallClockTests(unittest.TestCase):
    def test_compute_load_pool_materialization_excludes_registered_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            excluded_sha = None
            for source_index in range(2):
                source = root / f"source-{source_index}"
                split = source / "balanced_wall_clock"
                (split / "maps").mkdir(parents=True)
                source_config = root / "configs" / f"source-{source_index}.json"
                source_config.parent.mkdir(parents=True, exist_ok=True)
                source_config.write_text("{}", encoding="utf-8")
                fingerprint = hashlib.sha256(b"{}").hexdigest()
                (source / "dataset_summary.json").write_text(
                    json.dumps({"configuration_fingerprint": fingerprint}),
                    encoding="utf-8",
                )
                rows = []
                map_ids = (
                    ("keep-a", "drop-duplicate")
                    if source_index == 0
                    else ("keep-b",)
                )
                for map_id in map_ids:
                    relative = Path("maps") / f"{map_id}.map"
                    payload = f"map:{map_id}".encode()
                    (split / relative).write_bytes(payload)
                    if map_id == "drop-duplicate":
                        excluded_sha = hashlib.sha256(payload).hexdigest()
                    rows.append(
                        {
                            "split": "balanced_wall_clock",
                            "map_id": map_id,
                            "task_id": f"task-{map_id}",
                            "layout_mode": "regular_beltway",
                            "map_file": relative.as_posix(),
                        }
                    )
                (split / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                sources.append(
                    {
                        "config": source_config.relative_to(root).as_posix(),
                        "dataset": source.relative_to(root).as_posix(),
                    }
                )
            registry = {
                "schema_version": 1,
                "generated_sources": sources,
                "excluded_map_ids": [
                    {
                        "map_id": "drop-duplicate",
                        "map_sha256": excluded_sha,
                        "reason": "test",
                    }
                ],
                "expected_usable": {
                    "map_count": 2,
                    "instance_count": 2,
                    "layout_map_counts": {"regular_beltway": 2},
                },
            }
            registry_path = root / "configs" / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")

            summary = materialize_compute_load_candidate_pool(
                registry_path, root / "output"
            )

            self.assertEqual(
                summary["splits"]["balanced_wall_clock"]["map_count"], 2
            )
            rows = [
                json.loads(line)
                for line in (
                    root / "output" / "balanced_wall_clock" / "manifest.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual({row["map_id"] for row in rows}, {"keep-a", "keep-b"})
            self.assertTrue(all(row["source_group"] == "generated" for row in rows))
            self.assertFalse(
                (
                    root
                    / "output"
                    / "balanced_wall_clock"
                    / "maps"
                    / "drop-duplicate.map"
                ).exists()
            )

    def test_replacement_dataset_keeps_only_registered_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {
                "original": ("keep-original", "drop-original", "generated"),
                "movingai_candidates": ("keep-movingai", None, "movingai"),
                "generated_candidates": ("keep-generated", None, "generated"),
                "additional_generated_candidates": (
                    "keep-additional",
                    None,
                    "generated",
                ),
            }
            for source_name, (selected, removed, source_group) in sources.items():
                split = root / source_name / "balanced_wall_clock"
                split.mkdir(parents=True)
                rows = []
                for map_id in (selected, removed):
                    if map_id is None:
                        continue
                    relative = Path("maps") / f"{map_id}.map"
                    (split / relative).parent.mkdir(parents=True, exist_ok=True)
                    (split / relative).write_text(f"map:{map_id}", encoding="utf-8")
                    rows.append(
                        {
                            "map_id": map_id,
                            "task_id": f"task-{map_id}",
                            "source_group": source_group,
                            "map_file": relative.as_posix(),
                        }
                    )
                (split / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            selection = {
                "schema_version": 1,
                "dataset_revision": "test-replacement-v1",
                "expected_map_count": 4,
                "expected_instance_count": 4,
                "source_counts": {"generated": 3, "movingai": 1},
                "original_map_ids": ["keep-original"],
                "movingai_candidates_map_ids": ["keep-movingai"],
                "generated_candidates_map_ids": ["keep-generated"],
                "additional_generated_candidates_map_ids": ["keep-additional"],
                "removed_map_ids": ["drop-original"],
            }
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")

            summary = build_replacement_dataset(
                original=root / "original",
                movingai_candidates=root / "movingai_candidates",
                generated_candidates=root / "generated_candidates",
                additional_generated_candidates=root / "additional_generated_candidates",
                selection_config=selection_path,
                output=root / "output",
            )

            self.assertEqual(summary["splits"]["balanced_wall_clock"]["map_count"], 4)
            self.assertEqual(
                summary["splits"]["balanced_wall_clock"]["source_counts"],
                {"generated": 3, "movingai": 1},
            )
            output_rows = [
                json.loads(line)
                for line in (
                    root / "output" / "balanced_wall_clock" / "manifest.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertNotIn("drop-original", {row["map_id"] for row in output_rows})
            self.assertFalse(
                (root / "output" / "balanced_wall_clock" / "maps" / "drop-original.map").exists()
            )
            with self.assertRaisesRegex(ValueError, "output must differ"):
                build_replacement_dataset(
                    original=root / "original",
                    movingai_candidates=root / "movingai_candidates",
                    generated_candidates=root / "generated_candidates",
                    additional_generated_candidates=root / "additional_generated_candidates",
                    selection_config=selection_path,
                    output=root / "original",
                )

    def test_conflict_strata_are_fixed_and_exclude_zero_and_extreme(self) -> None:
        self.assertIsNone(conflict_stratum(0))
        self.assertEqual(conflict_stratum(1), "low")
        self.assertEqual(conflict_stratum(10), "low")
        self.assertEqual(conflict_stratum(11), "medium")
        self.assertEqual(conflict_stratum(100), "medium")
        self.assertEqual(conflict_stratum(101), "high")
        self.assertEqual(conflict_stratum(500), "high")
        self.assertIsNone(conflict_stratum(501))
        self.assertEqual(initial_pp_load_stratum(0), "low")
        self.assertEqual(initial_pp_load_stratum(100_000), "low")
        self.assertEqual(initial_pp_load_stratum(100_001), "medium")
        self.assertEqual(initial_pp_load_stratum(1_000_000), "medium")
        self.assertEqual(initial_pp_load_stratum(1_000_001), "high")
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            initial_pp_load_stratum(-1)

    def test_initial_complexity_distinguishes_pairs_events_and_pp_load(self) -> None:
        state = {
            "rows": 2,
            "cols": 3,
            "obstacles": [0] * 6,
            "agents": [
                {"id": 10, "path": [0, 0, 1]},
                {"id": 20, "path": [0, 0, 2]},
            ],
            "conflict_edges": [[10, 20]],
            "num_of_colliding_pairs": 1,
            "low_level": {"generated": 250_000, "expanded": 100_000, "reopened": 3, "runs": 2},
        }
        summary = summarize_initial_state_complexity(state)
        self.assertEqual(summary["conflict_pair_count"], 1)
        self.assertEqual(summary["conflict_event_count"], 2)
        self.assertEqual(summary["active_conflict_agent_ratio"], 1.0)
        self.assertEqual(summary["largest_conflict_component_size"], 2)
        self.assertEqual(summary["total_path_cost"], 4)
        self.assertEqual(summary["initial_low_level_generated"], 250_000)

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

            results[0]["initial_conflicts"] = 0
            # Remove one high-stratum result, leaving only eleven eligible jobs.
            high_index = next(
                index for index, row in enumerate(results) if row["initial_conflicts"] == 200
            )
            results[high_index]["initial_conflicts"] = 0
            (qualification / "qualification_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
            )
            failed_root = root / "failed"
            failed = select_balanced_cohort(root / "dataset", qualification, failed_root)
            self.assertFalse(failed["passed"])
            self.assertEqual(failed["decision"], "data_gate_failed")
            self.assertFalse((failed_root / "execution_schedule.json").exists())
            with self.assertRaisesRegex(ValueError, "did not pass"):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "unused.json",
                    qualification=qualification,
                    schedule_root=failed_root,
                    output=root / "unused-output",
                    original_bundle=root / "unused-v1",
                    mixed_bundle=root / "unused-mixed",
                    resume=False,
                )

    def test_compute_load_selector_requires_all_source_cell_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset" / "balanced_wall_clock"
            qualification = root / "qualification"
            dataset.mkdir(parents=True)
            qualification.mkdir(parents=True)
            tasks = []
            results = []
            conflict_values = {"low": 5, "medium": 50, "high": 200}
            load_values = {"low": 50_000, "medium": 500_000, "high": 2_000_000}
            for conflict_level, conflicts in conflict_values.items():
                for load_level, generated in load_values.items():
                    for source in ("generated", "movingai"):
                        for copy in range(2):
                            task_id = (
                                f"{conflict_level}-{load_level}-{source}-{copy}"
                            )
                            map_id = f"map-{task_id}"
                            tasks.append(
                                {
                                    "split": "balanced_wall_clock",
                                    "task_id": task_id,
                                    "map_id": map_id,
                                    "layout_mode": source,
                                    "source_group": source,
                                    "agent_count": 200,
                                }
                            )
                            results.append(
                                {
                                    "status": "ok",
                                    "initial_complete": True,
                                    "task_id": task_id,
                                    "map_id": map_id,
                                    "layout_mode": source,
                                    "agent_count": 200,
                                    "solver_seed": 1,
                                    "initial_conflicts": conflicts,
                                    "state_fingerprint": task_id,
                                    "initial_complexity": {
                                        "conflict_pair_count": conflicts,
                                        "initial_low_level_generated": generated,
                                        "initial_low_level_expanded": generated // 2,
                                        "total_path_cost": generated // 10,
                                        "conflict_event_count": conflicts * 2,
                                        "active_conflict_agent_ratio": 0.5,
                                        "largest_conflict_component_ratio": 0.25,
                                    },
                                }
                            )
            (dataset / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
            )
            manifest = qualification / "qualification_manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
            )

            report = select_compute_load_balanced_cohort(
                root / "dataset", qualification, root / "selected"
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_count"], 36)
            self.assertEqual(len(report["cell_reports"]), 9)
            self.assertTrue(all(row["passed"] for row in report["cell_reports"].values()))
            schedule = json.loads(
                (root / "selected" / "execution_schedule.json").read_text()
            )["entries"]
            self.assertEqual(
                {source: sum(row["source_group"] == source for row in schedule) for source in ("generated", "movingai")},
                {"generated": 18, "movingai": 18},
            )
            self.assertEqual(
                {load: sum(row["initial_pp_load_stratum"] == load for row in schedule) for load in ("low", "medium", "high")},
                {"low": 12, "medium": 12, "high": 12},
            )

            del results[0]["initial_complexity"]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "lacks initial_complexity"):
                select_compute_load_balanced_cohort(
                    root / "dataset", qualification, root / "invalid"
                )

    def test_analysis_uses_paired_map_and_stratum_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_rows = []
            for index in range(12):
                stratum = ("low", "medium", "high")[index % 3]
                initial_conflicts = {"low": 1, "medium": 11, "high": 101}[stratum]
                agent_count = initial_conflicts * 2
                source_group = "movingai" if stratum == "high" else "generated"
                schedule_rows.append(
                    {
                        "task_id": f"task-{index}",
                        "solver_seed": 1,
                        "map_id": f"map-{index}",
                        "source_group": source_group,
                        "layout_mode": source_group,
                        "agent_count": agent_count,
                        "agent_band": "small" if agent_count <= 200 else "medium",
                        "initial_conflicts": initial_conflicts,
                        "conflict_stratum": stratum,
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
                        initial_conflicts = int(item["initial_conflicts"])
                        agents = [
                            {"id": 2 * pair + offset, "path": [pair]}
                            for pair in range(initial_conflicts)
                            for offset in (0, 1)
                        ]
                        generated = (50_000, 500_000, 2_000_000)[
                            int(item["task_id"].split("-")[-1]) % 3
                        ]
                        state = {
                            "initialized": True,
                            "initial_solution_complete": True,
                            "feasible": False,
                            "done": False,
                            "iteration": 0,
                            "rows": 1,
                            "cols": initial_conflicts,
                            "sum_of_costs": 0,
                            "num_of_colliding_pairs": initial_conflicts,
                            "low_level": {
                                "generated": generated,
                                "expanded": generated // 2,
                                "reopened": 0,
                                "runs": len(agents),
                            },
                            "obstacles": [0] * initial_conflicts,
                            "conflict_edges": [
                                [2 * pair, 2 * pair + 1]
                                for pair in range(initial_conflicts)
                            ],
                            "agents": agents,
                        }
                        initial_state_ref = None
                        if controller == "official_adaptive":
                            initial_state_ref, _blob = write_state_blob(path, state)
                        ttf = 120.0 if controller == "official_adaptive" else 100.0
                        if controller == "mixed-full-v2":
                            ttf = 90.0
                        rows.append(
                            {
                                **{key: item[key] for key in ("task_id", "solver_seed", "map_id")},
                                "agent_count": int(item["agent_count"]),
                                "initial_state_ref": (
                                    initial_state_ref.as_posix()
                                    if isinstance(initial_state_ref, Path)
                                    else initial_state_ref
                                ),
                                "status": "ok",
                                "summary": {
                                    "success": True,
                                    "initial_fingerprint": state_fingerprint(state),
                                    "initial_conflicts": initial_conflicts,
                                    "capped_wall_time_to_feasible": ttf,
                                    "wall_time_to_feasible": ttf,
                                    "episode_observed_wall_seconds": ttf,
                                    "fixed_budget_conflict_auc": ttf,
                                    "normalized_fixed_budget_conflict_auc": ttf / 1000.0,
                                    "repair_iterations": 2,
                                    "repair_wall_seconds": ttf / 2.0,
                                    "reset_timings": {"initial_solution_seconds": ttf / 4.0},
                                    "final_low_level": {"generated": generated, "expanded": generated // 2},
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
            audit = audit_balanced_cohort_difficulty(
                collection, cohort, root / "difficulty-report"
            )
            self.assertEqual(
                audit["decision"],
                "conflict_count_balanced_pilot_with_compute_load_confounding",
            )
            self.assertEqual(
                audit["stratification"]["initial_pp_load_counts"],
                {"high": 4, "low": 4, "medium": 4},
            )
            self.assertFalse(
                audit["methodology_gates"][
                    "generated_and_movingai_overlap_in_every_load_tier"
                ]
            )
            self.assertTrue((root / "difficulty-report" / "difficulty_episodes.csv").is_file())


if __name__ == "__main__":
    unittest.main()
