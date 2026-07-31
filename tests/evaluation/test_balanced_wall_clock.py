from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from experiments.balanced_wall_clock import (
    _four_neighbor_distances,
    _paired_success_ttf,
    _successful_ttf,
    analyze_scheduled,
    analyze_success_only_ttf,
    audit_balanced_cohort_difficulty,
    build_replacement_dataset,
    collect_scheduled,
    conflict_stratum,
    initial_pp_load_stratum,
    materialize_compute_load_candidate_pool,
    materialize_qualified_compute_load_pool,
    materialize_registered_compute_load_cohort,
    prepare_movingai_map_derived_dataset,
    prepare_movingai_dataset,
    select_balanced_cohort,
    select_compute_load_balanced_cohort,
    verify_compute_load_cohort_registration,
)
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import summarize_initial_state_complexity


class BalancedWallClockTests(unittest.TestCase):
    def test_formal_collection_uses_uncapped_fixed_metric_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root = root / "schedule"
            schedule_root.mkdir()
            (schedule_root / "cohort_report.json").write_text(
                json.dumps({"formal_collection_allowed": True}),
                encoding="utf-8",
            )
            controller_order = [
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            ]
            entries = [
                {
                    "schedule_group": group,
                    "controller_order": controller_order,
                    "task_id": f"task-{group}",
                    "solver_seed": group,
                }
                for group in range(6)
            ]
            (schedule_root / "execution_schedule.json").write_text(
                json.dumps({"entries": entries}),
                encoding="utf-8",
            )
            with patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                return_value={"dry_run": True},
            ) as run:
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=root / "qualification",
                    schedule_root=schedule_root,
                    output=root / "output",
                    original_bundle=root / "v2",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            self.assertEqual(run.call_count, 18)
            for call in run.call_args_list:
                self.assertEqual(
                    call.kwargs["stopping_rule"],
                    "wall-clock-fixed-metric",
                )

    def test_success_only_ttf_excludes_failures_from_pairs(self) -> None:
        def successful(value: float) -> dict[str, object]:
            return {
                "summary": {"success": True, "wall_time_to_feasible": value}
            }

        failed = {
            "summary": {
                "success": False,
                "wall_time_to_feasible": None,
                "capped_wall_time_to_feasible": 600.0,
            }
        }
        baseline = {
            ("shared", 1): successful(10.0),
            ("candidate-only", 1): failed,
        }
        candidate = {
            ("shared", 1): successful(8.0),
            ("candidate-only", 1): successful(1.0),
        }
        schedule = {
            ("shared", 1): {"map_id": "shared-map"},
            ("candidate-only", 1): {"map_id": "candidate-only-map"},
        }

        self.assertIsNone(_successful_ttf(failed))
        comparison = _paired_success_ttf(
            baseline,
            candidate,
            schedule,
            schedule,
            samples=100,
            seed=7,
        )
        self.assertEqual(comparison["common_success_count"], 1)
        self.assertAlmostEqual(comparison["mean_improvement"], 0.2)
        self.assertEqual(comparison["candidate_faster_count"], 1)

    def test_map_derived_astar_distances_match_four_neighbor_paths(self) -> None:
        passable = {
            (row, col)
            for row in range(5)
            for col in range(5)
            if (row, col) not in {(1, 0), (1, 1), (1, 2), (1, 3)}
        }
        self.assertEqual(
            _four_neighbor_distances(
                passable,
                [(0, 0), (4, 0)],
                [(2, 0), (4, 4)],
            ),
            [10, 4],
        )

    def test_map_derived_movingai_tasks_are_pinned_unique_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fetched = root / "fetched"
            archives = fetched / "_archives"
            archives.mkdir(parents=True)
            map_text = (
                "type octile\nheight 5\nwidth 6\nmap\n"
                "......\n......\n......\n......\n......\n"
            )
            archive = archives / "maps.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("compact.map", map_text)
            map_sha = hashlib.sha256(map_text.encode("utf-8")).hexdigest()
            config = root / "config.json"
            payload = {
                "schema_version": 1,
                "dataset_revision": "test-derived-v1",
                "source": "https://example.test/maps",
                "master_seed": 1234,
                "task_seeds": [11],
                "task_variants": ["uniform_random", "opposite_exchange"],
                "map_archive": {
                    "url": "https://example.test/maps.zip",
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
                "expected_map_count": 1,
                "expected_instance_count": 2,
                "benchmarks": [
                    {
                        "id": "compact",
                        "layout_family": "test_compact",
                        "member": "compact.map",
                        "member_sha256": map_sha,
                        "agent_counts": [8],
                    }
                ],
            }
            config.write_text(json.dumps(payload), encoding="utf-8")

            first = prepare_movingai_map_derived_dataset(
                fetched, config, root / "first"
            )
            second = prepare_movingai_map_derived_dataset(
                fetched, config, root / "second"
            )

            self.assertEqual(first, second)
            split = root / "first" / "balanced_wall_clock"
            rows = [
                json.loads(line)
                for line in (split / "manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source_group"] for row in rows}, {"movingai"})
            self.assertEqual(
                {row["instance_origin"] for row in rows},
                {"movingai_map_project_derived_od"},
            )
            self.assertEqual(
                hashlib.sha256((split / "maps" / "compact.map").read_bytes()).hexdigest(),
                map_sha,
            )
            scenario_hashes = []
            for row in rows:
                scenario = split / row["scenario_file"]
                scenario_hashes.append(hashlib.sha256(scenario.read_bytes()).hexdigest())
                fields = [
                    line.split()
                    for line in scenario.read_text(encoding="utf-8").splitlines()[1:]
                ]
                starts = [(int(value[4]), int(value[5])) for value in fields]
                goals = [(int(value[6]), int(value[7])) for value in fields]
                self.assertEqual(len(starts), 8)
                self.assertEqual(len(set(starts)), 8)
                self.assertEqual(len(set(goals)), 8)
                self.assertTrue(all(start != goal for start, goal in zip(starts, goals)))
                self.assertTrue(all(int(value[8]) > 0 for value in fields))
            second_split = root / "second" / "balanced_wall_clock"
            self.assertEqual(
                scenario_hashes,
                [
                    hashlib.sha256(
                        (second_split / row["scenario_file"]).read_bytes()
                    ).hexdigest()
                    for row in rows
                ],
            )

    def test_map_derived_movingai_rejects_member_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "fetched" / "_archives"
            archives.mkdir(parents=True)
            archive = archives / "maps.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "compact.map",
                    "type octile\nheight 2\nwidth 2\nmap\n..\n..\n",
                )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_revision": "test",
                        "source": "https://example.test",
                        "master_seed": 1,
                        "task_seeds": [1],
                        "task_variants": [
                            "uniform_random",
                            "opposite_exchange",
                        ],
                        "map_archive": {
                            "url": "https://example.test/maps.zip",
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        },
                        "expected_map_count": 1,
                        "expected_instance_count": 2,
                        "benchmarks": [
                            {
                                "id": "compact",
                                "layout_family": "test",
                                "member": "compact.map",
                                "member_sha256": "0" * 64,
                                "agent_counts": [2],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "member SHA mismatch"):
                prepare_movingai_map_derived_dataset(
                    root / "fetched", config, root / "output"
                )

    def test_qualified_compute_load_pool_pins_and_merges_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for index, source_group in enumerate(("generated", "movingai")):
                dataset = root / f"dataset-{index}"
                split = dataset / "balanced_wall_clock"
                qualification = root / f"qualification-{index}"
                task_id = f"task-{index}"
                map_id = f"map-{index}"
                files = {
                    "map_file": Path("maps") / f"{map_id}.map",
                    "scenario_file": Path("scenarios") / f"{task_id}.scen",
                    "task_file": Path("tasks") / f"{task_id}.json",
                }
                for field, relative in files.items():
                    path = split / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{field}:{index}\n", encoding="utf-8")
                manifest_path = split / "manifest.jsonl"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "split": "balanced_wall_clock",
                            "task_id": task_id,
                            "map_id": map_id,
                            "layout_mode": f"layout-{index}",
                            "source_group": "stale-value",
                            "agent_count": 10,
                            **{name: value.as_posix() for name, value in files.items()},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                qualification.mkdir()
                qualification_path = qualification / "qualification_manifest.jsonl"
                qualification_path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "status": "ok",
                                "initial_complete": True,
                                "task_id": task_id,
                                "map_id": map_id,
                                "solver_seed": seed,
                            }
                        )
                        + "\n"
                        for seed in (1, 2)
                    ),
                    encoding="utf-8",
                )
                sources.append(
                    {
                        "id": f"source-{index}",
                        "source_group": source_group,
                        "dataset": dataset.relative_to(root).as_posix(),
                        "qualification": qualification.relative_to(root).as_posix(),
                        "dataset_manifest_sha256": hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                        "qualification_manifest_sha256": hashlib.sha256(
                            qualification_path.read_bytes()
                        ).hexdigest(),
                        "task_count": 1,
                        "qualification_count": 2,
                    }
                )
            config = root / "configs" / "pool.json"
            config.parent.mkdir()
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_revision": "test-qualified-pool-v1",
                        "sources": sources,
                        "expected": {
                            "map_count": 2,
                            "task_count": 2,
                            "qualification_count": 4,
                            "source_task_counts": {"generated": 1, "movingai": 1},
                            "source_qualification_counts": {
                                "generated": 2,
                                "movingai": 2,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = materialize_qualified_compute_load_pool(
                config, root / "output-dataset", root / "output-qualification"
            )

            self.assertEqual(
                report["dataset"]["splits"]["balanced_wall_clock"]["task_count"],
                2,
            )
            merged = [
                json.loads(line)
                for line in (
                    root
                    / "output-dataset"
                    / "balanced_wall_clock"
                    / "manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["source_group"] for row in merged}, {"generated", "movingai"}
            )
            qualified = [
                json.loads(line)
                for line in (
                    root / "output-qualification" / "qualification_manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(qualified), 4)
            self.assertEqual(
                {row["qualification_source_id"] for row in qualified},
                {"source-0", "source-1"},
            )

    def test_movingai_preparation_can_select_a_registered_source_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fetched = root / "fetched"
            (fetched / "maps").mkdir(parents=True)
            (fetched / "scenarios").mkdir(parents=True)
            source_rows = []
            for map_id in ("keep", "extra"):
                map_path = fetched / "maps" / f"{map_id}.map"
                map_path.write_text(
                    "type octile\nheight 3\nwidth 3\nmap\n...\n...\n...\n",
                    encoding="utf-8",
                )
                scenario_path = fetched / "scenarios" / f"{map_id}-random-11.scen"
                scenario_path.write_text(
                    f"version 1\n0\t{map_id}.map\t3\t3\t0\t0\t2\t2\t4\n",
                    encoding="utf-8",
                )
                source_rows.append(
                    {
                        "id": map_id,
                        "map_file": f"maps/{map_id}.map",
                        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                        "scenarios": [
                            {
                                "index": 11,
                                "file": f"scenarios/{map_id}-random-11.scen",
                                "sha256": hashlib.sha256(
                                    scenario_path.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                )
            (fetched / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            config = root / "config.json"
            payload = {
                "scenario_indices": [11],
                "benchmarks": [
                    {"id": "keep", "layout_family": "test", "agent_counts": [1]}
                ],
            }
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from registration"):
                prepare_movingai_dataset(fetched, config, root / "strict")

            payload["allow_fetched_superset"] = True
            config.write_text(json.dumps(payload), encoding="utf-8")
            report = prepare_movingai_dataset(fetched, config, root / "subset")
            self.assertEqual(report["splits"]["balanced_wall_clock"]["map_count"], 1)
            rows = [
                json.loads(line)
                for line in (
                    root / "subset" / "balanced_wall_clock" / "manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["map_id"] for row in rows}, {"keep"})

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

    def test_compute_load_selector_supports_preregistered_asymmetric_cell_quotas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset" / "balanced_wall_clock"
            qualification = root / "qualification"
            dataset.mkdir(parents=True)
            qualification.mkdir()
            movingai_quotas = {
                "low__low": 2,
                "low__medium": 2,
                "low__high": 2,
                "medium__low": 3,
                "medium__medium": 3,
                "medium__high": 0,
                "high__low": 3,
                "high__medium": 3,
                "high__high": 0,
            }
            conflicts = {"low": 5, "medium": 50, "high": 200}
            loads = {"low": 50_000, "medium": 500_000, "high": 2_000_000}
            tasks = []
            results = []
            for conflict_level in ("low", "medium", "high"):
                for load_level in ("low", "medium", "high"):
                    cell = f"{conflict_level}__{load_level}"
                    movingai_count = movingai_quotas[cell]
                    for index in range(4):
                        source = "movingai" if index < movingai_count else "generated"
                        task_id = f"{cell}-{source}-{index}"
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
                                "initial_conflicts": conflicts[conflict_level],
                                "state_fingerprint": task_id,
                                "initial_complexity": {
                                    "conflict_pair_count": conflicts[conflict_level],
                                    "initial_low_level_generated": loads[load_level],
                                    "initial_low_level_expanded": loads[load_level] // 2,
                                    "total_path_cost": 100,
                                    "conflict_event_count": conflicts[conflict_level],
                                    "active_conflict_agent_ratio": 0.5,
                                    "largest_conflict_component_ratio": 0.5,
                                },
                            }
                        )
            (dataset / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
            )
            (qualification / "qualification_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in results),
                encoding="utf-8",
            )
            config = root / "difficulty.json"
            config.write_text(
                json.dumps(
                    {
                        "conflict_strata": {
                            "low": [1, 10],
                            "medium": [11, 100],
                            "high": [101, 500],
                        },
                        "initial_pp_load_strata": {
                            "low": [0, 100000],
                            "medium": [100001, 1000000],
                            "high": [1000001, None],
                        },
                        "future_cohort_selection": {
                            "jobs_per_conflict_load_cell": 4,
                            "movingai_jobs_by_cell": movingai_quotas,
                            "global_jobs_per_map_cap": 2,
                            "minimum_distinct_maps": 18,
                            "exact_total_source_counts": {
                                "generated": 18,
                                "movingai": 18,
                            },
                            "source_counts_per_conflict_tier": {
                                level: {"generated": 6, "movingai": 6}
                                for level in ("low", "medium", "high")
                            },
                            "minimum_source_count_per_load_tier": {
                                "generated": 2,
                                "movingai": 2,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = select_compute_load_balanced_cohort(
                root / "dataset",
                qualification,
                root / "selected",
                config,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["selected_source_counts"],
                {"generated": 18, "movingai": 18},
            )
            self.assertTrue(report["overall_checks"]["unique_tasks"])
            self.assertTrue(
                report["overall_checks"]["source_balance_per_conflict_tier"]
            )
            self.assertTrue(
                report["overall_checks"]["source_overlap_per_load_tier"]
            )

    def test_compute_load_registration_pins_cohort_and_balanced_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            build = root / "build"
            configs.mkdir()
            build.mkdir()
            registered_paths = {
                "pool_registry": configs / "pool.json",
                "selection_config": configs / "selection.json",
                "merged_dataset_manifest": build / "dataset.jsonl",
                "merged_qualification_manifest": build / "qualification.jsonl",
                "cohort": build / "cohort.jsonl",
                "execution_schedule": build / "schedule.json",
                "cohort_report": build / "report.json",
            }
            for name in (
                "pool_registry",
                "selection_config",
                "merged_qualification_manifest",
            ):
                registered_paths[name].write_text(f"{name}\n", encoding="utf-8")
            movingai_quotas = {
                ("low", "low"): 2,
                ("low", "medium"): 2,
                ("low", "high"): 2,
                ("medium", "low"): 3,
                ("medium", "medium"): 3,
                ("medium", "high"): 0,
                ("high", "low"): 3,
                ("high", "medium"): 3,
                ("high", "high"): 0,
            }
            cohort = []
            source_rows = []
            for conflict in ("low", "medium", "high"):
                for load in ("low", "medium", "high"):
                    movingai_count = movingai_quotas[(conflict, load)]
                    for index in range(4):
                        task_id = f"{conflict}-{load}-{index}"
                        cohort.append(
                            {
                                "task_id": task_id,
                                "solver_seed": 1,
                                "map_id": f"map-{task_id}",
                                "conflict_stratum": conflict,
                                "initial_pp_load_stratum": load,
                                "source_group": (
                                    "movingai"
                                    if index < movingai_count
                                    else "generated"
                                ),
                            }
                        )
                        map_file = Path("maps") / f"map-{task_id}.map"
                        scenario_file = Path("scenarios") / f"{task_id}.scen"
                        (build / map_file).parent.mkdir(parents=True, exist_ok=True)
                        (build / scenario_file).parent.mkdir(parents=True, exist_ok=True)
                        (build / map_file).write_text("type octile\nheight 1\nwidth 1\nmap\n.\n")
                        (build / scenario_file).write_text("version 1\n")
                        source_rows.append(
                            {
                                "task_id": task_id,
                                "map_id": f"map-{task_id}",
                                "source_group": cohort[-1]["source_group"],
                                "map_file": map_file.as_posix(),
                                "scenario_file": scenario_file.as_posix(),
                            }
                        )
            registered_paths["merged_dataset_manifest"].write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            registered_paths["cohort"].write_text(
                "".join(json.dumps(row) + "\n" for row in cohort),
                encoding="utf-8",
            )
            orders = list(itertools.permutations((
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            )))
            entries = [
                {
                    **row,
                    "schedule_group": index % 6,
                    "controller_order": list(orders[index % 6]),
                }
                for index, row in enumerate(cohort)
            ]
            registered_paths["execution_schedule"].write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )
            registered_paths["cohort_report"].write_text(
                json.dumps({"formal_collection_allowed": True}), encoding="utf-8"
            )
            counts = {
                "jobs": 36,
                "tasks": 36,
                "maps": 36,
                "jobs_per_conflict_load_cell": 4,
                "source": {"generated": 18, "movingai": 18},
                "source_per_conflict_tier": {
                    level: {"generated": 6, "movingai": 6}
                    for level in ("low", "medium", "high")
                },
                "source_per_initial_pp_load_tier": {
                    "low": {"generated": 4, "movingai": 8},
                    "medium": {"generated": 4, "movingai": 8},
                    "high": {"generated": 10, "movingai": 2},
                },
            }
            registration = {
                "schema": "test.registration.v1",
                "status": "preregistered_before_controller_collection",
                "counts": counts,
            }
            for path_key, path in registered_paths.items():
                registration[path_key] = path.relative_to(root).as_posix()
                registration[f"{path_key}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            registration_path = configs / "registration.json"
            registration_path.write_text(json.dumps(registration), encoding="utf-8")

            report = verify_compute_load_cohort_registration(registration_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"], counts)

            dataset = materialize_registered_compute_load_cohort(
                registration_path, root / "formal-dataset"
            )
            self.assertEqual(dataset["splits"]["balanced_wall_clock"]["map_count"], 36)
            self.assertEqual(
                dataset["splits"]["balanced_wall_clock"]["instance_count"], 36
            )

            registered_paths["cohort"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registration mismatch: cohort"):
                verify_compute_load_cohort_registration(registration_path)

    def test_analysis_uses_paired_map_and_stratum_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_rows = []
            for index in range(12):
                stratum = ("low", "medium", "high")[index % 3]
                load_stratum = ("low", "medium", "high")[index % 3]
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
                        "initial_pp_load_stratum": load_stratum,
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
            self.assertEqual(
                set(report["comparisons"]["mixed_vs_v2_by_initial_pp_load"]),
                {"low", "medium", "high"},
            )
            self.assertEqual(
                report["promotion_gate_counts"]["map_not_worse_requirement"], 8
            )
            success_ttf = analyze_success_only_ttf(
                collection, cohort, root / "success-ttf"
            )
            self.assertEqual(
                success_ttf["definition"]["primary_comparison"],
                "paired_common_success",
            )
            self.assertAlmostEqual(
                success_ttf["success_only"]["official_adaptive"]["mean_seconds"],
                120.0,
            )
            self.assertAlmostEqual(
                success_ttf["paired"]["overall"]["mixed_vs_v2"][
                    "mean_improvement"
                ],
                0.1,
            )
            self.assertTrue(
                (
                    root
                    / "success-ttf"
                    / "success_only_ttf_instances.csv"
                ).is_file()
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
            markdown = (
                root / "difficulty-report" / "difficulty_audit_zh.md"
            ).read_text(encoding="utf-8")
            self.assertIn("# V2 / Mixed Full 分层墙钟确认", markdown)
            self.assertIn("未通过计算负载平衡门槛", markdown)


if __name__ == "__main__":
    unittest.main()
