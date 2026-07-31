from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from experiments.balanced_wall_clock import (
    CORRECTED_FULL_POOL_REPORT_SCHEMA,
    CORRECTED_FULL_POOL_SCHEDULE_SCHEMA,
    _four_neighbor_distances,
    _load_scheduled_controller_rows,
    _paired_success_ttf,
    _success_ttf_markdown,
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
    prepare_corrected_native_formal_config,
    qualify_corrected_native_schedule,
    qualify_corrected_native_seed_pool,
    rebind_corrected_native_schedule,
    select_balanced_cohort,
    select_compute_load_balanced_cohort,
    select_corrected_native_seed_schedule,
    verify_compute_load_cohort_registration,
)
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.repair_collection import _dataset_fingerprint, state_fingerprint
from experiments.state_analysis import summarize_initial_state_complexity


def _corrected_implementation_identity() -> dict[str, object]:
    native_identity = {
        "path": "lns2_env.so",
        "sha256": "b" * 64,
        "repair_timing_schema": "lns2.repair_timing.v2",
        "native_semantics_schema": "lns2.corrected_native.v1",
    }
    files = {"experiments/closed_loop_confirmation.py": "c" * 64}
    implementation: dict[str, object] = {
        "files": files,
        "native_module": native_identity,
    }
    implementation["sha256"] = hashlib.sha256(
        json.dumps(
            {"files": files, "native_module": native_identity},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return implementation


def _structured_corrected_producer_identity() -> dict[str, object]:
    return {
        "schema": "lns2.producer_identity.v2",
        "source_sha256": {
            "experiments/closed_loop_confirmation.py": "c" * 64,
        },
        "python": {
            "implementation": "CPython",
            "version": "3.10.12",
        },
        "packages": {
            "joblib": "1.3.2",
            "numpy": "1.26.4",
            "scikit-learn": "1.4.2",
        },
        "native_required": True,
        "native": {
            "path": "/tmp/lns2_env.so",
            "sha256": "b" * 64,
            "repair_timing_schema": "lns2.repair_timing.v2",
            "native_semantics_schema": "lns2.corrected_native.v1",
        },
    }


def _write_corrected_seed_pool_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    source_root = root / "source"
    dataset_root = root / "dataset"
    qualification_root = root / "qualification"
    source_root.mkdir()
    (dataset_root / "balanced_wall_clock").mkdir(parents=True)
    qualification_root.mkdir()
    controllers = (
        "official_adaptive",
        "v2-full",
        "mixed-full-v2",
    )
    orders = list(itertools.permutations(controllers))
    conflict_specs = (("low", 5), ("medium", 50), ("high", 200))
    load_specs = (
        ("low", 50_000),
        ("medium", 500_000),
        ("high", 2_000_000),
    )
    source_rows = []
    dataset_rows = []
    qualification_rows = []
    index = 0
    for conflict_name, conflicts in conflict_specs:
        for load_name, generated in load_specs:
            for copy in range(4):
                source_group = "generated" if copy < 2 else "movingai"
                task_id = f"task-{index:02d}"
                map_id = f"map-{index:02d}"
                group = index % 6
                source_rows.append(
                    {
                        "task_id": task_id,
                        "solver_seed": 1,
                        "map_id": map_id,
                        "layout_mode": f"{source_group}-layout",
                        "source_group": source_group,
                        "agent_count": 400,
                        "agent_band": "medium",
                        "initial_conflicts": conflicts,
                        "conflict_stratum": conflict_name,
                        "initial_pp_load_stratum": load_name,
                        "initial_low_level_generated": generated,
                        "state_fingerprint": f"legacy-{index}",
                        "schedule_group": group,
                        "controller_order": list(orders[group]),
                    }
                )
                dataset_rows.append(
                    {
                        "task_id": task_id,
                        "map_id": map_id,
                        "layout_mode": f"{source_group}-layout",
                        "source_group": source_group,
                        "agent_count": 400,
                    }
                )
                for solver_seed in (1, 2, 3):
                    feasible = index == 0 and solver_seed == 1
                    candidate_conflicts = (
                        0
                        if feasible
                        else conflicts
                        if solver_seed in {1, 2}
                        else conflicts + 1
                    )
                    candidate_generated = (
                        generated if solver_seed in {1, 2} else generated + 100
                    )
                    qualification_rows.append(
                        {
                            "schema": "lns2.repair_collection.v2",
                            "schema_version": 2,
                            "status": "ok",
                            "error": None,
                            "initial_complete": True,
                            "initial_feasible": feasible,
                            "repairable": not feasible,
                            "task_id": task_id,
                            "solver_seed": solver_seed,
                            "map_id": map_id,
                            "layout_mode": f"{source_group}-layout",
                            "agent_count": 400,
                            "initial_conflicts": candidate_conflicts,
                            "state_fingerprint": hashlib.sha256(
                                f"{task_id}-{solver_seed}".encode("utf-8")
                            ).hexdigest(),
                            "initial_complexity": {
                                "conflict_pair_count": candidate_conflicts,
                                "active_conflict_agent_ratio": (
                                    candidate_conflicts / 400.0
                                ),
                                "conflict_event_count": candidate_conflicts,
                                "initial_low_level_expanded": (
                                    candidate_generated // 2
                                ),
                                "initial_low_level_generated": (
                                    candidate_generated
                                ),
                                "largest_conflict_component_ratio": (
                                    0.0 if feasible else 0.1
                                ),
                                "total_path_cost": 1000 + index,
                            },
                        }
                    )
                index += 1
    source_schedule_path = source_root / "execution_schedule.json"
    source_schedule_path.write_text(
        json.dumps({"entries": source_rows}), encoding="utf-8"
    )
    dataset_manifest = (
        dataset_root / "balanced_wall_clock" / "manifest.jsonl"
    )
    dataset_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in dataset_rows),
        encoding="utf-8",
    )
    qualification_manifest = (
        qualification_root / "qualification_manifest.jsonl"
    )
    qualification_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in qualification_rows),
        encoding="utf-8",
    )
    keys = [
        [row["task_id"], solver_seed]
        for row in source_rows
        for solver_seed in (1, 2, 3)
    ]
    registration_path = root / "registration.json"
    registration = {
        "schema": "lns2.compute_load_balanced_wall_clock_registration.v1",
        "selection_blind_to_controller_outcomes": True,
        "execution_schedule_sha256": hashlib.sha256(
            source_schedule_path.read_bytes()
        ).hexdigest(),
        "formal_dataset_manifest_sha256": hashlib.sha256(
            dataset_manifest.read_bytes()
        ).hexdigest(),
    }
    registration_path.write_text(
        json.dumps(registration), encoding="utf-8"
    )
    configuration = {
        "cohort_job_keys_override": sorted(keys),
        "solver_seeds": [1, 2, 3],
        "formal": True,
        "stopping_rule": "wall-clock-fixed-metric",
        "max_decisions": 0,
        "metric_iteration_budget": 100,
        "wall_time_budget_seconds": 600.0,
        "episode_process_timeout_seconds": 660.0,
        "environment": {
            "max_repair_iterations": 0,
            "time_limit": 600.0,
        },
        "dataset_design": {
            "mode": "balanced_wall_clock",
            "dataset_revision": "balanced-wall-clock-formal-cohort-v6",
            "map_count": 36,
            "instance_count": 36,
            "source_counts": {"generated": 18, "movingai": 18},
            "formal_dataset_manifest_sha256": hashlib.sha256(
                dataset_manifest.read_bytes()
            ).hexdigest(),
            "source_task_registration": str(registration_path),
            "source_task_registration_sha256": hashlib.sha256(
                registration_path.read_bytes()
            ).hexdigest(),
        },
    }
    configuration_fingerprint = hashlib.sha256(
        json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    (qualification_root / "run_config.json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "configuration_fingerprint": configuration_fingerprint,
                "run_fingerprint": "a" * 64,
                "dataset_fingerprint": "d" * 64,
                "controller_implementation": (
                    _corrected_implementation_identity()
                ),
            }
        ),
        encoding="utf-8",
    )
    repairable = [
        [row["task_id"], row["solver_seed"]]
        for row in qualification_rows
        if not row["initial_feasible"]
    ]
    (qualification_root / "qualification_report.json").write_text(
        json.dumps(
            {
                "schema": "lns2.closed_loop_confirmation.v1",
                "formal": False,
                "passed": True,
                "valid_count": 108,
                "expected_reset_count": 108,
                "registered_solver_seeds": [1, 2, 3],
                "errors": [],
                "incomplete_reset_count": 0,
                "inconsistent_initial_state_count": 0,
                "initial_feasible_count": 1,
                "nonzero_state_count": 107,
                "gates": {"all_resets_valid": True},
                "repairable_episode_keys": sorted(repairable),
            }
        ),
        encoding="utf-8",
    )
    observed_root = root / "observed"
    probe_root = root / "probe"
    for evidence_root, evidence_rows in (
        (
            observed_root,
            [
                row
                for row in qualification_rows
                if row["solver_seed"] == 1
            ],
        ),
        (probe_root, qualification_rows[:2]),
    ):
        evidence_root.mkdir()
        (evidence_root / "qualification_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in evidence_rows),
            encoding="utf-8",
        )
        (evidence_root / "qualification_report.json").write_text(
            json.dumps(
                {
                    "schema": "lns2.closed_loop_confirmation.v1",
                    "passed": True,
                    "valid_count": len(evidence_rows),
                    "expected_reset_count": len(evidence_rows),
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        (evidence_root / "run_config.json").write_bytes(
            (qualification_root / "run_config.json").read_bytes()
        )
    return (
        source_root,
        dataset_root,
        qualification_root,
        observed_root,
        probe_root,
    )


def _write_full_pool_collection_fixture(
    root: Path,
) -> tuple[Path, Path, dict[str, object]]:
    schedule_root = root / "schedule"
    schedule_root.mkdir()
    controllers = (
        "official_adaptive",
        "v2-full",
        "mixed-full-v2",
    )
    entries = [
        {
            "task_id": f"task-{group}",
            "solver_seed": group % 3 + 1,
            "map_id": f"map-{group}",
            "agent_count": 100,
            "schedule_group": group,
            "controller_order": list(controllers),
        }
        for group in range(6)
    ]
    producer = _structured_corrected_producer_identity()
    producer_fingerprint = hashlib.sha256(
        json.dumps(
            producer, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    provenance: dict[str, object] = {
        "native_semantics_schema": "lns2.corrected_native.v1",
        "qualification_producer_identity": producer,
        "qualification_producer_identity_fingerprint": producer_fingerprint,
        "formal_dataset_fingerprint": "b" * 64,
        "stopping_contract": {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 600.0,
            "environment_time_limit_seconds": 600.0,
            "episode_process_timeout_seconds": 660.0,
            "safety_max_decisions": 100_000,
            "safety_limit_is_not_metric_cap": True,
        },
    }
    schedule: dict[str, object] = {
        "schema": CORRECTED_FULL_POOL_SCHEDULE_SCHEMA,
        "provenance": provenance,
        "entries": entries,
    }
    schedule_path = schedule_root / "execution_schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    report: dict[str, object] = {
        "schema": CORRECTED_FULL_POOL_REPORT_SCHEMA,
        "passed": True,
        "formal_collection_allowed": True,
        "execution_schedule_sha256": hashlib.sha256(
            schedule_path.read_bytes()
        ).hexdigest(),
        "provenance": provenance,
    }
    (schedule_root / "cohort_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (root / "formal-config.json").write_text(
        json.dumps(
            {
                "stopping_rule": "wall-clock-fixed-metric",
                "max_decisions": 0,
                "metric_iteration_budget": 100,
                "wall_time_budget_seconds": 600.0,
                "episode_process_timeout_seconds": 660.0,
                "environment": {
                    "max_repair_iterations": 0,
                    "time_limit": 600.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return schedule_root, root / "dataset", report


class BalancedWallClockTests(unittest.TestCase):
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

    def test_success_only_ttf_requires_strict_boolean_and_failure_null(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            _successful_ttf(
                {
                    "summary": {
                        "success": "false",
                        "wall_time_to_feasible": None,
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "failed episode must use null"):
            _successful_ttf(
                {
                    "summary": {
                        "success": False,
                        "wall_time_to_feasible": 12.0,
                    }
                }
            )

    def test_success_only_markdown_uses_na_for_zero_successes(self) -> None:
        empty_summary = {
            "success_count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
        empty_pair = {
            "common_success_count": 0,
            "baseline": empty_summary,
            "candidate": empty_summary,
            "mean_improvement": None,
            "candidate_faster_count": 0,
            "map_bootstrap": {"improvement_95_ci": [None, None]},
        }
        report = {
            "episode_count": 1,
            "success_only": {
                controller: dict(empty_summary)
                for controller in (
                    "official_adaptive",
                    "v2-full",
                    "mixed-full-v2",
                )
            },
            "paired": {
                "overall": {"empty": empty_pair},
                "conflict_stratum": {},
                "initial_pp_load_stratum": {},
                "source_group": {},
                "conflict_load_cell": {},
            },
            "instances": [],
        }
        markdown = _success_ttf_markdown(report)
        self.assertIn("| empty | 0 | NA | NA | NA | NA | 0/0 |", markdown)

    def test_formal_loader_rejects_duplicate_lane_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_rows = []
            for group in range(6):
                schedule_rows.append(
                    {
                        "task_id": f"task-{group}",
                        "solver_seed": group,
                        "map_id": f"map-{group}",
                        "agent_count": 10,
                        "schedule_group": group,
                        "controller_order": [
                            "official_adaptive",
                            "v2-full",
                            "mixed-full-v2",
                        ],
                    }
                )
            for group, scheduled in enumerate(schedule_rows):
                for controller in (
                    "official_adaptive",
                    "v2-full",
                    "mixed-full-v2",
                ):
                    phase = (
                        "official_adaptive"
                        if controller == "official_adaptive"
                        else "realized_dynamic"
                    )
                    lane = root / f"order_{group}" / controller
                    lane.mkdir(parents=True)
                    row = {
                        "task_id": scheduled["task_id"],
                        "solver_seed": scheduled["solver_seed"],
                        "map_id": scheduled["map_id"],
                        "agent_count": scheduled["agent_count"],
                        "status": "error",
                    }
                    rows = [row, row] if group == 0 and controller == "v2-full" else [row]
                    (lane / f"{phase}_manifest.jsonl").write_text(
                        "".join(json.dumps(value) + "\n" for value in rows),
                        encoding="utf-8",
                    )
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                _load_scheduled_controller_rows(
                    root, {"entries": schedule_rows}
                )

    def test_formal_loader_rejects_trace_sha_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controllers = (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            )
            schedule_rows = [
                {
                    "task_id": f"task-{group}",
                    "solver_seed": group,
                    "map_id": f"map-{group}",
                    "agent_count": 10,
                    "schedule_group": group,
                    "controller_order": list(controllers),
                }
                for group in range(6)
            ]
            for group, scheduled in enumerate(schedule_rows):
                key = [scheduled["task_id"], scheduled["solver_seed"]]
                for controller in controllers:
                    phase = (
                        "official_adaptive"
                        if controller == "official_adaptive"
                        else "realized_dynamic"
                    )
                    lane = root / f"order_{group}" / controller
                    lane.mkdir(parents=True)
                    trace = lane / "trace.jsonl"
                    trace.write_text("{}\n", encoding="utf-8")
                    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
                    summary = {"success": True}
                    row = {
                        "task_id": scheduled["task_id"],
                        "solver_seed": scheduled["solver_seed"],
                        "map_id": scheduled["map_id"],
                        "agent_count": scheduled["agent_count"],
                        "episode_id": (
                            f"{scheduled['task_id']}__seed_"
                            f"{scheduled['solver_seed']:04d}__{phase}"
                        ),
                        "policy": phase,
                        "status": "ok",
                        "summary": summary,
                        "trace_file": trace.name,
                        "trace_sha256": trace_sha,
                        "trace_bytes": len(trace.read_bytes()),
                        "trace_event_count": 1,
                        "trace_format": "full-v1",
                        "initial_state_ref": "state",
                        "storage_fingerprint": "storage",
                    }
                    (lane / f"{phase}_manifest.jsonl").write_text(
                        json.dumps(row) + "\n", encoding="utf-8"
                    )
                    configuration = {
                        "cohort_job_keys_override": [key],
                        "formal": True,
                        "metric_iteration_budget": 100,
                    }
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            configuration,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    (lane / "run_config.json").write_text(
                        json.dumps(
                            {
                                "configuration": configuration,
                                "configuration_fingerprint": fingerprint,
                                "run_fingerprint": "a" * 64,
                                "verification_profile": "deployment",
                                "storage_fingerprint": "storage",
                                "trace_format": "full-v1",
                                "controller_implementation": {},
                            }
                        ),
                        encoding="utf-8",
                    )
            (root / "collection_progress.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "group": group,
                                "controller": controller,
                                "job_count": 1,
                                "dry_run": False,
                            }
                            for group in range(6)
                            for controller in controllers
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "experiments.balanced_wall_clock.validate_closed_loop_trace",
                return_value={
                    "summary": {"success": True},
                    "event_count": 1,
                    "initial_state_ref": "state",
                },
            ):
                loaded = _load_scheduled_controller_rows(
                    root, {"entries": schedule_rows}
                )
                self.assertEqual(loaded[3]["validation_mode"], "run_config_trace_semantic")
                progress_path = root / "collection_progress.json"
                progress = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
                progress["entries"][0], progress["entries"][1] = (
                    progress["entries"][1],
                    progress["entries"][0],
                )
                progress_path.write_text(
                    json.dumps(progress), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, "execution order differs"
                ):
                    _load_scheduled_controller_rows(
                        root, {"entries": schedule_rows}
                    )
                progress["entries"][0], progress["entries"][1] = (
                    progress["entries"][1],
                    progress["entries"][0],
                )
                progress_path.write_text(
                    json.dumps(progress), encoding="utf-8"
                )
                tampered_path = (
                    root
                    / "order_0"
                    / "v2-full"
                    / "realized_dynamic_manifest.jsonl"
                )
                tampered = json.loads(
                    tampered_path.read_text(encoding="utf-8")
                )
                tampered["trace_sha256"] = "0" * 64
                tampered_path.write_text(
                    json.dumps(tampered) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "trace SHA256 mismatch"):
                    _load_scheduled_controller_rows(
                        root, {"entries": schedule_rows}
                    )

    def test_full_pool_collection_uses_derived_schedule_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, report = (
                _write_full_pool_collection_fixture(root)
            )
            validated = {
                "report": report,
                "passed": True,
                "formal_dataset_fingerprint": "b" * 64,
            }
            with mock.patch(
                "experiments.corrected_native_full_pool."
                "validate_corrected_native_full_pool_schedule",
                return_value=validated,
            ) as validate, mock.patch(
                "experiments.balanced_wall_clock._dataset_fingerprint",
                return_value="b" * 64,
            ), mock.patch(
                "experiments.balanced_wall_clock._controller_producer_identity",
                return_value=report["provenance"][
                    "qualification_producer_identity"
                ],
            ), mock.patch(
                "experiments.balanced_wall_clock."
                "_validate_corrected_seed_schedule_semantics"
            ) as validate_seed, mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                return_value={"status": "dry-run"},
            ) as run:
                result = collect_scheduled(
                    dataset=dataset,
                    config=root / "formal-config.json",
                    qualification=None,
                    schedule_root=schedule_root,
                    output=root / "collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                    registration=None,
                    stopping_rule="wall-clock-fixed-metric",
                )
            validate.assert_called_once_with(
                schedule_root.resolve(),
                dataset=dataset,
            )
            validate_seed.assert_not_called()
            self.assertEqual(run.call_count, 18)
            self.assertEqual(result["lane_count"], 18)
            self.assertTrue(
                all(
                    call.kwargs["stopping_rule"]
                    == "wall-clock-fixed-metric"
                    for call in run.call_args_list
                )
            )

    def test_full_pool_collection_contract_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, report = (
                _write_full_pool_collection_fixture(root)
            )
            validated = {
                "report": report,
                "passed": True,
                "formal_dataset_fingerprint": "b" * 64,
            }
            config_path = root / "formal-config.json"
            schedule_path = schedule_root / "execution_schedule.json"
            base_config = json.loads(config_path.read_text(encoding="utf-8"))
            base_schedule = json.loads(
                schedule_path.read_text(encoding="utf-8")
            )
            cases = (
                (
                    "python repair cap",
                    "config",
                    ("max_decisions",),
                    100,
                    "0/0/100/600/600/660",
                ),
                (
                    "native repair cap",
                    "config",
                    ("environment", "max_repair_iterations"),
                    100,
                    "0/0/100/600/600/660",
                ),
                (
                    "fixed metric drift",
                    "config",
                    ("metric_iteration_budget",),
                    101,
                    "0/0/100/600/600/660",
                ),
                (
                    "diagnostic safety drift",
                    "schedule",
                    ("provenance", "stopping_contract", "safety_max_decisions"),
                    100,
                    "stopping contract",
                ),
            )
            for label, target, path, value, error in cases:
                with self.subTest(label=label):
                    bad_config = json.loads(json.dumps(base_config))
                    bad_schedule = json.loads(json.dumps(base_schedule))
                    payload = (
                        bad_config if target == "config" else bad_schedule
                    )
                    cursor = payload
                    for key in path[:-1]:
                        cursor = cursor[key]
                    cursor[path[-1]] = value
                    config_path.write_text(
                        json.dumps(bad_config), encoding="utf-8"
                    )
                    schedule_path.write_text(
                        json.dumps(bad_schedule), encoding="utf-8"
                    )
                    with mock.patch(
                        "experiments.corrected_native_full_pool."
                        "validate_corrected_native_full_pool_schedule",
                        return_value=validated,
                    ), mock.patch(
                        "experiments.balanced_wall_clock._dataset_fingerprint",
                        return_value="b" * 64,
                    ), mock.patch(
                        "experiments.balanced_wall_clock."
                        "run_closed_loop_collection"
                    ) as run:
                        with self.assertRaisesRegex(ValueError, error):
                            collect_scheduled(
                                dataset=dataset,
                                config=config_path,
                                qualification=None,
                                schedule_root=schedule_root,
                                output=root / "collection",
                                original_bundle=root / "original",
                                mixed_bundle=root / "mixed",
                                resume=False,
                                dry_run=True,
                            )
                    run.assert_not_called()

    def test_full_pool_v1_is_read_only_for_formal_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, _report = (
                _write_full_pool_collection_fixture(root)
            )
            schedule_path = schedule_root / "execution_schedule.json"
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            schedule["schema"] = (
                "lns2.controller_execution_schedule."
                "corrected_native_full_pool.v1"
            )
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection"
            ) as run:
                with self.assertRaisesRegex(
                    ValueError,
                    "v1/v2 artifacts are read-only",
                ):
                    collect_scheduled(
                        dataset=dataset,
                        config=root / "formal-config.json",
                        qualification=None,
                        schedule_root=schedule_root,
                        output=root / "collection",
                        original_bundle=root / "original",
                        mixed_bundle=root / "mixed",
                        resume=False,
                        dry_run=True,
                    )
            run.assert_not_called()

    def test_full_pool_collection_rejects_current_producer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, report = (
                _write_full_pool_collection_fixture(root)
            )
            validated = {
                "report": report,
                "passed": True,
                "formal_dataset_fingerprint": "b" * 64,
            }
            current = _structured_corrected_producer_identity()
            current["packages"] = {
                **current["packages"],
                "numpy": "0.0-source-mismatch",
            }
            with mock.patch(
                "experiments.corrected_native_full_pool."
                "validate_corrected_native_full_pool_schedule",
                return_value=validated,
            ), mock.patch(
                "experiments.balanced_wall_clock._dataset_fingerprint",
                return_value="b" * 64,
            ), mock.patch(
                "experiments.balanced_wall_clock._controller_producer_identity",
                return_value=current,
            ), mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection"
            ) as run:
                with self.assertRaisesRegex(
                    ValueError, "package/source/native producer identity"
                ):
                    collect_scheduled(
                        dataset=dataset,
                        config=root / "formal-config.json",
                        qualification=None,
                        schedule_root=schedule_root,
                        output=root / "collection",
                        original_bundle=root / "original",
                        mixed_bundle=root / "mixed",
                        resume=False,
                        dry_run=True,
                    )
            run.assert_not_called()

    def test_full_pool_collection_rejects_schedule_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, _report = (
                _write_full_pool_collection_fixture(root)
            )
            with mock.patch(
                "experiments.corrected_native_full_pool."
                "validate_corrected_native_full_pool_schedule",
                side_effect=ValueError(
                    "corrected-native schedule/report identity is invalid"
                ),
            ), mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection"
            ) as run:
                with self.assertRaisesRegex(
                    ValueError, "schedule/report identity"
                ):
                    collect_scheduled(
                        dataset=dataset,
                        config=root / "formal-config.json",
                        qualification=None,
                        schedule_root=schedule_root,
                        output=root / "collection",
                        original_bundle=root / "original",
                        mixed_bundle=root / "mixed",
                        resume=False,
                        dry_run=True,
                    )
            run.assert_not_called()

    def test_full_pool_collection_rejects_dataset_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root, dataset, report = (
                _write_full_pool_collection_fixture(root)
            )
            with mock.patch(
                "experiments.corrected_native_full_pool."
                "validate_corrected_native_full_pool_schedule",
                return_value={
                    "report": report,
                    "passed": True,
                    "formal_dataset_fingerprint": "b" * 64,
                },
            ), mock.patch(
                "experiments.balanced_wall_clock._dataset_fingerprint",
                return_value="c" * 64,
            ), mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection"
            ) as run:
                with self.assertRaisesRegex(
                    ValueError, "not bound to the supplied derived dataset"
                ):
                    collect_scheduled(
                        dataset=dataset,
                        config=root / "formal-config.json",
                        qualification=None,
                        schedule_root=schedule_root,
                        output=root / "collection",
                        original_bundle=root / "original",
                        mixed_bundle=root / "mixed",
                        resume=False,
                        dry_run=True,
                    )
            run.assert_not_called()

    def test_prepare_full_pool_formal_config_binds_uncapped_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "formal-dataset"
            split = dataset / "balanced_wall_clock"
            split.mkdir(parents=True)
            rows = []
            for index in range(36):
                source = "generated" if index % 2 == 0 else "movingai"
                layout = "generated-layout" if source == "generated" else "game"
                row = {
                    "task_id": f"task-{index:02d}",
                    "map_id": f"map-{index:02d}",
                    "split": "balanced_wall_clock",
                    "layout_mode": layout,
                    "source_group": source,
                    "agent_count": 100,
                    "map_file": f"maps/map-{index:02d}.map",
                    "scenario_file": f"scenarios/task-{index:02d}.scen",
                    "map_metadata_file": f"maps/map-{index:02d}.json",
                    "task_file": f"tasks/task-{index:02d}.json",
                }
                for field in (
                    "map_file",
                    "scenario_file",
                    "map_metadata_file",
                    "task_file",
                ):
                    path = split / row[field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        f"{field}:{index}\n", encoding="utf-8"
                    )
                rows.append(row)
            manifest = split / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = {
                "schema_version": 1,
                "dataset_revision": (
                    "balanced-wall-clock-corrected-native-selected-v3"
                ),
                "splits": {
                    "balanced_wall_clock": {
                        "map_count": 36,
                        "instance_count": 36,
                        "source_counts": {
                            "generated": 18,
                            "movingai": 18,
                        },
                        "layout_counts": {
                            "game": 18,
                            "generated-layout": 18,
                        },
                    }
                },
            }
            (dataset / "dataset_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            schedule_root = root / "formal-schedule"
            schedule_root.mkdir()
            (schedule_root / "execution_schedule.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (schedule_root / "cohort_report.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (schedule_root / "materialization_report.json").write_text(
                "{}\n", encoding="utf-8"
            )
            base = root / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "formal": True,
                        "experiment_revision": "corrected-pool-v1",
                        "split": "balanced_wall_clock",
                        "solver_seeds": [1, 2, 3],
                        "policies": [
                            "official_adaptive",
                            "realized_dynamic",
                        ],
                        "environment": {
                            "time_limit": 600.0,
                            "max_repair_iterations": 0,
                        },
                        "max_decisions": 0,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "episode_process_timeout_seconds": 660.0,
                        "dataset_design": {
                            "historical_map_ids": ["historical-map"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            dataset_fingerprint = _dataset_fingerprint(dataset)
            validated = {
                "provenance": {
                    "formal_dataset_fingerprint": dataset_fingerprint
                },
                "formal_dataset_fingerprint": dataset_fingerprint,
                "execution_schedule_sha256": hashlib.sha256(
                    (schedule_root / "execution_schedule.json").read_bytes()
                ).hexdigest(),
                "cohort_report_sha256": hashlib.sha256(
                    (schedule_root / "cohort_report.json").read_bytes()
                ).hexdigest(),
            }
            output = root / "formal-config.json"
            with mock.patch(
                "experiments.corrected_native_full_pool."
                "validate_corrected_native_full_pool_schedule",
                return_value=validated,
            ):
                result = prepare_corrected_native_formal_config(
                    base_config=base,
                    schedule_root=schedule_root,
                    dataset=dataset,
                    output=output,
                )
                config = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(config["max_decisions"], 0)
                self.assertEqual(
                    config["environment"]["max_repair_iterations"], 0
                )
                self.assertEqual(config["metric_iteration_budget"], 100)
                self.assertFalse(
                    result["metric_iteration_budget_is_execution_cap"]
                )
                self.assertEqual(
                    config["experiment_revision"],
                    "corrected-native-selected-v3",
                )
                self.assertEqual(
                    config["dataset_design"]["layout_counts"],
                    {"game": 18, "generated-layout": 18},
                )
                self.assertEqual(
                    config["dataset_design"][
                        "formal_dataset_manifest_sha256"
                    ],
                    hashlib.sha256(manifest.read_bytes()).hexdigest(),
                )
                with self.assertRaisesRegex(ValueError, "new file"):
                    prepare_corrected_native_formal_config(
                        base_config=base,
                        schedule_root=schedule_root,
                        dataset=dataset,
                        output=output,
                    )
                (split / rows[0]["task_file"]).write_text(
                    "tampered\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, "not bound to the supplied formal dataset"
                ):
                    prepare_corrected_native_formal_config(
                        base_config=base,
                        schedule_root=schedule_root,
                        dataset=dataset,
                        output=root / "tampered-config.json",
                    )

    def test_corrected_formal_loader_rejects_capped_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controllers = (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            )
            orders = list(itertools.permutations(controllers))
            native = {
                "path": "lns2_env.so",
                "sha256": "1" * 64,
                "repair_timing_schema": "lns2.repair_timing.v2",
                "native_semantics_schema": "lns2.corrected_native.v1",
            }
            files = {"experiments/closed_loop_confirmation.py": "2" * 64}
            producer = {"files": files, "native_module": native}
            producer["sha256"] = hashlib.sha256(
                json.dumps(
                    {"files": files, "native_module": native},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            producer_fingerprint = hashlib.sha256(
                json.dumps(
                    producer, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            provenance = {
                "native_semantics_schema": "lns2.corrected_native.v1",
                "qualification_producer_identity_fingerprint": producer_fingerprint,
                "qualification_dataset_fingerprint": "3" * 64,
                "stopping_contract": {
                    "stopping_rule": "wall-clock-fixed-metric",
                    "max_decisions": 0,
                    "max_repair_iterations": 0,
                    "metric_iteration_budget": 100,
                    "wall_time_budget_seconds": 600.0,
                    "environment_time_limit_seconds": 600.0,
                },
            }
            schedule_rows = []
            summaries = {}
            for index in range(36):
                group = index % 6
                state_digest = hashlib.sha256(
                    f"state-{index}".encode("utf-8")
                ).hexdigest()
                scheduled = {
                    "task_id": f"task-{index:02d}",
                    "solver_seed": index,
                    "map_id": f"map-{index:02d}",
                    "layout_mode": "generated",
                    "agent_count": 100,
                    "initial_conflicts": 5,
                    "state_fingerprint": state_digest,
                    "conflict_stratum": "low",
                    "initial_low_level_generated": 50_000,
                    "initial_pp_load_stratum": "low",
                    "schedule_group": group,
                    "controller_order": list(orders[group]),
                }
                schedule_rows.append(scheduled)
                for controller in controllers:
                    phase = (
                        "official_adaptive"
                        if controller == "official_adaptive"
                        else "realized_dynamic"
                    )
                    summaries[
                        f"{scheduled['task_id']}__seed_{index:04d}__{phase}"
                    ] = {
                        "initial_fingerprint": state_digest,
                        "initial_conflicts": 5,
                    }
            schedule = {
                "schema": "lns2.controller_execution_schedule.corrected_native_v1",
                "provenance": provenance,
                "entries": schedule_rows,
            }
            for group in range(6):
                group_rows = [
                    row for row in schedule_rows if row["schedule_group"] == group
                ]
                for controller in controllers:
                    phase = (
                        "official_adaptive"
                        if controller == "official_adaptive"
                        else "realized_dynamic"
                    )
                    lane = root / f"order_{group}" / controller
                    lane.mkdir(parents=True)
                    manifest_rows = []
                    for scheduled in group_rows:
                        episode_id = (
                            f"{scheduled['task_id']}__seed_"
                            f"{scheduled['solver_seed']:04d}__{phase}"
                        )
                        trace = lane / f"{scheduled['task_id']}.jsonl"
                        trace.write_text("{}\n", encoding="utf-8")
                        manifest_rows.append(
                            {
                                "task_id": scheduled["task_id"],
                                "solver_seed": scheduled["solver_seed"],
                                "map_id": scheduled["map_id"],
                                "layout_mode": scheduled["layout_mode"],
                                "agent_count": scheduled["agent_count"],
                                "episode_id": episode_id,
                                "policy": phase,
                                "status": "ok",
                                "summary": summaries[episode_id],
                                "trace_file": trace.name,
                                "trace_sha256": hashlib.sha256(
                                    trace.read_bytes()
                                ).hexdigest(),
                                "trace_bytes": len(trace.read_bytes()),
                                "trace_event_count": 1,
                                "trace_format": "full-v1",
                                "initial_state_ref": "state",
                                "storage_fingerprint": "storage",
                            }
                        )
                    (lane / f"{phase}_manifest.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in manifest_rows),
                        encoding="utf-8",
                    )
                    configuration = {
                        "cohort_job_keys_override": [
                            [row["task_id"], row["solver_seed"]]
                            for row in group_rows
                        ],
                        "formal": True,
                        "stopping_rule": "wall-clock-fixed-metric",
                        "max_decisions": 0,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "environment": {
                            "max_repair_iterations": 0,
                            "time_limit": 600.0,
                        },
                    }
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            configuration,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    (lane / "run_config.json").write_text(
                        json.dumps(
                            {
                                "configuration": configuration,
                                "configuration_fingerprint": fingerprint,
                                "run_fingerprint": "4" * 64,
                                "dataset_fingerprint": "3" * 64,
                                "verification_profile": "deployment",
                                "storage_fingerprint": "storage",
                                "trace_format": "full-v1",
                                "controller_implementation": producer,
                            }
                        ),
                        encoding="utf-8",
                    )
            (root / "collection_progress.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "group": group,
                                "controller": controller,
                                "job_count": 6,
                                "dry_run": False,
                            }
                            for group in range(6)
                            for controller in orders[group]
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def validated(*_args, **kwargs):
                return {
                    "summary": summaries[kwargs["expected_episode_id"]],
                    "event_count": 1,
                    "initial_state_ref": "state",
                }

            with mock.patch(
                "experiments.balanced_wall_clock.validate_closed_loop_trace",
                side_effect=validated,
            ):
                loaded = _load_scheduled_controller_rows(root, schedule)
                self.assertEqual(
                    loaded[3]["validation_mode"], "run_config_trace_semantic"
                )
                structured_producer = (
                    _structured_corrected_producer_identity()
                )
                structured_fingerprint = hashlib.sha256(
                    json.dumps(
                        structured_producer,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                schedule["provenance"][
                    "qualification_producer_identity"
                ] = structured_producer
                schedule["provenance"][
                    "qualification_producer_identity_fingerprint"
                ] = structured_fingerprint
                schedule["provenance"]["stopping_contract"].update(
                    {
                        "episode_process_timeout_seconds": 660.0,
                        "safety_max_decisions": 100_000,
                        "safety_limit_is_not_metric_cap": True,
                    }
                )
                for group in range(6):
                    for controller in controllers:
                        lane_config_path = (
                            root
                            / f"order_{group}"
                            / controller
                            / "run_config.json"
                        )
                        lane_config = json.loads(
                            lane_config_path.read_text(encoding="utf-8")
                        )
                        lane_config.pop("controller_implementation")
                        lane_config[
                            "producer_identity"
                        ] = structured_producer
                        lane_config[
                            "producer_identity_fingerprint"
                        ] = structured_fingerprint
                        lane_config["configuration"][
                            "episode_process_timeout_seconds"
                        ] = 660.0
                        lane_config["configuration_fingerprint"] = (
                            hashlib.sha256(
                                json.dumps(
                                    lane_config["configuration"],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                        )
                        lane_config_path.write_text(
                            json.dumps(lane_config), encoding="utf-8"
                        )
                derived_schedule = json.loads(json.dumps(schedule))
                derived_schedule["schema"] = (
                    CORRECTED_FULL_POOL_SCHEDULE_SCHEMA
                )
                derived_schedule["provenance"][
                    "qualification_dataset_fingerprint"
                ] = "9" * 64
                derived_schedule["provenance"][
                    "formal_dataset_fingerprint"
                ] = "3" * 64
                derived_loaded = _load_scheduled_controller_rows(
                    root, derived_schedule
                )
                self.assertEqual(
                    derived_loaded[3]["validation_mode"],
                    "run_config_trace_semantic",
                )
                run_config_path = (
                    root / "order_0" / "official_adaptive" / "run_config.json"
                )
                run_config = json.loads(
                    run_config_path.read_text(encoding="utf-8")
                )
                run_config["configuration"]["max_decisions"] = 100
                run_config["configuration_fingerprint"] = hashlib.sha256(
                    json.dumps(
                        run_config["configuration"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                run_config_path.write_text(
                    json.dumps(run_config), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "stopping contract"):
                    _load_scheduled_controller_rows(root, schedule)

    def test_corrected_seed_pool_qualification_runs_full_cartesian_product(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            entries = [
                {"task_id": f"task-{index:02d}", "solver_seed": 1}
                for index in range(36)
            ]
            (source / "execution_schedule.json").write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "formal": True,
                        "solver_seeds": [1, 2, 3],
                        "max_decisions": 0,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "episode_process_timeout_seconds": 660.0,
                        "environment": {
                            "max_repair_iterations": 0,
                            "time_limit": 600.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            def qualify_side_effect(**kwargs):
                output = Path(kwargs["output"])
                output.mkdir(parents=True)
                keys = sorted(kwargs["job_keys"])
                rows = [
                    {"task_id": task_id, "solver_seed": solver_seed}
                    for task_id, solver_seed in keys
                ]
                report = {
                    "schema": "lns2.closed_loop_confirmation.v1",
                    "passed": True,
                    "valid_count": 108,
                    "expected_reset_count": 108,
                    "errors": [],
                    "incomplete_reset_count": 0,
                    "inconsistent_initial_state_count": 0,
                }
                (output / "qualification_manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (output / "qualification_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                (output / "run_config.json").write_text(
                    "{}", encoding="utf-8"
                )
                return {"qualification": report}

            output = root / "qualification"
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                side_effect=qualify_side_effect,
            ) as run:
                report = qualify_corrected_native_seed_pool(
                    dataset=root / "dataset",
                    config=config,
                    source_schedule_root=source,
                    output=output,
                    original_bundle=root / "bundle",
                    workers=2,
                )
            self.assertEqual(report["task_count"], 36)
            self.assertEqual(report["job_count"], 108)
            self.assertEqual(report["solver_seeds"], [1, 2, 3])
            self.assertEqual(len(run.call_args.kwargs["job_keys"]), 108)
            self.assertEqual(
                run.call_args.kwargs["job_keys"],
                run.call_args.kwargs["cohort_job_keys"],
            )
            self.assertEqual(
                run.call_args.kwargs["stopping_rule"],
                "wall-clock-fixed-metric",
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                qualify_corrected_native_seed_pool(
                    dataset=root / "dataset",
                    config=config,
                    source_schedule_root=source,
                    output=output,
                    original_bundle=root / "bundle",
                )
            invalid = root / "invalid.json"
            invalid.write_text(
                json.dumps(
                    {
                        "formal": True,
                        "solver_seeds": [1, True, 3],
                        "max_decisions": 0,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "episode_process_timeout_seconds": 660.0,
                        "environment": {
                            "max_repair_iterations": 0,
                            "time_limit": 600.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly three"):
                qualify_corrected_native_seed_pool(
                    dataset=root / "dataset",
                    config=invalid,
                    source_schedule_root=source,
                    output=root / "invalid-output",
                    original_bundle=root / "bundle",
                )

    def test_corrected_seed_selection_is_exact_and_preserves_frozen_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dataset, qualification, observed, probe = (
                _write_corrected_seed_pool_fixture(root)
            )
            output = root / "selected"
            report = select_corrected_native_seed_schedule(
                dataset=dataset,
                source_schedule_root=source,
                qualification=qualification,
                observed_qualification=observed,
                seed_probe_qualification=probe,
                output=output,
            )
            self.assertTrue(report["formal_collection_allowed"])
            self.assertEqual(report["qualification_count"], 108)
            self.assertEqual(report["selected_count"], 36)
            self.assertEqual(report["original_seed_retained_count"], 35)
            self.assertEqual(report["solver_seed_changed_count"], 1)
            self.assertTrue(all(report["gates"].values()))
            self.assertEqual(
                set(report["counts"]["cells"].values()), {4}
            )
            schedule = json.loads(
                (output / "execution_schedule.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                schedule["schema"],
                (
                    "lns2.controller_execution_schedule."
                    "corrected_native_seed_reselected_v1"
                ),
            )
            selected = {
                row["task_id"]: row for row in schedule["entries"]
            }
            self.assertEqual(selected["task-00"]["solver_seed"], 2)
            self.assertTrue(
                all(
                    selected[f"task-{index:02d}"]["solver_seed"] == 1
                    for index in range(1, 36)
                )
            )
            source_schedule = json.loads(
                (source / "execution_schedule.json").read_text(
                    encoding="utf-8"
                )
            )
            for old, new in zip(
                source_schedule["entries"], schedule["entries"]
            ):
                self.assertEqual(old["task_id"], new["task_id"])
                self.assertEqual(
                    old["schedule_group"], new["schedule_group"]
                )
                self.assertEqual(
                    old["controller_order"], new["controller_order"]
                )
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                return_value={"dry_run": True},
            ) as run:
                collected = collect_scheduled(
                    dataset=dataset,
                    config=root / "config.json",
                    qualification=qualification,
                    schedule_root=output,
                    output=root / "collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            self.assertEqual(collected["lane_count"], 18)
            self.assertEqual(run.call_count, 18)
            with self.assertRaisesRegex(ValueError, "already exists"):
                select_corrected_native_seed_schedule(
                    dataset=dataset,
                    source_schedule_root=source,
                    qualification=qualification,
                    observed_qualification=observed,
                    seed_probe_qualification=probe,
                    output=output,
                )

    def test_corrected_seed_selection_rejects_tampered_coverage_and_dataset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dataset, qualification, observed, probe = (
                _write_corrected_seed_pool_fixture(root)
            )
            manifest = qualification / "qualification_manifest.jsonl"
            rows = manifest.read_text(encoding="utf-8").splitlines()
            manifest.write_text(
                "\n".join(rows[:-1]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                select_corrected_native_seed_schedule(
                    dataset=dataset,
                    source_schedule_root=source,
                    qualification=qualification,
                    observed_qualification=observed,
                    seed_probe_qualification=probe,
                    output=root / "missing-row",
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dataset, qualification, observed, probe = (
                _write_corrected_seed_pool_fixture(root)
            )
            dataset_manifest = (
                dataset / "balanced_wall_clock" / "manifest.jsonl"
            )
            rows = [
                json.loads(line)
                for line in dataset_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            rows[0]["map_id"] = "tampered-map"
            dataset_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "differs from source schedule"
            ):
                select_corrected_native_seed_schedule(
                    dataset=dataset,
                    source_schedule_root=source,
                    qualification=qualification,
                    observed_qualification=observed,
                    seed_probe_qualification=probe,
                    output=root / "tampered-dataset",
                )

    def test_corrected_rebind_reports_feasible_reset_as_failed_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _dataset, pool, _observed, _probe = (
                _write_corrected_seed_pool_fixture(root)
            )
            qualification = root / "single-seed"
            qualification.mkdir()
            rows = [
                json.loads(line)
                for line in (
                    pool / "qualification_manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if json.loads(line)["solver_seed"] == 1
            ]
            (qualification / "qualification_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            run_config = json.loads(
                (pool / "run_config.json").read_text(encoding="utf-8")
            )
            configuration = run_config["configuration"]
            configuration["cohort_job_keys_override"] = sorted(
                [[row["task_id"], 1] for row in rows]
            )
            run_config["configuration_fingerprint"] = hashlib.sha256(
                json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            (qualification / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            repairable = [
                [row["task_id"], 1]
                for row in rows
                if not row["initial_feasible"]
            ]
            (qualification / "qualification_report.json").write_text(
                json.dumps(
                    {
                        "schema": "lns2.closed_loop_confirmation.v1",
                        "passed": True,
                        "valid_count": 36,
                        "expected_reset_count": 36,
                        "errors": [],
                        "incomplete_reset_count": 0,
                        "inconsistent_initial_state_count": 0,
                        "initial_feasible_count": 1,
                        "nonzero_state_count": 35,
                        "gates": {"all_resets_valid": True},
                        "repairable_episode_keys": repairable,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "rebound"
            report = rebind_corrected_native_schedule(
                source_schedule_root=source,
                qualification=qualification,
                output=output,
            )
            self.assertFalse(report["formal_collection_allowed"])
            self.assertFalse(
                report["gates"]["all_resets_nonzero_nonextreme"]
            )
            self.assertEqual(
                report["decision"],
                "corrected_native_reset_changed_balance_reselect_required",
            )
            self.assertTrue((output / "cohort_report.json").is_file())

    def test_corrected_native_rebind_preserves_design_and_refreshes_reset_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            qualification_root = root / "qualification"
            source_root.mkdir()
            qualification_root.mkdir()
            controllers = (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            )
            orders = list(itertools.permutations(controllers))
            conflict_specs = (("low", 5), ("medium", 50), ("high", 200))
            load_specs = (("low", 50_000), ("medium", 500_000), ("high", 2_000_000))
            schedule_rows = []
            qualification_rows = []
            index = 0
            for conflict_name, conflicts in conflict_specs:
                for load_name, generated in load_specs:
                    for copy in range(4):
                        group = index % 6
                        task_id = f"task-{index:02d}"
                        source = "generated" if index % 2 == 0 else "movingai"
                        schedule_rows.append(
                            {
                                "task_id": task_id,
                                "solver_seed": index,
                                "map_id": f"map-{index:02d}",
                                "layout_mode": source,
                                "source_group": source,
                                "agent_count": 200,
                                "agent_band": "small",
                                "initial_conflicts": conflicts,
                                "conflict_stratum": conflict_name,
                                "initial_pp_load_stratum": load_name,
                                "state_fingerprint": f"old-{index}",
                                "schedule_group": group,
                                "controller_order": list(orders[group]),
                            }
                        )
                        qualification_rows.append(
                            {
                                "status": "ok",
                                "initial_complete": True,
                                "initial_feasible": False,
                                "task_id": task_id,
                                "solver_seed": index,
                                "map_id": f"map-{index:02d}",
                                "layout_mode": source,
                                "source_group": source,
                                "agent_count": 200,
                                "initial_conflicts": conflicts,
                                "state_fingerprint": hashlib.sha256(
                                    f"new-{index}".encode("utf-8")
                                ).hexdigest(),
                                "initial_complexity": {
                                    "conflict_pair_count": conflicts,
                                    "active_conflict_agent_ratio": conflicts / 200.0,
                                    "conflict_event_count": conflicts,
                                    "initial_low_level_expanded": generated // 2,
                                    "initial_low_level_generated": generated,
                                    "largest_conflict_component_ratio": 0.1,
                                    "total_path_cost": 1000 + index,
                                },
                            }
                        )
                        index += 1
            (source_root / "execution_schedule.json").write_text(
                json.dumps({"entries": schedule_rows}), encoding="utf-8"
            )
            (qualification_root / "qualification_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in qualification_rows),
                encoding="utf-8",
            )
            configuration = {
                "cohort_job_keys_override": [
                    [row["task_id"], row["solver_seed"]]
                    for row in schedule_rows
                ],
                "formal": True,
                "stopping_rule": "wall-clock-fixed-metric",
                "max_decisions": 0,
                "metric_iteration_budget": 100,
                "wall_time_budget_seconds": 600.0,
                "environment": {
                    "max_repair_iterations": 0,
                    "time_limit": 600.0,
                },
            }
            configuration_fingerprint = hashlib.sha256(
                json.dumps(
                    configuration, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            native_identity = {
                "path": "lns2_env.so",
                "sha256": "b" * 64,
                "repair_timing_schema": "lns2.repair_timing.v2",
                "native_semantics_schema": "lns2.corrected_native.v1",
            }
            implementation_files = {
                "experiments/closed_loop_confirmation.py": "c" * 64
            }
            implementation = {
                "files": implementation_files,
                "native_module": native_identity,
            }
            implementation["sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        "files": implementation_files,
                        "native_module": native_identity,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            run_config = {
                "configuration": configuration,
                "configuration_fingerprint": configuration_fingerprint,
                "run_fingerprint": "a" * 64,
                "dataset_fingerprint": "d" * 64,
                "controller_implementation": implementation,
            }
            (qualification_root / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            qualification_report = {
                "schema": "lns2.closed_loop_confirmation.v1",
                "passed": True,
                "valid_count": 36,
                "expected_reset_count": 36,
                "errors": [],
                "incomplete_reset_count": 0,
                "inconsistent_initial_state_count": 0,
                "initial_feasible_count": 0,
                "nonzero_state_count": 36,
                "gates": {"all_resets_valid": True},
                "repairable_episode_keys": [
                    [row["task_id"], row["solver_seed"]]
                    for row in schedule_rows
                ],
            }
            (qualification_root / "qualification_report.json").write_text(
                json.dumps(qualification_report), encoding="utf-8"
            )
            output = root / "rebound"
            report = rebind_corrected_native_schedule(
                source_schedule_root=source_root,
                qualification=qualification_root,
                output=output,
            )
            self.assertTrue(report["formal_collection_allowed"])
            self.assertEqual(report["changed_cell_count"], 0)
            rebound = json.loads(
                (output / "execution_schedule.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [
                    (
                        row["task_id"],
                        row["solver_seed"],
                        row["schedule_group"],
                        row["controller_order"],
                    )
                    for row in rebound["entries"]
                ],
                [
                    (
                        row["task_id"],
                        row["solver_seed"],
                        row["schedule_group"],
                        row["controller_order"],
                    )
                    for row in sorted(
                        schedule_rows,
                        key=lambda row: (
                            row["schedule_group"],
                            row["task_id"],
                            row["solver_seed"],
                        ),
                    )
                ],
            )
            self.assertTrue(
                all(
                    len(str(row["state_fingerprint"])) == 64
                    for row in rebound["entries"]
                )
            )
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                return_value={"dry_run": True},
            ) as run:
                collected = collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=qualification_root,
                    schedule_root=output,
                    output=root / "collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            self.assertEqual(collected["lane_count"], 18)
            self.assertEqual(run.call_count, 18)
            schedule_path = output / "execution_schedule.json"
            schedule_bytes = schedule_path.read_bytes()
            schedule_path.write_bytes(schedule_bytes + b"\n")
            with self.assertRaisesRegex(
                ValueError, "does not bind its execution schedule"
            ):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=qualification_root,
                    schedule_root=output,
                    output=root / "schedule-tampered-collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            schedule_path.write_bytes(schedule_bytes)
            with self.assertRaisesRegex(
                ValueError, "not the legacy cohort registration"
            ):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=qualification_root,
                    schedule_root=output,
                    output=root / "registered-collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                    registration=root / "legacy-registration.json",
                )
            qualification_manifest = (
                qualification_root / "qualification_manifest.jsonl"
            )
            qualification_manifest.write_text(
                qualification_manifest.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "differs from schedule provenance"
            ):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=qualification_root,
                    schedule_root=output,
                    output=root / "tampered-collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            with self.assertRaisesRegex(ValueError, "already exists"):
                rebind_corrected_native_schedule(
                    source_schedule_root=source_root,
                    qualification=qualification_root,
                    output=output,
                )

    def test_formal_collection_forces_unbounded_execution_with_fixed_metric(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root = root / "schedule"
            schedule_root.mkdir()
            (schedule_root / "cohort_report.json").write_text(
                json.dumps({"formal_collection_allowed": True}),
                encoding="utf-8",
            )
            orders = list(
                itertools.permutations(
                    ("official_adaptive", "v2-full", "mixed-full-v2")
                )
            )
            entries = [
                {
                    "task_id": f"task-{group}",
                    "solver_seed": group,
                    "schedule_group": group,
                    "controller_order": list(orders[group]),
                }
                for group in range(6)
            ]
            (schedule_root / "execution_schedule.json").write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                return_value={"dry_run": True},
            ) as run:
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=root / "qualification",
                    schedule_root=schedule_root,
                    output=root / "collection",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )
            self.assertEqual(run.call_count, 18)
            self.assertTrue(
                all(
                    call.kwargs["stopping_rule"]
                    == "wall-clock-fixed-metric"
                    for call in run.call_args_list
                )
            )
            with self.assertRaisesRegex(
                ValueError, "requires wall-clock-fixed-metric"
            ):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=root / "qualification",
                    schedule_root=schedule_root,
                    output=root / "collection-legacy",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                    stopping_rule="historical",
                )
            (schedule_root / "cohort_report.json").write_text(
                json.dumps({"formal_collection_allowed": "false"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "did not pass the preregistered data gate"
            ):
                collect_scheduled(
                    dataset=root / "dataset",
                    config=root / "config.json",
                    qualification=root / "qualification",
                    schedule_root=schedule_root,
                    output=root / "collection-string-gate",
                    original_bundle=root / "original",
                    mixed_bundle=root / "mixed",
                    resume=False,
                    dry_run=True,
                )

    def test_formal_collection_rejects_manifest_without_lane_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root = root / "schedule"
            schedule_root.mkdir()
            controllers = (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
            )
            orders = list(itertools.permutations(controllers))
            entries = [
                {
                    "task_id": f"task-{group}",
                    "solver_seed": group,
                    "schedule_group": group,
                    "controller_order": list(orders[group]),
                }
                for group in range(6)
            ]
            (schedule_root / "cohort_report.json").write_text(
                json.dumps({"formal_collection_allowed": True}),
                encoding="utf-8",
            )
            (schedule_root / "execution_schedule.json").write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )
            orphan_lane = (
                root
                / "collection"
                / "order_0"
                / controllers[0]
            )
            orphan_lane.mkdir(parents=True)
            (orphan_lane / "qualification_manifest.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection"
            ) as run:
                with self.assertRaisesRegex(
                    ValueError, "manifest without its run identity"
                ):
                    collect_scheduled(
                        dataset=root / "dataset",
                        config=root / "config.json",
                        qualification=root / "qualification",
                        schedule_root=schedule_root,
                        output=root / "collection",
                        original_bundle=root / "original",
                        mixed_bundle=root / "mixed",
                        resume=True,
                    )
            run.assert_not_called()

    def test_corrected_qualification_uses_frozen_keys_and_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule_root = root / "schedule"
            schedule_root.mkdir()
            entries = [
                {
                    "task_id": f"task-{index:02d}",
                    "solver_seed": index % 3,
                }
                for index in range(36)
            ]
            (schedule_root / "execution_schedule.json").write_text(
                json.dumps({"entries": entries}), encoding="utf-8"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "formal": True,
                        "max_decisions": 0,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "episode_process_timeout_seconds": 660.0,
                        "environment": {
                            "max_repair_iterations": 0,
                            "time_limit": 600.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            def qualify_side_effect(**kwargs):
                output = Path(kwargs["output"])
                output.mkdir(parents=True)
                report = {
                    "passed": True,
                    "valid_count": 36,
                    "expected_reset_count": 36,
                    "errors": [],
                    "incomplete_reset_count": 0,
                    "inconsistent_initial_state_count": 0,
                }
                (output / "qualification_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                (output / "qualification_manifest.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
                (output / "run_config.json").write_text(
                    "{}", encoding="utf-8"
                )
                return {"qualification": report}

            with mock.patch(
                "experiments.balanced_wall_clock.run_closed_loop_collection",
                side_effect=qualify_side_effect,
            ) as run:
                report = qualify_corrected_native_schedule(
                    dataset=root / "dataset",
                    config=config,
                    source_schedule_root=schedule_root,
                    output=root / "qualification",
                    original_bundle=root / "bundle",
                    workers=2,
                )
            self.assertEqual(report["job_count"], 36)
            self.assertEqual(report["metric_iteration_budget"], 100)
            self.assertEqual(run.call_args.kwargs["phase"], "qualify")
            self.assertEqual(
                run.call_args.kwargs["stopping_rule"],
                "wall-clock-fixed-metric",
            )
            self.assertEqual(len(run.call_args.kwargs["job_keys"]), 36)
            invalid_config = root / "invalid-config.json"
            invalid_config.write_text(
                json.dumps(
                    {
                        "formal": True,
                        "max_decisions": False,
                        "metric_iteration_budget": 100,
                        "wall_time_budget_seconds": 600.0,
                        "episode_process_timeout_seconds": 660.0,
                        "environment": {
                            "max_repair_iterations": 0,
                            "time_limit": 600.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "remove both 100-step"):
                qualify_corrected_native_schedule(
                    dataset=root / "dataset",
                    config=invalid_config,
                    source_schedule_root=schedule_root,
                    output=root / "invalid-qualification",
                    original_bundle=root / "bundle",
                )

    def test_corrected_cli_phases_use_only_corrected_v6_paths(self) -> None:
        from scripts import run_balanced_wall_clock as cli

        with mock.patch(
            "sys.argv",
            ["run_balanced_wall_clock.py", "qualify-corrected-schedule"],
        ), mock.patch.object(
            cli,
            "qualify_corrected_native_schedule",
            return_value={"status": "ok"},
        ) as qualify:
            self.assertEqual(cli.main(), 0)
        qualification_arguments = qualify.call_args.kwargs
        self.assertEqual(
            qualification_arguments["dataset"],
            (
                cli.PROJECT_ROOT
                / "build/initlns-v2-mixed-qualified-compute-load-formal-dataset-v6"
            ).resolve(),
        )
        self.assertEqual(
            qualification_arguments["config"],
            (
                cli.PROJECT_ROOT
                / "configs/balanced_wall_clock_corrected_native_v1.json"
            ).resolve(),
        )

        with mock.patch(
            "sys.argv",
            ["run_balanced_wall_clock.py", "collect-corrected"],
        ), mock.patch.object(
            cli, "collect_scheduled", return_value={"status": "ok"}
        ) as collect:
            self.assertEqual(cli.main(), 0)
        collection_arguments = collect.call_args.kwargs
        self.assertEqual(
            collection_arguments["schedule_root"],
            (
                cli.PROJECT_ROOT
                / "build/initlns-v2-corrected-native-seed-reselected-cohort-v1"
            ).resolve(),
        )
        self.assertEqual(
            collection_arguments["qualification"],
            (
                cli.PROJECT_ROOT
                / "build/initlns-v2-corrected-native-seed-pool-qualification-v1"
            ).resolve(),
        )
        self.assertEqual(
            collection_arguments["output"],
            (
                cli.PROJECT_ROOT
                / "build/initlns-v2-corrected-native-collection-v1"
            ).resolve(),
        )
        self.assertEqual(
            collection_arguments["stopping_rule"],
            "wall-clock-fixed-metric",
        )

        extension = json.dumps(
            {
                "id": "high-high-v1",
                "dataset": "build/extension-dataset",
                "qualification": "build/extension-qualification",
                "config": "configs/extension.json",
            }
        )
        with mock.patch(
            "sys.argv",
            [
                "run_balanced_wall_clock.py",
                "select-corrected-full-pool",
                "--corrected-extension",
                extension,
            ],
        ), mock.patch.object(
            cli,
            "select_corrected_native_full_pool_schedule",
            return_value={"status": "ok"},
        ) as select_full:
            self.assertEqual(cli.main(), 0)
        extension_spec = select_full.call_args.kwargs["extensions"][0]
        self.assertEqual(extension_spec["id"], "high-high-v1")
        self.assertEqual(
            extension_spec["dataset"],
            str(
                (
                    cli.PROJECT_ROOT / "build/extension-dataset"
                ).resolve()
            ),
        )

        with mock.patch(
            "sys.argv",
            [
                "run_balanced_wall_clock.py",
                "materialize-corrected-full-pool",
                "--corrected-extension",
                extension,
            ],
        ), mock.patch.object(
            cli,
            "materialize_corrected_native_selected_dataset",
            return_value={"status": "ok"},
        ) as materialize:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(
            materialize.call_args.kwargs["extensions"][0]["id"],
            "high-high-v1",
        )

        with mock.patch(
            "sys.argv",
            [
                "run_balanced_wall_clock.py",
                "prepare-corrected-full-config",
            ],
        ), mock.patch.object(
            cli,
            "prepare_corrected_native_formal_config",
            return_value={"status": "ok"},
        ) as prepare:
            self.assertEqual(cli.main(), 0)
        formal_root = (
            cli.PROJECT_ROOT
            / "build/initlns-v2-corrected-native-selected-formal-v3"
        ).resolve()
        self.assertEqual(
            prepare.call_args.kwargs["dataset"],
            formal_root / "dataset",
        )

        with mock.patch(
            "sys.argv",
            [
                "run_balanced_wall_clock.py",
                "collect-corrected-full-pool",
            ],
        ), mock.patch.object(
            cli, "collect_scheduled", return_value={"status": "ok"}
        ) as collect_full:
            self.assertEqual(cli.main(), 0)
        self.assertIsNone(
            collect_full.call_args.kwargs["qualification"]
        )
        self.assertEqual(
            collect_full.call_args.kwargs["schedule_root"],
            formal_root,
        )

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
