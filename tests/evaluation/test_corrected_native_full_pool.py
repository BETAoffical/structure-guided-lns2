from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import experiments.corrected_native_full_pool as full_pool
from experiments.corrected_native_full_pool import (
    CONTROLLERS,
    REPORT_SCHEMA,
    SAFETY_MAX_DECISIONS,
    SCHEDULE_SCHEMA,
    _assert_extension_disjoint,
    _candidate,
    _fingerprint,
    _min_cost_assignment,
    _native_identity,
    _seal_selection_evidence,
    _selection_products,
    _selector_identity,
    _validated_selection_evidence_bundle,
    _validate_run_config,
    materialize_corrected_native_selected_dataset,
    validate_corrected_native_full_pool_schedule,
)
from experiments._common import NATIVE_SEMANTICS_SCHEMA, sha256_file
from experiments.repair_collection import _dataset_fingerprint


def _slot(
    group: int,
    *,
    task: str,
    map_id: str,
    source: str = "generated",
    conflict: str = "low",
    load: str = "low",
    seed: int = 1,
) -> dict:
    return {
        "task_id": task,
        "map_id": map_id,
        "solver_seed": seed,
        "source_group": source,
        "conflict_stratum": conflict,
        "initial_pp_load_stratum": load,
        "initial_conflicts": 5,
        "initial_low_level_generated": 50,
        "schedule_group": group,
        "controller_order": [
            "official_adaptive",
            "v2-full",
            "mixed-full-v2",
        ],
    }


def _selection_candidate(
    task: str,
    map_id: str,
    *,
    source: str = "generated",
    conflict: str = "low",
    load: str = "low",
    seed: int = 1,
    conflicts: int = 5,
    generated: int = 50,
) -> dict:
    return {
        "task_id": task,
        "map_id": map_id,
        "solver_seed": seed,
        "source_group": source,
        "conflict_stratum": conflict,
        "initial_pp_load_stratum": load,
        "initial_conflicts": conflicts,
        "initial_low_level_generated": generated,
        "state_fingerprint": f"{seed:064x}",
        "_initial_feasible": False,
    }


def _formal_rows() -> list[dict]:
    rows = []
    orders = list(itertools.permutations(CONTROLLERS))
    conflicts = {"low": 5, "medium": 50, "high": 200}
    generated = {"low": 50, "medium": 200_000, "high": 2_000_000}
    index = 0
    for conflict in ("low", "medium", "high"):
        for load in ("low", "medium", "high"):
            for _position in range(4):
                source = "generated" if index % 2 == 0 else "movingai"
                task_id = f"task-{index:02d}"
                rows.append(
                    {
                        "task_id": task_id,
                        "map_id": f"map-{index:02d}",
                        "solver_seed": index % 3 + 1,
                        "layout_mode": f"layout-{index % 2}",
                        "source_group": source,
                        "agent_count": 100,
                        "agent_band": "small",
                        "initial_conflicts": conflicts[conflict],
                        "conflict_stratum": conflict,
                        "initial_pp_load_stratum": load,
                        "initial_low_level_generated": generated[load],
                        "initial_low_level_expanded": generated[load] // 2,
                        "conflict_event_count": conflicts[conflict],
                        "total_path_cost": 1000 + index,
                        "active_conflict_agent_ratio": 0.2,
                        "largest_conflict_component_ratio": 0.1,
                        "state_fingerprint": f"{index + 1:064x}",
                        "schedule_group": index % 6,
                        "controller_order": list(orders[index % 6]),
                    }
                )
                index += 1
    return rows


def _producer_identity() -> dict:
    return {
        "schema": "lns2.producer_identity.v2",
        "source_sha256": {
            "experiments/closed_loop_confirmation.py": "a" * 64,
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
            "native_semantics_schema": NATIVE_SEMANTICS_SCHEMA,
            "repair_timing_schema": "lns2.repair_timing.v2",
        },
    }


def _write_schedule(
    root: Path,
    *,
    dataset_manifest_sha256: str,
    dataset_fingerprint: str,
) -> None:
    entries = _formal_rows()
    identity = _producer_identity()
    native = {
        key: identity["native"][key]
        for key in (
            "sha256",
            "native_semantics_schema",
            "repair_timing_schema",
        )
    }
    producer = _fingerprint(identity)
    provenance = {
        "schema": "lns2.corrected_native_full_pool_provenance.v3",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "extensions": [],
        "native": native,
        "native_semantics_schema": NATIVE_SEMANTICS_SCHEMA,
        "qualification_producer_identity": identity,
        "qualification_producer_identity_fingerprint": producer,
        "qualification_producer_identities": [
            {
                "id": "base-full-pool",
                "fingerprint": producer,
                "identity": identity,
                "native": native,
            }
        ],
        "selector_identity": _selector_identity(),
        "stopping_contract": {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 600.0,
            "environment_time_limit_seconds": 600.0,
            "episode_process_timeout_seconds": 660.0,
            "safety_max_decisions": SAFETY_MAX_DECISIONS,
            "safety_limit_is_not_metric_cap": True,
        },
    }
    schedule = {
        "schema": SCHEDULE_SCHEMA,
        "selection_blind_to_controller_outcomes": True,
        "selection_uses_initial_reset_metrics_only": True,
        "provenance": provenance,
        "entries": entries,
    }
    root.mkdir()
    schedule_path = root / "execution_schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    cells = {}
    for conflict in ("low", "medium", "high"):
        for load in ("low", "medium", "high"):
            for source in ("generated", "movingai"):
                cells[f"{conflict}__{load}__{source}"] = sum(
                    row["conflict_stratum"] == conflict
                    and row["initial_pp_load_stratum"] == load
                    and row["source_group"] == source
                    for row in entries
                )
    assignments = [
        {
            "schedule_group": row["schedule_group"],
            "source_task_id": row["task_id"],
            "selected_task_id": row["task_id"],
            "source_map_id": row["map_id"],
            "selected_map_id": row["map_id"],
            "source_solver_seed": row["solver_seed"],
            "selected_solver_seed": row["solver_seed"],
            "original_task_retained": True,
            "original_map_retained": True,
            "original_seed_retained": True,
            "relative_covariate_drift": 0.0,
            "relative_covariate_drift_exact": {
                "numerator": 0,
                "denominator": 1,
            },
            "salted_tie_sha256": f"{100 + index:064x}",
        }
        for index, row in enumerate(entries)
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "formal_collection_allowed": True,
        "selected_count": 36,
        "execution_schedule_sha256": sha256_file(schedule_path),
        "provenance": provenance,
        "gates": {"all": True},
        "slot_assignments": assignments,
        "counts": {
            "cell_source": cells,
            "source": {"generated": 18, "movingai": 18},
            "tasks": 36,
            "maps": 36,
        },
    }
    (root / "cohort_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def _qualification_candidates() -> list[dict]:
    candidates = []
    for row in _formal_rows():
        for seed in (1, 2, 3):
            candidate = {
                key: value
                for key, value in row.items()
                if key not in {"schedule_group", "controller_order"}
            }
            candidate["solver_seed"] = seed
            if seed != row["solver_seed"]:
                candidate["state_fingerprint"] = _fingerprint(
                    [row["task_id"], seed]
                )
            candidate["_initial_feasible"] = False
            candidates.append(candidate)
    return candidates


def _rebase_schedule_report(schedule_root: Path) -> None:
    schedule_path = schedule_root / "execution_schedule.json"
    report_path = schedule_root / "cohort_report.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"] = schedule["provenance"]
    report["execution_schedule_sha256"] = sha256_file(schedule_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _attach_stub_selection_evidence(
    schedule_root: Path,
    project_root: Path,
) -> None:
    input_path = project_root / "sealed-input.json"
    input_path.write_text('{"sealed":true}\n', encoding="utf-8")
    registration = _seal_selection_evidence(
        schedule_root,
        project_root=project_root,
        artifacts={"stub": input_path},
        descriptor={"test_fixture": True},
    )
    schedule_path = schedule_root / "execution_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["provenance"]["selection_evidence"] = registration
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    _rebase_schedule_report(schedule_root)


@contextmanager
def _mocked_complete_replay(schedule_root: Path):
    schedule_path = schedule_root / "execution_schedule.json"
    report_path = schedule_root / "cohort_report.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    slots = [
        {**row, "_source_slot_index": index}
        for index, row in enumerate(_formal_rows())
    ]
    candidates = _qualification_candidates()
    products = _selection_products(slots, candidates)
    assert products is not None
    schedule["entries"] = products["entries"]
    identity = schedule["provenance"]["qualification_producer_identity"]
    producer = _fingerprint(identity)
    native = schedule["provenance"]["native"]
    qualification_evidence = {
        "qualification_manifest_sha256": "1" * 64,
        "qualification_report_sha256": "2" * 64,
        "qualification_run_config_sha256": "3" * 64,
        "qualification_producer_identity": identity,
        "qualification_producer_identity_fingerprint": producer,
    }
    provenance = schedule["provenance"]
    provenance.update(qualification_evidence)
    provenance.update(
        {
            "extension_count": 0,
            "merged_task_count": 36,
            "merged_map_count": 36,
            "merged_qualification_count": len(candidates),
            "selection": {
                "schema": full_pool.SELECTOR_SCHEMA,
                "objective": [
                    "maximize_original_task_retention",
                    "maximize_original_map_retention",
                    "maximize_original_seed_retention",
                    "minimize_total_relative_conflict_and_pp_load_drift",
                    (
                        "minimize_additive_fixed_salt_sha256:"
                        f"{full_pool.SELECTION_SALT}"
                    ),
                ],
                "constraints": {
                    "jobs_per_conflict_load_cell": 4,
                    "cell_source_quotas": "exactly_preserve_v6",
                    "one_task_per_map": True,
                    "map_count": 36,
                    "source_counts": {
                        "generated": 18,
                        "movingai": 18,
                    },
                },
                "controller_outcomes_used": False,
            },
        }
    )
    total_cost = products["total_cost"]
    drift = total_cost[3]
    report.update(
        {
            "qualification_count": len(candidates),
            "base_qualification_count": len(candidates),
            "extension_qualification_count": 0,
            "selected_count": 36,
            "original_task_retained_count": -total_cost[0],
            "original_map_retained_count": -total_cost[1],
            "original_seed_retained_count": -total_cost[2],
            "total_relative_covariate_drift": float(drift),
            "total_relative_covariate_drift_exact": {
                "numerator": drift.numerator,
                "denominator": drift.denominator,
            },
            "salted_sha256_additive_cost": str(total_cost[4]),
            "slot_assignments": products["audit_rows"],
            "gates": products["gates"],
        }
    )
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    report["provenance"] = provenance
    report["execution_schedule_sha256"] = sha256_file(schedule_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    qualification_root = schedule_root / "mock" / "qualification"
    role_paths = {
        "source_registration": schedule_root / "mock/source-registration.json",
        "source_pool_registry": schedule_root / "mock/source-registry.json",
        "source_selection_config": schedule_root / "mock/source-config.json",
        "source_merged_dataset_manifest": schedule_root / "mock/source-dataset.jsonl",
        "source_merged_qualification_manifest": schedule_root / "mock/source-qualification.jsonl",
        "source_cohort": schedule_root / "mock/source-cohort.json",
        "source_execution_schedule": schedule_root / "mock/source-schedule.json",
        "source_cohort_report": schedule_root / "mock/source-report.json",
        "source_formal_dataset_manifest": schedule_root / "mock/source-formal.jsonl",
        "base_config": schedule_root / "mock/base-config.json",
        "base_pool_registry": schedule_root / "mock/base-registry.json",
        "base_dataset_manifest": schedule_root / "mock/base-dataset.jsonl",
        "base_dataset_summary": schedule_root / "mock/base-summary.json",
        "base_qualification_manifest": (
            qualification_root / "qualification_manifest.jsonl"
        ),
        "base_qualification_report": (
            qualification_root / "qualification_report.json"
        ),
        "base_qualification_run_config": qualification_root / "run_config.json",
    }
    manifest = {
        "roles": {role: role for role in role_paths},
        "descriptor": {
            "base_dataset_fingerprint": provenance["dataset_fingerprint"],
            "base_task_count": 36,
            "base_qualification_count": len(candidates),
            "extension_ids": [],
            "merged_task_count": 36,
            "merged_qualification_count": len(candidates),
        },
    }
    base_rows = [
        {"task_id": row["task_id"], "map_id": row["map_id"]}
        for row in _formal_rows()
    ]

    def bound_artifact(*_args, **kwargs):
        return role_paths[kwargs["role"]]

    with (
        mock.patch.object(
            full_pool,
            "_validated_selection_evidence_bundle",
            return_value=(manifest, role_paths),
        ),
        mock.patch.object(
            full_pool,
            "_bound_evidence_artifact",
            side_effect=bound_artifact,
        ),
        mock.patch.object(
            full_pool,
            "_validate_original_design",
            return_value=(slots, {}, {}),
        ),
        mock.patch.object(
            full_pool,
            "_validate_config",
            return_value=(
                {},
                role_paths["base_pool_registry"],
                {},
            ),
        ),
        mock.patch.object(
            full_pool,
            "_validate_dataset_snapshot",
            return_value=base_rows,
        ),
        mock.patch.object(
            full_pool,
            "_validate_qualification",
            return_value=(
                candidates,
                {},
                native,
                dict(qualification_evidence),
            ),
        ),
    ):
        yield products


class CorrectedNativeFullPoolTests(unittest.TestCase):
    def test_qualification_run_requires_structured_identity_and_exact_payload(
        self,
    ) -> None:
        producer = _producer_identity()
        producer_fingerprint = _fingerprint(producer)
        effective = {
            "formal": True,
            "stopping_rule": "wall-clock-fixed-metric",
            "task_ids_override": None,
            "cohort_job_keys_override": None,
        }
        dataset_fingerprint = "d" * 64
        run_config = {
            "schema": "lns2.closed_loop_confirmation.v1",
            "schema_version": 1,
            "formal": True,
            "controller": "v2-full",
            "dataset_fingerprint": dataset_fingerprint,
            "configuration": effective,
            "configuration_fingerprint": _fingerprint(effective),
            "frozen_models": None,
            "controller_bundle": None,
            "stall_shadow_config": None,
            "critical_seed_config": None,
            "cost_top3_config": None,
            "v3_bundle": None,
            "v3_s3_bundle": None,
            "producer_identity": producer,
            "producer_identity_fingerprint": producer_fingerprint,
        }
        run_config["run_fingerprint"] = _fingerprint(
            {
                "dataset_fingerprint": dataset_fingerprint,
                "configuration_fingerprint": _fingerprint(effective),
                "freeze_manifest": None,
                "controller_bundle_manifest": None,
                "stall_shadow_config": None,
                "repair_aware_bundle_manifest": None,
                "critical_seed_config": None,
                "cost_top3_config": None,
                "v3_bundle_manifest": None,
                "v3_s3_bundle_manifest": None,
                "producer_identity": producer,
                "producer_identity_fingerprint": producer_fingerprint,
            }
        )
        validated, native = _validate_run_config(
            run_config,
            {"formal": True},
            dataset_fingerprint,
        )
        self.assertEqual(validated, producer)
        self.assertEqual(native["sha256"], "b" * 64)

        legacy = dict(run_config)
        legacy.pop("producer_identity")
        legacy.pop("producer_identity_fingerprint")
        legacy["controller_implementation"] = {"files": {}}
        with self.assertRaisesRegex(ValueError, "legacy controller_implementation"):
            _validate_run_config(
                legacy,
                {"formal": True},
                dataset_fingerprint,
            )

        for field in ("source_sha256", "packages"):
            with self.subTest(field=field):
                tampered = json.loads(json.dumps(run_config))
                if field == "source_sha256":
                    tampered["producer_identity"][field][
                        "experiments/closed_loop_confirmation.py"
                    ] = "e" * 64
                else:
                    tampered["producer_identity"][field]["numpy"] = "0.0"
                with self.assertRaisesRegex(
                    ValueError, "producer fingerprint is invalid"
                ):
                    _validate_run_config(
                        tampered,
                        {"formal": True},
                        dataset_fingerprint,
                    )

    def test_lexicographic_task_retention_precedes_covariate_drift(self) -> None:
        slots = [_slot(0, task="old-task", map_id="old-map")]
        selected = _min_cost_assignment(
            slots,
            [
                _selection_candidate(
                    "old-task",
                    "old-map",
                    conflicts=500,
                    generated=2_000_000,
                ),
                _selection_candidate("new-task", "new-map"),
            ],
        )
        self.assertIsNotNone(selected)
        assignments, cost = selected  # type: ignore[misc]
        self.assertEqual(assignments[0][1]["task_id"], "old-task")
        self.assertEqual(cost[0], -1)

    def test_map_capacity_prevents_two_tasks_from_same_map(self) -> None:
        slots = [
            _slot(0, task="old-a", map_id="old-a"),
            _slot(1, task="old-b", map_id="old-b"),
        ]
        selected = _min_cost_assignment(
            slots,
            [
                _selection_candidate("a", "shared"),
                _selection_candidate("b", "shared", seed=2),
                _selection_candidate("c", "other"),
            ],
        )
        self.assertIsNotNone(selected)
        assignments, _cost = selected  # type: ignore[misc]
        self.assertEqual(
            len({assignment[1]["map_id"] for assignment in assignments}),
            2,
        )

    def test_unsatisfied_slot_returns_none(self) -> None:
        slots = [
            _slot(0, task="a", map_id="a", source="generated"),
            _slot(1, task="b", map_id="b", source="movingai"),
        ]
        self.assertIsNone(
            _min_cost_assignment(
                slots,
                [_selection_candidate("only", "map", source="generated")],
            )
        )

    def test_extension_candidate_can_fill_missing_registered_slot(self) -> None:
        slots = [
            _slot(0, task="old-a", map_id="old-a"),
            _slot(
                1,
                task="old-b",
                map_id="old-b",
                conflict="high",
                load="high",
            ),
        ]
        base = [_selection_candidate("base", "base-map")]
        self.assertIsNone(_min_cost_assignment(slots, base))
        extension = _selection_candidate(
            "extension",
            "extension-map",
            conflict="high",
            load="high",
            conflicts=200,
            generated=2_000_000,
        )
        selected = _min_cost_assignment(slots, [*base, extension])
        self.assertIsNotNone(selected)
        assignments, _cost = selected  # type: ignore[misc]
        self.assertEqual(
            {row[1]["map_id"] for row in assignments},
            {"base-map", "extension-map"},
        )

    def test_extension_rejects_repeated_task_map_or_file_identity(self) -> None:
        shared_file = {("map_file", "a" * 64)}
        cases = (
            ({"task"}, {"new-map"}, set(), "task identity"),
            ({"new-task"}, {"map"}, set(), "map identity"),
            ({"new-task"}, {"new-map"}, shared_file, "file identity"),
        )
        for tasks, maps, files, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _assert_extension_disjoint(
                        "extension",
                        existing_task_ids={"task"},
                        existing_map_ids={"map"},
                        existing_file_identities=shared_file,
                        extension_task_ids=tasks,
                        extension_map_ids=maps,
                        extension_file_identities=files,
                    )

    def test_candidate_rejects_controller_outcome_fields(self) -> None:
        task = {
            "task_id": "task",
            "map_id": "map",
            "layout_mode": "layout",
            "source_group": "generated",
            "agent_count": 10,
        }
        result = {
            "schema": "lns2.repair_collection.v2",
            "schema_version": 2,
            "status": "ok",
            "error": None,
            "task_id": "task",
            "solver_seed": 1,
            "map_id": "map",
            "layout_mode": "layout",
            "source_group": "generated",
            "agent_count": 10,
            "split": "balanced_wall_clock",
            "initial_complete": True,
            "initial_feasible": False,
            "repairable": True,
            "initial_conflicts": 1,
            "state_fingerprint": "a" * 64,
            "initial_complexity": {
                "conflict_pair_count": 1,
                "initial_low_level_generated": 10,
                "initial_low_level_expanded": 8,
                "conflict_event_count": 1,
                "total_path_cost": 20,
                "active_conflict_agent_ratio": 0.2,
                "largest_conflict_component_ratio": 0.2,
            },
            "final_conflicts": 0,
        }
        with self.assertRaisesRegex(ValueError, "controller outcomes"):
            _candidate(result, task)

    def test_native_identity_rejects_wrong_semantics(self) -> None:
        producer = _producer_identity()
        producer["native"]["native_semantics_schema"] = "lns2.legacy.v0"
        with self.assertRaisesRegex(ValueError, "producer identity"):
            _native_identity(producer, label="test")

    def test_schedule_rejects_legacy_or_mutated_producer_identity(self) -> None:
        mutations = {
            "legacy": lambda identity: {
                "files": identity["source_sha256"],
                "native_module": identity["native"],
            },
            "source SHA": lambda identity: {
                **identity,
                "source_sha256": {
                    **identity["source_sha256"],
                    "experiments/closed_loop_confirmation.py": "f" * 64,
                },
            },
            "package": lambda identity: {
                **identity,
                "packages": {
                    **identity["packages"],
                    "numpy": "0.0-tampered",
                },
            },
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                schedule_root = root / "schedule"
                _write_schedule(
                    schedule_root,
                    dataset_manifest_sha256="d" * 64,
                    dataset_fingerprint="e" * 64,
                )
                schedule_path = schedule_root / "execution_schedule.json"
                schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
                identity = mutate(
                    schedule["provenance"][
                        "qualification_producer_identity"
                    ]
                )
                schedule["provenance"][
                    "qualification_producer_identity"
                ] = identity
                schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
                report_path = schedule_root / "cohort_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["provenance"] = schedule["provenance"]
                report["execution_schedule_sha256"] = sha256_file(schedule_path)
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_corrected_native_full_pool_schedule(schedule_root)

    def test_schedule_rejects_selector_dependency_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_root = root / "schedule"
            _write_schedule(
                schedule_root,
                dataset_manifest_sha256="d" * 64,
                dataset_fingerprint="e" * 64,
            )
            schedule_path = schedule_root / "execution_schedule.json"
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            selector = schedule["provenance"]["selector_identity"]
            selector["producer_identity"]["source_sha256"][
                "experiments/_common.py"
            ] = "0" * 64
            selector["producer_identity_fingerprint"] = _fingerprint(
                selector["producer_identity"]
            )
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            report_path = schedule_root / "cohort_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["provenance"] = schedule["provenance"]
            report["execution_schedule_sha256"] = sha256_file(schedule_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selector identity"):
                validate_corrected_native_full_pool_schedule(schedule_root)

    def test_strict_schedule_validator_rejects_semantic_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_root = root / "schedule"
            _write_schedule(
                schedule_root,
                dataset_manifest_sha256="d" * 64,
                dataset_fingerprint="e" * 64,
            )
            with _mocked_complete_replay(schedule_root):
                self.assertTrue(
                    validate_corrected_native_full_pool_schedule(schedule_root)[
                        "passed"
                    ]
                )
                schedule_path = schedule_root / "execution_schedule.json"
                schedule = json.loads(
                    schedule_path.read_text(encoding="utf-8")
                )
                schedule["entries"][0]["conflict_stratum"] = "medium"
                schedule_path.write_text(
                    json.dumps(schedule), encoding="utf-8"
                )
                _rebase_schedule_report(schedule_root)
                with self.assertRaisesRegex(ValueError, "difficulty strata"):
                    validate_corrected_native_full_pool_schedule(schedule_root)

    def test_sealed_replay_rejects_rebased_selection_tamper(self) -> None:
        cases = ("solver_seed", "state", "cell", "source", "slot")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                schedule_root = root / "schedule"
                _write_schedule(
                    schedule_root,
                    dataset_manifest_sha256="d" * 64,
                    dataset_fingerprint="e" * 64,
                )
                with _mocked_complete_replay(schedule_root):
                    self.assertTrue(
                        validate_corrected_native_full_pool_schedule(
                            schedule_root
                        )["passed"]
                    )
                    schedule_path = schedule_root / "execution_schedule.json"
                    report_path = schedule_root / "cohort_report.json"
                    schedule = json.loads(
                        schedule_path.read_text(encoding="utf-8")
                    )
                    report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    entries = schedule["entries"]
                    if case == "solver_seed":
                        entries[0]["solver_seed"] = 99
                        report["slot_assignments"][0][
                            "selected_solver_seed"
                        ] = 99
                    elif case == "state":
                        entries[0]["state_fingerprint"] = "f" * 64
                    elif case == "cell":
                        for field in (
                            "initial_low_level_generated",
                            "initial_low_level_expanded",
                            "initial_pp_load_stratum",
                        ):
                            entries[0][field], entries[4][field] = (
                                entries[4][field],
                                entries[0][field],
                            )
                    elif case == "source":
                        entries[0]["source_group"], entries[1]["source_group"] = (
                            entries[1]["source_group"],
                            entries[0]["source_group"],
                        )
                    else:
                        for field in ("schedule_group", "controller_order"):
                            entries[0][field], entries[1][field] = (
                                entries[1][field],
                                entries[0][field],
                            )
                        by_task = {
                            row["selected_task_id"]: row
                            for row in report["slot_assignments"]
                        }
                        for entry in entries[:2]:
                            by_task[entry["task_id"]]["schedule_group"] = entry[
                                "schedule_group"
                            ]
                    schedule_path.write_text(
                        json.dumps(schedule), encoding="utf-8"
                    )
                    report["provenance"] = schedule["provenance"]
                    report["execution_schedule_sha256"] = sha256_file(
                        schedule_path
                    )
                    report_path.write_text(
                        json.dumps(report), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "schedule entries differ from sealed reset-row replay",
                    ):
                        validate_corrected_native_full_pool_schedule(
                            schedule_root
                        )

    def test_selection_evidence_rejects_byte_and_manifest_tamper(self) -> None:
        for case in ("sealed_input", "manifest"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                schedule_root = root / "schedule"
                schedule_root.mkdir()
                source = root / "source.json"
                source.write_text('{"value":1}\n', encoding="utf-8")
                registration = _seal_selection_evidence(
                    schedule_root,
                    project_root=root,
                    artifacts={"source": source},
                    descriptor={"test": True},
                )
                provenance = {"selection_evidence": registration}
                _validated_selection_evidence_bundle(
                    schedule_root,
                    provenance,
                )
                if case == "sealed_input":
                    sealed = (
                        schedule_root
                        / "selection_evidence"
                        / "project"
                        / "source.json"
                    )
                    sealed.write_text('{"value":2}\n', encoding="utf-8")
                    message = "project file set or hash differs"
                else:
                    manifest_path = (
                        schedule_root / full_pool.EVIDENCE_MANIFEST_PATH
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["descriptor"]["test"] = False
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    message = "manifest hash differs"
                with self.assertRaisesRegex(ValueError, message):
                    _validated_selection_evidence_bundle(
                        schedule_root,
                        provenance,
                    )

    def test_materializer_builds_and_binds_formal_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "base"
            split = dataset / "balanced_wall_clock"
            split.mkdir(parents=True)
            manifest_rows = []
            for entry in _formal_rows():
                index = int(entry["task_id"].split("-")[1])
                row = {
                    "task_id": entry["task_id"],
                    "map_id": entry["map_id"],
                    "split": "balanced_wall_clock",
                    "layout_mode": entry["layout_mode"],
                    "source_group": entry["source_group"],
                    "agent_count": entry["agent_count"],
                    "task_variant": f"variant-{index}",
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
                        f"{field}:{index}\n",
                        encoding="utf-8",
                    )
                manifest_rows.append(row)
            manifest = split / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )
            (dataset / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "splits": {
                            "balanced_wall_clock": {
                                "map_count": 36,
                                "instance_count": 36,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            schedule_root = root / "selection"
            _write_schedule(
                schedule_root,
                dataset_manifest_sha256=sha256_file(manifest),
                dataset_fingerprint=_dataset_fingerprint(dataset),
            )
            _attach_stub_selection_evidence(schedule_root, root)
            output = root / "formal"
            with _mocked_complete_replay(schedule_root):
                result = materialize_corrected_native_selected_dataset(
                    schedule_root=schedule_root,
                    base_dataset=dataset,
                    output=output,
                )
                self.assertTrue(result["passed"])
                verified = validate_corrected_native_full_pool_schedule(output)
            self.assertEqual(
                verified["formal_dataset_fingerprint"],
                result["dataset_fingerprint"],
            )
            self.assertEqual(
                len(
                    (output / "dataset" / "balanced_wall_clock" / "manifest.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                36,
            )
            formal_manifest = (
                output / "dataset" / "balanced_wall_clock" / "manifest.jsonl"
            )
            rows = [
                json.loads(line)
                for line in formal_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            rows[0]["agent_count"] += 1
            formal_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            materialization_path = output / "materialization_report.json"
            materialization = json.loads(
                materialization_path.read_text(encoding="utf-8")
            )
            materialization["manifest_sha256"] = sha256_file(
                formal_manifest
            )
            materialization["dataset_fingerprint"] = _dataset_fingerprint(
                output / "dataset"
            )
            materialization_path.write_text(
                json.dumps(materialization),
                encoding="utf-8",
            )
            schedule_path = output / "execution_schedule.json"
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            schedule["provenance"]["formal_dataset_manifest_sha256"] = (
                sha256_file(formal_manifest)
            )
            schedule["provenance"]["formal_dataset_fingerprint"] = (
                _dataset_fingerprint(output / "dataset")
            )
            schedule["provenance"][
                "formal_materialization_report_sha256"
            ] = sha256_file(materialization_path)
            schedule_path.write_text(
                json.dumps(schedule),
                encoding="utf-8",
            )
            report_path = output / "cohort_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["provenance"] = schedule["provenance"]
            report["execution_schedule_sha256"] = sha256_file(schedule_path)
            report_path.write_text(
                json.dumps(report),
                encoding="utf-8",
            )
            with _mocked_complete_replay(output):
                with self.assertRaisesRegex(
                    ValueError,
                    "metadata differs from the schedule",
                ):
                    validate_corrected_native_full_pool_schedule(output)


if __name__ == "__main__":
    unittest.main()
