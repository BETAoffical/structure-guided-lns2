from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _four_neighbor_distances,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _CollectionRunLock,
    _fingerprint,
    _qualification_worker,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import prepare_resumable_output
from experiments.warehouse_crossaisle_tasks import (
    CROSS_AISLE_GROUP_COUNT,
    MATCHED_VARIANT,
    STRUCTURED_VARIANT,
    SUPPORTED_Q,
    TASK_PAIR_SCHEMA,
    TASK_VARIANTS,
    audit_paired_crossaisle_tasks,
    crossaisle_task_specs,
    detect_cross_aisle_geometry,
    generate_paired_crossaisle_tasks,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_crossaisle_paired_config.v1"
RUNTIME_SCHEMA = 1
REGISTRY_SCHEMA = "lns2.stride.warehouse_crossaisle_task_registry.v1"
Q0_SCHEMA = "lns2.stride.warehouse_crossaisle_q0_geometry_audit.v1"
Q0_MANIFEST_SCHEMA = "lns2.stride.warehouse_crossaisle_q0_manifest.v1"
QUALIFICATION_STATUS_SCHEMA = (
    "lns2.stride.warehouse_crossaisle_qualification_status.v1"
)
QUALIFICATION_REPORT_SCHEMA = (
    "lns2.stride.warehouse_crossaisle_qualification_report.v1"
)
SELECTION_SCHEMA = "lns2.stride.warehouse_crossaisle_benchmark_selection.v1"
EXPERIMENT_ID = "stride-warehouse-crossaisle-paired-v1"

TASK_SEEDS = (419, 463)
Q_VALUES = (16, 20)
SOLVER_SEEDS = (61, 62, 63, 64)
MAPS = (
    {
        "id": "warehouse-10-20-10-2-1",
        "code": "w1020a",
        "path": "build/movingai-dev/maps/warehouse-10-20-10-2-1.map",
        "sha256": "c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0",
        "expected_internal_group_count": 19,
        "expected_aisle_width": 1,
        "expected_agent_count_by_q": {"16": 384, "20": 480},
    },
    {
        "id": "warehouse-10-20-10-2-2",
        "code": "w1020b",
        "path": "build/movingai-ood-dev/maps/warehouse-10-20-10-2-2.map",
        "sha256": "4f06e82c2b87238daa8e308086afdba701112bf023e94e740e5bf9508a6adec3",
        "expected_internal_group_count": 19,
        "expected_aisle_width": 2,
        "expected_agent_count_by_q": {"16": 768, "20": 960},
    },
    {
        "id": "warehouse-20-40-10-2-1",
        "code": "w2040a",
        "path": "build/movingai-dev/maps/warehouse-20-40-10-2-1.map",
        "sha256": "bd3bec2d1c20a8bbf900583cb4c2cf9dc16101470fbfae93b272ee0fb575d3ec",
        "expected_internal_group_count": 39,
        "expected_aisle_width": 1,
        "expected_agent_count_by_q": {"16": 384, "20": 480},
    },
    {
        "id": "warehouse-20-40-10-2-2",
        "code": "w2040b",
        "path": "build/movingai-ood-dev/maps/warehouse-20-40-10-2-2.map",
        "sha256": "eae2a3f5298b1e113bfc32b5b4404f5de5105365d237df63cc9dc9e2d1c73713",
        "expected_internal_group_count": 39,
        "expected_aisle_width": 2,
        "expected_agent_count_by_q": {"16": 768, "20": 960},
    },
)
MAP_BY_ID = {str(row["id"]): row for row in MAPS}

Q0_FILENAME = "q0_geometry_audit.json"
Q0_MANIFEST_FILENAME = "q0_manifest.jsonl"
Q0_REGISTRATION_PROPOSAL_FILENAME = "q0_task_registry_proposal.json"
DATASET_SUMMARY_FILENAME = "dataset_summary.json"
QUALIFICATION_STATUS_FILENAME = "qualification_status.json"
QUALIFICATION_REPORT_FILENAME = "qualification_report.json"
QUALIFICATION_MANIFEST_FILENAME = "qualification_manifest.jsonl"
SELECTION_FILENAME = "benchmark_selection.json"

PRODUCER_SOURCE_FILES = (
    "experiments/stride_warehouse_crossaisle_paired.py",
    "experiments/warehouse_crossaisle_tasks.py",
    "experiments/balanced_wall_clock.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/run_output_guard.py",
)


def _resolve_registered_input(
    root: Path, specification: Mapping[str, Any], label: str
) -> Path:
    return registered_input(root, dict(specification), label=label)


def _pending_input(root: Path, specification: Mapping[str, Any], label: str) -> Path:
    path = (root / str(specification.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes project root") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _load_registry(root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    registration = dict(config.get("registration") or {})
    registry_path = _resolve_registered_input(
        root,
        {
            "path": registration.get("task_registry_path"),
            "sha256": registration.get("task_registry_sha256"),
        },
        "Warehouse cross-aisle task registry",
    )
    registry = _read_json(registry_path)
    tasks = dict(registry.get("tasks") or {})
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("experiment_id") != EXPERIMENT_ID
        or int(registry.get("task_count", -1)) != 32
        or len(tasks) != 32
        or len(set(tasks)) != 32
        or not isinstance(registry.get("q0_manifest_sha256"), str)
        or len(str(registry["q0_manifest_sha256"])) != 64
        or str(registry.get("task_generator_sha256"))
        != str(dict(config["inputs"])["task_generator"]["sha256"])
    ):
        raise ValueError("Warehouse cross-aisle task registry identity changed")
    return registry_path, registry


def load_config(
    path: str | Path, *, require_frozen_registry: bool = False
) -> tuple[Path, Path, dict[str, Any], dict[str, Any] | None]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "3f9548b8206fd449f0ea1ce6f897cdd620701da0"
    ):
        raise ValueError("Warehouse cross-aisle benchmark identity changed")
    phase = dict(config.get("phase_contract") or {})
    if phase != {
        "allowed_phases": [
            "q0_geometry_and_task_registration",
            "q1_reset_only",
        ],
        "controller_step_allowed": False,
        "policy_episode_allowed": False,
        "formal_ttf_allowed": False,
        "formal_schedule_allowed": False,
        "q1_requires_frozen_task_registry": True,
    }:
        raise ValueError("Warehouse cross-aisle phase contract changed")
    if dict(config.get("q0_gates") or {}) != {
        "required_chosen_cross_aisle_group_count_per_map": 12,
        "exclude_map_boundary_groups": True,
        "exclude_central_group": True,
        "chosen_group_balance": (
            "six_above_and_six_below_the_excluded_central_group"
        ),
        "unique_starts_required": True,
        "unique_goals_required": True,
        "zero_fixed_points_required": True,
        "all_endpoints_reachable_required": True,
        "same_start_multiset_between_variants_required": True,
        "same_goal_multiset_between_variants_required": True,
        "maximum_relative_mean_shortest_distance_difference": 0.05,
        "maximum_relative_p95_shortest_distance_difference": 0.10,
        "all_32_tasks_required": True,
    }:
        raise ValueError("Warehouse cross-aisle Q0 gates changed")
    if dict(config.get("q1_gates") or {}) != {
        "all_128_resets_required": True,
        "all_initial_solutions_complete_and_consistent": True,
        "structured": {
            "variant": STRUCTURED_VARIANT,
            "minimum_initial_conflicts": 16,
            "minimum_active_conflict_agents": 32,
            "minimum_largest_conflict_component_size": 16,
            "initial_feasible": False,
        },
        "matched": {
            "variant": MATCHED_VARIANT,
            "minimum_initial_conflicts": 1,
            "initial_feasible": False,
        },
        "selection": (
            "per_map_lowest_q_where_both_variants_pass_every_task_seed_"
            "solver_seed_reset"
        ),
        "all_four_maps_must_select_a_q": True,
        "no_map_q_task_variant_or_seed_replacement": True,
    }:
        raise ValueError("Warehouse cross-aisle Q1 gates changed")
    if dict(config.get("result_interpretation") or {}) != {
        "matched_control_label": (
            "same_start_and_goal_multisets_distance_near_matched_secondary_control"
        ),
        "fully_difficulty_matched_control_claim_allowed": False,
        "controller_performance_claim_allowed": False,
        "ttf_claim_allowed": False,
        "benchmark_ready_means": (
            "q0_passed_and_all_four_maps_selected_a_common_q_in_q1_only"
        ),
    }:
        raise ValueError("Warehouse cross-aisle result interpretation changed")
    runtime_contract = dict(config.get("runtime") or {})
    if runtime_contract != {
        "workers": 16,
        "qualification_process_timeout_seconds": 240.0,
        "stop_on_first_execution_error_or_process_timeout": True,
        "resume_only_after_zero_execution_errors_and_zero_process_timeouts": True,
    }:
        raise ValueError("Warehouse cross-aisle reset runtime contract changed")
    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("split") != SPLIT
        or tuple(map(int, cohort.get("task_seeds") or ())) != TASK_SEEDS
        or tuple(map(str, cohort.get("task_variants") or ())) != TASK_VARIANTS
        or tuple(map(int, cohort.get("q_values") or ())) != Q_VALUES
        or tuple(map(int, cohort.get("solver_seeds") or ())) != SOLVER_SEEDS
        or tuple(dict(row) for row in cohort.get("maps") or ()) != MAPS
        or int(cohort.get("expected_task_count", -1)) != 32
        or int(cohort.get("expected_reset_count", -1)) != 128
        or cohort.get("claim_boundary")
        != (
            "four_registered_standard_warehouse_maps_and_two_preregistered_"
            "cross_aisle_task_families_only"
        )
    ):
        raise ValueError("Warehouse cross-aisle cohort changed")
    if TASK_VARIANTS != (STRUCTURED_VARIANT, MATCHED_VARIANT):
        raise ValueError("Warehouse cross-aisle helper variant identity changed")
    if tuple(map(int, SUPPORTED_Q)) != Q_VALUES or int(CROSS_AISLE_GROUP_COUNT) != 12:
        raise ValueError("Warehouse cross-aisle helper load/group identity changed")
    for specification in MAPS:
        _resolve_registered_input(
            root, specification, f"Warehouse map {specification['id']}"
        )

    inputs = dict(config.get("inputs") or {})
    runtime_path = _resolve_registered_input(
        root,
        inputs.get("runtime_config") or {},
        "Warehouse cross-aisle runtime",
    )
    registration = dict(config.get("registration") or {})
    frozen = (
        registration.get("status") == "frozen_before_q1"
        and registration.get("q1_allowed") is True
        and isinstance(registration.get("task_registry_sha256"), str)
        and len(str(registration["task_registry_sha256"])) == 64
    )
    generator_spec = dict(inputs.get("task_generator") or {})
    if frozen:
        if config.get("scientific_status") != (
            "preregistered_hash_frozen_q0_q1_reset_only_benchmark_qualification"
        ):
            raise ValueError("Warehouse cross-aisle frozen status changed")
        _resolve_registered_input(
            root, generator_spec, "Warehouse cross-aisle task generator"
        )
    else:
        if (
            registration
            != {
                "status": "pending_q0_task_hash_registration",
                "task_registry_path": (
                    "configs/stride_warehouse_crossaisle_paired_tasks_v1.json"
                ),
                "task_registry_sha256": None,
                "q1_allowed": False,
            }
            or config.get("scientific_status")
            != "preregistered_q0_generation_pending_task_hash_registration_reset_only"
        ):
            raise ValueError("Warehouse cross-aisle pending registration changed")
        _pending_input(root, generator_spec, "Warehouse cross-aisle task generator")
    if require_frozen_registry and not frozen:
        raise ValueError(
            "Warehouse cross-aisle Q1 is blocked until all 32 task hashes and "
            "the Q0 manifest hash are frozen in the tracked registry"
        )
    runtime = _read_json(runtime_path)
    if (
        int(runtime.get("schema_version", -1)) != RUNTIME_SCHEMA
        or runtime.get("formal") is not False
        or str(runtime.get("split")) != SPLIT
        or tuple(map(int, runtime.get("solver_seeds") or ())) != SOLVER_SEEDS
        or int(runtime.get("max_decisions", -1)) != 0
        or int(runtime.get("workers", -1)) != 16
        or int(dict(runtime.get("dataset_design") or {}).get("map_count", -1)) != 4
        or int(dict(runtime.get("dataset_design") or {}).get("instance_count", -1))
        != 32
    ):
        raise ValueError("Warehouse cross-aisle runtime input changed")
    registry = _load_registry(root, config)[1] if frozen else None
    return path, root, config, registry


def benchmark_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    del config
    rows = [
        dict(row)
        for row in crossaisle_task_specs(
            task_seeds=TASK_SEEDS,
            q_values=Q_VALUES,
        )
    ]
    if len(rows) != 32:
        raise ValueError("Warehouse cross-aisle helper did not define 32 tasks")
    identities = {
        (
            str(row.get("map_id")),
            str(row.get("variant")),
            int(row.get("q", -1)),
            int(row.get("task_seed", -1)),
        )
        for row in rows
    }
    expected = {
        (str(spec["id"]), variant, q, task_seed)
        for spec in MAPS
        for q in Q_VALUES
        for task_seed in TASK_SEEDS
        for variant in TASK_VARIANTS
    }
    if identities != expected or len(identities) != len(rows):
        raise ValueError("Warehouse cross-aisle helper task cohort changed")
    for row in rows:
        expected_agents = int(
            MAP_BY_ID[str(row["map_id"])]["expected_agent_count_by_q"][str(row["q"])]
        )
        if int(row.get("agent_count", -1)) != expected_agents:
            raise ValueError("Warehouse cross-aisle helper agent count changed")
        task_id = str(row.get("task_id") or "")
        if not task_id or len(task_id) != len(task_id.encode("ascii")):
            raise ValueError("Warehouse cross-aisle helper task id changed")
    return sorted(rows, key=lambda row: str(row["task_id"]))


def qualification_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**task, "solver_seed": solver_seed}
        for task in benchmark_task_specs(config)
        for solver_seed in SOLVER_SEEDS
    ]


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config, registry = load_config(config_path)
    return {
        "schema": QUALIFICATION_STATUS_SCHEMA,
        "command": "plan",
        "experiment_id": EXPERIMENT_ID,
        "registration_status": config["registration"]["status"],
        "q1_allowed": registry is not None,
        "map_count": 4,
        "q_values": list(Q_VALUES),
        "task_count": len(benchmark_task_specs(config)),
        "qualification_reset_count": len(qualification_schedule(config)),
        "workers": 16,
        "controller_count": 0,
        "policy_episode_count": 0,
        "formal_ttf_episode_count": 0,
        "claim_boundary": config["cohort"]["claim_boundary"],
    }


def _schedule_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id")), int(row.get("solver_seed", -1))


def _task_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in benchmark_task_specs(config)}


def _qualification_failure_result(
    job: dict[str, Any], status: str, error: str
) -> dict[str, Any]:
    row = dict(job["row"])
    return {
        "split": str(row["split"]),
        "map_id": str(row["map_id"]),
        "task_id": str(row["task_id"]),
        "layout_mode": str(row["layout_mode"]),
        "task_variant": str(row["task_variant"]),
        "agent_count": int(row["agent_count"]),
        "solver_seed": int(job["solver_seed"]),
        "status": status,
        "error": error,
    }


def _qualification_report(
    config: Mapping[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    schedule = qualification_schedule(config)
    expected = {_schedule_key(row): row for row in schedule}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_keys: list[list[Any]] = []
    for raw in rows:
        row = dict(raw)
        key = _schedule_key(row)
        if key in indexed:
            duplicate_keys.append([key[0], key[1]])
        indexed[key] = row
    unexpected_keys = sorted(
        [[task_id, seed] for task_id, seed in set(indexed) - set(expected)]
    )
    tasks: list[dict[str, Any]] = []
    execution_errors: list[dict[str, Any]] = []
    for key, row in sorted(indexed.items()):
        item = expected.get(key)
        status = str(row.get("status"))
        if status != "ok":
            execution_errors.append(
                {
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "status": status,
                    "error": str(row.get("error")),
                }
            )
            continue
        if item is None:
            continue
        complexity = dict(row.get("initial_complexity") or {})
        conflicts = int(row.get("initial_conflicts", -1))
        initial_complete = row.get("initial_complete") is True
        reported_feasible = row.get("initial_feasible") is True
        expected_feasible = bool(initial_complete and conflicts == 0)
        tasks.append(
            {
                "map_id": str(item["map_id"]),
                "task_id": str(item["task_id"]),
                "variant": str(item["variant"]),
                "q": int(item["q"]),
                "task_seed": int(item["task_seed"]),
                "agent_count": int(item["agent_count"]),
                "solver_seed": int(item["solver_seed"]),
                "initial_conflicts": conflicts,
                "active_conflict_agent_count": int(
                    complexity.get("active_conflict_agent_count", -1)
                ),
                "largest_conflict_component_size": int(
                    complexity.get("largest_conflict_component_size", -1)
                ),
                "initial_complete": initial_complete,
                "initial_feasible": expected_feasible,
                "reported_initial_feasible": reported_feasible,
                "initial_state_consistent": reported_feasible == expected_feasible,
                "state_fingerprint": str(row.get("state_fingerprint") or ""),
            }
        )
    by_seed = {
        seed: tuple(
            str(row["state_fingerprint"])
            for row in sorted(
                (task for task in tasks if int(task["solver_seed"]) == seed),
                key=lambda task: str(task["task_id"]),
            )
        )
        for seed in SOLVER_SEEDS
    }
    duplicate_seed_trajectories = [
        [left, right]
        for left_index, left in enumerate(SOLVER_SEEDS)
        for right in SOLVER_SEEDS[left_index + 1 :]
        if by_seed[left] and by_seed[left] == by_seed[right]
    ]
    return {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "q1_reset_only",
        "expected_reset_count": 128,
        "observed_result_count": len(indexed),
        "valid_reset_count": len(tasks),
        "execution_error_count": sum(
            row["status"] == "error" for row in execution_errors
        ),
        "process_timeout_count": sum(
            row["status"] == "timeout" for row in execution_errors
        ),
        "duplicate_keys": duplicate_keys,
        "unexpected_keys": unexpected_keys,
        "duplicate_solver_seed_trajectories": duplicate_seed_trajectories,
        "all_expected_results_present": set(indexed) == set(expected),
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "tasks": sorted(
            tasks, key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))
        ),
        "execution_errors": execution_errors,
    }


def select_qualified_benchmark(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = [dict(row) for row in report.get("tasks") or ()]
    indexed = {_schedule_key(row): row for row in tasks}
    schedule = qualification_schedule(config)
    expected = {_schedule_key(row): row for row in schedule}
    global_errors: list[str] = []
    if int(report.get("execution_error_count", -1)) != 0:
        global_errors.append("Q1 contains an execution error")
    if int(report.get("process_timeout_count", -1)) != 0:
        global_errors.append("Q1 contains a process timeout")
    if list(report.get("duplicate_keys") or ()):
        global_errors.append("Q1 contains duplicate reset keys")
    if list(report.get("unexpected_keys") or ()):
        global_errors.append("Q1 contains unexpected reset keys")
    if (
        report.get("all_expected_results_present") is not True
        or set(indexed) != set(expected)
        or len(indexed) != 128
    ):
        global_errors.append("Q1 reset coverage differs from the frozen 128 keys")

    map_results: dict[str, Any] = {}
    selected_q_by_map: dict[str, int] = {}
    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        q_results: dict[str, Any] = {}
        selected_q: int | None = None
        for q in Q_VALUES:
            variant_results: dict[str, Any] = {}
            for variant in TASK_VARIANTS:
                reset_rows: list[dict[str, Any]] = []
                failures: list[str] = []
                for task_seed in TASK_SEEDS:
                    task = next(
                        item
                        for item in schedule
                        if str(item["map_id"]) == map_id
                        and str(item["variant"]) == variant
                        and int(item["q"]) == q
                        and int(item["task_seed"]) == task_seed
                    )
                    for solver_seed in SOLVER_SEEDS:
                        key = (str(task["task_id"]), solver_seed)
                        row = indexed.get(key)
                        if row is None:
                            failures.append(f"missing:{key[0]}/seed{solver_seed}")
                            continue
                        reset_rows.append(row)
                        common_ok = bool(
                            row.get("initial_complete") is True
                            and row.get("initial_state_consistent") is True
                            and row.get("initial_feasible") is False
                            and len(str(row.get("state_fingerprint") or "")) == 64
                        )
                        if variant == STRUCTURED_VARIANT:
                            passed = bool(
                                common_ok
                                and int(row.get("initial_conflicts", -1)) >= 16
                                and int(row.get("active_conflict_agent_count", -1)) >= 32
                                and int(
                                    row.get("largest_conflict_component_size", -1)
                                )
                                >= 16
                            )
                        else:
                            passed = bool(
                                common_ok
                                and int(row.get("initial_conflicts", -1)) >= 1
                            )
                        if not passed:
                            failures.append(f"gate:{key[0]}/seed{solver_seed}")
                variant_results[variant] = {
                    "expected_reset_count": 8,
                    "observed_reset_count": len(reset_rows),
                    "minimum_initial_conflicts": (
                        min(int(row["initial_conflicts"]) for row in reset_rows)
                        if reset_rows
                        else None
                    ),
                    "minimum_active_conflict_agents": (
                        min(
                            int(row["active_conflict_agent_count"])
                            for row in reset_rows
                        )
                        if reset_rows
                        else None
                    ),
                    "minimum_largest_conflict_component_size": (
                        min(
                            int(row["largest_conflict_component_size"])
                            for row in reset_rows
                        )
                        if reset_rows
                        else None
                    ),
                    "passed": len(reset_rows) == 8 and not failures,
                    "failures": failures,
                }
            common_passed = all(
                variant_results[variant]["passed"] for variant in TASK_VARIANTS
            )
            q_results[str(q)] = {
                "both_variants_passed": common_passed,
                "variant_results": variant_results,
            }
            if selected_q is None and common_passed:
                selected_q = q
        map_results[map_id] = {
            "passed": selected_q is not None,
            "selected_q": selected_q,
            "selection_rule": "lowest_common_passing_q",
            "q_results": q_results,
        }
        if selected_q is not None:
            selected_q_by_map[map_id] = selected_q

    benchmark_ready = not global_errors and len(selected_q_by_map) == 4
    return {
        "schema": SELECTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "reset_only_benchmark_qualification_not_a_ttf_claim",
        "benchmark_ready": benchmark_ready,
        "decision": (
            "benchmark_ready_for_separately_preregistered_controller_evaluation"
            if benchmark_ready
            else "stop_do_not_resample_or_replace"
        ),
        "map_count": 4,
        "task_count": 32,
        "expected_reset_count": 128,
        "selected_map_count": len(selected_q_by_map),
        "selected_q_by_map": selected_q_by_map,
        "map_results": map_results,
        "global_errors": global_errors,
        "controller_performance_claim": False,
        "ttf_claim": False,
        "matched_control_interpretation": config["result_interpretation"][
            "matched_control_label"
        ],
        "fully_difficulty_matched_control_claim": False,
        "claim_boundary": config["cohort"]["claim_boundary"],
    }


def _generation_fingerprint(root: Path) -> str:
    return _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "maps": [
                {key: value for key, value in row.items() if key != "path"}
                for row in MAPS
            ],
            "task_seeds": list(TASK_SEEDS),
            "q_values": list(Q_VALUES),
            "task_variants": list(TASK_VARIANTS),
            "cross_aisle_group_count": CROSS_AISLE_GROUP_COUNT,
            "task_generator_sha256": sha256_file(
                root / "experiments" / "warehouse_crossaisle_tasks.py"
            ),
        }
    )


def _paired_payload_from_task_files(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = {str(left["variant"]): dict(left), str(right["variant"]): dict(right)}
    first = tasks[STRUCTURED_VARIANT]
    return {
        "schema": TASK_PAIR_SCHEMA,
        "pairing_semantics": first["pairing_semantics"],
        "causal_control_claim": first["causal_control_claim"],
        "map_id": first["map_id"],
        "map_sha256": first["map_sha256"],
        "task_seed": first["task_seed"],
        "q": first["q"],
        "agent_count": first["agent_count"],
        "geometry": first["geometry"],
        "variants": {
            variant: {
                key: value
                for key, value in tasks[variant].items()
                if key
                in {
                    "task_id",
                    "map_id",
                    "variant",
                    "task_seed",
                    "q",
                    "agent_count",
                    "starts",
                    "goals",
                    "shortest_path_distances",
                }
            }
            for variant in TASK_VARIANTS
        },
    }


def _parse_and_audit_scenario(
    scenario_path: Path,
    *,
    map_name: str,
    rows: int,
    cols: int,
    passable: set[tuple[int, int]],
    task: Mapping[str, Any],
) -> None:
    lines = scenario_path.read_text(encoding="utf-8").splitlines()
    expected_count = int(task["agent_count"])
    if not lines or lines[0] != "version 1" or len(lines) != expected_count + 1:
        raise ValueError(f"Warehouse cross-aisle scenario shape changed: {scenario_path}")
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    recorded_distances: list[int] = []
    for index, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(
                f"Warehouse cross-aisle scenario row has {len(fields)} fields: "
                f"{scenario_path}/{index}"
            )
        bucket, row_map, width, height, sx, sy, gx, gy, distance = fields
        try:
            parsed = tuple(map(int, (bucket, width, height, sx, sy, gx, gy, distance)))
        except ValueError as error:
            raise ValueError(
                f"Warehouse cross-aisle scenario row is non-integral: "
                f"{scenario_path}/{index}"
            ) from error
        bucket_i, width_i, height_i, sx_i, sy_i, gx_i, gy_i, distance_i = parsed
        if (
            bucket_i != 0
            or row_map != map_name
            or width_i != cols
            or height_i != rows
            or distance_i < 0
        ):
            raise ValueError(
                f"Warehouse cross-aisle scenario identity changed: "
                f"{scenario_path}/{index}"
            )
        starts.append((sy_i, sx_i))
        goals.append((gy_i, gx_i))
        recorded_distances.append(distance_i)
    task_starts = [tuple(map(int, cell)) for cell in task.get("starts") or ()]
    task_goals = [tuple(map(int, cell)) for cell in task.get("goals") or ()]
    task_distances = list(map(int, task.get("shortest_path_distances") or ()))
    recomputed = _four_neighbor_distances(passable, starts, goals)
    if (
        starts != task_starts
        or goals != task_goals
        or recorded_distances != task_distances
        or recomputed != task_distances
        or len(recomputed) != expected_count
    ):
        raise ValueError(
            f"Warehouse cross-aisle scenario/task coordinate or distance mismatch: "
            f"{scenario_path}"
        )


def _contained_file(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the registered dataset split") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _audit_registered_dataset(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    dataset = output / "dataset"
    summary_path = dataset / DATASET_SUMMARY_FILENAME
    q0_path = dataset / Q0_FILENAME
    q0_manifest_path = dataset / Q0_MANIFEST_FILENAME
    split = dataset / SPLIT
    manifest_path = split / "manifest.jsonl"
    if not all(
        path.is_file()
        for path in (summary_path, q0_path, q0_manifest_path, manifest_path)
    ):
        raise ValueError("Warehouse cross-aisle Q0 dataset evidence is incomplete")
    summary = _read_json(summary_path)
    q0 = _read_json(q0_path)
    q0_rows = _read_jsonl(q0_manifest_path)
    manifest = _read_jsonl(manifest_path)
    if (
        summary.get("dataset_revision") != EXPERIMENT_ID
        or summary.get("configuration_fingerprint") != _generation_fingerprint(root)
        or int(summary.get("task_count", -1)) != 32
        or str(summary.get("q0_manifest_sha256")) != sha256_file(q0_manifest_path)
        or q0.get("schema") != Q0_SCHEMA
        or q0.get("passed") is not True
        or int(q0.get("map_count", -1)) != 4
        or int(q0.get("task_count", -1)) != 32
        or q0.get("solver_or_controller_invoked") is not False
        or len(q0_rows) != 32
        or len(manifest) != 32
    ):
        raise ValueError("Warehouse cross-aisle Q0 dataset identity changed")
    q0_by_task = {str(row.get("task_id")): dict(row) for row in q0_rows}
    manifest_by_task = {str(row.get("task_id")): dict(row) for row in manifest}
    expected_specs = _task_index(config)
    if (
        set(q0_by_task) != set(expected_specs)
        or set(manifest_by_task) != set(expected_specs)
        or len(q0_by_task) != 32
        or len(manifest_by_task) != 32
    ):
        raise ValueError("Warehouse cross-aisle Q0 task coverage changed")
    registered_tasks = dict(registry.get("tasks") or {}) if registry else None
    if registry is not None and str(registry["q0_manifest_sha256"]) != sha256_file(
        q0_manifest_path
    ):
        raise ValueError("Warehouse cross-aisle registered Q0 manifest changed")

    map_inputs: dict[str, tuple[int, int, set[tuple[int, int]]]] = {}
    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        map_path = split / "maps" / f"{map_id}.map"
        metadata_path = split / "maps" / f"{map_id}.json"
        if (
            not map_path.is_file()
            or not metadata_path.is_file()
            or sha256_file(map_path) != str(map_spec["sha256"])
        ):
            raise ValueError(f"Warehouse cross-aisle map bytes changed: {map_id}")
        geometry = detect_cross_aisle_geometry(map_path, map_id)
        rows, cols, _grid, passable = _movingai_passable_cells(map_path)
        map_inputs[map_id] = (rows, cols, passable)
        metadata = _read_json(metadata_path)
        if (
            int(geometry["internal_group_count"])
            != int(map_spec["expected_internal_group_count"])
            or int(geometry["aisle_width"])
            != int(map_spec["expected_aisle_width"])
            or int(geometry["group_count"]) != CROSS_AISLE_GROUP_COUNT
            or str(metadata.get("benchmark_id")) != map_id
            or str(metadata.get("source"))
            != "checksum-pinned MovingAI Warehouse map"
            or str(metadata.get("source_path")) != str(map_spec["path"])
            or str(metadata.get("map_sha256")) != str(map_spec["sha256"])
            or dict(metadata.get("cross_aisle_geometry") or {}) != geometry
        ):
            raise ValueError(f"Warehouse cross-aisle geometry changed: {map_id}")

    for task_id, spec in expected_specs.items():
        row = manifest_by_task[task_id]
        q0_row = q0_by_task[task_id]
        expected_map_file = f"maps/{spec['map_id']}.map"
        expected_metadata_file = f"maps/{spec['map_id']}.json"
        expected_scenario_file = f"scenarios/{task_id}.scen"
        expected_task_file = f"tasks/{task_id}.json"
        if (
            str(row.get("split")) != SPLIT
            or str(row.get("source_group")) != "movingai"
            or str(row.get("instance_origin"))
            != "warehouse_crossaisle_paired_v1_geometry_only"
            or str(row.get("map_id")) != str(spec["map_id"])
            or str(row.get("task_variant")) != str(spec["variant"])
            or str(row.get("variant")) != str(spec["variant"])
            or int(row.get("agent_count", -1)) != int(spec["agent_count"])
            or int(row.get("q", -1)) != int(spec["q"])
            or int(row.get("task_seed", -1)) != int(spec["task_seed"])
            or str(row.get("map_file")) != expected_map_file
            or str(row.get("map_metadata_file")) != expected_metadata_file
            or str(row.get("scenario_file")) != expected_scenario_file
            or str(row.get("task_file")) != expected_task_file
            or str(row.get("layout_mode")) != "warehouse"
            or str(row.get("layout_variant")) != str(spec["map_id"])
            or str(row.get("scenario_type"))
            != f"warehouse_crossaisle_{spec['variant']}"
        ):
            raise ValueError(f"Warehouse cross-aisle manifest row changed: {task_id}")
        _contained_file(split, expected_map_file, f"map for {task_id}")
        _contained_file(split, expected_metadata_file, f"metadata for {task_id}")
        scenario_path = _contained_file(
            split, expected_scenario_file, f"scenario for {task_id}"
        )
        task_path = _contained_file(split, expected_task_file, f"task for {task_id}")
        hashes = {
            "scenario_sha256": sha256_file(scenario_path),
            "task_sha256": sha256_file(task_path),
        }
        task = _read_json(task_path)
        if (
            str(task.get("task_id")) != task_id
            or str(task.get("map_id")) != str(spec["map_id"])
            or str(task.get("variant")) != str(spec["variant"])
            or int(task.get("task_seed", -1)) != int(spec["task_seed"])
            or int(task.get("q", -1)) != int(spec["q"])
            or int(task.get("agent_count", -1)) != int(spec["agent_count"])
            or str(task.get("map_sha256"))
            != str(MAP_BY_ID[str(spec["map_id"])]["sha256"])
            or str(task.get("scenario_sha256")) != hashes["scenario_sha256"]
        ):
            raise ValueError(f"Warehouse cross-aisle task metadata changed: {task_id}")
        rows, cols, passable = map_inputs[str(spec["map_id"])]
        _parse_and_audit_scenario(
            scenario_path,
            map_name=f"{spec['map_id']}.map",
            rows=rows,
            cols=cols,
            passable=passable,
            task=task,
        )
        if (
            q0_row.get("schema") != Q0_MANIFEST_SCHEMA
            or q0_row.get("passed") is not True
            or str(q0_row.get("task_id")) != task_id
            or str(q0_row.get("map_id")) != str(spec["map_id"])
            or str(q0_row.get("variant")) != str(spec["variant"])
            or int(q0_row.get("task_seed", -1)) != int(spec["task_seed"])
            or int(q0_row.get("q", -1)) != int(spec["q"])
            or int(q0_row.get("agent_count", -1)) != int(spec["agent_count"])
            or str(q0_row.get("scenario_sha256")) != hashes["scenario_sha256"]
            or str(q0_row.get("task_sha256")) != hashes["task_sha256"]
        ):
            raise ValueError(f"Warehouse cross-aisle Q0 row changed: {task_id}")
        if registered_tasks is not None:
            registered = dict(registered_tasks.get(task_id) or {})
            for field in (
                "map_id",
                "variant",
                "task_seed",
                "q",
                "agent_count",
                "scenario_sha256",
                "task_sha256",
            ):
                expected_value = hashes[field] if field in hashes else spec[field]
                if registered.get(field) != expected_value:
                    raise ValueError(
                        f"Warehouse cross-aisle registered task bytes changed: {task_id}"
                    )

    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        map_path = split / "maps" / f"{map_id}.map"
        for q in Q_VALUES:
            for task_seed in TASK_SEEDS:
                pair_tasks = []
                for variant in TASK_VARIANTS:
                    spec = next(
                        row
                        for row in expected_specs.values()
                        if str(row["map_id"]) == map_id
                        and str(row["variant"]) == variant
                        and int(row["q"]) == q
                        and int(row["task_seed"]) == task_seed
                    )
                    pair_tasks.append(
                        _read_json(
                            split
                            / str(manifest_by_task[str(spec["task_id"])]["task_file"])
                        )
                    )
                pair_audit = audit_paired_crossaisle_tasks(
                    map_path,
                    _paired_payload_from_task_files(pair_tasks[0], pair_tasks[1]),
                )
                if pair_audit.get("passed") is not True:
                    raise ValueError(
                        f"Warehouse cross-aisle paired Q0 re-audit failed: "
                        f"{map_id}/q{q}/task{task_seed}"
                    )
    return summary


def prepare_q0(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    _path, root, config, registry = load_config(config_path)
    output = Path(output).resolve()
    dataset = output / "dataset"
    summary_path = dataset / DATASET_SUMMARY_FILENAME
    if summary_path.is_file():
        return _audit_registered_dataset(root, output, config, registry)
    if dataset.is_dir() and any(dataset.iterdir()):
        raise ValueError(
            "Warehouse cross-aisle Q0 dataset is non-empty without trusted identity"
        )

    split = dataset / SPLIT
    geometry_by_map: dict[str, dict[str, Any]] = {}
    geometry_errors: list[dict[str, str]] = []
    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        source = _resolve_registered_input(root, map_spec, f"Warehouse map {map_id}")
        map_path = split / "maps" / f"{map_id}.map"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, map_path)
        try:
            geometry = detect_cross_aisle_geometry(map_path, map_id)
            if (
                int(geometry["internal_group_count"])
                != int(map_spec["expected_internal_group_count"])
                or int(geometry["aisle_width"])
                != int(map_spec["expected_aisle_width"])
                or int(geometry["group_count"]) != CROSS_AISLE_GROUP_COUNT
            ):
                raise ValueError("registered geometry count/width changed")
            geometry_by_map[map_id] = geometry
        except (OSError, ValueError) as error:
            geometry_errors.append({"map_id": map_id, "error": str(error)})
    if geometry_errors:
        q0 = {
            "schema": Q0_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "passed": False,
            "map_count": len(geometry_by_map),
            "task_count": 0,
            "geometry_errors": geometry_errors,
            "decision": "stop_before_any_native_reset",
            "solver_or_controller_invoked": False,
        }
        _write_json(dataset / Q0_FILENAME, q0)
        return q0

    manifest: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    for map_spec in MAPS:
        map_id = str(map_spec["id"])
        map_path = split / "maps" / f"{map_id}.map"
        rows, cols, _grid, _passable = _movingai_passable_cells(map_path)
        metadata_path = split / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "checksum-pinned MovingAI Warehouse map",
                "source_path": str(map_spec["path"]),
                "map_sha256": sha256_file(map_path),
                "cross_aisle_geometry": geometry_by_map[map_id],
                "topology_metrics": _map_metrics(map_path),
            },
        )
        for q in Q_VALUES:
            for task_seed in TASK_SEEDS:
                pair = generate_paired_crossaisle_tasks(
                    map_path, map_id, task_seed, q
                )
                pair_audit = audit_paired_crossaisle_tasks(map_path, pair)
                pair_audits.append(pair_audit)
                if pair_audit.get("passed") is not True:
                    raise RuntimeError(
                        f"Warehouse cross-aisle Q0 pair audit failed: "
                        f"{map_id}/q{q}/task{task_seed}: {pair_audit.get('errors')}"
                    )
                for variant in TASK_VARIANTS:
                    task = dict(pair["variants"][variant])
                    task_id = str(task["task_id"])
                    starts = [tuple(map(int, cell)) for cell in task["starts"]]
                    goals = [tuple(map(int, cell)) for cell in task["goals"]]
                    distances = list(map(int, task["shortest_path_distances"]))
                    expected_agents = int(
                        map_spec["expected_agent_count_by_q"][str(q)]
                    )
                    if int(task["agent_count"]) != expected_agents:
                        raise RuntimeError(
                            f"Warehouse cross-aisle Q0 agent count changed: {task_id}"
                        )
                    scenario_path = split / "scenarios" / f"{task_id}.scen"
                    _write_derived_scenario(
                        scenario_path,
                        map_path.name,
                        rows,
                        cols,
                        starts,
                        goals,
                        distances,
                    )
                    task_payload = {
                        **task,
                        "schema_version": 1,
                        "task_semantics": "paired Warehouse cross-aisle benchmark Q0",
                        "pairing_semantics": pair["pairing_semantics"],
                        "causal_control_claim": pair["causal_control_claim"],
                        "map_sha256": pair["map_sha256"],
                        "geometry": pair["geometry"],
                        "scenario_sha256": sha256_file(scenario_path),
                    }
                    task_path = split / "tasks" / f"{task_id}.json"
                    _write_json(task_path, task_payload)
                    q0_row = {
                        "schema": Q0_MANIFEST_SCHEMA,
                        "task_id": task_id,
                        "map_id": map_id,
                        "variant": variant,
                        "task_seed": task_seed,
                        "q": q,
                        "agent_count": expected_agents,
                        "scenario_sha256": sha256_file(scenario_path),
                        "task_sha256": sha256_file(task_path),
                        "pair_distance_mean_relative_difference": pair_audit[
                            "metrics"
                        ]["paired_distance"]["mean_relative_difference"],
                        "pair_distance_p95_relative_difference": pair_audit[
                            "metrics"
                        ]["paired_distance"]["p95_relative_difference"],
                        "passed": True,
                    }
                    q0_rows.append(q0_row)
                    manifest.append(
                        {
                            "split": SPLIT,
                            "source_group": "movingai",
                            "instance_origin": (
                                "warehouse_crossaisle_paired_v1_geometry_only"
                            ),
                            "map_id": map_id,
                            "task_id": task_id,
                            "map_file": f"maps/{map_path.name}",
                            "scenario_file": f"scenarios/{scenario_path.name}",
                            "map_metadata_file": f"maps/{metadata_path.name}",
                            "task_file": f"tasks/{task_path.name}",
                            "layout_mode": "warehouse",
                            "layout_variant": map_id,
                            "scenario_type": f"warehouse_crossaisle_{variant}",
                            "task_variant": variant,
                            "variant": variant,
                            "task_seed": task_seed,
                            "q": q,
                            "agent_count": expected_agents,
                            "topology_metrics": _map_metrics(map_path),
                            "dominant_flow_ratio": 0.0,
                            "hotspot_skew": 0.0,
                            "required_bottleneck_crossing_ratio": 0.0,
                            "mean_shortest_distance": sum(distances) / len(distances),
                        }
                    )
    manifest.sort(key=lambda row: str(row["task_id"]))
    q0_rows.sort(key=lambda row: str(row["task_id"]))
    if len(manifest) != 32 or len(q0_rows) != 32:
        raise RuntimeError("Warehouse cross-aisle Q0 task count changed")
    _write_jsonl(split / "manifest.jsonl", manifest)
    _write_jsonl(dataset / Q0_MANIFEST_FILENAME, q0_rows)
    q0 = {
        "schema": Q0_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "passed": all(row["passed"] for row in q0_rows),
        "map_count": 4,
        "task_count": 32,
        "map_geometry": geometry_by_map,
        "pair_audits": pair_audits,
        "solver_or_controller_invoked": False,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
        "control_interpretation": config["result_interpretation"][
            "matched_control_label"
        ],
    }
    _write_json(dataset / Q0_FILENAME, q0)
    q0_manifest_sha = sha256_file(dataset / Q0_MANIFEST_FILENAME)
    proposal = {
        "schema": REGISTRY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_count": 32,
        "q0_manifest_sha256": q0_manifest_sha,
        "task_generator_sha256": sha256_file(
            root / "experiments" / "warehouse_crossaisle_tasks.py"
        ),
        "tasks": {
            str(row["task_id"]): {
                field: row[field]
                for field in (
                    "map_id",
                    "variant",
                    "task_seed",
                    "q",
                    "agent_count",
                    "scenario_sha256",
                    "task_sha256",
                )
            }
            for row in q0_rows
        },
        "geometry_summary": {
            map_id: {
                "internal_group_count": int(value["internal_group_count"]),
                "aisle_width": int(value["aisle_width"]),
                "chosen_group_count": int(value["group_count"]),
                "excluded_central_group": value["excluded_central_group"],
                "chosen_groups": value["chosen_groups"],
            }
            for map_id, value in geometry_by_map.items()
        },
    }
    _write_json(dataset / Q0_REGISTRATION_PROPOSAL_FILENAME, proposal)
    summary = {
        "schema_version": 1,
        "dataset_revision": EXPERIMENT_ID,
        "configuration_fingerprint": _generation_fingerprint(root),
        "source": "four checksum-pinned MovingAI Warehouse maps",
        "task_semantics": "Q0-only paired cross-aisle task registration",
        "map_count": 4,
        "task_count": 32,
        "q0_manifest_sha256": q0_manifest_sha,
        "registration_proposal_sha256": sha256_file(
            dataset / Q0_REGISTRATION_PROPOSAL_FILENAME
        ),
        "solver_or_controller_invoked": False,
    }
    _write_json(dataset / DATASET_SUMMARY_FILENAME, summary)
    _audit_registered_dataset(root, output, config, registry)
    return {
        **summary,
        "q0_passed": True,
        "registration_status": config["registration"]["status"],
        "q1_allowed": registry is not None,
    }


def _audit_qualification_evidence(
    config: Mapping[str, Any],
    qualification: Path,
    *,
    require_complete: bool,
    registry: Mapping[str, Any],
    runtime_preflight_run_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    manifest_path = qualification / QUALIFICATION_MANIFEST_FILENAME
    rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    expected = {_schedule_key(row): row for row in qualification_schedule(config)}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = _schedule_key(row)
        if key in indexed:
            raise ValueError("Warehouse cross-aisle Q1 manifest has duplicate keys")
        if key not in expected:
            raise ValueError("Warehouse cross-aisle Q1 manifest has an unknown key")
        item = expected[key]
        registry_fields = dict(row.get("registered_task_identity") or {})
        registered = {
            "map_id": str(item["map_id"]),
            "variant": str(item["variant"]),
            "task_seed": int(item["task_seed"]),
            "q": int(item["q"]),
            "agent_count": int(item["agent_count"]),
        }
        registered_task = dict(registry["tasks"])[str(item["task_id"])]
        if (
            str(row.get("split")) != SPLIT
            or str(row.get("map_id")) != str(item["map_id"])
            or str(row.get("task_id")) != str(item["task_id"])
            or str(row.get("task_variant")) != str(item["variant"])
            or int(row.get("agent_count", -1)) != int(item["agent_count"])
            or int(row.get("solver_seed", -1)) != int(item["solver_seed"])
            or registry_fields != registered
            or str(row.get("registered_scenario_sha256"))
            != str(registered_task["scenario_sha256"])
            or str(row.get("registered_task_sha256"))
            != str(registered_task["task_sha256"])
            or str(row.get("registered_q0_manifest_sha256"))
            != str(registry["q0_manifest_sha256"])
            or str(row.get("runtime_preflight_run_fingerprint"))
            != runtime_preflight_run_fingerprint
            or str(row.get("status")) not in {"ok", "error", "timeout"}
        ):
            raise ValueError(f"Warehouse cross-aisle Q1 row identity changed: {key}")
        if str(row.get("status")) == "ok":
            complexity = dict(row.get("initial_complexity") or {})
            if (
                row.get("error") is not None
                or not isinstance(row.get("initial_conflicts"), int)
                or int(complexity.get("agent_count", -1)) != int(item["agent_count"])
                or int(complexity.get("conflict_pair_count", -1))
                != int(row["initial_conflicts"])
                or int(complexity.get("active_conflict_agent_count", -1)) < 0
                or int(complexity.get("largest_conflict_component_size", -1)) < 0
                or not isinstance(row.get("state_fingerprint"), str)
                or len(str(row["state_fingerprint"])) != 64
            ):
                raise ValueError(
                    f"Warehouse cross-aisle Q1 reset evidence changed: {key}"
                )
        elif not str(row.get("error") or ""):
            raise ValueError(
                f"Warehouse cross-aisle Q1 failure lacks an error: {key}"
            )
        indexed[key] = row
    if require_complete and set(indexed) != set(expected):
        raise ValueError("Warehouse cross-aisle Q1 evidence is incomplete")

    report_path = qualification / QUALIFICATION_REPORT_FILENAME
    stored_report = _read_json(report_path) if report_path.is_file() else None
    recomputed = _qualification_report(config, rows)
    if stored_report is not None and stored_report != recomputed:
        raise ValueError("Warehouse cross-aisle Q1 report/manifest mismatch")
    if require_complete and stored_report is None:
        raise ValueError("Warehouse cross-aisle Q1 report is missing")
    return rows, stored_report


def _runtime_preflight(
    root: Path,
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    runtime_path = _resolve_registered_input(
        root,
        dict(config["inputs"])["runtime_config"],
        "Warehouse cross-aisle runtime",
    )
    keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in qualification_schedule(config)
    }
    return run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        output / "qualification-runtime-preflight-not-written",
        phase="qualify",
        workers=16,
        dry_run=True,
        controller="official_adaptive",
        job_keys=keys,
        cohort_job_keys=keys,
        qualification_process_timeout_seconds=240.0,
    )


def _producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=PRODUCER_SOURCE_FILES,
        native_required=True,
    )


def _dataset_rows(output: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(output / "dataset" / SPLIT / "manifest.jsonl")
    indexed = {str(row.get("task_id")): dict(row) for row in rows}
    if len(rows) != 32 or len(indexed) != 32:
        raise ValueError("Warehouse cross-aisle Q1 dataset manifest changed")
    return indexed


def _write_qualification_status(
    qualification: Path,
    base: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    complete: bool,
    terminal_error: bool,
    benchmark_ready: bool = False,
    report_sha256: str | None = None,
) -> None:
    errors = [row for row in rows if str(row.get("status")) != "ok"]
    payload = {
        **dict(base),
        "completed_schedule_entries": len(rows),
        "complete": complete,
        "terminal_error": terminal_error,
        "error_jobs": sum(str(row.get("status")) == "error" for row in errors),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in errors),
        "benchmark_ready": benchmark_ready,
        "controller_step_invoked": False,
        "policy_episode_invoked": False,
        "formal_ttf_invoked": False,
    }
    if report_sha256 is not None:
        payload["report_sha256"] = report_sha256
    _write_json(qualification / QUALIFICATION_STATUS_FILENAME, payload)


def qualify(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config, registry = load_config(
        config_path, require_frozen_registry=True
    )
    assert registry is not None
    output = Path(output).resolve()
    prepare_q0(path, output)
    _audit_registered_dataset(root, output, config, registry)
    preflight = _runtime_preflight(root, config, output)
    if dry_run:
        return {
            **plan(path),
            "command": "qualify",
            "dry_run": True,
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
            "native_reset_invoked": False,
        }

    qualification = output / "qualification"
    schedule = [
        {
            **row,
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
        }
        for row in qualification_schedule(config)
    ]
    producer = _producer(root)
    prepared = prepare_resumable_output(
        qualification,
        status_filename=QUALIFICATION_STATUS_FILENAME,
        status_schema=QUALIFICATION_STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=producer,
        resume=resume,
        report_filename=SELECTION_FILENAME,
        report_schema=SELECTION_SCHEMA,
        label="Warehouse cross-aisle Q1 qualification",
    )
    if prepared.completed_report is not None:
        rows, report = _audit_qualification_evidence(
            config,
            qualification,
            require_complete=True,
            registry=registry,
            runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
        )
        assert report is not None
        recomputed = select_qualified_benchmark(config, report)
        stored = dict(prepared.completed_report)
        for field in (
            "config_sha256",
            "q0_manifest_sha256",
            "qualification_manifest_sha256",
            "qualification_report_sha256",
            "producer_identity",
            "runtime_preflight_run_fingerprint",
        ):
            recomputed[field] = stored[field]
        if stored != recomputed:
            raise ValueError("Warehouse cross-aisle completed Q1 result changed")
        return stored

    existing, _stored_report = _audit_qualification_evidence(
        config,
        qualification,
        require_complete=False,
        registry=registry,
        runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
    )
    if int(prepared.status.get("completed_schedule_entries", -1)) != len(existing):
        raise ValueError("Warehouse cross-aisle Q1 resume progress changed")
    if prepared.status.get("terminal_error") is True or any(
        str(row.get("status")) != "ok" for row in existing
    ):
        raise ValueError(
            "Warehouse cross-aisle Q1 output is terminal after an execution "
            "error or process timeout; use a new preregistered experiment"
        )
    if (qualification / QUALIFICATION_REPORT_FILENAME).is_file():
        raise ValueError(
            "Warehouse cross-aisle incomplete Q1 unexpectedly has a final report"
        )

    schedule_index = {
        _schedule_key(row): index for index, row in enumerate(schedule)
    }
    schedule_by_key = {_schedule_key(row): row for row in schedule}
    results_by_key = {_schedule_key(row): dict(row) for row in existing}
    pending = [row for row in schedule if _schedule_key(row) not in results_by_key]
    dataset_by_task = _dataset_rows(output)
    jobs = [
        {
            "row": dataset_by_task[str(item["task_id"])],
            "solver_seed": int(item["solver_seed"]),
            "dataset_root": str(output / "dataset"),
            "environment": _read_json(
                _resolve_registered_input(
                    root,
                    dict(config["inputs"])["runtime_config"],
                    "Warehouse cross-aisle runtime",
                )
            )["environment"],
        }
        for item in pending
    ]

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            results_by_key[key]
            for key in sorted(results_by_key, key=schedule_index.__getitem__)
        ]

    def record(result: dict[str, Any]) -> None:
        key = _schedule_key(result)
        if key not in schedule_index or key in results_by_key:
            raise ValueError("Warehouse cross-aisle Q1 worker returned a bad key")
        item = schedule_by_key[key]
        registered_identity = {
            "map_id": str(item["map_id"]),
            "variant": str(item["variant"]),
            "task_seed": int(item["task_seed"]),
            "q": int(item["q"]),
            "agent_count": int(item["agent_count"]),
        }
        registered_task = dict(registry["tasks"])[str(item["task_id"])]
        results_by_key[key] = {
            **dict(result),
            "registered_task_identity": registered_identity,
            "registered_scenario_sha256": registered_task["scenario_sha256"],
            "registered_task_sha256": registered_task["task_sha256"],
            "registered_q0_manifest_sha256": registry["q0_manifest_sha256"],
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
        }
        rows = ordered_rows()
        _write_jsonl(qualification / QUALIFICATION_MANIFEST_FILENAME, rows)
        terminal = str(result.get("status")) in {"error", "timeout"}
        _write_qualification_status(
            qualification,
            prepared.base_status,
            rows,
            complete=False,
            terminal_error=terminal,
        )

    if not existing:
        _write_jsonl(qualification / QUALIFICATION_MANIFEST_FILENAME, [])
        _write_qualification_status(
            qualification,
            prepared.base_status,
            [],
            complete=False,
            terminal_error=False,
        )
    if jobs:
        with _CollectionRunLock(
            qualification,
            str(prepared.base_status["run_fingerprint"]),
            "warehouse-crossaisle-q1-reset-only",
        ):
            _run_jobs(
                _qualification_worker,
                jobs,
                16,
                phase="warehouse-crossaisle-q1-reset-only",
                output_root=qualification,
                run_fingerprint=str(prepared.base_status["run_fingerprint"]),
                timeout_seconds=240.0,
                on_result=record,
                failure_result=_qualification_failure_result,
                stop_on_failure=True,
            )

    rows = ordered_rows()
    failed = [row for row in rows if str(row.get("status")) != "ok"]
    if failed:
        report = _qualification_report(config, rows)
        selection = select_qualified_benchmark(config, report)
        _write_json(qualification / QUALIFICATION_REPORT_FILENAME, report)
        selection.update(
            {
                "terminal_error": True,
                "config_sha256": sha256_file(path),
                "q0_manifest_sha256": sha256_file(
                    output / "dataset" / Q0_MANIFEST_FILENAME
                ),
                "qualification_manifest_sha256": sha256_file(
                    qualification / QUALIFICATION_MANIFEST_FILENAME
                ),
                "qualification_report_sha256": sha256_file(
                    qualification / QUALIFICATION_REPORT_FILENAME
                ),
                "producer_identity": producer,
                "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
            }
        )
        _write_json(qualification / SELECTION_FILENAME, selection)
        _write_qualification_status(
            qualification,
            prepared.base_status,
            rows,
            complete=False,
            terminal_error=True,
            benchmark_ready=False,
        )
        return selection
    if len(rows) != 128:
        raise RuntimeError(
            "Warehouse cross-aisle Q1 stopped without an execution error but "
            "did not cover all 128 resets; resume the identical output"
        )

    report = _qualification_report(config, rows)
    _write_json(qualification / QUALIFICATION_REPORT_FILENAME, report)
    _audit_qualification_evidence(
        config,
        qualification,
        require_complete=True,
        registry=registry,
        runtime_preflight_run_fingerprint=preflight["run_fingerprint"],
    )
    selection = select_qualified_benchmark(config, report)
    selection.update(
        {
            "config_sha256": sha256_file(path),
            "q0_manifest_sha256": sha256_file(
                output / "dataset" / Q0_MANIFEST_FILENAME
            ),
            "qualification_manifest_sha256": sha256_file(
                qualification / QUALIFICATION_MANIFEST_FILENAME
            ),
            "qualification_report_sha256": sha256_file(
                qualification / QUALIFICATION_REPORT_FILENAME
            ),
            "producer_identity": producer,
            "runtime_preflight_run_fingerprint": preflight["run_fingerprint"],
        }
    )
    _write_json(qualification / SELECTION_FILENAME, selection)
    _write_qualification_status(
        qualification,
        prepared.base_status,
        rows,
        complete=True,
        terminal_error=False,
        benchmark_ready=bool(selection["benchmark_ready"]),
        report_sha256=sha256_file(qualification / SELECTION_FILENAME),
    )
    return selection


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return qualify(config_path, output, resume=resume, dry_run=dry_run)


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "MAPS",
    "MATCHED_VARIANT",
    "Q0_FILENAME",
    "Q0_MANIFEST_FILENAME",
    "Q0_REGISTRATION_PROPOSAL_FILENAME",
    "Q_VALUES",
    "QUALIFICATION_REPORT_SCHEMA",
    "QUALIFICATION_STATUS_SCHEMA",
    "REGISTRY_SCHEMA",
    "SELECTION_SCHEMA",
    "SOLVER_SEEDS",
    "STRUCTURED_VARIANT",
    "TASK_SEEDS",
    "TASK_VARIANTS",
    "benchmark_task_specs",
    "load_config",
    "plan",
    "prepare_q0",
    "qualification_schedule",
    "qualify",
    "run",
    "select_qualified_benchmark",
]
