from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments import stride_warehouse_fixed16_development as legacy
from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _derived_endpoint_seed,
    _derived_endpoints,
    _four_neighbor_distances,
    _largest_four_connected_component,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import (
    RESUMABLE_RUN_IDENTITY_SCHEMA,
    RUNNER_CONFIG_SCHEMA,
    load_completed_report,
    prepare_resumable_output,
)
from experiments.stride_hybridstructpool_routed_confirmation import (
    _bounded_paired_comparison,
    _bounded_summary,
)
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA


CONFIG_SCHEMA = "lns2.stride.warehouse_fixed16_development_config.v2"
STATUS_SCHEMA = "lns2.stride.warehouse_fixed16_development_status.v2"
REPORT_SCHEMA = "lns2.stride.warehouse_fixed16_development_report.v2"
QUALIFICATION_SELECTION_SCHEMA = (
    "lns2.stride.warehouse_fixed16_qualification_selection.v2"
)
Q0_SCHEMA = "lns2.stride.warehouse_fixed16_geometry_audit.v2"
EXPERIMENT_ID = "stride-warehouse-fixed16-development-v2"
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "development_report.json"
SELECTION_FILENAME = "selected_cohort.json"
Q0_FILENAME = "q0_geometry_audit.json"

CONTROLLERS = legacy.CONTROLLERS
CHALLENGERS = legacy.CHALLENGERS
CONTROLLER_CODES = legacy.CONTROLLER_CODES
BOUNDARY16_STATIC_CACHE = legacy.BOUNDARY16_STATIC_CACHE
SINGLE_FAMILY_PROFILES = legacy.SINGLE_FAMILY_PROFILES
PERFORMANCE_GATES = legacy.PERFORMANCE_GATES

MAP_ID = "warehouse-10-20-10-2-1"
MAP_CODE = "w1020a"
MAP_SHA256 = "c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0"
TASK_SEEDS = (233, 277)
TASK_VARIANTS = ("opposite_exchange", "uniform_random")
AGENT_COUNT = 600
SOLVER_SEEDS = (19, 20, 21, 22)
HALVES = {
    "A": ((233, 19), (277, 21)),
    "B": ((233, 20), (277, 22)),
}
RUNTIME_CONTRACT = legacy.RUNTIME_CONTRACT
DESIGN_PROVENANCE = {
    "role": "design_source_only_not_imported_or_read_as_runtime_input",
    "r2_qualification_report_path": (
        "build/wh-f16-v1-r2/qualification/qualification_report.json"
    ),
    "r2_qualification_report_sha256": (
        "725ba67b95bb263440e897cc900a0a59637f696facf8796f6867610ab12cfbfc"
    ),
    "r2_overall_qualification_passed": False,
    "r2_formal_episode_count": 0,
    "registered_observation": (
        "w1020a_at_n600_was_the_only_per_layout_slice_that_passed_both_"
        "variant_reset_gates"
    ),
    "controller_results_consulted": False,
    "reset_or_episode_artifacts_imported": False,
}
REGISTERED_DEVELOPMENT_TASKS = {
    "w1020a__oe__t0233__n0600": {
        "scenario_sha256": "d41bfb1ac8d69d90a84d8fde8701192e480f64917d75ca03dbe38f686c8178b5",
        "task_sha256": "9770356c056a691dfa4c903d3b1e4821744232fd2c465201b36aa5c45f1949cc",
    },
    "w1020a__oe__t0277__n0600": {
        "scenario_sha256": "64b7e1a97510aab9d6c790891af18a35feea95a11f69f445c0769545e39b8399",
        "task_sha256": "e41c844c0fb3b237b3d3b4d1aeb265fa881c6cb2a9a445bff7024fd45c36b1be",
    },
    "w1020a__ur__t0233__n0600": {
        "scenario_sha256": "032202ae4c1b55830e7a769cd73a9b970da666df568fa16a23594f6c6d30d331",
        "task_sha256": "d70c32f23931ddcb85d12a20279f12c8b22be51b7915826129b3d09e11e3da2d",
    },
    "w1020a__ur__t0277__n0600": {
        "scenario_sha256": "719640c2dfda7ae26a00480b3a407aa7f3b897138d83a3583ff9ce8b5fee3c51",
        "task_sha256": "a46d2eb9d97f4043c0f322eec970fa679c1191103c1f93e9e7987fadee3663fc",
    },
}
FINAL_NAMESPACE = {
    "status": "strictly_reserved_not_generated_not_reset",
    "master_seed": 20260831,
    "task_seeds": [941, 977, 1013],
    "solver_seeds": [1201, 1301, 1409],
    "planned_task_definition_count": 20,
    "planned_paired_key_count": 60,
    "exact_task_selection": (
        "deferred_until_development_winner_then_frozen_before_any_generation_or_reset"
    ),
    "generated": False,
    "reset": False,
    "may_be_touched_by_this_runner": False,
    "map_disjointness_available": False,
    "claim_boundary": "future_confirmation_must_be_fresh_and_result_blind",
}
PRODUCER_SOURCE_FILES = (
    "experiments/stride_warehouse_fixed16_development_v2.py",
    "experiments/stride_warehouse_fixed16_development.py",
    "experiments/balanced_wall_clock.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/stride_hybridstructpool_routed_confirmation.py",
    "experiments/stride_structpool_ttf_quick.py",
    "lns2_selector/runtime/structshell_single_family.py",
    "lns2_selector/runtime/topology_candidates.py",
)


def _registered_input(root: Path, specification: Mapping[str, Any], label: str) -> Path:
    return registered_input(root, dict(specification), label=label)


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != (
            "preregistered_qualification_conditioned_single_narrow_aisle_"
            "high_density_warehouse_development_ablation"
        )
        or config.get("pre_registration_parent_commit")
        != "9f6374d55113ab7d4382fb41b34046c79c8a7c91"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Warehouse fixed16 v2 development identity changed")
    if dict(config.get("runtime") or {}) != RUNTIME_CONTRACT:
        raise ValueError("Warehouse fixed16 v2 runtime contract changed")
    if dict(config.get("performance_gates") or {}) != PERFORMANCE_GATES:
        raise ValueError("Warehouse fixed16 v2 performance gates changed")
    if dict(config.get("design_provenance") or {}) != DESIGN_PROVENANCE:
        raise ValueError("Warehouse fixed16 v2 design provenance changed")
    if dict(config.get("registered_development_tasks") or {}) != REGISTERED_DEVELOPMENT_TASKS:
        raise ValueError("Warehouse fixed16 v2 registered task hashes changed")
    if dict(config.get("final_namespace") or {}) != FINAL_NAMESPACE:
        raise ValueError("Warehouse fixed16 v2 final namespace changed")

    cohort = dict(config.get("cohort") or {})
    halves = {
        str(name): tuple(tuple(map(int, pair)) for pair in pairs)
        for name, pairs in dict(cohort.get("paired_key_halves") or {}).items()
    }
    if (
        cohort.get("split") != SPLIT
        or int(cohort.get("master_seed", -1)) != 20260808
        or tuple(map(int, cohort.get("task_seeds") or ())) != TASK_SEEDS
        or tuple(map(str, cohort.get("task_variants") or ())) != TASK_VARIANTS
        or int(cohort.get("agent_count", -1)) != AGENT_COUNT
        or tuple(map(int, cohort.get("solver_seeds") or ())) != SOLVER_SEEDS
        or halves != HALVES
        or cohort.get("claim_boundary")
        != (
            "qualification_conditioned_single_narrow_aisle_high_density_"
            "warehouse_development_ablation_not_a_four_map_or_warehouse_"
            "generalization_claim"
        )
    ):
        raise ValueError("Warehouse fixed16 v2 cohort changed")
    map_spec = dict(cohort.get("map") or {})
    if map_spec.get("id") != MAP_ID or map_spec.get("code") != MAP_CODE:
        raise ValueError("Warehouse fixed16 v2 map changed")
    _registered_input(root, map_spec, "Warehouse fixed16 v2 map")
    if dict(cohort.get("qualification") or {}) != {
        "mode": "fresh_reset_only",
        "expected_reset_count": 16,
        "opposite_exchange_minimum_initial_conflicts": 16,
        "uniform_random_minimum_initial_conflicts": 1,
        "all_task_seed_solver_seed_resets_required": True,
        "no_map_load_task_or_seed_replacement": True,
        "forbidden_agent_counts": [800, 1000],
    }:
        raise ValueError("Warehouse fixed16 v2 qualification contract changed")

    contract = dict(config.get("controller_contract") or {})
    if (
        contract.get("base") != "frozen_v2_full_native_features_optimized_copeland"
        or contract.get("single_family_semantics") != "all_state_fixed_family_ablation"
        or int(contract.get("nominal_size", -1)) != 16
        or int(
            contract.get("single_family_maximum_added_candidates_per_decision", -1)
        )
        != 1
        or dict(contract.get("controller_output_codes") or {}) != CONTROLLER_CODES
        or dict(contract.get("profiles") or {}) != SINGLE_FAMILY_PROFILES
        or dict(contract.get("boundary16_static_cache") or {})
        != BOUNDARY16_STATIC_CACHE
        or set(map(str, contract.get("forbidden_interpretations") or ()))
        != {
            "high_stress_router",
            "map_or_conflict_structure_classifier",
            "structshell_boundary_subset",
        }
    ):
        raise ValueError("Warehouse fixed16 v2 controller contract changed")

    development_values = {20260808, *TASK_SEEDS, *SOLVER_SEEDS}
    reserved_values = {
        int(FINAL_NAMESPACE["master_seed"]),
        *map(int, FINAL_NAMESPACE["task_seeds"]),
        *map(int, FINAL_NAMESPACE["solver_seeds"]),
    }
    if development_values & reserved_values:
        raise ValueError("Warehouse fixed16 v2 development/final namespaces overlap")

    inputs = dict(config.get("inputs") or {})
    runtime_path = _registered_input(
        root, inputs.get("runtime_config") or {}, "Warehouse fixed16 v2 runtime"
    )
    controller_manifest = _registered_input(
        root,
        inputs.get("controller_manifest") or {},
        "Warehouse fixed16 v2 controller manifest",
    )
    runtime = _read_json(runtime_path)
    if (
        tuple(map(int, runtime.get("solver_seeds") or ())) != SOLVER_SEEDS
        or bool(runtime.get("formal"))
        or float(runtime.get("wall_time_budget_seconds", -1.0)) != 180.0
        or float(runtime.get("episode_process_timeout_seconds", -1.0)) != 240.0
        or int(dict(runtime.get("dataset_design") or {}).get("map_count", -1)) != 1
        or int(dict(runtime.get("dataset_design") or {}).get("instance_count", -1))
        != 4
    ):
        raise ValueError("Warehouse fixed16 v2 runtime input changed")
    if controller_manifest.name != "controller_manifest.json":
        raise ValueError("Warehouse fixed16 v2 controller bundle input changed")
    return path, root, config


def _guard_output(root: Path, output: str | Path) -> Path:
    resolved = Path(output).resolve()
    forbidden = (root / "build" / "wh-f16-v1-r2").resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise ValueError("Warehouse fixed16 v2 output must not reuse or nest under r2")
    return resolved


def _task_id(variant: str, task_seed: int) -> str:
    variant_code = {"opposite_exchange": "oe", "uniform_random": "ur"}[variant]
    return f"{MAP_CODE}__{variant_code}__t{task_seed:04d}__n{AGENT_COUNT:04d}"


def development_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    del config
    return [
        {
            "map_id": MAP_ID,
            "variant": variant,
            "task_seed": task_seed,
            "agent_count": AGENT_COUNT,
            "task_id": _task_id(variant, task_seed),
        }
        for task_seed in TASK_SEEDS
        for variant in TASK_VARIANTS
    ]


def qualification_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**task, "solver_seed": solver_seed}
        for task in development_task_specs(config)
        for solver_seed in SOLVER_SEEDS
    ]


def formal_schedule(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    del config
    if (
        selection.get("passed") is not True
        or str(selection.get("map_id")) != MAP_ID
        or int(selection.get("agent_count", -1)) != AGENT_COUNT
    ):
        raise ValueError("Warehouse fixed16 v2 formal requires the frozen passed cohort")
    qualification_manifest_sha256 = str(
        selection.get("qualification_manifest_sha256") or ""
    )
    qualification_report_sha256 = str(
        selection.get("qualification_report_sha256") or ""
    )
    if len(qualification_manifest_sha256) != 64 or len(qualification_report_sha256) != 64:
        raise ValueError("Warehouse fixed16 v2 formal lacks qualification hash binding")
    rows: list[dict[str, Any]] = []
    key_index = 0
    for variant in TASK_VARIANTS:
        for half in ("A", "B"):
            for task_seed, solver_seed in HALVES[half]:
                task_id = _task_id(variant, task_seed)
                offset = key_index % len(CONTROLLERS)
                for position in range(len(CONTROLLERS)):
                    controller = CONTROLLERS[(offset + position) % len(CONTROLLERS)]
                    rows.append(
                        {
                            "map_id": MAP_ID,
                            "variant": variant,
                            "agent_count": AGENT_COUNT,
                            "task_seed": task_seed,
                            "task_id": task_id,
                            "solver_seed": solver_seed,
                            "half": half,
                            "controller": controller,
                            "within_key_position": position,
                            "qualification_manifest_sha256": (
                                qualification_manifest_sha256
                            ),
                            "qualification_report_sha256": qualification_report_sha256,
                        }
                    )
                key_index += 1
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    return {
        "schema": STATUS_SCHEMA,
        "command": "plan",
        "scientific_status": config["scientific_status"],
        "map_count": 1,
        "agent_count": AGENT_COUNT,
        "dataset_task_count": len(development_task_specs(config)),
        "q0_geometry_audit_count": len(development_task_specs(config)),
        "qualification_reset_count": len(qualification_schedule(config)),
        "formal_paired_key_count_after_qualification": 8,
        "formal_episode_count_after_qualification": 8 * len(CONTROLLERS),
        "controllers": list(CONTROLLERS),
        "qualification_required_before_collect": True,
        "old_r2_artifacts_imported": False,
        "final_namespace_touched": False,
    }


def _audit_registered_dataset(dataset: Path, summary: Mapping[str, Any]) -> None:
    split = dataset / SPLIT
    manifest_path = split / "manifest.jsonl"
    q0_path = dataset / Q0_FILENAME
    if not manifest_path.is_file() or not q0_path.is_file():
        raise ValueError("Warehouse fixed16 v2 dataset audit artifacts are missing")
    manifest = _read_jsonl(manifest_path)
    split_summary = dict(dict(summary.get("splits") or {}).get(SPLIT) or {})
    if (
        summary.get("dataset_revision") != EXPERIMENT_ID
        or summary.get("source")
        != "one existing checksum-pinned MovingAI Warehouse map"
        or summary.get("task_semantics")
        != "qualification_conditioned_development_only"
        or summary.get("old_r2_artifacts_imported") is not False
        or summary.get("final_namespace_touched") is not False
        or split_summary
        != {
            "map_count": 1,
            "instance_count": 4,
            "source_counts": {"movingai": 4},
            "layout_counts": {"warehouse": 4},
        }
    ):
        raise ValueError("Warehouse fixed16 v2 dataset summary identity changed")
    by_task = {str(row.get("task_id")): dict(row) for row in manifest}
    if len(manifest) != 4 or set(by_task) != set(REGISTERED_DEVELOPMENT_TASKS):
        raise ValueError("Warehouse fixed16 v2 dataset manifest identity changed")
    if len(by_task) != len(manifest):
        raise ValueError("Warehouse fixed16 v2 dataset manifest has duplicates")
    q0 = _read_json(q0_path)
    q0_rows = {
        str(row.get("task_id")): dict(row) for row in q0.get("checks") or ()
    }
    if (
        q0.get("schema") != Q0_SCHEMA
        or q0.get("experiment_id") != EXPERIMENT_ID
        or q0.get("passed") is not True
        or int(q0.get("task_count", -1)) != 4
        or q0.get("solver_or_controller_invoked") is not False
        or q0.get("old_r2_artifacts_imported") is not False
        or q0.get("final_namespace_touched") is not False
        or set(q0_rows) != set(REGISTERED_DEVELOPMENT_TASKS)
        or len(q0_rows) != 4
        or str(summary.get("q0_geometry_audit_sha256")) != sha256_file(q0_path)
    ):
        raise ValueError("Warehouse fixed16 v2 Q0 audit identity changed")
    map_path = split / "maps" / f"{MAP_ID}.map"
    metadata_path = split / "maps" / f"{MAP_ID}.json"
    if (
        not map_path.is_file()
        or not metadata_path.is_file()
        or sha256_file(map_path) != MAP_SHA256
    ):
        raise ValueError("Warehouse fixed16 v2 dataset map bytes changed")
    metadata = _read_json(metadata_path)
    if (
        str(metadata.get("benchmark_id")) != MAP_ID
        or str(metadata.get("map_sha256")) != MAP_SHA256
        or str(metadata.get("source"))
        != "checksum-pinned existing MovingAI Warehouse map"
    ):
        raise ValueError("Warehouse fixed16 v2 map metadata changed")
    for task_id, expected_hashes in REGISTERED_DEVELOPMENT_TASKS.items():
        row = by_task[task_id]
        expected_variant = "opposite_exchange" if "__oe__" in task_id else "uniform_random"
        expected_seed = 233 if "__t0233__" in task_id else 277
        if (
            str(row.get("map_id")) != MAP_ID
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or str(row.get("split")) != SPLIT
            or str(row.get("source_group")) != "movingai"
            or str(row.get("instance_origin"))
            != "warehouse_fixed16_development_v2_derived_od"
            or str(row.get("map_file")) != f"maps/{MAP_ID}.map"
            or str(row.get("map_metadata_file")) != f"maps/{MAP_ID}.json"
            or str(row.get("scenario_type"))
            != f"warehouse_derived_{expected_variant}"
            or str(row.get("scenario_file")) != f"scenarios/{task_id}.scen"
            or str(row.get("task_file")) != f"tasks/{task_id}.json"
        ):
            raise ValueError(f"Warehouse fixed16 v2 manifest row changed: {task_id}")
        scenario = split / str(row["scenario_file"])
        task_path = split / str(row["task_file"])
        if (
            not scenario.is_file()
            or not task_path.is_file()
            or sha256_file(scenario) != expected_hashes["scenario_sha256"]
            or sha256_file(task_path) != expected_hashes["task_sha256"]
        ):
            raise ValueError(f"Warehouse fixed16 v2 registered task bytes changed: {task_id}")
        task = _read_json(task_path)
        if (
            str(task.get("benchmark_id")) != MAP_ID
            or str(task.get("od_variant")) != expected_variant
            or int(task.get("task_seed", -1)) != expected_seed
            or int(task.get("agent_count", -1)) != AGENT_COUNT
            or task.get("unique_starts") is not True
            or task.get("unique_goals") is not True
            or int(task.get("fixed_point_count", -1)) != 0
            or str(task.get("scenario_sha256")) != expected_hashes["scenario_sha256"]
        ):
            raise ValueError(f"Warehouse fixed16 v2 task metadata changed: {task_id}")
        audit = q0_rows[task_id]
        if (
            str(audit.get("variant")) != expected_variant
            or int(audit.get("task_seed", -1)) != expected_seed
            or int(audit.get("agent_count", -1)) != AGENT_COUNT
            or str(audit.get("scenario_sha256")) != expected_hashes["scenario_sha256"]
            or not isinstance(audit.get("endpoint_fingerprint"), str)
            or len(str(audit["endpoint_fingerprint"])) != 64
            or audit.get("unique_starts") is not True
            or audit.get("unique_goals") is not True
            or int(audit.get("fixed_point_count", -1)) != 0
            or audit.get("all_endpoints_reachable") is not True
            or audit.get("passed") is not True
        ):
            raise ValueError(f"Warehouse fixed16 v2 Q0 row changed: {task_id}")


def prepare_development_dataset(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = _guard_output(root, output)
    dataset = output / "dataset"
    summary_path = dataset / "dataset_summary.json"
    audit_path = dataset / Q0_FILENAME
    fingerprint = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "experiment_id": EXPERIMENT_ID,
            "map": config["cohort"]["map"],
            "master_seed": 20260808,
            "task_seeds": list(TASK_SEEDS),
            "task_variants": list(TASK_VARIANTS),
            "agent_count": AGENT_COUNT,
        }
    )
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if (
            summary.get("configuration_fingerprint") != fingerprint
        ):
            raise ValueError("Warehouse fixed16 v2 dataset/Q0 identity changed")
        _audit_registered_dataset(dataset, summary)
        return summary
    if dataset.is_dir() and any(dataset.iterdir()):
        raise ValueError("Warehouse fixed16 v2 dataset is non-empty without identity")

    split = dataset / SPLIT
    specification = dict(config["cohort"]["map"])
    source_map = _registered_input(root, specification, "Warehouse fixed16 v2 map")
    map_path = split / "maps" / f"{MAP_ID}.map"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_map, map_path)
    rows, cols, _grid, passable = _movingai_passable_cells(map_path)
    component = _largest_four_connected_component(passable)
    metrics = _map_metrics(map_path)
    metadata_path = split / "maps" / f"{MAP_ID}.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "benchmark_id": MAP_ID,
            "source": "checksum-pinned existing MovingAI Warehouse map",
            "source_path": str(specification["path"]),
            "map_sha256": sha256_file(map_path),
            "largest_four_connected_component": len(component),
            "topology_metrics": metrics,
        },
    )
    manifest: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    for task_seed in TASK_SEEDS:
        for variant in TASK_VARIANTS:
            endpoint_seed = _derived_endpoint_seed(
                20260808, MAP_ID, task_seed, variant, AGENT_COUNT
            )
            starts, goals = _derived_endpoints(
                component, AGENT_COUNT, variant, endpoint_seed
            )
            distances = _four_neighbor_distances(passable, starts, goals)
            task_id = _task_id(variant, task_seed)
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
                "schema_version": 1,
                "task_semantics_version": 1,
                "task_semantics": "development-only derived Warehouse MAPF OD",
                "benchmark_id": MAP_ID,
                "source_map_sha256": sha256_file(map_path),
                "od_variant": variant,
                "task_seed": task_seed,
                "endpoint_seed": endpoint_seed,
                "agent_count": AGENT_COUNT,
                "unique_starts": len(set(starts)) == AGENT_COUNT,
                "unique_goals": len(set(goals)) == AGENT_COUNT,
                "fixed_point_count": sum(
                    start == goal for start, goal in zip(starts, goals)
                ),
                "distance_metric": "four_neighbor_unit",
                "minimum_shortest_distance": min(distances),
                "maximum_shortest_distance": max(distances),
                "mean_shortest_distance": statistics.fmean(distances),
                "scenario_sha256": sha256_file(scenario_path),
            }
            task_path = split / "tasks" / f"{task_id}.json"
            _write_json(task_path, task_payload)
            registered_hashes = REGISTERED_DEVELOPMENT_TASKS[task_id]
            if (
                sha256_file(scenario_path)
                != registered_hashes["scenario_sha256"]
                or sha256_file(task_path) != registered_hashes["task_sha256"]
            ):
                raise RuntimeError(
                    f"Warehouse fixed16 v2 regenerated task hash changed: {task_id}"
                )
            q0_passed = bool(
                task_payload["unique_starts"]
                and task_payload["unique_goals"]
                and task_payload["fixed_point_count"] == 0
                and len(distances) == AGENT_COUNT
                and min(distances) > 0
            )
            q0_rows.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "task_seed": task_seed,
                    "agent_count": AGENT_COUNT,
                    "endpoint_fingerprint": _fingerprint(
                        {"starts": starts, "goals": goals}
                    ),
                    "scenario_sha256": task_payload["scenario_sha256"],
                    "unique_starts": task_payload["unique_starts"],
                    "unique_goals": task_payload["unique_goals"],
                    "fixed_point_count": task_payload["fixed_point_count"],
                    "all_endpoints_reachable": len(distances) == AGENT_COUNT,
                    "passed": q0_passed,
                }
            )
            manifest.append(
                {
                    "split": SPLIT,
                    "source_group": "movingai",
                    "instance_origin": "warehouse_fixed16_development_v2_derived_od",
                    "map_id": MAP_ID,
                    "task_id": task_id,
                    "map_file": f"maps/{map_path.name}",
                    "scenario_file": f"scenarios/{scenario_path.name}",
                    "map_metadata_file": f"maps/{metadata_path.name}",
                    "task_file": f"tasks/{task_path.name}",
                    "layout_mode": "warehouse",
                    "layout_variant": MAP_ID,
                    "scenario_type": f"warehouse_derived_{variant}",
                    "task_variant": (
                        f"{variant}_seed_{task_seed}_agents_{AGENT_COUNT}"
                    ),
                    "agent_count": AGENT_COUNT,
                    "topology_metrics": metrics,
                    "dominant_flow_ratio": 0.0,
                    "hotspot_skew": 0.0,
                    "required_bottleneck_crossing_ratio": 0.0,
                    "mean_shortest_distance": task_payload["mean_shortest_distance"],
                }
            )
    manifest.sort(key=lambda row: str(row["task_id"]))
    q0_rows.sort(key=lambda row: str(row["task_id"]))
    if len(manifest) != 4 or any(int(row["agent_count"]) != 600 for row in manifest):
        raise RuntimeError("Warehouse fixed16 v2 dataset dimensions changed")
    if any(value in {800, 1000} for value in (row["agent_count"] for row in manifest)):
        raise RuntimeError("Warehouse fixed16 v2 generated a forbidden load")
    q0 = {
        "schema": Q0_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_count": len(q0_rows),
        "checks": q0_rows,
        "passed": len(q0_rows) == 4 and all(row["passed"] for row in q0_rows),
        "solver_or_controller_invoked": False,
        "old_r2_artifacts_imported": False,
        "final_namespace_touched": False,
    }
    if q0["passed"] is not True:
        raise RuntimeError("Warehouse fixed16 v2 Q0 geometry audit failed")
    _write_jsonl(split / "manifest.jsonl", manifest)
    _write_json(audit_path, q0)
    summary = {
        "schema_version": 1,
        "dataset_revision": EXPERIMENT_ID,
        "configuration_fingerprint": fingerprint,
        "source": "one existing checksum-pinned MovingAI Warehouse map",
        "task_semantics": "qualification_conditioned_development_only",
        "claim_boundary": config["cohort"]["claim_boundary"],
        "q0_geometry_audit_sha256": sha256_file(audit_path),
        "splits": {
            SPLIT: {
                "map_count": 1,
                "instance_count": 4,
                "source_counts": {"movingai": 4},
                "layout_counts": {"warehouse": 4},
            }
        },
        "old_r2_artifacts_imported": False,
        "final_namespace_touched": False,
    }
    _write_json(summary_path, summary)
    _audit_registered_dataset(dataset, summary)
    return summary


def select_qualified_cohort(
    config: Mapping[str, Any], qualification_report: Mapping[str, Any]
) -> dict[str, Any]:
    expected = qualification_schedule(config)
    raw_rows = list(
        dict(qualification_report.get("natural_distribution") or {}).get("tasks", ())
    )
    index: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for raw in raw_rows:
        row = dict(raw)
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in index:
            errors.append(f"duplicate reset evidence: {key}")
        index[key] = row
    expected_keys = {(row["task_id"], row["solver_seed"]) for row in expected}
    if set(index) != expected_keys:
        errors.append("qualification reset coverage differs from frozen 16-key cohort")
    if list(qualification_report.get("errors") or ()):
        errors.append("qualification contains reset errors")
    if int(qualification_report.get("valid_count", -1)) != 16:
        errors.append("qualification valid reset count differs from 16")
    if (
        int(qualification_report.get("incomplete_reset_count", -1)) != 0
        or int(qualification_report.get("inconsistent_initial_state_count", -1)) != 0
        or dict(qualification_report.get("gates") or {}).get("all_resets_valid")
        is not True
    ):
        errors.append("qualification reset completeness/consistency gate failed")
    thresholds = {"opposite_exchange": 16, "uniform_random": 1}
    minimum_conflicts: dict[str, int | None] = {}
    for variant in TASK_VARIANTS:
        values: list[int] = []
        for task_seed in TASK_SEEDS:
            task_id = _task_id(variant, task_seed)
            for solver_seed in SOLVER_SEEDS:
                row = index.get((task_id, solver_seed))
                if row is None:
                    continue
                values.append(int(row.get("initial_conflicts", -1)))
                if not (
                    int(row.get("initial_conflicts", -1)) >= thresholds[variant]
                    and row.get("initial_feasible") is False
                    and row.get("initial_complete") is True
                    and row.get("initial_state_consistent") is True
                    and isinstance(row.get("state_fingerprint"), str)
                    and len(str(row["state_fingerprint"])) == 64
                ):
                    errors.append(
                        f"{task_id}/seed{solver_seed}: reset gate failed"
                    )
        minimum_conflicts[variant] = min(values) if values else None
    return {
        "schema": QUALIFICATION_SELECTION_SCHEMA,
        "selection_rule": "frozen_single_map_load_requires_all_16_resets",
        "passed": not errors and len(index) == 16,
        "map_id": MAP_ID,
        "agent_count": AGENT_COUNT,
        "expected_reset_count": 16,
        "observed_reset_count": len(index),
        "minimum_initial_conflicts": minimum_conflicts,
        "errors": errors,
        "claim_boundary": config["cohort"]["claim_boundary"],
        "old_r2_artifacts_imported": False,
        "final_namespace_touched": False,
    }


def _audit_qualification_evidence(
    config: Mapping[str, Any], qualification: Path
) -> None:
    manifest_path = qualification / "qualification_manifest.jsonl"
    report_path = qualification / "qualification_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError("Warehouse fixed16 v2 qualification evidence is incomplete")
    expected = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in qualification_schedule(config)
    }
    manifest_rows = _read_jsonl(manifest_path)
    manifest: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in manifest_rows:
        row = dict(raw)
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in manifest:
            raise ValueError("Warehouse fixed16 v2 qualification manifest has duplicates")
        manifest[key] = row
    report = _read_json(report_path)
    report_rows = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): dict(row)
        for row in dict(report.get("natural_distribution") or {}).get("tasks", ())
    }
    if (
        len(manifest_rows) != 16
        or set(manifest) != set(expected)
        or set(report_rows) != set(expected)
    ):
        raise ValueError("Warehouse fixed16 v2 qualification coverage changed")
    for key, item in expected.items():
        row = manifest[key]
        anchor = report_rows[key]
        expected_task_variant = (
            f"{item['variant']}_seed_{item['task_seed']}_agents_{AGENT_COUNT}"
        )
        if (
            str(row.get("status")) != "ok"
            or row.get("error") is not None
            or str(row.get("map_id")) != MAP_ID
            or str(row.get("split")) != SPLIT
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or str(row.get("task_variant")) != expected_task_variant
        ):
            raise ValueError(
                f"Warehouse fixed16 v2 qualification manifest row changed: {key}"
            )
        # The qualification manifest is the raw reset record and does not carry
        # ``initial_state_consistent``.  That field is computed by the report
        # builder after cross-seed fingerprint auditing, so require it directly
        # on the report instead of comparing it with a missing manifest field.
        if anchor.get("initial_state_consistent") is not True:
            raise ValueError(
                "Warehouse fixed16 v2 qualification report marks an "
                f"inconsistent initial state: {key}"
            )
        if (
            "initial_state_consistent" in row
            and row.get("initial_state_consistent") is not True
        ):
            raise ValueError(
                "Warehouse fixed16 v2 qualification manifest carries an "
                f"inconsistent initial state: {key}"
            )
        for field in (
            "initial_conflicts",
            "initial_feasible",
            "initial_complete",
            "state_fingerprint",
        ):
            if row.get(field) != anchor.get(field):
                raise ValueError(
                    "Warehouse fixed16 v2 qualification report/manifest mismatch: "
                    f"{key}/{field}"
                )


def qualify(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    if dry_run:
        return plan(path)
    output = _guard_output(root, output)
    prepare_development_dataset(path, output)
    qualification = output / "qualification"
    keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in qualification_schedule(config)
    }
    runtime_path = _registered_input(
        root, dict(config["inputs"])["runtime_config"], "Warehouse fixed16 v2 runtime"
    )
    run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        qualification,
        phase="qualify",
        workers=16,
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_process_timeout_seconds=240.0,
        **legacy.controller_kwargs(root, config, "v2_only"),
    )
    report_path = qualification / "qualification_report.json"
    manifest_path = qualification / "qualification_manifest.jsonl"
    if not manifest_path.is_file():
        raise RuntimeError("Warehouse fixed16 v2 qualification manifest is missing")
    _audit_qualification_evidence(config, qualification)
    selection = select_qualified_cohort(config, _read_json(report_path))
    selection.update(
        {
            "config_sha256": sha256_file(path),
            "q0_geometry_audit_sha256": sha256_file(
                output / "dataset" / Q0_FILENAME
            ),
            "qualification_report_sha256": sha256_file(report_path),
            "qualification_manifest_sha256": sha256_file(manifest_path),
        }
    )
    _write_json(output / SELECTION_FILENAME, selection)
    return selection


def _load_selection(
    config_path: Path, output: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    selection_path = output / SELECTION_FILENAME
    qualification_path = output / "qualification" / "qualification_report.json"
    qualification_manifest_path = output / "qualification" / "qualification_manifest.jsonl"
    q0_path = output / "dataset" / Q0_FILENAME
    if (
        not selection_path.is_file()
        or not qualification_path.is_file()
        or not qualification_manifest_path.is_file()
        or not q0_path.is_file()
    ):
        raise ValueError("Warehouse fixed16 v2 qualification/Q0 evidence is missing")
    q0 = _read_json(q0_path)
    if q0.get("schema") != Q0_SCHEMA or q0.get("passed") is not True:
        raise ValueError("Warehouse fixed16 v2 Q0 geometry audit did not pass")
    _audit_qualification_evidence(config, output / "qualification")
    selection = _read_json(selection_path)
    recomputed = select_qualified_cohort(config, _read_json(qualification_path))
    recomputed.update(
        {
            "config_sha256": sha256_file(config_path),
            "q0_geometry_audit_sha256": sha256_file(q0_path),
            "qualification_report_sha256": sha256_file(qualification_path),
            "qualification_manifest_sha256": sha256_file(
                qualification_manifest_path
            ),
        }
    )
    if selection != recomputed or recomputed.get("passed") is not True:
        raise ValueError(
            "Warehouse fixed16 v2 formal is blocked by changed or failed qualification"
        )
    return selection


def _producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=PRODUCER_SOURCE_FILES,
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    path, root, config = load_config(job["config_path"])
    output = _guard_output(root, job["output_root"])
    selection = _load_selection(path, output, config)
    items = formal_schedule(config, selection)
    item = dict(job["item"])
    controller = str(item["controller"])
    collection = legacy._controller_dir(output, controller)
    cohort_keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in items
        if row["controller"] == controller
    }
    kwargs = legacy.controller_kwargs(root, config, controller)
    runtime_path = _registered_input(
        root, dict(config["inputs"])["runtime_config"], "Warehouse fixed16 v2 runtime"
    )
    run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=cohort_keys,
        job_keys=cohort_keys,
        qualification_source=output / "qualification",
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        cohort_job_keys=cohort_keys,
        job_keys={(str(item["task_id"]), int(item["solver_seed"]))},
        qualification_source=output / "qualification",
        use_global_collection_lock=False,
        **kwargs,
    )
    row = legacy._manifest(
        output, controller, str(item["task_id"]), int(item["solver_seed"])
    )
    if row is None:
        raise RuntimeError("Warehouse fixed16 v2 episode completed without manifest")
    status = str(row.get("status"))
    return {**item, "status": status if status in {"error", "timeout"} else "ok"}


def _audit_formal_identity(
    config_path: Path,
    formal: Path,
    items: list[dict[str, Any]],
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    status_path = formal / STATUS_FILENAME
    runner_path = formal / "runner_config.json"
    schedule_path = formal / "execution_schedule.jsonl"
    if not status_path.is_file() or not runner_path.is_file() or not schedule_path.is_file():
        raise ValueError("Warehouse fixed16 v2 formal identity artifacts are incomplete")
    status = _read_json(status_path)
    runner = _read_json(runner_path)
    schedule_rows = _read_jsonl(schedule_path)
    schedule_sha = _fingerprint(items)
    expected_identity = {
        "schema": RESUMABLE_RUN_IDENTITY_SCHEMA,
        "status_schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "schedule_sha256": schedule_sha,
        "producer_identity": dict(producer),
        "report_schema": REPORT_SCHEMA,
    }
    if (
        schedule_rows != items
        or _fingerprint(schedule_rows) != schedule_sha
        or runner.get("schema") != RUNNER_CONFIG_SCHEMA
        or runner.get("identity") != expected_identity
        or runner.get("identity_fingerprint") != _fingerprint(expected_identity)
        or status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != expected_identity["config_sha256"]
        or status.get("schedule_sha256") != schedule_sha
        or status.get("producer_identity") != dict(producer)
        or status.get("run_fingerprint") != runner.get("identity_fingerprint")
        or int(status.get("total_schedule_entries", -1)) != len(items)
    ):
        raise ValueError("Warehouse fixed16 v2 formal producer/run identity changed")
    return status


def collect(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = _guard_output(root, output)
    if dry_run:
        return plan(path)
    prepare_development_dataset(path, output)
    selection = _load_selection(path, output, config)
    items = formal_schedule(config, selection)
    formal = output / "formal"
    producer = _producer(root)
    prepared = prepare_resumable_output(
        formal,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=items,
        producer=producer,
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="Warehouse fixed16 v2 narrow development ablation",
    )
    if prepared.status.get("terminal_failure") is not None:
        raise RuntimeError("Warehouse fixed16 v2 has terminal failure; resume forbidden")
    failures = legacy._terminal_manifest_failures(output)
    if failures:
        raise RuntimeError(
            "Warehouse fixed16 v2 contains error/timeout manifest; resume forbidden: "
            f"{failures[0]}"
        )
    legacy._audit_inner_collections(
        root,
        config,
        output,
        items,
        require_all=prepared.completed_report is not None,
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    completed_prefix = legacy._completed_schedule_prefix_length(output, items)
    pending = items[completed_prefix:]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase="warehouse-fixed16-development-v2",
            output_root=formal,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=300.0,
            failure_result=legacy._failed_episode_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row.get("status") in {"error", "timeout"}]
        if terminal:
            status = legacy._status(output, items, prepared.base_status)
            status["terminal_failure"] = terminal[0]
            _write_json(formal / STATUS_FILENAME, status)
            return status
    report = analyze(path, output, producer=producer)
    status = legacy._status(output, items, prepared.base_status, complete=True)
    status["report_sha256"] = sha256_file(formal / REPORT_FILENAME)
    _write_json(formal / STATUS_FILENAME, status)
    return report


def _key(item: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(item["map_id"]), str(item["task_id"]), int(item["solver_seed"])


def _registered_next_step(selected_controller: str | None) -> str:
    if selected_controller == "boundary16_static_cache":
        return (
            "freeze_boundary16_development_winner_then_preregister_fresh_final_"
            "confirmation_with_official_adaptive"
        )
    if selected_controller in SINGLE_FAMILY_PROFILES:
        return (
            "freeze_single_family_development_winner_then_preregister_independent_"
            "size_study_8_16_24_32_before_any_final_confirmation"
        )
    return "stop_fixed16_narrow_slice_branch_keep_v2_default"


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = _guard_output(root, output)
    formal = output / "formal"
    expected_producer = _producer(root)
    selection = _load_selection(path, output, config)
    items = formal_schedule(config, selection)
    _audit_formal_identity(path, formal, items, expected_producer)
    inner_run_fingerprints = legacy._audit_inner_collections(
        root, config, output, items, require_all=True
    )
    completed = load_completed_report(
        formal,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        if completed.get("producer_identity") != expected_producer:
            raise ValueError("Warehouse fixed16 v2 completed report producer changed")
        return completed

    expected = {_key(item) for item in items}
    half_keys = {
        half: {_key(item) for item in items if item["half"] == half}
        for half in HALVES
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    manifest_hashes: dict[str, str] = {}
    for controller in CONTROLLERS:
        manifest = legacy._manifest_path(output, controller)
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        if not manifest.is_file():
            errors.append(f"{controller}: missing manifest")
        else:
            manifest_hashes[controller] = sha256_file(manifest)
            for row in _read_jsonl(manifest):
                key = (MAP_ID, str(row.get("task_id")), int(row.get("solver_seed", -1)))
                if key in rows:
                    errors.append(f"{controller}: duplicate {key}")
                rows[key] = legacy._normalized_manifest_row(row)
        if set(rows) != expected:
            errors.append(f"{controller}: incomplete or extra paired coverage")
        indexed[controller] = rows

    qualification_report = _read_json(
        output / "qualification" / "qualification_report.json"
    )
    qualification_index = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in dict(qualification_report.get("natural_distribution") or {}).get(
            "tasks", ()
        )
    }
    fingerprint_mismatches = 0
    conflict_mismatches = 0
    qualification_fingerprint_mismatches = 0
    qualification_conflict_mismatches = 0
    missing_qualification_anchors = 0
    bad_clock = 0
    bad_capped = 0
    bad_stop_reason = 0
    accepted_stops = {"success", "wall_timeout", "controller_stalled", "native_terminal"}
    for key in sorted(expected):
        rows = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len(
            {row.get("initial_fingerprint") for row in summaries}
        ) != 1
        conflict_mismatches += len(
            {row.get("initial_conflicts") for row in summaries}
        ) != 1
        anchor = qualification_index.get((key[1], key[2]))
        if anchor is None:
            missing_qualification_anchors += 1
        else:
            qualification_fingerprint_mismatches += sum(
                row.get("initial_fingerprint") != anchor.get("state_fingerprint")
                for row in summaries
            )
            qualification_conflict_mismatches += sum(
                int(row.get("initial_conflicts", -1))
                != int(anchor.get("initial_conflicts", -2))
                for row in summaries
            )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries
        )
        bad_capped += sum(
            row.get("capped_wall_time_to_feasible") is None for row in summaries
        )
        bad_stop_reason += sum(
            str(row.get("stop_reason")) not in accepted_stops for row in summaries
        )

    summaries = {
        name: _bounded_summary(list(indexed[name].values())) for name in CONTROLLERS
    }
    report_metrics = {
        name: legacy._controller_report_metrics(list(indexed[name].values()), name)
        for name in CONTROLLERS
    }
    integrity = {
        "qualification_selection_passed": selection.get("passed") is True,
        "q0_geometry_audit_passed": _read_json(
            output / "dataset" / Q0_FILENAME
        ).get("passed")
        is True,
        "complete_paired_coverage": not any(
            "coverage" in error or "missing" in error for error in errors
        ),
        "zero_execution_errors_or_process_timeouts": all(
            row.get("status") == "ok"
            for values in indexed.values()
            for row in values.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "formal_initial_fingerprints_match_qualification": (
            missing_qualification_anchors == 0
            and qualification_fingerprint_mismatches == 0
        ),
        "formal_initial_conflicts_match_qualification": (
            missing_qualification_anchors == 0
            and qualification_conflict_mismatches == 0
        ),
        "registered_reset_inclusive_ttf_clock": bad_clock == 0,
        "bounded_ttf_values_complete": bad_capped == 0,
        "valid_bounded_stop_reasons": bad_stop_reason == 0,
        "zero_invalid_actions": all(
            summary.get("invalid_action_count") == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary.get("fingerprint_mismatch_count") == 0
            for summary in summaries.values()
        ),
        "strict_rotating_serial_schedule": len(items) == 48
        and all(
            {row["controller"] for row in items[index : index + 6]}
            == set(CONTROLLERS)
            and [row["within_key_position"] for row in items[index : index + 6]]
            == list(range(6))
            for index in range(0, len(items), 6)
        ),
        "single_registered_map_and_load": all(
            item["map_id"] == MAP_ID and item["agent_count"] == AGENT_COUNT
            for item in items
        ),
        "old_r2_artifacts_not_imported": True,
        "final_namespace_untouched_by_runner": True,
    }
    integrity_passed = not errors and all(integrity.values())
    keys = sorted(expected)
    overall_comparisons: dict[str, dict[str, Any]] = {}
    half_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for challenger in CHALLENGERS:
        overall = _bounded_paired_comparison(
            indexed["v2_only"], indexed[challenger], keys
        )
        halves = {
            half: _bounded_paired_comparison(
                indexed["v2_only"], indexed[challenger], sorted(half_keys[half])
            )
            for half in HALVES
        }
        overall_comparisons[challenger] = overall
        half_comparisons[challenger] = halves
        gates[challenger] = legacy.performance_gate(
            integrity_passed=integrity_passed,
            v2_summary=summaries["v2_only"],
            challenger_summary=summaries[challenger],
            overall=overall,
            half_a=halves["A"],
            half_b=halves["B"],
        )
    variant_by_key = {_key(item): str(item["variant"]) for item in items}
    diagnostics = {
        variant: {
            "paired_key_count": len(
                [key for key in expected if variant_by_key[key] == variant]
            ),
            "controller_summaries": {
                controller: _bounded_summary(
                    [
                        indexed[controller][key]
                        for key in sorted(expected)
                        if variant_by_key[key] == variant
                        and key in indexed[controller]
                    ]
                )
                for controller in CONTROLLERS
            },
        }
        for variant in TASK_VARIANTS
    }
    winner = legacy.choose_winner(
        gates,
        overall_comparisons,
        report_metrics,
        relative_margin=float(config["performance_gates"]["near_tie_relative_margin"]),
    )
    if producer is None:
        producer = expected_producer
    elif producer != expected_producer:
        raise ValueError("Warehouse fixed16 v2 analysis producer identity changed")
    next_step = _registered_next_step(winner["selected_controller"])
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "qualification_conditioned_single_narrow_aisle_high_density_"
            "warehouse_development_ablation_not_a_speed_or_generalization_claim"
        ),
        "map_count": 1,
        "map_id": MAP_ID,
        "agent_count": AGENT_COUNT,
        "paired_key_count": 8,
        "episode_count": sum(len(rows) for rows in indexed.values()),
        "controllers": list(CONTROLLERS),
        "official_adaptive_included": False,
        "official_adaptive_policy": (
            "boundary16_winner_enters_fresh_final_confirmation_directly;single_"
            "family_winner_requires_independent_8_16_24_32_size_study_and_frozen_"
            "size_winner_before_fresh_final_confirmation"
        ),
        "claim_boundary": config["cohort"]["claim_boundary"],
        "controller_summaries": summaries,
        "report_only_metrics": report_metrics,
        "overall_comparisons_vs_v2": overall_comparisons,
        "half_comparisons_vs_v2": half_comparisons,
        "diagnostic_by_variant": diagnostics,
        "integrity_gates": integrity,
        "challenger_gates": gates,
        "integrity_passed": integrity_passed,
        "development_gate_passed": winner["selected_controller"] is not None,
        **winner,
        "formal_speed_claim": False,
        "four_map_or_warehouse_generalization_claim": False,
        "default_replacement_allowed": False,
        "old_r2_artifacts_imported": False,
        "final_namespace_touched": False,
        "next_step": next_step,
        "errors": errors,
        "producer_identity": producer,
        "design_provenance": DESIGN_PROVENANCE,
        "inputs": {
            "config_sha256": sha256_file(path),
            "q0_geometry_audit_sha256": sha256_file(
                output / "dataset" / Q0_FILENAME
            ),
            "selection_sha256": sha256_file(output / SELECTION_FILENAME),
            "schedule_sha256": sha256_file(formal / "execution_schedule.jsonl"),
            "controller_manifest_sha256": manifest_hashes,
            "controller_inner_run_fingerprints": inner_run_fingerprints,
        },
    }
    _write_json(formal / REPORT_FILENAME, report)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return plan(config_path)
    selection = qualify(config_path, output, resume=resume)
    if selection.get("passed") is not True:
        return selection
    return collect(config_path, output, resume=resume)


__all__ = [
    "AGENT_COUNT",
    "CHALLENGERS",
    "CONTROLLERS",
    "DESIGN_PROVENANCE",
    "FINAL_NAMESPACE",
    "HALVES",
    "MAP_ID",
    "Q0_SCHEMA",
    "REPORT_SCHEMA",
    "SOLVER_SEEDS",
    "STATUS_SCHEMA",
    "TASK_SEEDS",
    "TASK_VARIANTS",
    "analyze",
    "collect",
    "development_task_specs",
    "formal_schedule",
    "load_config",
    "plan",
    "prepare_development_dataset",
    "qualification_schedule",
    "qualify",
    "run",
    "select_qualified_cohort",
]
