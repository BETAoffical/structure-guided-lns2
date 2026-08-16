from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    episode_id as closed_loop_episode_id,
    registered_input,
    sha256_file,
)
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
    _repair_tail,
)
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.evaluation.trace_validation import validate_closed_loop_trace
from lns2_selector.runtime.structshell_single_family import (
    structshell_single_family_augmentation,
    validate_structshell_single_family_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_fixed16_development_config.v1"
STATUS_SCHEMA = "lns2.stride.warehouse_fixed16_development_status.v1"
REPORT_SCHEMA = "lns2.stride.warehouse_fixed16_development_report.v1"
QUALIFICATION_SELECTION_SCHEMA = (
    "lns2.stride.warehouse_fixed16_qualification_selection.v1"
)
EXPERIMENT_ID = "stride-warehouse-fixed16-development-v1"
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "development_report.json"
SELECTION_FILENAME = "selected_cohort.json"

CONTROLLERS = (
    "v2_only",
    "boundary16_static_cache",
    "bottleneck16",
    "component16",
    "hotspot16",
    "path_overlap16",
)
CHALLENGERS = CONTROLLERS[1:]
CONTROLLER_CODES = {
    "v2_only": "v2",
    "boundary16_static_cache": "b16",
    "bottleneck16": "bn16",
    "component16": "cc16",
    "hotspot16": "hs16",
    "path_overlap16": "po16",
}
SINGLE_FAMILY_PROFILES = {
    "bottleneck16": "bottleneck",
    "component16": "conflict_component",
    "hotspot16": "hotspot",
    "path_overlap16": "path_overlap",
}
MAP_IDS = (
    "warehouse-10-20-10-2-1",
    "warehouse-10-20-10-2-2",
    "warehouse-20-40-10-2-1",
    "warehouse-20-40-10-2-2",
)
MAP_CODES = {
    "warehouse-10-20-10-2-1": "w1020a",
    "warehouse-10-20-10-2-2": "w1020b",
    "warehouse-20-40-10-2-1": "w2040a",
    "warehouse-20-40-10-2-2": "w2040b",
}
TASK_SEEDS = (233, 277)
TASK_VARIANTS = ("opposite_exchange", "uniform_random")
AGENT_COUNT_LADDER = (400, 600, 800, 1000)
SOLVER_SEEDS = (19, 20, 21, 22)
HALVES = {
    "A": ((233, 19), (277, 21)),
    "B": ((233, 20), (277, 22)),
}
BOUNDARY16_STATIC_CACHE = {
    "enabled": True,
    "generator_id": "stride-topoboundary-v1",
    "neighborhood_size": 16,
    "core_budget": 4,
    "maximum_added_candidates": 2,
    "runtime_id": "stride-boundary-static-cache-v1",
    "static_grid_cache": True,
}
RUNTIME_CONTRACT = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 180.0,
    "environment_time_limit_seconds": 180.0,
    "episode_process_timeout_seconds": 240.0,
    "outer_job_timeout_seconds": 300.0,
    "workers_for_qualification": 16,
    "workers_for_timed_episodes": 1,
    "execution_order": "rotating_strict_six_controller_serial",
}
PERFORMANCE_GATES = {
    "integrity_and_zero_execution_errors_required": True,
    "success_count_not_lower_than_v2": True,
    "mean_restricted_ttf_strictly_lower_than_v2": True,
    "half_a_mean_restricted_ttf_strictly_lower_than_v2": True,
    "half_b_mean_restricted_ttf_strictly_lower_than_v2": True,
    "report_only_not_gates": [
        "normalized_wall_clock_conflict_auc",
        "repair_iterations",
        "pp_replan_seconds",
        "neighborhood_selection_seconds",
        "paired_faster_fraction",
    ],
    "winner": "lowest_mean_restricted_ttf",
    "near_tie_relative_margin": 0.01,
    "near_tie_breakers": [
        "lowest_mean_added_candidate_count",
        "lowest_mean_neighborhood_selection_seconds",
        "controller_name",
    ],
    "tie_metric_definitions": {
        "mean_added_candidate_count": (
            "total_source_candidate_count_divided_by_controller_decisions"
        ),
        "mean_neighborhood_selection_seconds": (
            "episode_mean_of_total_neighborhood_selection_seconds"
        ),
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
    "claim_boundary": (
        "future_confirmation_is_task_and_solver_result_blind_on_the_same_four_"
        "available_standard_warehouse_layouts"
    ),
}


def _registered_warehouse_input(
    root: Path, specification: Mapping[str, Any], label: str
) -> Path:
    return registered_input(root, dict(specification), label=label)


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_development_screen_not_a_speed_or_runtime_claim"
        or config.get("pre_registration_parent_commit")
        != "b2806fb4b9092aae1489ee1dc1e73cbe4920a1f0"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Warehouse fixed16 development identity changed")
    if dict(config.get("runtime") or {}) != RUNTIME_CONTRACT:
        raise ValueError("Warehouse fixed16 runtime contract changed")
    if dict(config.get("performance_gates") or {}) != PERFORMANCE_GATES:
        raise ValueError("Warehouse fixed16 performance gates changed")

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
        or tuple(map(int, cohort.get("agent_count_ladder") or ()))
        != AGENT_COUNT_LADDER
        or tuple(map(int, cohort.get("solver_seeds") or ())) != SOLVER_SEEDS
        or halves != HALVES
    ):
        raise ValueError("Warehouse fixed16 development cohort changed")
    maps = list(cohort.get("maps") or ())
    if (
        tuple(str(value.get("id")) for value in maps) != MAP_IDS
        or {str(value["id"]): str(value.get("code")) for value in maps}
        != MAP_CODES
    ):
        raise ValueError("Warehouse fixed16 map cohort changed")
    for specification in maps:
        _registered_warehouse_input(
            root, specification, f"Warehouse map {specification.get('id')}"
        )

    contract = dict(config.get("controller_contract") or {})
    if (
        contract.get("base") != "frozen_v2_full_native_features_optimized_copeland"
        or contract.get("single_family_semantics")
        != "all_state_fixed_family_ablation"
        or int(contract.get("nominal_size", -1)) != 16
        or int(contract.get("maximum_added_candidates_per_decision", -1)) != 1
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
        raise ValueError("Warehouse fixed16 controller contract changed")

    qualification = dict(cohort.get("qualification") or {})
    if qualification != {
        "mode": "reset_only",
        "selection": "per_map_lowest_common_load_for_both_variants",
        "opposite_exchange_minimum_initial_conflicts": 16,
        "uniform_random_minimum_initial_conflicts": 1,
        "all_task_seed_solver_seed_resets_required": True,
        "no_map_or_load_replacement": True,
    }:
        raise ValueError("Warehouse fixed16 qualification rule changed")

    final_namespace = dict(config.get("final_namespace") or {})
    if final_namespace != FINAL_NAMESPACE:
        raise ValueError("Warehouse fixed16 final namespace changed")
    development_values = {
        20260808,
        *TASK_SEEDS,
        *SOLVER_SEEDS,
    }
    reserved_values = {
        int(final_namespace["master_seed"]),
        *map(int, final_namespace["task_seeds"]),
        *map(int, final_namespace["solver_seeds"]),
    }
    if development_values & reserved_values:
        raise ValueError("development and final namespaces overlap")

    inputs = dict(config.get("inputs") or {})
    runtime_path = _registered_warehouse_input(
        root, inputs.get("runtime_config") or {}, "runtime config"
    )
    controller_manifest = _registered_warehouse_input(
        root, inputs.get("controller_manifest") or {}, "V2 controller manifest"
    )
    runtime = _read_json(runtime_path)
    if (
        tuple(map(int, runtime.get("solver_seeds") or ())) != SOLVER_SEEDS
        or bool(runtime.get("formal"))
        or float(runtime.get("wall_time_budget_seconds", -1.0)) != 180.0
        or float(runtime.get("episode_process_timeout_seconds", -1.0)) != 240.0
        or int(dict(runtime.get("dataset_design") or {}).get("map_count", -1)) != 4
        or int(dict(runtime.get("dataset_design") or {}).get("instance_count", -1))
        != 64
    ):
        raise ValueError("Warehouse fixed16 runtime input changed")
    if controller_manifest.name != "controller_manifest.json":
        raise ValueError("Warehouse fixed16 controller bundle input changed")
    return path, root, config


def _task_id(map_id: str, variant: str, task_seed: int, agent_count: int) -> str:
    variant_code = {"opposite_exchange": "oe", "uniform_random": "ur"}[variant]
    return (
        f"{MAP_CODES[map_id]}__{variant_code}__t{task_seed:04d}"
        f"__n{agent_count:04d}"
    )


def development_task_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for map_spec in config["cohort"]["maps"]:
        map_id = str(map_spec["id"])
        for task_seed in TASK_SEEDS:
            for variant in TASK_VARIANTS:
                for agent_count in AGENT_COUNT_LADDER:
                    rows.append(
                        {
                            "map_id": map_id,
                            "variant": variant,
                            "task_seed": task_seed,
                            "agent_count": agent_count,
                            "task_id": _task_id(
                                map_id, variant, task_seed, agent_count
                            ),
                        }
                    )
    return rows


def qualification_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**task, "solver_seed": solver_seed}
        for task in development_task_specs(config)
        for solver_seed in SOLVER_SEEDS
    ]


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    tasks = development_task_specs(config)
    qualification = qualification_schedule(config)
    return {
        "schema": STATUS_SCHEMA,
        "command": "plan",
        "map_count": len(config["cohort"]["maps"]),
        "dataset_task_count": len(tasks),
        "qualification_reset_count": len(qualification),
        "formal_paired_key_count_after_qualification": 32,
        "formal_episode_count_after_qualification": 32 * len(CONTROLLERS),
        "controllers": list(CONTROLLERS),
        "final_namespace_touched": False,
    }


def prepare_development_dataset(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    dataset = output / "dataset"
    summary_path = dataset / "dataset_summary.json"
    fingerprint = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "master_seed": config["cohort"]["master_seed"],
            "maps": config["cohort"]["maps"],
            "task_seeds": list(TASK_SEEDS),
            "task_variants": list(TASK_VARIANTS),
            "agent_count_ladder": list(AGENT_COUNT_LADDER),
        }
    )
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("configuration_fingerprint") != fingerprint:
            raise ValueError("Warehouse development dataset identity changed")
        manifest = _read_jsonl(dataset / SPLIT / "manifest.jsonl")
        if len(manifest) != 64:
            raise ValueError("Warehouse development dataset is incomplete")
        return summary
    if dataset.is_dir() and any(dataset.iterdir()):
        raise ValueError("Warehouse development dataset is non-empty without identity")

    split = dataset / SPLIT
    manifest: list[dict[str, Any]] = []
    map_specs = {str(row["id"]): dict(row) for row in config["cohort"]["maps"]}
    for map_id in MAP_IDS:
        specification = map_specs[map_id]
        source_map = _registered_warehouse_input(
            root, specification, f"Warehouse map {map_id}"
        )
        map_path = split / "maps" / f"{map_id}.map"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_map, map_path)
        rows, cols, _grid, passable = _movingai_passable_cells(map_path)
        component = _largest_four_connected_component(passable)
        metrics = _map_metrics(map_path)
        metadata_path = split / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "checksum-pinned existing MovingAI Warehouse map",
                "source_path": str(specification["path"]),
                "map_sha256": sha256_file(map_path),
                "largest_four_connected_component": len(component),
                "topology_metrics": metrics,
            },
        )
        for task_seed in TASK_SEEDS:
            for variant in TASK_VARIANTS:
                for agent_count in AGENT_COUNT_LADDER:
                    endpoint_seed = _derived_endpoint_seed(
                        20260808, map_id, task_seed, variant, agent_count
                    )
                    starts, goals = _derived_endpoints(
                        component, agent_count, variant, endpoint_seed
                    )
                    distances = _four_neighbor_distances(passable, starts, goals)
                    task_id = _task_id(map_id, variant, task_seed, agent_count)
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
                    task_path = split / "tasks" / f"{task_id}.json"
                    task_payload = {
                        "schema_version": 1,
                        "task_semantics_version": 1,
                        "task_semantics": "development-only derived Warehouse MAPF OD",
                        "benchmark_id": map_id,
                        "source_map_sha256": sha256_file(map_path),
                        "od_variant": variant,
                        "task_seed": task_seed,
                        "endpoint_seed": endpoint_seed,
                        "agent_count": agent_count,
                        "unique_starts": len(set(starts)) == agent_count,
                        "unique_goals": len(set(goals)) == agent_count,
                        "fixed_point_count": sum(
                            start == goal for start, goal in zip(starts, goals)
                        ),
                        "distance_metric": "four_neighbor_unit",
                        "minimum_shortest_distance": min(distances),
                        "maximum_shortest_distance": max(distances),
                        "mean_shortest_distance": statistics.fmean(distances),
                        "scenario_sha256": sha256_file(scenario_path),
                    }
                    _write_json(task_path, task_payload)
                    manifest.append(
                        {
                            "split": SPLIT,
                            "source_group": "movingai",
                            "instance_origin": "warehouse_fixed16_development_derived_od",
                            "map_id": map_id,
                            "task_id": task_id,
                            "map_file": f"maps/{map_path.name}",
                            "scenario_file": f"scenarios/{scenario_path.name}",
                            "map_metadata_file": f"maps/{metadata_path.name}",
                            "task_file": f"tasks/{task_path.name}",
                            "layout_mode": "warehouse",
                            "layout_variant": map_id,
                            "scenario_type": f"warehouse_derived_{variant}",
                            "task_variant": (
                                f"{variant}_seed_{task_seed}_agents_{agent_count}"
                            ),
                            "agent_count": agent_count,
                            "topology_metrics": metrics,
                            "dominant_flow_ratio": 0.0,
                            "hotspot_skew": 0.0,
                            "required_bottleneck_crossing_ratio": 0.0,
                            "mean_shortest_distance": task_payload[
                                "mean_shortest_distance"
                            ],
                        }
                    )
    manifest.sort(key=lambda row: str(row["task_id"]))
    if len(manifest) != 64:
        raise RuntimeError("Warehouse development dataset dimensions changed")
    _write_jsonl(split / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": "stride-warehouse-fixed16-development-v1",
        "configuration_fingerprint": fingerprint,
        "source": "four existing checksum-pinned MovingAI Warehouse maps",
        "task_semantics": "derived_development_only_not_official_scenarios",
        "claim_boundary": config["cohort"]["claim_boundary"],
        "splits": {
            SPLIT: {
                "map_count": 4,
                "instance_count": 64,
                "source_counts": {"movingai": 64},
                "layout_counts": {"warehouse": 64},
            }
        },
        "final_namespace_touched": False,
    }
    _write_json(summary_path, summary)
    return summary


def _v2_kwargs(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    controller_manifest = _registered_warehouse_input(
        root,
        dict(config["inputs"])["controller_manifest"],
        "V2 controller manifest",
    )
    return {
        "controller": "v2-full",
        "controller_bundle": str(controller_manifest.parent),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    result = _v2_kwargs(root, config)
    if controller == "v2_only":
        return result
    if controller == "boundary16_static_cache":
        result["topology_boundary_augmentation"] = dict(
            BOUNDARY16_STATIC_CACHE
        )
        return result
    try:
        profile = SINGLE_FAMILY_PROFILES[controller]
    except KeyError as error:
        raise ValueError(f"unknown Warehouse fixed16 controller: {controller}") from error
    augmentation = structshell_single_family_augmentation(profile, 16)
    validated = validate_structshell_single_family_augmentation(augmentation)
    if (
        validated is None
        or validated.get("activation_gate")
        != {"gate_id": "single_family_ablation_all_states"}
        or validated.get("source_mode") != "structshell_single_family"
        or validated.get("nominal_size") != 16
    ):
        raise ValueError("single-family runtime is not an all-state fixed16 ablation")
    result["hybridstructpool_augmentation"] = validated
    return result


def _controller_cohort_keys(
    items: list[dict[str, Any]], controller: str
) -> set[tuple[str, int]]:
    return {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in items
        if row["controller"] == controller
    }


def _expected_inner_run(
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    controller: str,
    cohort_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    runtime_path = _registered_warehouse_input(
        root, dict(config["inputs"])["runtime_config"], "runtime config"
    )
    return run_closed_loop_collection(
        output / "dataset",
        runtime_path,
        _controller_dir(output, controller),
        phase="realized_dynamic",
        workers=1,
        dry_run=True,
        cohort_job_keys=cohort_keys,
        job_keys=cohort_keys,
        qualification_source=output / "qualification",
        use_global_collection_lock=False,
        **controller_kwargs(root, config, controller),
    )


def _validate_inner_run_config_payload(
    payload: Mapping[str, Any],
    *,
    expected_run_fingerprint: str,
    controller: str,
    cohort_keys: set[tuple[str, int]],
) -> None:
    configuration = dict(payload.get("configuration") or {})
    proposal = dict(configuration.get("proposal") or {})
    observed_keys = {
        (str(value[0]), int(value[1]))
        for value in configuration.get("cohort_job_keys_override") or ()
        if isinstance(value, list) and len(value) == 2
    }
    expected_boundary = (
        BOUNDARY16_STATIC_CACHE if controller == "boundary16_static_cache" else None
    )
    expected_hybrid = (
        structshell_single_family_augmentation(
            SINGLE_FAMILY_PROFILES[controller], 16
        )
        if controller in SINGLE_FAMILY_PROFILES
        else None
    )
    if (
        str(payload.get("run_fingerprint")) != expected_run_fingerprint
        or str(configuration.get("controller")) != "v2-full"
        or str(configuration.get("feature_backend")) != "native"
        or str(configuration.get("controller_runtime")) != "optimized"
        or str(configuration.get("verification_profile")) != "deployment"
        or str(configuration.get("stopping_rule")) != "wall-clock"
        or str(configuration.get("repair_seed_policy")) != "episode_stream"
        or configuration.get("deterministic_pp_replay") is not False
        or float(configuration.get("wall_time_budget_seconds", -1.0)) != 180.0
        or float(configuration.get("episode_process_timeout_seconds", -1.0))
        != 240.0
        or float(dict(configuration.get("environment") or {}).get("time_limit", -1.0))
        != 180.0
        or observed_keys != cohort_keys
        or proposal.get("topology_boundary") != expected_boundary
        or proposal.get("hybridstructpool") != expected_hybrid
    ):
        raise ValueError(
            f"Warehouse fixed16 inner run identity changed for {controller}"
        )


def _validate_manifest_trace_row(
    collection: Path,
    row: Mapping[str, Any],
    *,
    run_fingerprint: str,
    expected_item: Mapping[str, Any],
    expected_dataset_row: Mapping[str, Any],
    metric_iteration_budget: int | None,
) -> None:
    status = str(row.get("status"))
    if status not in {"ok", "resumed"}:
        raise ValueError("Warehouse fixed16 manifest contains a terminal status")
    key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
    expected_key = (
        str(expected_item["task_id"]),
        int(expected_item["solver_seed"]),
    )
    expected_episode = closed_loop_episode_id(
        dict(expected_dataset_row), expected_key[1], "realized_dynamic"
    )
    if (
        key != expected_key
        or str(row.get("episode_id")) != expected_episode
        or str(row.get("policy")) != "realized_dynamic"
        or str(row.get("split")) != SPLIT
        or str(row.get("map_id")) != str(expected_item["map_id"])
        or str(row.get("map_id")) != str(expected_dataset_row["map_id"])
        or str(row.get("task_variant"))
        != str(expected_dataset_row["task_variant"])
        or int(row.get("agent_count", -1)) != int(expected_item["agent_count"])
        or int(row.get("agent_count", -1))
        != int(expected_dataset_row["agent_count"])
    ):
        raise ValueError("Warehouse fixed16 manifest metadata changed")
    trace_value = row.get("trace_file")
    if not isinstance(trace_value, str) or not trace_value:
        raise ValueError("Warehouse fixed16 manifest trace path is missing")
    trace = (collection / trace_value).resolve()
    try:
        trace.relative_to(collection.resolve())
    except ValueError as error:
        raise ValueError("Warehouse fixed16 manifest trace escapes collection") from error
    if not trace.is_file() or sha256_file(trace) != str(row.get("trace_sha256") or ""):
        raise ValueError("Warehouse fixed16 manifest trace hash changed")
    validated = validate_closed_loop_trace(
        trace,
        run_fingerprint,
        expected_episode_id=expected_episode,
        expected_policy="realized_dynamic",
        expected_solver_seed=int(row.get("solver_seed", -1)),
        metric_iteration_budget=metric_iteration_budget,
        collection_root=collection,
    )
    if validated.get("summary") != row.get("summary"):
        raise ValueError("Warehouse fixed16 trace/manifest summary mismatch")
    if int(validated.get("event_count", -1)) != int(
        row.get("trace_event_count", -2)
    ):
        raise ValueError("Warehouse fixed16 trace event count mismatch")


def _audit_inner_collections(
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    items: list[dict[str, Any]],
    *,
    require_all: bool,
) -> dict[str, str]:
    run_fingerprints: dict[str, str] = {}
    dataset_rows = {
        str(row["task_id"]): dict(row)
        for row in _read_jsonl(output / "dataset" / SPLIT / "manifest.jsonl")
    }
    for controller in CONTROLLERS:
        collection = _controller_dir(output, controller)
        run_path = collection / "run_config.json"
        manifest_path = _manifest_path(output, controller)
        if not run_path.is_file():
            if manifest_path.is_file() or require_all:
                raise ValueError(
                    f"Warehouse fixed16 {controller} manifest/run identity is incomplete"
                )
            continue
        cohort_keys = _controller_cohort_keys(items, controller)
        expected_items = {
            (str(row["task_id"]), int(row["solver_seed"])): dict(row)
            for row in items
            if row["controller"] == controller
        }
        expected = _expected_inner_run(root, config, output, controller, cohort_keys)
        payload = _read_json(run_path)
        _validate_inner_run_config_payload(
            payload,
            expected_run_fingerprint=str(expected["run_fingerprint"]),
            controller=controller,
            cohort_keys=cohort_keys,
        )
        run_fingerprint = str(payload["run_fingerprint"])
        run_fingerprints[controller] = run_fingerprint
        configuration = dict(payload["configuration"])
        raw_metric_budget = configuration.get("metric_iteration_budget")
        metric_budget = (
            int(raw_metric_budget) if raw_metric_budget is not None else None
        )
        rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
        if require_all and len(rows) != len(cohort_keys):
            raise ValueError(
                f"Warehouse fixed16 {controller} manifest coverage is incomplete"
            )
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
            if key in seen:
                raise ValueError("Warehouse fixed16 manifest contains duplicate keys")
            seen.add(key)
            expected_item = expected_items.get(key)
            expected_dataset_row = dataset_rows.get(key[0])
            if expected_item is None or expected_dataset_row is None:
                raise ValueError(
                    "Warehouse fixed16 manifest key is absent from schedule/dataset"
                )
            _validate_manifest_trace_row(
                collection,
                row,
                run_fingerprint=run_fingerprint,
                expected_item=expected_item,
                expected_dataset_row=expected_dataset_row,
                metric_iteration_budget=metric_budget,
            )
        if require_all and seen != cohort_keys:
            raise ValueError(
                f"Warehouse fixed16 {controller} trace coverage differs from cohort"
            )
    return run_fingerprints


def select_qualified_cohort(
    config: Mapping[str, Any], qualification_report: Mapping[str, Any]
) -> dict[str, Any]:
    expected = qualification_schedule(config)
    raw_rows = list(
        dict(qualification_report.get("natural_distribution") or {}).get(
            "tasks", ()
        )
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
        errors.append("qualification reset coverage differs from the 256-key registration")
    if list(qualification_report.get("errors") or ()):
        errors.append("qualification contains reset errors")
    if int(qualification_report.get("valid_count", -1)) != len(expected):
        errors.append("qualification valid reset count differs from 256")
    if (
        int(qualification_report.get("incomplete_reset_count", -1)) != 0
        or int(qualification_report.get("inconsistent_initial_state_count", -1))
        != 0
        or dict(qualification_report.get("gates") or {}).get("all_resets_valid")
        is not True
    ):
        errors.append("qualification reset completeness/consistency gate failed")

    thresholds = {"opposite_exchange": 16, "uniform_random": 1}
    selected_loads: dict[str, int] = {}
    details: list[dict[str, Any]] = []
    for map_id in MAP_IDS:
        load_details: list[dict[str, Any]] = []
        for load in AGENT_COUNT_LADDER:
            rows = []
            missing = []
            for task_seed in TASK_SEEDS:
                for variant in TASK_VARIANTS:
                    task_id = _task_id(map_id, variant, task_seed, load)
                    for solver_seed in SOLVER_SEEDS:
                        row = index.get((task_id, solver_seed))
                        if row is None:
                            missing.append((task_id, solver_seed))
                        else:
                            rows.append((variant, row))
            passed = bool(
                not missing
                and len(rows) == 16
                and all(
                    int(row.get("initial_conflicts", -1)) >= thresholds[variant]
                    and row.get("initial_feasible") is False
                    and row.get("initial_complete") is True
                    and row.get("initial_state_consistent") is True
                    and isinstance(row.get("state_fingerprint"), str)
                    and len(str(row["state_fingerprint"])) == 64
                    for variant, row in rows
                )
            )
            load_details.append(
                {
                    "agent_count": load,
                    "passed": passed,
                    "valid_reset_count": len(rows),
                    "missing_reset_count": len(missing),
                    "minimum_initial_conflicts": {
                        variant: (
                            min(
                                int(row["initial_conflicts"])
                                for current, row in rows
                                if current == variant
                            )
                            if any(current == variant for current, _row in rows)
                            else None
                        )
                        for variant in TASK_VARIANTS
                    },
                }
            )
        eligible = [row["agent_count"] for row in load_details if row["passed"]]
        selected = min(eligible) if eligible else None
        if selected is None:
            errors.append(f"{map_id}: no common qualified load")
        else:
            selected_loads[map_id] = int(selected)
        details.append(
            {
                "map_id": map_id,
                "selected_agent_count": selected,
                "loads": load_details,
            }
        )
    return {
        "schema": QUALIFICATION_SELECTION_SCHEMA,
        "selection_rule": "per_map_lowest_common_load_for_both_variants",
        "passed": not errors and len(selected_loads) == 4,
        "expected_reset_count": 256,
        "observed_reset_count": len(index),
        "selected_loads": selected_loads,
        "details": details,
        "errors": errors,
        "claim_boundary": config["cohort"]["claim_boundary"],
        "final_namespace_touched": False,
    }


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
    output = Path(output).resolve()
    prepare_development_dataset(path, output)
    dataset = output / "dataset"
    qualification = output / "qualification"
    keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in qualification_schedule(config)
    }
    runtime_path = _registered_warehouse_input(
        root, dict(config["inputs"])["runtime_config"], "runtime config"
    )
    run_closed_loop_collection(
        dataset,
        runtime_path,
        qualification,
        phase="qualify",
        workers=16,
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_process_timeout_seconds=240.0,
        **_v2_kwargs(root, config),
    )
    report_path = qualification / "qualification_report.json"
    selection = select_qualified_cohort(config, _read_json(report_path))
    selection.update(
        {
            "config_sha256": sha256_file(path),
            "qualification_report_sha256": sha256_file(report_path),
        }
    )
    _write_json(output / SELECTION_FILENAME, selection)
    return selection


def _load_selection(
    config_path: Path, output: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    selection_path = output / SELECTION_FILENAME
    if not selection_path.is_file():
        raise ValueError("Warehouse fixed16 selected cohort is missing; qualify first")
    selection = _read_json(selection_path)
    qualification_path = output / "qualification" / "qualification_report.json"
    if not qualification_path.is_file():
        raise ValueError("Warehouse fixed16 qualification evidence changed")
    recomputed = select_qualified_cohort(config, _read_json(qualification_path))
    recomputed.update(
        {
            "config_sha256": sha256_file(config_path),
            "qualification_report_sha256": sha256_file(qualification_path),
        }
    )
    if selection != recomputed or recomputed.get("passed") is not True:
        raise ValueError(
            "Warehouse fixed16 selected cohort differs from recomputed lowest "
            "common qualified loads"
        )
    return selection


def formal_schedule(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected = {
        str(map_id): int(load)
        for map_id, load in dict(selection["selected_loads"]).items()
    }
    rows: list[dict[str, Any]] = []
    key_index = 0
    for map_id in MAP_IDS:
        load = selected[map_id]
        for variant in TASK_VARIANTS:
            for half in ("A", "B"):
                for task_seed, solver_seed in HALVES[half]:
                    task_id = _task_id(map_id, variant, task_seed, load)
                    offset = key_index % len(CONTROLLERS)
                    for position in range(len(CONTROLLERS)):
                        controller = CONTROLLERS[(offset + position) % len(CONTROLLERS)]
                        rows.append(
                            {
                                "map_id": map_id,
                                "variant": variant,
                                "agent_count": load,
                                "task_seed": task_seed,
                                "task_id": task_id,
                                "solver_seed": solver_seed,
                                "half": half,
                                "controller": controller,
                                "within_key_position": position,
                            }
                        )
                    key_index += 1
    return rows


def _controller_dir(output: Path, controller: str) -> Path:
    try:
        code = CONTROLLER_CODES[controller]
    except KeyError as error:
        raise ValueError(f"unknown Warehouse fixed16 controller: {controller}") from error
    return output / "formal" / "controllers" / code


def _manifest_path(output: Path, controller: str) -> Path:
    return _controller_dir(output, controller) / "realized_dynamic_manifest.jsonl"


def _manifest(
    output: Path, controller: str, task_id: str, solver_seed: int
) -> dict[str, Any] | None:
    path = _manifest_path(output, controller)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else ())
        if str(row.get("task_id")) == task_id
        and int(row.get("solver_seed", -1)) == solver_seed
    ]
    if len(matches) > 1:
        raise ValueError("Warehouse fixed16 manifest contains a duplicate key")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    path, root, config = load_config(job["config_path"])
    output = Path(job["output_root"]).resolve()
    selection = _load_selection(path, output, config)
    items = formal_schedule(config, selection)
    item = dict(job["item"])
    controller = str(item["controller"])
    collection = _controller_dir(output, controller)
    cohort_keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in items
        if row["controller"] == controller
    }
    kwargs = controller_kwargs(root, config, controller)
    runtime_path = _registered_warehouse_input(
        root, dict(config["inputs"])["runtime_config"], "runtime config"
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
    row = _manifest(
        output, controller, str(item["task_id"]), int(item["solver_seed"])
    )
    if row is None:
        raise RuntimeError("Warehouse fixed16 episode completed without a manifest")
    status = str(row.get("status"))
    return {**item, "status": status if status in {"error", "timeout"} else "ok"}


def _failed_episode_job(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    """Preserve a worker failure without assuming another runner's job schema."""

    item = dict(job.get("item") or {})
    return {
        **item,
        "job_id": str(job.get("job_id") or ""),
        "status": status,
        "manifest_status": status,
        "error": message,
        "collection_path": str(
            _controller_dir(
                Path(str(job["output_root"])).resolve(),
                str(item.get("controller") or "unknown"),
            )
        ),
    }


def _terminal_manifest_failures(output: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        manifest = _manifest_path(output, controller)
        if not manifest.is_file():
            continue
        for row in _read_jsonl(manifest):
            if str(row.get("status")) in {"error", "timeout"}:
                failures.append(
                    {
                        "controller": controller,
                        "task_id": str(row.get("task_id") or ""),
                        "solver_seed": int(row.get("solver_seed", -1)),
                        "status": str(row.get("status")),
                        "error": row.get("error"),
                    }
                )
    return failures


def _completed_schedule_prefix_length(
    output: Path, items: list[dict[str, Any]]
) -> int:
    """Require serial resume evidence to be an exact schedule prefix."""

    completed = 0
    missing_seen = False
    for item in items:
        row = _manifest(
            output,
            str(item["controller"]),
            str(item["task_id"]),
            int(item["solver_seed"]),
        )
        if row is None:
            missing_seen = True
        elif missing_seen:
            raise ValueError(
                "Warehouse fixed16 completed manifests are not a strict "
                "prefix of the registered serial schedule"
            )
        else:
            completed += 1
    return completed


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    rows = [
        _manifest(
            output,
            str(item["controller"]),
            str(item["task_id"]),
            int(item["solver_seed"]),
        )
        for item in items
    ]
    present = [row for row in rows if row is not None]
    return {
        **dict(base),
        "completed_schedule_entries": len(present),
        "completed_jobs": len(present),
        "total_jobs": len(items),
        "completed_by_controller": {
            controller: sum(
                row is not None
                for row, item in zip(rows, items)
                if item["controller"] == controller
            )
            for controller in CONTROLLERS
        },
        "error_jobs": sum(row.get("status") == "error" for row in present),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in present),
        "complete": bool(complete and len(present) == len(items)),
        "final_namespace_touched": False,
    }


def _producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_warehouse_fixed16_development.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/structshell_single_family.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
    )


def collect(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
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
        label="Warehouse fixed16 development screen",
    )
    if prepared.status.get("terminal_failure") is not None:
        raise RuntimeError(
            "Warehouse fixed16 output has a recorded terminal failure; "
            "resume is forbidden"
        )
    existing_failures = _terminal_manifest_failures(output)
    if existing_failures:
        raise RuntimeError(
            "Warehouse fixed16 output contains an error/timeout manifest; "
            f"resume is forbidden: {existing_failures[0]}"
        )
    _audit_inner_collections(
        root,
        config,
        output,
        items,
        require_all=prepared.completed_report is not None,
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    completed_prefix = _completed_schedule_prefix_length(output, items)
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
            phase="warehouse-fixed16-development",
            output_root=formal,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=300.0,
            failure_result=_failed_episode_job,
            stop_on_failure=True,
        )
        failures = [row for row in results if row.get("status") in {"error", "timeout"}]
        if failures:
            status = _status(output, items, prepared.base_status)
            status["terminal_failure"] = failures[0]
            _write_json(formal / STATUS_FILENAME, status)
            return status
    report = analyze(path, output, producer=producer)
    status = _status(output, items, prepared.base_status, complete=True)
    status["report_sha256"] = sha256_file(formal / REPORT_FILENAME)
    _write_json(formal / STATUS_FILENAME, status)
    return report


def _key(item: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(item["map_id"]),
        str(item["task_id"]),
        int(item["solver_seed"]),
    )


def _normalized_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Let resumed atomic episodes participate without mutating source evidence."""

    result = dict(row)
    status = str(result.get("status"))
    result["source_manifest_status"] = status
    if status in {"ok", "resumed"}:
        result["status"] = "ok"
    return result


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
        raise ValueError("Warehouse fixed16 formal identity artifacts are incomplete")
    status = _read_json(status_path)
    runner = _read_json(runner_path)
    schedule_rows = _read_jsonl(schedule_path)
    schedule_sha = _fingerprint(items)
    if schedule_rows != items or _fingerprint(schedule_rows) != schedule_sha:
        raise ValueError("Warehouse fixed16 formal schedule changed")
    expected_identity = {
        "schema": RESUMABLE_RUN_IDENTITY_SCHEMA,
        "status_schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "schedule_sha256": schedule_sha,
        "producer_identity": dict(producer),
        "report_schema": REPORT_SCHEMA,
    }
    if (
        runner.get("schema") != RUNNER_CONFIG_SCHEMA
        or runner.get("identity") != expected_identity
        or runner.get("identity_fingerprint") != _fingerprint(expected_identity)
        or status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != expected_identity["config_sha256"]
        or status.get("schedule_sha256") != schedule_sha
        or status.get("producer_identity") != dict(producer)
        or status.get("run_fingerprint") != runner.get("identity_fingerprint")
        or int(status.get("total_schedule_entries", -1)) != len(items)
    ):
        raise ValueError("Warehouse fixed16 formal producer/run identity changed")
    return status


def _controller_report_metrics(rows: list[dict[str, Any]], controller: str) -> dict[str, Any]:
    summaries = [
        dict(row["summary"])
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    totals = [dict(row.get("controller_totals") or {}) for row in summaries]
    decisions = sum(int(row.get("model_decision_count", 0)) for row in totals)
    if controller == "boundary16_static_cache":
        # The registered near-tie metric counts source candidates before
        # exact-set de-duplication.  Use the generated Boundary count so it is
        # symmetric with HybridStructPool's structural source count below.
        added = sum(
            int(row.get("topology_boundary_generated_count", 0))
            for row in totals
        )
    elif controller in SINGLE_FAMILY_PROFILES:
        added = sum(int(row.get("hybridstructpool_structural_candidate_count", 0)) for row in totals)
    else:
        added = 0
    return {
        "mean_repair_iterations": (
            statistics.fmean(float(row["repair_iterations"]) for row in summaries)
            if summaries
            else None
        ),
        "mean_pp_replan_seconds": (
            statistics.fmean(float(row.get("pp_replan_seconds", 0.0)) for row in totals)
            if totals
            else None
        ),
        "mean_neighborhood_selection_seconds": (
            statistics.fmean(
                float(row.get("neighborhood_selection_seconds", 0.0))
                for row in totals
            )
            if totals
            else None
        ),
        "total_added_candidate_count": added,
        "controller_decision_count": decisions,
        "mean_added_candidate_count_per_decision": (
            added / decisions if decisions else 0.0
        ),
        "repair_iteration_tail": _repair_tail(rows),
    }


def performance_gate(
    *,
    integrity_passed: bool,
    v2_summary: Mapping[str, Any],
    challenger_summary: Mapping[str, Any],
    overall: Mapping[str, Any],
    half_a: Mapping[str, Any],
    half_b: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "integrity_and_zero_execution_errors": bool(integrity_passed),
        "success_count_not_lower_than_v2": int(challenger_summary.get("success_count", -1))
        >= int(v2_summary.get("success_count", 0)),
        "mean_restricted_ttf_strictly_lower_than_v2": bool(overall.get("valid"))
        and float(overall.get("mean_restricted_ttf_delta_seconds", 0.0)) < 0.0,
        "half_a_mean_restricted_ttf_strictly_lower_than_v2": bool(
            half_a.get("valid")
        )
        and float(half_a.get("mean_restricted_ttf_delta_seconds", 0.0)) < 0.0,
        "half_b_mean_restricted_ttf_strictly_lower_than_v2": bool(
            half_b.get("valid")
        )
        and float(half_b.get("mean_restricted_ttf_delta_seconds", 0.0)) < 0.0,
    }


def choose_winner(
    gates: Mapping[str, Mapping[str, bool]],
    comparisons: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    relative_margin: float = 0.01,
) -> dict[str, Any]:
    eligible = [name for name in CHALLENGERS if all(dict(gates.get(name) or {}).values())]
    if not eligible:
        return {
            "selected_controller": None,
            "eligible_controllers": [],
            "near_tie_controllers": [],
            "decision": "no_fixed16_challenger_passed_stop",
        }
    means = {
        name: float(comparisons[name]["challenger_mean_restricted_ttf"])
        for name in eligible
    }
    best = min(means.values())
    if best > 0.0:
        near = [
            name
            for name in eligible
            if (means[name] - best) / best < relative_margin
        ]
    else:
        near = [name for name in eligible if means[name] == 0.0]
    selected = min(
        near,
        key=lambda name: (
            float(metrics[name]["mean_added_candidate_count_per_decision"]),
            float(metrics[name]["mean_neighborhood_selection_seconds"]),
            name,
        ),
    )
    return {
        "selected_controller": selected,
        "eligible_controllers": eligible,
        "near_tie_controllers": near,
        "minimum_mean_restricted_ttf": best,
        "near_tie_relative_margin": relative_margin,
        "decision": "development_winner_only_requires_fresh_final_confirmation",
    }


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    formal = output / "formal"
    expected_producer = _producer(root)
    selection = _load_selection(path, output, config)
    items = formal_schedule(config, selection)
    _audit_formal_identity(path, formal, items, expected_producer)
    inner_run_fingerprints = _audit_inner_collections(
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
            raise ValueError("Warehouse fixed16 completed report producer changed")
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
        manifest = _manifest_path(output, controller)
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        if not manifest.is_file():
            errors.append(f"{controller}: missing manifest")
        else:
            manifest_hashes[controller] = sha256_file(manifest)
            task_to_map = {
                str(item["task_id"]): str(item["map_id"])
                for item in items
                if item["controller"] == controller
            }
            for row in _read_jsonl(manifest):
                task_id = str(row.get("task_id"))
                key = (
                    task_to_map.get(task_id, ""),
                    task_id,
                    int(row.get("solver_seed", -1)),
                )
                if key in rows:
                    errors.append(f"{controller}: duplicate {key}")
                rows[key] = _normalized_manifest_row(row)
        if set(rows) != expected:
            errors.append(f"{controller}: incomplete or extra paired coverage")
        indexed[controller] = rows

    qualification_report = _read_json(
        output / "qualification" / "qualification_report.json"
    )
    qualification_index = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in dict(
            qualification_report.get("natural_distribution") or {}
        ).get("tasks", ())
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
        conflict_mismatches += len({row.get("initial_conflicts") for row in summaries}) != 1
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
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        bad_capped += sum(row.get("capped_wall_time_to_feasible") is None for row in summaries)
        bad_stop_reason += sum(str(row.get("stop_reason")) not in accepted_stops for row in summaries)

    summaries = {
        name: _bounded_summary(list(indexed[name].values())) for name in CONTROLLERS
    }
    report_metrics = {
        name: _controller_report_metrics(list(indexed[name].values()), name)
        for name in CONTROLLERS
    }
    integrity = {
        "qualification_selection_passed": selection.get("passed") is True,
        "complete_paired_coverage": not any("coverage" in error or "missing" in error for error in errors),
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
        "strict_rotating_serial_schedule": len(items) == 192
        and all(
            {row["controller"] for row in items[index : index + 6]}
            == set(CONTROLLERS)
            and [row["within_key_position"] for row in items[index : index + 6]]
            == list(range(6))
            for index in range(0, len(items), 6)
        ),
        "final_namespace_untouched_by_runner": True,
    }
    integrity_passed = not errors and all(integrity.values())

    overall_comparisons: dict[str, dict[str, Any]] = {}
    half_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    gates: dict[str, dict[str, bool]] = {}
    keys = sorted(expected)
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
        gates[challenger] = performance_gate(
            integrity_passed=integrity_passed,
            v2_summary=summaries["v2_only"],
            challenger_summary=summaries[challenger],
            overall=overall,
            half_a=halves["A"],
            half_b=halves["B"],
        )

    variant_by_key = {_key(item): str(item["variant"]) for item in items}

    def diagnostic_slice(slice_keys: list[tuple[str, str, int]]) -> dict[str, Any]:
        return {
            "paired_key_count": len(slice_keys),
            "controller_summaries": {
                controller: _bounded_summary(
                    [
                        indexed[controller][key]
                        for key in slice_keys
                        if key in indexed[controller]
                    ]
                )
                for controller in CONTROLLERS
            },
            "comparisons_vs_v2": {
                challenger: _bounded_paired_comparison(
                    indexed["v2_only"], indexed[challenger], slice_keys
                )
                for challenger in CHALLENGERS
            },
        }

    diagnostic_breakdowns = {
        "per_variant": {
            variant: diagnostic_slice(
                sorted(key for key in expected if variant_by_key[key] == variant)
            )
            for variant in TASK_VARIANTS
        },
        "per_map": {
            map_id: diagnostic_slice(
                sorted(key for key in expected if key[0] == map_id)
            )
            for map_id in MAP_IDS
        },
        "per_map_variant": {
            map_id: {
                variant: diagnostic_slice(
                    sorted(
                        key
                        for key in expected
                        if key[0] == map_id and variant_by_key[key] == variant
                    )
                )
                for variant in TASK_VARIANTS
            }
            for map_id in MAP_IDS
        },
        "gate_status": "diagnostic_only_not_used_for_promotion",
    }
    winner = choose_winner(
        gates,
        overall_comparisons,
        report_metrics,
        relative_margin=float(config["performance_gates"]["near_tie_relative_margin"]),
    )
    if producer is None:
        producer = expected_producer
    elif producer != expected_producer:
        raise ValueError("Warehouse fixed16 analysis producer identity changed")
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_screen_not_a_speed_or_runtime_claim",
        "map_count": 4,
        "paired_key_count": 32,
        "episode_count": sum(len(rows) for rows in indexed.values()),
        "controllers": list(CONTROLLERS),
        "official_adaptive_included": False,
        "official_adaptive_policy": (
            "add_only_after_a_development_winner_is_frozen_for_fresh_confirmation"
        ),
        "selected_loads": dict(selection["selected_loads"]),
        "variant_pairing_claim": config["cohort"]["claim_boundary"],
        "controller_summaries": summaries,
        "report_only_metrics": report_metrics,
        "overall_comparisons_vs_v2": overall_comparisons,
        "half_comparisons_vs_v2": half_comparisons,
        "diagnostic_breakdowns": diagnostic_breakdowns,
        "integrity_gates": integrity,
        "challenger_gates": gates,
        "integrity_passed": integrity_passed,
        "development_gate_passed": winner["selected_controller"] is not None,
        **winner,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "final_namespace_touched": False,
        "next_step": (
            "freeze_development_winner_then_preregister_size_study_or_fresh_official_confirmation"
            if winner["selected_controller"] is not None
            else "stop_fixed16_branch_keep_v2_default"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
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
    "AGENT_COUNT_LADDER",
    "BOUNDARY16_STATIC_CACHE",
    "CHALLENGERS",
    "CONTROLLERS",
    "FINAL_NAMESPACE",
    "HALVES",
    "MAP_IDS",
    "SOLVER_SEEDS",
    "TASK_SEEDS",
    "TASK_VARIANTS",
    "analyze",
    "choose_winner",
    "collect",
    "controller_kwargs",
    "development_task_specs",
    "formal_schedule",
    "load_config",
    "performance_gate",
    "plan",
    "prepare_development_dataset",
    "qualification_schedule",
    "qualify",
    "run",
    "select_qualified_cohort",
]
