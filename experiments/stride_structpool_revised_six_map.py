from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


CONFIG_SCHEMA = "lns2.stride.structpool_revised_six_map_qualification_design.v1"
REPORT_SCHEMA = "lns2.stride.structpool_revised_six_map_qualification_report.v1"
SPLIT = "balanced_wall_clock"
FORBIDDEN_FIELDS = {
    "conflicts_after",
    "repair_iterations",
    "repair_runtime",
    "pp_replan_seconds",
    "selected_candidate",
    "controller_action",
    "controller_ttf",
    "time_to_feasible_seconds",
}


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered revised six-map input changed: {path}")
    return path


def validate_revised_six_map_design(
    config: dict[str, Any], *, project_root: Path
) -> dict[str, Path]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "qualification_conditioned_formal_reset_confirmation_preregistered_before_combined_resets"
        or config.get("experiment_id") != "stride-structpool-revised-six-map-reset-v1"
        or config.get("pre_registration_parent_commit") != "1b8a1be"
        or config.get("dataset") != "build/stride-structpool-revised-six-map-dataset-v1"
        or config.get("runtime") != "configs/stride_structpool_revised_six_map_runtime.json"
        or config.get("split") != SPLIT
        or list(config.get("solver_seeds") or ()) != [1, 2, 3]
        or int(config.get("expected_map_count", -1)) != 6
        or int(config.get("expected_task_count", -1)) != 12
        or int(config.get("expected_reset_count", -1)) != 36
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("revised six-map qualification identity changed")
    if dict(config.get("qualification_gates") or {}) != {
        "minimum_nonzero_states": 24,
        "minimum_nonzero_states_per_layout": 1,
        "minimum_active_maps": 6,
        "minimum_nonzero_states_per_solver_seed": 6,
        "require_exact_source_state_reproduction": True,
        "require_zero_errors_and_timeouts": True,
    }:
        raise ValueError("revised six-map qualification gates changed")
    boundary = dict(config.get("outcome_boundary") or {})
    if (
        set(boundary.get("allowed_observed_fields") or ())
        != {
            "status",
            "error",
            "initial_complete",
            "initial_conflicts",
            "initial_complexity",
            "state_fingerprint",
            "initial_feasible",
        }
        or boundary.get("repair_outcomes_read") is not False
        or boundary.get("controller_outcomes_read") is not False
        or boundary.get("ttf_outcomes_read") is not False
    ):
        raise ValueError("revised six-map outcome boundary changed")
    if (
        config.get("next_step_on_pass")
        != "preregister_revised_six_map_paired_raw_ttf"
        or config.get("next_step_on_failure")
        != "stop_before_controller_timing_and_report_exact_reset_mismatch"
    ):
        raise ValueError("revised six-map decision sequence changed")
    expected_inputs = {
        "materialization_config",
        "dataset_summary",
        "dataset_manifest",
        "artifact_registry",
        "runtime",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("revised six-map input registry changed")
    paths = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    source_specs = list(config.get("source_qualification_manifests") or ())
    if [str(row.get("id")) for row in source_specs] != [
        "retained_original",
        "maze_replacement",
        "warehouse_replacement",
        "game_replacement",
    ]:
        raise ValueError("revised six-map source qualification order changed")
    for source in source_specs:
        paths[f"source:{source['id']}"] = _registered(project_root, dict(source))
    runtime = _read_json(paths["runtime"])
    if (
        runtime.get("experiment_runtime_id")
        != "stride-structpool-revised-six-map-reset-v1"
        or runtime.get("formal") is not True
        or list(runtime.get("solver_seeds") or ()) != [1, 2, 3]
        or dict(runtime.get("qualification") or {})
        != {
            "mode": "revised_six_map",
            "minimum_nonzero_states": 24,
            "minimum_nonzero_states_per_layout": 1,
            "minimum_active_maps": 6,
            "minimum_nonzero_states_per_solver_seed": 6,
        }
    ):
        raise ValueError("revised six-map runtime changed")
    return paths


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "mean": sum(ordered) / len(ordered),
        "max": max(ordered),
    }


def analyze_revised_six_map_qualification(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    paths = validate_revised_six_map_design(config, project_root=project_root)
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    dataset_manifest_path = dataset / SPLIT / "manifest.jsonl"
    qualification_manifest_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    dataset_rows = _read_jsonl(dataset_manifest_path)
    rows = _read_jsonl(qualification_manifest_path)
    qualification_report = _read_json(qualification_report_path)

    expected_tasks = {str(row["task_id"]) for row in dataset_rows}
    expected_keys = {(task_id, seed) for task_id in expected_tasks for seed in (1, 2, 3)}
    observed = {(str(row["task_id"]), int(row["solver_seed"])): row for row in rows}
    source_index: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_source_keys: list[list[Any]] = []
    for source in config["source_qualification_manifests"]:
        for row in _read_jsonl(paths[f"source:{source['id']}"]):
            key = (str(row["task_id"]), int(row["solver_seed"]))
            if key not in expected_keys:
                continue
            if key in source_index:
                duplicate_source_keys.append([key[0], key[1]])
            source_index[key] = row

    reproduction_fields = (
        "status",
        "error",
        "initial_complete",
        "initial_conflicts",
        "initial_feasible",
        "initial_complexity",
        "state_fingerprint",
    )
    mismatches: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        current = observed.get(key)
        source = source_index.get(key)
        if current is None or source is None:
            mismatches.append(
                {"task_id": key[0], "solver_seed": key[1], "field": "missing_row"}
            )
            continue
        for field in reproduction_fields:
            if current.get(field) != source.get(field):
                mismatches.append(
                    {"task_id": key[0], "solver_seed": key[1], "field": field}
                )

    forbidden = sorted(
        {field for row in rows for field in FORBIDDEN_FIELDS if field in row}
    )
    errors = [
        [str(row.get("task_id")), int(row.get("solver_seed", -1)), row.get("error")]
        for row in rows
        if row.get("status") != "ok" or row.get("error") is not None
    ]
    incomplete = [
        [str(row["task_id"]), int(row["solver_seed"])]
        for row in rows
        if not bool(row.get("initial_complete"))
    ]
    by_map: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(int(row["initial_conflicts"]))
    map_summaries = [
        {"map_id": map_id, **_summary(values)}
        for map_id, values in sorted(by_map.items())
    ]
    gates = {
        "registered_dimensions": len(dataset_rows) == 12
        and len(expected_tasks) == 12
        and len(rows) == 36
        and set(observed) == expected_keys,
        "six_maps_two_tasks_each": len({str(row["map_id"]) for row in dataset_rows}) == 6
        and set(collections.Counter(str(row["map_id"]) for row in dataset_rows).values())
        == {2},
        "source_state_product_complete": set(source_index) == expected_keys
        and not duplicate_source_keys,
        "exact_source_state_reproduction": not mismatches,
        "zero_errors_and_incomplete_resets": not errors and not incomplete,
        "forbidden_outcomes_absent": not forbidden,
        "collector_qualification_passed": qualification_report.get("passed") is True,
        "all_six_maps_active": int(qualification_report.get("active_map_count", -1)) == 6,
        "registered_nonzero_thresholds_passed": all(
            bool(value) for name, value in qualification_report.get("gates", {}).items()
            if name.startswith("minimum_nonzero") or name == "minimum_active_maps"
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "qualification_conditioned_formal_reset_confirmation",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "qualification_manifest_sha256": sha256_file(qualification_manifest_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "map_summaries": map_summaries,
        "nonzero_state_count": int(qualification_report.get("nonzero_state_count", -1)),
        "active_map_count": int(qualification_report.get("active_map_count", -1)),
        "source_reproduction_mismatches": mismatches,
        "duplicate_source_keys": duplicate_source_keys,
        "errors": errors,
        "incomplete_resets": incomplete,
        "forbidden_fields_found": forbidden,
        "repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "ttf_outcomes_read": False,
        "formal_speed_claim": False,
        "gates": gates,
        "passed": passed,
        "next_step": config[
            "next_step_on_pass" if passed else "next_step_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "revised_six_map_qualification_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_revised_six_map_qualification",
    "validate_revised_six_map_design",
]
