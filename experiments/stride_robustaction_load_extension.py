from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_expansion import (
    LOAD_EXTENSION_SCHEMA,
    validate_robustaction_expansion_design,
)
from experiments.stride_robustaction_qualification import (
    FORBIDDEN_FIELDS,
    registered_runtime_matches,
    select_qualified_tasks,
)


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_load_extension_qualification_design.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_load_extension_qualification_report.v1"
)
SPLIT = "balanced_wall_clock"
UNDERLOADED_MAPS = {
    "brc200d", "brc201d", "den900d", "hrt001d", "lak106d", "lak308d",
    "lak526d", "lgt600d", "lgt604d", "orz601d", "ost001d", "oth000d",
}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"load-extension qualification input changed: {spec['path']}")
    return path


def _task_variant(task_id: str) -> str:
    if "__derived_opposite_exchange__" in task_id:
        return "opposite_exchange"
    if "__derived_uniform_random__" in task_id:
        return "uniform_random"
    raise ValueError(f"unregistered load-extension task variant: {task_id}")


def validate_load_extension_qualification_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_structpool_load_extension_initial_pp"
        or config.get("data_line_id") != "stride-robustaction-structpool-data-v2"
        or config.get("pre_registration_git_commit")
        != "bcefd16818a1d54695bb3e55a57b92bbb2ffe59d"
        or int(config.get("expected_extension_map_count", -1)) != 12
        or int(config.get("expected_extension_task_count", -1)) != 48
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_extension_qualification_job_count", -1)) != 96
        or int(config.get("expected_final_map_count", -1)) != 20
        or int(config.get("tasks_selected_per_map", -1)) != 2
        or int(config.get("expected_final_selected_task_count", -1)) != 40
        or list(config.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0)) != 0.5
        or float(config.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or not bool(config.get("prefer_distinct_task_variants"))
        or not bool(config.get("extension_qualification_must_pass"))
        or not bool(config.get("all_extension_resets_must_be_complete"))
        or not bool(config.get("all_extension_maps_must_have_nonzero_state"))
        or not bool(config.get("final_selection_must_cover_all_twenty_maps"))
        or bool(config.get("candidate_outcomes_read"))
        or bool(config.get("controller_outcomes_read"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("StructPool load-extension qualification identity changed")
    if set(map(str, config.get("forbidden_selection_inputs") or ())) != {
        "candidate_repair_outcome", "candidate_conflicts_after",
        "candidate_runtime", "selected_candidate", "controller_action",
        "controller_ttf", "future_trajectory", "repair_runtime",
    }:
        raise ValueError("StructPool load-extension qualification outcome boundary changed")
    if (
        config.get("next_decision_on_failure")
        != "register_map_replacement_revision_without_outcome_filtering"
    ):
        raise ValueError("StructPool load-extension qualification failure action changed")

    inputs = dict(config.get("inputs") or {})
    expected_inputs = {
        "extension_data_design", "runtime", "extension_dataset_manifest",
        "extension_dataset_summary", "v1_dataset_manifest", "v1_selection_report",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("StructPool load-extension qualification input registry changed")
    paths = {name: _registered(project_root, dict(spec)) for name, spec in inputs.items()}
    design = _read_json(paths["extension_data_design"])
    validate_robustaction_expansion_design(design)
    if design.get("schema") != LOAD_EXTENSION_SCHEMA:
        raise ValueError("qualification does not reference the load extension")
    runtime = _read_json(paths["runtime"])
    environment = dict(runtime.get("environment") or {})
    qualification = dict(runtime.get("qualification") or {})
    dataset_design = dict(runtime.get("dataset_design") or {})
    if (
        bool(runtime.get("formal"))
        or list(runtime.get("solver_seeds") or ()) != [1, 2]
        or float(environment.get("time_limit", -1.0)) != 600.0
        or int(environment.get("max_repair_iterations", -1)) != 0
        or int(runtime.get("max_decisions", -1)) != 0
        or float(runtime.get("episode_process_timeout_seconds", -1.0)) != 660.0
        or int(dataset_design.get("map_count", -1)) != 12
        or int(dataset_design.get("instance_count", -1)) != 48
        or int(qualification.get("minimum_active_maps", -1)) != 12
        or int(qualification.get("minimum_nonzero_states", -1)) != 24
        or not bool(qualification.get("enforce_registered_thresholds"))
    ):
        raise ValueError("StructPool load-extension qualification runtime changed")


def _summaries(
    manifest_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    solver_seeds: list[int],
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    source_index = {str(row["task_id"]): row for row in manifest_rows}
    expected = {
        (task_id, seed) for task_id in source_index for seed in map(int, solver_seeds)
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    forbidden: set[str] = set()
    for row in result_rows:
        forbidden.update(FORBIDDEN_FIELDS & set(row))
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source = source_index.get(key[0])
        if source is None or key not in expected:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
            continue
        if (
            str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
            or str(row.get("map_id")) != str(source["map_id"])
            or int(row.get("agent_count", -1)) != int(source["agent_count"])
        ):
            errors.append(f"invalid:{key[0]}:{key[1]}")
    if set(indexed) != expected:
        errors.append("incomplete_extension_product")

    summaries = []
    for task_id, source in sorted(source_index.items()):
        values = [indexed.get((task_id, seed)) for seed in map(int, solver_seeds)]
        if any(value is None for value in values):
            continue
        conflicts = [int(value["initial_conflicts"]) for value in values if value is not None]
        summaries.append(
            {
                "map_id": str(source["map_id"]),
                "task_id": task_id,
                "task_variant": _task_variant(task_id),
                "layout_mode": str(source["layout_mode"]),
                "agent_count": int(source["agent_count"]),
                "solver_seed_count": len(conflicts),
                "nonzero_solver_seed_count": sum(value > 0 for value in conflicts),
                "nonzero_solver_seed_fraction": sum(value > 0 for value in conflicts) / len(conflicts),
                "mean_initial_conflicts": statistics.fmean(conflicts),
                "minimum_initial_conflicts": min(conflicts),
                "maximum_initial_conflicts": max(conflicts),
            }
        )
    return summaries, errors, forbidden


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"load-extension selected dataset collision: {target}")
        return
    shutil.copy2(source, target)


def analyze_load_extension_qualification(
    *, config_path: str | Path, extension_dataset: str | Path,
    extension_qualification: str | Path, v1_dataset: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_load_extension_qualification_design(config, project_root=project_root)
    extension_dataset = Path(extension_dataset).resolve()
    extension_qualification = Path(extension_qualification).resolve()
    v1_dataset = Path(v1_dataset).resolve()
    output = Path(output).resolve()

    extension_manifest_path = extension_dataset / SPLIT / "manifest.jsonl"
    result_path = extension_qualification / "qualification_manifest.jsonl"
    qualification_report_path = extension_qualification / "qualification_report.json"
    run_config_path = extension_qualification / "run_config.json"
    v1_report_path = project_root / str(config["inputs"]["v1_selection_report"]["path"])
    extension_rows = _read_jsonl(extension_manifest_path)
    result_rows = _read_jsonl(result_path)
    qualification_report = _read_json(qualification_report_path)
    run_config = _read_json(run_config_path)
    registered_runtime = _read_json(
        project_root / str(config["inputs"]["runtime"]["path"])
    )
    v1_report = _read_json(v1_report_path)
    extension_summaries, row_errors, forbidden = _summaries(
        extension_rows, result_rows, list(map(int, config["solver_seeds"]))
    )
    combined_summaries = [dict(row) for row in v1_report["task_summaries"]]
    combined_summaries.extend(extension_summaries)
    selected, underloaded = select_qualified_tasks(combined_summaries, config)
    active_maps = {
        str(row["map_id"])
        for row in extension_summaries
        if int(row["maximum_initial_conflicts"]) > 0
    }
    selected_maps = {str(row["map_id"]) for row in selected}
    gates = {
        "extension_dimensions": len(extension_rows) == 48 and len(result_rows) == 96,
        "complete_valid_extension_product": not row_errors,
        "forbidden_outcomes_absent": not forbidden,
        "registered_runtime_exact": registered_runtime_matches(run_config, registered_runtime),
        "extension_qualification_passed": bool(qualification_report.get("passed")),
        "all_extension_maps_active": active_maps == UNDERLOADED_MAPS,
        "v1_failure_source_frozen": (
            v1_report.get("passed") is False
            and set(map(str, v1_report.get("underloaded_maps") or ())) == UNDERLOADED_MAPS
            and int(dict(v1_report.get("counts") or {}).get("selected_task_count", -1)) == 16
        ),
        "final_two_tasks_per_map": (
            not underloaded and len(selected) == 40 and len(selected_maps) == 20
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_blind_initial_pp_load_extension_qualification",
        "data_line_id": "stride-robustaction-structpool-data-v2",
        "config_sha256": sha256_file(config_path),
        "extension_manifest_sha256": sha256_file(extension_manifest_path),
        "extension_qualification_manifest_sha256": sha256_file(result_path),
        "extension_qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "v1_selection_report_sha256": sha256_file(v1_report_path),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "extension_map_count": len({str(row["map_id"]) for row in extension_rows}),
            "extension_task_count": len(extension_rows),
            "extension_job_count": len(result_rows),
            "extension_active_map_count": len(active_maps),
            "final_selected_task_count": len(selected),
            "final_selected_map_count": len(selected_maps),
        },
        "extension_active_maps": sorted(active_maps),
        "underloaded_maps": underloaded,
        "extension_task_summaries": extension_summaries,
        "selected_tasks": selected,
        "forbidden_fields_found": sorted(forbidden),
        "row_errors": row_errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "load_extension_qualification_report.json", report)
    if not passed:
        return report

    v1_rows = _read_jsonl(v1_dataset / SPLIT / "manifest.jsonl")
    sources = {
        **{str(row["task_id"]): (v1_dataset, row) for row in v1_rows},
        **{str(row["task_id"]): (extension_dataset, row) for row in extension_rows},
    }
    final_rows = []
    for chosen in selected:
        task_id = str(chosen["task_id"])
        if task_id not in sources:
            raise ValueError(f"selected load-extension task has no source: {task_id}")
        root, row = sources[task_id]
        for field in ("map_file", "scenario_file", "map_metadata_file", "task_file"):
            relative = Path(str(row[field]))
            _copy(root / SPLIT / relative, output / SPLIT / relative)
        final_rows.append(dict(row))
    final_rows.sort(key=lambda row: str(row["task_id"]))
    manifest_path = output / SPLIT / "manifest.jsonl"
    _write_jsonl(manifest_path, final_rows)
    summary = {
        "schema": "lns2.stride.robustaction_structpool_source_dataset.v2",
        "data_line_id": "stride-robustaction-structpool-data-v2",
        "scientific_status": "qualified_outcome_blind_source_task_dataset",
        "map_count": 20,
        "task_count": 40,
        "solver_seed_count": 2,
        "source_policy_count": 2,
        "projected_independent_episode_count": 160,
        "manifest_sha256": sha256_file(manifest_path),
        "qualification_report_sha256": sha256_file(
            output / "load_extension_qualification_report.json"
        ),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
    }
    _write_json(output / "dataset_summary.json", summary)
    return report


__all__ = [
    "CONFIG_SCHEMA", "REPORT_SCHEMA", "UNDERLOADED_MAPS",
    "analyze_load_extension_qualification",
    "validate_load_extension_qualification_design",
]
