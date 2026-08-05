from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_expansion import (
    MAP_REPLACEMENT_SCHEMA,
    validate_robustaction_expansion_design,
)
from experiments.stride_robustaction_load_extension import _copy, _summaries
from experiments.stride_robustaction_qualification import (
    registered_runtime_matches,
    select_qualified_tasks,
)


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_map_replacement_qualification_design.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_map_replacement_qualification_report.v1"
)
SPLIT = "balanced_wall_clock"
REPLACED_MAPS = {"brc201d", "lgt600d", "ost001d", "oth000d"}
REPLACEMENT_MAPS = {"lak203d", "orz201d", "ost101d", "rmtst"}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"map-replacement qualification input changed: {spec['path']}")
    return path


def validate_map_replacement_qualification_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_structpool_map_replacement_initial_pp"
        or config.get("data_line_id") != "stride-robustaction-structpool-data-v3"
        or config.get("pre_registration_git_commit")
        != "5d451383834988e428a0fefd0b03f614c5360711"
        or int(config.get("expected_replacement_map_count", -1)) != 4
        or int(config.get("expected_replacement_task_count", -1)) != 24
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_replacement_qualification_job_count", -1)) != 48
        or set(map(str, config.get("replaced_map_ids") or ())) != REPLACED_MAPS
        or set(map(str, config.get("replacement_map_ids") or ())) != REPLACEMENT_MAPS
        or int(config.get("expected_final_map_count", -1)) != 20
        or int(config.get("tasks_selected_per_map", -1)) != 2
        or int(config.get("expected_final_selected_task_count", -1)) != 40
        or list(config.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0)) != 0.5
        or float(config.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or not bool(config.get("prefer_distinct_task_variants"))
        or not bool(config.get("replacement_qualification_must_pass"))
        or not bool(config.get("all_replacement_resets_must_be_complete"))
        or not bool(config.get("all_replacement_maps_must_have_nonzero_state"))
        or not bool(config.get("final_selection_must_cover_twenty_maps"))
        or bool(config.get("candidate_outcomes_read"))
        or bool(config.get("controller_outcomes_read"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("StructPool map-replacement qualification identity changed")
    if set(map(str, config.get("forbidden_selection_inputs") or ())) != {
        "candidate_repair_outcome", "candidate_conflicts_after",
        "candidate_runtime", "selected_candidate", "controller_action",
        "controller_ttf", "future_trajectory", "repair_runtime",
    }:
        raise ValueError("StructPool map-replacement qualification outcome boundary changed")
    inputs = dict(config.get("inputs") or {})
    expected_inputs = {
        "replacement_design", "runtime", "replacement_dataset_manifest",
        "replacement_dataset_summary", "v1_selection_report", "v2_analysis_report",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("StructPool map-replacement qualification input registry changed")
    paths = {name: _registered(project_root, dict(spec)) for name, spec in inputs.items()}
    design = _read_json(paths["replacement_design"])
    validate_robustaction_expansion_design(design)
    if design.get("schema") != MAP_REPLACEMENT_SCHEMA:
        raise ValueError("qualification does not reference the map replacement")
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
        or int(dataset_design.get("map_count", -1)) != 4
        or int(dataset_design.get("instance_count", -1)) != 24
        or int(qualification.get("minimum_active_maps", -1)) != 4
        or int(qualification.get("minimum_nonzero_states", -1)) != 8
        or not bool(qualification.get("enforce_registered_thresholds"))
    ):
        raise ValueError("StructPool map-replacement qualification runtime changed")


def analyze_map_replacement_qualification(
    *, config_path: str | Path, replacement_dataset: str | Path,
    replacement_qualification: str | Path, v1_dataset: str | Path,
    v2_extension_dataset: str | Path, output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_map_replacement_qualification_design(config, project_root=project_root)
    replacement_dataset = Path(replacement_dataset).resolve()
    replacement_qualification = Path(replacement_qualification).resolve()
    v1_dataset = Path(v1_dataset).resolve()
    v2_extension_dataset = Path(v2_extension_dataset).resolve()
    output = Path(output).resolve()

    manifest_path = replacement_dataset / SPLIT / "manifest.jsonl"
    result_path = replacement_qualification / "qualification_manifest.jsonl"
    qualification_report_path = replacement_qualification / "qualification_report.json"
    run_config_path = replacement_qualification / "run_config.json"
    rows = _read_jsonl(manifest_path)
    results = _read_jsonl(result_path)
    qualification_report = _read_json(qualification_report_path)
    run_config = _read_json(run_config_path)
    registered_runtime = _read_json(
        project_root / str(config["inputs"]["runtime"]["path"])
    )
    replacement_summaries, row_errors, forbidden = _summaries(
        rows, results, list(map(int, config["solver_seeds"]))
    )
    v1_report_path = project_root / str(config["inputs"]["v1_selection_report"]["path"])
    v2_report_path = project_root / str(config["inputs"]["v2_analysis_report"]["path"])
    v1_report = _read_json(v1_report_path)
    v2_report = _read_json(v2_report_path)
    combined = [
        dict(row) for row in v1_report["task_summaries"]
        if str(row["map_id"]) not in REPLACED_MAPS
    ]
    combined.extend(
        dict(row) for row in v2_report["extension_task_summaries"]
        if str(row["map_id"]) not in REPLACED_MAPS
    )
    combined.extend(replacement_summaries)
    selected, underloaded = select_qualified_tasks(combined, config)
    active_maps = {
        str(row["map_id"])
        for row in replacement_summaries
        if int(row["maximum_initial_conflicts"]) > 0
    }
    selected_maps = {str(row["map_id"]) for row in selected}
    selected_groups = Counter(str(row["layout_mode"]) for row in selected[::2])
    gates = {
        "replacement_dimensions": len(rows) == 24 and len(results) == 48,
        "complete_valid_replacement_product": not row_errors,
        "forbidden_outcomes_absent": not forbidden,
        "registered_runtime_exact": registered_runtime_matches(run_config, registered_runtime),
        "replacement_qualification_passed": bool(qualification_report.get("passed")),
        "all_replacement_maps_active": active_maps == REPLACEMENT_MAPS,
        "v1_v2_failure_sources_frozen": (
            v1_report.get("passed") is False
            and v2_report.get("passed") is False
            and set(map(str, v2_report.get("underloaded_maps") or ())) == REPLACED_MAPS
        ),
        "final_two_tasks_per_map": (
            not underloaded and len(selected) == 40 and len(selected_maps) == 20
        ),
        "final_topology_balance": selected_groups == Counter({
            "dao_high_topology": 12,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 2,
        }),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_blind_initial_pp_map_replacement_qualification",
        "data_line_id": "stride-robustaction-structpool-data-v3",
        "config_sha256": sha256_file(config_path),
        "replacement_manifest_sha256": sha256_file(manifest_path),
        "replacement_qualification_manifest_sha256": sha256_file(result_path),
        "replacement_qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "replacement_task_count": len(rows),
            "replacement_job_count": len(results),
            "replacement_active_map_count": len(active_maps),
            "final_selected_task_count": len(selected),
            "final_selected_map_count": len(selected_maps),
        },
        "replacement_active_maps": sorted(active_maps),
        "underloaded_maps": underloaded,
        "replacement_task_summaries": replacement_summaries,
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
    report_path = output / "map_replacement_qualification_report.json"
    _write_json(report_path, report)
    if not passed:
        return report

    source_rows = []
    for root in (v1_dataset, v2_extension_dataset, replacement_dataset):
        source_rows.extend((root, row) for row in _read_jsonl(root / SPLIT / "manifest.jsonl"))
    sources = {str(row["task_id"]): (root, row) for root, row in source_rows}
    final_rows = []
    for chosen in selected:
        task_id = str(chosen["task_id"])
        if task_id not in sources:
            raise ValueError(f"selected map-replacement task has no source: {task_id}")
        root, row = sources[task_id]
        for field in ("map_file", "scenario_file", "map_metadata_file", "task_file"):
            relative = Path(str(row[field]))
            _copy(root / SPLIT / relative, output / SPLIT / relative)
        final_rows.append(dict(row))
    final_rows.sort(key=lambda row: str(row["task_id"]))
    final_manifest = output / SPLIT / "manifest.jsonl"
    _write_jsonl(final_manifest, final_rows)
    _write_json(
        output / "dataset_summary.json",
        {
            "schema": "lns2.stride.robustaction_structpool_source_dataset.v3",
            "data_line_id": "stride-robustaction-structpool-data-v3",
            "scientific_status": "qualified_outcome_blind_source_task_dataset",
            "map_count": 20,
            "task_count": 40,
            "solver_seed_count": 2,
            "source_policy_count": 2,
            "projected_independent_episode_count": 160,
            "manifest_sha256": sha256_file(final_manifest),
            "qualification_report_sha256": sha256_file(report_path),
            "candidate_outcomes_read": False,
            "controller_outcomes_read": False,
            "formal_speed_claim": False,
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA", "REPORT_SCHEMA", "REPLACED_MAPS", "REPLACEMENT_MAPS",
    "analyze_map_replacement_qualification",
    "validate_map_replacement_qualification_design",
]
