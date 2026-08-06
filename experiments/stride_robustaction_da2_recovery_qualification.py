from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_da2_recovery import (
    DESIGN_SCHEMA,
    RETAINED_TASK_IDS,
    validate_da2_recovery_design,
)
from experiments.stride_robustaction_load_extension import _copy
from experiments.stride_robustaction_qualification import (
    FORBIDDEN_FIELDS,
    registered_runtime_matches,
    select_qualified_tasks,
)


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_recovery_qualification_design.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_recovery_qualification_report.v1"
)
SPLIT = "balanced_wall_clock"


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"DA2 recovery qualification input changed: {spec['path']}")
    return path


def _task_variant(task_id: str) -> str:
    if "__derived_opposite_exchange__" in task_id:
        return "opposite_exchange"
    if "__derived_uniform_random__" in task_id:
        return "uniform_random"
    raise ValueError(f"unregistered DA2 recovery task variant: {task_id}")


def validate_da2_recovery_qualification_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_primary_da2_recovery_initial_pp"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v1"
        or config.get("recovery_mode") != "primary_repair"
        or config.get("pre_registration_git_commit")
        != "036d06bc8946be3ea448a84adc11980065322f7f"
        or int(config.get("expected_map_count", -1)) != 1
        or int(config.get("expected_task_count", -1)) != 8
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_reset_count", -1)) != 16
        or config.get("recovery_map_id") != "ca_caverns2"
        or int(config.get("tasks_selected_per_map", -1)) != 2
        or int(config.get("expected_recovery_selected_task_count", -1)) != 2
        or list(config.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0))
        != 0.5
        or float(config.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or not bool(config.get("prefer_distinct_task_variants"))
        or not bool(config.get("all_registered_resets_must_be_terminal"))
        or not bool(config.get("candidate_ladder_errors_or_timeouts_allowed"))
        or not bool(config.get("selected_tasks_must_have_both_solver_seeds_ok"))
        or not bool(config.get("collector_qualification_pass_not_required"))
        or int(config.get("final_map_count", -1)) != 8
        or int(config.get("final_task_count", -1)) != 16
        or dict(config.get("final_layout_counts") or {})
        != {
            "dao_high_topology": 6,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        }
        or bool(config.get("candidate_repair_outcomes_read"))
        or bool(config.get("controller_outcomes_read"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("DA2 recovery qualification identity changed")
    if set(map(str, config.get("forbidden_selection_inputs") or ())) != {
        "candidate_repair_outcome",
        "candidate_conflicts_after",
        "candidate_runtime",
        "selected_candidate",
        "controller_action",
        "controller_ttf",
        "future_trajectory",
        "repair_runtime",
    }:
        raise ValueError("DA2 recovery qualification outcome boundary changed")
    if (
        config.get("next_decision_on_pass")
        != "merge_retained_14_with_primary_2_then_reaudit_combined_source_capacity"
        or config.get("next_decision_on_failure")
        != "activate_preregistered_w_encounter1_replacement"
    ):
        raise ValueError("DA2 recovery qualification decision sequence changed")

    inputs = dict(config.get("inputs") or {})
    expected_inputs = {
        "recovery_design",
        "runtime",
        "recovery_dataset_manifest",
        "recovery_dataset_summary",
        "recovery_source_adapter",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("DA2 recovery qualification input registry changed")
    paths = {
        name: _registered(project_root, dict(spec))
        for name, spec in inputs.items()
    }
    design = _read_json(paths["recovery_design"])
    validate_da2_recovery_design(design, project_root=project_root)
    if design.get("schema") != DESIGN_SCHEMA:
        raise ValueError("qualification does not reference DA2 recovery design")
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
        or int(dataset_design.get("map_count", -1)) != 1
        or int(dataset_design.get("instance_count", -1)) != 8
        or dict(dataset_design.get("layout_counts") or {})
        != {"dao_high_topology": 8}
        or int(qualification.get("minimum_active_maps", -1)) != 1
        or int(qualification.get("minimum_nonzero_states", -1)) != 2
        or not bool(qualification.get("enforce_registered_thresholds"))
    ):
        raise ValueError("DA2 recovery qualification runtime changed")


def recovery_task_summaries(
    manifest_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    solver_seeds: list[int],
) -> tuple[list[dict[str, Any]], list[str], set[str], list[dict[str, Any]]]:
    source_index = {str(row["task_id"]): row for row in manifest_rows}
    expected = {
        (task_id, seed)
        for task_id in source_index
        for seed in map(int, solver_seeds)
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    forbidden: set[str] = set()
    failed_rows: list[dict[str, Any]] = []
    for raw in result_rows:
        row = dict(raw)
        forbidden.update(FORBIDDEN_FIELDS & set(row))
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source = source_index.get(key[0])
        if source is None or key not in expected:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
            continue
        status = str(row.get("status"))
        if status != "ok":
            failed_rows.append(
                {
                    "map_id": str(row.get("map_id")),
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "status": status,
                    "error": str(row.get("error", "")),
                }
            )
            continue
        if (
            not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
            or "initial_conflicts" not in row
            or str(row.get("map_id")) != str(source["map_id"])
            or int(row.get("agent_count", -1)) != int(source["agent_count"])
        ):
            errors.append(f"invalid_ok:{key[0]}:{key[1]}")
    if set(indexed) != expected:
        errors.append("incomplete_recovery_reset_product")

    summaries: list[dict[str, Any]] = []
    for task_id, source in sorted(source_index.items()):
        values = [indexed.get((task_id, seed)) for seed in map(int, solver_seeds)]
        if any(value is None or str(value.get("status")) != "ok" for value in values):
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
                "nonzero_solver_seed_fraction": sum(value > 0 for value in conflicts)
                / len(conflicts),
                "mean_initial_conflicts": statistics.fmean(conflicts),
                "minimum_initial_conflicts": min(conflicts),
                "maximum_initial_conflicts": max(conflicts),
            }
        )
    return summaries, errors, forbidden, failed_rows


def analyze_da2_recovery_qualification(
    *,
    config_path: str | Path,
    recovery_dataset: str | Path,
    qualification: str | Path,
    retained_dataset: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_da2_recovery_qualification_design(config, project_root=project_root)
    recovery_dataset = Path(recovery_dataset).resolve()
    qualification = Path(qualification).resolve()
    retained_dataset = Path(retained_dataset).resolve()
    output = Path(output).resolve()

    manifest_path = recovery_dataset / SPLIT / "manifest.jsonl"
    result_path = qualification / "qualification_manifest.jsonl"
    collector_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    rows = _read_jsonl(manifest_path)
    results = _read_jsonl(result_path)
    collector_report = _read_json(collector_report_path)
    run_config = _read_json(run_config_path)
    registered_runtime = _read_json(
        project_root / str(config["inputs"]["runtime"]["path"])
    )
    summaries, row_errors, forbidden, failed_rows = recovery_task_summaries(
        rows, results, list(map(int, config["solver_seeds"]))
    )
    selected, underloaded = select_qualified_tasks(summaries, config)
    selected_ids = {str(row["task_id"]) for row in selected}
    gates = {
        "registered_dimensions": len(rows) == 8 and len(results) == 16,
        "all_registered_resets_terminal": not row_errors,
        "forbidden_outcomes_absent": not forbidden,
        "registered_runtime_exact": registered_runtime_matches(
            run_config, registered_runtime
        ),
        "two_complete_paired_tasks_selected": (
            not underloaded
            and len(selected) == 2
            and {str(row["map_id"]) for row in selected} == {"ca_caverns2"}
        ),
        "selected_tasks_have_both_solver_seeds": all(
            int(row["solver_seed_count"]) == 2 for row in selected
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_informed_initial_pp_recovery_qualification",
        "data_line_id": str(config["data_line_id"]),
        "config_sha256": sha256_file(config_path),
        "recovery_manifest_sha256": sha256_file(manifest_path),
        "qualification_manifest_sha256": sha256_file(result_path),
        "collector_qualification_report_sha256": sha256_file(collector_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "collector_qualification_passed": bool(collector_report.get("passed")),
        "candidate_ladder_error_count": sum(
            str(row["status"]) == "error" for row in failed_rows
        ),
        "candidate_ladder_timeout_count": sum(
            str(row["status"]) == "timeout" for row in failed_rows
        ),
        "complete_paired_task_count": len(summaries),
        "candidate_repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "failed_reset_rows": failed_rows,
        "paired_task_summaries": summaries,
        "selected_recovery_tasks": selected,
        "selected_recovery_task_ids": sorted(selected_ids),
        "underloaded_maps": underloaded,
        "forbidden_fields_found": sorted(forbidden),
        "row_errors": row_errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "da2_recovery_qualification_report.json"
    _write_json(report_path, report)
    if not passed:
        return report

    retained_rows = _read_jsonl(retained_dataset / SPLIT / "manifest.jsonl")
    retained_index = {str(row["task_id"]): row for row in retained_rows}
    recovery_index = {str(row["task_id"]): row for row in rows}
    final_sources: list[tuple[Path, dict[str, Any]]] = []
    for task_id in RETAINED_TASK_IDS:
        if task_id not in retained_index:
            raise ValueError(f"retained recovery source task is missing: {task_id}")
        final_sources.append((retained_dataset, retained_index[task_id]))
    for task_id in sorted(selected_ids):
        if task_id not in recovery_index:
            raise ValueError(f"selected recovery task is missing: {task_id}")
        final_sources.append((recovery_dataset, recovery_index[task_id]))

    final_rows: list[dict[str, Any]] = []
    for root, row in final_sources:
        for field in ("map_file", "scenario_file", "map_metadata_file", "task_file"):
            relative = Path(str(row[field]))
            _copy(root / SPLIT / relative, output / SPLIT / relative)
        final_rows.append(dict(row))
    final_rows.sort(key=lambda row: str(row["task_id"]))
    final_layouts = Counter(str(row["layout_mode"]) for row in final_rows)
    final_maps = {str(row["map_id"]) for row in final_rows}
    if (
        len(final_rows) != int(config["final_task_count"])
        or len(final_maps) != int(config["final_map_count"])
        or dict(final_layouts) != dict(config["final_layout_counts"])
    ):
        raise ValueError("final DA2 recovery dataset dimensions changed")
    final_manifest = output / SPLIT / "manifest.jsonl"
    _write_jsonl(final_manifest, final_rows)
    _write_json(
        output / "dataset_summary.json",
        {
            "schema": "lns2.stride.robustaction_da2_recovery_source_dataset.v1",
            "data_line_id": str(config["data_line_id"]),
            "scientific_status": "user_authorized_outcome_informed_recovery_cohort",
            "map_count": len(final_maps),
            "task_count": len(final_rows),
            "layout_counts": dict(final_layouts),
            "solver_seed_count": 2,
            "source_policy_count": 2,
            "projected_independent_episode_count": 64,
            "projected_state_capacity": 128,
            "manifest_sha256": sha256_file(final_manifest),
            "qualification_report_sha256": sha256_file(report_path),
            "candidate_repair_outcomes_read": False,
            "controller_outcomes_read": False,
            "formal_speed_claim": False,
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_da2_recovery_qualification",
    "recovery_task_summaries",
    "validate_da2_recovery_qualification_design",
]
