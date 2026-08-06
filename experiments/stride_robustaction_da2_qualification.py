from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_robustaction_expansion import (
    DA2_SUPPLEMENT_SCHEMA,
    validate_robustaction_expansion_design,
)
from experiments.stride_robustaction_load_extension import _copy, _summaries
from experiments.stride_robustaction_qualification import (
    registered_runtime_matches,
    select_qualified_tasks,
)


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_supplement_qualification_design.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_supplement_qualification_report.v1"
)
STATIC_AUDIT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_supplement_static_audit.v1"
)
SPLIT = "balanced_wall_clock"
SUPPLEMENT_MAPS = {
    "ca_cave",
    "ca_caverns2",
    "dr_primevalentrance",
    "ht_bartrand_n",
    "lt_hangedman",
    "lt_undercitydungeon",
    "lt_undercityserialkiller",
    "w_encounter3",
}
FORBIDDEN_INPUTS = {
    "candidate_repair_outcome",
    "candidate_conflicts_after",
    "candidate_runtime",
    "selected_candidate",
    "controller_action",
    "controller_ttf",
    "future_trajectory",
    "repair_runtime",
}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(
            f"DA2 supplement qualification input changed: {spec['path']}"
        )
    return path


def validate_da2_supplement_qualification_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_structpool_da2_supplement_initial_pp"
        or config.get("data_line_id") != "stride-robustaction-structpool-data-v4"
        or config.get("pre_registration_git_commit")
        != "486515cc272ff7e2215d993b10b7f8c5fa481cd7"
        or int(config.get("expected_map_count", -1)) != 8
        or int(config.get("expected_task_count", -1)) != 48
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_qualification_job_count", -1)) != 96
        or set(map(str, config.get("supplement_map_ids") or ()))
        != SUPPLEMENT_MAPS
        or int(config.get("tasks_selected_per_map", -1)) != 2
        or int(config.get("expected_selected_task_count", -1)) != 16
        or list(config.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0))
        != 0.5
        or float(config.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or not bool(config.get("prefer_distinct_task_variants"))
        or dict(config.get("expected_selected_layout_counts") or {})
        != {
            "dao_high_topology": 6,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        }
        or not bool(config.get("qualification_must_pass"))
        or not bool(config.get("all_resets_must_be_complete"))
        or not bool(config.get("all_maps_must_have_nonzero_state"))
        or not bool(config.get("selection_must_cover_all_eight_maps"))
        or bool(config.get("candidate_outcomes_read"))
        or bool(config.get("controller_outcomes_read"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("StructPool DA2 supplement qualification identity changed")
    if set(map(str, config.get("forbidden_selection_inputs") or ())) != FORBIDDEN_INPUTS:
        raise ValueError(
            "StructPool DA2 supplement qualification outcome boundary changed"
        )
    if (
        config.get("next_decision_on_pass")
        != "collect_16_tasks_times_2_seeds_times_2_frozen_source_policies_under_historical_12_step_rule"
        or config.get("next_decision_on_failure")
        != "report_exact_underloaded_maps_and_reassess_task_generator_without_outcome_filtering"
    ):
        raise ValueError(
            "StructPool DA2 supplement qualification decision rule changed"
        )

    inputs = dict(config.get("inputs") or {})
    expected_inputs = {
        "supplement_design",
        "runtime",
        "dataset_manifest",
        "dataset_summary",
        "source_adapter",
        "static_audit_report",
    }
    if set(inputs) != expected_inputs:
        raise ValueError(
            "StructPool DA2 supplement qualification input registry changed"
        )
    paths = {
        name: _registered(project_root, dict(spec))
        for name, spec in inputs.items()
    }
    design = _read_json(paths["supplement_design"])
    validate_robustaction_expansion_design(design)
    if design.get("schema") != DA2_SUPPLEMENT_SCHEMA:
        raise ValueError("qualification does not reference the DA2 supplement")
    if {str(row["id"]) for row in design["benchmarks"]} != SUPPLEMENT_MAPS:
        raise ValueError("DA2 supplement map registry differs from qualification")

    audit = _read_json(paths["static_audit_report"])
    if (
        audit.get("schema") != STATIC_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or bool(audit.get("solver_or_controller_run"))
        or bool(audit.get("performance_measurements_run"))
        or int(dict(audit.get("counts") or {}).get("selected_map_count", -1)) != 8
    ):
        raise ValueError("DA2 supplement static audit is not admissible")

    summary = _read_json(paths["dataset_summary"])
    if (
        summary.get("dataset_revision")
        != "stride-robustaction-structpool-da2-supplement-v1"
        or int(summary["splits"][SPLIT].get("map_count", -1)) != 8
        or int(summary["splits"][SPLIT].get("instance_count", -1)) != 48
    ):
        raise ValueError("DA2 supplement dataset dimensions changed")

    runtime = _read_json(paths["runtime"])
    environment = dict(runtime.get("environment") or {})
    qualification = dict(runtime.get("qualification") or {})
    dataset_design = dict(runtime.get("dataset_design") or {})
    historical = set(map(str, dataset_design.get("historical_map_ids") or ()))
    if (
        bool(runtime.get("formal"))
        or list(runtime.get("solver_seeds") or ()) != [1, 2]
        or float(environment.get("time_limit", -1.0)) != 600.0
        or int(environment.get("max_repair_iterations", -1)) != 0
        or int(runtime.get("max_decisions", -1)) != 0
        or float(runtime.get("episode_process_timeout_seconds", -1.0)) != 660.0
        or int(dataset_design.get("map_count", -1)) != 8
        or int(dataset_design.get("instance_count", -1)) != 48
        or dict(dataset_design.get("layout_counts") or {})
        != {
            "dao_high_topology": 18,
            "dao_mid_topology": 18,
            "dao_low_topology_control": 12,
        }
        or bool(SUPPLEMENT_MAPS & historical)
        or int(qualification.get("minimum_active_maps", -1)) != 8
        or int(qualification.get("minimum_nonzero_states", -1)) != 16
        or not bool(qualification.get("enforce_registered_thresholds"))
    ):
        raise ValueError("StructPool DA2 supplement qualification runtime changed")


def analyze_da2_supplement_qualification(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_da2_supplement_qualification_design(
        config, project_root=project_root
    )
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()

    manifest_path = dataset / SPLIT / "manifest.jsonl"
    result_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    rows = _read_jsonl(manifest_path)
    results = _read_jsonl(result_path)
    qualification_report = _read_json(qualification_report_path)
    run_config = _read_json(run_config_path)
    registered_runtime = _read_json(
        project_root / str(config["inputs"]["runtime"]["path"])
    )
    summaries, row_errors, forbidden = _summaries(
        rows, results, list(map(int, config["solver_seeds"]))
    )
    selected, underloaded = select_qualified_tasks(summaries, config)
    active_maps = {
        str(row["map_id"])
        for row in summaries
        if int(row["maximum_initial_conflicts"]) > 0
    }
    selected_maps = {str(row["map_id"]) for row in selected}
    selected_layouts = Counter(str(row["layout_mode"]) for row in selected)
    expected_layouts = Counter(
        {
            str(name): int(count)
            for name, count in dict(
                config["expected_selected_layout_counts"]
            ).items()
        }
    )
    gates = {
        "supplement_dimensions": len(rows) == 48 and len(results) == 96,
        "complete_valid_reset_product": not row_errors,
        "forbidden_outcomes_absent": not forbidden,
        "registered_runtime_exact": registered_runtime_matches(
            run_config, registered_runtime
        ),
        "qualification_passed": bool(qualification_report.get("passed")),
        "all_supplement_maps_active": active_maps == SUPPLEMENT_MAPS,
        "two_tasks_per_map": (
            not underloaded
            and len(selected) == 16
            and selected_maps == SUPPLEMENT_MAPS
        ),
        "selected_topology_balance_exact": selected_layouts == expected_layouts,
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "outcome_blind_initial_pp_da2_supplement_qualification"
        ),
        "data_line_id": "stride-robustaction-structpool-data-v4",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "qualification_manifest_sha256": sha256_file(result_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "task_count": len(rows),
            "qualification_job_count": len(results),
            "active_map_count": len(active_maps),
            "selected_task_count": len(selected),
            "selected_map_count": len(selected_maps),
        },
        "active_maps": sorted(active_maps),
        "underloaded_maps": underloaded,
        "task_summaries": summaries,
        "selected_tasks": selected,
        "selected_layout_counts": dict(selected_layouts),
        "forbidden_fields_found": sorted(forbidden),
        "row_errors": row_errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "da2_supplement_qualification_report.json"
    _write_json(report_path, report)
    if not passed:
        return report

    indexed = {str(row["task_id"]): row for row in rows}
    selected_rows: list[dict[str, Any]] = []
    for chosen in selected:
        task_id = str(chosen["task_id"])
        if task_id not in indexed:
            raise ValueError(f"selected DA2 supplement task has no source: {task_id}")
        row = indexed[task_id]
        for field in ("map_file", "scenario_file", "map_metadata_file", "task_file"):
            relative = Path(str(row[field]))
            _copy(dataset / SPLIT / relative, output / SPLIT / relative)
        selected_rows.append(dict(row))
    selected_rows.sort(key=lambda row: str(row["task_id"]))
    final_manifest = output / SPLIT / "manifest.jsonl"
    _write_jsonl(final_manifest, selected_rows)
    _write_json(
        output / "dataset_summary.json",
        {
            "schema": (
                "lns2.stride.robustaction_structpool_da2_supplement_source_dataset.v1"
            ),
            "data_line_id": "stride-robustaction-structpool-data-v4",
            "scientific_status": "qualified_outcome_blind_supplement_source_tasks",
            "map_count": 8,
            "task_count": 16,
            "solver_seed_count": 2,
            "source_policy_count": 2,
            "projected_independent_episode_count": 64,
            "projected_state_capacity": 128,
            "manifest_sha256": sha256_file(final_manifest),
            "qualification_report_sha256": sha256_file(report_path),
            "candidate_outcomes_read": False,
            "controller_outcomes_read": False,
            "formal_speed_claim": False,
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPLEMENT_MAPS",
    "analyze_da2_supplement_qualification",
    "validate_da2_supplement_qualification_design",
]
