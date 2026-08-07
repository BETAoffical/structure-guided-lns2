from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


CONFIG_SCHEMA = "lns2.stride.structpool_fresh_load_recovery_design.v1"
REPORT_SCHEMA = "lns2.stride.structpool_fresh_load_recovery_report.v1"
SPLIT = "balanced_wall_clock"


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered StructPool load-recovery input changed: {path}")
    return path


def validate_structpool_fresh_load_recovery(
    config: dict[str, Any], *, project_root: Path
) -> dict[str, Path]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "outcome_informed_reset_only_recovery_preregistered_before_candidate_load_resets"
        or config.get("experiment_id")
        != "stride-structpool-fresh-load-recovery-v1"
        or config.get("pre_registration_parent_commit")
        != "872ed104bbaf47149ab5c7160cf747a9b016cac4"
        or config.get("failed_fresh_experiment_id")
        != "stride-structpool-ttf-fresh-v1"
        or config.get("dataset")
        != "build/stride-structpool-fresh-load-recovery-dataset-v1"
        or config.get("split") != SPLIT
        or config.get("runtime")
        != "configs/stride_structpool_fresh_load_recovery_runtime.json"
        or list(config.get("solver_seeds") or ()) != [1, 2, 3]
        or int(config.get("expected_map_count", -1)) != 3
        or int(config.get("expected_task_count", -1)) != 18
        or int(config.get("expected_reset_count", -1)) != 54
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("StructPool fresh load-recovery identity changed")
    boundary = dict(config.get("outcome_boundary") or {})
    if (
        boundary.get("outcome_informed") is not True
        or set(boundary.get("allowed_observed_fields") or ())
        != {
            "reset_status",
            "initial_complete",
            "initial_conflicts",
            "initial_complexity",
            "initial_state_fingerprint",
        }
        or boundary.get("candidate_repair_outcomes_read") is not False
        or boundary.get("controller_outcomes_read") is not False
        or boundary.get("ttf_outcomes_read") is not False
        or boundary.get("claim_boundary")
        != "qualification_conditioned_fresh_map_evidence_not_untouched_formal_ood"
    ):
        raise ValueError("StructPool fresh load-recovery outcome boundary changed")
    ladders = list(config.get("candidate_ladders") or ())
    if ladders != [
        {
            "map_id": "maze-128-128-10",
            "layout_family": "maze",
            "agent_counts": [400, 600, 800],
        },
        {
            "map_id": "warehouse-20-40-10-2-2",
            "layout_family": "warehouse",
            "agent_counts": [600, 800, 1000],
        },
        {
            "map_id": "lak303d",
            "layout_family": "game",
            "agent_counts": [600, 800, 1000],
        },
    ]:
        raise ValueError("StructPool fresh load-recovery ladder changed")
    if dict(config.get("selection_rule") or {}) != {
        "structpool_minimum_conflict_pair_count": 16,
        "minimum_gate_eligible_resets_per_selected_load": 3,
        "require_each_scenario_index_gate_eligible": True,
        "require_all_six_selected_load_resets_ok_and_complete": True,
        "choose_lowest_qualifying_agent_count": True,
        "scenario_indices": [4, 5],
    }:
        raise ValueError("StructPool fresh load-recovery selection rule changed")
    if set(config.get("forbidden_selection_inputs") or ()) != {
        "conflicts_after",
        "repair_iterations",
        "repair_runtime",
        "pp_replan_seconds",
        "selected_candidate",
        "controller_action",
        "controller_ttf",
        "future_trajectory",
    }:
        raise ValueError("StructPool fresh load-recovery forbidden inputs changed")
    if (
        config.get("next_step_on_pass")
        != "materialize_six_map_recovered_cohort_then_repeat_formal_reset_qualification"
        or config.get("next_step_on_failure")
        != "preregister_derived_congestion_task_recovery_without_ttf"
    ):
        raise ValueError("StructPool fresh load-recovery decision sequence changed")
    expected_inputs = {
        "failed_fresh_config",
        "failed_qualification_manifest",
        "failed_qualification_report",
        "recovery_dataset_config",
        "recovery_runtime",
        "recovery_dataset_summary",
        "recovery_dataset_manifest",
        "fetched_manifest",
        "controller_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool fresh load-recovery input registry changed")
    paths = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    failed = _read_json(paths["failed_qualification_report"])
    if (
        failed.get("passed") is not False
        or failed.get("decision") != "inconclusive_do_not_resample"
        or list(failed.get("active_maps") or ())
        != ["den312d", "random-64-64-20", "room-64-64-16"]
        or failed.get("gates", {}).get("minimum_active_maps") is not False
        or failed.get("gates", {}).get("required_layout_families_active")
        is not False
    ):
        raise ValueError("StructPool fresh qualification failure evidence changed")
    return paths


def _scenario_index(task_id: str) -> int:
    marker = "__random_"
    if marker not in task_id:
        raise ValueError(f"unregistered MovingAI task identity: {task_id}")
    return int(task_id.split(marker, 1)[1].split("__", 1)[0])


def summarize_recovery_loads(
    manifest_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], set[str]]:
    source = {str(row["task_id"]): row for row in manifest_rows}
    seeds = list(map(int, config["solver_seeds"]))
    expected = {(task_id, seed) for task_id in source for seed in seeds}
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    forbidden: set[str] = set()
    forbidden_names = set(map(str, config["forbidden_selection_inputs"]))
    for raw in qualification_rows:
        row = dict(raw)
        forbidden.update(forbidden_names & set(row))
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source_row = source.get(key[0])
        if source_row is None or key not in expected:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
        elif (
            str(row.get("map_id")) != str(source_row["map_id"])
            or int(row.get("agent_count", -1))
            != int(source_row["agent_count"])
        ):
            errors.append(f"identity:{key[0]}:{key[1]}")
    if set(indexed) != expected:
        errors.append("incomplete_reset_product")

    threshold = int(config["selection_rule"]["structpool_minimum_conflict_pair_count"])
    minimum_eligible = int(
        config["selection_rule"]["minimum_gate_eligible_resets_per_selected_load"]
    )
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for task_id, source_row in source.items():
        for seed in seeds:
            result = indexed.get((task_id, seed), {})
            grouped[(str(source_row["map_id"]), int(source_row["agent_count"]))].append(
                (source_row, result)
            )

    summaries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for ladder in config["candidate_ladders"]:
        map_id = str(ladder["map_id"])
        map_summaries = []
        for agent_count in map(int, ladder["agent_counts"]):
            rows = grouped.get((map_id, agent_count), [])
            ok = [
                result
                for _source, result in rows
                if result.get("status") == "ok"
                and result.get("initial_complete") is True
                and bool(result.get("state_fingerprint"))
            ]
            eligible = [
                result
                for result in ok
                if int(result.get("initial_conflicts", -1)) >= threshold
            ]
            eligible_scenarios = sorted(
                {_scenario_index(str(result["task_id"])) for result in eligible}
            )
            conflicts = [int(result.get("initial_conflicts", -1)) for result in ok]
            qualifies = (
                len(rows) == 6
                and len(ok) == 6
                and len(eligible) >= minimum_eligible
                and eligible_scenarios == [4, 5]
            )
            summary = {
                "map_id": map_id,
                "layout_family": str(ladder["layout_family"]),
                "agent_count": agent_count,
                "reset_count": len(rows),
                "ok_complete_reset_count": len(ok),
                "gate_eligible_reset_count": len(eligible),
                "gate_eligible_scenario_indices": eligible_scenarios,
                "minimum_initial_conflicts": min(conflicts) if conflicts else None,
                "maximum_initial_conflicts": max(conflicts) if conflicts else None,
                "mean_initial_conflicts": (
                    sum(conflicts) / len(conflicts) if conflicts else None
                ),
                "qualifies": qualifies,
            }
            summaries.append(summary)
            map_summaries.append(summary)
        winner = next((row for row in map_summaries if row["qualifies"]), None)
        if winner is not None:
            selected.append(dict(winner))
    return summaries, selected, errors, forbidden


def analyze_structpool_fresh_load_recovery(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_structpool_fresh_load_recovery(config, project_root=project_root)
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    manifest_path = dataset / SPLIT / "manifest.jsonl"
    qualification_manifest_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    manifest_rows = _read_jsonl(manifest_path)
    qualification_rows = _read_jsonl(qualification_manifest_path)
    load_summaries, selected, errors, forbidden = summarize_recovery_loads(
        manifest_rows, qualification_rows, config
    )
    selected_maps = {str(row["map_id"]) for row in selected}
    gates = {
        "registered_dimensions": len(manifest_rows) == 18
        and len(qualification_rows) == 54,
        "complete_reset_product": not errors,
        "forbidden_outcomes_absent": not forbidden,
        "one_qualifying_load_per_map": len(selected) == 3
        and selected_maps
        == {"maze-128-128-10", "warehouse-20-40-10-2-2", "lak303d"},
        "lowest_qualifying_load_selected": all(
            row["agent_count"]
            == min(
                candidate["agent_count"]
                for candidate in load_summaries
                if candidate["map_id"] == row["map_id"]
                and candidate["qualifies"]
            )
            for row in selected
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_informed_reset_only_load_recovery",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "qualification_manifest_sha256": sha256_file(
            qualification_manifest_path
        ),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "candidate_repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "ttf_outcomes_read": False,
        "formal_speed_claim": False,
        "load_summaries": load_summaries,
        "selected_loads": selected,
        "errors": errors,
        "forbidden_fields_found": sorted(forbidden),
        "gates": gates,
        "passed": passed,
        "next_step": config[
            "next_step_on_pass" if passed else "next_step_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "fresh_load_recovery_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_structpool_fresh_load_recovery",
    "summarize_recovery_loads",
    "validate_structpool_fresh_load_recovery",
]
