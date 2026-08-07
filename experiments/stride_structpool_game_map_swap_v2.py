from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_structpool_fresh_congestion_recovery import (
    summarize_congestion_loads,
)


CONFIG_SCHEMA = "lns2.stride.structpool_game_map_swap_design.v2"
REPORT_SCHEMA = "lns2.stride.structpool_game_map_swap_report.v2"
SPLIT = "balanced_wall_clock"
EXPECTED_MAP = "orz200d"


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered second Game map-swap input changed: {path}")
    return path


def validate_structpool_game_map_swap_v2(
    config: dict[str, Any], *, project_root: Path
) -> dict[str, Path]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "outcome_informed_second_game_map_only_diagnostic_preregistered_before_resets"
        or config.get("experiment_id") != "stride-structpool-game-map-swap-v2"
        or config.get("pre_registration_parent_commit") != "695eb7a"
        or config.get("dataset") != "build/stride-structpool-game-map-swap-v2-dataset"
        or config.get("runtime") != "configs/stride_structpool_game_map_swap_v2_runtime.json"
        or config.get("split") != SPLIT
        or list(config.get("solver_seeds") or ()) != [1, 2, 3]
        or int(config.get("expected_map_count", -1)) != 1
        or int(config.get("expected_task_count", -1)) != 6
        or int(config.get("expected_reset_count", -1)) != 18
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("second Game map-swap identity changed")
    if dict(config.get("map_swap") or {}) != {
        "layout_family": "game",
        "replaced_map": "lt_hangedman",
        "replacement_map": EXPECTED_MAP,
    }:
        raise ValueError("second Game map swap changed")
    if dict(config.get("retained_selection") or {}) != {
        "map_id": "warehouse-10-20-10-2-1",
        "agent_count": 600,
    }:
        raise ValueError("retained Warehouse selection changed")
    if dict(config.get("unchanged_inputs") or {}) != {
        "task_generator": "opposite_exchange",
        "task_seeds": [233, 277],
        "agent_counts": [200, 400, 600],
        "solver_seeds": [1, 2, 3],
        "master_seed": 20260808,
        "initialization_time_limit_seconds": 300.0,
        "replan_algorithm": "PP",
        "use_sipp": True,
    }:
        raise ValueError("second Game controlled input changed")
    if list(config.get("candidate_ladders") or ()) != [
        {
            "map_id": EXPECTED_MAP,
            "layout_family": "game",
            "agent_counts": [200, 400, 600],
        }
    ]:
        raise ValueError("second Game map-swap ladder changed")
    if dict(config.get("selection_rule") or {}) != {
        "structpool_minimum_conflict_pair_count": 16,
        "minimum_gate_eligible_resets_per_selected_load": 3,
        "require_each_task_seed_gate_eligible": True,
        "require_all_six_selected_load_resets_ok_and_complete": True,
        "choose_lowest_qualifying_agent_count": True,
    }:
        raise ValueError("second Game selection rule changed")
    boundary = dict(config.get("outcome_boundary") or {})
    if (
        set(boundary.get("allowed_observed_fields") or ())
        != {
            "reset_status",
            "initial_complete",
            "initial_conflicts",
            "initial_complexity",
            "state_fingerprint",
        }
        or boundary.get("candidate_repair_outcomes_read") is not False
        or boundary.get("controller_outcomes_read") is not False
        or boundary.get("ttf_outcomes_read") is not False
        or boundary.get("claim_boundary")
        != "second_game_map_factor_diagnostic_not_fresh_ood_or_speed_evidence"
    ):
        raise ValueError("second Game outcome boundary changed")
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
        raise ValueError("second Game forbidden inputs changed")
    if (
        config.get("next_step_on_pass")
        != "materialize_revised_six_map_cohort_then_repeat_formal_reset_qualification"
        or config.get("next_step_on_failure")
        != "stop_map_only_replacements_and_reassess_game_task_flow"
    ):
        raise ValueError("second Game decision sequence changed")
    expected_inputs = {
        "raw_source_config",
        "raw_source_manifest",
        "source_config",
        "predecessor_report",
        "dataset_summary",
        "dataset_manifest",
        "runtime",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("second Game input registry changed")
    paths = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    source = _read_json(paths["source_config"])
    if (
        source.get("dataset_revision") != "stride-structpool-game-map-swap-v2"
        or list(source.get("task_seeds") or ()) != [233, 277]
        or list(source.get("task_variants") or ()) != ["opposite_exchange"]
        or [str(row.get("id")) for row in source.get("benchmarks") or ()]
        != [EXPECTED_MAP]
    ):
        raise ValueError("second Game task source changed")
    predecessor = _read_json(paths["predecessor_report"])
    if (
        predecessor.get("passed") is not False
        or predecessor.get("passing_map_ids") != ["warehouse-10-20-10-2-1"]
        or predecessor.get("failed_map_ids") != ["lt_hangedman"]
        or predecessor.get("ttf_outcomes_read") is not False
    ):
        raise ValueError("second Game predecessor changed")
    return paths


def analyze_structpool_game_map_swap_v2(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    paths = validate_structpool_game_map_swap_v2(config, project_root=project_root)
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    manifest_path = dataset / SPLIT / "manifest.jsonl"
    qualification_manifest_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    manifest_rows = _read_jsonl(manifest_path)
    qualification_rows = _read_jsonl(qualification_manifest_path)
    load_summaries, selected, errors, forbidden = summarize_congestion_loads(
        manifest_rows, qualification_rows, config
    )
    expected_task_product = (
        {str(row.get("map_id")) for row in manifest_rows} == {EXPECTED_MAP}
        and {int(row.get("agent_count", -1)) for row in manifest_rows}
        == {200, 400, 600}
        and {int(row.get("task_seed", -1)) for row in manifest_rows}
        == {233, 277}
        and {str(row.get("scenario_type")) for row in manifest_rows}
        == {"movingai_map_derived_opposite_exchange"}
    )
    selected_maps = {str(row["map_id"]) for row in selected}
    gates = {
        "registered_dimensions": len(manifest_rows) == 6
        and len(qualification_rows) == 18,
        "single_variable_task_product": expected_task_product,
        "complete_reset_product": not errors,
        "forbidden_outcomes_absent": not forbidden,
        "one_qualifying_load": len(selected) == 1
        and selected_maps == {EXPECTED_MAP},
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
    predecessor = _read_json(paths["predecessor_report"])
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_informed_second_game_map_only_diagnostic",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "qualification_manifest_sha256": sha256_file(qualification_manifest_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "failed_first_game_load_summaries": [
            row
            for row in predecessor.get("replacement_map_load_summaries", [])
            if row.get("map_id") == "lt_hangedman"
        ],
        "retained_warehouse_selection": predecessor.get("selected_loads", []),
        "replacement_map_load_summaries": load_summaries,
        "selected_loads": selected,
        "errors": errors,
        "forbidden_fields_found": sorted(forbidden),
        "candidate_repair_outcomes_read": False,
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
    _write_json(output / "game_map_swap_v2_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_structpool_game_map_swap_v2",
    "validate_structpool_game_map_swap_v2",
]
