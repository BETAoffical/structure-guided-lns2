from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.balanced_wall_clock import prepare_movingai_map_derived_dataset
from experiments.repair_collection import _fingerprint, _read_json, _write_json


DESIGN_SCHEMA = "lns2.stride.robustaction_structpool_da2_recovery_design.v1"
SPLIT = "balanced_wall_clock"
MODES = ("primary_repair", "fallback_replacement")
RETAINED_TASK_IDS = (
    "ca_cave__derived_opposite_exchange__task_seed_0331__agents_1062",
    "ca_cave__derived_uniform_random__task_seed_0331__agents_0532",
    "dr_primevalentrance__derived_opposite_exchange__task_seed_0331__agents_1066",
    "dr_primevalentrance__derived_uniform_random__task_seed_0331__agents_1420",
    "ht_bartrand_n__derived_opposite_exchange__task_seed_0331__agents_0880",
    "ht_bartrand_n__derived_uniform_random__task_seed_0331__agents_1320",
    "lt_hangedman__derived_opposite_exchange__task_seed_0331__agents_0692",
    "lt_hangedman__derived_uniform_random__task_seed_0331__agents_0922",
    "lt_undercitydungeon__derived_opposite_exchange__task_seed_0331__agents_1788",
    "lt_undercitydungeon__derived_uniform_random__task_seed_0331__agents_0894",
    "lt_undercityserialkiller__derived_opposite_exchange__task_seed_0331__agents_1362",
    "lt_undercityserialkiller__derived_uniform_random__task_seed_0331__agents_1816",
    "w_encounter3__derived_opposite_exchange__task_seed_0331__agents_0952",
    "w_encounter3__derived_uniform_random__task_seed_0331__agents_1268",
)


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"DA2 recovery input changed: {spec['path']}")
    return path


def _ceil_even(value: float) -> int:
    rounded = math.ceil(value)
    return rounded if rounded % 2 == 0 else rounded + 1


def _validate_case(case: dict[str, Any], *, primary: bool) -> None:
    expected = {
        "primary_repair": {
            "map_id": "ca_caverns2",
            "member_sha256": "28b2345a79145fde0e5751a142fe8bc1e3c210d2875d4a673814424699679dba",
            "component": 13714,
            "fractions": [0.03, 0.05, 0.07, 0.09],
            "counts": [412, 686, 960, 1236],
            "tasks": 8,
            "resets": 16,
        },
        "fallback_replacement": {
            "map_id": "w_encounter1",
            "member_sha256": "fc306932cd53bc57b5d812dd13e5aa3ffe6aa2a21a8ce9201507cd6b8300a42d",
            "component": 6041,
            "fractions": [0.12, 0.18, 0.24],
            "counts": [726, 1088, 1450],
            "tasks": 6,
            "resets": 12,
        },
    }["primary_repair" if primary else "fallback_replacement"]
    fractions = list(map(float, case.get("load_fractions") or ()))
    counts = list(map(int, case.get("agent_counts") or ()))
    component = int(case.get("largest_four_connected_component", -1))
    if (
        str(case.get("map_id")) != expected["map_id"]
        or str(case.get("member")) != f"{expected['map_id']}.map"
        or str(case.get("member_sha256")) != expected["member_sha256"]
        or str(case.get("topology_group")) != "dao_high_topology"
        or component != expected["component"]
        or fractions != expected["fractions"]
        or counts != expected["counts"]
        or counts != [_ceil_even(component * value) for value in fractions]
        or int(case.get("master_seed", -1)) != 20260813
        or list(case.get("task_seeds") or ()) != [337]
        or list(case.get("task_variants") or ())
        != ["uniform_random", "opposite_exchange"]
        or int(case.get("expected_task_count", -1)) != expected["tasks"]
        or list(case.get("solver_seeds") or ()) != [1, 2]
        or int(case.get("expected_reset_count", -1)) != expected["resets"]
        or int(case.get("minimum_complete_paired_tasks", -1)) != 2
        or list(case.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(case.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or bool(case.get("require_zero_errors_across_candidate_ladder"))
        or not bool(case.get("selection_uses_only_complete_paired_tasks"))
    ):
        raise ValueError("DA2 recovery case registration changed")


def validate_da2_recovery_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "user_authorized_posthoc_initial_pp_recovery_after_da2_qualification_failure"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v1"
        or config.get("pre_registration_git_commit")
        != "ecfecc5b0ce9e8909262e307bcfb50863b46f163"
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("DA2 recovery identity changed")
    boundary = dict(config.get("outcome_boundary") or {})
    if (
        not bool(boundary.get("outcome_informed"))
        or set(map(str, boundary.get("allowed_observed_fields") or ()))
        != {
            "reset_status",
            "initial_complete",
            "initial_conflicts",
            "initial_state_fingerprint",
        }
        or bool(boundary.get("candidate_repair_outcomes_read"))
        or bool(boundary.get("controller_outcomes_read"))
        or bool(boundary.get("ttf_outcomes_read"))
        or boundary.get("claim_boundary")
        != "recovery_training_data_only_not_clean_outcome_blind_evidence"
    ):
        raise ValueError("DA2 recovery outcome boundary changed")

    inputs = dict(config.get("inputs") or {})
    expected_inputs = {
        "failed_qualification_design",
        "failed_qualification_manifest",
        "failed_qualification_report",
        "failed_analysis_report",
        "source_dataset_manifest",
        "source_dataset_summary",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("DA2 recovery input registry changed")
    paths = {
        name: _registered(project_root, dict(spec))
        for name, spec in inputs.items()
    }
    analysis = _read_json(paths["failed_analysis_report"])
    if (
        analysis.get("passed") is not False
        or int(dict(analysis.get("counts") or {}).get("qualification_job_count", -1))
        != 96
        or len(analysis.get("row_errors") or ()) != 18
        or bool(analysis.get("candidate_outcomes_read"))
        or bool(analysis.get("controller_outcomes_read"))
    ):
        raise ValueError("DA2 recovery failure evidence changed")

    retained = dict(config.get("retained") or {})
    observed_tasks = tuple(
        str(row["task_id"]) for row in analysis.get("selected_tasks") or ()
    )
    if (
        int(retained.get("map_count", -1)) != 7
        or int(retained.get("task_count", -1)) != 14
        or tuple(map(str, retained.get("task_ids") or ())) != RETAINED_TASK_IDS
        or observed_tasks != RETAINED_TASK_IDS
        or dict(retained.get("layout_counts") or {})
        != {
            "dao_high_topology": 4,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        }
        or not bool(retained.get("failed_attempts_remain_preserved"))
    ):
        raise ValueError("DA2 recovery retained cohort changed")

    _validate_case(dict(config.get("primary_repair") or {}), primary=True)
    fallback = dict(config.get("fallback_replacement") or {})
    _validate_case(fallback, primary=False)
    if fallback.get("activation") != (
        "only_if_primary_repair_has_fewer_than_two_complete_paired_tasks"
    ):
        raise ValueError("DA2 recovery fallback activation changed")

    final = dict(config.get("final_dataset_gate") or {})
    if final != {
        "map_count": 8,
        "task_count": 16,
        "layout_counts": {
            "dao_high_topology": 6,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        },
        "solver_seed_count": 2,
        "source_policy_count": 2,
        "projected_episode_count": 64,
        "maximum_states_per_episode": 2,
        "projected_state_capacity": 128,
        "candidate_repair_outcomes_may_not_select_tasks": True,
    }:
        raise ValueError("DA2 recovery final dataset gate changed")
    if (
        config.get("next_decision_on_primary_pass")
        != "merge_retained_14_with_primary_2_then_reaudit_combined_source_capacity"
        or config.get("next_decision_on_primary_failure")
        != "activate_preregistered_w_encounter1_replacement"
        or config.get("next_decision_on_fallback_failure")
        != "stop_recovery_and_report_without_additional_outcome_search"
    ):
        raise ValueError("DA2 recovery decision sequence changed")


def recovery_source_adapter(
    config: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported DA2 recovery mode: {mode}")
    case = dict(config[mode])
    return {
        "schema_version": 1,
        "dataset_revision": str(case["dataset_revision"]),
        "source": str(config["map_archive"]["source"]),
        "task_semantics": (
            "posthoc_initial_pp_recovery_on_checksum_pinned_da2_map"
        ),
        "master_seed": int(case["master_seed"]),
        "task_seeds": list(map(int, case["task_seeds"])),
        "task_variants": list(map(str, case["task_variants"])),
        "map_archive": {
            "url": str(config["map_archive"]["url"]),
            "sha256": str(config["map_archive"]["sha256"]),
        },
        "expected_map_count": 1,
        "expected_instance_count": int(case["expected_task_count"]),
        "benchmarks": [
            {
                "id": str(case["map_id"]),
                "layout_family": str(case["topology_group"]),
                "member": str(case["member"]),
                "member_sha256": str(case["member_sha256"]),
                "agent_counts": list(map(int, case["agent_counts"])),
            }
        ],
    }


def prepare_da2_recovery_dataset(
    *,
    config_path: str | Path,
    mode: str,
    fetched: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_da2_recovery_design(config, project_root=project_root)
    adapter = recovery_source_adapter(config, mode=mode)
    adapter_path = output.parent / f"{output.name}.source_adapter.json"
    summary_path = output / "dataset_summary.json"
    adapter_fingerprint = _fingerprint(adapter)
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("configuration_fingerprint") != adapter_fingerprint:
            raise ValueError("DA2 recovery output belongs to another adapter")
    elif output.is_dir() and any(output.iterdir()):
        raise ValueError("DA2 recovery output is non-empty but incomplete")
    if adapter_path.is_file() and _read_json(adapter_path) != adapter:
        raise ValueError("DA2 recovery adapter path belongs to another run")
    _write_json(adapter_path, adapter)
    summary = prepare_movingai_map_derived_dataset(
        Path(fetched).resolve(), adapter_path, output
    )
    manifest = output / SPLIT / "manifest.jsonl"
    return {
        "schema": "lns2.stride.robustaction_da2_recovery_dataset.v1",
        "data_line_id": str(config["data_line_id"]),
        "mode": mode,
        "design_sha256": sha256_file(config_path),
        "source_adapter_sha256": sha256_file(adapter_path),
        "manifest_sha256": sha256_file(manifest),
        "map_count": int(summary["splits"][SPLIT]["map_count"]),
        "task_count": int(summary["splits"][SPLIT]["instance_count"]),
        "solver_or_controller_run": False,
        "performance_measurement_run": False,
        "summary": summary,
    }


__all__ = [
    "DESIGN_SCHEMA",
    "MODES",
    "RETAINED_TASK_IDS",
    "prepare_da2_recovery_dataset",
    "recovery_source_adapter",
    "validate_da2_recovery_design",
]
