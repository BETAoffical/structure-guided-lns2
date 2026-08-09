from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl
from experiments.stride_structpool_lean_quick import (
    CONTROLLERS,
    analyze_structpool_lean_multigroup,
    run_structpool_lean_multigroup,
    structpool_lean_schedule,
)
from experiments.stride_structpool_revised_six_map_ttf import (
    load_revised_six_map_ttf_config,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_lean_confirmation_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_lean_confirmation_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_lean_confirmation_report.v1"
STATUS_FILENAME = "confirmation_status.json"
REPORT_FILENAME = "structpool_lean_confirmation_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="LeanPool confirmation")


def load_structpool_lean_confirmation_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_label_map_disjoint_confirmation_before_timing"
        or config.get("experiment_id")
        != "stride-structpool-lean-confirmation-v1"
        or config.get("pre_registration_parent_commit")
        != "32dea699e4b4148f379c3e9c1b68460869041333"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
        or config.get("qualification_source")
        != "build/stride-structpool-revised-six-map-qualification-v1"
        or config.get("reuse_qualification_source") is not False
    ):
        raise ValueError("LeanPool confirmation identity changed")
    if dict(config.get("comparison") or {}) != {
        "lns2_baseline": "native_official_adaptive_neighborhood_generation",
        "v2_baseline": "frozen_v2_ranking_over_exact_base_pool",
        "full_structpool": "same_frozen_v2_over_base_plus_full_structpool",
        "lean_structpool": "same_frozen_v2_after_pure_bottleneck_filter",
        "execution_order": "strict_four_controller_rotation",
        "paired_solver_seed_required": True,
        "workers": 1,
    }:
        raise ValueError("LeanPool confirmation comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_lean_confirmation_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
    }:
        raise ValueError("LeanPool confirmation runtime changed")
    full = validate_structpool_augmentation(
        dict(config["structpool_augmentation"])
    )
    lean = validate_structpool_augmentation(
        dict(config["lean_structpool_augmentation"])
    )
    if full is None or "lean_filter" in full or lean is None or "lean_filter" not in lean:
        raise ValueError("LeanPool confirmation treatment changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_full_structpool": 0.02,
        "minimum_paired_faster_fraction_vs_full_structpool": 0.5,
        "maximum_group_raw_ttf_regression_vs_full_structpool": 0.1,
        "maximum_mean_repair_iterations_delta_vs_full_structpool": 0.0,
        "minimum_success_count_delta_vs_full_structpool": 0,
        "minimum_selection_seconds_improvement_vs_full_structpool": 0.0,
    }:
        raise ValueError("LeanPool confirmation performance gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "label_map_disjoint_confirmation_only": True,
        "source_maps_have_prior_full_structpool_ttf": True,
        "group_selection_uses_only_label_map_overlap": True,
        "formal_speed_claim": False,
        "fresh_map_claim": False,
        "default_replacement_allowed": False,
        "new_ranker_trained": False,
        "candidate_generator_changed": False,
        "pass_authorizes_only_never_timed_map_replication_design": True,
    }:
        raise ValueError("LeanPool confirmation claim boundary changed")

    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role")
        != "all_label_disjoint_groups_from_fixed_six_map_qualification"
        or cohort.get("dataset")
        != "build/stride-structpool-revised-six-map-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2, 3)
        or int(cohort.get("episode_count_per_controller", -1)) != 30
        or [str(row.get("id")) for row in groups]
        != ["den300", "maze100", "random500", "room400", "warehouse600"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("LeanPool confirmation cohort changed")
    expected_inputs = {
        "source_ttf_config",
        "source_qualification_analysis",
        "source_qualification_report",
        "source_qualification_manifest",
        "label_state_manifest",
        "lean_quick_config",
        "lean_quick_report",
        "runtime_config",
        "controller_manifest",
        "dataset_summary",
        "dataset_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("LeanPool confirmation input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    _source_path, _source_root, source = load_revised_six_map_ttf_config(
        inputs["source_ttf_config"]
    )
    if dict(source["structpool_augmentation"]) != dict(
        config["structpool_augmentation"]
    ):
        raise ValueError("LeanPool confirmation changed full StructPool")
    if str(source["controller_bundle"]) != str(config["controller_bundle"]):
        raise ValueError("LeanPool confirmation changed frozen V2")
    source_runtime = _read_json(
        (root / str(source["runtime"]["config"])).resolve()
    )
    expected_runtime = {
        **source_runtime,
        "experiment_runtime_id": "stride-structpool-lean-confirmation-v1",
        "qualification": {
            **dict(source_runtime["qualification"]),
            "mode": "label_map_disjoint_five_map_confirmation",
            "minimum_active_maps": 5,
        },
    }
    if _read_json(inputs["runtime_config"]) != expected_runtime:
        raise ValueError("LeanPool confirmation changed runtime beyond qualification scope")
    if _read_json(inputs["source_qualification_analysis"]).get("passed") is not True:
        raise ValueError("LeanPool confirmation source analysis did not pass")
    if _read_json(inputs["source_qualification_report"]).get("passed") is not True:
        raise ValueError("LeanPool confirmation source qualification did not pass")
    quick_report = _read_json(inputs["lean_quick_report"])
    if (
        str(dict(quick_report.get("inputs") or {}).get("config_sha256"))
        != sha256_file(inputs["lean_quick_config"])
        or quick_report.get("schema")
        != "lns2.stride.structpool_lean_quick_report.v1"
        or quick_report.get("scientific_status")
        != "development_four_controller_lean_quick"
        or quick_report.get("integrity_passed") is not True
        or quick_report.get("performance_passed") is not True
    ):
        raise ValueError("LeanPool confirmation Quick did not authorize confirmation")

    label_maps = {
        str(row["map_id"]) for row in _read_jsonl(inputs["label_state_manifest"])
    }
    source_groups = [dict(row) for row in source["cohort"]["groups"]]
    expected_groups = [
        row for row in source_groups if str(row["map_id"]) not in label_maps
    ]
    if groups != expected_groups:
        raise ValueError("LeanPool confirmation is not the exact label-disjoint subset")
    excluded = [
        row for row in source_groups if str(row["map_id"]) in label_maps
    ]
    if [(str(row["id"]), str(row["map_id"])) for row in excluded] != [
        ("orz600", "orz200d")
    ]:
        raise ValueError("LeanPool confirmation exclusion identity changed")
    return path, root, config


def run_structpool_lean_confirmation(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_lean_confirmation_config(config_path)
    return run_structpool_lean_multigroup(
        path,
        root,
        config,
        output,
        status_schema=STATUS_SCHEMA,
        status_filename=STATUS_FILENAME,
        report_schema=REPORT_SCHEMA,
        report_filename=REPORT_FILENAME,
        report_scientific_status="label_map_disjoint_four_controller_confirmation",
        next_step_on_pass="preregister_never_timed_movingai_map_replication",
        next_step_on_failure="retain_full_structpool_and_reject_leanpool_preference",
        producer_source_files=(
            "experiments/stride_structpool_lean_confirmation.py",
        ),
        resume=resume,
        dry_run=dry_run,
    )


def analyze_structpool_lean_confirmation(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_lean_confirmation_config(config_path)
    return analyze_structpool_lean_multigroup(
        path,
        config,
        output,
        status_filename=STATUS_FILENAME,
        report_schema=REPORT_SCHEMA,
        report_filename=REPORT_FILENAME,
        report_scientific_status="label_map_disjoint_four_controller_confirmation",
        next_step_on_pass="preregister_never_timed_movingai_map_replication",
        next_step_on_failure="retain_full_structpool_and_reject_leanpool_preference",
        producer_source_files=(
            "experiments/stride_structpool_lean_confirmation.py",
        ),
    )


__all__ = [
    "analyze_structpool_lean_confirmation",
    "load_structpool_lean_confirmation_config",
    "run_structpool_lean_confirmation",
    "structpool_lean_schedule",
]
